"""
DyGIE++ Standalone — Span Extractor

DyGIE++ 原論文と同様に、スパン表現を次の 3 要素の連結で構成します:
  1. span start の subword 表現
  2. span end   の subword 表現
  3. span width embedding（幅のバイアスを学習）
  4. (オプション) span 内の attended 表現（attention pooling）

入力:
  sequence_output : [B, L, H]   Transformer エンコーダの出力
  token_to_subword: [B, N]      各トークンの先頭 subword インデックス
  spans           : [B, K, 2]   スパン (start_tok, end_tok), end inclusive
  span_mask       : [B, K]      有効スパンマスク

出力:
  span_repr : [B, K, span_dim]
    span_dim = 2H + width_embedding_dim  (+ H if use_attentive_pooling)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpanExtractor(nn.Module):
    """
    Parameters
    ----------
    hidden_size : int
        Transformer の隠れ層サイズ H。
    max_span_width : int
        最大スパン幅（width embedding のボキャブラリサイズ）。
    width_embedding_dim : int
        スパン幅 embedding の次元数。
    use_attentive_pooling : bool
        True で span 内トークンの attention pooling を追加する。
    dropout : float
    """

    def __init__(
        self,
        hidden_size: int,
        max_span_width: int = 8,
        width_embedding_dim: int = 128,
        use_attentive_pooling: bool = True,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_span_width = max_span_width
        self.use_attentive_pooling = use_attentive_pooling

        # width embedding: index 0 は幅 1, index max_span_width-1 は幅 max_span_width
        self.width_embedding = nn.Embedding(max_span_width, width_embedding_dim)

        # attentive pooling 用の attention スコアラー
        if use_attentive_pooling:
            self.attention_scorer = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(dropout)

        # 出力次元
        self.span_dim = 2 * hidden_size + width_embedding_dim
        if use_attentive_pooling:
            self.span_dim += hidden_size

    def forward(
        self,
        sequence_output: torch.Tensor,   # [B, L, H]
        token_to_subword: torch.Tensor,  # [B, N]
        spans: torch.Tensor,             # [B, K, 2]
        span_mask: torch.Tensor,         # [B, K]
    ) -> torch.Tensor:
        """Return span representations [B, K, span_dim]."""
        B, L, H = sequence_output.shape
        _, K, _ = spans.shape

        start_toks = spans[:, :, 0]  # [B, K]
        end_toks   = spans[:, :, 1]  # [B, K]

        # トークンインデックス → subword インデックス
        # token_to_subword の範囲外アクセスを防ぐため clamp
        N = token_to_subword.size(1)
        start_toks_clamped = start_toks.clamp(0, N - 1)
        end_toks_clamped   = end_toks.clamp(0, N - 1)

        start_sw = token_to_subword.gather(1, start_toks_clamped)  # [B, K]
        end_sw   = token_to_subword.gather(1, end_toks_clamped)    # [B, K]

        # subword インデックスを L の範囲内にクランプ
        start_sw = start_sw.clamp(0, L - 1)
        end_sw   = end_sw.clamp(0, L - 1)

        # endpoint representations
        start_repr = sequence_output.gather(
            1, start_sw.unsqueeze(-1).expand(-1, -1, H)
        )  # [B, K, H]
        end_repr = sequence_output.gather(
            1, end_sw.unsqueeze(-1).expand(-1, -1, H)
        )  # [B, K, H]

        # width embedding
        widths = (end_toks - start_toks).clamp(0, self.max_span_width - 1)  # [B, K]
        width_repr = self.width_embedding(widths)  # [B, K, W]

        parts = [start_repr, end_repr, width_repr]

        if self.use_attentive_pooling:
            attended = self._attentive_pooling(
                sequence_output, start_sw, end_sw, span_mask
            )  # [B, K, H]
            parts.append(attended)

        span_repr = self.dropout(torch.cat(parts, dim=-1))  # [B, K, span_dim]
        return span_repr

    def _attentive_pooling(
        self,
        sequence_output: torch.Tensor,  # [B, L, H]
        start_sw: torch.Tensor,         # [B, K]
        end_sw: torch.Tensor,           # [B, K]
        span_mask: torch.Tensor,        # [B, K]
    ) -> torch.Tensor:
        """Attention pooling over subwords within each span."""
        B, L, H = sequence_output.shape
        K = start_sw.size(1)

        # attention score for every subword: [B, L]
        attn_scores = self.attention_scorer(sequence_output).squeeze(-1)  # [B, L]

        # subword 位置マスク: スパン内のみ有効
        # [B, K, L] の in-span マスクを作る
        positions = torch.arange(L, device=sequence_output.device)  # [L]
        # [B, K, L] : start_sw[b,k] <= pos <= end_sw[b,k]
        in_span = (
            (positions.unsqueeze(0).unsqueeze(0) >= start_sw.unsqueeze(-1)) &
            (positions.unsqueeze(0).unsqueeze(0) <= end_sw.unsqueeze(-1))
        )  # [B, K, L]

        # スパン外を -inf でマスク
        attn_scores_expanded = attn_scores.unsqueeze(1).expand(-1, K, -1)  # [B, K, L]
        attn_scores_expanded = attn_scores_expanded.masked_fill(~in_span, float("-inf"))
        attn_weights = torch.softmax(attn_scores_expanded, dim=-1)  # [B, K, L]

        # 全 subword が -inf の場合（スパン長 0 等）を安全に処理
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # weighted sum: [B, K, H]
        attended = torch.bmm(attn_weights, sequence_output)  # [B, K, H]
        return attended
