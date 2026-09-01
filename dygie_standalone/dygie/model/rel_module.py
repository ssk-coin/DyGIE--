"""
DyGIE++ Standalone — Relation Extraction Module

NER で検出されたエンティティスパンのペアに対して関係ラベルを分類する。
学習時は gold NER スパンを使用し、推論時は predicted NER スパンを使用する。

入力:
  span_repr     : [B, K, span_dim]
  span_mask     : [B, K]
  ner_preds     : [B, K]   予測 NER ラベル (0=none)
  rel_labels    : [B, K, K] (学習時のみ)
  ner_labels    : [B, K]   gold NER ラベル (学習時のみ)
  use_gold_spans: bool     学習時 True

出力:
  rel_loss    : scalar (学習時のみ)
  rel_preds   : [B, K, K]
  pair_mask   : [B, K, K]  両スパンがエンティティのペアのみ True

メモリ最適化 (v3):
  元の実装は全スパン K 個の K×K ペア行列 [B, K, K, 2*ff_dim] を作成していたため、
  K=1572 のとき batch=4 で約 12 GB の中間テンソルが発生していた。

  修正後はエンティティスパン E 個のみの E×E ペア行列を各バッチアイテムごとに
  個別に計算する（E ≈ 20〜50 が典型的）。
  メモリ使用量は O(K²) → O(E²) に削減される。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationModule(nn.Module):
    """
    Parameters
    ----------
    span_dim : int
        スパン表現の次元数。
    num_rel_labels : int
        関係ラベルの種類数（"no relation" を除く）。
    feedforward_dim : int
        MLP 中間層の次元。
    dropout : float
    """

    def __init__(
        self,
        span_dim: int,
        num_rel_labels: int,
        feedforward_dim: int = 150,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.num_rel_labels = num_rel_labels

        # 各スパン表現を個別に変換してからペア結合
        self.span_proj = nn.Sequential(
            nn.Linear(span_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # ペア結合後の MLP: 2 * feedforward_dim → feedforward_dim
        self.pair_mlp = nn.Sequential(
            nn.Linear(feedforward_dim * 2, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # +1 for "no relation" (class 0)
        self.classifier = nn.Linear(feedforward_dim, num_rel_labels + 1)

    def forward(
        self,
        span_repr: torch.Tensor,                    # [B, K, span_dim]
        span_mask: torch.Tensor,                    # [B, K]
        ner_preds: torch.Tensor,                    # [B, K]
        rel_labels: torch.Tensor | None = None,    # [B, K, K]
        ner_labels: torch.Tensor | None = None,    # [B, K] gold
        use_gold_spans: bool = True,
    ) -> dict[str, torch.Tensor]:
        B, K, _ = span_repr.shape
        device = span_repr.device

        # ---- エンティティフラグ ----
        # 学習時は gold NER、推論時は predicted NER を使用
        if use_gold_spans and ner_labels is not None:
            entity_flags = (ner_labels > 0) & span_mask   # [B, K]
        else:
            entity_flags = (ner_preds > 0) & span_mask    # [B, K]

        # ---- ペアマスク: [B, K, K]  (出力用・両スパンともエンティティ且つ自己ループ除外) ----
        pair_mask = entity_flags.unsqueeze(2) & entity_flags.unsqueeze(1)  # [B, K, K]
        diag = torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0)
        pair_mask = pair_mask & ~diag

        # ---- スパン射影: [B, K, ff_dim]  (K 個なので軽い) ----
        proj = self.span_proj(span_repr)  # [B, K, ff_dim]

        # ---- エンティティスパンのみで E×E ペアを計算 ----
        # K×K 全体の行列を作成するのではなく、
        # エンティティスパン E 個のペアだけを各バッチアイテムで個別処理する。
        # これにより O(K²) → O(E²) にメモリを削減する。
        losses: list[torch.Tensor] = []
        rel_preds = torch.zeros(B, K, K, dtype=torch.long, device=device)

        for b in range(B):
            # エンティティスパンのインデックス [E]
            e_idx = entity_flags[b].nonzero(as_tuple=False).view(-1)
            E = e_idx.size(0)

            if E < 2:
                # エンティティが 0 or 1 個ならペアなし
                if rel_labels is not None:
                    # 損失項を 0 として追加（勾配グラフを維持）
                    losses.append(proj[b].sum() * 0.0)
                continue

            # エンティティスパンの射影表現を収集: [E, ff_dim]
            e_proj = proj[b].index_select(0, e_idx)

            # E×E ペア表現（K×K の代わり）
            # unsqueeze + expand は view のみで実メモリを使わないが、
            # cat で E×E×(2*ff_dim) のテンソルを生成する（E<<K なので小さい）
            src = e_proj.unsqueeze(1).expand(-1, E, -1)   # [E, E, ff_dim]
            tgt = e_proj.unsqueeze(0).expand(E, -1, -1)   # [E, E, ff_dim]
            pair_repr = torch.cat([src, tgt], dim=-1)      # [E, E, 2*ff_dim]

            hidden = self.pair_mlp(pair_repr)              # [E, E, ff_dim]
            e_logits = self.classifier(hidden)             # [E, E, num_rel+1]

            # ---- 損失: E×E のラベルから直接計算 ----
            if rel_labels is not None:
                # gold ラベルをエンティティスパンに絞り込む
                e_labels = rel_labels[b].index_select(0, e_idx).index_select(1, e_idx)  # [E, E]
                # 対角（自己ループ）を ignore
                e_diag = torch.eye(E, dtype=torch.bool, device=device)
                e_labels = e_labels.masked_fill(e_diag, -100)
                loss_b = F.cross_entropy(
                    e_logits.reshape(-1, self.num_rel_labels + 1),
                    e_labels.reshape(-1),
                    ignore_index=-100,
                )
                losses.append(loss_b)

            # ---- 予測: E×E → K×K にスキャッタ ----
            with torch.no_grad():
                e_preds = e_logits.argmax(dim=-1)   # [E, E]
                # 行インデックス [E, E], 列インデックス [E, E]
                ri = e_idx.unsqueeze(1).expand(-1, E)   # [E, E]
                ci = e_idx.unsqueeze(0).expand(E, -1)   # [E, E]
                rel_preds[b, ri, ci] = e_preds

        # 対角（自己ループ）の予測をゼロに
        rel_preds = rel_preds * pair_mask.long()

        output: dict[str, torch.Tensor] = {
            "rel_preds": rel_preds,
            "pair_mask": pair_mask,
        }

        if rel_labels is not None:
            if losses:
                output["rel_loss"] = torch.stack(losses).mean()
            else:
                # 全バッチアイテムにエンティティなし → 損失 0（勾配グラフを維持）
                output["rel_loss"] = proj.sum() * 0.0

        return output
