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
  rel_logits  : [B, K, K, num_rel_labels+1]  (full スパン行列)
  rel_loss    : scalar (学習時のみ)
  rel_preds   : [B, K, K]
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

        # スパン表現を射影
        proj = self.span_proj(span_repr)            # [B, K, ff_dim]

        # ペアリング: [B, K, K, 2*ff_dim]
        src = proj.unsqueeze(2).expand(-1, -1, K, -1)  # [B, K, K, ff_dim]
        tgt = proj.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, K, ff_dim]
        pair_repr = torch.cat([src, tgt], dim=-1)       # [B, K, K, 2*ff_dim]

        hidden = self.pair_mlp(pair_repr)               # [B, K, K, ff_dim]
        logits = self.classifier(hidden)                # [B, K, K, num_rel+1]

        output: dict[str, torch.Tensor] = {"rel_logits": logits}

        # ---- ペアマスク: 両スパンがエンティティであるペアのみ ----
        # 学習時は gold NER、推論時は predicted NER を使用
        entity_flags: torch.Tensor
        if use_gold_spans and ner_labels is not None:
            entity_flags = (ner_labels > 0) & span_mask  # [B, K]
        else:
            entity_flags = (ner_preds > 0) & span_mask   # [B, K]

        # ペアマスク: [B, K, K]  src != tgt (自己ループ除外)
        pair_mask = (
            entity_flags.unsqueeze(2) & entity_flags.unsqueeze(1)
        )  # [B, K, K]
        diag = torch.eye(K, dtype=torch.bool, device=span_repr.device).unsqueeze(0)
        pair_mask = pair_mask & ~diag

        # ---- loss ----
        if rel_labels is not None:
            masked_labels = rel_labels.clone()
            masked_labels[~pair_mask] = -100
            loss = F.cross_entropy(
                logits.view(-1, self.num_rel_labels + 1),
                masked_labels.view(-1),
                ignore_index=-100,
            )
            output["rel_loss"] = loss

        preds = logits.argmax(dim=-1)               # [B, K, K]
        preds = preds * pair_mask.long()
        output["rel_preds"] = preds
        output["pair_mask"] = pair_mask

        return output
