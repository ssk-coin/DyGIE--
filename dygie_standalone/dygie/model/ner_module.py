"""
DyGIE-- — NER Module

スパン表現を受け取り、各スパンが Named Entity かどうかを分類する。
ラベル 0 = "no entity"、1 以上が実エンティティラベル。

入力:
  span_repr  : [B, K, span_dim]
  span_mask  : [B, K]
  ner_labels : [B, K]  (学習時のみ)

出力:
  logits     : [B, K, num_ner_labels + 1]
  loss       : scalar (学習時のみ)
  predicted_ner_labels : [B, K]  (推論時)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NERModule(nn.Module):
    """
    Parameters
    ----------
    span_dim : int
        スパン表現の次元数。
    num_ner_labels : int
        エンティティラベルの種類数（"no entity" を除く）。
    feedforward_dim : int
        MLP 中間層の次元。
    dropout : float
    """

    def __init__(
        self,
        span_dim: int,
        num_ner_labels: int,
        feedforward_dim: int = 150,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.num_ner_labels = num_ner_labels

        self.mlp = nn.Sequential(
            nn.Linear(span_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # +1 for "no entity" (class 0)
        self.classifier = nn.Linear(feedforward_dim, num_ner_labels + 1)

    def forward(
        self,
        span_repr: torch.Tensor,            # [B, K, span_dim]
        span_mask: torch.Tensor,            # [B, K]
        ner_labels: torch.Tensor | None = None,  # [B, K]
    ) -> dict[str, torch.Tensor]:
        hidden = self.mlp(span_repr)            # [B, K, ff_dim]
        logits = self.classifier(hidden)        # [B, K, num_labels+1]

        output: dict[str, torch.Tensor] = {"ner_logits": logits}

        if ner_labels is not None:
            # マスク外のスパンは loss 計算から除外
            # CrossEntropyLoss は ignore_index でマスク
            # span_mask が False のスパンは -100 にする
            masked_labels = ner_labels.clone()
            masked_labels[~span_mask] = -100
            loss = F.cross_entropy(
                logits.view(-1, self.num_ner_labels + 1),
                masked_labels.view(-1),
                ignore_index=-100,
            )
            output["ner_loss"] = loss

        preds = logits.argmax(dim=-1)          # [B, K]
        preds = preds * span_mask.long()       # 無効スパンは 0
        output["ner_preds"] = preds

        return output
