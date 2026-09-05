"""
DyGIE-- — Event Extraction Module (v5)

イベント抽出は 2 段階のタスクで構成される:
  1. トリガー検出 (Trigger Detection):
     各スパンがイベントトリガーであるかどうかを分類する。
     NERModule と同じ構造（2 層 MLP → softmax）。

  2. 引数抽出 (Argument Extraction):
     各 (トリガースパン, 候補スパン) ペアに対して引数ロールを分類する。
     RelationModule と類似した構造だが、ペアは (trigger, any_span) 形式。
     メモリ効率化として、全 K×K ではなく T×K（T=トリガー数）のみ計算する。

DyGIE++ での実行順序（v5 でも変わらず）:
  span_repr → Coref → SpanProp → NER → RE → Event

入力:
  span_repr             : [B, K, span_dim]
  span_mask             : [B, K]
  event_trigger_labels  : [B, K]  gold トリガーラベル (0=none, 学習時)
  event_arg_labels      : [B, K, K] gold 引数ロール (0=none, 学習時)
  use_gold_triggers     : bool  学習時 True（gold trigger で引数を計算）

出力:
  trigger_loss    : scalar (学習時のみ)
  arg_loss        : scalar (学習時のみ)
  event_loss      : scalar (trigger_loss + arg_loss, 学習時のみ)
  trigger_logits  : [B, K, num_event_types+1]
  trigger_preds   : [B, K]
  arg_preds       : [B, K, K]  arg_preds[b, trigger_k, arg_k] = role
  arg_mask        : [B, K, K]  予測トリガーがある (trigger, span) ペアが True

参考:
  Wadden et al. (2019) Section 3.4
  https://arxiv.org/abs/1909.03546
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EventModule(nn.Module):
    """
    イベントトリガー検出 + 引数抽出モジュール。

    Parameters
    ----------
    span_dim : int
        スパン表現の次元数。
    num_event_types : int
        イベントタイプの種類数（"no trigger" を除く）。
    num_arg_roles : int
        引数ロールの種類数（"no role" を除く）。
    feedforward_dim : int
        MLP 中間層の次元。
    dropout : float
    """

    def __init__(
        self,
        span_dim: int,
        num_event_types: int,
        num_arg_roles: int,
        feedforward_dim: int = 150,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.num_event_types = num_event_types
        self.num_arg_roles = num_arg_roles

        # ---- トリガースコアラー (like NERModule) ----
        self.trigger_mlp = nn.Sequential(
            nn.Linear(span_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # +1 for "no trigger" (class 0)
        self.trigger_classifier = nn.Linear(feedforward_dim, num_event_types + 1)

        # ---- 引数スパン射影: span_dim → feedforward_dim ----
        self.span_proj = nn.Sequential(
            nn.Linear(span_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- 引数ペア MLP: 2 層 + LayerNorm (like RelationModule) ----
        # 入力: [trigger_proj; arg_proj] = 2 * feedforward_dim
        pair_input_dim = feedforward_dim * 2
        self.arg_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, feedforward_dim),
            nn.LayerNorm(feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, feedforward_dim),
            nn.LayerNorm(feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # +1 for "no role" (class 0)
        self.arg_classifier = nn.Linear(feedforward_dim, num_arg_roles + 1)

    def forward(
        self,
        span_repr: torch.Tensor,                       # [B, K, span_dim]
        span_mask: torch.Tensor,                       # [B, K]
        event_trigger_labels: torch.Tensor | None = None,  # [B, K]
        event_arg_labels: torch.Tensor | None = None,      # [B, K, K]
        use_gold_triggers: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict:
          trigger_loss   : scalar (学習時のみ)
          arg_loss       : scalar (学習時のみ)
          event_loss     : scalar (学習時のみ)
          trigger_logits : [B, K, num_event_types+1]
          trigger_preds  : [B, K]
          arg_preds      : [B, K, K]
          arg_mask       : [B, K, K]  評価用の有効 (trigger, arg) ペアマスク
        """
        B, K, _ = span_repr.shape
        device = span_repr.device

        # ---- (1) トリガー検出 ----
        trigger_hidden = self.trigger_mlp(span_repr)         # [B, K, ff_dim]
        trigger_logits = self.trigger_classifier(trigger_hidden)  # [B, K, num_types+1]

        trigger_preds = trigger_logits.argmax(dim=-1)        # [B, K]
        trigger_preds = trigger_preds * span_mask.long()     # 無効スパンは 0

        # ---- (2) トリガー損失 ----
        trigger_loss = torch.tensor(0.0, device=device)
        if event_trigger_labels is not None:
            masked_labels = event_trigger_labels.clone()
            masked_labels[~span_mask] = -100
            trigger_loss = F.cross_entropy(
                trigger_logits.view(-1, self.num_event_types + 1),
                masked_labels.view(-1),
                ignore_index=-100,
            )

        # ---- (3) 引数抽出 ----
        # 学習時: gold triggers を使用 (use_gold_triggers=True)
        # 推論時: predicted triggers を使用
        if use_gold_triggers and event_trigger_labels is not None:
            trigger_flags = (event_trigger_labels > 0) & span_mask  # [B, K]
        else:
            trigger_flags = (trigger_preds > 0) & span_mask         # [B, K]

        # スパン射影: [B, K, ff_dim]
        proj = self.span_proj(span_repr)

        # 引数マスク: trigger × any_valid_span
        # arg_mask[b, t, k] = True iff span t is a trigger and span k is valid
        arg_mask = trigger_flags.unsqueeze(2) & span_mask.unsqueeze(1)  # [B, K, K]

        # arg_preds: [B, K, K]
        arg_preds = torch.zeros(B, K, K, dtype=torch.long, device=device)

        all_arg_logits: list[torch.Tensor] = []
        all_arg_labels: list[torch.Tensor] = []
        grad_anchors:   list[torch.Tensor] = []

        for b in range(B):
            # トリガースパンのインデックス [T]
            t_idx = trigger_flags[b].nonzero(as_tuple=False).view(-1)
            T = t_idx.size(0)

            if T == 0:
                # トリガーなし → 損失への寄与なし（勾配グラフ維持のみ）
                if event_trigger_labels is not None:
                    grad_anchors.append(proj[b].sum() * 0.0)
                continue

            # トリガースパンの射影表現: [T, ff_dim]
            t_proj = proj[b].index_select(0, t_idx)
            # 全スパンの射影表現: [K, ff_dim]
            a_proj = proj[b]

            # ---- ペア表現の構築: [T, K, 2*ff_dim] ----
            src = t_proj.unsqueeze(1).expand(-1, K, -1)   # [T, K, ff_dim]
            tgt = a_proj.unsqueeze(0).expand(T, -1, -1)   # [T, K, ff_dim]
            pair_repr = torch.cat([src, tgt], dim=-1)     # [T, K, 2*ff_dim]

            hidden = self.arg_mlp(pair_repr)              # [T, K, ff_dim]
            b_logits = self.arg_classifier(hidden)        # [T, K, num_roles+1]

            # ---- 損失用ラベル収集 ----
            if event_arg_labels is not None:
                # gold ラベルをトリガースパンに絞り込む: [T, K]
                b_labels = event_arg_labels[b].index_select(0, t_idx)  # [T, K]
                # 無効スパン (span_mask=False) を ignore
                b_labels = b_labels.masked_fill(~span_mask[b].unsqueeze(0), -100)
                all_arg_logits.append(b_logits.reshape(-1, self.num_arg_roles + 1))
                all_arg_labels.append(b_labels.reshape(-1))

            # ---- 予測値を T×K → K×K にスキャッタ ----
            with torch.no_grad():
                b_preds = b_logits.argmax(dim=-1)          # [T, K]
                b_preds = b_preds * span_mask[b].unsqueeze(0)  # 無効スパンは 0
                ri = t_idx.unsqueeze(1).expand(-1, K)      # [T, K]
                ci = torch.arange(K, device=device).unsqueeze(0).expand(T, -1)  # [T, K]
                arg_preds[b, ri, ci] = b_preds

        # ---- 引数損失 ----
        arg_loss = torch.tensor(0.0, device=device)
        if event_trigger_labels is not None:
            if all_arg_logits:
                stacked_logits = torch.cat(all_arg_logits, dim=0)   # [N_total, num_roles+1]
                stacked_labels = torch.cat(all_arg_labels, dim=0)   # [N_total]
                arg_loss = F.cross_entropy(
                    stacked_logits, stacked_labels, ignore_index=-100
                )
                if grad_anchors:
                    arg_loss = arg_loss + sum(grad_anchors)
            elif grad_anchors:
                arg_loss = sum(grad_anchors)
            else:
                arg_loss = proj.sum() * 0.0

        output: dict[str, torch.Tensor] = {
            "trigger_logits": trigger_logits,
            "trigger_preds":  trigger_preds,
            "arg_preds":      arg_preds,
            "arg_mask":       arg_mask,
        }

        if event_trigger_labels is not None:
            output["trigger_loss"] = trigger_loss
            output["arg_loss"]     = arg_loss
            output["event_loss"]   = trigger_loss + arg_loss

        return output
