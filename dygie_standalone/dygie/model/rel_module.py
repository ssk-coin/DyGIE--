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
  spans         : [B, K, 2] スパン位置 (距離特徴量に使用)
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

RE スコア改善 (v4):
  1. エンティティタイプ埋め込み: NER ラベルをペア表現に追加
     → タイプ間の関係パターン（Method→Task など）を直接学習できる
  2. スパン間距離特徴: エンティティ位置差をログスケールバケットで埋め込み
     → 近接ペアほど関係ありの確率が高いというバイアスを学習
  3. Focal Loss: クラス不均衡（大多数の「関係なし」ペア）への対処
     → 難しい正例に損失を集中 (gamma=2.0 推奨)
  4. 深い Pair MLP: 2 層 + LayerNorm で表現力向上
  5. 損失集約の修正: バッチアイテムごとの平均 → 全ペアを集約した単一 CE
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: focal loss
# ---------------------------------------------------------------------------

def _focal_loss(
    logits: torch.Tensor,    # [N, C]
    labels: torch.Tensor,    # [N]
    gamma: float = 2.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Multi-class focal loss with ignore_index support.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    gamma=0 は通常の CrossEntropyLoss と等価。
    gamma>0 は「簡単な」例（p_t が高い）の損失を小さくし、
    難しい正例に学習を集中させる。SciERC の RE では「関係なし」が多数派で
    簡単な負例が学習を支配しやすいため、focal loss が効果的。
    """
    valid = labels != ignore_index
    if not valid.any():
        return logits.sum() * 0.0

    logits_v = logits[valid]    # [M, C]
    labels_v = labels[valid]    # [M]

    log_probs = F.log_softmax(logits_v, dim=-1)                           # [M, C]
    probs     = log_probs.exp()                                            # [M, C]
    p_t       = probs.gather(1, labels_v.unsqueeze(1)).squeeze(1)         # [M]
    focal_w   = (1.0 - p_t).pow(gamma).detach()   # 勾配を flow させない
    ce        = F.nll_loss(log_probs, labels_v, reduction="none")         # [M]
    return (focal_w * ce).mean()


# ---------------------------------------------------------------------------
# Helper: distance bucketing (log-scale)
# ---------------------------------------------------------------------------

def _dist_to_bucket(dist: torch.Tensor, num_buckets: int) -> torch.Tensor:
    """非負の距離をログスケールバケットインデックスに変換する。

    バケット境界 (num_buckets=10 の例):
      0: dist=0
      1: dist 1-2
      2: dist 3-6
      3: dist 7-14
      4: dist 15-30
      5: dist 31-62
      6: dist 63-126
      7: dist 127-254
      8: dist 255-510
      9: dist 511+
    """
    dist = dist.clamp(min=0)
    bucket = (dist.float() + 1.0).log2().floor().long()
    return bucket.clamp(max=num_buckets - 1)


# ---------------------------------------------------------------------------
# RelationModule
# ---------------------------------------------------------------------------

class RelationModule(nn.Module):
    """
    Parameters
    ----------
    span_dim : int
        スパン表現の次元数。
    num_rel_labels : int
        関係ラベルの種類数（"no relation" を除く）。
    num_ner_labels : int
        NER ラベルの種類数（"no entity" を除く）。
        0 のときエンティティタイプ埋め込みを無効化。
    feedforward_dim : int
        MLP 中間層の次元。
    dropout : float
    type_embedding_dim : int
        エンティティタイプ埋め込みの次元。0 で無効。
    use_distance_feature : bool
        スパン間距離特徴を使用するか。spans が None の場合は無効。
    num_distance_buckets : int
        距離バケット数（ログスケール）。
    distance_embedding_dim : int
        距離埋め込みの次元。0 で無効。
    focal_loss_gamma : float
        Focal Loss の gamma パラメータ。0 で通常の CrossEntropyLoss。
    """

    def __init__(
        self,
        span_dim: int,
        num_rel_labels: int,
        num_ner_labels: int = 0,
        feedforward_dim: int = 150,
        dropout: float = 0.4,
        type_embedding_dim: int = 0,
        use_distance_feature: bool = False,
        num_distance_buckets: int = 10,
        distance_embedding_dim: int = 64,
        focal_loss_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_rel_labels = num_rel_labels
        self.focal_loss_gamma = focal_loss_gamma
        self.use_distance_feature = use_distance_feature
        self.num_distance_buckets = num_distance_buckets

        # ---- エンティティタイプ埋め込み ----
        self.use_type_embedding = (num_ner_labels > 0 and type_embedding_dim > 0)
        if self.use_type_embedding:
            # 0 = no entity (padding_idx), 1..num_ner_labels = entity types
            self.type_embedding = nn.Embedding(
                num_ner_labels + 1, type_embedding_dim, padding_idx=0
            )
            type_feat_dim = type_embedding_dim * 2   # src タイプ + tgt タイプ
        else:
            type_feat_dim = 0

        # ---- 距離埋め込み ----
        self.use_dist_embedding = (use_distance_feature and distance_embedding_dim > 0)
        if self.use_dist_embedding:
            self.distance_embedding = nn.Embedding(num_distance_buckets, distance_embedding_dim)
            dist_feat_dim = distance_embedding_dim
        else:
            dist_feat_dim = 0

        # ---- スパン射影: span_dim → feedforward_dim ----
        self.span_proj = nn.Sequential(
            nn.Linear(span_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- Pair MLP: 2 層 + LayerNorm ----
        # ペア入力: [src_proj; tgt_proj; src_type; tgt_type; dist_emb]
        pair_input_dim = feedforward_dim * 2 + type_feat_dim + dist_feat_dim
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, feedforward_dim),
            nn.LayerNorm(feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, feedforward_dim),
            nn.LayerNorm(feedforward_dim),
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
        spans: torch.Tensor | None = None,         # [B, K, 2] (start, end) for distance
        use_gold_spans: bool = True,
    ) -> dict[str, torch.Tensor]:
        B, K, _ = span_repr.shape
        device = span_repr.device

        # ---- エンティティフラグ ----
        # 学習時は gold NER スパンを使用（use_gold_spans=True）
        if use_gold_spans and ner_labels is not None:
            entity_flags = (ner_labels > 0) & span_mask   # [B, K]
            entity_types  = ner_labels                     # [B, K] gold types
        else:
            entity_flags = (ner_preds > 0) & span_mask    # [B, K]
            entity_types  = ner_preds                      # [B, K] predicted types

        # ---- ペアマスク: [B, K, K]  (両スパンともエンティティ且つ自己ループ除外) ----
        pair_mask = entity_flags.unsqueeze(2) & entity_flags.unsqueeze(1)   # [B, K, K]
        diag = torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0)
        pair_mask = pair_mask & ~diag

        # ---- スパン射影: [B, K, ff_dim] ----
        proj = self.span_proj(span_repr)   # [B, K, ff_dim]

        # ---- エンティティスパンのみで E×E ペアを計算 ----
        # K×K 全体の行列を作成するのではなく、エンティティスパン E 個のペアだけを
        # 各バッチアイテムで個別処理する（O(K²) → O(E²) のメモリ削減）。
        # 損失は全バッチアイテムのペアを集約して一括計算する（公正な損失尺度）。
        all_pair_logits: list[torch.Tensor] = []   # List of [E×E, num_rel+1]
        all_pair_labels: list[torch.Tensor] = []   # List of [E×E]
        grad_anchors:    list[torch.Tensor] = []   # 0-loss for items with no entity pairs

        rel_preds = torch.zeros(B, K, K, dtype=torch.long, device=device)

        for b in range(B):
            # エンティティスパンのインデックス [E]
            e_idx = entity_flags[b].nonzero(as_tuple=False).view(-1)
            E = e_idx.size(0)

            if E < 2:
                # エンティティが 0 or 1 個ならペアなし → 損失への寄与なし
                if rel_labels is not None:
                    grad_anchors.append(proj[b].sum() * 0.0)
                continue

            # エンティティスパンの射影表現を収集: [E, ff_dim]
            e_proj = proj[b].index_select(0, e_idx)

            # ---- ペア表現の構築 ----
            src = e_proj.unsqueeze(1).expand(-1, E, -1)    # [E, E, ff_dim]
            tgt = e_proj.unsqueeze(0).expand(E, -1, -1)    # [E, E, ff_dim]
            parts = [src, tgt]

            # エンティティタイプ埋め込み [E, E, type_emb_dim] × 2
            if self.use_type_embedding:
                e_types = entity_types[b].index_select(0, e_idx)            # [E]
                src_type = self.type_embedding(e_types).unsqueeze(1).expand(-1, E, -1)  # [E, E, te]
                tgt_type = self.type_embedding(e_types).unsqueeze(0).expand(E, -1, -1)  # [E, E, te]
                parts.extend([src_type, tgt_type])

            # スパン間距離埋め込み [E, E, dist_emb_dim]
            if self.use_dist_embedding and spans is not None:
                e_spans = spans[b].index_select(0, e_idx)              # [E, 2]
                e_centers = (e_spans[:, 0].float() + e_spans[:, 1].float()) / 2.0  # [E]
                dist = (e_centers.unsqueeze(1) - e_centers.unsqueeze(0)).abs()     # [E, E]
                dist_bucket = _dist_to_bucket(dist, self.num_distance_buckets)     # [E, E]
                dist_emb = self.distance_embedding(dist_bucket)                    # [E, E, de]
                parts.append(dist_emb)

            pair_repr = torch.cat(parts, dim=-1)    # [E, E, pair_input_dim]
            hidden    = self.pair_mlp(pair_repr)    # [E, E, ff_dim]
            e_logits  = self.classifier(hidden)     # [E, E, num_rel+1]

            # ---- 損失用ラベル収集 ----
            if rel_labels is not None:
                # gold ラベルをエンティティスパンに絞り込む
                e_labels = rel_labels[b].index_select(0, e_idx).index_select(1, e_idx)  # [E, E]
                # 対角（自己ループ）を ignore
                e_diag = torch.eye(E, dtype=torch.bool, device=device)
                e_labels = e_labels.masked_fill(e_diag, -100)
                all_pair_logits.append(e_logits.reshape(-1, self.num_rel_labels + 1))
                all_pair_labels.append(e_labels.reshape(-1))

            # ---- 予測: E×E → K×K にスキャッタ ----
            with torch.no_grad():
                e_preds = e_logits.argmax(dim=-1)           # [E, E]
                ri = e_idx.unsqueeze(1).expand(-1, E)       # [E, E]
                ci = e_idx.unsqueeze(0).expand(E, -1)       # [E, E]
                rel_preds[b, ri, ci] = e_preds

        # 対角（自己ループ）の予測をゼロに（pair_mask との整合）
        rel_preds = rel_preds * pair_mask.long()

        output: dict[str, torch.Tensor] = {
            "rel_preds": rel_preds,
            "pair_mask": pair_mask,
        }

        if rel_labels is not None:
            if all_pair_logits:
                # 全バッチアイテムの全ペアを一括で損失計算
                # （バッチアイテムごとの平均ではなく、全ペアに対する公正な損失）
                stacked_logits = torch.cat(all_pair_logits, dim=0)   # [N_total, num_rel+1]
                stacked_labels = torch.cat(all_pair_labels, dim=0)   # [N_total]

                if self.focal_loss_gamma > 0.0:
                    rel_loss = _focal_loss(
                        stacked_logits, stacked_labels,
                        gamma=self.focal_loss_gamma,
                    )
                else:
                    rel_loss = F.cross_entropy(
                        stacked_logits, stacked_labels, ignore_index=-100
                    )

                # エンティティなしのバッチアイテムのための勾配アンカー（値は 0）
                if grad_anchors:
                    rel_loss = rel_loss + sum(grad_anchors)
            elif grad_anchors:
                # 全バッチアイテムにエンティティなし → 損失 0（勾配グラフを維持）
                rel_loss = sum(grad_anchors)
            else:
                rel_loss = proj.sum() * 0.0

            output["rel_loss"] = rel_loss

        return output
