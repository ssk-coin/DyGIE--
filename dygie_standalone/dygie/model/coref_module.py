"""
DyGIE-- Standalone — Coreference Resolution Module

DyGIE++ / Lee et al. (2018) スタイルの coref ヘッド。

主なステップ:
  1. スパンスコアリング（mention スコア）
  2. Top-λ mention pruning
  3. antecedent スコアリング（mention ペアスコア + mention スコアの合計）
  4. softmax で antecedent 分布を計算し、損失を計算

入力:
  span_repr    : [B, K, span_dim]
  span_mask    : [B, K]
  coref_clusters : list[list[list[tuple]]]  (gold クラスタ、学習時のみ)

出力:
  coref_loss   : scalar (学習時のみ)
  mention_scores : [B, K]
  top_spans    : [B, T]  pruned span indices
  antecedent_scores : [B, T, T+1]  (0 番目はダミー antecedent)

バグ修正 (v2):
  - antecedent 距離の計算を修正: j-i → i-j (逆方向だったため距離埋め込みが機能せず)
  - max_top_antecedents を _score_antecedents で正しく適用
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CorefModule(nn.Module):
    """
    Parameters
    ----------
    span_dim : int
        スパン表現の次元数。
    feature_dim : int
        各種 feature embedding の次元（距離など）。
    max_top_antecedents : int
        各 mention に対して考慮する antecedent の最大数。
    spans_per_word : float
        mention pruning 後に残すスパンの割合（ドキュメントトークン数 × spans_per_word）。
    feedforward_dim : int
    dropout : float
    """

    def __init__(
        self,
        span_dim: int,
        feature_dim: int = 20,
        max_top_antecedents: int = 50,
        spans_per_word: float = 0.4,
        feedforward_dim: int = 150,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.max_top_antecedents = max_top_antecedents
        self.spans_per_word = spans_per_word
        self.feature_dim = feature_dim

        # mention スコアラー
        self.mention_scorer = nn.Sequential(
            nn.Linear(span_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, 1),
        )

        # antecedent スコアラー（ペア表現）
        # 入力: [span_i, span_j, span_i * span_j, distance_emb]
        pair_input_dim = 3 * span_dim + feature_dim
        self.antecedent_scorer = nn.Sequential(
            nn.Linear(pair_input_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, 1),
        )

        # antecedent 距離 embedding（10 バケット）
        self.distance_emb = nn.Embedding(10, feature_dim)

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        span_repr: torch.Tensor,                      # [B, K, span_dim]
        span_mask: torch.Tensor,                      # [B, K]
        spans: torch.Tensor,                          # [B, K, 2]
        num_tokens: torch.Tensor,                     # [B]
        coref_clusters: list[list[list[tuple]]] | None = None,
    ) -> dict[str, torch.Tensor]:
        B, K, D = span_repr.shape
        device = span_repr.device

        # ---- (1) mention スコア ----
        mention_scores = self.mention_scorer(span_repr).squeeze(-1)  # [B, K]
        mention_scores = mention_scores + (~span_mask).float() * -1e9

        # ---- (2) Top-T pruning ----
        T_list = [
            max(1, min(K, int(n.item() * self.spans_per_word)))
            for n in num_tokens
        ]
        T = max(T_list)

        top_indices_list = []
        for b in range(B):
            T_b = T_list[b]
            scores_b = mention_scores[b]
            _, top_idx = scores_b.topk(T_b, dim=0)
            top_idx, _ = top_idx.sort()
            # T にパディング（同じインデックスを繰り返し）
            if T_b < T:
                pad = top_idx[-1:].expand(T - T_b)
                top_idx = torch.cat([top_idx, pad], dim=0)
            top_indices_list.append(top_idx)

        top_indices = torch.stack(top_indices_list, dim=0)  # [B, T]

        # top span 表現・スコア
        top_repr = span_repr.gather(
            1, top_indices.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, T, D]
        top_mention_scores = mention_scores.gather(1, top_indices)  # [B, T]

        # top span マスク（パディングした部分を除く）
        top_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        for b in range(B):
            top_mask[b, : T_list[b]] = True

        # ---- (3) antecedent スコアリング ----
        # 各 mention i に対して i より前の mention j を antecedent 候補とする
        # antecedent スコア: [B, T, T]  (i, j) = score(i→j), j < i のみ有効
        ant_scores = self._score_antecedents(
            top_repr, top_mention_scores, top_indices, top_mask, spans
        )  # [B, T, T+1]  (index 0 = dummy antecedent)

        output: dict[str, torch.Tensor] = {
            "mention_scores": mention_scores,
            "top_span_indices": top_indices,
            "top_span_mask": top_mask,
            "antecedent_scores": ant_scores,
        }

        # ---- (4) loss ----
        if coref_clusters is not None:
            loss = self._coref_loss(
                ant_scores, top_indices, top_mask, spans, coref_clusters, B, T, device
            )
            output["coref_loss"] = loss

        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_antecedents(
        self,
        top_repr: torch.Tensor,          # [B, T, D]
        top_scores: torch.Tensor,        # [B, T]
        top_indices: torch.Tensor,       # [B, T]
        top_mask: torch.Tensor,          # [B, T]
        spans: torch.Tensor,             # [B, K, 2]
    ) -> torch.Tensor:
        """Return antecedent scores [B, T, T+1].  Index 0 = dummy."""
        B, T, D = top_repr.shape
        device = top_repr.device

        # ---- 修正: antecedent 距離バケット ----
        # dist[i, j] = max(0, i - j):  mention i から antecedent j までの距離
        # 旧コードは (j - i) を使っており、j < i のとき常に 0 になるバグがあった。
        dist = torch.clamp(
            torch.arange(T, device=device).unsqueeze(1) -   # [T, 1] → i
            torch.arange(T, device=device).unsqueeze(0),    # [1, T] → j
            min=0,
        )  # [T, T]  dist[i, j] = max(0, i - j)
        dist_buckets = self._bucket_distance(dist)  # [T, T]
        dist_emb = self.distance_emb(dist_buckets)   # [T, T, feat_dim]
        dist_emb = dist_emb.unsqueeze(0).expand(B, -1, -1, -1)  # [B, T, T, feat_dim]

        # ペア表現: mention i が subject, mention j が antecedent
        src = top_repr.unsqueeze(2).expand(-1, -1, T, -1)   # [B, T, T, D]
        tgt = top_repr.unsqueeze(1).expand(-1, T, -1, -1)   # [B, T, T, D]
        prod = src * tgt                                     # [B, T, T, D]

        pair_input = torch.cat([src, tgt, prod, dist_emb], dim=-1)  # [B, T, T, 3D+f]
        pair_scores = self.antecedent_scorer(pair_input).squeeze(-1)  # [B, T, T]

        # mention スコアを加算: s(i, j) = s_m(i) + s_m(j) + s_a(i, j)
        pair_scores = (
            pair_scores
            + top_scores.unsqueeze(2)   # s_m(j) broadcast
            + top_scores.unsqueeze(1)   # s_m(i) broadcast
        )

        # j >= i のペアをマスク（antecedent は必ず前に現れる）
        causal_mask = torch.tril(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1
        )  # [T, T]  j < i が True
        pair_scores = pair_scores.masked_fill(
            ~causal_mask.unsqueeze(0), float("-inf")
        )

        # ---- 修正: max_top_antecedents の適用 ----
        # mention i に対して、位置が離れすぎている antecedent j をマスク
        # (i - j > max_top_antecedents のペアを無効化)
        if self.max_top_antecedents < T - 1:
            too_far = dist > self.max_top_antecedents  # [T, T]
            pair_scores = pair_scores.masked_fill(
                too_far.unsqueeze(0), float("-inf")
            )

        # top_mask で無効 mention をマスク
        pair_scores = pair_scores.masked_fill(
            ~top_mask.unsqueeze(2), float("-inf")
        )
        pair_scores = pair_scores.masked_fill(
            ~top_mask.unsqueeze(1), float("-inf")
        )

        # dummy antecedent (index 0) のスコアは 0
        dummy = torch.zeros(B, T, 1, device=device)
        ant_scores = torch.cat([dummy, pair_scores], dim=-1)  # [B, T, T+1]

        return ant_scores

    def _coref_loss(
        self,
        ant_scores: torch.Tensor,                  # [B, T, T+1]
        top_indices: torch.Tensor,                 # [B, T]
        top_mask: torch.Tensor,                    # [B, T]
        spans: torch.Tensor,                       # [B, K, 2]
        coref_clusters: list[list[list[tuple]]],
        B: int,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Lee et al. (2018) スタイルのマージン損失。
        gold antecedent が存在する mention に対して、
        gold antecedent のスコアの対数尤度を最大化する。
        """
        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for b in range(B):
            # mention インデックス → スパン (start, end) のマッピング
            top_span_list = [
                (spans[b, top_indices[b, t], 0].item(),
                 spans[b, top_indices[b, t], 1].item())
                for t in range(T)
            ]
            span_to_top: dict[tuple, int] = {
                sp: t for t, sp in enumerate(top_span_list)
                if top_mask[b, t].item()
            }

            # クラスタから gold antecedent マッピングを構築
            # gold_ant[i] = i が持つ gold antecedent の top インデックスリスト
            gold_ants: dict[int, list[int]] = {}
            for cluster in coref_clusters[b]:
                sorted_cluster = sorted(cluster)
                for idx, mention in enumerate(sorted_cluster):
                    if mention not in span_to_top:
                        continue
                    t_i = span_to_top[tuple(mention)]
                    for ant_mention in sorted_cluster[:idx]:
                        if ant_mention not in span_to_top:
                            continue
                        t_j = span_to_top[tuple(ant_mention)]
                        gold_ants.setdefault(t_i, []).append(t_j + 1)  # +1 for dummy offset

            for t_i, ant_list in gold_ants.items():
                if not ant_list:
                    continue
                log_norm = torch.logsumexp(ant_scores[b, t_i], dim=0)
                log_gold = torch.logsumexp(
                    ant_scores[b, t_i, ant_list], dim=0
                )
                total_loss = total_loss + log_norm - log_gold
                count += 1

        if count == 0:
            return total_loss
        return total_loss / count

    @staticmethod
    def _bucket_distance(dist: torch.Tensor) -> torch.Tensor:
        """距離を 10 バケットに変換。Lee et al. 2018 方式。"""
        # 0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+
        buckets = torch.zeros_like(dist)
        buckets = torch.where(dist >= 1,  torch.ones_like(dist),     buckets)
        buckets = torch.where(dist >= 2,  torch.full_like(dist, 2),  buckets)
        buckets = torch.where(dist >= 3,  torch.full_like(dist, 3),  buckets)
        buckets = torch.where(dist >= 4,  torch.full_like(dist, 4),  buckets)
        buckets = torch.where(dist >= 5,  torch.full_like(dist, 5),  buckets)
        buckets = torch.where(dist >= 8,  torch.full_like(dist, 6),  buckets)
        buckets = torch.where(dist >= 16, torch.full_like(dist, 7),  buckets)
        buckets = torch.where(dist >= 32, torch.full_like(dist, 8),  buckets)
        buckets = torch.where(dist >= 64, torch.full_like(dist, 9),  buckets)
        return buckets
