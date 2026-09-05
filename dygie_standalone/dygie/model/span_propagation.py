"""
DyGIE-- — Span Graph Propagation Module (v5)

Wadden et al. (2019) Section 3.3 の GRU スタイルスパングラフ伝播を実装する。

Coreference クラスタを辺とするスパングラフを構築し、
attention-weighted 近傍集約 + GRU ゲーティングでスパン表現を更新する。

アルゴリズム:
  1. antecedent_scores から coref クラスタをデコード（argmax + union-find, no_grad）
  2. 各スパン i の近傍集合 N(i) = 同一クラスタ内の他スパン
  3. 注意重み: α_ij = softmax_{j∈N(i)} (w^T g_j)
  4. 近傍集約: f_i = Σ_{j∈N(i)} α_ij * g_j
  5. GRU 更新:
       u_i = σ(W_u [g_i; f_i])
       r_i = σ(W_r [g_i; f_i])
       c_i = tanh(W_c [r_i⊙g_i; f_i])
       g_i' = (1-u_i)⊙g_i + u_i⊙c_i
  近傍なし（シングルトン）スパンは g_i' = g_i のまま。

参考:
  Wadden et al. (2019) "Entity, Relation, and Event Extraction with Contextualized
  Span Representations." EMNLP. https://arxiv.org/abs/1909.03546
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpanPropagation(nn.Module):
    """
    GRU スタイルスパングラフ伝播モジュール。

    Coref クラスタを辺とするスパングラフを構築し、
    GRU スタイルのゲーティングでスパン表現を更新する。
    NER・RE ヘッドが実行される前に span_repr を更新することで、
    同一エンティティの複数スパン間で情報を共有できる。

    Parameters
    ----------
    span_dim : int
        スパン表現の次元数。
    dropout : float
    """

    def __init__(self, span_dim: int, dropout: float = 0.4) -> None:
        super().__init__()
        self.span_dim = span_dim

        # 近傍注意スコアラー: g_j → scalar  (w^T g_j, bias なし)
        self.neighbor_attention = nn.Linear(span_dim, 1, bias=False)

        # GRU ゲート (入力次元: [g_i; f_i] = 2 * span_dim)
        self.gate_u = nn.Linear(span_dim * 2, span_dim)  # update gate
        self.gate_r = nn.Linear(span_dim * 2, span_dim)  # reset gate
        self.gate_c = nn.Linear(span_dim * 2, span_dim)  # candidate

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Static helper: cluster decoding (no gradient)
    # ------------------------------------------------------------------

    @staticmethod
    def decode_clusters(
        ant_scores: torch.Tensor,   # [T, T+1]
        top_indices: torch.Tensor,  # [T]
        top_mask: torch.Tensor,     # [T]
    ) -> list[list[int]]:
        """Antecedent スコアから coref クラスタをデコードする。

        各 mention の最良 antecedent（argmax; index 0 = dummy）を
        union-find で繋いでクラスタを形成する。

        Returns
        -------
        clusters : list[list[int]]
            K-space スパンインデックスのクラスタリスト。
            サイズ 2 以上のクラスタのみ含む（シングルトン除外）。
        """
        T = top_indices.size(0)

        # union-find（path compression）
        parent = list(range(T))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # argmax で各 mention の最良 antecedent を選択
        with torch.no_grad():
            best_ants = ant_scores.argmax(dim=-1).tolist()  # [T]

        for t in range(T):
            if not top_mask[t].item():
                continue
            ant = best_ants[t] - 1  # index 0 = dummy → -1
            if ant >= 0 and top_mask[ant].item():
                union(t, ant)

        # クラスタ化: T-space index → K-space span index に変換
        top_idx_list = top_indices.tolist()
        cluster_dict: dict[int, list[int]] = {}
        for t in range(T):
            if not top_mask[t].item():
                continue
            root = find(t)
            k_idx = int(top_idx_list[t])
            cluster_dict.setdefault(root, []).append(k_idx)

        # シングルトンを除外（近傍がないと伝播不要）
        return [members for members in cluster_dict.values() if len(members) >= 2]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        span_repr: torch.Tensor,           # [B, K, span_dim]
        top_span_indices: torch.Tensor,    # [B, T]
        top_span_mask: torch.Tensor,       # [B, T]
        antecedent_scores: torch.Tensor,   # [B, T, T+1]
    ) -> torch.Tensor:
        """スパン表現を GRU スタイルのグラフ伝播で更新する。

        Parameters
        ----------
        span_repr : [B, K, span_dim]
            全スパンの表現。
        top_span_indices : [B, T]
            Coref モジュールが選択した top-T スパンの K-space インデックス。
        top_span_mask : [B, T]
            top-T スパンの有効マスク（パディング部分が False）。
        antecedent_scores : [B, T, T+1]
            Coref antecedent スコア（index 0 = dummy antecedent）。

        Returns
        -------
        updated_span_repr : [B, K, span_dim]
            GRU スタイルで更新されたスパン表現。
            Coref クラスタに属さないスパンは元の表現のまま。
        """
        B = span_repr.size(0)
        device = span_repr.device

        # 出力テンソル（クラスタに属さないスパンは元表現を保持）
        out = span_repr.clone()

        for b in range(B):
            clusters = self.decode_clusters(
                antecedent_scores[b],
                top_span_indices[b],
                top_span_mask[b],
            )
            if not clusters:
                continue

            for cluster in clusters:
                E = len(cluster)
                e_idx = torch.tensor(cluster, dtype=torch.long, device=device)

                # クラスタ内スパンの元表現: [E, D]
                e_repr = span_repr[b].index_select(0, e_idx)
                e_repr = self.dropout(e_repr)

                # ---- 近傍注意重み ----
                # α[i, j] = softmax_{j ≠ i} (w^T g_j)
                attn_scores = self.neighbor_attention(e_repr).squeeze(-1)  # [E]
                # 対角を -inf でマスク（自己ループ除外）
                diag_mask = torch.eye(E, dtype=torch.bool, device=device)
                attn_matrix = (
                    attn_scores.unsqueeze(0).expand(E, -1)
                    .masked_fill(diag_mask, float("-inf"))
                )  # [E, E]
                alpha = F.softmax(attn_matrix, dim=1)  # [E, E]

                # ---- 近傍集約: f_i = Σ_{j≠i} α_ij * g_j ----
                f = alpha @ e_repr  # [E, D]

                # ---- GRU スタイル更新 ----
                # update gate: u = σ(W_u [g; f])
                concat_gf = torch.cat([e_repr, f], dim=-1)  # [E, 2D]
                u = torch.sigmoid(self.gate_u(concat_gf))   # [E, D]

                # reset gate: r = σ(W_r [g; f])
                r = torch.sigmoid(self.gate_r(concat_gf))   # [E, D]

                # candidate: c = tanh(W_c [r⊙g; f])
                concat_rf = torch.cat([r * e_repr, f], dim=-1)  # [E, 2D]
                c = torch.tanh(self.gate_c(concat_rf))           # [E, D]

                # 更新: g' = (1-u)⊙g + u⊙c
                g_new = (1.0 - u) * e_repr + u * c  # [E, D]

                # K-space に書き戻す
                out[b].index_copy_(0, e_idx, g_new)

        return out
