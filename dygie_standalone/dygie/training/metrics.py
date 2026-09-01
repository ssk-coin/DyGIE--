"""
DyGIE++ Standalone — Metrics

NER: span-level micro F1（ラベル込み）
RE : span-pair-level micro F1（ラベル込み）
Coref: MUC / B³ / CEAFφ4 の平均（CoNLL スコア）

バグ修正 (v2):
  - CorefMetrics の recall 計算を修正。
    _muc / _b3 / _ceaf_phi4 は (precision, recall) を返すように変更し、
    _f1 内での recall 計算を正しく mean recall に修正した。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


# =====================================================================
# NER Metrics
# =====================================================================

class NERMetrics:
    """スパンレベルの NER F1 を計算する。

    Usage::
        metrics = NERMetrics()
        metrics.update(ner_preds, ner_labels, span_mask, spans)
        result = metrics.compute()
        metrics.reset()
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        # ラベルごとの集計
        self.tp_per_label: dict[int, int] = defaultdict(int)
        self.fp_per_label: dict[int, int] = defaultdict(int)
        self.fn_per_label: dict[int, int] = defaultdict(int)

    def update(
        self,
        preds: torch.Tensor,    # [B, K]
        golds: torch.Tensor,    # [B, K]
        span_mask: torch.Tensor,  # [B, K]
        spans: torch.Tensor | None = None,  # [B, K, 2] (未使用、API 統一のため)
    ) -> None:
        B, K = preds.shape
        for b in range(B):
            mask = span_mask[b]
            pred = preds[b][mask]
            gold = golds[b][mask]
            for p, g in zip(pred.tolist(), gold.tolist()):
                if g > 0 and p == g:
                    self.tp += 1
                    self.tp_per_label[g] += 1
                elif p > 0 and p != g:
                    self.fp += 1
                    self.fp_per_label[p] += 1
                    if g > 0:
                        self.fn += 1
                        self.fn_per_label[g] += 1
                elif g > 0 and p == 0:
                    self.fn += 1
                    self.fn_per_label[g] += 1
                elif p > 0 and g == 0:
                    self.fp += 1
                    self.fp_per_label[p] += 1

    def compute(self) -> dict[str, float]:
        p = self.tp / (self.tp + self.fp + 1e-9)
        r = self.tp / (self.tp + self.fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        return {"ner_precision": p, "ner_recall": r, "ner_f1": f1}


# =====================================================================
# Relation Metrics
# =====================================================================

class RelationMetrics:
    """スパンペアレベルの RE F1 を計算する。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(
        self,
        preds: torch.Tensor,      # [B, K, K]
        golds: torch.Tensor,      # [B, K, K]
        pair_mask: torch.Tensor,  # [B, K, K]
    ) -> None:
        for b in range(preds.size(0)):
            mask = pair_mask[b]
            pred = preds[b][mask]
            gold = golds[b][mask]
            for p, g in zip(pred.tolist(), gold.tolist()):
                if g > 0 and p == g:
                    self.tp += 1
                elif p > 0 and p != g:
                    self.fp += 1
                    if g > 0:
                        self.fn += 1
                elif g > 0 and p == 0:
                    self.fn += 1
                elif p > 0 and g == 0:
                    self.fp += 1

    def compute(self) -> dict[str, float]:
        p = self.tp / (self.tp + self.fp + 1e-9)
        r = self.tp / (self.tp + self.fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        return {"rel_precision": p, "rel_recall": r, "rel_f1": f1}


# =====================================================================
# Coreference Metrics  (MUC / B³ / CEAFφ4)
# =====================================================================

class CorefMetrics:
    """
    CoNLL-2012 スタイルの Coref スコアを計算する。
    MUC, B³, CEAFφ4 の平均 F1 = CoNLL スコア。

    Usage::
        metrics = CorefMetrics()
        metrics.update(pred_clusters, gold_clusters)
        result = metrics.compute()   # {"muc_f1": ..., "b3_f1": ..., "ceaf_f1": ..., "conll_f1": ...}
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # (precision, recall) ペアの累積リスト
        self._muc: list[tuple[float, float]] = []
        self._b3: list[tuple[float, float]] = []
        self._ceaf: list[tuple[float, float]] = []

    def update(
        self,
        pred_clusters: list[list[tuple[int, int]]],
        gold_clusters: list[list[tuple[int, int]]],
    ) -> None:
        self._muc.append(_muc(pred_clusters, gold_clusters))
        self._b3.append(_b3(pred_clusters, gold_clusters))
        self._ceaf.append(_ceaf_phi4(pred_clusters, gold_clusters))

    def compute(self) -> dict[str, float]:
        def _f1(pairs: list[tuple[float, float]]) -> dict[str, float]:
            """
            (precision, recall) ペアのリストから平均 P / R / F1 を計算する。
            修正: precision と recall を各々の平均として正しく計算する。
            """
            n = len(pairs) + 1e-9
            p = sum(t[0] for t in pairs) / n
            r = sum(t[1] for t in pairs) / n
            f = 2 * p * r / (p + r + 1e-9)
            return {"p": p, "r": r, "f1": f}

        muc  = _f1(self._muc)
        b3   = _f1(self._b3)
        ceaf = _f1(self._ceaf)
        conll_f1 = (muc["f1"] + b3["f1"] + ceaf["f1"]) / 3
        return {
            "muc_p": muc["p"],   "muc_r": muc["r"],   "muc_f1": muc["f1"],
            "b3_p":  b3["p"],    "b3_r":  b3["r"],    "b3_f1":  b3["f1"],
            "ceaf_p": ceaf["p"], "ceaf_r": ceaf["r"],  "ceaf_f1": ceaf["f1"],
            "conll_f1": conll_f1,
        }


# ------------------------------------------------------------------
# Coref scoring functions
# ------------------------------------------------------------------

Cluster = list[tuple[int, int]]


def _mentions(clusters: list[Cluster]) -> set[tuple[int, int]]:
    return {m for c in clusters for m in c}


def _muc(
    pred: list[Cluster], gold: list[Cluster]
) -> tuple[float, float]:
    """MUC metric. Returns (precision, recall)."""
    def _score(key: list[Cluster], response: list[Cluster]) -> float:
        total = 0.0
        for cluster in key:
            if len(cluster) == 1:
                continue
            # partition of cluster induced by response
            partitions: dict[int, set] = {}
            for m in cluster:
                assigned = -1
                for i, rc in enumerate(response):
                    if m in rc:
                        assigned = i
                        break
                partitions.setdefault(assigned, set()).add(m)
            total += len(cluster) - len(partitions)
        denom = sum(len(c) - 1 for c in key if len(c) > 1)
        return total / (denom + 1e-9)

    recall    = _score(gold, pred)
    precision = _score(pred, gold)
    return precision, recall


def _b3(
    pred: list[Cluster], gold: list[Cluster]
) -> tuple[float, float]:
    """B³ metric. Returns (precision, recall)."""
    gold_map: dict[tuple, int] = {}
    for i, c in enumerate(gold):
        for m in c:
            gold_map[tuple(m) if isinstance(m, list) else m] = i

    pred_map: dict[tuple, int] = {}
    for i, c in enumerate(pred):
        for m in c:
            pred_map[tuple(m) if isinstance(m, list) else m] = i

    all_mentions = set(gold_map) | set(pred_map)
    tp_p = tp_r = 0.0

    for m in all_mentions:
        g_id = gold_map.get(m, -1)
        p_id = pred_map.get(m, -1)
        # gold cluster mates
        gold_set = {x for x, gi in gold_map.items() if gi == g_id} if g_id >= 0 else set()
        pred_set = {x for x, pi in pred_map.items() if pi == p_id} if p_id >= 0 else set()
        if gold_set:
            tp_r += len(gold_set & pred_set) / len(gold_set)
        if pred_set:
            tp_p += len(gold_set & pred_set) / len(pred_set)

    n = len(all_mentions)
    p = tp_p / (n + 1e-9)
    r = tp_r / (n + 1e-9)
    return p, r


def _ceaf_phi4(
    pred: list[Cluster], gold: list[Cluster]
) -> tuple[float, float]:
    """CEAFφ4 metric using optimal alignment (Hungarian algorithm). Returns (precision, recall)."""
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    def phi4(c1: Cluster, c2: Cluster) -> float:
        s1 = {tuple(m) if isinstance(m, list) else m for m in c1}
        s2 = {tuple(m) if isinstance(m, list) else m for m in c2}
        return 2 * len(s1 & s2) / (len(s1) + len(s2) + 1e-9)

    if not pred or not gold:
        return 0.0, 0.0

    cost = [[phi4(g, p) for p in pred] for g in gold]
    cost_arr = np.array(cost)
    row_ind, col_ind = linear_sum_assignment(-cost_arr)
    numerator = sum(cost_arr[r, c] for r, c in zip(row_ind, col_ind))
    p = numerator / (len(pred) + 1e-9)
    r = numerator / (len(gold) + 1e-9)
    return p, r
