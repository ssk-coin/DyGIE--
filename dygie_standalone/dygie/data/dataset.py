"""
DyGIE++ Standalone — Dataset
SciERC / DyGIE++ JSON Lines 形式を読み込み、PyTorch Dataset として提供します。

各行の形式:
{
  "doc_key": str,
  "dataset": str,
  "sentences": [[token, ...], ...],          # 必須
  "ner":       [[[start, end, label], ...], ...],   # optional
  "relations": [[[s1, e1, s2, e2, label], ...], ...], # optional
  "clusters":  [[[start, end], ...], ...]    # optional (coref)
}
すべてのインデックスはドキュメント全体でのフラットなトークンインデックス。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class DyGIEDataset(Dataset):
    """
    DyGIE++ JSON Lines データセット。

    Parameters
    ----------
    path : str | Path
        .jsonl ファイルのパス。
    tokenizer : PreTrainedTokenizerFast
        HuggingFace トークナイザ（fast tokenizer 推奨）。
    max_span_width : int
        列挙するスパンの最大幅（トークン数）。
    max_total_length : int
        入力 token id の最大長（超えるドキュメントは先頭から切り捨て）。
    ner_labels : list[str] | None
        NER ラベル一覧（None で自動収集）。
    rel_labels : list[str] | None
        Relation ラベル一覧（None で自動収集）。
    use_ner : bool
        NER タスクを有効にするか。
    use_rel : bool
        RE タスクを有効にするか。
    use_coref : bool
        Coref タスクを有効にするか。
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerFast,
        max_span_width: int = 8,
        max_total_length: int = 512,
        ner_labels: list[str] | None = None,
        rel_labels: list[str] | None = None,
        use_ner: bool = True,
        use_rel: bool = True,
        use_coref: bool = True,
        max_spans: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_span_width = max_span_width
        self.max_total_length = max_total_length
        self.use_ner = use_ner
        self.use_rel = use_rel
        self.use_coref = use_coref
        self.max_spans = max_spans  # 0 = 上限なし

        # ---- ラベル辞書（外部指定 or ファイルから自動構築） ----
        self._ner_labels: list[str] = list(ner_labels) if ner_labels else []
        self._rel_labels: list[str] = list(rel_labels) if rel_labels else []

        self.raw_docs: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.raw_docs.append(json.loads(line))

        # ラベル自動収集（外部指定がない場合）
        if not self._ner_labels and use_ner:
            self._ner_labels = self._collect_labels("ner", index=2)
        if not self._rel_labels and use_rel:
            self._rel_labels = self._collect_labels("relations", index=4)

        self.ner_label2id: dict[str, int] = {
            lbl: i + 1 for i, lbl in enumerate(self._ner_labels)
        }  # 0 = "no entity"
        self.rel_label2id: dict[str, int] = {
            lbl: i + 1 for i, lbl in enumerate(self._rel_labels)
        }  # 0 = "no relation"

        # キャッシュ済みサンプル
        self._cache: list[dict[str, Any]] = [
            self._process(doc) for doc in self.raw_docs
        ]

        logger.info(
            "Loaded %d documents from %s | NER labels: %d | RE labels: %d",
            len(self._cache),
            path,
            len(self._ner_labels),
            len(self._rel_labels),
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def ner_labels(self) -> list[str]:
        return self._ner_labels

    @property
    def rel_labels(self) -> list[str]:
        return self._rel_labels

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._cache[idx]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_labels(self, field: str, index: int) -> list[str]:
        """フィールドからユニークなラベルをソート済みリストとして収集する。"""
        labels: set[str] = set()
        for doc in self.raw_docs:
            for sent_annots in doc.get(field, []):
                for annot in sent_annots:
                    if len(annot) > index:
                        labels.add(annot[index])
        return sorted(labels)

    def _process(self, doc: dict[str, Any]) -> dict[str, Any]:
        """
        生ドキュメントをモデル入力テンソルに変換する。

        Returns
        -------
        dict with keys:
          doc_key          : str
          input_ids        : LongTensor [L]
          attention_mask   : LongTensor [L]
          token_to_subword : LongTensor [N]   各トークンの先頭 subword インデックス
          subword_to_token : LongTensor [L]   各 subword のトークンインデックス (-1 = special)
          sentence_offsets : LongTensor [S, 2] 文ごとのトークン開始・終了
          spans            : LongTensor [K, 2] 列挙済みスパン (start, end) ※ end inclusive
          ner_labels       : LongTensor [K]    スパンごとの NER ラベル (0=none)
          rel_labels       : LongTensor [K, K] スパンペアの RE ラベル (0=none)
          coref_clusters   : list[list[tuple]] クラスタ（評価用）
        """
        sentences: list[list[str]] = doc["sentences"]
        doc_key: str = doc.get("doc_key", "")

        # ---- (1) トークン全結合 & subword エンコード ----
        flat_tokens: list[str] = []
        sentence_offsets: list[tuple[int, int]] = []
        for sent in sentences:
            start = len(flat_tokens)
            flat_tokens.extend(sent)
            sentence_offsets.append((start, len(flat_tokens) - 1))

        num_tokens = len(flat_tokens)

        # word_ids() を使うために word ごとにエンコード
        encoding = self.tokenizer(
            flat_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_total_length,
            padding=False,
        )
        input_ids: torch.Tensor = encoding["input_ids"][0]       # [L]
        attention_mask: torch.Tensor = encoding["attention_mask"][0]  # [L]
        word_ids: list[int | None] = encoding.word_ids(batch_index=0)

        L = input_ids.size(0)

        # subword → token マッピング（-1 = [CLS]/[SEP] 等の special token）
        subword_to_token = torch.full((L,), -1, dtype=torch.long)
        # token → 先頭 subword マッピング
        token_to_subword = torch.zeros(num_tokens, dtype=torch.long)
        seen_words: set[int] = set()
        for sw_idx, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            subword_to_token[sw_idx] = w_id
            if w_id not in seen_words:
                token_to_subword[w_id] = sw_idx
                seen_words.add(w_id)

        # トークン数が truncation で減っている場合の補正
        valid_num_tokens = int(subword_to_token.max().item()) + 1 if len(seen_words) > 0 else 0

        # ---- (2) スパン列挙 ----
        spans: list[tuple[int, int]] = []
        for start in range(valid_num_tokens):
            for end in range(start, min(start + self.max_span_width, valid_num_tokens)):
                spans.append((start, end))

        spans_tensor = torch.tensor(spans, dtype=torch.long)  # [K, 2]
        K = len(spans)
        span_index: dict[tuple[int, int], int] = {s: i for i, s in enumerate(spans)}

        # ---- (3) NER ラベル ----
        ner_labels_tensor = torch.zeros(K, dtype=torch.long)
        if self.use_ner:
            # DyGIE++ の NER インデックスはドキュメント全体のフラットインデックスなので
            # 文ごとのループでも tok_offset 補正は不要
            for s_idx, sent in enumerate(sentences):
                for annot in doc.get("ner", [[]] * len(sentences))[s_idx] if s_idx < len(doc.get("ner", [])) else []:
                    gs, ge, lbl = annot[0], annot[1], annot[2]
                    key = (gs, ge)
                    if key in span_index and lbl in self.ner_label2id:
                        ner_labels_tensor[span_index[key]] = self.ner_label2id[lbl]

        # ---- (4) RE ラベル ----
        rel_labels_tensor = torch.zeros((K, K), dtype=torch.long)
        if self.use_rel:
            for s_idx, sent in enumerate(sentences):
                for annot in doc.get("relations", [[]] * len(sentences))[s_idx] if s_idx < len(doc.get("relations", [])) else []:
                    s1, e1, s2, e2, lbl = annot[0], annot[1], annot[2], annot[3], annot[4]
                    k1 = span_index.get((s1, e1))
                    k2 = span_index.get((s2, e2))
                    if k1 is not None and k2 is not None and lbl in self.rel_label2id:
                        rel_labels_tensor[k1, k2] = self.rel_label2id[lbl]

        # ---- (5) Coref クラスタ（評価用・生データのまま保持） ----
        coref_clusters: list[list[tuple[int, int]]] = []
        if self.use_coref:
            for cluster in doc.get("clusters", []):
                coref_clusters.append([tuple(mention) for mention in cluster])

        # ---- (6) スパン数の上限設定（メモリ節約）----
        # max_spans > 0 のとき、先頭 max_spans 件に切り捨てる。
        # RE の pair 行列は O(K²) のため、K を抑えるとメモリが大幅に削減される。
        if self.max_spans > 0 and K > self.max_spans:
            cap = self.max_spans
            spans_tensor = spans_tensor[:cap]
            ner_labels_tensor = ner_labels_tensor[:cap]
            rel_labels_tensor = rel_labels_tensor[:cap, :cap]

        return {
            "doc_key": doc_key,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_to_subword": token_to_subword,   # [N]
            "subword_to_token": subword_to_token,   # [L]
            "sentence_offsets": torch.tensor(sentence_offsets, dtype=torch.long),  # [S, 2]
            "spans": spans_tensor,                  # [K, 2]
            "ner_labels": ner_labels_tensor,        # [K]
            "rel_labels": rel_labels_tensor,        # [K, K]
            "coref_clusters": coref_clusters,       # list[list[tuple]]
            "num_tokens": valid_num_tokens,
        }
