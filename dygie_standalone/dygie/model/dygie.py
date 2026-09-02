"""
DyGIE++ Standalone — メインモデル

Transformer エンコーダ + スパン抽出 + NER / RE / Coref ヘッドを統合します。
AllenNLP には一切依存しません。

設計方針:
  - バッチ内でスパン数が異なるため、span_mask でパディングを管理
  - 学習時は gold NER スパンで RE を計算（use_gold_spans=True）
  - 推論時は predicted NER スパンで RE を計算
  - 損失の合計は各タスクの重み付き和

改善点 (v2):
  - save_pretrained / from_pretrained が dygie_config.json に DyGIE 固有の
    ハイパーパラメータを保存・復元するように対応。
    モデルのロードに kwargs を全て手動で指定する必要がなくなった。
  - torch.load に weights_only=True を追加（セキュリティ改善・PyTorch 警告抑制）

メモリ最適化 (v3):
  - use_gradient_checkpointing=True で Transformer エンコーダに勾配チェックポイントを
    適用。エンコーダの活性化メモリを約 50〜70% 削減（学習速度は約 1.3x 低下）。
  - RE モジュールを K×K → E×E（エンティティスパンのみ）に変更済み。

RE スコア改善 (v4):
  - エンティティタイプ埋め込みを RE ペア表現に追加（type_embedding_dim で制御）
  - スパン間距離特徴（use_distance_feature / distance_embedding_dim で制御）
  - Focal Loss による RE クラス不均衡への対処（focal_loss_gamma で制御）
  - Pair MLP を 2 層 + LayerNorm に深化

グラフ構造統合 (v5):
  - DyGIE++ 論文 (Wadden et al., 2019) Section 3.3 のスパングラフ伝播を実装。
  - フォワードパスの実行順序を論文に従って変更:
      span_repr → Coref → SpanPropagation → NER → RE
  - Coref クラスタを辺とするスパングラフを構築し、GRU スタイルのゲーティングで
    スパン表現を更新してから NER・RE を計算する。
    同一エンティティの複数スパン間で情報共有できるため、NER・RE 精度が向上する。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from .span_extractor import SpanExtractor
from .ner_module import NERModule
from .rel_module import RelationModule
from .coref_module import CorefModule
from .span_propagation import SpanPropagation
from .event_module import EventModule

logger = logging.getLogger(__name__)


class DyGIE(nn.Module):
    """
    Parameters
    ----------
    transformer_model : str
        HuggingFace モデル名またはローカルパス。
        例: "bert-base-cased", "allenai/scibert_scivocab_cased"
    ner_labels : list[str]
        NER ラベル一覧（"no entity" を除く）。
    rel_labels : list[str]
        Relation ラベル一覧（"no relation" を除く）。
    max_span_width : int
        最大スパン幅（トークン数）。
    use_ner : bool
    use_rel : bool
    use_coref : bool
    ner_loss_weight : float
    rel_loss_weight : float
    coref_loss_weight : float
    width_embedding_dim : int
    feedforward_dim : int
    use_attentive_pooling : bool
    spans_per_word : float
        Coref mention pruning の割合。
    max_top_antecedents : int
    dropout : float
    type_embedding_dim : int
        RE エンティティタイプ埋め込みの次元 (0 で無効)。
    use_distance_feature : bool
        RE スパン間距離特徴を使用するか。
    num_distance_buckets : int
        距離バケット数。
    distance_embedding_dim : int
        距離埋め込みの次元 (0 で無効)。
    focal_loss_gamma : float
        RE Focal Loss の gamma 値 (0 で通常の CE)。
    use_event : bool
        イベント抽出を有効にするか。
    event_type_labels : list[str]
        イベントタイプラベル一覧（"no trigger" を除く）。
    arg_role_labels : list[str]
        引数ロールラベル一覧（"no role" を除く）。
    event_loss_weight : float
        イベント抽出損失の重み（trigger_loss + arg_loss の合計に掛ける）。
    """

    def __init__(
        self,
        transformer_model: str,
        ner_labels: list[str],
        rel_labels: list[str],
        max_span_width: int = 8,
        use_ner: bool = True,
        use_rel: bool = True,
        use_coref: bool = True,
        ner_loss_weight: float = 1.0,
        rel_loss_weight: float = 1.0,
        coref_loss_weight: float = 1.0,
        width_embedding_dim: int = 128,
        feedforward_dim: int = 150,
        use_attentive_pooling: bool = True,
        spans_per_word: float = 0.4,
        max_top_antecedents: int = 50,
        dropout: float = 0.4,
        use_gradient_checkpointing: bool = False,
        type_embedding_dim: int = 0,
        use_distance_feature: bool = False,
        num_distance_buckets: int = 10,
        distance_embedding_dim: int = 64,
        focal_loss_gamma: float = 0.0,
        # イベント抽出 (v5)
        use_event: bool = False,
        event_type_labels: list[str] | None = None,
        arg_role_labels: list[str] | None = None,
        event_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.use_ner = use_ner
        self.use_rel = use_rel
        self.use_coref = use_coref
        self.use_event = use_event
        self.ner_loss_weight = ner_loss_weight
        self.rel_loss_weight = rel_loss_weight
        self.coref_loss_weight = coref_loss_weight
        self.event_loss_weight = event_loss_weight
        self.ner_labels = ner_labels
        self.rel_labels = rel_labels
        self.event_type_labels = list(event_type_labels) if event_type_labels else []
        self.arg_role_labels = list(arg_role_labels) if arg_role_labels else []

        # save_pretrained / from_pretrained 用に初期化パラメータを記録
        self._init_config: dict[str, Any] = {
            "transformer_model": transformer_model,
            "ner_labels": list(ner_labels),
            "rel_labels": list(rel_labels),
            "max_span_width": max_span_width,
            "use_ner": use_ner,
            "use_rel": use_rel,
            "use_coref": use_coref,
            "ner_loss_weight": ner_loss_weight,
            "rel_loss_weight": rel_loss_weight,
            "coref_loss_weight": coref_loss_weight,
            "width_embedding_dim": width_embedding_dim,
            "feedforward_dim": feedforward_dim,
            "use_attentive_pooling": use_attentive_pooling,
            "spans_per_word": spans_per_word,
            "max_top_antecedents": max_top_antecedents,
            "dropout": dropout,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            # v4: RE スコア改善
            "type_embedding_dim": type_embedding_dim,
            "use_distance_feature": use_distance_feature,
            "num_distance_buckets": num_distance_buckets,
            "distance_embedding_dim": distance_embedding_dim,
            "focal_loss_gamma": focal_loss_gamma,
            # v5: イベント抽出
            "use_event": use_event,
            "event_type_labels": self.event_type_labels,
            "arg_role_labels": self.arg_role_labels,
            "event_loss_weight": event_loss_weight,
        }

        # ---- Transformer encoder ----
        self.encoder = AutoModel.from_pretrained(transformer_model)

        # 勾配チェックポイント: エンコーダの活性化メモリを 50〜70% 削減
        # （学習速度は約 1.3x 低下するトレードオフ）
        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for encoder.")
        hidden_size: int = self.encoder.config.hidden_size

        # ---- Span extractor ----
        self.span_extractor = SpanExtractor(
            hidden_size=hidden_size,
            max_span_width=max_span_width,
            width_embedding_dim=width_embedding_dim,
            use_attentive_pooling=use_attentive_pooling,
            dropout=dropout,
        )
        span_dim = self.span_extractor.span_dim

        # ---- Task heads ----
        if use_ner and ner_labels:
            self.ner_module = NERModule(
                span_dim=span_dim,
                num_ner_labels=len(ner_labels),
                feedforward_dim=feedforward_dim,
                dropout=dropout,
            )
        else:
            self.ner_module = None  # type: ignore

        if use_rel and rel_labels:
            self.rel_module = RelationModule(
                span_dim=span_dim,
                num_rel_labels=len(rel_labels),
                num_ner_labels=len(ner_labels) if (use_ner and ner_labels) else 0,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                type_embedding_dim=type_embedding_dim,
                use_distance_feature=use_distance_feature,
                num_distance_buckets=num_distance_buckets,
                distance_embedding_dim=distance_embedding_dim,
                focal_loss_gamma=focal_loss_gamma,
            )
        else:
            self.rel_module = None  # type: ignore

        if use_coref:
            self.coref_module = CorefModule(
                span_dim=span_dim,
                feedforward_dim=feedforward_dim,
                spans_per_word=spans_per_word,
                max_top_antecedents=max_top_antecedents,
                dropout=dropout,
            )
            # スパングラフ伝播: Coref クラスタを辺として NER/RE 前に span_repr を更新
            # (DyGIE++ 論文 Section 3.3 / Wadden et al., 2019)
            self.span_prop = SpanPropagation(span_dim=span_dim, dropout=dropout)
        else:
            self.coref_module = None  # type: ignore
            self.span_prop = None  # type: ignore

        # ---- イベント抽出ヘッド ----
        if use_event and event_type_labels:
            self.event_module = EventModule(
                span_dim=span_dim,
                num_event_types=len(self.event_type_labels),
                num_arg_roles=len(self.arg_role_labels),
                feedforward_dim=feedforward_dim,
                dropout=dropout,
            )
        else:
            self.event_module = None  # type: ignore

        logger.info(
            "DyGIE initialized | encoder=%s | NER=%s | RE=%s | Coref=%s | Event=%s | span_dim=%d",
            transformer_model,
            use_ner,
            use_rel,
            use_coref,
            use_event,
            span_dim,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,           # [B, L]
        attention_mask: torch.Tensor,      # [B, L]
        token_to_subword: torch.Tensor,    # [B, N]
        spans: torch.Tensor,               # [B, K, 2]
        span_mask: torch.Tensor,           # [B, K]
        num_tokens: torch.Tensor,          # [B]
        ner_labels: torch.Tensor | None = None,                 # [B, K]
        rel_labels: torch.Tensor | None = None,                 # [B, K, K]
        coref_clusters: list | None = None,
        event_trigger_labels: torch.Tensor | None = None,       # [B, K]
        event_arg_labels: torch.Tensor | None = None,           # [B, K, K]
        use_gold_spans: bool = True,
    ) -> dict[str, Any]:
        """
        Returns
        -------
        dict:
          loss          : total weighted loss (学習時のみ)
          ner_loss      : NER loss (学習時のみ)
          rel_loss      : RE loss (学習時のみ)
          coref_loss    : Coref loss (学習時のみ)
          ner_logits    : [B, K, num_ner_labels+1]
          ner_preds     : [B, K]
          rel_logits    : [B, K, K, num_rel_labels+1]
          rel_preds     : [B, K, K]
          mention_scores      : [B, K]       (coref)
          top_span_indices    : [B, T]       (coref)
          antecedent_scores   : [B, T, T+1]  (coref)
        """
        # ---- エンコード ----
        encoder_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output: torch.Tensor = encoder_out.last_hidden_state  # [B, L, H]

        # ---- スパン表現抽出 ----
        span_repr = self.span_extractor(
            sequence_output=sequence_output,
            token_to_subword=token_to_subword,
            spans=spans,
            span_mask=span_mask,
        )  # [B, K, span_dim]

        output: dict[str, Any] = {}
        total_loss = torch.tensor(0.0, device=input_ids.device)

        # ---- Coreference (論文に従い NER/RE より先に実行) ----
        # DyGIE++ (Wadden et al., 2019) の実行順序:
        #   span_repr → Coref → SpanPropagation → NER → RE
        if self.coref_module is not None:
            coref_out = self.coref_module(
                span_repr=span_repr,
                span_mask=span_mask,
                spans=spans,
                num_tokens=num_tokens,
                coref_clusters=coref_clusters,
            )
            output.update(coref_out)
            if "coref_loss" in coref_out:
                total_loss = total_loss + self.coref_loss_weight * coref_out["coref_loss"]

            # ---- Span Graph Propagation (Section 3.3) ----
            # Coref クラスタを辺として span_repr を GRU スタイルで更新する。
            # 更新後の span_repr を NER・RE で使用することで、
            # 同一エンティティの複数スパン間で情報共有が可能になる。
            if self.span_prop is not None:
                span_repr = self.span_prop(
                    span_repr=span_repr,
                    top_span_indices=coref_out["top_span_indices"],
                    top_span_mask=coref_out["top_span_mask"],
                    antecedent_scores=coref_out["antecedent_scores"],
                )

        # ---- NER (伝播後の span_repr を使用) ----
        if self.ner_module is not None:
            ner_out = self.ner_module(
                span_repr=span_repr,
                span_mask=span_mask,
                ner_labels=ner_labels,
            )
            output.update(ner_out)
            if "ner_loss" in ner_out:
                total_loss = total_loss + self.ner_loss_weight * ner_out["ner_loss"]
        else:
            output["ner_preds"] = torch.zeros(
                span_repr.shape[:2], dtype=torch.long, device=input_ids.device
            )

        # ---- Relation Extraction (伝播後の span_repr を使用) ----
        if self.rel_module is not None:
            rel_out = self.rel_module(
                span_repr=span_repr,
                span_mask=span_mask,
                ner_preds=output["ner_preds"],
                rel_labels=rel_labels,
                ner_labels=ner_labels,
                spans=spans,                   # 距離特徴量のためにスパン位置を渡す
                use_gold_spans=use_gold_spans,
            )
            output.update(rel_out)
            if "rel_loss" in rel_out:
                total_loss = total_loss + self.rel_loss_weight * rel_out["rel_loss"]

        # ---- Event Extraction ----
        # 実行順序: Coref → SpanProp → NER → RE → Event
        # イベントトリガー検出と引数抽出は伝播後の span_repr を使用する。
        if self.event_module is not None:
            event_out = self.event_module(
                span_repr=span_repr,
                span_mask=span_mask,
                event_trigger_labels=event_trigger_labels,
                event_arg_labels=event_arg_labels,
                use_gold_triggers=use_gold_spans,
            )
            # キー名に "event_" プレフィックスを付けて output に統合
            for k, v in event_out.items():
                output[f"event_{k}" if not k.startswith("event_") else k] = v
            if "event_loss" in event_out:
                total_loss = total_loss + self.event_loss_weight * event_out["event_loss"]

        if (
            ner_labels is not None
            or rel_labels is not None
            or coref_clusters is not None
            or event_trigger_labels is not None
        ):
            output["loss"] = total_loss

        return output

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_pretrained(self, output_dir: str | Path) -> None:
        """モデル全体を output_dir に保存する。

        保存ファイル:
          model.pt          — PyTorch state dict
          config.json       — Transformer encoder config
          dygie_config.json — DyGIE 固有のハイパーパラメータ
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "model.pt")
        # encoder の config も保存
        self.encoder.config.save_pretrained(output_dir)
        # DyGIE 固有の設定を保存
        with open(output_dir / "dygie_config.json", "w", encoding="utf-8") as f:
            json.dump(self._init_config, f, indent=2, ensure_ascii=False)
        logger.info("Model saved to %s", output_dir)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        **kwargs: Any,
    ) -> "DyGIE":
        """保存済みモデルを読み込む。

        dygie_config.json が存在する場合はそこからハイパーパラメータを復元する。
        kwargs で個別パラメータを上書きすることも可能。

        読み込み手順:
          1. dygie_config.json からハイパーパラメータ（transformer_model 名を含む）を復元
          2. transformer_model 名からエンコーダのアーキテクチャを初期化
          3. model.pt から fine-tuned 全パラメータをロード（エンコーダ含む）

        NOTE: transformer_model は元の HuggingFace モデル名のまま使用する。
        model_dir には encoder の HuggingFace 形式の重みファイルは不要。
        """
        model_dir = Path(model_dir)

        # DyGIE 固有の設定を読み込む
        dygie_config_path = model_dir / "dygie_config.json"
        if dygie_config_path.exists():
            with open(dygie_config_path, encoding="utf-8") as f:
                saved_config: dict[str, Any] = json.load(f)
            # kwargs で上書き（transformer_model の上書きも可能）
            saved_config.update(kwargs)
            kwargs = saved_config
        else:
            # 旧形式: kwargs に全ハイパーパラメータが必要
            # transformer_model が未指定の場合のみ model_dir を使う（後方互換）
            logger.warning(
                "dygie_config.json not found in %s. "
                "All DyGIE hyperparameters must be supplied as kwargs.",
                model_dir,
            )
            if "transformer_model" not in kwargs:
                kwargs["transformer_model"] = str(model_dir)

        # cls(**kwargs) でエンコーダのアーキテクチャを初期化
        # (transformer_model は元の HuggingFace モデル名 or ローカルパス)
        model = cls(**kwargs)

        # model.pt から fine-tuned 全パラメータをロード（エンコーダ重みも上書き）
        state_dict = torch.load(
            model_dir / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        logger.info("Model loaded from %s", model_dir)
        return model
