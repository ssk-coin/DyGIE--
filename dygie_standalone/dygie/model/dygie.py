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
    use_lora : bool
        LoRA (Low-Rank Adaptation) を有効にするか。
        True にすると BERT エンコーダの大半を凍結し、LoRA アダプタ重みのみ学習する。
        小規模データでの過学習抑制・GPU メモリ削減に有効。
        peft ライブラリが必要: pip install peft
    lora_r : int
        LoRA のランク。低いほどパラメータが少ない（推奨: 4〜16）。
    lora_alpha : int
        LoRA のスケーリング係数。通常 lora_r の 2〜4 倍に設定する。
    lora_dropout : float
        LoRA アダプタ内の Dropout 率。
    lora_target_modules : list[str] | None
        LoRA を適用する BERT のモジュール名。
        None のとき BERT/SciBERT デフォルト ["query", "value"] を使用。
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
        # LoRA アダプタ (v6)
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: list[str] | None = None,
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
            # v6: LoRA
            "use_lora": use_lora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": lora_target_modules or ["query", "value"],
        }

        # ---- Transformer encoder ----
        self.encoder = AutoModel.from_pretrained(transformer_model)

        # ---- LoRA アダプタ (v6) ----
        # BERT エンコーダの大半を凍結し、低ランク行列のみ学習する。
        # 小規模データ (SciERC, ACE05 等) での過学習抑制・メモリ削減に有効。
        self.use_lora = use_lora
        if use_lora:
            try:
                from peft import get_peft_model, LoraConfig
            except ImportError as e:
                raise ImportError(
                    "LoRA を使用するには peft ライブラリが必要です: pip install peft"
                ) from e
            _target_modules = lora_target_modules or ["query", "value"]
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, lora_cfg)
            # 勾配チェックポイントと LoRA を併用する場合は入力勾配を有効化
            self.encoder.enable_input_require_grads()
            trainable, total = self._count_lora_params()
            logger.info(
                "LoRA enabled | r=%d, alpha=%d, target=%s | "
                "trainable params: %d / %d (%.2f%%)",
                lora_r, lora_alpha, _target_modules,
                trainable, total, 100 * trainable / max(total, 1),
            )

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
            "DyGIE initialized | encoder=%s | NER=%s | RE=%s | Coref=%s | Event=%s"
            " | LoRA=%s | span_dim=%d",
            transformer_model,
            use_ner, use_rel, use_coref, use_event, use_lora, span_dim,
        )

    # ------------------------------------------------------------------
    # LoRA ユーティリティ
    # ------------------------------------------------------------------

    def _count_lora_params(self) -> tuple[int, int]:
        """学習可能パラメータ数と総パラメータ数を返す。"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return trainable, total

    def print_trainable_parameters(self) -> None:
        """学習可能パラメータ数を表示する（診断用）。"""
        trainable, total = self._count_lora_params()
        print(
            f"Trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / max(total, 1):.2f}%)"
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

        # ====================================================================
        # DyGIE++ (Wadden et al., 2019) の実行順序を忠実に再現する:
        #
        #   [Phase 1] coref.compute_representations()
        #       mention scoring + Top-T pruning + antecedent scoring
        #   [Phase 2] coref_propagation (SpanPropagation, Section 3.3)
        #       antecedent scores を辺として span_repr をグラフ伝播で更新
        #   [Phase 3] NER   ← 伝播後の span_repr を使用
        #   [Phase 4] coref.predict_labels()
        #       クラスタ割り当て + coref 損失計算  ← NER の後、RE の前
        #   [Phase 5] RE    ← 伝播後の span_repr を使用
        #   [Phase 6] Events ← 伝播後の span_repr を使用
        # ====================================================================

        # ---- Phase 1: Coref compute_representations ----
        coref_repr: dict[str, Any] | None = None
        if self.coref_module is not None:
            coref_repr = self.coref_module.compute_representations(
                span_repr=span_repr,
                span_mask=span_mask,
                spans=spans,
                num_tokens=num_tokens,
            )
            # antecedent_scores は SpanProp と出力に使用（損失はまだ計算しない）
            output["mention_scores"]    = coref_repr["mention_scores"]
            output["top_span_indices"]  = coref_repr["top_span_indices"]
            output["top_span_mask"]     = coref_repr["top_span_mask"]
            output["antecedent_scores"] = coref_repr["antecedent_scores"]

        # ---- Phase 2: Span Graph Propagation (Section 3.3) ----
        # Coref antecedent scores を辺として GRU スタイルでスパン表現を更新する。
        # 同一エンティティの複数スパン間で情報を共有し、NER・RE の精度を向上させる。
        if self.span_prop is not None and coref_repr is not None:
            span_repr = self.span_prop(
                span_repr=span_repr,
                top_span_indices=coref_repr["top_span_indices"],
                top_span_mask=coref_repr["top_span_mask"],
                antecedent_scores=coref_repr["antecedent_scores"],
            )

        # ---- Phase 3: NER (伝播後の span_repr を使用) ----
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

        # ---- Phase 4: coref.predict_labels (NER の後・RE の前) ----
        # 原論文の順序に従い、クラスタ割り当てと coref 損失計算を NER の後に実行する。
        if self.coref_module is not None and coref_repr is not None:
            coref_repr = self.coref_module.predict_labels(
                coref_repr=coref_repr,
                spans=spans,
                coref_clusters=coref_clusters,
            )
            if "coref_loss" in coref_repr:
                output["coref_loss"] = coref_repr["coref_loss"]
                total_loss = total_loss + self.coref_loss_weight * coref_repr["coref_loss"]

        # ---- Phase 5: Relation Extraction (伝播後の span_repr を使用) ----
        if self.rel_module is not None:
            rel_out = self.rel_module(
                span_repr=span_repr,
                span_mask=span_mask,
                ner_preds=output["ner_preds"],
                rel_labels=rel_labels,
                ner_labels=ner_labels,
                spans=spans,
                use_gold_spans=use_gold_spans,
            )
            output.update(rel_out)
            if "rel_loss" in rel_out:
                total_loss = total_loss + self.rel_loss_weight * rel_out["rel_loss"]

        # ---- Phase 6: Event Extraction (伝播後の span_repr を使用) ----
        if self.event_module is not None:
            event_out = self.event_module(
                span_repr=span_repr,
                span_mask=span_mask,
                event_trigger_labels=event_trigger_labels,
                event_arg_labels=event_arg_labels,
                use_gold_triggers=use_gold_spans,
            )
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
          model.pt          — PyTorch state dict（LoRA 重みを含む）
          config.json       — Transformer encoder config
          dygie_config.json — DyGIE 固有のハイパーパラメータ（LoRA 設定を含む）

        LoRA 有効時:
          state dict には LoRA の A/B 行列と凍結 BERT 重みの両方が保存される。
          from_pretrained() 時に use_lora=True で自動的に LoRA 構造を再構築し、
          state dict を読み込む。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "model.pt")
        # encoder の config を保存（peft モデルは base_model.config にプロキシされる）
        self.encoder.config.save_pretrained(output_dir)
        # DyGIE 固有の設定（LoRA 設定を含む）を保存
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
