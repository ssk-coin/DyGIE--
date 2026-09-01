# DyGIE++ Standalone

AllenNLP **不要**の DyGIE++ 完全再実装です。  
PyTorch + HuggingFace Transformers のみで動作します。

## 対応タスク
- **NER** — Named Entity Recognition（スパン分類）
- **RE** — Relation Extraction（スパンペア分類）
- **Coref** — Coreference Resolution（スパンペアスコアリング + antecedentランキング）

## 必要環境
```
Python >= 3.9
PyTorch >= 2.0
transformers >= 4.30
scipy
```

## インストール
```bash
pip install torch transformers scipy
```

## ディレクトリ構成
```
dygie_standalone/
├── dygie/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py        # データローダ（SciERC/DyGIE++ JSON形式）
│   │   └── collate.py        # バッチ化ユーティリティ
│   ├── model/
│   │   ├── __init__.py
│   │   ├── dygie.py          # メインモデル
│   │   ├── span_extractor.py # スパン表現抽出
│   │   ├── ner_module.py     # NERヘッド
│   │   ├── rel_module.py     # Relationヘッド
│   │   └── coref_module.py   # Corefヘッド
│   └── training/
│       ├── __init__.py
│       ├── trainer.py        # 学習ループ
│       └── metrics.py        # F1スコア等
├── configs/
│   └── scierc.json           # SciERC設定例
├── scripts/
│   ├── train.py              # 学習スクリプト
│   ├── predict.py            # 推論スクリプト
│   └── evaluate.py           # 評価スクリプト
└── data/sample/
    └── sample.jsonl          # サンプルデータ
```

## データ形式（DyGIE++ JSON Lines）
```json
{
  "doc_key": "doc_id",
  "dataset": "scierc",
  "sentences": [["Token", "list", "..."], ["Next", "sentence"]],
  "ner": [[[0, 2, "Method"], [5, 5, "Task"]], []],
  "relations": [[[0, 2, 5, 5, "Used-for"]], []],
  "clusters": [[[0, 2], [10, 12]], [[5, 5], [20, 20]]]
}
```

## 学習
```bash
python scripts/train.py \
  --config configs/scierc.json \
  --train_path data/train.jsonl \
  --dev_path data/dev.jsonl \
  --output_dir output/scierc_bert
```

### 学習オプション（v2 追加）

| オプション | 説明 | デフォルト |
|---|---|---|
| `--use_amp` | 自動混合精度 (AMP) を有効化（CUDA 環境のみ） | false |
| `--patience` | Early stopping の待機エポック数（0=無効） | 0 |
| `--gradient_accumulation_steps` | 勾配蓄積ステップ数（実効バッチサイズを増やす） | 1 |

例: GPU メモリが少ない場合の設定
```bash
python scripts/train.py \
  --config configs/scierc.json \
  --train_path data/train.jsonl \
  --dev_path data/dev.jsonl \
  --output_dir output/scierc_bert \
  --use_amp \
  --patience 5 \
  --gradient_accumulation_steps 4
```

## 推論
```bash
python scripts/predict.py \
  --model_dir output/scierc_bert/checkpoint_best \
  --input_path data/test.jsonl \
  --output_path predictions.jsonl
```

## 評価
```bash
python scripts/evaluate.py \
  --gold_path data/test.jsonl \
  --pred_path predictions.jsonl
```

## モデルの保存・復元（v2）

`DyGIE.save_pretrained()` は `dygie_config.json`（NER/REラベル、ハイパーパラメータ）を保存します。
`DyGIE.from_pretrained()` はこれを自動読み込みするため、引数なしでロード可能です。

```python
from dygie.model import DyGIE

# 保存
model.save_pretrained("output/scierc_bert/checkpoint_best")

# 復元（引数不要）
model = DyGIE.from_pretrained("output/scierc_bert/checkpoint_best")
```

## バグ修正と改善点 (v2)

### バグ修正
1. **Coref metrics recall の計算誤り** (`metrics.py`)  
   `_muc` / `_b3` / `_ceaf_phi4` の集計で、recall 分子に precision スコアを使っていたバグを修正。
   precision と recall が独立して正しく計算されるように変更。

2. **Coref antecedent 距離埋め込みの方向が逆** (`coref_module.py`)  
   `dist[i,j] = j - i`（clamped to 0）のため、有効な antecedent ペア（j < i）で
   常に距離 0 になり距離埋め込みが機能しないバグを修正。
   正しくは `dist[i,j] = max(0, i - j)`。

3. **`max_top_antecedents` が未適用** (`coref_module.py`)  
   `__init__` に受け取るが `_score_antecedents` で使われていなかった。
   位置距離ウィンドウとして正しく適用。

4. **モデル保存時に DyGIE 設定が失われる** (`dygie.py`)  
   `save_pretrained` が `dygie_config.json` を出力し、
   `from_pretrained` が自動で復元するよう対応。

5. **デッドコード `tok_offset`** (`dataset.py`)  
   使われていない変数を削除。

6. **`torch.load` のセキュリティ警告** (`dygie.py`, `predict.py`)  
   `weights_only=True` を追加。

### 機能追加
7. **AMP（自動混合精度）学習** (`trainer.py`)  
   `use_amp=True` で CUDA 学習を高速化・メモリ削減。

8. **Early stopping** (`trainer.py`)  
   `patience > 0` で dev スコアが改善しないエポックが続く場合に自動停止。

9. **Gradient accumulation** (`trainer.py`)  
   `gradient_accumulation_steps > 1` で実効バッチサイズを増加。
