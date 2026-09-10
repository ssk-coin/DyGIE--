"""
DyGIE-- Standalone — バージョン情報

学習・評価スクリプトの起動時に表示し、ログにどの時点の
コードによる実験結果かを明記するためのモジュール。

バージョン番号は git コミット番号（v{N}）に対応する。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# -----------------------------------------------------------------------
# バージョン番号
# -----------------------------------------------------------------------

__version__ = "1.16.0"
"""
バージョン番号: major.minor.patch
  major = 大きな機能追加 / アーキテクチャ変更
  minor = 論文記載の機能番号（v1〜v16 に相当）
  patch = バグ修正・細かな改良
"""

# -----------------------------------------------------------------------
# 変更履歴
# -----------------------------------------------------------------------

CHANGELOG: list[tuple[str, str]] = [
    ("1.0.0",  "初期リリース: NER / RE / Coref の基本実装"),
    ("1.1.0",  "v2: CorefMetrics の recall 計算バグ修正"),
    ("1.2.0",  "v3: RE のメモリ最適化 O(K²)→O(E²)・スパンプルーニング改善"),
    ("1.3.0",  "v4: RE スコア改善 — エンティティタイプ埋め込み・距離特徴・Focal Loss・損失集約修正"),
    ("1.4.0",  "v5: イベント抽出 (Event Extraction) 追加"),
    ("1.5.0",  "v6: LoRA アダプタ (PEFT) 対応"),
    ("1.6.0",  "v7: 乱数シード固定オプション追加"),
    ("1.7.0",  "v8: early stopping メトリクス設定ファイルで選択可能に"),
    ("1.8.0",  "v9: trainer の RE/Event 評価を end-to-end 評価に統一 (evaluate.py との整合)"),
    ("1.9.0",  "v10: coref_prop パラメータによるスパングラフ伝播回数の制御"),
    ("1.10.0", "v11: coref_prop を use_coref と独立させ DyGIE++ 本家の動作に合わせる"),
    ("1.11.0", "v12: max_span_width で除外された gold アノテーションを FN として正しくカウント"),
    ("1.12.0", "v13: freeze_encoder — Transformer エンコーダ凍結による学習高速化"),
    ("1.13.0", "v14: num_workers≥1 での Too many open files 修正 (file_system 共有戦略 + persistent_workers)"),
    ("1.14.0", "v15: RE ペア表現アーキテクチャ修正 — span_proj(512-dim)でボトルネック解消"),
    ("1.15.0", "v15b: RE pair MLP 計算量削減 — pair_input_dim 5184→1344 (速度復帰)"),
    ("1.16.0", "v16: RE F1 改善 — 要素積特徴量(DyGIE++論文準拠)・gold NER 診断評価追加"),
]


def get_git_info() -> dict[str, str]:
    """git のコミット情報を取得する。git が使えない場合は空辞書を返す。"""
    info: dict[str, str] = {}
    repo_root = Path(__file__).parent.parent
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root, stderr=subprocess.DEVNULL,
        ).decode().strip()
        # 未コミットの変更があるか確認
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root, stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["dirty"] = "yes" if dirty else "no"
    except Exception:
        pass
    return info


def format_version_header() -> str:
    """ログ冒頭に出力するバージョンヘッダー文字列を生成する。"""
    lines = [
        "=" * 60,
        f"  DyGIE-- Standalone  v{__version__}",
    ]

    # git 情報
    git = get_git_info()
    if git:
        dirty_mark = " (uncommitted changes)" if git.get("dirty") == "yes" else ""
        lines.append(f"  git: {git.get('branch', '?')}@{git.get('commit', '?')}{dirty_mark}")

    # 最新の変更履歴を 3 件表示
    lines.append("")
    lines.append("  最新の変更:")
    for ver, desc in CHANGELOG[-3:]:
        marker = ">>>" if ver == __version__ else "   "
        lines.append(f"  {marker} v{ver}: {desc}")

    lines.append("=" * 60)
    return "\n".join(lines)


def print_version_header() -> None:
    """バージョンヘッダーを標準出力と logging に出力する。"""
    import logging
    header = format_version_header()
    print(header)
    logging.getLogger(__name__).info("Version: %s", __version__)
