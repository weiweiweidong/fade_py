# FADE — Fog Aware Density Evaluator

[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

[English](../README.md) | [简体中文](README_CN.md) | [日本語](README_JA.md)

FADE は、12 個の知覚的特徴量と事前計算された参照統計量を用いて、1 枚の RGB 画像から霧の濃度を推定します。本プロジェクトは NumPy/SciPy による実装で、画像全体のスコアと局所的な霧の濃度マップを取得できます。**スコアが高いほど、霧が濃いことを示します。**

## インストール

**Python 3.10 以降が必要です。** リポジトリをクローンし、プロジェクトのルートディレクトリで以下のコマンドを実行してください。

```bash
git clone https://github.com/weiweiweidong/fade_py.git
cd fade_py
```

### uv（推奨）

[uv](https://github.com/astral-sh/uv) のインストール後に実行します。

```bash
uv sync --extra demo
```

### pip + venv

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell の場合：.venv\Scripts\Activate.ps1
pip install -e ".[demo]"
```

アルゴリズム自体に必要なのは NumPy と SciPy のみです（`uv sync` または `pip install -e .` でインストールできます）。オプションの `demo` には、画像読み込み用の Pillow、描画用の Matplotlib、Notebook 用の IPykernel が含まれます。以下のサンプルを実行する場合は `demo` をインストールしてください。

## クイックスタート

プロジェクトのルートディレクトリで、プロジェクトの Python 環境を使って実行してください。uv を使う場合は `uv run --extra demo python` で Python を起動します。pip を使う場合は仮想環境を有効にしてください。

```python
import numpy as np
from PIL import Image
from fade import fade, fade_with_map

with Image.open("test_image/test_image1.png") as image:
    img = np.array(image.convert("RGB"))

# 画像全体の霧濃度スコアを計算
score = fade(img)
print(f"FADE score: {score:.4f}")

# スコアとパッチごとの霧濃度マップを計算
score, density_map = fade_with_map(img)
print(f"Score: {score:.4f} | Map shape: {density_map.shape}")
```

実行結果：

```text
FADE score: 0.9113
Score: 0.9113 | Map shape: (96, 72)
```

## サンプル結果

リポジトリに含まれる 3 枚のサンプル画像のスコアです。

| 霧の少ない風景 | 中程度の霧 | 濃い霧 |
| --- | --- | --- |
| <img src="../test_image/test_image1.png" width="240" alt="霧の少ない山の風景"> | <img src="../test_image/test_image2.JPG" width="240" alt="中程度の霧に包まれた街路"> | <img src="../test_image/test_image3.jpg" width="240" alt="濃い霧に包まれた湖"> |
| `test_image1.png` | `test_image2.JPG` | `test_image3.jpg` |
| **FADE: 0.9113** | **FADE: 2.7122** | **FADE: 5.7674** |

## 入力と出力

- 入力：形状 `(H, W, 3)`、データ型 `uint8`（値の範囲は 0–255）の RGB `np.ndarray`。
- `fade(img)` は `float` を返し、`fade_with_map(img)` は `(score, density_map)` を返します。
- 画像は重なりのない 8×8 ピクセルのパッチに分割されます。下端と右端の余りは切り捨てられ、マップの形状は `(H // 8, W // 8)` になります。
- 参照統計量はパッケージに含まれており、自動的に読み込まれます。
- 単色の画像や変化の乏しいパッチでは、`NaN` が生じることがあります。

## デモとテスト

Notebook 対応エディターで [test_fade.ipynb](../test_fade.ipynb) を開き、プロジェクトの `.venv` 内の Python インタープリターをカーネルとして選択してください。サンプルのスコア、推論時間、霧濃度マップを確認できます。上記の `demo` 依存パッケージが必要です。

uv でテストを実行する場合：

```bash
uv run --extra test pytest
uv run --extra test pytest -m "not slow"  # サンプル画像を読み込むテストを除外
```

pip を使う場合は、有効にした仮想環境でテスト用パッケージをインストールします。

```bash
pip install -e ".[test]"
pytest
```

## 参考文献とライセンス

アルゴリズム、構成、性能測定の詳細は、[技術ドキュメント（中国語）](technical_documentation.md) を参照してください。

> Choi, L. K., You, J., & Bovik, A. C. (2015).
> *Referenceless prediction of perceptual fog density and perceptual image quality.*
> IEEE Transactions on Image Processing, 24(11), 3888–3901.

[論文 PDF](../reference/FADE.pdf)

本プロジェクトは [MIT ライセンス](../LICENSE) で公開されています。
