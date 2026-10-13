# FADE — Fog Aware Density Evaluator

[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

[English](../README.md) | [简体中文](README_CN.md) | [日本語](README_JA.md)

FADE 通过 12 个感知特征和预计算参考统计量，从单张 RGB 图像估计雾浓度。本项目基于 NumPy/SciPy 实现，可返回整体评分及局部雾密度图。**评分越高，表示雾浓度越大。**

## 安装

**需要 Python 3.10 或更高版本。** 克隆仓库后，在项目根目录执行以下命令：

```bash
git clone https://github.com/weiweiweidong/fade_py.git
cd fade_py
```

### uv（推荐）

安装 [uv](https://github.com/astral-sh/uv) 后执行：

```bash
uv sync --extra demo
```

### pip + venv

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell 使用：.venv\Scripts\Activate.ps1
pip install -e ".[demo]"
```

算法本身仅依赖 NumPy 和 SciPy（使用 `uv sync` 或 `pip install -e .` 安装）。`demo` 可选依赖包含用于读取图像的 Pillow、用于绘图的 Matplotlib，以及 Notebook 所需的 IPykernel；运行下方示例时请安装它。

## 快速开始

在项目根目录使用项目环境运行这段代码。使用 uv 时，通过 `uv run --extra demo python` 启动 Python；使用 pip 时，请先激活虚拟环境。

```python
import numpy as np
from PIL import Image
from fade import fade, fade_with_map

with Image.open("test_image/test_image1.png") as image:
    img = np.array(image.convert("RGB"))

# 计算整体雾浓度评分
score = fade(img)
print(f"FADE score: {score:.4f}")

# 计算评分和逐图像块的雾密度图
score, density_map = fade_with_map(img)
print(f"Score: {score:.4f} | Map shape: {density_map.shape}")
```

预期输出：

```text
FADE score: 0.9113
Score: 0.9113 | Map shape: (96, 72)
```

## 示例结果

仓库中三张示例图像的评分如下：

| 清晰场景 | 中等雾 | 浓雾 |
| --- | --- | --- |
| <img src="../test_image/test_image1.png" width="240" alt="清晰的山地景观"> | <img src="../test_image/test_image2.JPG" width="240" alt="中等雾中的城市街道"> | <img src="../test_image/test_image3.jpg" width="240" alt="浓雾中的湖面"> |
| `test_image1.png` | `test_image2.JPG` | `test_image3.jpg` |
| **FADE: 0.9113** | **FADE: 2.7122** | **FADE: 5.7674** |

## 输入与输出

- 输入：RGB `np.ndarray`，形状为 `(H, W, 3)`，数据类型为 `uint8`（取值 0–255）。
- `fade(img)` 返回一个 `float`；`fade_with_map(img)` 返回 `(score, density_map)`。
- 图像按不重叠的 8×8 图像块计算。底部和右侧不足一个图像块的部分会被裁剪；密度图形状为 `(H // 8, W // 8)`。
- 参考统计量随包提供，由程序自动加载。
- 纯色图像或变化不足的图像块可能产生 `NaN` 值。

## 演示与测试

在支持 Notebook 的编辑器中打开 [test_fade.ipynb](../test_fade.ipynb)，选择项目 `.venv` 中的 Python 解释器作为内核。Notebook 展示示例评分、推理耗时和雾密度图，依赖上方安装的 `demo` 工具。

使用 uv 运行测试：

```bash
uv run --extra test pytest
uv run --extra test pytest -m "not slow"  # 跳过加载示例图像的测试
```

使用 pip 时，在已激活的虚拟环境中安装测试依赖：

```bash
pip install -e ".[test]"
pytest
```

## 参考与许可证

算法细节、架构和性能测量见[中文技术文档](technical_documentation.md)。

> Choi, L. K., You, J., & Bovik, A. C. (2015).
> *Referenceless prediction of perceptual fog density and perceptual image quality.*
> IEEE Transactions on Image Processing, 24(11), 3888–3901.

[论文 PDF](../reference/FADE.pdf)

本项目采用 [MIT 许可证](../LICENSE)。
