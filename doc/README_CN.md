# FADE — 雾感知密度评估器

[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

[English README](../README.md)

本项目是 **FADE** 雾浓度估计算法的高性能 Python 实现，基于以下论文：

> Choi, L. K., You, J., & Bovik, A. C. (2015).
> *Referenceless prediction of perceptual fog density and perceptual image quality.*
> IEEE Transactions on Image Processing, 24(11), 3888–3901.

FADE 仅凭**单张 RGB 图像**，通过 12 个手工感知特征与预计算参考统计量，无需参考图像即可估计雾浓度。

---

## 特性

- **无参考雾浓度估计** — 仅需单张图像
- **向量化流水线** — 基于 NumPy/SciPy 的快速实现
- **完整 12 特征 FADE 流水线** — 忠实复现原始算法
- **模块化架构** — `features`、`ops`、`models`、`pipeline` 子包
- **简洁公共 API** — `fade(image)` 与 `fade_with_map(image)`
- **完整测试套件** — 覆盖所有模块的 pytest 测试套件

---

## 安装

**环境要求：** Python >= 3.10

本项目使用 [uv](https://github.com/astral-sh/uv) 进行包管理，推荐使用 `uv`，同时也支持标准 `pip` 工作流。

### 方式一：uv（推荐）

```bash
# 若尚未安装 uv，先执行安装
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装算法依赖
uv sync

# 安装图像读取、绘图和 Notebook 依赖
uv sync --extra demo

# 同时安装演示与测试依赖
uv sync --extra demo --extra test

# 在托管环境中运行命令
uv run python -c "from fade import fade; print('OK')"
```

### 方式二：pip + venv

```bash
python -m venv .venv
source .venv/bin/activate      # Windows：.venv\Scripts\activate

pip install -e ".[demo]"
```

---

## 快速开始

```python
from PIL import Image
import numpy as np
from fade import fade, fade_with_map

img = np.array(Image.open("test_image/test_image1.png").convert("RGB"))

# 获取雾浓度标量分数
score = fade(img)
print(f"FADE 分数：{score:.4f}")

# 获取分数 + 逐 patch 雾密度图
score, density_map = fade_with_map(img)
print(f"分数：{score:.4f} | 密度图形状：{density_map.shape}")
```

**FADE 分数越高，表示雾浓度越大。**

---

## 项目结构

```text
fade_py/
├── src/fade/
│   ├── api.py            # 公共 API：fade() 与 fade_with_map()
│   ├── pipeline.py       # 端到端流水线编排（FadePipeline）
│   ├── features/         # f01 ~ f12 — 各特征模块
│   │   ├── base.py       # 特征抽象基类
│   │   ├── registry.py   # 特征注册表
│   │   └── f01.py ~ f12.py
│   ├── ops/              # 底层算子（MSCN、熵、patch 划分等）
│   ├── models/           # 上下文模型 + 参考统计量加载器
│   └── data/             # 预计算参考 .mat 文件
├── tests/                # pytest 测试套件
├── reference/            # 原始论文（FADE.pdf）
├── test_image/           # 示例图像
├── test_fade.ipynb       # 交互式演示 Notebook
└── pyproject.toml
```

---

## 运行测试

使用 `uv`：

```bash
uv run --extra test pytest
uv run --extra test pytest -m "not slow"   # 跳过需要加载图像的慢速测试
```

使用标准 `pytest`：

```bash
pip install -e ".[test]"
pytest
pytest -m "not slow"
```

---

## 演示 Notebook

打开 `test_fade.ipynb`，可交互式地对示例图像计算 FADE 分数。
请先安装演示依赖：`uv sync --extra demo` 或 `pip install -e ".[demo]"`。

输出表格各字段说明：

| 字段 | 说明 |
|---|---|
| Image name | 输入图像文件名 |
| Shape | 高 × 宽 × 通道数 |
| FADE score | 估计雾浓度（越高越浓） |
| Elapsed time | 推理耗时 |

---

## 使用说明

- 输入须为 RGB `np.ndarray`，形状为 `(H, W, 3)`，数据类型为 `uint8`。
- 流水线在不重叠的 `8×8` patch 上进行运算。
- 参考统计量从 `src/fade/data/` 目录下的 `.mat` 文件加载。
- 算法运行仅依赖 NumPy 和 SciPy；图像读取、绘图与 Notebook 工具属于可选的 `demo` 依赖。
- 图像块熵与协方差求解按最多 4096 个图像块分批处理，控制临时内存占用。

---

## 参考资源

- `reference/FADE.pdf` — 原始论文

---

## 许可证

[MIT](../LICENSE)
