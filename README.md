# FADE — Fog Aware Density Evaluator

[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) | [简体中文](doc/README_CN.md) | [日本語](doc/README_JA.md)

FADE estimates fog density from a single RGB image using 12 perceptual features and precomputed reference statistics. This NumPy/SciPy implementation returns a scalar score and an optional map of local fog density. **Higher scores indicate denser fog.**

## Installation

**Requires Python 3.10 or newer.** Clone the repository and run the commands below from its root directory:

```bash
git clone https://github.com/weiweiweidong/fade_py.git
cd fade_py
```

### uv (recommended)

With [uv](https://github.com/astral-sh/uv) installed:

```bash
uv sync --extra demo
```

### pip + venv

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[demo]"
```

The algorithm itself requires only NumPy and SciPy (`uv sync` or `pip install -e .`). The `demo` extra adds Pillow for image loading, Matplotlib for plotting, and IPykernel for the notebook; install it to run the examples below.

## Quick Start

Run this code from the repository root using the project environment. With uv, start Python with `uv run --extra demo python`; with pip, use the activated virtual environment.

```python
import numpy as np
from PIL import Image
from fade import fade, fade_with_map

with Image.open("test_image/test_image1.png") as image:
    img = np.array(image.convert("RGB"))

# Compute the scalar fog-density score
score = fade(img)
print(f"FADE score: {score:.4f}")

# Compute the score and per-patch density map
score, density_map = fade_with_map(img)
print(f"Score: {score:.4f} | Map shape: {density_map.shape}")
```

Expected output:

```text
FADE score: 0.9113
Score: 0.9113 | Map shape: (96, 72)
```

## Sample Results

Scores for the three images included in the repository:

| Clear scene | Moderate fog | Dense fog |
| --- | --- | --- |
| <img src="test_image/test_image1.png" width="240" alt="Clear mountain landscape"> | <img src="test_image/test_image2.JPG" width="240" alt="City street in moderate fog"> | <img src="test_image/test_image3.jpg" width="240" alt="Lake in dense fog"> |
| `test_image1.png` | `test_image2.JPG` | `test_image3.jpg` |
| **FADE: 0.9113** | **FADE: 2.7122** | **FADE: 5.7674** |

## Input and Output

- Input: an RGB `np.ndarray` with shape `(H, W, 3)` and dtype `uint8` (values 0–255).
- `fade(img)` returns a `float`; `fade_with_map(img)` returns `(score, density_map)`.
- The image is divided into non-overlapping 8×8 patches. Incomplete patches at the bottom and right edges are cropped; the map has shape `(H // 8, W // 8)`.
- Reference statistics are included in the package and loaded automatically.
- Uniform images or patches with insufficient variation may produce `NaN` values.

## Demo and Tests

Open [test_fade.ipynb](test_fade.ipynb) in a notebook editor and select the project’s `.venv` Python interpreter as the kernel. The notebook displays sample scores, inference times, and density maps. It uses the `demo` dependencies installed above.

Run the test suite with uv:

```bash
uv run --extra test pytest
uv run --extra test pytest -m "not slow"  # Skip tests that load sample images
```

With pip, install the test dependencies in the activated environment:

```bash
pip install -e ".[test]"
pytest
```

## Reference and License

For algorithm details, architecture, and performance measurements, see the [technical documentation in Chinese](doc/technical_documentation.md).

> Choi, L. K., You, J., & Bovik, A. C. (2015).
> *Referenceless prediction of perceptual fog density and perceptual image quality.*
> IEEE Transactions on Image Processing, 24(11), 3888–3901.

[Paper PDF](reference/FADE.pdf)

Released under the [MIT License](LICENSE).
