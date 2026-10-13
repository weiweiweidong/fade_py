"""
CE (Contrast Energy) computation for FADE.

Applies a Laplacian-of-Gaussian (LoG) filter to the gray, by, rg channels
and returns the rectified/normalized contrast energy maps.
"""

import numpy as np
from scipy import ndimage


def _build_log_filter(sigma: float = 3.25, break_off_sigma: float = 3.0):
    """Build 1-D LoG kernel (same as MATLAB CE reference)."""
    filtersize = break_off_sigma * sigma
    x = np.arange(-filtersize, filtersize, 1.0).reshape(1, -1)
    gauss = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(x**2 / (-2.0 * sigma**2))
    gauss /= gauss.sum()
    gx = (x**2 / sigma**4 - 1.0 / sigma**2) * gauss
    gx -= gx.sum() / x.size
    gx /= np.sum(0.5 * x * x * gx)
    return gx  # shape (1, kernel_len)


def _pad_border(img: np.ndarray, border: int) -> np.ndarray:
    """Replicate-pad `img` by `border` pixels on each side (matches MATLAB border_in)."""
    uc = border // 2
    dc = border // 2 - 1 if border % 2 == 0 else border // 2

    top = img[:uc, :]
    bottom = img[-dc - 1 :, :]
    mid = np.vstack([top, img, bottom])

    left = mid[:, :uc]
    right = mid[:, -dc - 1 :]
    return np.hstack([left, mid, right])


def _unpad_border(img: np.ndarray, border: int) -> np.ndarray:
    """Remove the border added by `_pad_border` (matches MATLAB border_out)."""
    uc = border // 2
    dc = border // 2 - 1 if border % 2 == 0 else border // 2

    img = img[:, uc:]
    if dc > 0:
        img = img[:, : -dc - 1]
    img = img[uc:, :]
    if dc > 0:
        img = img[: -dc - 1, :]
    return img


def _contrast_energy(channel: np.ndarray, gx: np.ndarray, threshold: float, border: int = 20) -> np.ndarray:
    """Compute contrast energy for a single channel."""
    padded = _pad_border(channel, border)
    cx = ndimage.convolve(padded, gx, mode="constant")
    cy = ndimage.convolve(padded, gx.T, mode="constant")
    c_map = np.sqrt(cx**2 + cy**2)
    c_map = _unpad_border(c_map, border)

    c_max = c_map.max()
    r_map = (c_map * c_max) / (c_map + c_max * 0.1)
    result = np.zeros_like(r_map)
    mask = (r_map - threshold) > 1e-7
    result[mask] = r_map[mask] - threshold
    return result


def compute_ce(I: np.ndarray):
    """
    Compute contrast energy maps for gray, by, rg channels.

    Parameters
    ----------
    I : np.ndarray
        RGB image, shape (H, W, 3), uint8 or float.

    Returns
    -------
    CE_gray, CE_by, CE_rg : np.ndarray
        Contrast energy maps, each shape (H, W).
    """
    img = I.astype(np.float64)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    gray = 0.299 * R + 0.587 * G + 0.114 * B
    by = 0.5 * R + 0.5 * G - B
    rg = R - G

    return compute_ce_from_channels(gray, by, rg)


def compute_ce_from_channels(
    gray: np.ndarray, by: np.ndarray, rg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute CE while reusing prepared gray and color-opponent channels."""
    gx = _build_log_filter()

    t1 = 9.225496406318721e-4 * 255
    t2 = 8.969246659629488e-4 * 255
    t3 = 2.069284034165411e-4 * 255

    CE_gray = _contrast_energy(gray, gx, t1)
    CE_by = _contrast_energy(by, gx, t2)
    CE_rg = _contrast_energy(rg, gx, t3)

    return CE_gray, CE_by, CE_rg
