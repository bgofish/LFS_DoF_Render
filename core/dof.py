# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Core depth-of-field — fully vectorised, scene-unit agnostic."""

from __future__ import annotations
import numpy as np


def compute_coc(
    depth: np.ndarray,
    focal_length_mm: float,
    f_stop: float,
    focus_distance: float,
    sensor_height_mm: float = 24.0,
    image_height_px: int = 1080,
    max_coc_px: float = 32.0,
) -> np.ndarray:
    """
    Per-pixel CoC in pixels [0, max_coc_px].

    DoF zone is symmetric around focus_distance.
    f_stop controls the width: high f_stop = wide in-focus band.
    focal_length controls steepness: long lens = narrow band.
    """
    depth = np.maximum(depth, 1e-6).astype(np.float32)
    fd = max(abs(focus_distance), 1e-6)

    # Use focus distance as the reference point for normalisation.
    # scene_range spans from focus outward in both directions, so the
    # normalisation is always centred on the user's chosen focus plane
    # regardless of whether geometry exists there.
    d_min = float(depth.min())
    d_max = float(depth.max())
    # Range = furthest distance from focus plane to scene edge
    scene_range = max(max(abs(d_max - fd), abs(fd - d_min)), 1e-6)

    # Normalised signed defocus: 0 = in focus, ±1 = scene edge
    defocus = (depth - fd) / scene_range   # signed, centred on focus plane

    # DoF half-width in normalised units.
    # f_stop=1.8 → narrow band (0.01), f_stop=32 → wide band (0.18)
    # focal_length=85mm → narrower than 35mm
    dof_half = min((f_stop / max(focal_length_mm, 1.0)) * 1.0, 0.45)

    # CoC rises from 0 at focus to 1 at max defocus, with dof_half dead zone
    beyond = np.maximum(np.abs(defocus) - dof_half, 0.0)
    coc_normalised = beyond / max(1.0 - dof_half, 0.01)

    coc_px = np.clip(coc_normalised * max_coc_px, 0.0, max_coc_px).astype(np.float32)

    # Smooth CoC map to eliminate per-splat chunkiness
    try:
        from scipy.ndimage import gaussian_filter
        coc_px = gaussian_filter(coc_px, sigma=1.5).astype(np.float32)
        coc_px = np.clip(coc_px, 0.0, max_coc_px)
    except Exception:
        pass

    return coc_px


def _gauss_blur(img: np.ndarray, radius: float) -> np.ndarray:
    if radius < 0.3:
        return img.copy()
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(img, sigma=[radius, radius, 0], mode='reflect').astype(np.float32)
    except ImportError:
        r = max(1, int(round(radius)))
        pad = np.pad(img, ((r, r), (r, r), (0, 0)), mode='reflect')
        k = np.ones(2 * r + 1) / (2 * r + 1)
        tmp = np.apply_along_axis(lambda x: np.convolve(x, k, 'valid'), 1, pad)
        return np.apply_along_axis(lambda x: np.convolve(x, k, 'valid'), 0, tmp).astype(np.float32)


def _apply_separable_blur(img, coc, max_coc, num_bands):
    H, W, C = img.shape
    radii = np.linspace(0.0, max_coc, num_bands + 1)
    layers = [_gauss_blur(img, float(r)) for r in radii]
    stack = np.stack(layers, axis=0).astype(np.float32)
    t = np.clip(coc / max(max_coc, 1e-6), 0.0, 1.0) * num_bands
    lo = np.floor(t).astype(np.int32)
    hi = np.minimum(lo + 1, num_bands)
    alpha = (t - lo).astype(np.float32)[:, :, None]
    iy = np.arange(H)[:, None]
    ix = np.arange(W)[None, :]
    return ((1.0 - alpha) * stack[lo, iy, ix, :] + alpha * stack[hi, iy, ix, :]).astype(np.float32)


def apply_dof(
    color: np.ndarray,
    depth: np.ndarray,
    focal_length_mm: float,
    f_stop: float,
    focus_distance: float,
    sensor_height_mm: float = 24.0,
    max_coc_px: float = 24.0,
    num_bands: int = 6,
) -> np.ndarray:
    # Clamp focus to actual depth range so it's never outside the scene
    d_min = float(depth.min())
    d_max = float(depth.max())
    focus_distance = float(np.clip(focus_distance, d_min, d_max))

    coc = compute_coc(
        depth, focal_length_mm=focal_length_mm, f_stop=f_stop,
        focus_distance=focus_distance, sensor_height_mm=sensor_height_mm,
        image_height_px=color.shape[0], max_coc_px=max_coc_px,
    )
    result = _apply_separable_blur(color, coc, max_coc_px, num_bands)
    return np.clip(result, 0.0, 1.0).astype(np.float32)
