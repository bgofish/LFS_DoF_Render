# Depth of Field Plugin — LichtFeld Studio

Adds a **depth-of-field (bokeh blur)** post-process to the
LichtFeld Studio viewer and a one-click render export, driven by the
scene's own depth buffer.

---

## Features

| Feature | Description |
|---|---|
| Pick focus point | Click anywhere on the splat to auto-set focus distance |
| f-Stop / focal length | Standard camera lens controls |
| CoC debug overlay | Visualise the per-pixel Circle-of-Confusion radius |
| Render with DoF | Saves a full-resolution DoF-composited PNG  |
| Render with DoF | Saves a Stills (PNG/Video (MP4) of  DoF-composited of Camera Path  |
---

## Installation

**In-app (LichtFeld Studio ≥ 0.4.2):**

1. Open the **Plugins** panel.
2. Paste: `https://github.com/bgofish/LFS-DoF-Render`
3. Click **Install**.

---

## Usage

1. Load a 3DGS scene.
2. Switch the renderer to **RGB_D** mode (required for the depth buffer).
3. Open the **DoF Render** panel in the sidebar.
4. Click **Enable**.
5. Use **Pick Focus Point** to click your subject on the splat — the focus
   distance updates instantly.
6. Adjust **Focal Length** and **F-Stop** to taste.
7. Click **Render with DoF** to export a composited PNG & Debug information: Default output = C:\temp
   
	DOF_RENDER.TXT       =  Information additional runs appended, so can be deleted

	debug_compare.png    =  Side by Side full depth scene & blurred

	render_dof.png       =  Blurred result

	debug_coc.png        =  Circle-of-Confusion image in greyscale

	debug_depth.png      =  Calculated depth image in greyscale

	debug_raw.png        =  Normal RGB image of scene

---

## Controls

| Control | Default | Notes |
|---|---|---|
| Focal Length (mm) | 50 | Longer = narrower DoF |
| F-Stop | 2.8 | Lower = more blur |
| Focus Distance | 5.0 | Scene units; use Pick button for ease |
| Sensor Height (mm) | 24 | Full-frame 35mm default |
| *Max CoC (px) | 24 | Clamps the maximum blur radius |
| *Blur Bands | 6 | More bands = smoother transition, slower |

* currently no controls
---

## How it works

The plugin hooks into the renderer's `add_post_process_hook` callback and
runs a **variable-radius Gaussian blur** on every frame:

```
CoC_px = (aperture_mm × focal_mm × |depth - focus_dist|)
         ─────────────────────────────────────────────────
         (depth × |focus_dist - focal_mm / 1000|)
         × (image_height_px / sensor_height_mm)
```

The blur is implemented as a band-based separable Gaussian (using
`scipy.ndimage` when available, or a NumPy integral-image box-blur
fallback).  The same mathematics apply equally to CPU and GPU paths.

---

## Requirements

- LichtFeld Studio ≥ 0.4.2
- Python ≥ 3.10 (bundled with LichtFeld Studio)
- `numpy` (auto-installed via plugin deps)
- `scipy` *(optional — installed automatically; improves blur quality)*

---

## License

GPL-3.0-or-later — same as LichtFeld Studio.
