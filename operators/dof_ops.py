# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operators for the Depth of Field plugin."""

import datetime
import numpy as np
import lichtfeld as lf
import lichtfeld.selection as sel
import lfs_plugins

Operator = lfs_plugins.Operator
Event = lf.ui.Event

from ..panels.dof_panel import state


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

DOF_LOG_PATH = r"c:\temp\DOF_RENDER.TXT"
DOF_VERSION = "v12-viewport-api"

def _dof_log(msg: str):
    try:
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(DOF_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass

_dof_log(f"=== dof_ops.py loaded: {DOF_VERSION} ===")


# ---------------------------------------------------------------------------
# Depth map from splat positions
# Computes a per-pixel depth image from scene splat world positions
# projected through the current camera.
# ---------------------------------------------------------------------------

def _compute_depth_image(color_shape, cam_pos_np, view):
    """
    Build a float32 depth image (H, W) using lf.get_viewport_render().screen_positions
    — the same API depthmap uses for its picker.
    """
    _dof_log(f"depth: _compute_depth_image {DOF_VERSION}")
    H, W = color_shape[:2]
    depth_img = np.full((H, W), state.focus_distance, dtype=np.float32)

    try:
        # --- Get splat world positions and camera distances ---
        scene = lf.get_scene()
        if scene is None:
            _dof_log("depth: scene is None")
            return depth_img
        combined = scene.combined_model()
        if combined is None:
            _dof_log("depth: combined_model() is None")
            return depth_img
        means = combined.get_means().numpy()  # (N, 3)
        if means.shape[0] == 0:
            _dof_log("depth: 0 splats")
            return depth_img

        diffs = means - cam_pos_np[None, :]
        dists = np.linalg.norm(diffs, axis=1).astype(np.float32)
        _dof_log(f"depth: {means.shape[0]} splats  dist [{dists.min():.2f}, {dists.max():.2f}]")

        # --- Get pre-computed screen positions from the engine ---
        _dof_log(f"depth: calling get_viewport_render()  hasattr={hasattr(lf, 'get_viewport_render')}")
        try:
            vp_render = lf.get_viewport_render()
            _dof_log(f"depth: vp_render={vp_render}  type={type(vp_render)}")
        except Exception as e:
            _dof_log(f"depth: get_viewport_render() raised: {e}")
            vp_render = None

        if vp_render is None:
            _dof_log("depth: get_viewport_render() returned None — trying dir(lf)...")
            lf_attrs = [a for a in dir(lf) if 'render' in a.lower() or 'viewport' in a.lower() or 'screen' in a.lower()]
            _dof_log(f"depth: lf render/viewport attrs: {lf_attrs}")
            return depth_img

        _dof_log(f"depth: vp_render attrs: {[a for a in dir(vp_render) if not a.startswith('_')]}")
        screen_pos_tensor = vp_render.screen_positions
        if screen_pos_tensor is None:
            _dof_log("depth: screen_positions is None")
            return depth_img

        screen_pos = screen_pos_tensor.numpy()  # (N, 2)  x=col, y=row
        _dof_log(f"depth: screen_pos shape={screen_pos.shape}  means shape={means.shape}")

        if screen_pos.shape[0] != means.shape[0]:
            _dof_log(f"depth: shape mismatch {screen_pos.shape[0]} vs {means.shape[0]}")
            return depth_img

        # --- Scatter min-distance into depth image ---
        px = np.round(screen_pos[:, 0]).astype(int)
        py = np.round(screen_pos[:, 1]).astype(int)
        # Filter out sentinel off-screen values (LichtFeld uses -10000)
        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        _dof_log(f"depth: {valid.sum()} splats on screen (W={W} H={H})")
        # Diagnose near splats specifically
        near_mask = dists < 120.0
        _dof_log(f"depth: {near_mask.sum()} splats closer than 120m")
        if near_mask.sum() > 0:
            near_px = px[near_mask]
            near_py = py[near_mask]
            near_on = valid[near_mask]
            _dof_log(f"depth: near splat px range [{near_px.min()},{near_px.max()}] py [{near_py.min()},{near_py.max()}]  on_screen={near_on.sum()}")
        _dof_log(f"depth: screen_pos x [{screen_pos[:,0].min():.1f}, {screen_pos[:,0].max():.1f}]  y [{screen_pos[:,1].min():.1f}, {screen_pos[:,1].max():.1f}]")

        if not np.any(valid):
            return depth_img

        # Start with +inf so np.minimum.at writes every splat distance correctly
        depth_img = np.full((H, W), np.inf, dtype=np.float32)
        flat_idx = py[valid] * W + px[valid]
        d_valid = dists[valid]
        depth_flat = depth_img.ravel()
        np.minimum.at(depth_flat, flat_idx, d_valid)
        depth_img = depth_flat.reshape(H, W)

        # Fill pixels with no splat (still inf) using nearest neighbour
        unfilled = ~np.isfinite(depth_img)
        _dof_log(f"depth: {unfilled.sum()} unfilled pixels after scatter")
        if unfilled.any() and (~unfilled).any():
            try:
                from scipy.ndimage import distance_transform_edt
                _, idx2d = distance_transform_edt(unfilled, return_indices=True)
                depth_img[unfilled] = depth_img[idx2d[0][unfilled], idx2d[1][unfilled]]
            except Exception:
                depth_img[unfilled] = state.focus_distance

        # Any remaining inf (all pixels unfilled) → focus_distance
        depth_img[~np.isfinite(depth_img)] = state.focus_distance


        # Joint bilateral smooth — fills splat gaps while preserving depth edges.
        # Only blends pixels with similar depth values, preventing near/far mixing.
        try:
            from scipy.ndimage import uniform_filter
            depth_range = float(depth_img.max() - depth_img.min())
            sigma_d = max(depth_range * 0.02, 1.0)  # depth similarity threshold (2% of range)
            # Approximate bilateral: iterative small-radius Gaussian on depth-weighted patches
            # Fast version: guided filter using the depth map itself as guide
            r = 4  # spatial radius
            eps = (sigma_d) ** 2
            mean_d = uniform_filter(depth_img, size=2*r+1)
            mean_d2 = uniform_filter(depth_img * depth_img, size=2*r+1)
            var_d = mean_d2 - mean_d * mean_d
            a = var_d / (var_d + eps)
            b = mean_d - a * mean_d
            mean_a = uniform_filter(a, size=2*r+1)
            mean_b = uniform_filter(b, size=2*r+1)
            depth_img = (mean_a * depth_img + mean_b).astype(np.float32)
        except Exception:
            pass

        _dof_log(f"depth image: min={depth_img.min():.3f}  max={depth_img.max():.3f}")
        return depth_img

    except Exception as e:
        import traceback
        _dof_log(f"_compute_depth_image error: {e}\n{traceback.format_exc()}")
        return depth_imgdatetime
import numpy as np
import lichtfeld as lf
import lichtfeld.selection as sel
import lfs_plugins

Operator = lfs_plugins.Operator
Event = lf.ui.Event

from ..panels.dof_panel import state


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

DOF_LOG_PATH = r"c:\temp\DOF_RENDER.TXT"
DOF_VERSION = "v12-viewport-api"

def _dof_log(msg: str):
    try:
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(DOF_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass

_dof_log(f"=== dof_ops.py loaded: {DOF_VERSION} ===")


# ---------------------------------------------------------------------------
# Depth map from splat positions
# Computes a per-pixel depth image from scene splat world positions
# projected through the current camera.
# ---------------------------------------------------------------------------

def _capture_frame():
    """
    Returns (color_np float32 HxWx3, depth_np float32 HxW) or (None, None).
    colour  — from lf.capture_viewport()
    depth   — computed from splat screen positions + camera distances
    """
    try:
        render = lf.capture_viewport()
        if render is None:
            _dof_log("capture_viewport() returned None")
            return None, None
        color_np = render.image.numpy().astype(np.float32)  # (H,W,3 or 4)
        if color_np.ndim == 3 and color_np.shape[2] == 4:
            color_np = color_np[:, :, :3]

        view = lf.get_current_view()
        if view is None:
            _dof_log("get_current_view() returned None")
            return color_np, np.full(color_np.shape[:2], state.focus_distance, np.float32)

        cam_pos = np.array(view.translation.numpy()).flatten()
        depth_np = _compute_depth_image(color_np.shape, cam_pos, view)
        return color_np, depth_np

    except Exception as e:
        _dof_log(f"_capture_frame error: {e}")
        import traceback
        _dof_log(traceback.format_exc())
        return None, None


# ---------------------------------------------------------------------------
# Enable / Disable  (purely a flag — no hook registration needed)
# ---------------------------------------------------------------------------

class DOF_OT_enable(Operator):
    label = "Enable DoF"
    localization_key = ""

    @classmethod
    def poll(cls, context) -> bool:
        return not state.enabled and lf.get_scene() is not None

    def execute(self, context) -> set:
        state.enabled = True
        lf.log.info("[DoF] Enabled")
        lf.get_scene().notify_changed()
        return {"FINISHED"}


class DOF_OT_disable(Operator):
    label = "Disable DoF"
    localization_key = ""

    @classmethod
    def poll(cls, context) -> bool:
        return state.enabled

    def execute(self, context) -> set:
        state.enabled = False
        lf.log.info("[DoF] Disabled")
        lf.get_scene().notify_changed()
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Focus picker — modal operator (same pattern as depthmap point_picker)
# ---------------------------------------------------------------------------

class DOF_OT_pick_focus_point(Operator):
    label = "Pick Focus Point"
    localization_key = ""
    options = {"BLOCKING"}

    @classmethod
    def poll(cls, context) -> bool:
        return lf.get_scene() is not None

    def invoke(self, context, event: Event) -> set:
        state.picking_focus = True
        lf.ui.request_redraw()
        return {"RUNNING_MODAL"}

    def modal(self, context, event: Event) -> set:
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            result = sel.pick_at_screen(event.mouse_region_x, event.mouse_region_y)
            if result is not None:
                pos = result.world_position
                state.focus_pick_pos = tuple(pos)
                view = lf.get_current_view()
                if view is not None:
                    cam = np.array(view.translation.numpy()).flatten()
                    dist = float(np.linalg.norm(np.array(pos) - cam))
                    state.focus_distance = max(dist, 0.01)
                    _dof_log(f"Focus set to {state.focus_distance:.3f}")
                lf.get_scene().notify_changed()
            lf.ui.request_redraw()
            return {"RUNNING_MODAL"}

        if event.type in {"RIGHTMOUSE", "ESC"}:
            state.picking_focus = False
            lf.ui.request_redraw()
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def cancel(self, context):
        state.picking_focus = False
        lf.ui.request_redraw()


class DOF_OT_cancel_pick_focus(Operator):
    label = "Stop Picking"
    localization_key = ""

    def execute(self, context) -> set:
        state.picking_focus = False
        lf.ui.ops.cancel_modal()
        lf.ui.request_redraw()
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Single-frame render
# ---------------------------------------------------------------------------


class DOF_OT_focus_nearest(Operator):
    """Set focus distance to the nearest splat in the current view."""
    label = "Focus on Nearest"
    localization_key = ""

    @classmethod
    def poll(cls, context) -> bool:
        return lf.get_scene() is not None

    def execute(self, context) -> set:
        vp_render = lf.get_viewport_render()
        if vp_render is None:
            return {"CANCELLED"}
        screen_pos = vp_render.screen_positions
        if screen_pos is None:
            return {"CANCELLED"}

        scene = lf.get_scene()
        combined = scene.combined_model()
        if combined is None:
            return {"CANCELLED"}

        means = combined.get_means().numpy()
        view = lf.get_current_view()
        if view is None:
            return {"CANCELLED"}

        cam_pos = np.array(view.translation.numpy()).flatten()
        dists = np.linalg.norm(means - cam_pos[None, :], axis=1)

        sp = screen_pos.numpy()
        W = int(view.width)
        H = int(view.height)
        on_screen = (sp[:, 0] >= 0) & (sp[:, 0] < W) & (sp[:, 1] >= 0) & (sp[:, 1] < H)

        if not np.any(on_screen):
            return {"CANCELLED"}

        nearest = float(dists[on_screen].min())
        state.focus_distance = nearest
        _dof_log(f"Focus on Nearest: {nearest:.3f}")
        lf.get_scene().notify_changed()
        return {"FINISHED"}

class DOF_OT_render_with_dof(Operator):
    label = "Render with DoF"
    localization_key = ""

    @classmethod
    def poll(cls, context) -> bool:
        return lf.get_scene() is not None

    def execute(self, context) -> set:
        import pathlib
        from ..core.dof import apply_dof

        _dof_log("Render with DoF: capturing frame...")
        color_np, depth_np = _capture_frame()

        if color_np is None:
            state.render_status = "ERROR: capture_viewport() failed — see log"
            lf.log.warning("[DoF] Capture failed. Check DOF_RENDER.TXT.")
            return {"CANCELLED"}

        import imageio
        out_dir = pathlib.Path(state.render_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save raw capture so we can verify capture_viewport() is working
        raw_uint8 = (np.clip(color_np, 0.0, 1.0) * 255).astype(np.uint8)
        imageio.imwrite(str(out_dir / "debug_raw.png"), raw_uint8)
        _dof_log(f"Raw capture: {color_np.shape}  range [{color_np.min():.3f}, {color_np.max():.3f}]")

        # Save depth map as greyscale
        d_min, d_max = float(depth_np.min()), float(depth_np.max())
        _dof_log(f"Depth map: min={d_min:.3f}  max={d_max:.3f}  focus={state.focus_distance:.3f}")
        if d_max > d_min:
            depth_vis = ((depth_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_vis = np.zeros(depth_np.shape, dtype=np.uint8)
        imageio.imwrite(str(out_dir / "debug_depth.png"), depth_vis)

        # Log CoC stats and save CoC map
        from ..core.dof import compute_coc
        coc = compute_coc(
            depth_np,
            focal_length_mm=state.focal_length, f_stop=state.f_stop,
            focus_distance=state.focus_distance, sensor_height_mm=state.sensor_height,
            image_height_px=color_np.shape[0], max_coc_px=state.max_coc,
        )
        _dof_log(f"CoC: min={coc.min():.2f}px  max={coc.max():.2f}px  mean={coc.mean():.2f}px")
        coc_vis = (np.clip(coc / max(state.max_coc, 1e-6), 0, 1) * 255).astype(np.uint8)
        imageio.imwrite(str(out_dir / "debug_coc.png"), coc_vis)

        result = apply_dof(
            color=color_np, depth=depth_np,
            focal_length_mm=state.focal_length, f_stop=state.f_stop,
            focus_distance=state.focus_distance, sensor_height_mm=state.sensor_height,
            max_coc_px=state.max_coc, num_bands=state.num_bands,
        )

        out_path = out_dir / "render_dof.png"
        uint8 = (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)
        imageio.imwrite(str(out_path), uint8)

        # Side-by-side comparison
        h = raw_uint8.shape[0]
        divider = np.full((h, 4, 3), 200, dtype=np.uint8)
        imageio.imwrite(str(out_dir / "debug_compare.png"),
                        np.concatenate([raw_uint8, divider, uint8], axis=1))

        state.render_status = f"Saved (+ 3 debug PNGs in {out_dir})"
        _dof_log(f"Done. Check debug_raw/depth/coc/compare.png in {out_dir}")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Render ALL frames
# ---------------------------------------------------------------------------

class DOF_OT_render_all_frames(Operator):
    label = "Render All Frames"
    localization_key = ""

    @classmethod
    def poll(cls, context) -> bool:
        return lf.get_scene() is not None and not state.render_active

    def execute(self, context) -> set:
        import os
        import threading
        import time
        from ..core.dof import apply_dof

        _dof_log("=== Render All Frames: execute() ===")

        output_dir = state.render_output_dir
        total_frames = state.render_total_frames
        fps = state.render_fps

        def catmull_rom(p0, p1, p2, p3, t):
            return 0.5 * (
                2*p1 + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t*t +
                (-p0+3*p1-3*p2+p3)*t*t*t
            )

        def interp_camera(kf_cameras, frame_idx, total):
            n = len(kf_cameras)
            if n == 1:
                return kf_cameras[0]
            t_global = (frame_idx / max(total-1, 1)) * (n-1)
            i = max(0, min(int(t_global), n-2))
            t = t_global - i
            i0, i1 = max(0, i-1), i
            i2, i3 = min(n-1, i+1), min(n-1, i+2)
            result = {}
            for key in ("eye", "target", "up"):
                p0 = np.array(kf_cameras[i0][key])
                p1 = np.array(kf_cameras[i1][key])
                p2 = np.array(kf_cameras[i2][key])
                p3 = np.array(kf_cameras[i3][key])
                result[key] = tuple(catmull_rom(p0, p1, p2, p3, t).tolist())
            return result

        def _run():
            try:
                _dof_log("--- thread started ---")
                state.render_status = "Reading keyframes..."

                rs = lf.get_render_scene()
                nodes = list(rs.get_nodes())
                node_names = [getattr(n, "name", "?") for n in nodes]
                _dof_log(f"nodes: {node_names}")

                kf_container = next(
                    (n for n in nodes if getattr(n, "name", "") == "Keyframes"), None
                )
                if kf_container is None:
                    state.render_status = "ERROR: No Keyframes node found"
                    _dof_log(f"ERROR: no Keyframes. nodes={node_names}")
                    state.render_active = False
                    return

                kf_cameras = []
                for child_id in kf_container.children:
                    child = rs.get_node_by_id(child_id)
                    t = child.world_transform
                    eye = (t[0][3], t[1][3], t[2][3])
                    fwd = (-t[0][2], -t[1][2], -t[2][2])
                    target = (eye[0]+fwd[0]*10, eye[1]+fwd[1]*10, eye[2]+fwd[2]*10)
                    up = (t[0][1], t[1][1], t[2][1])
                    kf_cameras.append({"eye": eye, "target": target, "up": up})

                _dof_log(f"keyframes: {len(kf_cameras)}")
                if not kf_cameras:
                    state.render_status = "ERROR: No keyframe cameras"
                    state.render_active = False
                    return

                frames_dir = os.path.join(output_dir, "dof_frames")
                os.makedirs(frames_dir, exist_ok=True)
                _dof_log(f"frames_dir: {frames_dir}")

                frame_paths = []
                out_w, out_h = 1920, 1080

                for i in range(total_frames):
                    if state.render_cancel:
                        state.render_status = "Cancelled"
                        _dof_log(f"Cancelled at frame {i}")
                        break

                    state.render_status = f"Rendering frame {i+1}/{total_frames}..."
                    state.render_progress = i / total_frames
                    lf.ui.request_redraw()

                    cam = interp_camera(kf_cameras, i, total_frames)
                    lf.set_camera(cam["eye"], cam["target"], cam["up"])
                    lf.ui.request_redraw()
                    time.sleep(0.15)

                    color_np, depth_np = _capture_frame()
                    if color_np is None:
                        state.render_status = f"ERROR: capture failed at frame {i}"
                        _dof_log(f"Frame {i}: capture_frame returned None")
                        state.render_active = False
                        return

                    _dof_log(f"Frame {i}: captured shape={color_np.shape}")

                    result = apply_dof(
                        color=color_np, depth=depth_np,
                        focal_length_mm=state.focal_length, f_stop=state.f_stop,
                        focus_distance=state.focus_distance, sensor_height_mm=state.sensor_height,
                        max_coc_px=state.max_coc, num_bands=state.num_bands,
                    )
                    img_uint8 = (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)

                    h, w = img_uint8.shape[:2]
                    if w != out_w or h != out_h:
                        try:
                            from PIL import Image as PILImage
                            pil = PILImage.fromarray(img_uint8)
                            pil = pil.resize((out_w, out_h), PILImage.LANCZOS)
                            img_uint8 = np.array(pil)
                        except Exception:
                            out_w, out_h = w, h

                    frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
                    try:
                        import imageio
                        imageio.imwrite(frame_path, img_uint8)
                        frame_paths.append(frame_path)
                        _dof_log(f"Frame {i}: saved {frame_path}")
                    except Exception as e:
                        _dof_log(f"Frame {i}: save error: {e}")

                if not state.render_cancel and frame_paths:
                    state.render_status = "Encoding video..."
                    video_path = os.path.join(output_dir, "dof_video.mp4")
                    _dof_log(f"Encoding -> {video_path}")
                    try:
                        import av, imageio
                        container = av.open(video_path, mode="w")
                        stream = container.add_stream("h264", rate=fps)
                        stream.width = out_w
                        stream.height = out_h
                        stream.pix_fmt = "yuv420p"
                        stream.options = {"crf": "18", "preset": "slow"}
                        for j, fp in enumerate(frame_paths):
                            if state.render_cancel:
                                break
                            state.render_status = f"Encoding {j+1}/{len(frame_paths)}..."
                            state.render_progress = j / len(frame_paths)
                            lf.ui.request_redraw()
                            img = imageio.imread(fp)
                            frame_av = av.VideoFrame.from_ndarray(img, format="rgb24")
                            for packet in stream.encode(frame_av):
                                container.mux(packet)
                        for packet in stream.encode():
                            container.mux(packet)
                        container.close()
                        state.render_status = f"Done! {video_path}"
                        state.render_progress = 1.0
                        _dof_log(f"Encode complete: {video_path}")
                    except Exception as e:
                        _dof_log(f"Encode error: {e}")
                        state.render_status = f"Encode error: {e}"

                if not state.render_keep_frames and not state.render_cancel:
                    for fp in frame_paths:
                        try:
                            os.remove(fp)
                        except Exception:
                            pass
                    try:
                        os.rmdir(frames_dir)
                    except Exception:
                        pass

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                _dof_log(f"EXCEPTION: {e}\n{tb}")
                state.render_status = f"ERROR: {e}"
            finally:
                state.render_active = False
                lf.ui.request_redraw()
                _dof_log("--- thread finished ---")

        state.render_active = True
        state.render_cancel = False
        state.render_progress = 0.0
        threading.Thread(target=_run, daemon=True).start()
        return {"FINISHED"}


class DOF_OT_cancel_render_all(Operator):
    label = "Cancel Render"
    localization_key = ""

    @classmethod
    def poll(cls, context) -> bool:
        return state.render_active

    def execute(self, context) -> set:
        state.render_cancel = True
        return {"FINISHED"}
