# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Side-panel UI for the Depth of Field plugin."""

from __future__ import annotations
import lichtfeld as lf


class _State:
    enabled: bool = False
    focal_length: float = 50.0
    f_stop: float = 2.8
    focus_distance: float = 5.0
    max_coc: float = 24.0
    sensor_height: float = 24.0
    num_bands: int = 6
    show_coc_debug: bool = False

    # Picker
    picking_focus: bool = False
    focus_pick_pos: tuple | None = None

    # Render-all-frames
    render_active: bool = False
    render_cancel: bool = False
    render_progress: float = 0.0
    render_status: str = ""
    render_output_dir: str = "c:\\temp"
    render_total_frames: int = 300
    render_fps: int = 24
    render_keep_frames: bool = True


state = _State()

# ---------------------------------------------------------------------------
# Viewport overlay draw handler
# ---------------------------------------------------------------------------

_draw_handler_registered = False


def _dof_draw_handler(ctx):
    if state.focus_pick_pos is not None:
        ctx.draw_point_3d(state.focus_pick_pos, (0.0, 0.8, 1.0, 1.0), 18.0)
        screen = ctx.world_to_screen(state.focus_pick_pos)
        if screen:
            ctx.draw_circle_2d(screen, 14.0, (0.0, 0.8, 1.0, 1.0), 2.0)
            ctx.draw_text_2d(
                (screen[0] + 16, screen[1] - 8),
                f"Focus: {state.focus_distance:.3f}",
                (0.0, 0.8, 1.0, 1.0),
            )
    if state.picking_focus:
        ctx.draw_text_2d(
            (20, 50),
            "PICK FOCUS: Click on model  (ESC / Right-click to cancel)",
            (0.0, 1.0, 0.5, 0.95),
        )


def _ensure_draw_handler():
    global _draw_handler_registered
    if not _draw_handler_registered:
        try:
            lf.remove_draw_handler("dof_pick_overlay")
        except Exception:
            pass
        lf.add_draw_handler("dof_pick_overlay", _dof_draw_handler, "POST_VIEW")
        _draw_handler_registered = True


# ---------------------------------------------------------------------------
# Panel — uses layout.button() returning bool, never ui.operator_()
# ---------------------------------------------------------------------------

class DoFPanel(lf.ui.Panel):
    id = "dof_viewer.panel"
    label = "DoF Render"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 20
    poll_dependencies = {lf.ui.PollDependency.SCENE}

    # operator id prefix
    _OPS = "lfs_plugins.dof_render.operators.dof_ops"

    @classmethod
    def poll(cls, context) -> bool:
        return lf.get_scene() is not None

    def draw(self, layout) -> None:
        _ensure_draw_handler()
        theme = lf.ui.theme()
        scale = layout.get_dpi_scale()

        # --- Status & toggle ---
        if not state.enabled:
            layout.text_colored("Status: Inactive", theme.palette.text_dim)
            if layout.button("Enable DoF##dof_enable", (-1, 28 * scale)):
                lf.ui.ops.invoke(f"{self._OPS}.DOF_OT_enable")
        else:
            layout.text_colored("Status: Active", (0.4, 1.0, 0.4, 1.0))
            if layout.button("Disable DoF##dof_disable", (-1, 28 * scale)):
                lf.ui.ops.invoke(f"{self._OPS}.DOF_OT_disable")

        # --- Lens ---
        layout.separator()
        if layout.collapsing_header("Lens##dof_lens", default_open=True):
            changed, val = layout.drag_float("Focal Length (mm)##fl", state.focal_length, 0.5, 1.0, 2000.0)
            if changed:
                state.focal_length = val
            changed, val = layout.drag_float("F-Stop##fs", state.f_stop, 0.05, 0.7, 32.0)
            if changed:
                state.f_stop = val
            changed, val = layout.drag_float("Sensor Height (mm)##sh", state.sensor_height, 0.1, 1.0, 100.0)
            if changed:
                state.sensor_height = val

        # --- Focus ---
        layout.separator()
        if layout.collapsing_header("Focus##dof_focus", default_open=True):
            changed, val = layout.drag_float("Focus Distance##fd", state.focus_distance, 0.05, 0.01, 10000.0)
            if changed:
                state.focus_distance = val

            if state.focus_pick_pos:
                layout.text_colored(
                    f"Picked: ({state.focus_pick_pos[0]:.2f}, "
                    f"{state.focus_pick_pos[1]:.2f}, "
                    f"{state.focus_pick_pos[2]:.2f})  d={state.focus_distance:.3f}",
                    (0.0, 0.8, 1.0, 1.0),
                )

            if layout.button("Focus on Nearest##dof_focusnearest", (-1, 24)):
                lf.ui.ops.invoke(f"{self._OPS}.DOF_OT_focus_nearest")
            if state.picking_focus:
                layout.text_colored("Click scene to set focus... (ESC / RMB to cancel)", (0.0, 1.0, 0.5, 1.0))
                if layout.button("[x] Stop Picking##dof_stoppick", (-1, 28 * scale)):
                    lf.ui.ops.invoke(f"{self._OPS}.DOF_OT_cancel_pick_focus")
            else:
                if layout.button("Pick Focus Point##dof_pick", (-1, 28 * scale)):
                    lf.ui.ops.invoke(f"{self._OPS}.DOF_OT_pick_focus_point")

        # --- Quality ---
        layout.separator()
        if layout.collapsing_header("Quality##dof_quality", default_open=False):
            changed, val = layout.drag_float("Max CoC (px)##coc", state.max_coc, 0.5, 1.0, 128.0)
            if changed:
                state.max_coc = val
            changed, val = layout.input_int("Blur Bands##bands", state.num_bands, 1, 4)
            if changed:
                state.num_bands = max(1, min(val, 16))
            changed, val = layout.checkbox("Show CoC Debug##cocdebug", state.show_coc_debug)
            if changed:
                state.show_coc_debug = val

        # --- Single-frame render ---
        layout.separator()
        if layout.button("Render with DoF##dof_render1", (-1, 28 * scale)):
            lf.ui.ops.invoke(f"{self._OPS}.DOF_OT_render_with_dof")

        # --- Render All Frames ---
        layout.separator()
        if layout.collapsing_header("Render All Frames##dof_renderall_hdr", default_open=True):
            layout.label("Output folder:")
            changed, val = layout.path_input("##outdir", state.render_output_dir)
            if changed:
                state.render_output_dir = val

            changed, val = layout.input_int("Frames##renderframes", state.render_total_frames, 1, 10)
            if changed:
                state.render_total_frames = max(1, val)

            changed, val = layout.input_int("FPS##renderfps", state.render_fps, 1, 5)
            if changed:
                state.render_fps = max(1, val)

            changed, val = layout.checkbox("Keep Frame PNGs##keepframes", state.render_keep_frames)
            if changed:
                state.render_keep_frames = val

            duration = state.render_total_frames / max(state.render_fps, 1)
            layout.text_colored(
                f"Duration: {duration:.1f}s  ({state.render_total_frames} frames @ {state.render_fps}fps)",
                theme.palette.text_dim,
            )

            layout.spacing()
            if state.render_active:
                layout.progress_bar(state.render_progress, height=20.0 * scale)
                layout.text_colored(state.render_status, (0.8, 0.8, 0.2, 1.0))
                if layout.button("Cancel Render##dof_cancelrender", (-1, 0)):
                    lf.ui.ops.invoke(f"{self._OPS}.DOF_OT_cancel_render_all")
            else:
                if layout.button("Render All Frames##dof_renderall", (-1, 28 * scale)):
                    lf.ui.ops.invoke(f"{self._OPS}.DOF_OT_render_all_frames")
                if state.render_status:
                    color = (1.0, 0.4, 0.4, 1.0) if "ERROR" in state.render_status else (0.4, 1.0, 0.4, 1.0)
                    layout.text_colored(state.render_status, color)
