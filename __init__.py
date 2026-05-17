# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Depth of Field Viewer Plugin for LichtFeld Studio."""

import lichtfeld as lf

from .panels.dof_panel import DoFPanel

try:
    from .operators.dof_ops import (
        DOF_OT_enable,
        DOF_OT_disable,
        DOF_OT_pick_focus_point,
        DOF_OT_cancel_pick_focus,
        DOF_OT_focus_nearest,
    DOF_OT_render_with_dof,
        DOF_OT_render_all_frames,
        DOF_OT_cancel_render_all,
    )
except Exception as _e:
    import traceback
    print("[DoF] operators import failed:\n" + traceback.format_exc())
    lf.log.error("[DoF] operators import failed - see console")
    raise

_classes = [
    DoFPanel,
    DOF_OT_enable,
    DOF_OT_disable,
    DOF_OT_pick_focus_point,
    DOF_OT_cancel_pick_focus,
    DOF_OT_focus_nearest,
    DOF_OT_focus_nearest,
    DOF_OT_render_with_dof,
    DOF_OT_render_all_frames,
    DOF_OT_cancel_render_all,
]


def on_load() -> None:
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("Depth of Field plugin loaded")


def on_unload() -> None:
    try:
        lf.remove_draw_handler("dof_pick_overlay")
    except Exception:
        pass
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("Depth of Field plugin unloaded")
