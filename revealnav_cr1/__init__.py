"""Versioned implementation helpers for MF2-CR1.

This package is deliberately separate from the frozen ``toporeveal`` and
ETP-R1 trees until the Phase-0C automatic-front-end gate is accepted.
"""

from .causal_frontend import (  # noqa: F401
    CausalPoseViewBuffer,
    apply_raw_view_mask,
    causal_vp_feature_variable,
    filter_waypoint_outputs,
)

