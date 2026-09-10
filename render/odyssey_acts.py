"""Pure-Python parameter curves for the Odyssey Director's three acts.

The Odyssey Director (odyssey_director.py) hard-codes the camera and
lighting choreography of its acts — the Jovian approach dolly, the
corridor tumble, the infall toward the horizon. To reuse those acts as
graph shots (studio.voyage_composer phrase templates → render.adapters
renderers), the choreography lives here as pure functions of the shot's
normalised time t in [0, 1], with every knob overridable through shot
params.

Everything in this module is numpy/stdlib only so the curves stay
testable on CPU-only hosts; the Taichi-side adapters consume the dicts
these functions return.
"""
from __future__ import annotations

import math


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def jovian_approach_pose(
    p: float,
    dist_start: float = 5.5,
    dist_fall: float = 3.6,
    dist_gamma: float = 1.6,
    pan_total: float = 2.2,
    tilt_start: float = -0.45,
    tilt_rise: float = 0.2,
    roll_onset: float = 0.66,
    roll_max: float = 0.6,
    sun_azimuth_offset: float = 0.3,
    sun_azimuth_sweep: float = 1.6,
    sun_ly_start: float = 0.5,
    sun_ly_fall: float = 0.65,
    flash_onset: float = 0.97,
    flash_gain: float = 7.0,
) -> dict:
    """Act I: slow accelerating dolly toward the cloud deck, the sun
    sinking behind the planet into a backlit crescent, white-out at the
    end. Returns camera pose, sun direction, and exposure for progress
    p in [0, 1]."""
    p = max(0.0, min(1.0, p))
    pan = p * pan_total
    roll_p = smoothstep((p - roll_onset) / max(1e-6, 1.0 - roll_onset))
    sun_azimuth = pan + sun_azimuth_offset + p * sun_azimuth_sweep
    flash = smoothstep((p - flash_onset) / max(1e-6, 1.0 - flash_onset))
    return {
        "cam_dist": dist_start - dist_fall * (p ** dist_gamma),
        "cam_pan": pan,
        "cam_tilt": tilt_start + p * tilt_rise,
        "cam_roll": roll_p * roll_max,
        "sun_dir": (
            -math.sin(sun_azimuth),
            sun_ly_start - p * sun_ly_fall,
            -math.cos(sun_azimuth),
        ),
        "exposure": 1.0 + flash * flash_gain,
    }


def corridor_pose(
    p: float,
    tumble_start: float = 0.45,
    tumble_width: float = 0.2,
    tumble_angle: float = math.pi / 2.0,
    entry_flash: float = 5.0,
    flash_decay: float = 30.0,
    base_exposure: float = 1.0,
) -> dict:
    """Act II: slit-scan corridor roll from horizontal to vertical planes
    plus the decaying entry flash."""
    p = max(0.0, min(1.0, p))
    roll = smoothstep((p - tumble_start) / max(1e-6, tumble_width)) * tumble_angle
    exposure = base_exposure + entry_flash * math.exp(-p * flash_decay)
    return {"roll": roll, "exposure": exposure, "progress": p}


def infinite_pose(
    p: float,
    dist_start: float = 6.5,
    dist_fall: float = 4.5,
    dist_gamma: float = 1.4,
    tilt_start: float = -0.42,
    tilt_rise: float = 0.22,
    pan_total: float = 0.35,
    mass_start: float = 0.4,
    mass_ramp: float = 0.25,
    fade_onset: float = 0.85,
) -> dict:
    """Act III: the fall toward the event horizon — dolly in, mass ramp,
    fade to black across the horizon."""
    p = max(0.0, min(1.0, p))
    fade = smoothstep((p - fade_onset) / max(1e-6, 1.0 - fade_onset))
    return {
        "cam_dist": dist_start - dist_fall * (p ** dist_gamma),
        "cam_tilt": tilt_start + p * tilt_rise,
        "cam_pan": p * pan_total,
        "cam_roll": 0.0,
        "bh_mass": mass_start + p * mass_ramp,
        "exposure": 1.0 - fade,
    }
