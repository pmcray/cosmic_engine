"""Voyage composer: procedural generation of multi-shot Stargate sequences.

The ultimate goal of the Cosmic Engine: produce sequences that are
as long or longer than the "Jupiter and Beyond the Infinite" section
of 2001 (~17 minutes), built from many different generators and
subsequences. Each compose run yields a different voyage; the same
seed reproduces it exactly. With dozens of phrase templates and a
narrative-aware phase model, the space of possible voyages is
effectively unbounded.

Architecture:

  Phase: a narrative beat (Approach, Threshold, Tunnel, Galaxies,
         Nebulae, Stars, Alien Worlds, Resolution) with a target
         duration fraction, allowed phrase templates, palette
         preferences, and transition style.

  Phrase template: a function that, given an RNG, builds one Shot
         using a specific renderer + parameter distribution. Each
         template encodes "what this shot type tries to do" -- a
         Jovian approach, a slit-scan tunnel, a cosmic web dive,
         an alien terrain reveal, etc.

  Composer: walks the phases in order, picks `n_shots` per phase
         (within a per-phase range), and within each phase picks
         phrase templates with the phase's preferred weighting.
         Between adjacent shots it inserts a Transition whose kind
         depends on whether they're inside the same phase or
         crossing a phase boundary.

The CLI emits a ShotGraph manifest that can be rendered directly
via GraphRunner.

Usage:

    python -m studio.voyage_composer --duration 600 --seed 42 \\
        --out voyages/voyage_42.json

    # ... then render:
    from director.graph import GraphRunner
    from scene.manifest import load_manifest
    GraphRunner(load_manifest("voyages/voyage_42.json"),
                "out.mp4").run()

For a Colab-friendly proxy render, the CLI also accepts --fast which
halves resolution and march steps. Use --duration 60 --fast for a
1-minute sanity render.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from scene.manifest import (
    Camera,
    PaletteRef,
    Shot,
    ShotGraph,
    Transition,
    dump_manifest,
)


# ============================================================
# Phrase templates -- one per "kind of shot we know how to make"
# ============================================================
#
# Each template returns a (params dict, Camera, motion_hint, default
# palette) tuple. The composer adds the boilerplate fields (id,
# duration, fps, resolution, seed).

TEMPLATE_RETURN = tuple  # (renderer_name, params, camera, motion_hint, default_palette)


def _r(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def _ri(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)


def _cam_push(rng: random.Random,
              z_start_range=(-4.5, -3.2),
              z_end_range=(-2.8, -1.8),
              jitter=0.15,
              fov_start=(34.0, 40.0),
              fov_end=(28.0, 32.0)) -> Camera:
    """Build a slow push-in camera that ends closer than it started."""
    return Camera(
        position=(_r(rng, -jitter, jitter), _r(rng, -jitter, jitter), _r(rng, *z_start_range)),
        target=(0.0, _r(rng, -0.1, 0.1), 0.0),
        up=(0.0, 1.0, 0.0),
        fov_deg=_r(rng, *fov_start),
        path_to=Camera(
            position=(_r(rng, -jitter, jitter), _r(rng, -jitter, jitter), _r(rng, *z_end_range)),
            target=(0.0, _r(rng, -0.15, 0.15), 0.0),
            up=(0.0, 1.0, 0.0),
            fov_deg=_r(rng, *fov_end),
        ),
    )


def _cam_drift(rng: random.Random, distance=3.2, jitter=0.4) -> Camera:
    """Lateral pan or drift across the disc, not zooming."""
    sx, sy, sz = _r(rng, -jitter, jitter), _r(rng, -jitter, jitter), -distance
    return Camera(
        position=(sx, sy, sz),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov_deg=_r(rng, 28.0, 36.0),
        path_to=Camera(
            position=(sx + _r(rng, -0.5, 0.5), sy + _r(rng, -0.25, 0.25), sz),
            target=(0.0, _r(rng, -0.1, 0.1), 0.0),
            up=(0.0, 1.0, 0.0),
            fov_deg=_r(rng, 28.0, 36.0),
        ),
    )


def _cam_tunnel_fly(rng: random.Random) -> Camera:
    """First-person tunnel fly-through."""
    return Camera(
        position=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        fov_deg=_r(rng, 85.0, 110.0),
    )


# ---- Phrase template functions -----------------------------------

def t_jovian_approach(rng: random.Random):
    params = {
        "atm_thickness": _r(rng, 0.04, 0.07),
        "march_steps": rng.choice([24, 28, 32]),
        "samples": 2,
        "sun_dir": [_r(rng, -0.7, -0.4), _r(rng, 0.1, 0.3), _r(rng, -0.85, -0.65)],
        "grs_lon": _r(rng, -120.0, -30.0),
        "seed": _ri(rng, 0, 999_999),
        "push_in_factor": _r(rng, 1.25, 1.55),
    }
    return ("volumetric_gas_giant", params, _cam_push(rng), "push_in", "trumbull_2001")


def t_jovian_grs(rng: random.Random):
    params = {
        "atm_thickness": _r(rng, 0.035, 0.055),
        "march_steps": rng.choice([28, 32, 36]),
        "samples": 2,
        "sun_dir": [_r(rng, -0.6, -0.3), _r(rng, 0.05, 0.2), _r(rng, -0.95, -0.75)],
        "grs_lon": _r(rng, -75.0, -45.0),
        "seed": _ri(rng, 0, 999_999),
    }
    cam = Camera(
        position=(_r(rng, 0.5, 0.8), _r(rng, -0.5, -0.3), _r(rng, -1.6, -1.3)),
        target=(_r(rng, 0.25, 0.4), _r(rng, -0.45, -0.35), 0.0),
        up=(0.0, 1.0, 0.0),
        fov_deg=_r(rng, 18.0, 26.0),
    )
    return ("volumetric_gas_giant", params, cam, "drift", "trumbull_2001")


def t_saturnian_drift(rng: random.Random):
    params = {
        "atm_thickness": _r(rng, 0.04, 0.06),
        "march_steps": rng.choice([24, 28]),
        "samples": 2,
        "sun_dir": [_r(rng, -0.6, -0.3), _r(rng, 0.15, 0.35), _r(rng, -0.85, -0.7)],
        "hex_enable": rng.choice([0, 1]),
        "hex_lat": 78.0,
        "hex_amp": _r(rng, 0.35, 0.55),
        "rings_enable": 1,
        "ring_brightness": _r(rng, 0.85, 1.1),
        "ring_shadows_enable": 1,
        "seed": _ri(rng, 0, 999_999),
    }
    cam = Camera(
        position=(0.0, _r(rng, 0.9, 1.5), _r(rng, -5.2, -4.0)),
        target=(0.0, -0.2, 0.0),
        up=(0.0, 1.0, 0.0),
        fov_deg=_r(rng, 26.0, 32.0),
    )
    return ("saturn_class", params, cam, "drift", "trumbull_2001")


def t_kerr_disc(rng: random.Random):
    params = {
        "spin": _r(rng, 0.4, 0.999),
        "inclination_deg": _r(rng, 60.0, 88.0),
        "disk_outer": _r(rng, 10.0, 18.0),
        "steps": rng.choice([220, 260, 300]),
        "step_size": _r(rng, 0.08, 0.14),
    }
    cam = Camera(
        position=(0.0, _r(rng, 1.0, 4.0), _r(rng, -22.0, -14.0)),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov_deg=_r(rng, 30.0, 40.0),
    )
    return ("kerr_black_hole", params, cam, "approach", "trumbull_2001")


def t_slitscan_tunnel(rng: random.Random):
    params = {"time_scale": _r(rng, 0.035, 0.075)}
    return ("slitscan_tunnel", params, _cam_tunnel_fly(rng), "tunnel", "trumbull_2001")


def t_cosmic_web_wide(rng: random.Random):
    params = {
        "box_size_mpc": 3000.0,
        "grid_dim": 128,
        "n_synth_particles": rng.choice([100_000, 150_000, 200_000]),
        "n_synth_clusters": rng.choice([400, 500, 700]),
        "synth_seed": _ri(rng, 0, 999_999),
        "march_steps": rng.choice([96, 128, 144]),
        "samples": 2,
        "emission_gain": _r(rng, 3.5, 5.5),
        "extinction_strength": _r(rng, 0.08, 0.18),
        "background_gain": _r(rng, 0.3, 0.6),
        "push_in_factor": _r(rng, 1.0, 1.3),
    }
    cam = Camera(
        position=(_r(rng, -200, 200), _r(rng, -100, 100), _r(rng, -2400, -1800)),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov_deg=_r(rng, 45.0, 60.0),
        path_to=Camera(
            position=(_r(rng, -200, 200), _r(rng, -100, 100), _r(rng, -1500, -1000)),
            target=(_r(rng, -100, 100), 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov_deg=_r(rng, 45.0, 60.0),
        ),
    )
    return ("cosmic_web", params, cam, "drift_in", "jwst_nircam")


def t_cosmic_web_dive(rng: random.Random):
    params = {
        "box_size_mpc": 3000.0,
        "grid_dim": 128,
        "n_synth_particles": 150_000,
        "n_synth_clusters": 500,
        "synth_seed": _ri(rng, 0, 999_999),
        "march_steps": rng.choice([128, 144, 160]),
        "samples": 2,
        "emission_gain": _r(rng, 5.0, 7.0),
        "extinction_strength": _r(rng, 0.15, 0.25),
        "background_gain": _r(rng, 0.2, 0.4),
        "push_in_factor": _r(rng, 1.4, 1.8),
    }
    cam = Camera(
        position=(0.0, 0.0, _r(rng, -1300, -1000)),
        target=(0.0, 0.0, _r(rng, 1000, 1500)),
        up=(0.0, 1.0, 0.0),
        fov_deg=_r(rng, 35.0, 45.0),
        path_to=Camera(
            position=(_r(rng, 0, 80), _r(rng, 0, 50), _r(rng, 100, 300)),
            target=(_r(rng, 0, 80), _r(rng, 0, 50), _r(rng, 1300, 1500)),
            up=(0.0, 1.0, 0.0),
            fov_deg=_r(rng, 30.0, 38.0),
        ),
    )
    return ("cosmic_web", params, cam, "deep_dive", "hubble_sii_ha_oiii")


def t_spiral_face_on(rng: random.Random):
    params = {
        "n_arms": rng.choice([2, 2, 3, 4]),
        "pitch_deg": _r(rng, 8.0, 18.0),
        "arm_strength": _r(rng, 0.75, 0.95),
        "arm_width": _r(rng, 0.35, 0.55),
        "dust_strength": _r(rng, 2.0, 3.0),
        "dust_freq": _r(rng, 9.0, 13.0),
        "hii_threshold": _r(rng, 0.18, 0.30),
        "hii_freq": _r(rng, 12.0, 16.0),
        "inclination_deg": _r(rng, 6.0, 18.0),
        "doppler_strength": _r(rng, 0.05, 0.08),
        "companion_mass_ratio": rng.choice([0.0, 0.0, 0.4, 0.5]),
        "bridge_strength": rng.choice([0.0, 0.5, 0.7]),
        "march_steps": rng.choice([60, 72]),
        "samples": 2,
        "seed": _ri(rng, 0, 999_999),
        "push_in_factor": _r(rng, 1.3, 1.6),
    }
    cam = _cam_push(rng, z_start_range=(-3.0, -2.4), z_end_range=(-1.7, -1.3),
                   fov_start=(32.0, 38.0), fov_end=(28.0, 32.0))
    return ("spiral_galaxy", params, cam, "push_in", "jwst_nircam")


def t_spiral_arm_closeup(rng: random.Random):
    params = {
        "n_arms": 2,
        "pitch_deg": _r(rng, 11.0, 16.0),
        "arm_strength": _r(rng, 0.8, 0.95),
        "arm_width": _r(rng, 0.35, 0.45),
        "dust_strength": _r(rng, 2.6, 3.4),
        "dust_freq": _r(rng, 11.0, 14.0),
        "hii_threshold": _r(rng, 0.25, 0.32),
        "inclination_deg": _r(rng, 25.0, 35.0),
        "companion_mass_ratio": 0.0,
        "bridge_strength": 0.0,
        "march_steps": rng.choice([72, 84]),
        "samples": 2,
        "seed": _ri(rng, 0, 999_999),
    }
    cam = Camera(
        position=(_r(rng, 0.3, 0.5), _r(rng, 0.4, 0.6), _r(rng, -1.4, -1.1)),
        target=(_r(rng, 0.2, 0.35), _r(rng, -0.05, 0.05), 0.0),
        up=(0.0, 1.0, 0.0),
        fov_deg=_r(rng, 20.0, 24.0),
    )
    return ("spiral_galaxy", params, cam, "drift", "hubble_sii_ha_oiii")


def _nebula_topology(rng):
    return rng.choice(["pillars", "crab", "helix", "veil", "orion", "pleiades"])


def t_nebula(rng: random.Random):
    topo = _nebula_topology(rng)
    params = {
        "topology": topo,
        "R_bound": 1.5,
        "intensity": _r(rng, 0.9, 1.2),
        "dust_strength": _r(rng, 0.8, 1.4),
        "detail_strength": _r(rng, 0.85, 1.15),
        "march_steps": rng.choice([72, 80, 96]),
        "samples": 2,
        "seed": _ri(rng, 0, 999_999),
        "push_in_factor": _r(rng, 1.1, 1.45),
    }
    if topo == "pillars":
        params["erosion_height"] = _r(rng, 0.18, 0.28)
        params["erosion_decay"] = _r(rng, 2.0, 3.0)
    if topo in ("crab", "veil"):
        params["R_shell"] = _r(rng, 0.9, 1.1)
        params["shell_thickness"] = _r(rng, 0.05, 0.20)
    if topo == "helix":
        params["R_torus"] = _r(rng, 0.6, 0.8)
        params["r_torus"] = _r(rng, 0.15, 0.22)
    cam = _cam_push(rng, z_start_range=(-3.0, -2.4), z_end_range=(-1.8, -1.3),
                   fov_start=(34.0, 40.0), fov_end=(30.0, 35.0))
    palette = rng.choice(["hubble_sii_ha_oiii", "jwst_nircam"])
    return ("diffuse_nebula", params, cam, "push_in", palette)


def t_stellar_sun(rng: random.Random):
    n_spots = rng.choice([2, 3, 4])
    spots = []
    for _ in range(n_spots):
        spots.append([_r(rng, -45, 45), _r(rng, -60, 60),
                      _r(rng, 3.0, 8.0), _r(rng, 0.75, 0.92)])
    n_prom = rng.choice([1, 2, 3])
    prom = []
    for _ in range(n_prom):
        prom.append([_r(rng, -55, 55), _r(rng, -180, 180),
                     _r(rng, 0.07, 0.14), _r(rng, 22, 38)])
    params = {
        "T_eff": _r(rng, 5400, 6200),
        "limb_u": _r(rng, 0.55, 0.65),
        "granule_scale": _r(rng, 18.0, 26.0),
        "supergranule_scale": _r(rng, 3.5, 5.5),
        "corona_thickness": _r(rng, 0.04, 0.07),
        "sunspots": spots,
        "prominences": prom,
        "samples": 2,
        "seed": _ri(rng, 0, 999_999),
        "push_in_factor": _r(rng, 1.15, 1.35),
    }
    return ("stellar_surface", params, _cam_push(rng), "push_in", "trumbull_2001")


def t_stellar_o(rng: random.Random):
    params = {
        "T_eff": _r(rng, 25000, 40000),
        "limb_u": _r(rng, 0.35, 0.5),
        "granule_scale": _r(rng, 26.0, 34.0),
        "supergranule_scale": _r(rng, 5.0, 7.0),
        "corona_thickness": _r(rng, 0.08, 0.14),
        "sunspots": [],
        "prominences": [
            [_r(rng, -15, 15), _r(rng, -180, -160), _r(rng, 0.18, 0.24), _r(rng, 45, 70)],
            [_r(rng, -15, 15), _r(rng, 160, 180), _r(rng, 0.16, 0.22), _r(rng, 40, 60)],
        ],
        "samples": 2,
        "seed": _ri(rng, 0, 999_999),
    }
    return ("stellar_surface", params, _cam_push(rng), "push_in", "jwst_nircam")


def t_stellar_m(rng: random.Random):
    params = {
        "T_eff": _r(rng, 2800, 3800),
        "limb_u": _r(rng, 0.45, 0.55),
        "granule_scale": _r(rng, 14.0, 20.0),
        "supergranule_scale": _r(rng, 3.0, 4.0),
        "corona_thickness": _r(rng, 0.05, 0.08),
        "sunspots": [
            [_r(rng, -25, 25), _r(rng, -45, 45), _r(rng, 9.0, 13.0), _r(rng, 0.88, 0.94)],
            [_r(rng, -25, 25), _r(rng, -45, 45), _r(rng, 7.0, 11.0), _r(rng, 0.85, 0.92)],
        ],
        "prominences": [
            [_r(rng, 35, 60), _r(rng, -120, -60), _r(rng, 0.10, 0.16), _r(rng, 25, 40)],
        ],
        "samples": 2,
        "seed": _ri(rng, 0, 999_999),
    }
    return ("stellar_surface", params, _cam_drift(rng), "drift", "trumbull_2001")


def _exoplanet_topology(rng):
    return rng.choice(["hot_jupiter", "mini_neptune", "brown_dwarf"])


def t_exoplanet(rng: random.Random):
    topo = _exoplanet_topology(rng)
    params = {
        "topology": topo,
        "atm_thickness": _r(rng, 0.06, 0.12),
        "march_steps": rng.choice([28, 32]),
        "samples": 2,
        "sun_dir": [_r(rng, -0.7, -0.4), _r(rng, 0.05, 0.2), _r(rng, -0.9, -0.7)],
        "seed": _ri(rng, 0, 999_999),
        "push_in_factor": _r(rng, 1.15, 1.45),
    }
    palette = "jwst_nircam" if topo == "mini_neptune" else "trumbull_2001"
    return ("exoplanet_atmosphere", params, _cam_push(rng), "push_in", palette)


def t_terrain(rng: random.Random):
    alien = rng.random() < 0.5
    params = {
        "terrain_amplitude": _r(rng, 0.9, 1.7),
        "terrain_freq": _r(rng, 0.04, 0.09),
        "octaves": rng.choice([6, 7, 8]),
        "snow_line": 5.0 if alien else _r(rng, 0.6, 0.85),
        "water_level": -10.0 if alien else _r(rng, -0.25, -0.10),
        "water_enable": 0 if alien else 1,
        "sun_dir": [_r(rng, 0.3, 0.7), _r(rng, 0.3, 0.6), _r(rng, -0.8, -0.5)],
        "haze_strength": _r(rng, 0.45, 0.8),
        "max_dist": _r(rng, 70.0, 110.0),
        "march_steps": rng.choice([130, 150]),
        "samples": 1,
        "seed": _ri(rng, 0, 999_999),
    }
    cam = Camera(
        position=(_r(rng, -5.0, -1.5), _r(rng, 0.9, 1.4), _r(rng, -6.0, -1.5)),
        target=(_r(rng, 2.0, 5.0), _r(rng, 0.4, 0.8), _r(rng, 3.0, 5.0)),
        up=(0.0, 1.0, 0.0),
        fov_deg=_r(rng, 48.0, 58.0),
        path_to=Camera(
            position=(_r(rng, -1.0, 2.0), _r(rng, 1.0, 1.4), _r(rng, -3.5, -0.5)),
            target=(_r(rng, 3.0, 5.0), _r(rng, 0.4, 0.8), _r(rng, 4.0, 5.5)),
            up=(0.0, 1.0, 0.0),
            fov_deg=_r(rng, 45.0, 55.0),
        ),
    )
    palette = "jwst_nircam" if alien else "trumbull_2001"
    return ("terrain", params, cam, "fly_through", palette)


def t_ca_creatures(rng: random.Random):
    params = {
        "grid_size": 256,
        "A_R": rng.choice([11, 13, 15]),
        "A_kernel_mu": _r(rng, 0.4, 0.55),
        "A_kernel_sigma": _r(rng, 0.12, 0.18),
        "A_growth_mu": _r(rng, 0.13, 0.17),
        "A_growth_sigma": _r(rng, 0.013, 0.018),
        "B_R": rng.choice([22, 24, 28]),
        "B_kernel_mu": _r(rng, 0.5, 0.6),
        "B_kernel_sigma": _r(rng, 0.10, 0.14),
        "B_growth_mu": _r(rng, 0.10, 0.14),
        "B_growth_sigma": _r(rng, 0.018, 0.022),
        "dt": 0.1,
        "substeps_per_frame": rng.choice([2, 3]),
        "warmup_steps": rng.choice([80, 100]),
        "samples": 1,
        "seed": _ri(rng, 0, 999_999),
    }
    cam = Camera(
        position=(_r(rng, -0.2, 0.2), _r(rng, -0.2, 0.2), _r(rng, -1.6, -1.0)),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov_deg=60.0,
    )
    return ("ca_creatures", params, cam, "zoom_in", "trumbull_2001")


def t_odyssey_jovian(rng: random.Random):
    """Act I of the Odyssey Director as a phrase: fluid-simulated Jovian
    approach with the sinking sun. The renderer drives its own camera
    choreography, so the manifest camera is a plain default."""
    params = {
        "fluid_res": 256,
        "samples": 2,
        "prewarm_steps": _ri(rng, 80, 140),
        "vorticity_strength": _r(rng, 1.6, 2.4),
        "band_freq": _r(rng, 10.0, 14.0),
        "wind_mult": _r(rng, 1.2, 1.8),
        "pan_total": _r(rng, 1.8, 2.6),
        "roll_max": _r(rng, 0.4, 0.75),
        "sun_azimuth_sweep": _r(rng, 1.2, 2.0),
        "flash": 0,  # the white-out belongs to the tunnel boundary, not mid-voyage
        "seed": _ri(rng, 0, 999_999),
    }
    return ("odyssey_jovian_approach", params, Camera(), "push_in", "trumbull_2001")


def t_odyssey_jovian_flash(rng: random.Random):
    """Act I variant that keeps the terminal white-out — the classic cut
    into a tunnel phase."""
    renderer, params, cam, motion, palette = t_odyssey_jovian(rng)
    params["flash"] = 1
    return (renderer, params, cam, motion, palette)


def t_stargate_corridor(rng: random.Random):
    """Act II of the Odyssey Director: widescreen slit-scan corridor with
    colour epochs and the horizontal-to-vertical tumble."""
    params = {
        "samples": 2,
        "tumble_start": _r(rng, 0.35, 0.55),
        "tumble_width": _r(rng, 0.15, 0.30),
        "entry_flash": _r(rng, 3.0, 6.0),
        "flash_decay": _r(rng, 20.0, 40.0),
        "seed": _ri(rng, 0, 999_999),
    }
    return ("stargate_corridor", params, _cam_tunnel_fly(rng), "tunnel", "trumbull_2001")


def t_odyssey_infinite(rng: random.Random):
    """Act III of the Odyssey Director: lensed black-hole infall with the
    mass ramping. Mid-voyage variant — no fade to black."""
    params = {
        "samples": 2,
        "bh_steps": rng.choice([260, 300, 320]),
        "bh_dt": 0.08,
        "mass_start": _r(rng, 0.35, 0.45),
        "mass_ramp": _r(rng, 0.18, 0.30),
        "dist_start": _r(rng, 6.0, 7.0),
        "dist_fall": _r(rng, 4.0, 5.0),
        "fade": 0,
        "seed": _ri(rng, 0, 999_999),
    }
    return ("odyssey_infinite", params, Camera(), "approach", "trumbull_2001")


def t_odyssey_infinite_finale(rng: random.Random):
    """Act III variant that keeps the fade to black across the horizon —
    meant for the resolution phase's closing shot."""
    renderer, params, cam, motion, palette = t_odyssey_infinite(rng)
    params["fade"] = 1
    return (renderer, params, cam, motion, palette)


# ---- Fractal dive phrases ------------------------------------------
#
# Deep-dive locations. Each is a point on the boundary with structure at
# every depth; the digit count bounds how deep that centre can be pushed
# before the reference orbit needs re-deriving.

FRACTAL_LOCATIONS = (
    # (name, center_re, center_im, power)
    # Each centre was bisected at high precision until its orbit survives
    # thousands of iterations, so the dive keeps finding structure all the
    # way down; the digit count bounds how deep it can be pushed before the
    # reference orbit needs re-deriving.
    ("seahorse_valley",
     "-0.737751206161254491181307210762363450",
     "0.1276207733591109064923511174750151381", 2),
    ("elephant_valley",
     "0.2924137270354271177935041176362840817",
     "0.0149056207922282518104476262507482955", 2),
    ("triple_spiral",
     "-0.087947259278037183807137999348274103",
     "0.6552960232344219817403402651021707434", 2),
    ("scepter_valley",
     "-1.749164760990536265979793535947397794",
     "-0.000354908863341066613549520963946711", 2),
    ("dendrite",
     "-0.129684478118252256638573814717243139",
     "0.8756983234785321890495082748370532110", 2),
    ("multibrot_3",
     "0.3575572475710187173097895550841719553",
     "0.7101740570828986189270892820783667134", 3),
    ("multibrot_4",
     "0.6370710737609568638007646787830163914",
     "0.2803614520918594356942172103114730702", 4),
    ("multibrot_5",
     "0.7892825805319152899034950303947405545",
     "-0.658215581388023392607858154002439068", 5),
)


def _fractal_location(rng: random.Random):
    return rng.choice(FRACTAL_LOCATIONS)


def t_fractal_dive(rng: random.Random):
    """A long exponential plunge into the set — the Multibrot answer to
    the slit-scan tunnel. Deep enough that only perturbation reaches it."""
    name, cre, cim, power = _fractal_location(rng)
    start = _r(rng, 1.2, 2.2)
    end = 10.0 ** _r(rng, -22.0, -14.0)
    params = {
        "center_re": cre,
        "center_im": cim,
        "power": power,
        "scale_start": start,
        "scale_end": end,
        "rotation_deg_total": _r(rng, -220.0, 220.0),
        "samples": 2,
        "color_cycles": _r(rng, 3.0, 7.0),
        "color_phase": _r(rng, 0.0, 1.0),
        "iter_per_decade": rng.choice([600, 800, 1000]),
        "seed": _ri(rng, 0, 999_999),
    }
    return ("fractal_dive", params, Camera(), "deep_dive", "trumbull_2001")


def t_fractal_dive_shallow(rng: random.Random):
    """A shorter dive that stays in reach of visible large-scale
    structure — used as an approach or resolution beat rather than a
    tunnel."""
    name, cre, cim, power = _fractal_location(rng)
    params = {
        "center_re": cre,
        "center_im": cim,
        "power": power,
        "scale_start": _r(rng, 0.6, 1.4),
        "scale_end": 10.0 ** _r(rng, -8.0, -5.0),
        "rotation_deg_total": _r(rng, -60.0, 60.0),
        "samples": 2,
        "color_cycles": _r(rng, 2.5, 5.5),
        "color_phase": _r(rng, 0.0, 1.0),
        "seed": _ri(rng, 0, 999_999),
    }
    palette = rng.choice(["trumbull_2001", "hubble_sii_ha_oiii"])
    return ("fractal_dive", params, Camera(), "push_in", palette)


# ============================================================
# Phases
# ============================================================

@dataclass
class Phase:
    name: str
    duration_frac: float       # share of total target frames
    n_shots: tuple[int, int]   # min, max shots within the phase
    templates: tuple[Callable, ...]
    palette_pool: tuple[str, ...]
    transition_kind: str = "crossfade"
    transition_to_next: str = "crossfade"


PHASES: tuple[Phase, ...] = (
    Phase(
        name="approach",
        duration_frac=0.12,
        n_shots=(3, 4),
        templates=(t_jovian_approach, t_odyssey_jovian, t_saturnian_drift, t_stellar_sun, t_exoplanet),
        palette_pool=("trumbull_2001", "nfb_universe_1960"),
    ),
    Phase(
        name="threshold",
        duration_frac=0.10,
        n_shots=(3, 4),
        templates=(t_kerr_disc, t_odyssey_infinite, t_jovian_grs, t_stellar_m,
                   t_nebula, t_fractal_dive_shallow),
        palette_pool=("trumbull_2001", "hubble_sii_ha_oiii"),
        transition_to_next="slitscan",
    ),
    Phase(
        name="tunnel",
        duration_frac=0.10,
        n_shots=(2, 3),
        templates=(t_slitscan_tunnel, t_stargate_corridor, t_fractal_dive,
                   t_kerr_disc, t_stargate_corridor, t_fractal_dive),
        palette_pool=("trumbull_2001",),
        transition_kind="hard_cut",
        transition_to_next="slitscan",
    ),
    Phase(
        name="galaxies",
        duration_frac=0.15,
        n_shots=(3, 5),
        templates=(t_cosmic_web_wide, t_cosmic_web_dive, t_spiral_face_on, t_spiral_arm_closeup),
        palette_pool=("jwst_nircam", "hubble_sii_ha_oiii"),
    ),
    Phase(
        name="nebulae",
        duration_frac=0.13,
        n_shots=(3, 5),
        templates=(t_nebula, t_nebula, t_nebula, t_nebula),
        palette_pool=("hubble_sii_ha_oiii", "jwst_nircam"),
    ),
    Phase(
        name="stars",
        duration_frac=0.12,
        n_shots=(3, 4),
        templates=(t_stellar_sun, t_stellar_o, t_stellar_m, t_exoplanet),
        palette_pool=("trumbull_2001", "jwst_nircam"),
    ),
    Phase(
        name="alien_worlds",
        duration_frac=0.16,
        n_shots=(4, 6),
        templates=(t_terrain, t_terrain, t_ca_creatures, t_exoplanet, t_terrain),
        palette_pool=("jwst_nircam", "trumbull_2001"),
    ),
    Phase(
        name="resolution",
        duration_frac=0.12,
        n_shots=(3, 4),
        templates=(t_saturnian_drift, t_spiral_face_on, t_nebula, t_jovian_approach,
                   t_odyssey_infinite_finale, t_fractal_dive_shallow),
        palette_pool=("nfb_universe_1960", "trumbull_2001"),
    ),
)


# ============================================================
# Composer
# ============================================================

def compose_voyage(
    target_seconds: float = 600.0,
    fps: int = 24,
    seed: int = 42,
    resolution: tuple[int, int] = (1280, 720),
    title: str | None = None,
) -> ShotGraph:
    """Build a full ShotGraph for a voyage of approximately `target_seconds`."""
    rng = random.Random(seed)
    target_frames = int(target_seconds * fps)
    if title is None:
        title = f"voyage_seed_{seed}"

    shots: list[Shot] = []
    transitions: list[Transition] = []

    for phase_idx, phase in enumerate(PHASES):
        n_shots = rng.randint(*phase.n_shots)
        phase_frames = int(target_frames * phase.duration_frac)
        # Distribute phase frames across n_shots with small variation.
        per_shot = max(48, phase_frames // n_shots)
        var = max(0, per_shot // 4)

        for shot_idx in range(n_shots):
            template = rng.choice(phase.templates)
            renderer, params, camera, motion, default_palette = template(rng)
            palette = rng.choice(phase.palette_pool) or default_palette
            duration = max(48, per_shot + rng.randint(-var, var))

            sid = f"{phase.name}_{shot_idx:02d}_{renderer}_{_ri(rng, 1000, 9999)}"
            shot = Shot(
                id=sid,
                renderer=renderer,
                params=params,
                camera=camera,
                palette=PaletteRef(name=palette),
                duration_frames=duration,
                fps=fps,
                resolution=resolution,
                seed=params.get("seed", 0),
                motion_hint=motion,
            )

            # Insert transition before this shot.
            if shots:
                prev_phase_idx = phase_idx if shot_idx > 0 else phase_idx - 1
                prev_phase = PHASES[prev_phase_idx] if prev_phase_idx >= 0 else phase
                if shot_idx == 0:
                    kind = prev_phase.transition_to_next
                else:
                    kind = phase.transition_kind
                t_duration = {
                    "hard_cut": 0,
                    "crossfade": rng.choice([14, 16, 18]),
                    "slitscan": rng.choice([20, 24, 28]),
                    "match_cut": rng.choice([10, 12, 14]),
                }.get(kind, 16)
                trans_params = {}
                if kind == "crossfade":
                    trans_params = {
                        "histogram_strength": _r(rng, 0.45, 0.7),
                        "smoothstep": True,
                    }
                elif kind == "slitscan":
                    trans_params = {"intensity": _r(rng, 1.1, 1.4)}
                if t_duration > 0:
                    transitions.append(Transition(
                        from_shot=shots[-1].id,
                        to_shot=sid,
                        kind=kind,
                        duration_frames=t_duration,
                        params=trans_params,
                    ))

            shots.append(shot)

    graph = ShotGraph(shots=shots, transitions=transitions, title=title)
    graph.validate()
    return graph


def total_duration_seconds(graph: ShotGraph) -> float:
    """Return the total voyage duration in seconds, accounting for
    overlapping transition frames (each transition shortens the timeline
    by its duration)."""
    total_frames = sum(s.duration_frames for s in graph.shots)
    overlap_frames = sum(t.duration_frames for t in graph.transitions)
    fps = graph.shots[0].fps if graph.shots else 24
    return (total_frames - overlap_frames) / fps


# ============================================================
# CLI
# ============================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voyage_composer")
    parser.add_argument("--duration", type=float, default=600.0,
                        help="target voyage duration in seconds")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", default="1280x720",
                        help="WIDTHxHEIGHT")
    parser.add_argument("--title", default=None)
    parser.add_argument("--out", required=True, help="output JSON manifest path")
    parser.add_argument("--fast", action="store_true",
                        help="halve resolution + reduce march_steps for proxy render")
    args = parser.parse_args(argv)

    w, h = (int(x) for x in args.resolution.split("x"))
    if args.fast:
        w, h = w // 2, h // 2

    graph = compose_voyage(
        target_seconds=args.duration,
        fps=args.fps,
        seed=args.seed,
        resolution=(w, h),
        title=args.title,
    )

    if args.fast:
        # Reduce march steps where the param is exposed.
        for s in graph.shots:
            for k in ("march_steps", "steps"):
                if k in s.params:
                    s.params[k] = max(16, int(s.params[k] * 0.6))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    text = dump_manifest(graph, args.out)

    actual = total_duration_seconds(graph)
    print(
        f"composed {len(graph.shots)} shots across {len(PHASES)} phases, "
        f"target {args.duration:.0f}s -> actual {actual:.1f}s, "
        f"transitions: {len(graph.transitions)}"
    )
    print(f"manifest written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
