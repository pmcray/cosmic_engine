"""
Odyssey Director: offline cinematic sequence renderer for the Cosmic Engine.

Generates long-form, widescreen sequences in the spirit of the 'Jupiter Space'
and 'Beyond the Infinite' movements of '2001: A Space Odyssey', at anything
from quick previews up to UHD (3840x2160):

  Act I   THE APPROACH   - photoreal Jovian fluid dynamics, sinking sun,
                           slow dolly toward the cloud deck
  Act II  THE STARGATE   - slit-scan light corridor with evolving colour
                           epochs, tumbling from horizontal to vertical
  Act III THE INFINITE   - gravitationally lensed black hole, mass ramping
                           as the camera falls past the event horizon

Unlike stargate_sequence.py (interactive GUI), this renders headless to an
MP4 or to numbered PNG frames. Frame ranges make multi-minute UHD renders
tractable: split the frame range across machines/sessions and encode the
frames with ffmpeg afterwards.

Examples:
  # 24-second preview straight to MP4
  python odyssey_director.py --preset preview --duration 24 --output preview.mp4

  # Full 4-minute UHD master, rendered in resumable chunks
  python odyssey_director.py --preset uhd --duration 240 --seed 42 \
      --frames-dir frames_uhd --start-frame 0 --end-frame 1440
  python odyssey_director.py --preset uhd --duration 240 --seed 42 \
      --frames-dir frames_uhd --start-frame 1440 --end-frame 2880
  ffmpeg -framerate 24 -i frames_uhd/frame_%06d.png -c:v libx264 \
      -crf 16 -pix_fmt yuv420p odyssey_uhd.mp4

  # Single UHD still (e.g. to check framing/quality)
  python odyssey_director.py --preset uhd --duration 240 --still 120 \
      --still-out uhd_frame.png
"""

import argparse
import math
import os
import sys

import numpy as np
import taichi as ti


PRESETS = {
    # name: (width, height, fluid_res, samples, fps)
    "preview": (640, 360, 192, 1, 12),
    "sd": (960, 540, 256, 2, 24),
    "hd": (1920, 1080, 512, 2, 24),
    "uhd": (3840, 2160, 768, 4, 24),
}

# Act boundaries as fractions of the total sequence
ACT_STARGATE = 0.55
ACT_INFINITE = 0.85


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


class OdysseyDirector:
    def __init__(self, width, height, fluid_res, samples, fps, duration, seed=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.total_frames = max(1, int(round(duration * fps)))

        if seed is not None:
            import random
            random.seed(seed)

        # Imported here so ti.init() (done in main) precedes field allocation
        from render.camera import SphereCamera
        from render.stargate_corridor import StargateCorridor
        from physics.fluid_solver import FluidEngine

        print(f"Odyssey Director: {width}x{height} @ {fps}fps, "
              f"{self.total_frames} frames ({duration:.1f}s)")

        # A calm, zonally constrained Jupiter. The stock 'jupiter' profile is
        # tuned for violent interactive storms; at cinematic pacing those
        # impulses shred the cloud bands into noise. These overrides must be
        # set before the first step() so the kernels compile with them.
        self.fluid = FluidEngine(res=fluid_res, dt=0.004, planet_type="jupiter")
        self.fluid.vorticity_strength = 2.0
        self.fluid.band_freq = 12.0
        self.fluid.wind_mult = 1.5
        self.fluid.shear_mult = 0.5
        self.fluid.has_spot = 1
        self.fluid.has_pearls = 1
        self.fluid.meridional_damping = 0.9
        self.camera = SphereCamera(fluid_res=fluid_res, render_res=max(width, height),
                                   has_rings=0, samples=samples,
                                   render_w=width, render_h=height)
        self.corridor = StargateCorridor(render_w=width, render_h=height,
                                         samples=samples)

        # Fully collapsed ZPHC: we want photoreal clouds, not the pi-lattice
        self.camera.geo_engine.update_observer_attention(1.0)

        # Act III frames the whole accretion disk from outside, which needs
        # a longer geodesic march than the interactive default
        self.camera.bh_steps = 320
        self.camera.bh_dt = 0.08

        print("Pre-warming Jovian fluid dynamics...")
        for pre_frame in range(100):
            self.fluid.step(pre_frame)
        self.fluid_frame = 100

    # ------------------------------------------------------------------ #
    # Per-act frame renderers. Each returns a float32 (W, H, 3) array.   #
    # ------------------------------------------------------------------ #
    def render_jupiter(self, progress, t):
        """Acts I & II run as one continuous gas-giant shot (0 .. ACT_STARGATE)."""
        p = progress / ACT_STARGATE

        cam = self.camera
        # Slow accelerating dolly from deep space down toward the cloud deck
        cam.cam_dist = 5.5 - 3.6 * (p ** 1.6)
        cam.cam_pan = p * 2.2
        cam.cam_tilt = -0.45 + p * 0.2
        # The push into the storm picks up a slow roll in the final third
        roll_p = smoothstep((p - 0.66) / 0.34)
        cam.cam_roll = roll_p * 0.6

        # The sun starts near the camera axis (fully lit disc) and drifts
        # behind the planet as it sinks -- ending on a thin backlit
        # crescent, the 2001 alignment. Azimuth is relative to the camera
        # so the lit face stays in frame for most of the act.
        sun_azimuth = cam.cam_pan + 0.3 + p * 1.6
        ly = 0.5 - p * 0.65
        lx = -math.sin(sun_azimuth)
        lz = -math.cos(sun_azimuth)

        # White-out flash in the last moments: the hard cut into the Stargate
        flash = smoothstep((p - 0.97) / 0.03)
        cam.exposure = 1.0 + flash * 7.0

        cam.render_gas_giant(self.fluid.dye, lx, ly, lz, t)
        return cam.get_image_data()

    def render_stargate(self, progress, t):
        p = (progress - ACT_STARGATE) / (ACT_INFINITE - ACT_STARGATE)
        p = max(0.0, min(1.0, p))

        # Tumble from horizontal planes to vertical mid-act, like the film's
        # cut between corridor orientations -- but as one continuous roll
        roll = smoothstep((p - 0.45) / 0.2) * (math.pi / 2.0)

        # Entry flash decays over the first ~8% of the act
        entry = math.exp(-p * 30.0)
        exposure = 1.0 + entry * 5.0

        self.corridor.render_frame(t, p, roll, exposure)
        return np.clip(self.corridor.pixels.to_numpy(), 0.0, 1.0)

    def render_infinite(self, progress, t):
        p = (progress - ACT_INFINITE) / (1.0 - ACT_INFINITE)
        p = max(0.0, min(1.0, p))

        cam = self.camera
        # A slow fall toward the event horizon, starting wide enough to see
        # the whole lensed accretion disk (mass 0.4 -> disk outer radius
        # ~9.6; render_black_hole puts the camera at 2x cam_dist, so this
        # runs from r=13 down to r=4). The march budget set in __init__
        # covers the full path.
        cam.cam_dist = 6.5 - 4.5 * (p ** 1.4)
        cam.cam_roll = 0.0
        # Looking down onto the disk plane so the far side lenses over the
        # shadow; levelling out as we fall in
        cam.cam_tilt = -0.42 + p * 0.22
        cam.cam_pan = p * 0.35

        cam.geo_engine.update_black_hole_mass(0.4 + p * 0.25)

        # Final fade to black as we cross the horizon
        fade = smoothstep((p - 0.85) / 0.15)
        cam.exposure = 1.0 - fade

        cam.render_black_hole(t)
        return cam.get_image_data()

    # ------------------------------------------------------------------ #
    # Frame assembly                                                      #
    # ------------------------------------------------------------------ #
    def render_frame(self, frame):
        """Returns the finished frame as a uint8 (H, W, 3) RGB image."""
        progress = frame / float(self.total_frames)
        t = frame / float(self.fps)

        if progress < ACT_STARGATE:
            img = self.render_jupiter(progress, t)
        elif progress < ACT_INFINITE:
            img = self.render_stargate(progress, t)
            # Crossfade into the black hole over the act's final 1.5 seconds
            overlap = 1.5 / self.duration
            blend = smoothstep((progress - (ACT_INFINITE - overlap)) / overlap)
            if blend > 0.0:
                bh = self.render_infinite(ACT_INFINITE, t)
                img = img * (1.0 - blend) + bh * blend
        else:
            img = self.render_infinite(progress, t)

        # Taichi fields are (x, y) bottom-up; video wants (row, col) top-down
        frame_img = np.flipud(img.transpose(1, 0, 2))
        return (np.clip(frame_img, 0.0, 1.0) * 255.0).astype(np.uint8)

    def step_simulation(self, frame):
        """Advances stateful simulations by one frame (fluid dynamics).

        The fluid only matters while the gas giant is on screen; skipping it
        afterwards saves hours on long UHD renders and keeps late-frame
        stills cheap, without affecting what any frame looks like.
        """
        if frame / float(self.total_frames) < ACT_STARGATE:
            self.fluid.step(self.fluid_frame)
            self.fluid_frame += 1

    # ------------------------------------------------------------------ #
    # Output drivers                                                      #
    # ------------------------------------------------------------------ #
    def run(self, output=None, frames_dir=None, start_frame=0, end_frame=None,
            overwrite=False):
        import cv2

        end_frame = self.total_frames if end_frame is None else min(end_frame, self.total_frames)

        writer = None
        if output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output, fourcc, float(self.fps),
                                     (self.width, self.height))
        if frames_dir:
            os.makedirs(frames_dir, exist_ok=True)

        # The fluid sim is stateful: earlier frames must be simulated (but
        # not rendered) so a chunk picks up the same atmosphere it would
        # have had in a single continuous run.
        if start_frame > 0:
            print(f"Advancing simulation through {start_frame} unrendered frames...")
            for f in range(start_frame):
                self.step_simulation(f)

        import time as _time
        t0 = _time.time()
        for frame in range(start_frame, end_frame):
            frame_path = None
            if frames_dir:
                frame_path = os.path.join(frames_dir, f"frame_{frame:06d}.png")
                if not overwrite and os.path.exists(frame_path) and writer is None:
                    self.step_simulation(frame)
                    continue

            rgb = self.render_frame(frame)
            self.step_simulation(frame)

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if writer is not None:
                writer.write(bgr)
            if frame_path:
                cv2.imwrite(frame_path, bgr)

            if (frame - start_frame) % 25 == 0:
                done = frame - start_frame + 1
                rate = done / (_time.time() - t0 + 1e-6)
                remaining = (end_frame - frame - 1) / max(rate, 1e-6)
                print(f"Frame {frame}/{self.total_frames} "
                      f"({frame / self.total_frames * 100.0:.1f}%) | "
                      f"{rate:.2f} fps | ~{remaining / 60.0:.1f} min left")

        if writer is not None:
            writer.release()
            print(f"Wrote {output}")
        if frames_dir:
            print(f"Frames in {frames_dir}. Encode with:\n"
                  f"  ffmpeg -framerate {self.fps} -i {frames_dir}/frame_%06d.png "
                  f"-c:v libx264 -crf 16 -pix_fmt yuv420p odyssey.mp4")

    def render_still(self, frame, out_path):
        import cv2
        frame = min(frame, self.total_frames - 1)
        if frame / float(self.total_frames) < ACT_STARGATE:
            print(f"Advancing simulation to frame {frame}...")
            for f in range(frame):
                self.step_simulation(f)
        rgb = self.render_frame(frame)
        cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render a '2001'-flavoured cosmic sequence (headless, up to UHD)")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="sd")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--samples", type=int, help="anti-aliasing samples per pixel")
    parser.add_argument("--fluid-res", type=int, help="fluid simulation grid resolution")
    parser.add_argument("--duration", type=float, default=240.0,
                        help="sequence length in seconds (default 240 = 4 minutes)")
    parser.add_argument("--seed", type=int, help="seed for reproducible planet parameters")
    parser.add_argument("--output", help="write an MP4 directly to this path")
    parser.add_argument("--frames-dir", help="write numbered PNG frames to this directory")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--overwrite", action="store_true",
                        help="re-render PNG frames that already exist")
    parser.add_argument("--still", type=int, help="render only this frame number")
    parser.add_argument("--still-out", default="odyssey_still.png")
    parser.add_argument("--cpu", action="store_true", help="force CPU backend")
    args = parser.parse_args()

    width, height, fluid_res, samples, fps = PRESETS[args.preset]
    width = args.width or width
    height = args.height or height
    fluid_res = args.fluid_res or fluid_res
    samples = args.samples or samples
    fps = args.fps or fps

    init_kwargs = {}
    if args.seed is not None:
        init_kwargs["random_seed"] = args.seed
    if args.cpu:
        ti.init(arch=ti.cpu, **init_kwargs)
    else:
        try:
            ti.init(arch=ti.gpu, **init_kwargs)
        except Exception:
            print("GPU backend unavailable; falling back to CPU.")
            ti.init(arch=ti.cpu, **init_kwargs)

    director = OdysseyDirector(width, height, fluid_res, samples, fps,
                               args.duration, seed=args.seed)

    if args.still is not None:
        director.render_still(args.still, args.still_out)
        return

    if not args.output and not args.frames_dir:
        args.output = "odyssey_sequence.mp4"

    director.run(output=args.output, frames_dir=args.frames_dir,
                 start_frame=args.start_frame, end_frame=args.end_frame,
                 overwrite=args.overwrite)


if __name__ == "__main__":
    main()
