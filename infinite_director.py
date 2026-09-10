import os
import sys
import subprocess
import time
import shutil
import taichi as ti

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Taichi is initialized lazily: importing this module must not seize the
# GPU, since the manifest path and the worlds package are both usable
# without it.
_TI_READY = False


def ensure_taichi():
    global _TI_READY
    if not _TI_READY:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
        _TI_READY = True


def get_astrometric_waypoint(seed=None):
    """Pick the next world to visit.

    Sources worlds through the `worlds` package rather than importing
    worldmaker from a hardcoded path: if a worldmaker checkout is
    reachable (COSMIC_WORLDMAKER_PATH) it is used, otherwise the
    built-in procedural Traveller generator stands in, so this runs on
    any machine. Returns (sector, world) — the sector replaces the old
    `system` object and carries the rest of the neighbourhood.
    """
    from worlds import generate_sector
    from worlds.providers import get_provider

    if seed is None:
        seed = int(time.time()) & 0x7FFFFFFF
    sector = generate_sector(seed=seed, name="Foreven",
                             provider=get_provider(sector="Foreven"))
    pool = sector.habitable() or sector.terrestrials() or sector.worlds
    chosen = pool[seed % len(pool)] if pool else None
    return sector, chosen


def generate_planetary_segment(world, output_name, test_mode=False):
    """Render a planetary sequence for `world`.

    Prefers a sister surface pipeline (weorold, then Erith) when one is
    reachable; otherwise renders the world through the engine's own
    terrain renderer via the manifest path, so a planetary segment is
    always produced.
    """
    from worlds.providers import (ErithProvider, ProviderUnavailable,
                                  WeoroldProvider)

    print(f"Generating planetary view for {world.name}: {world.summary()}")

    for provider in (WeoroldProvider(), ErithProvider()):
        if not provider.available():
            continue
        try:
            result = provider.generate_surface(world, output_name,
                                               test_mode=test_mode)
        except ProviderUnavailable as exc:
            print(f"  {provider.name} unavailable: {exc}")
            continue
        # Pipelines return a path, or write "<output_name>_orbit.mp4"
        # beside themselves.
        if isinstance(result, str) and os.path.exists(result):
            return result
        candidate = f"{output_name}_orbit.mp4"
        if os.path.exists(candidate):
            return candidate
        print(f"  {provider.name} produced no output; falling through")

    return _render_world_natively(world, output_name, test_mode=test_mode)


def _render_world_natively(world, output_name, test_mode=False):
    """Render one world as a single-shot graph with the engine's own
    renderers — the no-sister-repos path."""
    import render.adapters  # noqa: F401  ensures renderers are registered
    from director.graph import render_graph
    from scene.manifest import ShotGraph
    from worlds.adapter import shot_for_world

    ensure_taichi()
    shot = shot_for_world(
        world,
        duration_frames=30 if test_mode else 120,
        resolution=(512, 288) if test_mode else (1920, 1080),
    )
    out_path = f"{output_name}.mp4"
    render_graph(ShotGraph(shots=[shot], title=world.name), out_path)
    return out_path

def generate_exotic_segment(encounter_type, output_name, test_mode=False):
    """
    Hooks into cosmic_engine to render exotic encounters (L-systems or Geodesic Blackholes).
    """
    ensure_taichi()
    print(f"Generating exotic segment: {encounter_type}")
    if encounter_type in ["blackhole", "hypernova"]:
         from encounters.exotic_physics import ExoticPhysicsEncounter
         frames = 30 if test_mode else 120
         enc = ExoticPhysicsEncounter(encounter_type, res=512 if test_mode else 1024)
         full_path = f"{output_name}.mp4"
         enc.render_to_mp4(full_path, total_frames=frames)
         return full_path
    else:
         # Megastructure Encounter
         from encounters.megastructure import MegastructureEncounter
         frames = 60 # They control own scaling
         # Extract the base style from the anomaly name if needed
         style = "alien_structure"
         if "chthonic" in encounter_type:
              style = "prehistoric_mega_flora"
         elif "megastructure" in encounter_type:
              style = "alien_structure"
         
         enc = MegastructureEncounter(obj_style=style, res=512 if test_mode else 1024)
         full_path = f"{output_name}.mp4"
         enc.render_to_mp4(full_path, total_frames=frames, test_mode=test_mode)
         return full_path

@ti.data_oriented
class ZPHCTransitionRenderer:
    def __init__(self, res=1024):
        self.res = res
        self.pixels = ti.Vector.field(3, dtype=float, shape=(res, res))
        
    @ti.kernel
    def render_frame(self, time: float, progress: float):
        for i, j in self.pixels:
             u = (float(i) / self.res) * 2.0 - 1.0
             v = (float(j) / self.res) * 2.0 - 1.0
             
             # Relativistic Slit-Scan Tunnel
             # As progress approaches 1.0, the tunnel collapses into a singularity
             r = ti.sqrt(u**2 + v**2) + 1e-5
             angle = ti.atan2(v, u)
             
             # Slit-scan warp
             z = 1.0 / r + time * 10.0
             
             # Harmonic collapse noise (Zero-Point)
             noise = ti.sin(z * 5.0 + angle * 8.0) * ti.cos(z * 3.0 - time * 2.0)
             
             # Color shifting
             # Starts mostly dark blue/purple, flashes blinding white at peak collapse
             base_blue = ti.Vector([0.1, 0.3, 0.9])
             base_red = ti.Vector([0.9, 0.1, 0.3])
             
             # Mix based on angle and depth to get swirling colors
             mix_val = (ti.sin(angle * 3.0 + time) + 1.0) * 0.5
             color = base_blue * mix_val + base_red * (1.0 - mix_val)
             
             # Add the high frequency "slit" texture
             slit = ti.abs(ti.sin(z * 50.0))
             color += color * slit * 2.0
             
             # Fade to white based on progress
             # Peak intensity at progress = 0.5
             intensity = 1.0 - ti.abs(progress - 0.5) * 2.0
             
             # Vignette
             vignette = ti.exp(-r * 2.0)
             
             final_color = color * vignette * (noise + 1.0)
             
             # Blinding flash at peak
             if intensity > 0.8:
                  flash = (intensity - 0.8) * 5.0
                  final_color += ti.Vector([1.0, 1.0, 1.0]) * flash
             
             self.pixels[i, j] = final_color

    def render_to_mp4(self, output_path, total_frames=60):
        print(f"Generating ZPHC Transition to {output_path}...")
        import cv2
        import numpy as np
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (self.res, self.res))
        
        for f in range(total_frames):
             progress = f / total_frames
             self.render_frame(f * 0.05, progress)
             
             img = self.pixels.to_numpy()
             img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
             # OpenCV expects BGR
             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
             out.write(img)
             
        out.release()
        print(f"Saved {output_path}")

def generate_zphc_transition(output_name, test_mode=False):
    """
    Generates a ZPHC slit-scan transition video between encounters.
    """
    ensure_taichi()
    print("Generating ZPHC transition...")
    frames = 15 if test_mode else 60
    res = 512 if test_mode else 1024
    
    renderer = ZPHCTransitionRenderer(res=res)
    full_path = f"{output_name}.mp4"
    renderer.render_to_mp4(full_path, total_frames=frames)
    
    return full_path

def stitch_sequence(video_list, output_file):
    """Uses ffmpeg to stitch MP4s together."""
    if not video_list:
        print("No videos to stitch!")
        return
        
    print(f"Stitching {len(video_list)} segments into {output_file}...")
    
    list_file = "concat_list.txt"
    with open(list_file, "w") as f:
        for vid in video_list:
             if vid and os.path.exists(vid):
                  f.write(f"file '{vid}'\n")
                  
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", output_file]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.remove(list_file)
    print(f"Finished stitching {output_file}")


class InfiniteDirector:
    def __init__(self, output_dir="director_output", seed=1977):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.segments = []
        self.iteration = 0
        # Fixes which worlds this director visits: the same seed tours
        # the same sector in the same order.
        self.seed = int(seed)

    def run_from_manifest(self, manifest_path, output_name="sequence.mp4"):
        """Render a ShotGraph defined by a JSON manifest.

        This is the modern path: each shot is rendered by a registered
        Renderer, transitions are produced by the continuity engine, and
        the result is a single tone-mapped MP4 with no hard cuts. The
        legacy `run_infinite_loop` below stitches finished MP4 segments
        instead, and is kept for the sister-project surface pipelines
        that produce their own video.
        """
        import render.adapters  # noqa: F401  ensures renderers are registered
        from scene.manifest import load_manifest
        from director.graph import render_graph

        graph = load_manifest(manifest_path)
        out_path = os.path.join(self.output_dir, output_name)
        return render_graph(graph, out_path)
            
    def run_infinite_loop(self, max_iterations=3, test_mode=False):
        print("=== INITIATING INFINITE DIRECTOR SEQUENCE ===")
        while self.iteration < max_iterations:
             print(f"\n--- Sequence Iteration {self.iteration} ---")
             
             # 1. Astrometric Waypoint
             import random
             waypoint_seed = (self.seed * 7919 + self.iteration) & 0x7FFFFFFF
             sector, world = get_astrometric_waypoint(seed=waypoint_seed)
             rng = random.Random(waypoint_seed ^ 0xA710)

             # Every so often, detour to something the Traveller tables
             # have no code for at all.
             is_exotic = rng.random() < 0.3
             encounter_type = "planetary"
             if is_exotic:
                 encounter_type = rng.choice([
                     "blackhole",
                     "hypernova",
                     "megastructure_alien",
                     "chthonic_horror"
                 ])
             else:
                 print(f"Waypoint: {world.summary()}")

             safe_name = (world.name.replace(' ', '_') if world and not is_exotic
                          else encounter_type)
             segment_name = f"seg_{self.iteration}_{safe_name}"
             vid_path = None
             
             # 2. Pick Encounter type (Planetary vs Exotic)
             if is_exotic:
                 vid_path = generate_exotic_segment(encounter_type, segment_name, test_mode=test_mode)
             else:
                 vid_path = generate_planetary_segment(world, segment_name, test_mode=test_mode)
             
             if vid_path and os.path.exists(vid_path):
                  # Move to output dir
                  final_vid = os.path.join(self.output_dir, f"{segment_name}.mp4")
                  try:
                      shutil.move(vid_path, final_vid)
                  except shutil.Error:
                      # File is already there or same path
                      pass
                  self.segments.append(final_vid)
                  
                  # Generate Transition
                  trans_path = os.path.join(self.output_dir, f"trans_{self.iteration}")
                  trans_vid = generate_zphc_transition(trans_path, test_mode=test_mode)
                  if trans_vid and os.path.exists(trans_vid):
                       self.segments.append(trans_vid)
                  
             self.iteration += 1
             
        # Stitch
        stitch_sequence(self.segments, os.path.join(self.output_dir, "final_sequence.mp4"))
        print("=== SEQUENCE COMPLETE ===")


if __name__ == "__main__":
    director = InfiniteDirector()
    director.run_infinite_loop(max_iterations=1, test_mode=True)
