import os
import sys
import subprocess
import time
import shutil
import taichi as ti

# Ensure Taichi is initialized globally before any other classes try to map memory!
ti.init(arch=ti.gpu)

# Add project paths to import from other repos
sys.path.append('/home/pmc/worldmaker')
sys.path.append('/home/pmc/weorold')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import traveller_world_generator as worldmaker

def get_astrometric_waypoint():
    """Uses worldmaker to generate a new star system and select a planet/encounter."""
    system = worldmaker.generate_full_system()
    
    # Look for interesting worlds or just pick the first terrestrial
    chosen_world = None
    worlds = system.all_worlds
    if worlds:
        for w in worlds:
             if w.body_type == 'Terrestrial':
                 chosen_world = w
                 break
        if not chosen_world:
             chosen_world = worlds[0]
            
    return system, chosen_world

def generate_planetary_segment(world, output_name, test_mode=False):
    """Uses weorold to generate a planetary sequence."""
    from wp12_full_pipeline import run_full_pipeline
    
    mean_temp_k = world.mean_temperature if world.mean_temperature > 0 else 288.0
    
    hydro_code = 7
    atm_code = 6
    if hasattr(world, 'hydrographics_code') and world.hydrographics_code:
        try:
             hydro_code = int(world.hydrographics_code, 16)
        except ValueError:
             pass
    if hasattr(world, 'atmosphere_code') and world.atmosphere_code:
        try:
             atm_code = int(world.atmosphere_code, 16)
        except ValueError:
             pass
             
    print(f"Generating planetary view for {world.name or 'Unknown'} (Temp={mean_temp_k}K, Atm={atm_code}, Hyd={hydro_code})")
    
    # Change dir to weorold so outputs land there, or run from root
    cwd = os.getcwd()
    os.chdir('/home/pmc/weorold')
    
    # Needs a dummy sketch input if we want a new map, or we just rely on defaults for now
    run_full_pipeline(output_name=output_name, mean_temp_k=mean_temp_k, hydro_code=hydro_code, atm_code=atm_code, test_mode=test_mode)
    
    # The output MP4 should be at f"{output_name}_orbit.mp4"
    result_path = f"/home/pmc/weorold/{output_name}_orbit.mp4"
    
    os.chdir(cwd)
    return result_path

def generate_exotic_segment(encounter_type, output_name, test_mode=False):
    """
    Hooks into cosmic_engine to render exotic encounters (L-systems or Geodesic Blackholes).
    """
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
    def __init__(self, output_dir="director_output"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.segments = []
        self.iteration = 0
            
    def run_infinite_loop(self, max_iterations=3, test_mode=False):
        print("=== INITIATING INFINITE DIRECTOR SEQUENCE ===")
        while self.iteration < max_iterations:
             print(f"\n--- Sequence Iteration {self.iteration} ---")
             
             # 1. Astrometric Waypoint
             system, world = get_astrometric_waypoint()
             # Check if system has anomalies we can use for exotic encounters Let's override world
             # picking sometimes to show off the exotic ones!
             is_exotic = False
             encounter_type = "planetary"
             
             if system.anomalous_planets and len(system.anomalous_planets) > 0:
                 is_exotic = True
                 # Pick a random exotic type we have implemented
                 import random
                 encounter_type = random.choice([
                     "blackhole", 
                     "hypernova", 
                     "megastructure_alien", 
                     "chthonic_horror"
                 ])
             
             segment_name = f"seg_{self.iteration}_{world.name.replace(' ', '_') if world else 'exotic'}"
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
