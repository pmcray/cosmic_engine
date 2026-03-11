import os
import taichi as ti
import numpy as np
import subprocess
import glob

@ti.data_oriented
class MegastructureEncounter:
    def __init__(self, obj_style="alien_structure", res=512):
        self.res = res
        self.obj_style = obj_style
        self.lsystem_file = f"examples/{obj_style}.l"
        self.frame = 0

    def render_to_mp4(self, output_path, total_frames=60, test_mode=False):
        print(f"Generating {self.obj_style} megastructure sequence to {output_path}...")
        
        # We will generate a growing L-system by iterating the depth
        import tempfile
        import shutil
        import cv2
        
        tmp_dir = tempfile.mkdtemp(prefix="mega_")
        frame_files = []
        
        # In test mode, we drastically limit rendering
        actual_frames = 5 if test_mode else total_frames
        
        for f in range(actual_frames):
             iteration = int(1 + (f / actual_frames) * 2) if test_mode else int(2 + (f / total_frames) * 5)
             
             obj_path = os.path.join(tmp_dir, f"mega_{f}.obj")
             
             print(f"  Frame {f}/{actual_frames} (L-System Iteration {iteration})...")
             
             # Step 1: Generate OBJ via lsystem.py
             cmd_l = [
                 "python", "lsystem.py",
                 "--file", self.lsystem_file,
                 "--iter", str(iteration),
                 "--out", obj_path
             ]
             subprocess.run(cmd_l, capture_output=True, cwd="/home/pmc/cosmic_engine")
             
             # Step 2: Render via factory.py (Map generation + AI diffusion)
             # Use maps only for speed if we don't have SDXL weights loaded, 
             # but we assume the prompt calls for ultraphotorealism:
             style_preset = "alien_creature" if "alien" in self.obj_style else "prehistoric"
             
             cmd_f = [
                 "python", "factory.py",
                 "--input", obj_path,
                 "--maps-only",
                 "--angles", "iso",
                 "--output", tmp_dir,
                 "--prefix", f"frame_{f}"
             ]
             
             if test_mode:
                 cmd_f.extend(["--map-resolution", "256"]) # extremely low res map for speed
                 
             subprocess.run(cmd_f, capture_output=True, cwd="/home/pmc/cosmic_engine")
             
             # Find the generated rendered image (e.g. depth or normal map since maps_only)
             rendered = os.path.join(tmp_dir, f"frame_{f}_iso_depth.png")
             if not os.path.exists(rendered):
                  # Fallback to normal map
                  rendered = os.path.join(tmp_dir, f"frame_{f}_iso_normal.png")
             
             if os.path.exists(rendered):
                  frame_files.append(rendered)
                  
        # Step 3: Stitch the frames into an MP4
        if frame_files:
             print("  Stitching frames via OpenCV...")
             # Read first frame to get size
             first = cv2.imread(frame_files[0])
             h, w, layers = first.shape
             
             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
             out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))
             
             for img_path in frame_files:
                  frame = cv2.imread(img_path)
                  out.write(frame)
                  
             out.release()
             print(f"Saved {output_path}")
        else:
             print("Error: No frames were generated!")
             
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    enc = MegastructureEncounter("alien_structure", res=256)
    enc.render_to_mp4("megastructure.mp4", test_mode=True)
