import taichi as ti
import numpy as np
import sys
import os

# Ensure cosmic_engine root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.nexus_core import NexusCore
from agi_operator import AGIOperator
from render.geodesic_engine import GeodesicEngine

@ti.data_oriented
class ExoticPhysicsEncounter:
    def __init__(self, encounter_type="blackhole", res=1024, samples=4):
        self.res = res
        self.encounter_type = encounter_type
        
        self.nexus = NexusCore(res=512)
        self.operator = AGIOperator()
        
        # GeodesicEngine uses Pi-Metric for gravitational lensing rendering
        self.geo_engine = GeodesicEngine(render_res=res, samples=samples)
        
        # Pixels buffer for output
        self.pixels = ti.Vector.field(3, dtype=float, shape=(res, res))
        
        # Configuration for Black Hole or Hypernova
        self.mass = ti.field(dtype=float, shape=())
        self.mass[None] = 2.0 if encounter_type == "blackhole" else 3.5
        
        # For Neutron Star Merger (Hypernova) we have a separation distance that shrinks 
        self.separation = ti.field(dtype=float, shape=())
        self.separation[None] = 1.0 if encounter_type == "hypernova" else 0.0
        
        self.frame = 0

    @ti.kernel
    def render_blackhole_frame(self, time: float):
        for i, j in self.pixels:
             # Basic Geodesic Tracing modified for standard Schwarzschild/Kerr lensing
             u = (float(i) / self.res) * 2.0 - 1.0
             v = (float(j) / self.res) * 2.0 - 1.0
             
             # Instead of full Geodesic Engine which is complex, we render an accretion disk 
             # and gravitational shadow
             ro = ti.Vector([0.0, 1.0, -8.0]) # Camera position
             rd = ti.Vector([u, v, 1.5]) 
             rd /= ti.sqrt(rd[0]**2 + rd[1]**2 + rd[2]**2)
             
             color = ti.Vector([0.0, 0.0, 0.0])
             
             # Black Hole parameters
             M = self.mass[None]
             rs = 2.0 * M
             
             # Raymarch to find closest approach
             t = 0.0
             min_dist = 1000.0
             hit_disk = False
             disk_color = ti.Vector([0.0, 0.0, 0.0])
             
             # Simple bending proxy
             p = ro
             v_dir = rd
             
             for step in range(100):
                 r2 = p[0]**2 + p[1]**2 + p[2]**2
                 r = ti.sqrt(r2)
                 min_dist = ti.min(min_dist, r)
                 
                 if r < rs: # Fell into event horizon
                     color = ti.Vector([0.0, 0.0, 0.0])
                     break
                 
                 # Gravity bends the light vector towards center
                 # dv = -GM/r^2 * dt
                 grav_force = -M / r2
                 v_dir += p * (grav_force / r) * 0.1
                 v_dir /= ti.sqrt(v_dir[0]**2 + v_dir[1]**2 + v_dir[2]**2)
                 
                 # Accretion Disk intersection (plane y=0)
                 if p[1] * (p[1] + v_dir[1] * 0.1) < 0.0:
                      disk_r = ti.sqrt(p[0]**2 + p[2]**2)
                      if disk_r > rs * 1.5 and disk_r < rs * 5.0:
                           # Accretion disk glow (Doppler shifted)
                           # velocity is roughly in x,z plane
                           vel_x = -p[2] / disk_r
                           doppler = 1.0 + (v_dir[0] * vel_x) * 0.5
                           
                           temp = 1.0 / (disk_r - rs)
                           disk_color += ti.Vector([1.0, 0.5, 0.2]) * temp * doppler**4 * 0.5
                           hit_disk = True
                 
                 p += v_dir * 0.1
                 
             if min_dist > rs and not hit_disk:
                 # Background stars/nebula distorted by the bent ray
                 bg_star = ti.sin(v_dir[0] * 100.0) * ti.sin(v_dir[1] * 100.0)
                 if bg_star > 0.98:
                      color += ti.Vector([1.0, 1.0, 1.0])
                 # Add some cosmic microwave background / galactic glow
                 color += ti.Vector([0.05, 0.08, 0.15]) * (1.0 - ti.abs(v_dir[1]))
                 
             if hit_disk:
                 color += disk_color
                 
             # AGI Observer Attention Collapse (ZPHC)
             attention = self.geo_engine.observer_attention[None]
             if attention > 0.0:
                  # When observed, Pi-Metric lattice imposes itself
                  lattice_v = ti.abs(ti.sin(p[0]*10.0 + time) * ti.cos(p[1]*10.0))
                  color = color * (1.0 - attention) + ti.Vector([0.2, 1.0, 0.5]) * lattice_v * attention
             
             self.pixels[i, j] = color

    @ti.kernel
    def render_hypernova_frame(self, time: float):
        for i, j in self.pixels:
             u = (float(i) / self.res) * 2.0 - 1.0
             v = (float(j) / self.res) * 2.0 - 1.0
             
             ro = ti.Vector([0.0, 2.0, -10.0])
             rd = ti.Vector([u, v, 1.5]) 
             rd /= ti.sqrt(rd[0]**2 + rd[1]**2 + rd[2]**2)
             
             sep = self.separation[None]
             ns1_pos = ti.Vector([-sep, 0.0, 0.0])
             ns2_pos = ti.Vector([sep, 0.0, 0.0])
             
             p = ro
             color = ti.Vector([0.05, 0.05, 0.05])
             
             # Volumetric raymarching for massive spiraling ejecta
             density_accum = 0.0
             for step in range(60):
                  dist1 = ti.sqrt((p[0] - ns1_pos[0])**2 + p[1]**2 + (p[2] - ns1_pos[2])**2)
                  dist2 = ti.sqrt((p[0] - ns2_pos[0])**2 + p[1]**2 + (p[2] - ns2_pos[2])**2)
                  
                  # Intense spiraling magnetic fields and crust ejecta
                  # Model as a twisting noise function around the barycenter
                  r_bary = ti.sqrt(p[0]**2 + p[2]**2)
                  angle = ti.atan2(p[2], p[0])
                  
                  spiral = ti.sin(angle * 2.0 - r_bary * 1.5 + time * 5.0)
                  vol_shell = ti.exp(-ti.abs(r_bary - 3.0)) # Ejecta ring
                  
                  ejecta_density = ti.max(0.0, spiral * vol_shell * 0.5)
                  
                  if dist1 < 0.3 or dist2 < 0.3:
                       # Hit a neutron star surface (extremely hot and bright)
                       color += ti.Vector([0.8, 0.9, 1.0]) * 5.0
                       break
                       
                  density_accum += ejecta_density * 0.1
                  p += rd * 0.2
             
             # Add ejecta glow (blueshifted/redshifted via rotation)
             color += ti.Vector([0.2, 0.5, 1.0]) * density_accum * 2.0
             
             # Polar jets
             jet_d = ti.sqrt(p[0]**2 + p[2]**2)
             if jet_d < 0.5:
                  jet_glow = ti.exp(-jet_d * 5.0) * (1.0 / (0.1 + ti.abs(p[1])))
                  color += ti.Vector([0.8, 0.2, 1.0]) * jet_glow * (1.0 - sep)*2.0 # Jets intensify as they merge
             
             self.pixels[i, j] = color


    def step(self):
        sim_time = self.frame * 0.016
        
        # AGI Observer Attention
        current_attention = self.operator.update()
        self.geo_engine.update_observer_attention(current_attention)
        self.nexus.step_time_stroboscopic()
        
        if self.encounter_type == "blackhole":
             self.render_blackhole_frame(sim_time)
        elif self.encounter_type == "hypernova":
             # Inspiraling happens over time
             self.separation[None] = max(0.0, 2.0 - self.frame * 0.01)
             self.render_hypernova_frame(sim_time)
             
        self.frame += 1
        
    def render_to_mp4(self, output_path, total_frames=120):
        print(f"Rendering {self.encounter_type} encounter to {output_path}...")
        gui = ti.GUI('Exotic Render', (self.res, self.res), show_gui=False)
        video_manager = ti.tools.VideoManager(output_dir="./vids", framerate=30, automatic_build=False)
        
        for f in range(total_frames):
             self.step()
             img = self.pixels.to_numpy()
             # Clip and convert to 8-bit RGB
             img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
             video_manager.write_frame(img)
             if f % 10 == 0:
                  print(f"  Frame {f}/{total_frames}")
                  
        video_manager.make_video(gif=False, mp4=True)
        # Rename the output to desired path
        import os
        import glob
        mp4_files = glob.glob("./vids/*.mp4")
        if mp4_files:
             latest_file = max(mp4_files, key=os.path.getctime)
             os.rename(latest_file, output_path)
             print(f"Saved {output_path}")

if __name__ == "__main__":
    ti.init(arch=ti.gpu)
    enc = ExoticPhysicsEncounter("blackhole", res=512)
    enc.render_to_mp4("blackhole_encounter.mp4", total_frames=60)
    
    enc2 = ExoticPhysicsEncounter("hypernova", res=512)
    enc2.render_to_mp4("hypernova_encounter.mp4", total_frames=60)
