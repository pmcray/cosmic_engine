import taichi as ti
import numpy as np
import time
import math
from render.camera import SphereCamera

ti.init(arch=ti.gpu)

def run_demo():
    res = 1024
    camera = SphereCamera(render_res=res, samples=4)
    gui = ti.GUI('Black Hole Simulation: Mass Variance', (res, res))
    
    frame = 0
    while gui.running:
        # Vary mass over time to show range of black holes
        # Cycles between Micro (0.05), Stellar (1.0), and Supermassive (5.0)
        cycle_time = time.time() * 0.2
        mass = 1.0 + math.sin(cycle_time) * 1.5 
        if mass < 0.1: mass = 0.05 # Micro BH limit
        
        # Update engine parameters
        camera.geo_engine.update_black_hole_mass(mass)
        
        # Slowly rotate camera
        camera.cam_pan = time.time() * 0.1
        camera.cam_tilt = -0.3 + math.sin(time.time() * 0.05) * 0.2
        
        # Render
        camera.render_black_hole(time.time())
        
        # Output
        img = camera.get_image_data()
        gui.set_image(img)
        
        # Telemetry
        gui.text(f"BH Mass: {mass:.2f} Solar Masses", (0.05, 0.95), color=0xffffff)
        if mass < 0.2:
            gui.text("Type: Micro (Hawking Radiation Active)", (0.05, 0.92), color=0xaaaaff)
        elif mass < 3.0:
            gui.text("Type: Stellar Mass", (0.05, 0.92), color=0xffaa00)
        else:
            gui.text("Type: Supermassive", (0.05, 0.92), color=0xff5555)
            
        gui.show()
        frame += 1

if __name__ == "__main__":
    run_demo()
