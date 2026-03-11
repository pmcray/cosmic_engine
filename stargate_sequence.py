import taichi as ti
import numpy as np
import time
import math
import time
from render.camera import SphereCamera
from physics.fluid_solver import FluidEngine
from physics.nexus_core import NexusCore
from agi_operator import AGIOperator

ti.init(arch=ti.gpu)

class StargateDirector:
    """
    Orchestrates the grand sequence inspired by 2001: A Space Odyssey.
    Transitions from the fluid dynamics of a mundane gas giant into the
    mind-bending, ZPHC-driven abstract geometry of the Stargate.
    """
    def __init__(self, res=1024):
        self.res = res
        print("Initializing the Convergence Architecture...")
        
        # Core mathematical and physical modules
        self.nexus = NexusCore(res=512)
        self.operator = AGIOperator()
        
        # The visual components
        self.fluid = FluidEngine(res=512)
        self.camera = SphereCamera(fluid_res=512, render_res=res, has_rings=0, samples=4)
        
        # Frame and timing logic
        self.frame = 0
        self.max_frames = 1000 # The length of the cinematic sequence
        
        # Set up the initial Jovian atmosphere
        print("Igniting Jovian Fluid Dynamics...")
        for pre_frame in range(100):
            # Pre-warm the fluid sim to get interesting storms
            self.fluid.step(pre_frame)

    def update_cinematic_camera(self, progress: float):
        """
        Drives the camera through the sequence smoothly.
        progress ranges from 0.0 (start) to 1.0 (end).
        """
        # Phase 1 (0.0 - 0.3): Slowly panning across the Jovian surface
        if progress < 0.3:
            local_p = progress / 0.3
            self.camera.cam_pan = local_p * 2.0
            self.camera.cam_tilt = -0.45 + (local_p * 0.1)
            
            # The AGI Operator is passive, just watching the planet
            self.operator.target_attention = 0.05
            
        # Phase 2 (0.3 - 0.6): The Anomaly. Pushing in tight on a massive fractal storm.
        elif progress < 0.6:
            local_p = (progress - 0.3) / 0.3
            # The AGI Operator notices something deep in the informational structure
            self.operator.process_telemetry(0.85) # Triggers "investigating"
            
            # The camera rolls and the light dims as we enter the dense clouds
            self.camera.cam_roll = local_p * 0.5
            
        # Phase 3 (0.6 - 1.0): The Stargate. ZPHC collapse into the Domain Wall of Light.
        else:
            local_p = (progress - 0.6) / 0.4
            # Full AGI interrogating the anomaly, triggering ZPHC collapse
            self.operator.process_telemetry(1.0) # Triggers "locked"
            
            # The camera wildly distorts, representing the breakdown of standard spacetime
            self.camera.cam_roll += 0.05
            self.camera.cam_tilt = -0.3 + ti.sin(local_p * 20.0) * 0.5

    def run_sequence(self):
        print("\n--- INITIATING STARGATE SEQUENCE ---")
        
        gui = ti.GUI('Cosmic Engine: 2001 Stargate Sequence', (self.res, self.res))
        
        while gui.running and self.frame < self.max_frames:
            # 1. Update Core Logic and Time
            sim_time = self.frame * 0.016
            progress = self.frame / float(self.max_frames)
            
            self.nexus.step_time_stroboscopic()
            
            # 2. Update the AGI Operator's zero-point harmonic attention
            current_attention = self.operator.update()
            self.camera.geo_engine.update_observer_attention(current_attention)
            
            # 3. Step the planetary fluid dynamics
            self.fluid.step(self.frame)
            
            # 4. Cinematic Director updates camera and lighting
            self.update_cinematic_camera(progress)
            
            # Simulated orbit taking the light source behind the planet
            lx = math.sin(self.frame * 0.02)
            lz = math.cos(self.frame * 0.02)
            ly = 0.5 - progress # The sun sinks lower
            
            # 5. Render the frame
            # The camera automatically handles the ZPHC transition based on the AGI attention
            self.camera.render_gas_giant(self.fluid.dye, lx, ly, lz, sim_time)
            
            # 6. Output to screen
            img = self.camera.get_image_data()
            gui.set_image(img)
            gui.show()
            
            # Print periodic telemetry
            if self.frame % 50 == 0:
                status = self.operator.get_agent_status()
                print(f"Frame [{self.frame}/{self.max_frames}] | "
                      f"Seq Progress: {progress*100:.1f}% | "
                      f"Operator: {status['state'].upper()} | "
                      f"ZPHC Focus: {status['zphc_attention_level']:.2f}")
                
            self.frame += 1

        print("--- SEQUENCE COMPLETE ---")
        gui.close()

if __name__ == "__main__":
    director = StargateDirector()
    director.run_sequence()
