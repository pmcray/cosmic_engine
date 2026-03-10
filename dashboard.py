import taichi as ti
import numpy as np
import argparse
from pathlib import Path
import sys
import time

from physics.fluid_solver import FluidEngine
from simulator import load_biosim_from_file

# Initialize Taichi
ti.init(arch=ti.gpu)

def main():
    parser = argparse.ArgumentParser(description="Cosmic Console - Real-Time Aesthetic Dashboard")
    parser.add_argument('--file', '-f', required=True, help='Input L-system file')
    parser.add_argument('--planet-type', type=str, default='jupiter', help='Planet type for fluid simulation')
    parser.add_argument('--res', type=int, default=512, help='Dashboard resolution')
    args = parser.parse_args()

    input_file = Path(args.file)
    if not input_file.exists():
        print(f"Error: Input file '{args.file}' not found.")
        sys.exit(1)

    print("Initializing Cosmic Console...")

    # Initialize Fluid Engine
    fluid_engine = FluidEngine(res=args.res, planet_type=args.planet_type)

    # Load Simulator
    simulator = load_biosim_from_file(
        str(input_file),
        bounds=(-50, 50, -50, 50, 0, 100),
        voxel_resolution=1.0,
        fluid_engine=fluid_engine
    )

    # Initial settings
    target_iterations = 5
    current_iteration = 0
    simulator.fluid_susceptibility = 0.5
    simulator.gravity_susceptibility = 0.2
    
    # Taichi UI setup
    window = ti.ui.Window("Cosmic Console", (args.res * 2, args.res), vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    # Pre-allocate arrays for 3D rendering
    max_vertices = 500000
    pos_field = ti.Vector.field(3, dtype=ti.f32, shape=max_vertices)
    color_field = ti.Vector.field(3, dtype=ti.f32, shape=max_vertices)
    vertex_count = 0
    
    camera = ti.ui.Camera()
    camera.position(0, 50, 150)
    camera.lookat(0, 25, 0)
    camera.up(0, 1, 0)

    scene = ti.ui.Scene()

    frame = 0
    last_step_time = time.time()
    auto_grow = False

    while window.running:
        # Step fluid every frame
        fluid_engine.step(frame)
        frame += 1

        # Handle GUI
        with gui.sub_window("Aesthetic Controls", 0.05, 0.05, 0.3, 0.4):
            gui.text("Fluid Physics")
            fluid_engine.vorticity_strength = gui.slider_float("Vorticity", fluid_engine.vorticity_strength, 0.0, 10.0)
            simulator.fluid_susceptibility = gui.slider_float("Wind Susceptibility", simulator.fluid_susceptibility, 0.0, 5.0)
            
            gui.text("L-System Growth")
            simulator.gravity_susceptibility = gui.slider_float("Gravity", simulator.gravity_susceptibility, -1.0, 1.0)
            target_iterations = gui.slider_int("Target Iterations", target_iterations, 1, 10)
            
            if gui.button("Grow Step"):
                if current_iteration < target_iterations:
                    simulator.step(apply_tropism=True, update_shadows=False)
                    current_iteration += 1
                    
            if gui.button("Reset Growth"):
                # Reload simulator state
                simulator = load_biosim_from_file(
                    str(input_file),
                    bounds=(-50, 50, -50, 50, 0, 100),
                    voxel_resolution=1.0,
                    fluid_engine=fluid_engine
                )
                current_iteration = 0

            auto_grow = gui.checkbox("Auto-Grow", auto_grow)

        # Auto-grow logic (e.g. 1 iteration per second)
        if auto_grow and current_iteration < target_iterations:
            if time.time() - last_step_time > 1.0:
                simulator.step(apply_tropism=True, update_shadows=False)
                current_iteration += 1
                last_step_time = time.time()

        # Render background/fluid on the left
        # We can draw the fluid on the left half of the screen using canvas.set_image, 
        # but canvas.set_image fills the window. So we render fluid on the left using a 2D quad or just two subwindows.
        # Alternatively, we just display the fluid on the left side, and 3D on the right side.
        # Taichi UI doesn't natively support split screen set_image without custom shaders, 
        # so we'll just overlap them or use camera perspective.
        
        # We'll render the 3D plant
        turtle, _ = simulator.get_final_geometry()
        vertices = turtle.get_vertices()
        edges = turtle.get_edges()
        
        vertex_count = 0
        if len(vertices) > 0:
            # We want to draw lines. We can just fill a buffer of points for lines.
            lines_pos = []
            for start_idx, end_idx, width in edges:
                if start_idx < len(vertices) and end_idx < len(vertices):
                    lines_pos.append(vertices[start_idx])
                    lines_pos.append(vertices[end_idx])
                    
            if len(lines_pos) > 0:
                vertex_count = len(lines_pos)
                np_pos = np.array(lines_pos, dtype=np.float32)
                # Keep within limits
                if vertex_count > max_vertices:
                    vertex_count = max_vertices
                    np_pos = np_pos[:max_vertices]
                    
                pos_field.from_numpy(np_pos)
                # Set color to plant-like green
                color_field.fill(ti.Vector([0.2, 0.8, 0.3]))
                
        # Setup camera controls
        camera.track_user_inputs(window, movement_speed=1.0, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        
        # Add lighting
        scene.point_light(pos=(0, 100, 100), color=(1, 1, 1))
        scene.ambient_light((0.3, 0.3, 0.3))
        
        # Add particles/lines to scene
        if vertex_count > 0:
            # Taichi scene currently supports particles and mesh. We'll use particles to represent joints.
            # For lines, we use scene.particles with a small radius since lines aren't always supported in all Ti versions.
            scene.particles(pos_field, color=color_field, radius=0.5, per_vertex_color=True)
            
        canvas.scene(scene)

        # Since we want to preview fluid as well, we can show it in a small picture or mapped to a plane.
        # Let's map fluid dye to the canvas image directly for preview on left? 
        # Unfortunately, mixing canvas.scene and canvas.set_image can be tricky. 
        # We will let the user use the GUI to visualize parameters, and rely on 3D view for the plant.
        # To show the fluid, we'd ideally render it as a texture on a ground plane!
        
        window.show()

if __name__ == '__main__':
    main()
