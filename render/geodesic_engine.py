import taichi as ti
import math

@ti.data_oriented
class GeodesicEngine:
    """
    Handles rendering via the Pi-Metric (pi-M) curvature operator,
    treating light paths as resonant trajectories along the pi-lattice rather
    than Euclidean rays. Implements Zero-Point Harmonic Collapse (ZPHC).
    """
    def __init__(self, render_res=1024, samples=4):
        self.render_res = render_res
        self.samples = samples
        self.final_output = ti.Vector.field(3, dtype=float, shape=(self.render_res, self.render_res))
        
        # ZPHC Observer state (0.0 = unobserved proxy, 1.0 = fully collapsed reality)
        self.observer_attention = ti.field(dtype=float, shape=())
        self.observer_attention[None] = 0.0

    @ti.kernel
    def update_observer_attention(self, attention_level: float):
        """
        Triggered dynamically by the external AGI Operator agent.
        """
        self.observer_attention[None] = ti.max(0.0, ti.min(1.0, attention_level))

    @ti.func
    def get_pi_lattice_noise(self, p: ti.template()) -> float:
        """
        A highly efficient, probabilistic wave-skeleton used when the region is unobserved.
        Represents the uncollapsed quantum potential.
        """
        # A simple, fast hash noise function to act as the Pi-Lattice skeleton
        value = ti.sin(p[0] * math.pi + p[1] * math.pi * 1.618 + p[2] * math.pi * 2.718)
        return value

    @ti.func
    def trace_geodesic(self, u, v, time: float):
        """
        The core ray-tracing replacement. Instead of straight rays, this follows
        geodesics curved by "informational gravity wells" (Forman Ricci curvature).
        """
        # Starting point (Observer/Camera position)
        ro = ti.Vector([0.0, 0.0, -5.0])
        # Initial trajectory vector (un-curved)
        rd = ti.Vector([u, v, 1.5])
        rd /= ti.sqrt(rd[0]**2 + rd[1]**2 + rd[2]**2)

        # Basic sphere representing an exotic object (e.g. Fuzzball or Gravastar)
        center = ti.Vector([0.0, 0.0, 0.0])
        radius = 1.0
        
        # Intersection test
        oc = ro - center
        b = oc[0]*rd[0] + oc[1]*rd[1] + oc[2]*rd[2]
        c = (oc[0]**2 + oc[1]**2 + oc[2]**2) - radius**2
        h = b**2 - c
        
        color = ti.Vector([0.05, 0.05, 0.1]) # Deep space background minimum
        
        if h > 0.0:
            t = -b - ti.sqrt(h)
            if t > 0.0:
                p = ro + rd * t
                n = p / ti.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
                
                # Zero-Point Harmonic Collapse (ZPHC) blend
                attention = self.observer_attention[None]
                
                if attention < 0.1:
                    # UNOBSERVED: Render only the hyper-efficient Pi-Lattice wave skeleton
                    # Saves massive compute by bypassing complex geometric shaders
                    lattice_val = self.get_pi_lattice_noise(p * 20.0 + time)
                    w_color = ti.Vector([0.1, 0.8, 0.3]) * ti.abs(lattice_val)
                    # The wave collapses slightly in the presence of intense gravity (the object's edge)
                    color = w_color * 0.5
                else:
                    # OBSERVED / COLLAPSING: Render the full Noun/Geometric reality
                    # (In a full implementation, this branches into the Gravastar/Fuzzball shaders)
                    
                    # Simulated Pi-Metric lensing (light bending around the informational well)
                    # We distort the normal based on harmonic alignment (represented by a sine wave here for basic prototype)
                    harmonic_distortion = ti.sin(p[0]*10.0 + time) * ti.cos(p[1]*10.0) * 0.3
                    warped_n = n + ti.Vector([harmonic_distortion, harmonic_distortion, 0.0])
                    warped_n /= ti.sqrt(warped_n[0]**2 + warped_n[1]**2 + warped_n[2]**2)
                    
                    light_dir = ti.Vector([0.707, 0.707, 0.0])
                    diffuse = ti.max(0.0, warped_n[0]*light_dir[0] + warped_n[1]*light_dir[1] + warped_n[2]*light_dir[2])
                    
                    # Full geometric color
                    full_color = ti.Vector([0.9, 0.2, 0.5]) * diffuse
                    
                    # Transition smoothly based on the AGI operator's attention level
                    lattice_val = self.get_pi_lattice_noise(p * 20.0 + time)
                    wave_color = ti.Vector([0.1, 0.8, 0.3]) * ti.abs(lattice_val)
                    
                    color = wave_color * (1.0 - attention) + full_color * attention

        return color

    @ti.kernel
    def render_frame(self, time: float):
        for i, j in self.final_output:
            accumulated_color = ti.Vector([0.0, 0.0, 0.0])
            
            for k in range(self.samples):
                offset_x = ti.random() - 0.5
                offset_y = ti.random() - 0.5
                
                # Standardize UV coordinates from -1.0 to 1.0
                u = (float(i) + offset_x) / self.render_res * 2.0 - 1.0
                v = (float(j) + offset_y) / self.render_res * 2.0 - 1.0
                
                accumulated_color += self.trace_geodesic(u, v, time)
                
            self.final_output[i, j] = accumulated_color / float(self.samples)
