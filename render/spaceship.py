import taichi as ti
import math
import random

# A simpler, non-Taichi Python class handles the grammar generation,
# while the results are passed into the Taichi rendering kernel.

class SpacecraftLSystem:
    """
    Generates the rigid-body structure of the Discovery One spacecraft
    using a 3D Lindenmayer System to achieve a "mechanorganic" aesthetic.
    """
    def __init__(self, iterations=4):
        self.iterations = iterations
        # Start with the main spine
        self.axiom = "F"
        
        # Rules:
        # F: Move forward, drawing a cylinder (spine segment)
        # S: Draw a sphere (habitat/reactor module)
        # +: Pitch up 
        # -: Pitch down
        # ^: Yaw left
        # v: Yaw right
        # [: Push state (branch)
        # ]: Pop state
        
        # We want a long spine with clustered modules at the ends,
        # resembling the iconic bone/spaceship motif
        self.rules = {
            "F": "F[+S][-S]FF", # The spine grows and attaches modules
            "S": "S" # Modules remain modules
        }

    def generate_geometry(self):
         current = self.axiom
         for _ in range(self.iterations):
             next_seq = ""
             for char in current:
                 next_seq += self.rules.get(char, char)
             current = next_seq
             
         print(f"L-System Sequence Generated (Length: {len(current)})")
         return current

@ti.data_oriented
class SpacecraftRenderer:
    def __init__(self, l_system_sequence: str):
        # In a full implementation, we would parse the string locally here
        # or pre-compile it into a Taichi field of structs/AABB bounding boxes.
        # For prototype rendering inside the raymarcher, we simulate the SDF.
        self.seq = l_system_sequence

    @ti.func
    def get_sd_capsule(self, p, a: ti.Vector, b: ti.Vector, r: float):
        """Signed Distance Field for a capsule (used for the ship's spine)"""
        p_a = p - a
        b_a = b - a
        
        ba_len_sq = b_a[0]**2 + b_a[1]**2 + b_a[2]**2
        if ba_len_sq < 1e-5: return ti.sqrt(p_a[0]**2 + p_a[1]**2 + p_a[2]**2) - r
            
        h = ti.max(0.0, ti.min(1.0, (p_a[0]*b_a[0] + p_a[1]*b_a[1] + p_a[2]*b_a[2]) / ba_len_sq))
        # Vector rejection
        rejection = p_a - b_a * h
        return ti.sqrt(rejection[0]**2 + rejection[1]**2 + rejection[2]**2) - r

    @ti.func
    def get_sd_sphere(self, p, center: ti.Vector, r: float):
        """Signed Distance Field for a sphere (used for habitat modules)"""
        d = p - center
        return ti.sqrt(d[0]**2 + d[1]**2 + d[2]**2) - r

    @ti.func
    def evaluate_sdf(self, p):
        """
        Simulates the L-System geometry via a Distance Field for the raymarcher.
        Rather than parsing the string per-pixel (too slow), we define a procedural
        SDF that mimics the *results* of the L-System grammar.
        """
        # A long central spine along the Z axis
        spine = self.get_sd_capsule(p, ti.Vector([0.0, 0.0, -10.0]), ti.Vector([0.0, 0.0, 10.0]), 0.2)
        
        # The main spherical habitat section at the front (Z = -10.0)
        habitat = self.get_sd_sphere(p, ti.Vector([0.0, 0.0, -10.0]), 2.5)
        
        # The engine block/reactor array at the rear (Z = 10.0)
        # Using a clustered/modulo approach to simulate the L-System branching
        reactor_dist = 1e10
        for i in range(3):
            angle = float(i) * 2.0 * math.pi / 3.0
            r_center = ti.Vector([ti.cos(angle)*1.5, ti.sin(angle)*1.5, 9.5])
            reactor = self.get_sd_capsule(p, r_center, r_center + ti.Vector([0.0, 0.0, 2.0]), 0.6)
            reactor_dist = ti.min(reactor_dist, reactor)

        # Procedural "greebling" along the spine mimicking the F[+S][-S] rule
        # We use a modulo operation to place repeating antenna/cargo pods
        z_mod = (p[2] + 8.0) % 2.0 - 1.0 
        pod_center = ti.Vector([1.0, 0.0, p[2] - z_mod]) # Project onto spine side
        
        # Only place pods along the middle section
        pods = 1e10
        if -8.0 < p[2] < 8.0:
            pods = self.get_sd_sphere(p, pod_center, 0.3)
            # Add symmetry
            pod_center_2 = ti.Vector([-1.0, 0.0, p[2] - z_mod])
            pods = ti.min(pods, self.get_sd_sphere(p, pod_center_2, 0.3))

        # Smooth boolean union of all the parts (the 'mechanorganic' look)
        k = 0.5 # Smoothing factor
        
        # h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 )
        # return mix( d2, d1, h ) - k*h*(1.0-h)
        
        res = ti.min(spine, ti.min(habitat, ti.min(reactor_dist, pods)))
        return res

    @ti.func
    def get_normal(self, p):
        eps = 0.01
        dx = self.evaluate_sdf(p + ti.Vector([eps, 0.0, 0.0])) - self.evaluate_sdf(p - ti.Vector([eps, 0.0, 0.0]))
        dy = self.evaluate_sdf(p + ti.Vector([0.0, eps, 0.0])) - self.evaluate_sdf(p - ti.Vector([0.0, eps, 0.0]))
        dz = self.evaluate_sdf(p + ti.Vector([0.0, 0.0, eps])) - self.evaluate_sdf(p - ti.Vector([0.0, 0.0, eps]))
        
        n = ti.Vector([dx, dy, dz])
        return n / ti.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
