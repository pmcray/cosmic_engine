import taichi as ti
import math

@ti.data_oriented
class TopologicalDefects:
    """
    Handles rendering and physics for Cosmic Strings (1D) and Domain Walls (2D).
    Crucial for the abstract, mind-bending visuals of the Stargate sequence.
    """
    def __init__(self, string_tension=0.01):
        self.string_tension = string_tension # Energy per unit length (mu)

    @ti.func
    def render_domain_wall(self, ro, rd, time):
        """
        Domain Walls of Light create massive 2D planar light-sheets resulting
        from symmetry breaking in the early universe. Approaching them
        causes severe, fractal space-time rippling.
        """
        # Define a massive plane at Z = 50.0 (The Stargate threshold)
        wall_z = 50.0
        
        # Simple plane intersection
        color = ti.Vector([0.0, 0.0, 0.0])
        
        if rd[2] > 1e-4:
            t_wall = (wall_z - ro[2]) / rd[2]
            
            if t_wall > 0.0:
                 p_wall = ro + rd * t_wall
                 
                 # The domain wall surface isn't flat; it's a seething, recursive harmonic sea
                 # (Implementing the 'Stargate' ripple effect documented in the research report)
                 
                 # High-frequency cellular automaton-like noise pattern
                 ripple_1 = ti.sin(p_wall[0]*0.5 + time*5.0) * ti.cos(p_wall[1]*0.6 - time*4.0)
                 ripple_2 = ti.sin(p_wall[0]*0.1 - time*2.0 + ripple_1*3.0) * ti.cos(p_wall[1]*0.1 + time*3.0 - ripple_1*3.0)
                 
                 fractal_ripple = ti.abs(ripple_2)**0.5
                 
                 # The 'Domain Wall of Light' is blindingly bright white/blue
                 base_color = ti.Vector([0.8, 0.9, 1.0])
                 # With iridescent, oil-slick like interference patterns from the symmetry breaking
                 iridescent = ti.Vector([ti.sin(fractal_ripple*10.0)*0.5+0.5, ti.sin(fractal_ripple*11.0 + 2.0)*0.5+0.5, ti.sin(fractal_ripple*12.0 + 4.0)*0.5+0.5])
                 
                 wall_color = base_color * (0.5 + fractal_ripple) + iridescent * 0.5
                 
                 # Depth fog (the wall shines through the universe)
                 fog = ti.exp(-t_wall * 0.01)
                 color = wall_color * fog
                 
        return color

    @ti.func
    def calculate_conical_deficit(self, p, string_axis: ti.Vector):
        """
        Cosmic strings do not curve space conventionally; they cut a "wedge"
        out of it, creating a conical deficit angle. Passing near them
        causes double-imaging (lensing) without magnification.
        """
        # Simplified pseudo-lensing calculation 
        # Calculate distance to the string axis
        string_dir = string_axis / ti.sqrt(string_axis[0]**2 + string_axis[1]**2 + string_axis[2]**2)
        proj = p[0]*string_dir[0] + p[1]*string_dir[1] + p[2]*string_dir[2]
        closest_point = string_dir * proj
        dist_vec = p - closest_point
        radial_dist = ti.sqrt(dist_vec[0]**2 + dist_vec[1]**2 + dist_vec[2]**2)
        
        # The deflection angle depends directly on the string tension (energy density)
        # Deflection = 8 * pi * G * mu (where mu is string_tension)
        deflection_angle = 8.0 * math.pi * self.string_tension
        
        # Return a simple scalar indicating the severity of the deficit at this point
        return deflection_angle / ti.max(radial_dist, 0.001)

    @ti.func
    def apply_string_lensing(self, ray_dir: ti.Vector, p_intersect, string_axis: ti.Vector):
         """
         Bends a ray crossing near a cosmic string, creating the signature
         double-image effect without magnification.
         """
         deflection = self.calculate_conical_deficit(p_intersect, string_axis)
         
         # Bend the ray sharply perpendicularly to the string towards its axis
         string_dir = string_axis / ti.sqrt(string_axis[0]**2 + string_axis[1]**2 + string_axis[2]**2)
         proj = p_intersect[0]*string_dir[0] + p_intersect[1]*string_dir[1] + p_intersect[2]*string_dir[2]
         closest_point = string_dir * proj
         
         # Direction towards the string
         dir_to_string = closest_point - p_intersect
         dir_len = ti.sqrt(dir_to_string[0]**2 + dir_to_string[1]**2 + dir_to_string[2]**2)
         
         if dir_len > 1e-4:
             dir_to_string /= dir_len
             # Blend the original ray direction with the deflection
             bent_ray = ray_dir + dir_to_string * ti.min(deflection, 0.5)
             bent_ray /= ti.sqrt(bent_ray[0]**2 + bent_ray[1]**2 + bent_ray[2]**2)
             return bent_ray
             
         return ray_dir
