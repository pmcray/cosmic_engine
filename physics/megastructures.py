import taichi as ti
import math

@ti.data_oriented
class Megastructures:
    """
    Simulates the physics and rendering of vast stellar engineering projects.
    """
    def __init__(self, star_mass=1.0, ds_radius=1.0):
        self.M = star_mass
        self.R = ds_radius # Radius of the primary Dyson Swarm
        
        # Landauer Limit (Theoretical minimum energy to erase 1 bit of information)
        # E = k_B * T * ln(2)
        # Used to calculate the maximum computational capacity of the Matrioshka Brain
        self.KB = 1.380649e-23
        self.T_core = 5000.0 # Effective temperature of the innermost shell (Kelvin)

    @ti.func
    def render_dyson_swarm(self, ro, rd, time):
        """
        Renders a Matrioshka Brain—a nested series of Dyson Swarms.
        Instead of rendering billions of individual satellites, we render them
        as probabilistic density shells, simulating the occlusion of the central star.
        """
        # Central star (simplified sphere)
        b = ro[0]*rd[0] + ro[1]*rd[1] + ro[2]*rd[2]
        c = (ro[0]**2 + ro[1]**2 + ro[2]**2) - (self.R * 0.1)**2 # Star is 10% the radius of the inner swarm
        h = b**2 - c
        
        star_hit = False
        t_star = 1e10
        if h > 0.0:
            t = -b - ti.sqrt(h)
            if t > 0.0:
                t_star = t
                star_hit = True

        # Raymarch through the nested shells
        t = 0.0
        transmittance = 1.0
        color = ti.Vector([0.0, 0.0, 0.0])
        
        # We model 3 nested shells of the Matrioshka Brain
        # Inner: Hot, extracts primary energy
        # Middle: Computes using waste heat of inner
        # Outer: Cold, radiates waste heat into deep space
        shell_radii = ti.Vector([self.R, self.R * 2.0, self.R * 4.0])
        shell_temps = ti.Vector([5000.0, 1000.0, 100.0])
        
        for i in range(50):
            p = ro + rd * t
            r = ti.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
            
            if r > self.R * 5.0 or (star_hit and t > t_star):
                 break
                 
            # Check intersection/proximity with each shell
            for shell_idx in ti.static(range(3)):
                dist_to_shell = ti.abs(r - shell_radii[shell_idx])
                
                if dist_to_shell < 0.1:
                    # We are passing through a swarm layer.
                    # The density creates an occlusion fractal (the 'megastructure' look)
                    # We use high-frequency geometric noise to simulate the billions of habitats/nodes
                    swarm_pattern = ti.abs(ti.sin(p[0]*50.0) * ti.cos(p[1]*50.0) * ti.sin(p[2]*50.0))
                    
                    # Density of the swarm (0.0 to 1.0)
                    density = ti.max(0.0, 0.8 - swarm_pattern)
                    
                    # Compute thermodynamic color based on the shell's temperature
                    # Simplified blackbody approximation
                    temp_ratio = shell_temps[shell_idx] / 5000.0
                    emission = ti.Vector([1.0, 0.8 * temp_ratio, 0.4 * (temp_ratio**2)]) * density * temp_ratio
                    
                    absorption = density * 2.0
                     
                    transmittance *= ti.exp(-absorption * 0.1) # approx step size
                    color += emission * transmittance * 0.1
                    
            t += 0.1 # Fixed step size for the thin shells
            if transmittance < 0.01:
                break
                
        # Add the central star if we hit it and have transmittance left
        if star_hit and transmittance > 0.01:
            star_color = ti.Vector([5.0, 4.8, 4.0]) # Extreme white-hot center
            color += star_color * transmittance
            
        return color

    @ti.func
    def get_shkadov_thrust(self):
        """
        A Shkadov Thruster is a massive half-shell mirror placed over a star.
        It reflects stellar radiation asymmetrically to move the entire star system.
        Returns the net acceleration vector.
        """
        # Assuming the mirror covers exactly one hemisphere
        # Radiation pressure force = (Luminosity / c)
        c = 299792458.0
        luminosity = 3.828e26 * (self.M ** 3.5) # Mass-Luminosity relation (approximate)
        
        force = luminosity / c
        # The star's mass dominates the system
        mass_kg = 1.989e30 * self.M
        
        # Acceleration = F / m
        acceleration = force / mass_kg
        
        # Return as a tiny vector along the Z axis representing the thrust
        return ti.Vector([0.0, 0.0, acceleration])
