import taichi as ti
import math

@ti.data_oriented
class Fuzzball:
    """
    Models a horizonless microstate string geometry.
    Unlike standard black holes, Fuzzballs lack a true event horizon or singularity,
    replaced instead by a chaotic 'fuzz' of vibrating strings.
    """
    def __init__(self, mass=1.0, J=0.5):
        self.M = mass
        self.J = J # Angular momentum (spin)
        # The Fuzzball radius roughly corresponds to the Schwarzschild radius
        self.R_fuzz = 2.0 * self.M 

    @ti.func
    def get_string_density(self, p, time):
        """
        Calculates the local density of string microstates.
        This governs both the optical scattering (rendering) and the local
        informational curvature (physics).
        """
        r = ti.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
        
        # Outside the fuzzball boundary, density rapidly drops to 0
        if r > self.R_fuzz * 1.5:
             return 0.0
             
        # Normalize distance relative to the fuzzball core
        normalized_r = r / self.R_fuzz
        
        # High-frequency, chaotic "stringy" noise
        # This creates the signature 'fuzz' rather than a hard black shell
        string_noise = ti.sin(p[0] * 50.0 + time * 10.0) * ti.cos(p[1] * 70.0 - time * 5.0) * ti.sin(p[2] * 60.0)
        
        # Density peaks at the surface (r=R_fuzz) and remains chaotic inside
        density = ti.exp(-10.0 * (normalized_r - 1.0)**2) * (0.5 + 0.5 * string_noise)
        
        return ti.max(0.0, ti.min(1.0, density))

    @ti.func
    def render_photon_ring(self, ro, rd, time):
        """
        Fuzzballs exhibit chaotic photon ring scattering. Light does not just
        orbit; it interacts with the vibrating surface microstates.
        """
        # (This is a simplified ray-march for the photon interacton layer)
        t = 0.0
        color = ti.Vector([0.0, 0.0, 0.0])
        transmittance = 1.0
        
        for i in range(50):
            p = ro + rd * t
            r = ti.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
            
            if r > 5.0 * self.M: 
                # Escaped outer bounds
                break
                
            if r < 0.1:
                # Approaching the dense center
                break

            density = self.get_string_density(p, time)
            
            if density > 0.01:
                # Intense, chaotic Hawking-like radiation emission where strings scatter
                emission_color = ti.Vector([0.8, 0.2, 1.0]) * density * 5.0 
                
                # Attenuate the remaining light passing through the string mesh
                absorption = density * 2.0
                transmittance *= ti.exp(-absorption * 0.1) # step size approx 0.1
                
                color += emission_color * transmittance * 0.1
                
            # Adaptive step size: smaller when close to the fuzzball surface
            step_size = ti.max(0.02, (r - self.R_fuzz) * 0.5)
            t += step_size
            
            if transmittance < 0.01:
                break
                
        return color
