import taichi as ti

@ti.data_oriented
class Gravastar:
    """
    Models a Gravastar, a theoretical alternative to a black hole.
    Consists of an ultra-dense shell of 'stiff matter' encompassing
    an interior core governed by de Sitter space (repulsive dark energy).
    """
    def __init__(self, mass=1.0, J=0.1):
        self.M = mass
        self.J = J # Spin
        
        # Dimensions based on general theoretical constraints
        # The stiff matter shell sits just slightly larger than the Schwarzschild radius
        self.R_outer_shell = 2.0 * self.M + 0.05 
        self.R_inner_core = 2.0 * self.M - 0.05 
        
        # I-Love-Q parameter approximations (Moment of Inertia, Love number, Quadrupole moment)
        # Gravastars differ significantly from Black Holes in their tidal deformability (Love number)
        self.tidal_love_number = 0.05 # Non-zero compared to a Kerr Black Hole which has k2 = 0

    @ti.func
    def get_matter_pressure(self, r):
        """
        Calculates the equation of state (pressure vs density) at a given radius.
        """
        pressure = 0.0
        
        if r > self.R_outer_shell:
            # Exterior Vacuum (Schwarzschild/Kerr metric)
            pressure = 0.0
            
        elif r > self.R_inner_core:
            # Stiff Matter Shell: Pressure equals density (p = rho), speed of sound = speed of light
            # Highly incompressible, rigid state.
            shell_thickness = self.R_outer_shell - self.R_inner_core
            normalized_depth = (r - self.R_inner_core) / shell_thickness
            
            # Simplified parabolic density profile across the shell
            density = 1.0 - 4.0 * (normalized_depth - 0.5)**2
            pressure = density * 100.0 # Extreme stiff pressure
            
        else:
            # Interior de Sitter Core: Negative pressure (repulsive dark energy)
            # p = -rho. This prevents the singularity from forming.
            pressure = -50.0 
            
        return pressure

    @ti.func
    def calculate_tidal_deformation(self, external_gravitational_gradient: ti.math.vec3):
        """
        Calculates how the stiff matter shell bulges or warps in the presence
        of another massive object (like during a merger or binary orbit),
        using the non-zero Love Number.
        """
        # A simple linear response model based on the tidal Love number
        # Return the resulting quadrupole moment/deformation vector
        quadrupole = external_gravitational_gradient * self.tidal_love_number * (self.R_outer_shell**5)
        return quadrupole

    @ti.func
    def render_exterior(self, ro, rd, time):
        """
        Ray-marches the extremely luminous, rigid outer stiff-matter shell.
        Because there is no horizon, incoming matter impacts the shell and
        releases phenomenal amounts of radiation before thermalizing.
        """
        # Basic intersection test against the outer shell
        oc = ro
        b = oc[0]*rd[0] + oc[1]*rd[1] + oc[2]*rd[2]
        c = (oc[0]**2 + oc[1]**2 + oc[2]**2) - self.R_outer_shell**2
        h = b**2 - c
        
        color = ti.Vector([0.0, 0.0, 0.0])
        
        if h > 0.0:
            t = -b - ti.sqrt(h)
            if t > 0.0:
                p = ro + rd * t
                n = p / ti.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
                
                # The stiff matter shell is incredibly hot, emitting X-Rays and Gamma rays
                # We simulate an intense, rigid, crystalline glowing effect
                crystal_pattern = ti.abs(ti.cos(p[0]*20.0) * ti.sin(p[1]*20.0) * ti.cos(p[2]*20.0))
                
                # Accretion impact flashes (simulating matter striking the hard surface)
                impact_flash = ti.max(0.0, ti.sin(p[0]*5.0 + time*15.0) * ti.sin(p[2]*7.0 - time*10.0))
                impact_flash = impact_flash**10.0 # Sharp, explosive spots
                
                base_glow = ti.Vector([0.9, 0.95, 1.0]) * crystal_pattern * 0.8
                flare_color = ti.Vector([1.0, 0.4, 0.1]) * impact_flash * 5.0
                
                # The shell itself is largely opaque due to extreme density
                color = base_glow + flare_color

        return color
