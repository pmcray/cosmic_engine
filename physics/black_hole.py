import taichi as ti
import math

@ti.data_oriented
class BlackHole:
    """
    Models a Black Hole across a range of mass scales.
    Implements gravitational lensing, accretion disk dynamics,
    and mass-dependent Hawking radiation / Relativistic effects.
    """
    def __init__(self):
        # We don't store mass here, we pass it to the functions
        pass

    @ti.func
    def get_disk_color(self, r, time, mass, disk_inner, disk_outer):
        """
        Returns the color of the accretion disk at a given radius.
        Scales from blue (hot/small mass) to red (cool/large mass).
        """
        normalized_r = (r - disk_inner) / (disk_outer - disk_inner + 1e-6)
        
        # Temperature gradient across the disk (hotter inside)
        temp = (1.0 - normalized_r) * (2.0 / (mass**0.5 + 0.5))
        
        color = ti.Vector([0.0, 0.0, 0.0])
        
        # Base color based on mass (Small = Blue, Large = Red)
        if mass < 0.5: # Micro Black Hole
            color = ti.Vector([0.5, 0.8, 1.0]) * temp * 2.0
        elif mass < 5.0: # Stellar Mass
            color = ti.Vector([1.0, 0.6, 0.2]) * temp
        else: # Supermassive
            color = ti.Vector([1.0, 0.2, 0.05]) * temp * 0.5
            
        # Add some turbulence/flicker
        noise = ti.sin(r * 10.0 - time * 5.0 / (mass + 0.1)) * 0.2 + 0.8
        return color * noise

    @ti.func
    def render(self, ro, rd, time, mass):
        """
        Integrates the light path in the curved spacetime with the original
        march budget (100 steps of 0.1 -- adequate for close-in framing).
        """
        return self.render_n(ro, rd, time, mass, 100, 0.1)

    @ti.func
    def render_n(self, ro, rd, time, mass, max_steps: ti.i32, dt: float):
        """
        As render(), but with a caller-chosen march budget. Wide framings
        (camera outside the accretion disk) need the longer path length.
        """
        rs = 2.0 * mass
        r_isco = 3.0 * rs
        disk_inner = r_isco
        disk_outer = r_isco * 4.0
        hawking_temp = 1.0 / (mass + 1e-6)

        curr_p = ro
        curr_v = rd

        total_color = ti.Vector([0.0, 0.0, 0.0])
        transmittance = 1.0

        # Far-field bound must lie beyond the camera itself, or rays from a
        # wide framing would 'escape' on their first step
        r0 = ti.sqrt(ro[0]**2 + ro[1]**2 + ro[2]**2)
        escape_r = ti.max(15.0 * mass + 5.0, r0 * 1.05)

        for i in range(max_steps):
            r2 = curr_p[0]**2 + curr_p[1]**2 + curr_p[2]**2
            r = ti.sqrt(r2)

            if r < rs * 1.01:
                # Crossed the Event Horizon - perfectly black
                transmittance = 0.0
                break

            if r > escape_r:
                # Escaped to far field
                break
                
            # 1. Gravitational Deflection
            # The force is proportional to M/r^3 for light bending
            force_mag = (1.5 * rs) / (r2 * r + 1e-6)
            accel = -curr_p * force_mag
            
            # Update velocity and position (Symplectic Euler-like)
            prev_p = curr_p
            curr_v += accel * dt
            curr_v /= ti.sqrt(curr_v[0]**2 + curr_v[1]**2 + curr_v[2]**2) # Maintain speed of light
            curr_p += curr_v * dt

            # 2. Accretion Disk Intersection (thin plane at y=0). Detected
            # by the sign change of y across the step -- a fixed |y| window
            # misses crossings at coarse step sizes, producing stair-step
            # artifacts on the disk edge.
            if prev_p[1] * curr_p[1] < 0.0:
                # Interpolate to the actual plane crossing
                frac = prev_p[1] / (prev_p[1] - curr_p[1] + 1e-9)
                hit_p = prev_p + (curr_p - prev_p) * frac
                dist_xz = ti.sqrt(hit_p[0]**2 + hit_p[2]**2)
                if disk_inner < dist_xz < disk_outer:
                    disk_col = self.get_disk_color(dist_xz, time, mass, disk_inner, disk_outer)

                    # Relativistic Beaming (simplified)
                    # One side is brighter because it moves towards the observer
                    # Assuming counter-clockwise rotation
                    velocity_dot_view = -hit_p[2] / (dist_xz + 1e-6) # Z-component of tangential velocity
                    beaming = 1.0 + velocity_dot_view * 0.8

                    total_color += disk_col * beaming * transmittance
                    transmittance *= 0.1 # Disk is somewhat opaque
            
            if transmittance < 0.01:
                break
                
        # 3. Hawking Radiation (Micro BHs only)
        if mass < 0.1:
            hawking = ti.Vector([0.8, 0.9, 1.0]) * hawking_temp * 0.01
            # Add some "sparkle"
            sparkle = ti.random() * ti.exp(- (ti.sqrt(ro[0]**2 + ro[1]**2 + ro[2]**2) - rs)**2 * 10.0)
            total_color += hawking * sparkle
            
        return total_color
