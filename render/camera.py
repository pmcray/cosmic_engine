import taichi as ti
import math
import numpy as np

@ti.data_oriented
class SphereCamera:
    def __init__(self, fluid_res=512, render_res=1024, has_rings=0):
        self.fluid_res = fluid_res
        self.render_res = render_res
        self.has_rings = has_rings
        self.final_output = ti.Vector.field(3, dtype=float, shape=(self.render_res, self.render_res))

    @ti.func
    def bilinear_interp(self, field: ti.template(), p):
        u = p[0] % float(self.fluid_res)
        v = p[1] % float(self.fluid_res)
        i, j = int(u), int(v)
        fx, fy = u - i, v - j
        i_next = (i + 1) % self.fluid_res
        j_next = (j + 1) % self.fluid_res
        
        c00 = field[i, j]
        c10 = field[i_next, j]
        c01 = field[i, j_next]
        c11 = field[i_next, j_next]
        
        return (c00 * (1.0 - fx) + c10 * fx) * (1.0 - fy) + \
               (c01 * (1.0 - fx) + c11 * fx) * fy

    @ti.func
    def rot_x(self, v, angle):
        """Rotates a 3D vector around the X-axis to tilt the camera."""
        c = ti.cos(angle)
        s = ti.sin(angle)
        return ti.Vector([v[0], v[1]*c - v[2]*s, v[1]*s + v[2]*c])

    @ti.func
    def sample_ring(self, dist):
        """Generates procedural Cassini divisions and dust bands."""
        alpha = 0.0
        color = ti.Vector([0.0, 0.0, 0.0])
        
        n_dist = (dist - 1.2) / 1.2 # Normalize between inner (1.2) and outer (2.4) radius
        
        # High and low frequency waves for varied ring density
        density = ti.sin(n_dist * 40.0) * ti.cos(n_dist * 15.0) + ti.sin(n_dist * 150.0) * 0.5
        
        # Carve out the Cassini Division
        if 0.55 < n_dist < 0.62:
            density -= 1.0 
            
        if density > 0.2:
            alpha = 0.85
            color = ti.Vector([0.85, 0.80, 0.70]) # Dense, reflective icy gold
        elif density > -0.2:
            alpha = 0.45
            color = ti.Vector([0.65, 0.55, 0.45]) # Medium dusty rock
        elif density > -0.5:
            alpha = 0.15
            color = ti.Vector([0.40, 0.35, 0.30]) # Faint scattered dust
            
        return alpha, color

    @ti.kernel
    def render_gas_giant(self, render_buffer: ti.template(), lx: float, ly: float, lz: float):
        light_dir = ti.Vector([lx, ly, lz])
        light_dir /= ti.sqrt(light_dir[0]**2 + light_dir[1]**2 + light_dir[2]**2)
        
        for i, j in self.final_output:
            u = (float(i) / self.render_res) * 2.0 - 1.0
            v = (float(j) / self.render_res) * 2.0 - 1.0
            
            # Base camera ray
            ro = ti.Vector([0.0, 0.0, -4.0])
            rd = ti.Vector([u, v, 1.5]) 
            rd /= ti.sqrt(rd[0]**2 + rd[1]**2 + rd[2]**2)
            
            # Tilt the camera down by ~25 degrees to see the rings clearly
            tilt = -0.45 
            ro = self.rot_x(ro, tilt)
            rd = self.rot_x(rd, tilt)
            
            hit_sphere = False
            t_sphere = 1e10
            
            # 1. Sphere Intersection Test
            b = ro[0]*rd[0] + ro[1]*rd[1] + ro[2]*rd[2]
            c = (ro[0]**2 + ro[1]**2 + ro[2]**2) - 1.0
            h = b**2 - c
            if h > 0.0:
                t_cand = -b - ti.sqrt(h)
                if t_cand > 0.0:
                    t_sphere = t_cand
                    hit_sphere = True

            hit_ring = False
            t_ring = 1e10
            
            # 2. Ring (Plane) Intersection Test
            if self.has_rings == 1 and ti.abs(rd[1]) > 1e-5:
                t_cand = -ro[1] / rd[1]
                if t_cand > 0.0:
                    p_cand = ro + rd * t_cand
                    dist = ti.sqrt(p_cand[0]**2 + p_cand[2]**2)
                    if 1.2 < dist < 2.4: # Inner and outer bounds of the ring system
                        t_ring = t_cand
                        hit_ring = True

            # 3. Compositing and Lighting
            color = ti.Vector([0.0, 0.0, 0.0])
            
            if hit_sphere or hit_ring:
                sphere_color = ti.Vector([0.0, 0.0, 0.0])
                ring_color = ti.Vector([0.0, 0.0, 0.0])
                ring_alpha = 0.0
                
                # Render Sphere
                if hit_sphere:
                    p = ro + rd * t_sphere
                    n = p / ti.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
                    
                    phi = ti.math.atan2(n[2], n[0])
                    theta = ti.math.asin(n[1])
                    tex_u = (phi / (2.0 * math.pi)) + 0.5
                    tex_v = (theta / math.pi) + 0.5
                    
                    fluid_color = self.bilinear_interp(render_buffer, ti.Vector([tex_u * self.fluid_res, tex_v * self.fluid_res]))
                    
                    # Ring Shadows casting ON the planet
                    shadow = 1.0
                    if self.has_rings == 1 and ti.abs(light_dir[1]) > 1e-5:
                        t_shad = -p[1] / light_dir[1]
                        if t_shad > 0.0:
                            p_shad = p + light_dir * t_shad
                            d_shad = ti.sqrt(p_shad[0]**2 + p_shad[2]**2)
                            if 1.2 < d_shad < 2.4:
                                r_alpha, _ = self.sample_ring(d_shad)
                                shadow = 1.0 - (r_alpha * 0.9) # 0.9 leaves the shadows slightly transparent
                                
                    diffuse = ti.max(0.0, n[0]*light_dir[0] + n[1]*light_dir[1] + n[2]*light_dir[2]) * shadow
                    rim = (1.0 - ti.max(0.0, -(rd[0]*n[0] + rd[1]*n[1] + rd[2]*n[2])))**4.0 * 0.3
                    sphere_color = fluid_color * diffuse + ti.Vector([rim, rim, rim]) * diffuse
                    
                # Render Rings
                if hit_ring:
                    p_ring = ro + rd * t_ring
                    dist = ti.sqrt(p_ring[0]**2 + p_ring[2]**2)
                    ring_alpha, base_r_color = self.sample_ring(dist)
                    
                    # Planet Shadow casting ON the rings
                    b_shad = p_ring[0]*light_dir[0] + p_ring[1]*light_dir[1] + p_ring[2]*light_dir[2]
                    c_shad = (p_ring[0]**2 + p_ring[1]**2 + p_ring[2]**2) - 1.0
                    h_shad = b_shad**2 - c_shad
                    
                    planet_shadow = 1.0
                    if h_shad > 0.0:
                        t1 = -b_shad - ti.sqrt(h_shad)
                        if t1 > 0.0:
                            planet_shadow = 0.02 # Almost pitch black in the planet's umbra
                            
                    # Sunlight catching the dust
                    light_intensity = ti.max(0.2, ti.abs(light_dir[1]))
                    ring_color = base_r_color * light_intensity * planet_shadow

                # Final Z-Buffer Depth Blend
                if hit_ring and hit_sphere:
                    if t_ring < t_sphere:
                        color = ring_color * ring_alpha + sphere_color * (1.0 - ring_alpha) # Rings in front
                    else:
                        color = sphere_color # Planet in front
                elif hit_ring:
                    color = ring_color * ring_alpha
                elif hit_sphere:
                    color = sphere_color
                    
            self.final_output[i, j] = color

    def get_image_data(self):
        return np.clip(self.final_output.to_numpy(), 0.0, 1.0)
