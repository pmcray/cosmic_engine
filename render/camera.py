import taichi as ti
import math
import numpy as np

@ti.data_oriented
class SphereCamera:
    def __init__(self, fluid_res=512, render_res=1024, has_rings=0, samples=4):
        self.fluid_res = fluid_res
        self.render_res = render_res
        self.has_rings = has_rings
        self.samples = samples # Controls Anti-Aliasing quality
        
        # Camera transform parameters
        self.cam_tilt = -0.45
        self.cam_pan = 0.0
        self.cam_roll = 0.0
        
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
    def get_cloud_height(self, field: ti.template(), u, v):
        # We use the luminance of the fluid dye as a height map
        color = self.bilinear_interp(field, ti.Vector([u * self.fluid_res, v * self.fluid_res]))
        # Approximate luminance
        return color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114

    @ti.func
    def perturbed_normal(self, field: ti.template(), tex_u, tex_v, n):
        # Sample surrounding points to calculate gradient
        eps = 1.0 / self.fluid_res
        
        # We need a tangent space to perturb the normal along the surface
        # A simple way is to find two vectors perpendicular to n
        up = ti.Vector([0.0, 1.0, 0.0])
        if ti.abs(n[1]) > 0.999:
            up = ti.Vector([1.0, 0.0, 0.0])
            
        tangent = ti.math.cross(up, n)
        tangent /= ti.sqrt(tangent[0]**2 + tangent[1]**2 + tangent[2]**2)
        bitangent = ti.math.cross(n, tangent)
        
        h_center = self.get_cloud_height(field, tex_u, tex_v)
        h_right = self.get_cloud_height(field, tex_u + eps, tex_v)
        h_up = self.get_cloud_height(field, tex_u, tex_v + eps)
        
        # Gradient in texture space (Reduced from 50.0 to 12.0 for softer, more realistic clouds)
        du = (h_right - h_center) * 12.0 
        dv = (h_up - h_center) * 12.0
        
        # Perturb normal
        new_n = n - (tangent * du) - (bitangent * dv)
        new_n /= ti.sqrt(new_n[0]**2 + new_n[1]**2 + new_n[2]**2)
        
        # Mix the perturbed normal with the original to control the strength
        bump_strength = 0.85 # Softened bump strength
        final_n = n * (1.0 - bump_strength) + new_n * bump_strength
        final_n /= ti.sqrt(final_n[0]**2 + final_n[1]**2 + final_n[2]**2)
        
        return final_n

    @ti.func
    def fbm_noise(self, p):
        value = 0.0
        amp = 0.5
        freq = 1.0
        # 4 octaves of pseudo-random noise
        for _ in range(4):
            # Simple hash-based noise approximation for GPU
            qx = p[0] * freq
            qy = p[1] * freq
            qz = p[2] * freq
            n = ti.sin(qx*12.9898 + qy*78.233 + qz*37.719) * 43758.5453
            n = n - ti.floor(n) # fractional part
            
            value += n * amp
            amp *= 0.5
            freq *= 2.0
            
        return value

    @ti.func
    def rot_x(self, v, angle):
        c = ti.cos(angle)
        s = ti.sin(angle)
        return ti.Vector([v[0], v[1]*c - v[2]*s, v[1]*s + v[2]*c])

    @ti.func
    def rot_y(self, v, angle):
        c = ti.cos(angle)
        s = ti.sin(angle)
        return ti.Vector([v[0]*c + v[2]*s, v[1], -v[0]*s + v[2]*c])
        
    @ti.func
    def rot_z(self, v, angle):
        c = ti.cos(angle)
        s = ti.sin(angle)
        return ti.Vector([v[0]*c - v[1]*s, v[0]*s + v[1]*c, v[2]])

    @ti.func
    def sample_ring(self, dist):
        """Generates soft, continuous ring density and color."""
        n_dist = (dist - 1.2) / 1.2 
        
        # Smooth overlapping noise waves including a high-frequency grit
        density = ti.sin(n_dist * 40.0) * ti.cos(n_dist * 15.0) + ti.sin(n_dist * 150.0) * 0.4 + ti.sin(n_dist * 500.0) * 0.1
        
        # Carve out a soft-edged Cassini Division
        cassini = 1.0
        if 0.53 < n_dist < 0.65:
            cassini = ti.abs(n_dist - 0.59) * 20.0 
            cassini = ti.max(0.0, ti.min(1.0, cassini))
            
        # Continuous alpha map (no hard cutoffs)
        alpha = ti.max(0.0, ti.min(1.0, (density + 0.5) * 0.9)) * cassini
        
        # Smooth color blending based on density
        base_color1 = ti.Vector([0.85, 0.80, 0.70]) # Bright reflective ice
        base_color2 = ti.Vector([0.45, 0.40, 0.35]) # Darker, sparse dust
        mix_factor = ti.max(0.0, ti.min(1.0, density + 0.5))
        
        color = base_color2 * (1.0 - mix_factor) + base_color1 * mix_factor
        return alpha, color

    @ti.func
    def cast_ray(self, u, v, light_dir, render_buffer: ti.template()):
        ro = ti.Vector([0.0, 0.0, -4.0])
        rd = ti.Vector([u, v, 1.5]) 
        rd /= ti.sqrt(rd[0]**2 + rd[1]**2 + rd[2]**2)
        
        # Apply camera rotations (roll, tilt, pan)
        rd = self.rot_z(rd, self.cam_roll)
        
        ro = self.rot_x(ro, self.cam_tilt)
        rd = self.rot_x(rd, self.cam_tilt)
        
        ro = self.rot_y(ro, self.cam_pan)
        rd = self.rot_y(rd, self.cam_pan)
        
        hit_sphere = False
        t_sphere = 1e10
        
        # 1. Sphere Intersection
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
        
        # 2. Ring Plane Intersection
        if self.has_rings == 1 and ti.abs(rd[1]) > 1e-5:
            t_cand = -ro[1] / rd[1]
            if t_cand > 0.0:
                p_cand = ro + rd * t_cand
                dist = ti.sqrt(p_cand[0]**2 + p_cand[2]**2)
                if 1.2 < dist < 2.4: 
                    t_ring = t_cand
                    hit_ring = True

        color = ti.Vector([0.0, 0.0, 0.0])
        
        # 3. Compositing, Translucency, and Scattering
        if hit_sphere or hit_ring:
            sphere_color = ti.Vector([0.0, 0.0, 0.0])
            ring_color = ti.Vector([0.0, 0.0, 0.0])
            ring_alpha = 0.0
            
            if hit_sphere:
                p = ro + rd * t_sphere
                n = p / ti.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
                
                phi = ti.math.atan2(n[2], n[0])
                theta = ti.math.asin(n[1])
                tex_u = (phi / (2.0 * math.pi)) + 0.5
                tex_v = (theta / math.pi) + 0.5
                
                # 3D Cloud Relief (Perturb Normal)
                n = self.perturbed_normal(render_buffer, tex_u, tex_v, n)
                
                fluid_color = self.bilinear_interp(render_buffer, ti.Vector([tex_u * self.fluid_res, tex_v * self.fluid_res]))
                
                # Sub-grid Procedural Detail (fBM)
                # We distort the noise coordinates using the underlying fluid color to make the noise follow the flow
                flow_distortion = ti.Vector([fluid_color[0] - 0.5, fluid_color[1] - 0.5, fluid_color[2] - 0.5]) * 4.0
                noise_val = self.fbm_noise(p * 18.0 + flow_distortion)
                
                # Blend the fractal wispiness into the base color
                # White clouds get whiter/crisper, dark bands get deeper
                wisp_factor = (noise_val * 2.0 - 1.0) * 0.15 
                fluid_color = ti.Vector([
                    ti.max(0.0, ti.min(1.0, fluid_color[0] + wisp_factor)),
                    ti.max(0.0, ti.min(1.0, fluid_color[1] + wisp_factor)),
                    ti.max(0.0, ti.min(1.0, fluid_color[2] + wisp_factor))
                ])
                
                # Soft Translucent Shadows casting ON the planet
                shadow = 1.0
                if self.has_rings == 1 and ti.abs(light_dir[1]) > 1e-5:
                    t_shad = -p[1] / light_dir[1]
                    if t_shad > 0.0:
                        p_shad = p + light_dir * t_shad
                        d_shad = ti.sqrt(p_shad[0]**2 + p_shad[2]**2)
                        if 1.2 < d_shad < 2.4:
                            r_alpha, _ = self.sample_ring(d_shad)
                            shadow = 1.0 - (r_alpha * 0.85) # Allows partial light through
                            
                # Enhanced Self-Shadowing for clouds
                # The normal is perturbed, creating stark shadows near the terminator
                diffuse = ti.max(0.0, n[0]*light_dir[0] + n[1]*light_dir[1] + n[2]*light_dir[2]) * shadow
                
                # Rim lighting and volumetric aerosol scattering
                view_dir = ti.Vector([-rd[0], -rd[1], -rd[2]])
                ndotv = ti.max(0.0, n[0]*view_dir[0] + n[1]*view_dir[1] + n[2]*view_dir[2])
                rim = (1.0 - ndotv)**3.0 * 0.6
                
                # Rayleigh Atmospheric Halo (Limb Glow)
                # Adds a glowing rim even in shadow at extreme angles
                limb_glow = (1.0 - ndotv)**5.0 * 0.5
                limb_color = ti.Vector([0.3, 0.5, 0.9]) * limb_glow
                
                rim_color = ti.Vector([1.0, 0.9, 0.8]) * rim * diffuse
                
                # Volumetric aerosol scattering
                half_vec = light_dir + view_dir
                half_length = ti.sqrt(half_vec[0]**2 + half_vec[1]**2 + half_vec[2]**2) + 1e-5
                half_vec /= half_length
                ndoth = ti.max(0.0, n[0]*half_vec[0] + n[1]*half_vec[1] + n[2]*half_vec[2])
                aerosol_scatter = (1.0 - ndotv) * (ndoth**8.0) * 0.8
                aerosol_color = ti.Vector([0.95, 0.85, 0.75]) * aerosol_scatter * shadow

                sphere_color = fluid_color * diffuse + rim_color + aerosol_color + limb_color
                
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
                        planet_shadow = 0.05 # Deep umbra
                        
                # Mie Scattering Approximation for icy dust
                scatter = ti.max(0.0, rd[0]*light_dir[0] + rd[1]*light_dir[1] + rd[2]*light_dir[2])
                mie = 1.0 + (scatter**4) * 2.0
                
                light_intensity = ti.max(0.15, ti.abs(light_dir[1])) * mie
                ring_color = base_r_color * light_intensity * planet_shadow

            # Z-Buffer Depth Blending
            if hit_ring and hit_sphere:
                if t_ring < t_sphere:
                    color = ring_color * ring_alpha + sphere_color * (1.0 - ring_alpha) 
                else:
                    color = sphere_color 
            elif hit_ring:
                color = ring_color * ring_alpha
            elif hit_sphere:
                color = sphere_color
                
        return color

    @ti.kernel
    def render_gas_giant(self, render_buffer: ti.template(), lx: float, ly: float, lz: float):
        light_dir = ti.Vector([lx, ly, lz])
        light_dir /= ti.sqrt(light_dir[0]**2 + light_dir[1]**2 + light_dir[2]**2)
        
        for i, j in self.final_output:
            accumulated_color = ti.Vector([0.0, 0.0, 0.0])
            
            # Sub-pixel Jittering for Anti-Aliasing
            for k in range(self.samples):
                offset_x = ti.random() - 0.5
                offset_y = ti.random() - 0.5
                u = (float(i) + offset_x) / self.render_res * 2.0 - 1.0
                v = (float(j) + offset_y) / self.render_res * 2.0 - 1.0
                
                accumulated_color += self.cast_ray(u, v, light_dir, render_buffer)
                
            final_color = accumulated_color / float(self.samples)
            
            # ACES Tone Mapping
            a = 2.51
            b = 0.03
            c = 2.43
            d = 0.59
            e = 0.14
            color = (final_color * (a * final_color + b)) / (final_color * (c * final_color + d) + e)
            
            # Spacecraft Sensor Noise
            noise = (ti.random() - 0.5) * 0.04
            
            self.final_output[i, j] = color + ti.Vector([noise, noise, noise])

    def get_image_data(self):
        return np.clip(self.final_output.to_numpy(), 0.0, 1.0)
