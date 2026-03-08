import taichi as ti
import math
import numpy as np

@ti.data_oriented
class SphereCamera:
    def __init__(self, fluid_res=512, render_res=1024):
        self.fluid_res = fluid_res
        self.render_res = render_res
        # The final, high-resolution output frame
        self.final_output = ti.Vector.field(3, dtype=float, shape=(self.render_res, self.render_res))

    @ti.func
    def bilinear_interp(self, field: ti.template(), p):
        """Smoothly blends pixels on the fluid grid to prevent blocky artifacts when mapping to 3D."""
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

    @ti.kernel
    def render_gas_giant(self, render_buffer: ti.template(), lx: float, ly: float, lz: float):
        # Normalize the sun's position vector
        light_dir = ti.Vector([lx, ly, lz])
        length_l = ti.sqrt(light_dir[0]**2 + light_dir[1]**2 + light_dir[2]**2)
        light_dir /= length_l
        
        for i, j in self.final_output:
            # Map screen coordinates from -1.0 to 1.0
            u = (float(i) / self.render_res) * 2.0 - 1.0
            v = (float(j) / self.render_res) * 2.0 - 1.0
            
            # Ray origin and direction
            ro = ti.Vector([0.0, 0.0, -3.0])
            rd = ti.Vector([u, v, 1.5]) 
            length_rd = ti.sqrt(rd[0]**2 + rd[1]**2 + rd[2]**2)
            rd /= length_rd
            
            # Sphere intersection logic
            b = ro[0]*rd[0] + ro[1]*rd[1] + ro[2]*rd[2]
            c = (ro[0]**2 + ro[1]**2 + ro[2]**2) - 1.0
            h = b**2 - c
            
            if h > 0.0:
                # We hit the sphere geometry
                t = -b - ti.sqrt(h)
                p = ro + rd * t
                
                # Calculate the surface normal
                n = ti.Vector([p[0], p[1], p[2]])
                length_n = ti.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
                n /= length_n
                
                # Convert 3D hit to spherical coordinates
                phi = ti.math.atan2(n[2], n[0])
                theta = ti.math.asin(n[1])
                
                # Map spherical coordinates to our 2D fluid grid
                tex_u = (phi / (2.0 * math.pi)) + 0.5
                tex_v = (theta / math.pi) + 0.5
                
                grid_x = tex_u * self.fluid_res
                grid_y = tex_v * self.fluid_res
                
                fluid_color = self.bilinear_interp(render_buffer, ti.Vector([grid_x, grid_y]))
                
                # Calculate the terminator line
                diffuse = n[0]*light_dir[0] + n[1]*light_dir[1] + n[2]*light_dir[2]
                diffuse = ti.max(0.0, diffuse)
                
                # Calculate atmospheric rim lighting
                rim = 1.0 - ti.max(0.0, -(rd[0]*n[0] + rd[1]*n[1] + rd[2]*n[2]))
                rim = rim**4.0 * 0.3
                
                self.final_output[i, j] = fluid_color * diffuse + ti.Vector([rim, rim, rim]) * diffuse
            else:
                self.final_output[i, j] = ti.Vector([0.0, 0.0, 0.0])

    def get_image_data(self):
        """Helper to extract the data for matplotlib."""
        return np.clip(self.final_output.to_numpy(), 0.0, 1.0)
