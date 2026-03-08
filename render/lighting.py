import taichi as ti

@ti.data_oriented
class VolumetricIlluminator:
    def __init__(self, res=512):
        self.RES = res
        # This buffer holds the flattened, lit texture before it gets wrapped onto a 3D object
        self.render_buffer = ti.Vector.field(3, dtype=float, shape=(self.RES, self.RES))

    @ti.kernel
    def compute_lighting(self, dye: ti.template()):
        # Fixed light direction for the internal cloud shadows
        light_dir = ti.Vector([0.707, 0.707])
        
        for i, j in self.render_buffer:
            base_color = dye[i, j]
            # Treat brightness as physical cloud density/height
            density = ti.sqrt(base_color[0]**2 + base_color[1]**2 + base_color[2]**2)
            shadow = 0.0
            ray_pos = ti.Vector([float(i), float(j)])
            
            # March across the 2D plane to accumulate shadows
            for step in range(12):
                ray_pos += light_dir * 5.0 
                # Periodic wrap for seamless tiling
                sx = int(ray_pos[0]) % self.RES
                sy = int(ray_pos[1]) % self.RES
                
                sample_color = dye[sx, sy]
                sample_density = ti.sqrt(sample_color[0]**2 + sample_color[1]**2 + sample_color[2]**2)
                
                # If the cloud in the path is taller, cast a shadow
                if sample_density > density:
                    shadow += (sample_density - density) * 0.15
                    
            # Beer-Lambert Law exponential decay
            illumination = ti.exp(-shadow * 2.5)
            ambient = 0.2
            self.render_buffer[i, j] = base_color * (illumination + ambient)
