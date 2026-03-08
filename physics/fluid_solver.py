import taichi as ti
import math

@ti.data_oriented
class FluidEngine:
    def __init__(self, res=512, dt=0.03, jacobi_iters=40, vorticity_strength=2.8, planet_type="jupiter"):
        self.RES = res
        self.dt = dt
        self.jacobi_iters = jacobi_iters
        self.vorticity_strength = vorticity_strength
        self.planet_type = planet_type.lower()

        # --- PLANETARY PROFILES ---
        if self.planet_type == "venus":
            self.band_freq = 2.0      
            self.wind_mult = 0.5      
            self.shear_mult = 0.1     # Extremely smooth and smeary
            self.has_spot = 0
            self.color_1 = ti.Vector([0.85, 0.80, 0.65]) # Pale Yellow
            self.color_2 = ti.Vector([0.75, 0.65, 0.45]) # Mustard
            self.color_3 = ti.Vector([0.90, 0.85, 0.70]) # Cream
            self.color_storm = ti.Vector([0.95, 0.95, 0.85]) 
            self.color_spot = ti.Vector([0.0, 0.0, 0.0]) 

        elif self.planet_type == "saturn":
            self.band_freq = 10.0
            self.wind_mult = 1.8      
            self.shear_mult = 0.4     # Muted turbulence (hazy)
            self.has_spot = 0         
            self.color_1 = ti.Vector([0.85, 0.75, 0.55]) # Golden tan
            self.color_2 = ti.Vector([0.70, 0.60, 0.45]) # Deeper gold
            self.color_3 = ti.Vector([0.90, 0.85, 0.70]) # Pale gold
            self.color_storm = ti.Vector([0.95, 0.95, 0.90])
            self.color_spot = ti.Vector([0.0, 0.0, 0.0])

        elif self.planet_type == "uranus":
            self.band_freq = 6.0
            self.wind_mult = 0.8
            self.shear_mult = 0.3 
            self.has_spot = 1         
            self.color_1 = ti.Vector([0.40, 0.75, 0.85]) # Cyan
            self.color_2 = ti.Vector([0.25, 0.60, 0.75]) # Deep Cyan
            self.color_3 = ti.Vector([0.50, 0.80, 0.90]) # Pale Blue
            self.color_storm = ti.Vector([0.90, 0.95, 1.0]) # Methane cirrus
            self.color_spot = ti.Vector([0.15, 0.40, 0.55]) # Dark Methane Spot

        elif self.planet_type == "hot_jupiter":
            self.band_freq = 8.0
            self.wind_mult = 2.5      
            self.shear_mult = 2.0     
            self.has_spot = 0
            self.color_1 = ti.Vector([0.10, 0.10, 0.15]) # Ash grey
            self.color_2 = ti.Vector([0.90, 0.20, 0.05]) # Magma red
            self.color_3 = ti.Vector([1.00, 0.50, 0.10]) # Glowing orange
            self.color_storm = ti.Vector([1.00, 0.90, 0.50]) 
            self.color_spot = ti.Vector([0.0, 0.0, 0.0])

        else: # Default: Jupiter
            self.band_freq = 14.0     # Slight frequency increase
            self.wind_mult = 1.8      # Slightly faster jets
            self.shear_mult = 4.8     # Dialed back from the apocalyptic 6.0
            self.has_spot = 1
            self.color_1 = ti.Vector([1.00, 0.95, 0.88]) # Brighter Cream
            self.color_2 = ti.Vector([0.70, 0.20, 0.05]) # CRITICAL: Punchier, dark rust
            self.color_3 = ti.Vector([0.45, 0.55, 0.65]) # Saturated Polar Blue
            self.color_storm = ti.Vector([1.0, 1.0, 1.0]) # White
            self.color_spot = ti.Vector([0.80, 0.10, 0.00]) # Violent Red

        # Core Fields
        self.velocity = ti.Vector.field(2, dtype=float, shape=(self.RES, self.RES))
        self.new_velocity = ti.Vector.field(2, dtype=float, shape=(self.RES, self.RES))
        self.dye = ti.Vector.field(3, dtype=float, shape=(self.RES, self.RES))
        self.new_dye = ti.Vector.field(3, dtype=float, shape=(self.RES, self.RES))
        self.pressure = ti.field(dtype=float, shape=(self.RES, self.RES))
        self.new_pressure = ti.field(dtype=float, shape=(self.RES, self.RES))
        self.divergence = ti.field(dtype=float, shape=(self.RES, self.RES))
        self.curl = ti.field(dtype=float, shape=(self.RES, self.RES))

    @ti.func
    def bilinear_interp(self, field: ti.template(), p):
        u = p[0] % float(self.RES)
        v = p[1] % float(self.RES)
        i, j = int(u), int(v)
        fx, fy = u - i, v - j
        i_next = (i + 1) % self.RES
        j_next = (j + 1) % self.RES
        
        c00 = field[i, j]
        c10 = field[i_next, j]
        c01 = field[i, j_next]
        c11 = field[i_next, j_next]
        
        return (c00 * (1.0 - fx) + c10 * fx) * (1.0 - fy) + \
               (c01 * (1.0 - fx) + c11 * fx) * fy

    @ti.kernel
    def advect(self, field: ti.template(), new_field: ti.template(), vec_field: ti.template(), dissipation: float):
        for i, j in field:
            p = ti.Vector([float(i), float(j)])
            back_p = p - vec_field[i, j] * self.dt
            new_field[i, j] = self.bilinear_interp(field, back_p) * dissipation

    @ti.kernel
    def compute_vorticity(self):
        for i, j in self.curl:
            l = (i - 1 + self.RES) % self.RES
            r = (i + 1) % self.RES
            b = (j - 1 + self.RES) % self.RES
            t = (j + 1) % self.RES
            self.curl[i, j] = (self.velocity[r, j][1] - self.velocity[l, j][1] - self.velocity[i, t][0] + self.velocity[i, b][0]) * 0.5

    @ti.kernel
    def apply_vorticity_confinement(self):
        for i, j in self.velocity:
            l = (i - 1 + self.RES) % self.RES
            r = (i + 1) % self.RES
            b = (j - 1 + self.RES) % self.RES
            t = (j + 1) % self.RES
            cl = ti.abs(self.curl[l, j])
            cr = ti.abs(self.curl[r, j])
            cb = ti.abs(self.curl[i, b])
            ct = ti.abs(self.curl[i, t])
            dx = (cr - cl) * 0.5
            dy = (ct - cb) * 0.5
            length = ti.sqrt(dx**2 + dy**2) + 1e-5
            self.velocity[i, j][0] += ((dy / length) * self.curl[i, j]) * self.vorticity_strength * self.dt
            self.velocity[i, j][1] += ((-dx / length) * self.curl[i, j]) * self.vorticity_strength * self.dt

    @ti.kernel
    def compute_divergence(self):
        for i, j in self.divergence:
            l = (i - 1 + self.RES) % self.RES
            r = (i + 1) % self.RES
            b = (j - 1 + self.RES) % self.RES
            t = (j + 1) % self.RES
            self.divergence[i, j] = (self.velocity[r, j][0] - self.velocity[l, j][0] + self.velocity[i, t][1] - self.velocity[i, b][1]) * 0.5

    @ti.kernel
    def solve_pressure(self):
        for i, j in self.pressure:
            l = (i - 1 + self.RES) % self.RES
            r = (i + 1) % self.RES
            b = (j - 1 + self.RES) % self.RES
            t = (j + 1) % self.RES
            self.new_pressure[i, j] = (self.pressure[l, j] + self.pressure[r, j] + self.pressure[i, b] + self.pressure[i, t] - self.divergence[i, j]) * 0.25

    @ti.kernel
    def subtract_gradient(self):
        for i, j in self.velocity:
            l = (i - 1 + self.RES) % self.RES
            r = (i + 1) % self.RES
            b = (j - 1 + self.RES) % self.RES
            t = (j + 1) % self.RES
            self.velocity[i, j][0] -= (self.pressure[r, j] - self.pressure[l, j]) * 0.5
            self.velocity[i, j][1] -= (self.pressure[i, t] - self.pressure[i, b]) * 0.5

    @ti.kernel
    def inject_zonal_jets(self, frame: int):
        for i, j in self.velocity:
            lat = float(j) / self.RES
            lon = float(i) / self.RES
            time = float(frame) * 0.05
            
            # 1. Fractal Turbulence (fBM)
            turb = 0.0
            amp = 1.0
            freq = 1.0
            for _ in ti.static(range(4)):
                turb += ti.sin(lon * 80.0 * freq + time) * ti.cos(lat * 80.0 * freq - time) * amp
                amp *= 0.5
                freq *= 2.0
                
            wobble = ti.sin(lon * 15.0 + time) * 0.03 * self.shear_mult
            band_profile = ti.sin((lat + wobble) * self.band_freq * math.pi)
            shear_mask = 1.0 - ti.abs(band_profile)
            shear_mask = shear_mask * shear_mask 
            
            micro_turbulence = turb * 180.0 * self.shear_mult
            wind_speed = (band_profile * 140.0 * self.wind_mult) + (micro_turbulence * shear_mask)
            self.velocity[i, j][0] = wind_speed
            
            # 2. Dynamic Cyclogenesis (The Great Red/Dark Spot)
            if self.has_spot == 1:
                spot_x = self.RES * 0.45
                spot_y = self.RES * 0.35
                dx = float(i) - spot_x
                dy = float(j) - spot_y
                dist = ti.sqrt(dx**2 + dy**2)
                spot_radius = self.RES * 0.08
                
                spot_distortion = turb * 0.015 * self.RES * self.shear_mult
                
                if dist < spot_radius + spot_distortion:
                    tangent_v = (dist / spot_radius) * 250.0 * self.wind_mult
                    self.velocity[i, j][0] += -dy / (dist + 1e-5) * tangent_v
                    self.velocity[i, j][1] += dx / (dist + 1e-5) * tangent_v

                    if ti.random() > 0.6: # Significantly slows the red dye flood
                        self.dye[i, j] = self.color_spot
            
            # 3. Coloring and Polar Caps
            polar_mask = ti.abs(lat - 0.5) * 2.0 
            
            if ti.random() > 0.94: # Faster global repaint to contain the storms
                if wind_speed > 80.0 * self.wind_mult:
                    self.dye[i, j] = self.color_1 
                elif wind_speed < -80.0 * self.wind_mult:
                    self.dye[i, j] = self.color_2 
                else:
                    base_c = self.color_3 if polar_mask > 0.65 else self.color_1 * 0.8
                    self.dye[i, j] = base_c
                    
            # 4. "String of Pearls" Storm Pop-ups
            pearl_trigger = ti.sin(lon * 40.0 + time * 2.0) * ti.cos(lat * 30.0)
            if pearl_trigger > 0.98 and ti.random() > 0.7:
                self.dye[i, j] = self.color_storm
                self.velocity[i, j][0] += (ti.random() - 0.5) * 300.0 * self.shear_mult
                self.velocity[i, j][1] += (ti.random() - 0.5) * 300.0 * self.shear_mult

    @ti.kernel
    def copy_fields(self, f1: ti.template(), f2: ti.template()):
        for i, j in f1:
            f1[i, j] = f2[i, j]

    def step(self, frame: int):
        self.inject_zonal_jets(frame)
        self.compute_vorticity()
        self.apply_vorticity_confinement()
        self.advect(self.velocity, self.new_velocity, self.velocity, 0.999)
        self.copy_fields(self.velocity, self.new_velocity)
        self.compute_divergence()
        for _ in range(self.jacobi_iters):
            self.solve_pressure()
            self.copy_fields(self.pressure, self.new_pressure)
        self.subtract_gradient()
        self.advect(self.dye, self.new_dye, self.velocity, 0.995)
        self.copy_fields(self.dye, self.new_dye)
