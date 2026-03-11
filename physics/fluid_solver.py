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
        self.has_pearls = 0
        import random
        self.seed_offset = random.random() * 10000.0

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
            self.jacobi_iters = 80
            self.vorticity_strength = 8.0 + random.uniform(-2.0, 2.0)
            self.band_freq = 14.0 + random.uniform(-3.0, 3.0)
            self.wind_mult = 1.8 + random.uniform(-0.4, 0.4)
            self.shear_mult = 5.0 + random.uniform(-1.0, 1.5)
            self.has_spot = 1 if random.random() > 0.3 else 0
            self.has_pearls = 1 if random.random() > 0.3 else 0
            
            # Jupiter Palette Details (Zones = Light, Belts = Dark)
            self.color_zone_1 = ti.Vector([0.90, 0.90, 0.88]) # Pearl white (Equatorial Zone)
            self.color_zone_2 = ti.Vector([0.80, 0.75, 0.65]) # Tan/cream (Tropical Zones)
            self.color_belt_1 = ti.Vector([0.55, 0.35, 0.20]) # Dark Brown (North Equatorial Belt)
            self.color_belt_2 = ti.Vector([0.75, 0.40, 0.25]) # Rust/Orange (South Equatorial Belt)
            
            self.color_3 = ti.Vector([0.15, 0.35, 0.65]) # Deep Juno Polar Blue
            self.color_storm = ti.Vector([1.0, 1.0, 1.0]) # White
            self.color_spot = ti.Vector([0.70 + random.uniform(-0.1, 0.1), 0.15 + random.uniform(-0.05, 0.05), 0.05]) # Terracotta Red

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
            time = float(frame) * 0.05 + self.seed_offset
            
            # 1. Fractional Micro-Turbulence (fBM)
            turb = 0.0
            amp = 1.5
            freq = 1.0
            for _ in ti.static(range(8)):
                turb += ti.sin(lon * 80.0 * freq + time) * ti.cos(lat * 80.0 * freq - time) * amp
                amp *= 0.5
                freq *= 2.0
                
            wobble = ti.sin(lon * 15.0 + time) * 0.03 * self.shear_mult
            # Smooth step the sine wave to create sharper boundaries between jets
            raw_band = ti.sin((lat + wobble) * self.band_freq * math.pi)
            band_profile = ti.math.sign(raw_band) * (ti.abs(raw_band) ** 0.5) 
            
            shear_mask = 1.0 - ti.abs(band_profile)
            shear_mask = shear_mask * shear_mask 
            
            micro_turbulence = turb * 180.0 * self.shear_mult
            wind_speed = (band_profile * 160.0 * self.wind_mult) + (micro_turbulence * shear_mask)
            self.velocity[i, j][0] = wind_speed
            
            # 2. Dynamic Cyclogenesis (The Great Red/Dark Spot)
            if self.has_spot == 1:
                spot_x = self.RES * 0.45
                spot_y = self.RES * 0.35
                dx = float(i) - spot_x
                dy = float(j) - spot_y
                
                # Make the spot correctly elliptical (wider than it is tall)
                dist_elliptical = ti.sqrt((dx * 0.7)**2 + (dy * 1.5)**2)
                spot_radius = self.RES * 0.07
                
                spot_distortion = turb * 0.015 * self.RES * self.shear_mult
                
                if dist_elliptical < spot_radius + spot_distortion:
                    # Flow around the storm (hollow vortex)
                    tangent_v = (dist_elliptical / spot_radius) * 350.0 * self.wind_mult
                    self.velocity[i, j][0] += -dy / (dist_elliptical + 1e-5) * tangent_v
                    self.velocity[i, j][1] += dx / (dist_elliptical + 1e-5) * tangent_v

                    # Pale halo around the terracotta core
                    if dist_elliptical > spot_radius * 0.75:
                        if ti.random() > 0.8:
                            self.dye[i, j] = self.color_zone_1 * 1.1 
                    elif ti.random() > 0.6: 
                        self.dye[i, j] = self.color_spot
            
            # 3. Strict Zonal Coloring and Polar Caps
            polar_mask = ti.abs(lat - 0.5) * 2.0 
            
            if ti.random() > 0.92: # Continual repainting of the zonal bands
                if polar_mask > 0.65:
                    self.dye[i, j] = self.color_3
                else:
                    # Hardcoded latitudinal bounds for iconic Belts and Zones
                    if 0.45 < lat < 0.55: # Equatorial Zone
                        self.dye[i, j] = self.color_zone_1
                    elif 0.55 <= lat < 0.62: # North Equatorial Belt (Dark Brown)
                        self.dye[i, j] = self.color_belt_1
                    elif 0.38 < lat <= 0.45: # South Equatorial Belt (Rust/Orange)
                        self.dye[i, j] = self.color_belt_2
                    elif wind_speed > 60.0 * self.wind_mult: # Fast Prograde Zones
                        self.dye[i, j] = self.color_zone_2
                    elif wind_speed < -60.0 * self.wind_mult: # Fast Retrograde Belts
                        self.dye[i, j] = self.color_belt_1 * 0.9 # Darker
                    else:
                        # Fallback mixture
                        mix_factor = (ti.sin(lat * 30.0) + 1.0) * 0.5
                        self.dye[i, j] = self.color_zone_2 * mix_factor + self.color_belt_2 * (1.0 - mix_factor)
                    
            # 4. Explosive Storm Pop-ups
            storm_trigger = ti.sin(lon * 40.0 + time * 2.0) * ti.cos(lat * 30.0) + turb * 0.2
            if storm_trigger > 1.05 and ti.random() > 0.8:
                self.dye[i, j] = self.color_storm
                self.velocity[i, j][0] += (ti.random() - 0.5) * 800.0 * self.shear_mult
                self.velocity[i, j][1] += (ti.random() - 0.5) * 800.0 * self.shear_mult

            # 5. String of Pearls (Jupiter-specific)
            if self.has_pearls == 1:
                pearl_lat = 0.28 # Southern hemisphere
                lat_dist = ti.abs(lat - pearl_lat)
                if lat_dist < 0.025:
                    pearl_trigger = ti.sin(lon * 60.0 - time * 1.2)
                    if pearl_trigger > 0.90:
                        self.dye[i, j] = ti.Vector([0.95, 0.95, 1.0])
                        # Fast rotating anticyclones
                        dx_p = lon * 60.0 % (2.0 * math.pi) - math.pi
                        dy_p = (lat - pearl_lat) * 60.0
                        d_p = ti.sqrt(dx_p**2 + dy_p**2) + 1e-5
                        tangent_v_p = (d_p / 0.5) * 600.0 * self.wind_mult
                        self.velocity[i, j][0] += -dy_p / d_p * tangent_v_p
                        self.velocity[i, j][1] += dx_p / d_p * tangent_v_p

            # 6. Brown Barges (Dark equatorial cyclones)
            if self.planet_type == "jupiter":
                barge_lat = 0.62 # Northern hemisphere
                barge_dist = ti.abs(lat - barge_lat)
                if barge_dist < 0.03:
                    barge_trigger = ti.sin(lon * 25.0 + time * 0.8) + turb * 0.5
                    if barge_trigger > 1.2:
                        self.dye[i, j] = self.color_belt_1 * 0.5 # Very dark brown
                        # Strong cyclonic rotation
                        dx_b = lon * 25.0 % (2.0 * math.pi) - math.pi
                        dy_b = (lat - barge_lat) * 25.0
                        d_b = ti.sqrt(dx_b**2 + dy_b**2) + 1e-5
                        tangent_v_b = (d_b / 0.8) * 400.0 * self.wind_mult
                        self.velocity[i, j][0] += dy_b / d_b * tangent_v_b
                        self.velocity[i, j][1] += -dx_b / d_b * tangent_v_b

            # 7. Polar Cyclones (Juno style)
            if self.planet_type == "jupiter" and polar_mask > 0.85:
                # Dense clusters of vortices near poles
                polar_vortex_trigger = ti.sin(lon * 80.0) * ti.cos(lat * 120.0 + time) + turb
                if polar_vortex_trigger > 1.5:
                    self.dye[i, j] = self.color_3 * 1.2 # Bright blue centers
                    self.velocity[i, j][0] += (ti.random() - 0.5) * 1200.0 * self.shear_mult
                    self.velocity[i, j][1] += (ti.random() - 0.5) * 1200.0 * self.shear_mult

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
