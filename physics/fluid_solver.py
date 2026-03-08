import taichi as ti
import math

@ti.data_oriented
class FluidEngine:
    def __init__(self, res=512, dt=0.03, jacobi_iters=40, vorticity_strength=2.8):
        self.RES = res
        self.dt = dt
        self.jacobi_iters = jacobi_iters
        self.vorticity_strength = vorticity_strength

        # Core Fields
        self.velocity = ti.Vector.field(2, dtype=float, shape=(self.RES, self.RES))
        self.new_velocity = ti.Vector.field(2, dtype=float, shape=(self.RES, self.RES))
        self.dye = ti.Vector.field(3, dtype=float, shape=(self.RES, self.RES))
        self.new_dye = ti.Vector.field(3, dtype=float, shape=(self.RES, self.RES))

        # Pressure and Vorticity Fields
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
            noise = ti.sin(float(i) * 0.05 + float(frame) * 0.1) * 30.0
            wind_speed = ti.sin((float(j) / self.RES) * 8.0 * math.pi) * 100.0 + noise
            self.velocity[i, j][0] = wind_speed
            
            if ti.random() > 0.85:
                if wind_speed > 40.0:
                    self.dye[i, j] = ti.Vector([0.9, 0.8, 0.7]) 
                elif wind_speed < -40.0:
                    self.dye[i, j] = ti.Vector([0.7, 0.3, 0.2]) 
                else:
                    self.dye[i, j] = ti.Vector([0.8, 0.5, 0.3]) 
            if ti.random() > 0.998:
                self.dye[i, j] = ti.Vector([1.0, 0.95, 0.9])

    @ti.kernel
    def copy_fields(self, f1: ti.template(), f2: ti.template()):
        for i, j in f1:
            f1[i, j] = f2[i, j]

    def step(self, frame: int):
        """Executes one full frame of the fluid simulation."""
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
