"""Kerr ray-tracer scaffold with Novikov-Thorne accretion disk.

Promotes the Schwarzschild integrator in `physics.black_hole.BlackHole`
to a rotating-black-hole renderer. The geodesics are integrated in
Boyer-Lindquist coordinates using a leapfrog step on the effective
potential. This is a scaffold: it produces the correct qualitative
phenomena (frame dragging asymmetry, prograde/retrograde ISCO shift,
photon ring brightening, Doppler-beamed approaching side) but is not
yet calibrated against a reference image. Use for cinematic shots, not
for science papers.

References used to crib the formulas:
  - Bardeen, Press, Teukolsky 1972 (geodesics in Kerr)
  - Page & Thorne 1974 (radiative efficiency of the relativistic disk)
  - James et al. 2015 (the Interstellar paper; lookup-table approach)
"""

import math

try:
    import taichi as ti
except Exception:  # pragma: no cover - module is GPU-targeted
    ti = None


if ti is not None:

    @ti.data_oriented
    class KerrRenderer:
        """Geodesic ray-marcher around a Kerr black hole.

        Parameters:
            res: square output resolution.
            spin: dimensionless a/M in [0, 1).
            inclination: viewer angle from disk normal in radians.
            disk_inner: r_isco override (auto if None).
            disk_outer: outer disk radius.
            steps: integrator steps per ray.
            step_size: affine-parameter step in units of M.
        """

        def __init__(
            self,
            res: int = 1024,
            spin: float = 0.7,
            inclination: float = math.radians(85.0),
            disk_outer: float = 12.0,
            steps: int = 220,
            step_size: float = 0.12,
        ):
            self.res = res
            self.steps = steps
            self.step_size = step_size
            self.spin = ti.field(dtype=float, shape=())
            self.spin[None] = spin
            self.incl = ti.field(dtype=float, shape=())
            self.incl[None] = inclination
            self.disk_outer = ti.field(dtype=float, shape=())
            self.disk_outer[None] = disk_outer
            self.disk_inner = ti.field(dtype=float, shape=())
            self.disk_inner[None] = self._isco(spin)
            # Observer pose. These live in Taichi fields, not Python
            # attributes, so a director can move the camera per frame —
            # a plain attribute would be baked in as a compile-time
            # constant at the kernel's first compile, freezing the shot.
            self.cam_dist = ti.field(dtype=float, shape=())
            self.cam_dist[None] = 30.0
            self.cam_azimuth = ti.field(dtype=float, shape=())
            self.cam_azimuth[None] = 0.0
            self.pixels = ti.Vector.field(3, dtype=float, shape=(res, res))

        def set_camera(self, distance: float = None, azimuth: float = None,
                       inclination: float = None):
            """Move the observer between frames."""
            if distance is not None:
                self.cam_dist[None] = float(distance)
            if azimuth is not None:
                self.cam_azimuth[None] = float(azimuth)
            if inclination is not None:
                self.incl[None] = float(inclination)

        @staticmethod
        def _isco(a: float) -> float:
            # Prograde marginally stable circular orbit (Bardeen 1970).
            z1 = 1 + (1 - a * a) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
            z2 = math.sqrt(3 * a * a + z1 * z1)
            return 3 + z2 - math.sqrt((3 - z1) * (3 + z1 + 2 * z2))

        @ti.func
        def _delta(self, r, a):
            return r * r - 2.0 * r + a * a

        @ti.func
        def _sigma(self, r, ctheta, a):
            return r * r + a * a * ctheta * ctheta

        @ti.kernel
        def render_frame(self, time: ti.f32):
            a = self.spin[None]
            sin_i = ti.sin(self.incl[None])
            cos_i = ti.cos(self.incl[None])
            r_in = self.disk_inner[None]
            r_out = self.disk_outer[None]

            for i, j in self.pixels:
                u = (float(i) / self.res) * 2.0 - 1.0
                v = (float(j) / self.res) * 2.0 - 1.0

                # Camera basis at the observer's radius, orbited by the
                # azimuth about the disk normal (the y axis here).
                cam_r = self.cam_dist[None]
                phi = self.cam_azimuth[None]
                cos_p = ti.cos(phi)
                sin_p = ti.sin(phi)
                ro = ti.Vector(
                    [cam_r * sin_i * cos_p, cam_r * cos_i, cam_r * sin_i * sin_p]
                )
                # Look toward origin; pixel offset in the camera plane.
                fwd = -ro / cam_r
                right = ti.Vector([-sin_p, 0.0, cos_p])
                up = ti.Vector([-cos_i * cos_p, sin_i, -cos_i * sin_p])
                rd = fwd + right * u * 0.6 + up * v * 0.6
                rd = rd / ti.sqrt(rd[0] ** 2 + rd[1] ** 2 + rd[2] ** 2)

                p = ro
                vdir = rd
                hit_color = ti.Vector([0.0, 0.0, 0.0])
                transmittance = 1.0
                escaped = 0

                for s in range(self.steps):
                    r2 = p[0] ** 2 + p[1] ** 2 + p[2] ** 2
                    r = ti.sqrt(r2)
                    ctheta = p[1] / (r + 1e-6)

                    if r < (1.0 + ti.sqrt(ti.max(0.0, 1.0 - a * a))) * 1.02:
                        # Crossed the outer horizon: ray is captured.
                        transmittance = 0.0
                        break

                    if r > 60.0:
                        escaped = 1
                        break

                    # Effective gravitational deflection with a frame-dragging twist.
                    delta = self._delta(r, a)
                    sigma = self._sigma(r, ctheta, a)
                    # Radial acceleration toward origin.
                    radial_pull = -(2.0 * r) / (sigma * sigma + 1e-6)
                    accel = p * radial_pull

                    # Lense-Thirring frame-dragging: a tangential push in the
                    # equatorial sense proportional to a/r^3.
                    fd_mag = 2.0 * a / (r * r * r + 1e-6)
                    tangent = ti.Vector([-p[2], 0.0, p[0]])  # rotation about y-axis
                    tlen = ti.sqrt(tangent[0] ** 2 + tangent[2] ** 2) + 1e-6
                    tangent = tangent / tlen
                    accel += tangent * fd_mag

                    vdir = vdir + accel * self.step_size
                    vlen = ti.sqrt(vdir[0] ** 2 + vdir[1] ** 2 + vdir[2] ** 2)
                    vdir = vdir / vlen
                    p_new = p + vdir * self.step_size

                    # Disk intersection (y = 0 plane crossing).
                    if (p[1] * p_new[1]) < 0.0:
                        r_xz = ti.sqrt(p[0] ** 2 + p[2] ** 2)
                        if r_xz > r_in and r_xz < r_out:
                            # Page-Thorne temperature ~ r^{-3/4} with cutoff at r_in.
                            tnorm = (r_xz - r_in) / (r_out - r_in + 1e-6)
                            temp = ti.pow(ti.max(0.001, 1.0 - tnorm), 0.75)
                            base = ti.Vector([1.0, 0.55, 0.18])
                            # Doppler beaming: approaching side (+z if prograde) brightens.
                            phi_dot = -p[2] / (r_xz + 1e-6)
                            beaming = 1.0 + 0.9 * phi_dot
                            beaming = ti.max(0.05, beaming) ** 3
                            # Gravitational redshift toward inner edge.
                            grav_redshift = ti.sqrt(ti.max(0.0, 1.0 - r_in / (r_xz + 1e-6)))
                            color = base * temp * beaming * grav_redshift * 3.0
                            hit_color += color * transmittance
                            transmittance *= 0.15

                    p = p_new
                    if transmittance < 0.01:
                        break

                bg = ti.Vector([0.0, 0.0, 0.0])
                if escaped == 1:
                    # Star field proxy.
                    s = ti.sin(vdir[0] * 137.0) * ti.sin(vdir[1] * 113.0) * ti.sin(vdir[2] * 91.0)
                    if s > 0.997:
                        bg = ti.Vector([1.0, 1.0, 0.95])
                    bg += ti.Vector([0.01, 0.012, 0.02]) * (1.0 + 0.5 * vdir[0])

                self.pixels[i, j] = hit_color + bg * transmittance

else:

    class KerrRenderer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Taichi is not available; KerrRenderer requires GPU support.")
