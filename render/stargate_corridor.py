import taichi as ti
from render.detrng import rand_centered, tick, S_AA_X, S_AA_Y, S_GRAIN
import math


@ti.data_oriented
class StargateCorridor:
    """
    Widescreen slit-scan corridor renderer in the spirit of the Stargate
    sequence from '2001: A Space Odyssey' (Trumbull's slit-scan rig).

    Two infinite luminous planes converge on a vanishing line at the centre
    of frame. Streaming fBM light patterns rush past the camera while the
    palette moves through distinct colour 'epochs', mirroring the film's
    successive corridor movements. The corridor can roll continuously from
    a horizontal pair of planes to a vertical pair (the film cuts between
    both orientations).
    """

    def __init__(self, render_w=1024, render_h=576, samples=2):
        self.render_w = render_w
        self.render_h = render_h
        self.samples = samples
        self.pixels = ti.Vector.field(3, dtype=float, shape=(render_w, render_h))

    # ------------------------------------------------------------------ #
    # Noise primitives                                                    #
    # ------------------------------------------------------------------ #
    @ti.func
    def hash21(self, p):
        h = ti.sin(p[0] * 127.1 + p[1] * 311.7) * 43758.5453
        return h - ti.floor(h)

    @ti.func
    def vnoise(self, p):
        """Smooth 2D value noise (bilinear over hashed lattice corners)."""
        ip = ti.floor(p)
        fp = p - ip
        # Quintic smoothstep for C2 continuity
        w = fp * fp * fp * (fp * (fp * 6.0 - 15.0) + 10.0)

        a = self.hash21(ip)
        b = self.hash21(ip + ti.Vector([1.0, 0.0]))
        c = self.hash21(ip + ti.Vector([0.0, 1.0]))
        d = self.hash21(ip + ti.Vector([1.0, 1.0]))

        return a * (1.0 - w[0]) * (1.0 - w[1]) + b * w[0] * (1.0 - w[1]) + \
               c * (1.0 - w[0]) * w[1] + d * w[0] * w[1]

    @ti.func
    def fbm(self, p):
        value = 0.0
        amp = 0.5
        q = p
        for _ in ti.static(range(5)):
            value += self.vnoise(q) * amp
            q = q * 2.03 + ti.Vector([13.7, 7.1])
            amp *= 0.5
        return value

    # ------------------------------------------------------------------ #
    # Colour epochs                                                       #
    # ------------------------------------------------------------------ #
    @ti.func
    def epoch_palette(self, progress, band):
        """
        Returns the corridor colour for a pattern band value in [0,1].
        The palette sweeps through four epochs over the act:
          0: electric blue / violet     (the plunge)
          1: molten red / amber         (the furnace)
          2: emerald / cyan             (the crystalline reach)
          3: white-gold overload        (approaching the infinite)
        """
        e = progress * 4.0
        idx = ti.min(3, int(e))
        frac = e - float(idx)

        c_a = ti.Vector([0.05, 0.25, 0.95])
        c_b = ti.Vector([0.55, 0.10, 0.85])
        n_a = ti.Vector([0.95, 0.25, 0.05])
        n_b = ti.Vector([0.95, 0.65, 0.10])

        if idx == 1:
            c_a = ti.Vector([0.95, 0.25, 0.05])
            c_b = ti.Vector([0.95, 0.65, 0.10])
            n_a = ti.Vector([0.05, 0.85, 0.45])
            n_b = ti.Vector([0.10, 0.65, 0.95])
        elif idx == 2:
            c_a = ti.Vector([0.05, 0.85, 0.45])
            c_b = ti.Vector([0.10, 0.65, 0.95])
            n_a = ti.Vector([1.00, 0.95, 0.80])
            n_b = ti.Vector([0.95, 0.80, 0.45])
        elif idx == 3:
            c_a = ti.Vector([1.00, 0.95, 0.80])
            c_b = ti.Vector([0.95, 0.80, 0.45])
            n_a = ti.Vector([1.00, 1.00, 1.00])
            n_b = ti.Vector([1.00, 1.00, 1.00])

        # Colour within the current epoch, then ease toward the next epoch
        cur = c_a * (1.0 - band) + c_b * band
        nxt = n_a * (1.0 - band) + n_b * band
        blend = frac * frac * (3.0 - 2.0 * frac)
        return cur * (1.0 - blend) + nxt * blend

    # ------------------------------------------------------------------ #
    # The corridor itself                                                 #
    # ------------------------------------------------------------------ #
    @ti.func
    def corridor_sample(self, u, v, time, progress):
        # Perspective projection of two planes at y = +/-1:
        # depth to the plane grows as 1/|v|, world x scales with depth.
        av = ti.abs(v) + 1e-4
        depth = 1.0 / av
        world_x = u * depth

        # The rush: pattern streams toward the camera, accelerating over the act
        speed = 6.0 + progress * 26.0
        z = depth + time * speed

        # Streaked light texture: heavily anisotropic fBM (stretched along z)
        band = self.fbm(ti.Vector([world_x * 0.7, z * 0.16]))
        streak = self.fbm(ti.Vector([world_x * 3.0 + z * 0.02, z * 0.9]))
        pattern = band * 0.72 + streak * 0.28

        # Modulate luminance by the pattern so troughs fall to near-black,
        # keeping the corridor high-contrast rather than a wash of colour
        color = self.epoch_palette(progress, ti.min(1.0, pattern * 1.4)) * \
                (0.15 + pattern * pattern * 2.2)

        # Slit structure: hard luminous ribs sweeping past
        rib = ti.abs(ti.sin(z * 2.2 + world_x * 0.35))
        rib = rib ** 6.0
        color += color * rib * 1.6

        # Distance haze toward the vanishing line, plus its hot glow
        fog = ti.exp(-depth * 0.045)
        glow = ti.exp(-av * 9.0) * (1.1 + ti.sin(time * 3.0) * 0.12)
        glow_col = self.epoch_palette(progress, 0.15) * 2.2

        # Edges of frame fall off into blackness (the corridor has no walls)
        edge_fade = ti.exp(-ti.abs(u) * 0.55)

        return (color * fog + glow_col * glow) * edge_fade

    @ti.kernel
    def render_frame(self, time: float, progress: float, roll: float, exposure: float):
        """
        progress: position within the stargate act [0,1] -- drives palette
                  epochs and acceleration.
        roll:     corridor orientation in radians (0 = horizontal planes,
                  pi/2 = vertical). Continuous values give the slow tumble.
        """
        aspect = float(self.render_w) / float(self.render_h)
        for i, j in self.pixels:
            accumulated = ti.Vector([0.0, 0.0, 0.0])
            for k in range(self.samples):
                ox = rand_centered(i, j, k, tick(time) + S_AA_X)
                oy = rand_centered(i, j, k, tick(time) + S_AA_Y)
                u = ((float(i) + ox) / self.render_w * 2.0 - 1.0) * aspect
                v = (float(j) + oy) / self.render_h * 2.0 - 1.0

                # Roll the corridor around the view axis
                cu = u * ti.cos(roll) - v * ti.sin(roll)
                cv = u * ti.sin(roll) + v * ti.cos(roll)

                accumulated += self.corridor_sample(cu, cv, time, progress)

            color = accumulated / float(self.samples) * exposure

            # ACES tone mapping (matches SphereCamera output)
            a = 2.51
            b = 0.03
            c = 2.43
            d = 0.59
            e = 0.14
            color = (color * (a * color + b)) / (color * (c * color + d) + e)

            # Film grain consistent with the rest of the sequence
            noise = rand_centered(i, j, tick(time), S_GRAIN) * 0.04
            self.pixels[i, j] = color + ti.Vector([noise, noise, noise])
