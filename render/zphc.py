"""Zero-Point Harmonic Collapse slit-scan tunnel renderer.

Extracted from `infinite_director.ZPHCTransitionRenderer` so that the
manifest-driven adapter chain does not import `infinite_director`, which
pulls in `traveller_world_generator` (the external worldmaker repo) at
module load.

The class is otherwise unchanged from its original form: a Taichi
relativistic slit-scan tunnel that collapses through a peak flash and
re-opens. Used both as a standalone shot renderer
(`render.adapters.SlitScanTunnelRenderer`) and as the GPU backend for
the slit-scan family of transitions.
"""
import math

import taichi as ti


@ti.data_oriented
class ZPHCTransitionRenderer:
    def __init__(self, res=1024):
        self.res = res
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res, res))

    @ti.kernel
    def render_frame(self, time: ti.f32, progress: ti.f32):
        for i, j in self.pixels:
            u = (float(i) / self.res) * 2.0 - 1.0
            v = (float(j) / self.res) * 2.0 - 1.0

            r = ti.sqrt(u ** 2 + v ** 2) + 1e-5
            angle = ti.atan2(v, u)

            z = 1.0 / r + time * 10.0
            noise = ti.sin(z * 5.0 + angle * 8.0) * ti.cos(z * 3.0 - time * 2.0)

            base_blue = ti.Vector([0.1, 0.3, 0.9])
            base_red = ti.Vector([0.9, 0.1, 0.3])
            mix_val = (ti.sin(angle * 3.0 + time) + 1.0) * 0.5
            color = base_blue * mix_val + base_red * (1.0 - mix_val)

            slit = ti.abs(ti.sin(z * 50.0))
            color += color * slit * 2.0

            intensity = 1.0 - ti.abs(progress - 0.5) * 2.0
            vignette = ti.exp(-r * 2.0)
            final_color = color * vignette * (noise + 1.0)

            if intensity > 0.8:
                flash = (intensity - 0.8) * 5.0
                final_color += ti.Vector([1.0, 1.0, 1.0]) * flash

            self.pixels[i, j] = final_color
