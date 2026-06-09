"""CPU-side tests for the DESI fetcher CLI synth mode.

Download mode requires network + astropy; tested manually.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_synth_cli_writes_valid_npz() -> None:
    from render.desi_fetch import main

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "test.npz"
        rc = main([
            "synth",
            "--out", str(out),
            "--n-particles", "5000",
            "--n-clusters", "30",
            "--box", "1000",
            "--seed", "7",
        ])
        assert rc == 0
        assert out.exists()
        data = np.load(out)
        assert "positions" in data.files
        assert "tracer" in data.files
        assert "magnitudes" in data.files
        assert "redshifts" in data.files
        assert data["positions"].shape == (5000, 3)
        assert data["positions"].dtype == np.float32
        assert data["tracer"].dtype == np.int32
        assert 0 <= int(data["tracer"].min())
        assert int(data["tracer"].max()) < 4
    print("ok: test_synth_cli_writes_valid_npz")


def test_synth_output_is_cosmic_web_loadable() -> None:
    """The npz the CLI emits must be loadable by render.cosmic_pipeline.
    Round-trip via load_desi_npz."""
    from render.cosmic_pipeline import load_desi_npz
    from render.desi_fetch import main

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "test.npz"
        main([
            "synth", "--out", str(out),
            "--n-particles", "2000", "--n-clusters", "20",
            "--box", "500", "--seed", "3",
        ])
        pos, tracer, mag, zs = load_desi_npz(out)
        assert pos.shape == (2000, 3)
        assert tracer.dtype == np.int32
        assert mag.dtype == np.float32
        assert zs is not None
        assert zs.shape == (2000,)
    print("ok: test_synth_output_is_cosmic_web_loadable")


def test_cli_synth_reproducible_with_same_seed() -> None:
    from render.desi_fetch import main

    with tempfile.TemporaryDirectory() as d:
        out_a = Path(d) / "a.npz"
        out_b = Path(d) / "b.npz"
        for o in (out_a, out_b):
            main([
                "synth", "--out", str(o),
                "--n-particles", "1000", "--n-clusters", "10",
                "--seed", "42",
            ])
        a = np.load(out_a)["positions"]
        b = np.load(out_b)["positions"]
        assert np.array_equal(a, b)
    print("ok: test_cli_synth_reproducible_with_same_seed")


def test_download_mode_errors_cleanly_without_astropy_or_url() -> None:
    """Download mode should refuse gracefully when astropy isn't
    available, OR when --url is missing entirely."""
    from render.desi_fetch import main

    try:
        import astropy  # noqa: F401
        astropy_available = True
    except ImportError:
        astropy_available = False

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "x.npz"
        if astropy_available:
            # astropy is here but URL is empty -> rc != 0.
            rc = main([
                "download", "--out", str(out), "--url", "",
            ])
            assert rc != 0
        else:
            # missing astropy entirely -> rc != 0.
            rc = main([
                "download", "--out", str(out), "--url", "http://example.com/x.fits",
            ])
            assert rc != 0
    print("ok: test_download_mode_errors_cleanly_without_astropy_or_url")


def main() -> int:
    test_synth_cli_writes_valid_npz()
    test_synth_output_is_cosmic_web_loadable()
    test_cli_synth_reproducible_with_same_seed()
    test_download_mode_errors_cleanly_without_astropy_or_url()
    print("\nall desi_fetch tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
