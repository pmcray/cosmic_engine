"""DESI catalog → cosmic_web .npz converter CLI.

Three modes:

  synth: generate a procedural cosmic-web catalog. No network or
         astropy needed. Useful for local development and CI.

    python -m render.desi_fetch synth --out catalogs/web.npz \\
        --n-particles 200000 --n-clusters 600 --box 3000 --seed 42

  list: print the known public DESI EDR / DR1 LSS clustering catalog
         URLs hosted at LBNL so you don't have to remember them.

    python -m render.desi_fetch list

  download: pull a public DESI FITS catalog from a URL, convert the
         RA/Dec/z columns to comoving Cartesian using the user-
         configurable w0wa cosmology, and emit the npz the
         cosmic_web renderer consumes. Requires astropy.

    python -m render.desi_fetch download --out catalogs/bgs.npz \\
        --tracer BGS --max-rows 200000 \\
        --url https://data.desi.lbl.gov/public/edr/vac/edr/lss/v2.0/LSScats/BGS_BRIGHT-21.5_NGC_clustering.dat.fits \\
        --H0 67.4 --omega-m 0.315 --w0 -0.838 --wa -0.62

The output .npz format is what render.cosmic_web.CosmicWebEngine
expects when given `catalog_path`:
    positions  (N, 3) float32   Mpc comoving, origin = observer
    tracer     (N,)   int32     0=BGS 1=LRG 2=ELG 3=QSO
    magnitudes (N,)   float32   emission weight
    redshifts  (N,)   float32   per-object z (optional, but recommended)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from render.cosmic_pipeline import (
    DEFAULT_H0,
    DEFAULT_OMEGA_M,
    DEFAULT_W0,
    DEFAULT_WA,
    TRACER_INDEX,
    classify_by_redshift,
    radec_z_to_cartesian,
    synthesize_cosmic_web,
)


# Known public DESI catalog URLs hosted at LBNL.
# Confirmed against https://data.desi.lbl.gov/public/edr/ and DR1
# index listings. Files are large (multi-GB); use --max-rows to
# subsample for proxy renders.
KNOWN_DESI_URLS = {
    # Early Data Release (EDR) v2.0 clustering catalogs.
    "EDR_BGS_NGC": "https://data.desi.lbl.gov/public/edr/vac/edr/lss/v2.0/LSScats/BGS_BRIGHT-21.5_NGC_clustering.dat.fits",
    "EDR_BGS_SGC": "https://data.desi.lbl.gov/public/edr/vac/edr/lss/v2.0/LSScats/BGS_BRIGHT-21.5_SGC_clustering.dat.fits",
    "EDR_LRG_NGC": "https://data.desi.lbl.gov/public/edr/vac/edr/lss/v2.0/LSScats/LRG_NGC_clustering.dat.fits",
    "EDR_LRG_SGC": "https://data.desi.lbl.gov/public/edr/vac/edr/lss/v2.0/LSScats/LRG_SGC_clustering.dat.fits",
    "EDR_ELG_NGC": "https://data.desi.lbl.gov/public/edr/vac/edr/lss/v2.0/LSScats/ELG_LOPnotqso_NGC_clustering.dat.fits",
    "EDR_ELG_SGC": "https://data.desi.lbl.gov/public/edr/vac/edr/lss/v2.0/LSScats/ELG_LOPnotqso_SGC_clustering.dat.fits",
    "EDR_QSO_NGC": "https://data.desi.lbl.gov/public/edr/vac/edr/lss/v2.0/LSScats/QSO_NGC_clustering.dat.fits",
    "EDR_QSO_SGC": "https://data.desi.lbl.gov/public/edr/vac/edr/lss/v2.0/LSScats/QSO_SGC_clustering.dat.fits",
    # DR1 LSS catalogs (release pending public listing; same URL
    # convention applies). Update once DR2 is published in April 2026.
    "DR1_BGS_NGC": "https://data.desi.lbl.gov/public/dr1/vac/dr1/lss/iron/LSScats/v1/BGS_BRIGHT-21.5_NGC_clustering.dat.fits",
    "DR1_LRG_NGC": "https://data.desi.lbl.gov/public/dr1/vac/dr1/lss/iron/LSScats/v1/LRG_NGC_clustering.dat.fits",
    "DR1_ELG_NGC": "https://data.desi.lbl.gov/public/dr1/vac/dr1/lss/iron/LSScats/v1/ELG_LOPnotqso_NGC_clustering.dat.fits",
    "DR1_QSO_NGC": "https://data.desi.lbl.gov/public/dr1/vac/dr1/lss/iron/LSScats/v1/QSO_NGC_clustering.dat.fits",
}


def cmd_list(args: argparse.Namespace) -> int:
    print("Known public DESI LSS clustering catalog URLs:\n")
    width = max(len(k) for k in KNOWN_DESI_URLS) + 2
    for label, url in KNOWN_DESI_URLS.items():
        print(f"  {label:<{width}} {url}")
    print(
        "\nPick one and pass it to `download --url <URL>`. Default cosmology "
        "(w0=-0.838, wa=-0.62) is the DESI DR2 best fit; override with "
        "--H0, --omega-m, --w0, --wa.\n"
        "\nFor large catalogs use --max-rows to subsample (e.g. 200000)."
    )
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    positions, tracer, magnitudes, redshifts = synthesize_cosmic_web(
        n_particles=args.n_particles,
        box_size_mpc=args.box,
        n_clusters=args.n_clusters,
        seed=args.seed,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        positions=positions.astype(np.float32),
        tracer=tracer.astype(np.int32),
        magnitudes=magnitudes.astype(np.float32),
        redshifts=redshifts.astype(np.float32),
    )
    print(
        f"synthesised {len(positions)} galaxies to {args.out}: "
        f"{ {n: int((tracer == i).sum()) for n, i in TRACER_INDEX.items()} }"
    )
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    try:
        from astropy.io import fits
    except ImportError:
        print(
            "download mode requires astropy. Install with `pip install astropy`.",
            file=sys.stderr,
        )
        return 1

    if not args.url:
        print("download mode requires --url", file=sys.stderr)
        return 1

    # Either fetch the file or use a local path.
    if args.url.startswith(("http://", "https://")):
        print(f"fetching {args.url} ...", file=sys.stderr)
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
            tmp_path = tf.name
        try:
            urllib.request.urlretrieve(args.url, tmp_path)
            fits_path = tmp_path
        except Exception as e:
            print(f"fetch failed: {e}", file=sys.stderr)
            os.unlink(tmp_path)
            return 2
    else:
        fits_path = args.url  # treat as local path

    try:
        with fits.open(fits_path) as f:
            tbl = f[1].data
            cols = {c.lower(): c for c in tbl.columns.names}

            def _col(*candidates):
                for c in candidates:
                    if c.lower() in cols:
                        return tbl[cols[c.lower()]]
                raise KeyError(f"no column matched {candidates}")

            ra = _col("RA", "TARGET_RA")
            dec = _col("DEC", "TARGET_DEC")
            z = _col("Z", "Z_NOT4CLUS", "REDSHIFT")
            try:
                mag = _col("FLUX_R", "FLUX_G", "R_MAG", "MAG_R")
            except KeyError:
                mag = np.ones_like(z, dtype=np.float32)
    finally:
        if args.url.startswith(("http://", "https://")):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # Subsample if requested.
    if args.max_rows and len(z) > args.max_rows:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(z), args.max_rows, replace=False)
        ra, dec, z, mag = ra[idx], dec[idx], z[idx], mag[idx]

    # Skip bad redshifts.
    finite = np.isfinite(z) & (z > 0.0) & (z < 5.0)
    ra = np.asarray(ra)[finite].astype(np.float64)
    dec = np.asarray(dec)[finite].astype(np.float64)
    z = np.asarray(z)[finite].astype(np.float64)
    mag = np.asarray(mag)[finite].astype(np.float64)

    print(f"converting {len(z)} rows to Cartesian (w0={args.w0} wa={args.wa})...",
          file=sys.stderr)
    positions = radec_z_to_cartesian(
        ra, dec, z,
        H0=args.H0, Omega_m=args.omega_m, w0=args.w0, wa=args.wa,
    ).astype(np.float32)

    # If the user supplied a specific tracer label, assign it to every
    # row; otherwise classify by redshift.
    if args.tracer:
        idx = TRACER_INDEX.get(args.tracer.upper())
        if idx is None:
            print(f"unknown tracer label {args.tracer!r}", file=sys.stderr)
            return 3
        tracer = np.full(len(z), idx, dtype=np.int32)
    else:
        tracer = classify_by_redshift(z).astype(np.int32)

    # Magnitudes -> emission weight: normalise to sensible range.
    mag = np.asarray(mag, dtype=np.float64)
    mag = np.clip(mag, np.percentile(mag, 1), np.percentile(mag, 99))
    mag -= mag.min()
    if mag.max() > 0:
        mag /= mag.max()
    mag = (mag + 0.1).astype(np.float32)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        positions=positions,
        tracer=tracer,
        magnitudes=mag,
        redshifts=z.astype(np.float32),
    )
    print(
        f"wrote {len(z)} galaxies to {args.out}: "
        f"{ {n: int((tracer == i).sum()) for n, i in TRACER_INDEX.items()} }",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="desi_fetch")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_synth = sub.add_parser("synth", help="generate a procedural catalog")
    p_synth.add_argument("--out", required=True, help="output .npz path")
    p_synth.add_argument("--n-particles", type=int, default=200_000)
    p_synth.add_argument("--n-clusters", type=int, default=600)
    p_synth.add_argument("--box", type=float, default=3000.0,
                         help="box edge in Mpc")
    p_synth.add_argument("--seed", type=int, default=0)
    p_synth.set_defaults(func=cmd_synth)

    p_list = sub.add_parser("list", help="print known public DESI catalog URLs")
    p_list.set_defaults(func=cmd_list)

    p_dl = sub.add_parser("download", help="convert a FITS catalog to .npz")
    p_dl.add_argument("--out", required=True, help="output .npz path")
    p_dl.add_argument("--url", required=True,
                      help="HTTP(S) URL or local FITS path")
    p_dl.add_argument("--tracer", default=None,
                      help="force tracer label (BGS|LRG|ELG|QSO); "
                           "otherwise classify by redshift")
    p_dl.add_argument("--max-rows", type=int, default=None,
                      help="random subsample if catalog is larger")
    p_dl.add_argument("--seed", type=int, default=0)
    p_dl.add_argument("--H0", type=float, default=DEFAULT_H0)
    p_dl.add_argument("--omega-m", type=float, default=DEFAULT_OMEGA_M)
    p_dl.add_argument("--w0", type=float, default=DEFAULT_W0)
    p_dl.add_argument("--wa", type=float, default=DEFAULT_WA)
    p_dl.set_defaults(func=cmd_download)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
