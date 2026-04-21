"""
Visualize DBSCAN vs HDBSCAN on a preprocessed NYC taxi CSV.

Two modes (pick one with --mode):

    sweep : 2x3 grid of DBSCAN at multiple eps values + one HDBSCAN panel.
            Plain scatter, no basemap. Matches sweep_eps_<N>.png style.

    map   : 1xK panel of a few DBSCAN eps values + HDBSCAN, drawn on top
            of an OpenStreetMap basemap with NYC landmark annotations.
            Matches map_eps_compare_<N>.png style.

Input CSV must have columns `x, y` (lon, lat) -- produced by preprocess_taxi.py.

Usage:
    # sweep mode (no map)
    python scripts/visualize_taxi.py --file data/taxi_10000.csv --mode sweep

    # map mode (with OSM basemap + landmarks)
    python scripts/visualize_taxi.py --file data/taxi_10000.csv --mode map

    # customize eps values / HDBSCAN params / output dir
    python scripts/visualize_taxi.py --file data/taxi_10000.csv --mode map \
        --eps 0.0005 0.001 0.002 --min-cluster-size 30 --out scripts/out_taxi/
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from PIL import Image
from sklearn.cluster import DBSCAN, HDBSCAN

# ---------- NYC constants ----------
NYC_BBOX = (-74.05, -73.75, 40.58, 40.90)     # lon_min, lon_max, lat_min, lat_max
NYC_LANDMARKS = [
    ("JFK",           -73.7781, 40.6413),
    ("LaGuardia",     -73.8740, 40.7769),
    ("Times Sq",      -73.9855, 40.7580),
    ("Grand Central", -73.9772, 40.7527),
    ("Penn Station",  -73.9904, 40.7506),
    ("Wall St",       -74.0089, 40.7074),
    ("Central Park",  -73.9654, 40.7829),
    ("Williamsburg",  -73.9571, 40.7081),
    ("Brooklyn Hts",  -73.9961, 40.6959),
]


# ---------- clustering ----------
def run(name, model, X):
    t = time.perf_counter()
    labels = model.fit_predict(X)
    dt = time.perf_counter() - t
    nc = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean() * 100
    if nc > 0:
        _, counts = np.unique(labels[labels != -1], return_counts=True)
        biggest = counts.max() / len(labels) * 100
    else:
        biggest = 0.0
    print(f"[{name:<24}] {dt:5.2f}s  clusters={nc:4d}  "
          f"noise={noise_pct:5.1f}%  biggest={biggest:5.1f}%")
    return labels, dict(clusters=nc, noise=noise_pct, biggest=biggest, time=dt)


# ---------- OSM tile fetching (for map mode) ----------
def _deg2num(lat, lon, z):
    lat_rad = math.radians(lat); n = 2.0 ** z
    return ((lon + 180.0) / 360.0 * n,
            (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)


def _num2deg(x, y, z):
    n = 2.0 ** z
    return (math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n)))),
            x / n * 360.0 - 180.0)


def fetch_basemap(bbox, zoom, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    lon_min, lon_max, lat_min, lat_max = bbox
    x0, y1 = _deg2num(lat_max, lon_min, zoom)
    x1, y0 = _deg2num(lat_min, lon_max, zoom)
    tx0, tx1 = int(math.floor(x0)), int(math.floor(x1))
    ty0, ty1 = int(math.floor(y1)), int(math.floor(y0))
    w, h = (tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256
    img = Image.new("RGB", (w, h))
    headers = {"User-Agent": "distributed-clustering-course/1.0"}
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            cache = cache_dir / f"{zoom}_{tx}_{ty}.png"
            if not cache.exists():
                r = requests.get(
                    f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png",
                    headers=headers, timeout=15)
                r.raise_for_status()
                cache.write_bytes(r.content); time.sleep(0.1)
            img.paste(Image.open(cache), ((tx - tx0) * 256, (ty - ty0) * 256))
    lat_top, lon_left  = _num2deg(tx0,     ty0,     zoom)
    lat_bot, lon_right = _num2deg(tx1 + 1, ty1 + 1, zoom)
    return img, (lon_left, lon_right, lat_bot, lat_top)


# ---------- drawing helpers ----------
def _cluster_size_rank_colors(labels):
    """Biggest cluster gets brightest color (so small clusters stay visible
    even when one giant cluster dominates)."""
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    order = unique[np.argsort(-counts)]
    rank = {c: i for i, c in enumerate(order)}
    cmap = plt.get_cmap("tab20", 20)
    colors = np.full((len(labels), 4), [0.7, 0.7, 0.7, 0.2])
    for c, i in rank.items():
        colors[labels == c] = cmap(i % 20)
    return colors


def _add_landmarks(ax, bbox, fontsize=8, marker_size=60):
    lon_min, lon_max, lat_min, lat_max = bbox
    for name, lon, lat in NYC_LANDMARKS:
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            ax.scatter([lon], [lat], s=marker_size, c="red", marker="*",
                       edgecolors="white", linewidths=1.1, zorder=6)
            ax.annotate(name, (lon, lat), xytext=(5, 5),
                        textcoords="offset points", fontsize=fontsize,
                        bbox=dict(boxstyle="round,pad=0.18",
                                  fc="white", ec="none", alpha=0.8),
                        zorder=7)


def _panel_plain(ax, X, labels, title, bbox):
    lon_min, lon_max, lat_min, lat_max = bbox
    mask = labels != -1
    ax.scatter(X[~mask, 0], X[~mask, 1], s=1.5, c="lightgray", alpha=0.4)
    if mask.any():
        ax.scatter(X[mask, 0], X[mask, 1], s=1.5, c=labels[mask],
                   cmap="tab20", alpha=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_aspect("equal")
    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)


def _panel_map(ax, X, labels, title, basemap, extent, bbox):
    lon_min, lon_max, lat_min, lat_max = bbox
    ax.imshow(basemap, extent=extent, origin="upper", alpha=0.5, zorder=0)
    colors = _cluster_size_rank_colors(labels)
    noise = labels == -1
    ax.scatter(X[noise, 0], X[noise, 1], s=2, c="black", alpha=0.12, zorder=1)
    ax.scatter(X[~noise, 0], X[~noise, 1], s=3, c=colors[~noise],
               alpha=0.85, zorder=2)
    _add_landmarks(ax, bbox)
    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)


# ---------- main modes ----------
def render_sweep(X, args, out_dir: Path):
    """2x3 grid: 5 DBSCAN eps + 1 HDBSCAN (no basemap)."""
    results = []
    for eps in args.eps_sweep:
        labels, meta = run(
            f"DBSCAN eps={eps}",
            DBSCAN(eps=eps, min_samples=args.min_samples, n_jobs=-1), X)
        results.append((eps, labels, meta))

    hdb_labels, hdb_meta = run(
        f"HDBSCAN mc={args.min_cluster_size}",
        HDBSCAN(min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples, n_jobs=-1), X)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11)); axes = axes.flatten()
    for ax, (eps, labels, m) in zip(axes, results):
        title = (f"DBSCAN  eps={eps}  (~{int(eps*111_000)} m)\n"
                 f"clusters={m['clusters']}, noise={m['noise']:.1f}%, "
                 f"{m['time']:.2f}s")
        _panel_plain(ax, X, labels, title, NYC_BBOX)
    _panel_plain(axes[5], X, hdb_labels,
                 f"HDBSCAN  min_cluster={args.min_cluster_size}\n"
                 f"clusters={hdb_meta['clusters']}, "
                 f"noise={hdb_meta['noise']:.1f}%, {hdb_meta['time']:.2f}s",
                 NYC_BBOX)
    fig.suptitle(f"DBSCAN eps sweep vs HDBSCAN  |  "
                 f"NYC Yellow Taxi pickups, n={len(X):,}",
                 fontsize=14, y=1.00)
    plt.tight_layout()
    out = out_dir / f"sweep_eps_{len(X)}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[plot] saved {out}")


def render_map(X, args, out_dir: Path):
    """1xK panel: selected DBSCAN eps + HDBSCAN, all on an OSM basemap."""
    basemap, extent = fetch_basemap(NYC_BBOX, zoom=args.zoom,
                                    cache_dir=out_dir / "tiles")
    panels = []
    for eps in args.eps_map:
        labels, m = run(
            f"DBSCAN eps={eps}",
            DBSCAN(eps=eps, min_samples=args.min_samples, n_jobs=-1), X)
        panels.append((f"DBSCAN eps={eps} (~{int(eps*111_000)} m)\n"
                       f"clusters={m['clusters']}, noise={m['noise']:.1f}%, "
                       f"biggest={m['biggest']:.1f}%",
                       labels))

    hdb_labels, hdb_meta = run(
        f"HDBSCAN mc={args.min_cluster_size}",
        HDBSCAN(min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples, n_jobs=-1), X)
    panels.append((f"HDBSCAN min_cluster={args.min_cluster_size}\n"
                   f"clusters={hdb_meta['clusters']}, "
                   f"noise={hdb_meta['noise']:.1f}%, "
                   f"biggest={hdb_meta['biggest']:.1f}%",
                   hdb_labels))

    k = len(panels)
    fig, axes = plt.subplots(1, k, figsize=(6 * k, 8))
    if k == 1:
        axes = [axes]
    for ax, (title, labels) in zip(axes, panels):
        _panel_map(ax, X, labels, title, basemap, extent, NYC_BBOX)

    fig.suptitle(f"DBSCAN vs HDBSCAN on NYC taxi pickups over OSM basemap  "
                 f"(n={len(X):,})",
                 fontsize=14, y=1.00)
    plt.tight_layout()
    out = out_dir / f"map_eps_compare_{len(X)}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[plot] saved {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--file", type=Path, required=True,
                    help="CSV produced by preprocess_taxi.py")
    ap.add_argument("--mode", choices=["sweep", "map"], required=True)
    ap.add_argument("--out", type=Path, default=Path("scripts/out_taxi"))

    # eps values for the two modes
    ap.add_argument("--eps-sweep", type=float, nargs="+",
                    default=[0.0002, 0.0005, 0.001, 0.002, 0.005])
    ap.add_argument("--eps-map", type=float, nargs="+",
                    default=[0.0005, 0.001, 0.002])

    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--min-cluster-size", type=int, default=30)
    ap.add_argument("--zoom", type=int, default=11,
                    help="OSM zoom level for map mode (default: 11)")
    args = ap.parse_args()

    if not args.file.exists():
        raise SystemExit(f"[error] CSV not found: {args.file}")

    X = pd.read_csv(args.file)[["x", "y"]].to_numpy()
    print(f"[data] loaded {len(X):,} points from {args.file}")

    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode == "sweep":
        render_sweep(X, args, args.out)
    else:
        render_map(X, args, args.out)


if __name__ == "__main__":
    main()
