"""
Preprocess NYC Yellow Taxi parquet into clean 2-column (x, y) CSVs.

Output format:
    x,y           <- header
    -73.98,40.76
    ...

The output is consumable by:
  * sklearn (pandas.read_csv then .to_numpy())
  * this project's Spark pipeline (scripts/run_experiment.py --data ...)
  * scripts/visualize_taxi.py

Usage:
    # defaults: sizes 2k/10k/20k/50k -> data/taxi_<N>.csv
    python scripts/preprocess_taxi.py

    # custom sizes / output dir
    python scripts/preprocess_taxi.py --sizes 5000 30000 --out data/

    # single custom sample
    python scripts/preprocess_taxi.py --sizes 100000 --out data/ --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# --- NYC bounding box: 5 boroughs (drops GPS=0,0 and far outliers) ---
NYC_BBOX = dict(lon_min=-74.05, lon_max=-73.75,
                lat_min=40.58,  lat_max=40.90)


def clean(parquet_path: Path, bbox: dict) -> pd.DataFrame:
    """Read parquet, keep pickup lon/lat inside bbox."""
    df = pd.read_parquet(parquet_path, columns=["Start_Lon", "Start_Lat"])
    n0 = len(df)
    df = df.rename(columns={"Start_Lon": "x", "Start_Lat": "y"}).dropna()
    df = df[df["x"].between(bbox["lon_min"], bbox["lon_max"])
            & df["y"].between(bbox["lat_min"], bbox["lat_max"])]
    kept = len(df) / n0 * 100
    print(f"[clean] kept {len(df):,} / {n0:,} rows inside NYC bbox ({kept:.1f}%)")
    return df.reset_index(drop=True)


def sample_and_save(df: pd.DataFrame, n: int, out_dir: Path, seed: int) -> Path:
    n = min(n, len(df))
    subset = df.sample(n=n, random_state=seed)[["x", "y"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"taxi_{n}.csv"
    subset.to_csv(out, index=False)
    print(f"[save] {out}  ({n:,} rows)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--parquet", type=Path,
                    default=Path("yellow_tripdata_2009-01.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[2_000, 10_000, 20_000, 50_000],
                    help="sample sizes to export (default: 2k 10k 20k 50k)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.parquet.exists():
        raise SystemExit(f"[error] parquet not found: {args.parquet}")

    df = clean(args.parquet, NYC_BBOX)
    for n in args.sizes:
        sample_and_save(df, n, args.out, args.seed)

    print(f"\nDone. Feed these CSVs into sklearn or "
          f"`python scripts/run_experiment.py --data ...`")


if __name__ == "__main__":
    main()
