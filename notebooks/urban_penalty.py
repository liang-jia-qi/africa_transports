"""
urban_penalty.py
================
Computes urban-penalty travel times for each road edge by sampling
the Africapolis density-scenario TIF rasters.

For each scenario × year, every edge gets:
  time_new = l_urban / v_urban + l_rural / v_intercity

where l_urban = portion of the edge crossing pixels with value > 0.

Outputs
-------
  input/AfricaNetworkEdges_withUrban.csv
    Original edges CSV + new time columns:
      timeU_high2050, timeU_cons2050, timeU_low2050
      timeU_high2020, timeU_cons2020, timeU_low2020   (optional)

Usage
-----
  python urban_penalty.py

  # Only 2050 (faster):
  python urban_penalty.py --years 2050

  # Custom speeds (km/h):
  python urban_penalty.py --v_urban 30 --v_intercity 60
"""

import os
import glob
import re
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge as rio_merge

warnings.filterwarnings("ignore")

# ── defaults ────────────────────────────────────────────────────────────────
TIF_DIR      = "input/Density_Scenarios_Paper"
INPUT_EDGES  = "input/AfricaNetworkEdges.csv"
INPUT_NODES  = "input/AfricaNetworkNodes.csv"
OUTPUT_EDGES = "input/AfricaNetworkEdges_withUrban.csv"

V_URBAN      = 30.0   # km/h inside urban pixels
V_INTERCITY  = 60.0   # km/h outside urban pixels

SCENARIOS    = ["highdensity", "consdensity", "lowdensity"]
YEARS        = [2050]          # add 2020 if you want the current-footprint update

# Sampling: how many points along each edge to test for urban coverage
# More points = more accurate but slower.  ~1 point per 500m is sufficient.
SAMPLE_SPACING_KM = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# 1. BUILD TIF INDEX
# ══════════════════════════════════════════════════════════════════════════════

def build_tif_index(tif_dir: str) -> pd.DataFrame:
    """
    Scan all .tif files and parse their scenario / year / country from filename.

    Filename pattern: {CountryName}_low80_{scenario}_{year}.tif
    """
    records = []
    for f in glob.glob(os.path.join(tif_dir, "**/*.tif"), recursive=True):
        name = os.path.splitext(os.path.basename(f))[0]
        folder = os.path.basename(os.path.dirname(f))

        # extract year (4 digits at end)
        m_year = re.search(r"(\d{4})$", name)
        year   = int(m_year.group(1)) if m_year else None

        # extract scenario
        scenario = None
        for kw in SCENARIOS:
            if kw.lower() in name.lower():
                scenario = kw
                break

        if year and scenario:
            records.append({
                "path":     f,
                "folder":   folder,
                "scenario": scenario,
                "year":     year,
            })

    df = pd.DataFrame(records)
    print(f"TIF index: {len(df)} files  |  "
          f"scenarios: {df['scenario'].unique().tolist()}  |  "
          f"years: {sorted(df['year'].unique().tolist())}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD VIRTUAL MERGED RASTER FOR ONE SCENARIO × YEAR
# ══════════════════════════════════════════════════════════════════════════════

class MergedRaster:
    """
    Lazily opens and spatially queries a set of rasterio datasets
    without loading them all into RAM.

    For each (lon, lat) query point, finds which open dataset covers it
    and samples the pixel value.  Returns 0 for points not covered.
    """

    def __init__(self, paths: list):
        self._datasets = [rasterio.open(p) for p in paths]
        # Pre-compute bounding boxes for fast dispatch
        self._bounds = [
            (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
            for ds in self._datasets
        ]

    def sample_points(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """
        Return pixel values at arrays of (lon, lat) coordinates.
        Returns 0.0 for points outside all rasters or on nodata.
        """
        values = np.zeros(len(lons), dtype=np.float32)

        for i, (lon, lat) in enumerate(zip(lons, lats)):
            for j, (l, b, r, t) in enumerate(self._bounds):
                if l <= lon <= r and b <= lat <= t:
                    ds = self._datasets[j]
                    try:
                        row, col = ds.index(lon, lat)
                        v = ds.read(1, window=rasterio.windows.Window(col, row, 1, 1))
                        px = float(v[0, 0])
                        nodata = ds.nodata
                        if nodata is not None and px == nodata:
                            px = 0.0
                        values[i] = px
                    except Exception:
                        pass
                    break   # first matching dataset wins

        return values

    def close(self):
        for ds in self._datasets:
            ds.close()


# ══════════════════════════════════════════════════════════════════════════════
# 3. VECTORISED EDGE SAMPLER  (much faster than per-point loop)
# ══════════════════════════════════════════════════════════════════════════════

def sample_edges_vectorised(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    raster: MergedRaster,
    spacing_km: float = SAMPLE_SPACING_KM,
) -> np.ndarray:
    """
    For every edge, place sample points along the straight-line segment at
    `spacing_km` intervals and query the raster.

    Returns an array of shape (n_edges,) with the fraction of points that
    fall on urban pixels (value > 0).
    """
    # Build node coordinate lookup
    node_xy = nodes.set_index("name")[["x", "y"]]   # x=lon, y=lat

    n_edges      = len(edges)
    urban_fracs  = np.zeros(n_edges, dtype=np.float32)

    # Collect ALL sample points across all edges first, then batch-query raster
    all_lons, all_lats, edge_ids, n_samples_per_edge = [], [], [], []
    
    for i, row in edges.iterrows():
        try:
            x0, y0 = node_xy.loc[row['from'], ["x", "y"]]  # 用方括号
            x1, y1 = node_xy.loc[row['to'], ["x", "y"]]
            edge_len_km = row['l']
        except KeyError:
            n_samples_per_edge.append(0)
            continue

        n_pts = max(2, int(np.ceil(edge_len_km / spacing_km)) + 1)

        lons = np.linspace(x0, x1, n_pts)
        lats = np.linspace(y0, y1, n_pts)

        all_lons.extend(lons)
        all_lats.extend(lats)
        edge_ids.extend([i] * n_pts)
        n_samples_per_edge.append(n_pts)

    print(f"  Sampling {len(all_lons):,} points across {n_edges:,} edges …")

    all_lons = np.array(all_lons, dtype=np.float64)
    all_lats = np.array(all_lats, dtype=np.float64)
    edge_ids = np.array(edge_ids, dtype=np.int32)

    # Batch query  ── process in chunks to avoid huge memory spikes
    CHUNK = 50_000
    all_values = np.zeros(len(all_lons), dtype=np.float32)
    for start in range(0, len(all_lons), CHUNK):
        end = min(start + CHUNK, len(all_lons))
        all_values[start:end] = raster.sample_points(
            all_lons[start:end], all_lats[start:end]
        )
        if (start // CHUNK) % 10 == 0:
            print(f"    {100*end/len(all_lons):.0f}%", flush=True)

    is_urban = (all_values > 0).astype(np.float32)

    # Aggregate per edge
    for i in range(n_edges):
        mask = edge_ids == i
        if mask.sum() > 0:
            urban_fracs[i] = is_urban[mask].mean()

    return urban_fracs


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPUTE NEW TIME COLUMN
# ══════════════════════════════════════════════════════════════════════════════

def urban_fraction_to_time(
    edges: pd.DataFrame,
    urban_fracs: np.ndarray,
    v_urban: float = V_URBAN,
    v_intercity: float = V_INTERCITY,   # 保留参数签名但不再使用
) -> np.ndarray:
    """
    修正版：城外部分保留coauthor原始速度，只有城内部分降速到v_urban。

    原始 time 列已经按道路类型分级（motorway=100, trunk=60, primary=40 km/h），
    不能用固定的 v_intercity 覆盖城外部分，否则primary边会被"提速"导致时间变短。

    公式：
      time_new = time_rural_orig + time_urban_new
               = time * (1 - urban_frac)        # 城外：保持原始时间
               + l * urban_frac / v_urban * 60  # 城内：统一降到v_urban

    保证：urban_frac > 0 且 implied_speed > v_urban 时，time_new > time_original
    """
    l            = edges["l"].values      # km
    time_orig    = edges["time"].values   # 分钟，含道路类型分级速度

    time_rural   = time_orig * (1 - urban_fracs)          # 城外：原速不变
    time_urban   = l * urban_fracs / v_urban * 60.0       # 城内：降到v_urban

    time_new = time_rural + time_urban

    # 安全检查：确保不会比原始时间更短
    time_new = np.maximum(time_new, time_orig)

    return time_new


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(years=None, v_urban=V_URBAN, v_intercity=V_INTERCITY):
    if years is None:
        years = YEARS

    print("Loading network data …")
    nodes = pd.read_csv(INPUT_NODES)
    edges = pd.read_csv(INPUT_EDGES)
    print(f"  Nodes: {len(nodes):,}  |  Edges: {len(edges):,}")

    print("\nBuilding TIF index …")
    tif_index = build_tif_index(TIF_DIR)

    edges_out = edges.copy()

    for year in years:
        for scenario in SCENARIOS:
            col_name = f"timeU_{scenario[:4]}{year}"   # e.g. timeU_high2050
            print(f"\n{'─'*60}")
            print(f"Scenario: {scenario}  |  Year: {year}  →  column: {col_name}")
            print(f"{'─'*60}")

            # Get all TIF files for this scenario + year
            subset = tif_index[
                (tif_index["scenario"] == scenario) &
                (tif_index["year"]     == year)
            ]
            if len(subset) == 0:
                print(f"  ⚠ No TIF files found — skipping")
                continue
            print(f"  Found {len(subset)} TIF tiles")

            # Open merged raster
            raster = MergedRaster(subset["path"].tolist())

            # Sample edges
            urban_fracs = sample_edges_vectorised(nodes, edges, raster)

            # Compute new time
            time_new = urban_fraction_to_time(edges_out, urban_fracs, v_urban, v_intercity)
            edges_out[col_name] = time_new

            # Quick sanity check
            n_affected = (urban_fracs > 0).sum()
            print(f"  Edges with any urban portion: {n_affected:,} / {len(edges):,}")
            print(f"  Time stats (min): "
                  f"mean={time_new.mean():.1f}  "
                  f"median={np.median(time_new):.1f}  "
                  f"max={time_new.max():.1f}")
            print(f"  vs original 'time': "
                  f"mean={edges['time'].mean():.1f}  "
                  f"median={edges['time'].median():.1f}")

            raster.close()

    # Save
    edges_out.to_csv(OUTPUT_EDGES, index=False)
    new_cols = [c for c in edges_out.columns if c.startswith("timeU_")]
    print(f"\nSaved: {OUTPUT_EDGES}")
    print(f"New columns added: {new_cols}")
    return edges_out


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years",        nargs="+", type=int,   default=YEARS)
    parser.add_argument("--v_urban",      type=float, default=V_URBAN)
    parser.add_argument("--v_intercity",  type=float, default=V_INTERCITY)
    args = parser.parse_args()

    main(years=args.years, v_urban=args.v_urban, v_intercity=args.v_intercity)
