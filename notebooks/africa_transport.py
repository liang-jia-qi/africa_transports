"""
Africa Transport Network Analysis
Python translation of SlowNetworks20250715.R

Gravity-weighted edge betweenness for 2015 and 2050,
with support for multiple time-weight scenarios (baseline, urban penalty variants).

Inputs (relative to INPUT_DIR):
  AfricaNetworkNodes.csv
  AfricaNetworkEdges.csv
  Africapolis_2050.csv

Outputs (relative to OUTPUT_DIR):
  GravityPairs.pkl
  NetworkEdgeBetween_<scenario>.pkl
  Figures/Congestion_<scenario>_<year>.png
  Figures/LagosCongestion_<scenario>.png
"""

import os
import pickle
import numpy as np
import pandas as pd
import igraph as ig
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.use("Agg")

# ── paths ──────────────────────────────────────────────────────────────────────
INPUT_DIR  = "input"
OUTPUT_DIR = "output"
FIG_DIR    = os.path.join(OUTPUT_DIR, "Figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

# ── parameters ─────────────────────────────────────────────────────────────────
GRAVITY_EXPONENT = 2.8     # distance-decay exponent β
N_TOP_PAIRS      = 25_000  # top gravity pairs used for betweenness
NSTEPS           = 25_000  # alias kept for clarity

# Time-weight columns available in AfricaNetworkEdges.csv:
#   "time"    – intercity speeds only (no urban penalty)
#   "timeU"   – with urban area speed penalty (existing 2015 city footprint)
#   "timeUCB" – urban penalty with cross-border adjustment
# The Africapolis shapefile scenarios (High/Consolidated/Low density) will
# add further columns once GIS intersection is done (see urban_penalty.py).
# For now we run baseline + the two pre-computed urban columns.
TIME_SCENARIOS = {
    "baseline":        "time",
    "urban_2015":      "timeU",
    "urban_CB_2015":   "timeUCB",
    "urban_high_2050": "timeU_high2050",
    "urban_cons_2050": "timeU_cons2050",
    "urban_low_2050":  "timeU_lowd2050",   # 列名是lowd（来自urban_penalty.py输出）
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data(input_dir: str = INPUT_DIR):
    """Load and merge nodes, edges, and Africapolis 2050 population data."""

    nodes = pd.read_csv(os.path.join(input_dir, "AfricaNetworkNodes.csv"))
    # 优先读包含urban penalty时间列的版本，回退到原始版本
    edges_with_urban = os.path.join(input_dir, "AfricaNetworkEdges_withUrban.csv")
    edges_original   = os.path.join(input_dir, "AfricaNetworkEdges.csv")
    edges = pd.read_csv(edges_with_urban if os.path.exists(edges_with_urban) else edges_original)
    afp   = pd.read_csv(os.path.join(input_dir, "Africapolis_2050.csv"),
                        usecols=["Agglomeration_ID", "Population_2050"])

    # Merge 2050 population onto nodes (left join; missing → 0)
    afp = afp.rename(columns={"Agglomeration_ID": "name",
                               "Population_2050":  "Pop2050"})
    nodes = nodes.merge(afp, on="name", how="left")
    nodes["Pop2050"] = nodes["Pop2050"].fillna(0)

    # Numeric country code (needed for some downstream analyses)
    iso_codes = {iso: i+1 for i, iso in enumerate(sorted(nodes["ISO3"].unique()))}
    nodes["ISOCode"] = nodes["ISO3"].map(iso_codes)

    return nodes, edges


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> ig.Graph:
    """
    Construct an undirected igraph from node/edge dataframes.
    Node attributes: all columns in `nodes` (indexed by 'name').
    Edge attributes: all columns in `edges`.
    """
    # igraph vertex names must be strings
    node_names = nodes["name"].astype(str).tolist()

    G = ig.Graph(directed=False)
    G.add_vertices(len(nodes))
    G.vs["name"] = node_names

    # Attach all node columns as vertex attributes
    for col in nodes.columns:
        G.vs[col] = nodes[col].tolist()

    # Map node name → vertex index for edge lookup
    name_to_idx = {str(n): i for i, n in enumerate(nodes["name"])}

    # Build edge list – drop any edges whose endpoints are not in nodes
    from_idx = edges["from"].astype(str).map(name_to_idx)
    to_idx   = edges["to"].astype(str).map(name_to_idx)
    valid    = from_idx.notna() & to_idx.notna()
    if (~valid).sum() > 0:
        print(f"  ⚠ Dropped {(~valid).sum()} edges with unknown nodes.")

    edge_tuples = list(zip(from_idx[valid].astype(int),
                           to_idx[valid].astype(int)))
    G.add_edges(edge_tuples)

    # Attach all edge columns as edge attributes
    edges_valid = edges[valid].reset_index(drop=True)
    for col in edges_valid.columns:
        G.es[col] = edges_valid[col].tolist()

    print(f"Graph: {G.vcount()} vertices, {G.ecount()} edges")
    return G


# ══════════════════════════════════════════════════════════════════════════════
# 3. GRAVITY MODEL
# ══════════════════════════════════════════════════════════════════════════════

def compute_gravity(G: ig.Graph,
                    pop_attr_2015: str = "Pop2015",
                    pop_attr_2050: str = "Pop2050",
                    time_weight:   str = "time",
                    top_n:         int = N_TOP_PAIRS) -> pd.DataFrame:
    """
    Compute pairwise gravity scores for all city pairs.

    Returns a DataFrame sorted by GravM (2015) descending, with columns:
      from, to, GravM (2015), GravM50 (2050)
    Only city nodes (population > 0 in 2015) are considered.

    The distance matrix uses `time_weight` as the edge weight for shortest paths.
    """
    pop2015 = np.array(G.vs[pop_attr_2015], dtype=float)
    pop2050 = np.array(G.vs[pop_attr_2050], dtype=float)

    city_mask = pop2015 > 0
    city_idx  = np.where(city_mask)[0]
    n_cities  = len(city_idx)
    print(f"  Cities with Pop2015 > 0: {n_cities}")

    weights = G.es[time_weight]

    # Shortest-path time matrix (cities × all nodes, then subset)
    print("  Computing distance matrix …")
    D_full = np.array(G.distances(source=city_idx.tolist(),
                                  weights=weights))   # shape (n_cities, n_nodes)
    D = D_full[:, city_idx]                          # shape (n_cities, n_cities)

    # Gravity matrices
    P15 = pop2015[city_idx] / 1000.0
    P50 = pop2050[city_idx] / 1000.0

    Grav15 = np.outer(P15, P15) / (D ** GRAVITY_EXPONENT + 1)
    Grav50 = np.outer(P50, P50) / (D ** GRAVITY_EXPONENT + 1)

    # Lower-triangle only (undirected, no self-loops)
    rows_idx, cols_idx = np.tril_indices(n_cities, k=-1)

    gm_vals  = Grav15[rows_idx, cols_idx]
    gm50_vals = Grav50[rows_idx, cols_idx]

    # Map back to global vertex indices
    from_global = city_idx[cols_idx]
    to_global   = city_idx[rows_idx]

    GM = pd.DataFrame({
        "from":    from_global,
        "to":      to_global,
        "GravM":   gm_vals,
        "GravM50": gm50_vals,
    })

    # Sort by 2015 gravity descending; keep top N for betweenness
    GM = GM.sort_values("GravM", ascending=False).reset_index(drop=True)
    GM_top = GM.head(top_n).copy()

    print(f"  Total pairs: {len(GM):,}  |  Using top {len(GM_top):,}")
    return GM_top


# ══════════════════════════════════════════════════════════════════════════════
# 4. GRAVITY-WEIGHTED EDGE BETWEENNESS
# ══════════════════════════════════════════════════════════════════════════════

def compute_betweenness(G: ig.Graph,
                        GM: pd.DataFrame,
                        time_weight: str = "time") -> ig.Graph:
    """
    Accumulate gravity-weighted edge betweenness onto G.es["Between"]
    (2015 weights) and G.es["Between50"] (2050 weights).

    Shortest paths use `time_weight` as the edge weight.
    """
    n_edges   = G.ecount()
    between   = np.zeros(n_edges)
    between50 = np.zeros(n_edges)
    weights   = G.es[time_weight]

    n_pairs = len(GM)
    report_every = max(1, n_pairs // 20)

    print(f"  Computing betweenness over {n_pairs:,} pairs …")

    from_arr = GM["from"].values
    to_arr   = GM["to"].values
    gm_arr   = GM["GravM"].values
    gm50_arr = GM["GravM50"].values

    for k in range(n_pairs):
        if k % report_every == 0:
            print(f"    {100*k/n_pairs:.0f}%", flush=True)

        paths = G.get_shortest_paths(
            v       = int(from_arr[k]),
            to      = int(to_arr[k]),
            weights = weights,
            output  = "epath",
        )

        edge_ids = paths[0]
        if not edge_ids:
            continue

        edge_ids = np.array(edge_ids, dtype=int)
        between[edge_ids]   += gm_arr[k]
        between50[edge_ids] += gm50_arr[k]

    G.es["Between"]   = between.tolist()
    G.es["Between50"] = between50.tolist()

    print("  Done.")
    return G


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

REGION_COLORS = {
    "South":   "#FFC107",
    "East":    "#7A98D6",
    "Central": "tomato",
    "West":    "#00CC99",
    "North":   "#CC90C1",
}

def _edge_width(values, scale=2.0):
    arr = np.array(values, dtype=float)
    return (scale * np.log1p(arr)).tolist()


def plot_network(G: ig.Graph,
                 nodes: pd.DataFrame,
                 between_attr: str,
                 pop_attr: str,
                 title: str,
                 out_path: str,
                 figsize: tuple = (20, 20)):
    """Draw the full African road network coloured by region, edge width = congestion."""

    coords = np.column_stack([nodes["x"].values, nodes["y"].values])
    regions = G.vs["Region"]
    node_colors = [REGION_COLORS.get(r, "gray") for r in regions]
    pop = np.array(G.vs[pop_attr], dtype=float)
    vertex_sizes = np.log1p(pop) / 14

    between = G.es[between_attr]
    edge_widths = _edge_width(between)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw edges
    for e, w in zip(G.es, edge_widths):
        s, t = e.source, e.target
        xs = [coords[s, 0], coords[t, 0]]
        ys = [coords[s, 1], coords[t, 1]]
        ax.plot(xs, ys, color="gray", linewidth=w, alpha=0.6, solid_capstyle="round")

    # Draw nodes
    ax.scatter(coords[:, 0], coords[:, 1],
               s=vertex_sizes,
               c=node_colors,
               linewidths=0.3,
               edgecolors="black",
               zorder=3)

    # Legend
    handles = [mpatches.Patch(color=c, label=r)
               for r, c in REGION_COLORS.items()]
    ax.legend(handles=handles, loc="lower left", fontsize=10, framealpha=0.7)
    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_corridor_profile(G: ig.Graph,
                          nodes: pd.DataFrame,
                          origin_name: str,
                          dest_names: list,
                          between_attr: str,
                          title: str,
                          out_path: str,
                          ymax: float = 15_000):
    """
    Bar-profile plot: x = cumulative distance along shortest path,
    height = edge betweenness value.
    """
    name_to_idx = {row["agglosName"]: i for i, row in nodes.iterrows()
                   if pd.notna(row["agglosName"])}

    origin_idx = name_to_idx.get(origin_name)
    if origin_idx is None:
        print(f"  ⚠ City not found: {origin_name}")
        return

    n_dests = len(dest_names)
    colors = ["#FFC107", "#7A98D6", "tomato", "#00CC99", "#CC90C1"]

    fig, axes = plt.subplots(n_dests, 1, figsize=(8, n_dests * 1.8),
                             sharex=False)
    if n_dests == 1:
        axes = [axes]

    for ax, dest_name, color in zip(axes, dest_names, colors):
        dest_idx = name_to_idx.get(dest_name)
        if dest_idx is None:
            ax.set_visible(False)
            continue

        paths = G.get_shortest_paths(
            v       = origin_idx,
            to      = dest_idx,
            weights = G.es["l"],
            output  = "vpath",
        )
        vpath = paths[0]
        if len(vpath) < 2:
            ax.set_visible(False)
            continue

        cum_dist = 0.0
        rects = []
        for i in range(len(vpath) - 1):
            eid   = G.get_eid(vpath[i], vpath[i+1])
            l_val = G.es[eid]["l"]
            b_val = G.es[eid][between_attr]
            rects.append((cum_dist, cum_dist + l_val, b_val))
            cum_dist += l_val

        ax.set_yscale("log")
        ax.set_ylim(1, ymax * 1.1)
        ax.set_xlim(0, 900)
        ax.axis("off")

        for x0, x1, h in rects:
            ax.bar(x=(x0 + x1) / 2,
                   height=max(h, 1),
                   width=x1 - x0,
                   bottom=1,
                   color=color,
                   linewidth=0)

        ax.text(900, 0.8 * ymax, dest_name, ha="right", va="top", fontsize=8)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_scatter_2015_vs_2050(G: ig.Graph,
                               scenario_label: str,
                               out_path: str):
    """Scatter plot of edge congestion in 2015 vs 2050."""
    between   = np.array(G.es["Between"],   dtype=float)
    between50 = np.array(G.es["Between50"], dtype=float)
    mask = between > 0
    x = between[mask]  + 1
    y = between50[mask] + 1

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=3, alpha=0.4, color="#2196F3")
    lim = max(x.max(), y.max()) * 1.1
    ax.plot([1, lim], [1, lim], color="red", linewidth=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Congestion 2015")
    ax.set_ylabel("Congestion 2050")
    ax.set_title(f"Edge congestion 2015 vs 2050  [{scenario_label}]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_scenario(scenario_label: str,
                 time_col: str,
                 nodes: pd.DataFrame,
                 edges: pd.DataFrame,
                 force_recompute: bool = False,
                 GM_fixed: object = None):
    """
    Full pipeline for one time-weight scenario.
    Results are cached in output/ as pickle files.

    GM_fixed: if provided, skip gravity computation and use this pre-computed
              gravity pairs DataFrame instead. This ensures all scenarios share
              the same demand matrix (only routing changes, not demand).
              Pass the baseline GravityPairs for density scenarios.
    """
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_label}  (time col = '{time_col}')")
    print(f"{'='*60}")

    grav_path = os.path.join(OUTPUT_DIR, f"GravityPairs_{scenario_label}.pkl")
    net_path  = os.path.join(OUTPUT_DIR, f"Network_{scenario_label}.pkl")

    # ── Build graph ──────────────────────────────────────────────────────────
    G = build_graph(nodes, edges)

    # ── Gravity pairs ────────────────────────────────────────────────────────
    if GM_fixed is not None:
        # Use pre-computed gravity from baseline: demand is population-driven,
        # not affected by city boundary scenario
        GM = GM_fixed
        print("  Using fixed baseline gravity pairs (demand independent of routing)")
        # Save a copy so the cache reflects what was used
        pickle.dump(GM, open(grav_path, "wb"))
    elif not force_recompute and os.path.exists(grav_path):
        print("  Loading cached gravity pairs …")
        GM = pickle.load(open(grav_path, "rb"))
    else:
        print("  Computing gravity pairs …")
        GM = compute_gravity(G, time_weight=time_col)
        pickle.dump(GM, open(grav_path, "wb"))
        print(f"  Saved: {grav_path}")

    # ── Betweenness ──────────────────────────────────────────────────────────
    if not force_recompute and os.path.exists(net_path):
        print("  Loading cached betweenness …")
        G = pickle.load(open(net_path, "rb"))
    else:
        G = compute_betweenness(G, GM, time_weight=time_col)
        pickle.dump(G, open(net_path, "wb"))
        print(f"  Saved: {net_path}")

    # ── Figures ──────────────────────────────────────────────────────────────
    print("  Generating figures …")

    # Full-network congestion maps
    for year, b_attr, p_attr in [
        ("2015", "Between",   "Pop2015"),
        ("2050", "Between50", "Pop2050"),
    ]:
        plot_network(
            G, nodes,
            between_attr = b_attr,
            pop_attr     = p_attr,
            title        = f"Africa Road Congestion {year}  [{scenario_label}]",
            out_path     = os.path.join(FIG_DIR, f"Congestion_{scenario_label}_{year}.png"),
        )

    # Lagos corridor profiles
    lagos_dests = ["Kano", "Abuja", "Onitsha", "Accra", "Abidjan"]
    for year, b_attr in [("2015", "Between"), ("2050", "Between50")]:
        plot_corridor_profile(
            G, nodes,
            origin_name  = "Lagos",
            dest_names   = lagos_dests,
            between_attr = b_attr,
            title        = f"Lagos corridor congestion {year}  [{scenario_label}]",
            out_path     = os.path.join(FIG_DIR,
                                        f"LagosCongestion_{scenario_label}_{year}.png"),
        )

    # 2015 vs 2050 scatter
    plot_scatter_2015_vs_2050(
        G, scenario_label,
        out_path = os.path.join(FIG_DIR, f"Scatter_{scenario_label}.png"),
    )

    return G, GM


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Allow: python africa_transport.py [scenario] [--force]
    args = sys.argv[1:]
    force = "--force" in args
    args  = [a for a in args if a != "--force"]

    print("Loading data …")
    nodes, edges = load_data()
    print(f"  Nodes: {len(nodes):,}  |  Edges: {len(edges):,}")
    print(f"  Cities with Pop2015 > 0: {(nodes['Pop2015'] > 0).sum():,}")
    print(f"  Cities with Pop2050 > 0: {(nodes['Pop2050'] > 0).sum():,}")

    # Which scenarios to run
    if args:
        scenarios = {k: v for k, v in TIME_SCENARIOS.items() if k in args}
    else:
        scenarios = TIME_SCENARIOS

    # Density scenarios share the baseline gravity matrix (demand is
    # population-driven, independent of city boundary scenario).
    DENSITY_SCENARIOS = {"urban_high_2050", "urban_cons_2050", "urban_low_2050"}
    BASELINE_GRAV_PATH = os.path.join(OUTPUT_DIR, "GravityPairs_baseline.pkl")

    GM_baseline = None
    if any(s in DENSITY_SCENARIOS for s in scenarios):
        if os.path.exists(BASELINE_GRAV_PATH):
            print("  Loading baseline gravity pairs for density scenarios …")
            GM_baseline = pickle.load(open(BASELINE_GRAV_PATH, "rb"))
        else:
            print("  ⚠ Baseline gravity pairs not found.")
            print("    Run baseline scenario first: python africa_transport.py baseline")

    results = {}
    for label, col in scenarios.items():
        if col not in edges.columns:
            print(f"  ⚠ Column '{col}' not in edges – skipping scenario '{label}'")
            continue
        # Pass fixed baseline GM for density scenarios
        gm_fixed = GM_baseline if label in DENSITY_SCENARIOS else None
        G, GM = run_scenario(label, col, nodes, edges,
                             force_recompute=force, GM_fixed=gm_fixed)
        results[label] = (G, GM)

    print("\nAll done. Outputs in:", OUTPUT_DIR)
