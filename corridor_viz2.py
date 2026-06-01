"""
corridor_viz2.py
================
Corridor visualizations using the same physics model as urban_psi_sweep.py:

  r_sweep    = r_2015 * (1 + psi)
  urban_frac = min(r_sweep_from + r_sweep_to, L) / L
  time_new   = urban_frac * L/v_slow + (1-urban_frac) * L/v   (minutes)

  v       from road type (motorway=100, trunk=60, primary=40, added=15 km/h)
  v_slow  = 15 km/h (city internal speed)
  baseline = time (no urban penalty)

Outputs:
  results/corridor_profiles.png    — betweenness profile along each corridor
  results/corridor_times_line.png  — travel time vs psi for each corridor
"""

import os
import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree

DATA_DIR   = "input"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CORRIDORS = [
    ("Lagos",        "Accra"),
    ("Lagos",        "Abidjan"),
    ("Lagos",        "Kano"),
    ("Nairobi",      "Kampala"),
    ("Johannesburg", "Durban"),
    ("Dakar",        "Abidjan"),
]

PSI_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]
COLORS     = ["#2166ac", "#4dac26", "#f4a582", "#d6604d", "#b2182b"]

SPEED_TABLE = {
    "motorway":      100.0,
    "motorway_link": 100.0,
    "trunk":          60.0,
    "trunk_link":     60.0,
    "primary":        40.0,
    "primary_link":   40.0,
    "added":          15.0,
}
V_SLOW = 15.0

# ── load ─────────────────────────────────────────────────────────────────────
nodes = pd.read_csv(os.path.join(DATA_DIR, "AfricaNetworkNodes.csv"))
edges = pd.read_csv(os.path.join(DATA_DIR, "AfricaNetworkEdges.csv"))
afr   = pd.read_csv(os.path.join(DATA_DIR, "Africapolis_2050.csv"))

afp = afr[["Agglomeration_ID","Population_2050"]].rename(
    columns={"Agglomeration_ID":"name","Population_2050":"Pop2050"})
nodes = nodes.merge(afp, on="name", how="left")
nodes["Pop2050"] = nodes["Pop2050"].fillna(0)

name_to_idx  = {str(n): i for i, n in enumerate(nodes["name"])}
from_idx_arr = edges["from"].astype(str).map(name_to_idx).values.astype(int)
to_idx_arr   = edges["to"].astype(str).map(name_to_idx).values.astype(int)

G = ig.Graph(n=len(nodes), directed=False)
G.add_edges(list(zip(from_idx_arr, to_idx_arr)))
for col in edges.columns: G.es[col] = edges[col].tolist()
for col in nodes.columns:  G.vs[col] = nodes[col].tolist()

name_to_node = {str(row["agglosName"]): i for i, row in nodes.iterrows()}

# ── city radii (r_2015) ───────────────────────────────────────────────────────
city_mask  = nodes["Pop2015"] > 0
city_nodes = nodes[city_mask].copy().reset_index(drop=True)
tree = cKDTree(afr[["Longitude","Latitude"]].values)
_, idx = tree.query(city_nodes[["x","y"]].values, k=1)

city_nodes["Built_up_2015"]     = afr.iloc[idx]["Built up_2015"].values
city_nodes["Built_up_2045"]     = afr.iloc[idx]["Built up_2045"].values
city_nodes["Built_up_2050_raw"] = afr.iloc[idx]["Built up_2050"].values
anomaly = city_nodes["Built_up_2050_raw"] > 2 * city_nodes["Built_up_2045"]
city_nodes["Built_up_2050"] = city_nodes["Built_up_2050_raw"].copy()
city_nodes.loc[anomaly, "Built_up_2050"] = city_nodes.loc[anomaly, "Built_up_2045"]
city_nodes["r_2015"] = np.sqrt(city_nodes["Built_up_2015"].clip(lower=0))

city_r2015 = dict(zip(city_nodes["name"].astype(str), city_nodes["r_2015"].values))

# ── edge fixed attributes ─────────────────────────────────────────────────────
edge_l      = edges["l"].values
edge_v      = edges["h"].map(SPEED_TABLE).fillna(40.0).values
r_from_2015 = edges["from"].astype(str).map(city_r2015).fillna(0).values
r_to_2015   = edges["to"].astype(str).map(city_r2015).fillna(0).values
time_orig   = edges["time"].values.copy()

def compute_time_new(psi):
    r_sweep_from = r_from_2015 * (1.0 + psi)
    r_sweep_to   = r_to_2015   * (1.0 + psi)
    urban_frac   = np.where(edge_l > 0,
                            np.minimum(r_sweep_from + r_sweep_to, edge_l) / edge_l,
                            0.0)
    t_new = (urban_frac * edge_l / V_SLOW +
             (1.0 - urban_frac) * edge_l / edge_v) * 60.0
    return t_new, urban_frac

# precompute for PSI_VALUES
for psi in PSI_VALUES:
    t, _ = compute_time_new(psi)
    G.es[f"t_{psi}"] = t.tolist()

# ── load betweenness from sweep results ──────────────────────────────────────
eb   = pd.read_csv(os.path.join(OUTPUT_DIR, "edges_betweenness_delta.csv"))
bet0 = eb["betweenness_psi0"].values
bet1 = eb["betweenness_psi1"].values

def interp_bet(psi):
    alpha = min(max(psi, 0.0), 1.0)
    return bet0 * (1.0 - alpha) + bet1 * alpha

def get_path(orig, dest, wkey):
    o = name_to_node.get(orig)
    d = name_to_node.get(dest)
    if o is None or d is None:
        return None, None
    epath = G.get_shortest_paths(v=o, to=d, weights=wkey, output="epath")[0]
    if not epath:
        return None, None
    cum = np.concatenate([[0], np.cumsum([G.es[e]["l"] for e in epath])])
    return epath, cum

# ── FIGURE 1: corridor betweenness profiles ───────────────────────────────────
# urban_frac at psi=0 for shading
_, uf_psi0 = compute_time_new(0.0)

n_corr = len(CORRIDORS)
fig, axes = plt.subplots(n_corr, 1, figsize=(11, n_corr * 2.5), sharex=False)
plt.rcParams.update({"font.size": 9})

for ai, (orig, dest) in enumerate(CORRIDORS):
    ax = axes[ai]

    for pi, psi in enumerate(PSI_VALUES):
        eids, cum = get_path(orig, dest, f"t_{psi}")
        if eids is None:
            continue

        bvals = interp_bet(psi)
        x_mid = [(cum[i] + cum[i+1]) / 2 for i in range(len(eids))]
        y_val = [bvals[e] for e in eids]

        ax.plot(x_mid, y_val, color=COLORS[pi], linewidth=1.4, alpha=0.85,
                label=f"ψ={psi}" if ai == 0 else "_nolegend_")

        # shade by urban_frac at psi=0 (2015 city footprint)
        if psi == 0.0:
            for i, eid in enumerate(eids):
                uf = uf_psi0[eid]
                if uf > 0:
                    ax.axvspan(cum[i], cum[i+1],
                               color="#FFD700",
                               alpha=0.1 + 0.35 * uf,
                               zorder=0)

    eids0, cum0 = get_path(orig, dest, "t_0.0")
    if eids0 is None:
        continue

    t0 = G.distances(source=name_to_node[orig],
                     target=name_to_node[dest], weights="t_0.0")[0][0]
    t2 = G.distances(source=name_to_node[orig],
                     target=name_to_node[dest], weights="t_2.0")[0][0]

    ax.set_xlim(0, cum0[-1])
    ax.set_ylabel("Betweenness", fontsize=8)
    ax.set_title(
        f"{orig} → {dest}   "
        f"[ψ=0: {t0/60:.1f}h  →  ψ=2: {t2/60:.1f}h  (+{100*(t2-t0)/t0:.0f}%)]",
        fontsize=9, loc="left")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines[["top","right"]].set_visible(False)

    if ai == n_corr - 1:
        ax.set_xlabel("Distance along path (km)", fontsize=8)

handles = [mpatches.Patch(color=COLORS[i], label=f"ψ={p}")
           for i, p in enumerate(PSI_VALUES)]
handles.append(mpatches.Patch(color="#FFD700", alpha=0.45,
                               label="Urban segment (darker = higher urban_frac at ψ=0)"))
fig.legend(handles=handles, loc="upper right", fontsize=8,
           framealpha=0.9, ncol=2, bbox_to_anchor=(0.99, 0.99))

fig.suptitle(
    "Corridor Betweenness Profiles under Urban Expansion (ψ)\n"
    f"time_new = urban_frac×L/v_slow + (1−urban_frac)×L/v  |  "
    f"v_slow={V_SLOW}km/h  |  baseline = time (no urban penalty)",
    fontsize=9, y=1.01)
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "corridor_profiles.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ── FIGURE 2: corridor travel time vs psi ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
all_psi = np.arange(0, 2.1, 0.2)

for orig, dest in CORRIDORS:
    o = name_to_node.get(orig)
    d = name_to_node.get(dest)
    if o is None or d is None:
        continue
    times_h = []
    for psi in all_psi:
        t_new, _ = compute_time_new(psi)
        G.es["_t"] = t_new.tolist()
        tt = G.distances(source=o, target=d, weights="_t")[0][0]
        times_h.append(tt / 60)
    ax.plot(all_psi, times_h, marker="o", markersize=4,
            linewidth=1.8, label=f"{orig}→{dest}")

ax.set_xlabel("ψ (city radius growth ratio beyond 2015)", fontsize=10)
ax.set_ylabel("Travel time (hours)", fontsize=10)
ax.set_title("Corridor Travel Times vs Urban Expansion ψ\n"
             f"Baseline = time (no urban penalty) | v_slow={V_SLOW}km/h",
             fontsize=10)
ax.set_xticks(all_psi)
ax.set_xticklabels([f"{p:.1f}" for p in all_psi], rotation=45, fontsize=8)
ax.legend(fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "corridor_times_line.png")
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

print("\n✓ Done.")
