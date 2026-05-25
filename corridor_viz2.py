"""
corridor_viz2.py — 简洁风格，corridor折线profile + 时间对比柱状图
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

# ── load ─────────────────────────────────────────────────────────────────────
nodes = pd.read_csv(os.path.join(DATA_DIR, "AfricaNetworkNodes.csv"))
edges = pd.read_csv(os.path.join(DATA_DIR, "AfricaNetworkEdges.csv"))
afr   = pd.read_csv(os.path.join(DATA_DIR, "Africapolis_2050.csv"))

afp = afr[["Agglomeration_ID","Population_2050"]].rename(
    columns={"Agglomeration_ID":"name","Population_2050":"Pop2050"})
nodes = nodes.merge(afp, on="name", how="left")
nodes["Pop2050"] = nodes["Pop2050"].fillna(0)

name_to_idx = {str(n): i for i, n in enumerate(nodes["name"])}
from_idx = edges["from"].astype(str).map(name_to_idx).values.astype(int)
to_idx   = edges["to"].astype(str).map(name_to_idx).values.astype(int)

G = ig.Graph(n=len(nodes), directed=False)
G.add_edges(list(zip(from_idx, to_idx)))
for col in edges.columns: G.es[col] = edges[col].tolist()
for col in nodes.columns:  G.vs[col] = nodes[col].tolist()

name_to_node = {str(row["agglosName"]): i for i, row in nodes.iterrows()}

city_set  = set(nodes.loc[nodes["Pop2015"]>0, "name"].astype(str))
is_urban  = (edges["from"].astype(str).isin(city_set) |
             edges["to"].astype(str).isin(city_set)).values
time_orig    = edges["time"].values.copy()

# 城市半径查找表（与urban_psi_sweep.py一致）
afr   = pd.read_csv(os.path.join(DATA_DIR, "Africapolis_2050.csv"))
tree  = cKDTree(afr[["Longitude","Latitude"]].values)
city_mask2 = nodes["Pop2015"] > 0
city_nodes2 = nodes[city_mask2].copy().reset_index(drop=True)
_, idx2 = tree.query(city_nodes2[["x","y"]].values, k=1)
city_nodes2["Built_up_2045"]     = afr.iloc[idx2]["Built up_2045"].values
city_nodes2["Built_up_2050_raw"] = afr.iloc[idx2]["Built up_2050"].values
anomaly2 = city_nodes2["Built_up_2050_raw"] > 2 * city_nodes2["Built_up_2045"]
city_nodes2["Built_up_2050"] = city_nodes2["Built_up_2050_raw"].copy()
city_nodes2.loc[anomaly2, "Built_up_2050"] = city_nodes2.loc[anomaly2, "Built_up_2045"]
city_nodes2["r_2050"] = np.sqrt(city_nodes2["Built_up_2050"].clip(lower=0))
city_radius2 = dict(zip(city_nodes2["name"].astype(str), city_nodes2["r_2050"].values))

edge_l  = edges["l"].values
r_from2 = edges["from"].astype(str).map(city_radius2).fillna(0).values
r_to2   = edges["to"].astype(str).map(city_radius2).fillna(0).values
urban_frac2 = np.where(edge_l > 0,
                       np.minimum(r_from2 + r_to2, edge_l) / edge_l, 0.0)

eb = pd.read_csv(os.path.join(OUTPUT_DIR, "edges_betweenness_delta.csv"))

# precompute time weights: time_new = time * (1 + urban_frac * psi)
for psi in PSI_VALUES:
    t = time_orig * (1.0 + urban_frac2 * psi)
    G.es[f"t_{psi}"] = t.tolist()

# ── FIGURE 1: corridor profiles ──────────────────────────────────────────────
# For each corridor: x=cumulative distance (km), y=betweenness value along path
# One line per psi value. Betweenness comes from psi=0 and psi=1 sweep results;
# for intermediate values we use linear interpolation as approximation.
bet0 = eb["betweenness_psi0"].values
bet1 = eb["betweenness_psi1"].values

def get_path(orig, dest, wkey):
    o = name_to_node.get(orig)
    d = name_to_node.get(dest)
    if o is None or d is None:
        return None, None, None
    epath = G.get_shortest_paths(v=o, to=d, weights=wkey, output="epath")[0]
    if not epath:
        return None, None, None
    lens = [G.es[e]["l"] for e in epath]
    cum  = np.concatenate([[0], np.cumsum(lens)])
    return epath, cum, [is_urban[e] for e in epath]

def interp_bet(psi):
    """Linearly interpolate betweenness between psi=0 and psi=1."""
    alpha = min(psi, 1.0)
    return bet0 * (1 - alpha) + bet1 * alpha

n_corr = len(CORRIDORS)
fig, axes = plt.subplots(n_corr, 1, figsize=(11, n_corr * 2.5), sharex=False)
plt.rcParams.update({"font.size": 9})

for ai, (orig, dest) in enumerate(CORRIDORS):
    ax = axes[ai]

    for pi, psi in enumerate(PSI_VALUES):
        wkey = f"t_{psi}"
        eids, cum, urban_flags = get_path(orig, dest, wkey)
        if eids is None:
            continue

        bvals = interp_bet(psi)

        # midpoints of each edge segment (x-axis)
        x_mid = [(cum[i] + cum[i+1]) / 2 for i in range(len(eids))]
        y_val = [bvals[e] for e in eids]

        ax.plot(x_mid, y_val, color=COLORS[pi],
                linewidth=1.4, alpha=0.85,
                label=f"ψ={psi}" if ai == 0 else "_nolegend_")

        # shade urban segments on baseline
        if psi == 0.0:
            for i, eid in enumerate(eids):
                if urban_flags[i]:
                    ax.axvspan(cum[i], cum[i+1],
                               color="#FFD700", alpha=0.12, zorder=0)

    # total travel time annotations
    t0 = G.distances(source=name_to_node[orig],
                     target=name_to_node[dest], weights="t_0.0")[0][0]
    t1 = G.distances(source=name_to_node[orig],
                     target=name_to_node[dest], weights="t_2.0")[0][0]

    eids0, cum0, _ = get_path(orig, dest, "t_0.0")
    max_x = cum0[-1] if cum0 is not None else 1

    ax.set_xlim(0, max_x)
    ax.set_ylabel("Betweenness", fontsize=8)
    ax.set_title(
        f"{orig} → {dest}   "
        f"[ψ=0: {t0/60:.1f}h  →  ψ=2: {t1/60:.1f}h  (+{100*(t1-t0)/t0:.0f}%)]",
        fontsize=9, loc="left")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines[["top","right"]].set_visible(False)

    if ai == n_corr - 1:
        ax.set_xlabel("Distance along path (km)", fontsize=8)

# shared legend
handles = [mpatches.Patch(color=COLORS[i], label=f"ψ={p}")
           for i, p in enumerate(PSI_VALUES)]
handles.append(mpatches.Patch(color="#FFD700", alpha=0.4, label="Urban segment (ψ=0)"))
fig.legend(handles=handles, loc="upper right", fontsize=8,
           framealpha=0.9, ncol=3, bbox_to_anchor=(0.98, 0.98))

fig.suptitle(
    "Corridor Betweenness Profiles under Urban Expansion (ψ)\n"
    "Yellow shading = urban edge segments | Title = total travel time at ψ=0 and ψ=2",
    fontsize=10, y=1.01)
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "corridor_profiles.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ── FIGURE 2: corridor travel time lines (all psi values) ────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

all_psi = np.arange(0, 2.1, 0.2)
for ci, (orig, dest) in enumerate(CORRIDORS):
    o = name_to_node.get(orig)
    d = name_to_node.get(dest)
    if o is None or d is None:
        continue
    times_h = []
    for psi in all_psi:
        t = time_orig * (1.0 + urban_frac2 * psi)
        G.es["_t"] = t.tolist()
        tt = G.distances(source=o, target=d, weights="_t")[0][0]
        times_h.append(tt / 60)

    label = f"{orig}→{dest}"
    ax.plot(all_psi, times_h, marker="o", markersize=4,
            linewidth=1.8, label=label)

ax.set_xlabel("ψ (urban penalty multiplier beyond 2015)", fontsize=10)
ax.set_ylabel("Travel time (hours)", fontsize=10)
ax.set_title("Corridor Travel Times vs Urban Expansion ψ", fontsize=11)
ax.set_xticks(all_psi)
ax.set_xticklabels([f"{p:.1f}" for p in all_psi], rotation=45, fontsize=8)
ax.legend(fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.spines[["top","right"]].set_visible(False)

out2 = os.path.join(OUTPUT_DIR, "corridor_times_line.png")
plt.tight_layout()
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

print("\n✓ Done.")
