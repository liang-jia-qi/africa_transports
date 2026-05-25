"""
corridor_viz.py
===============
Visualizes betweenness congestion along key African transport corridors
for psi=0 (baseline) vs psi=0.7 (cons scenario) vs psi=1.4 (high scenario).

Outputs:
  results/corridor_profiles.png   — bar profiles per corridor (like R script)
  results/corridor_map.png        — map with top betweenness-delta edges highlighted
"""

import os
import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = "input"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── corridors to visualize ───────────────────────────────────────────────────
CORRIDORS = [
    ("Lagos",        "Accra",    "#E84855"),
    ("Lagos",        "Abidjan",  "#FF9F1C"),
    ("Lagos",        "Kano",     "#2EC4B6"),
    ("Nairobi",      "Kampala",  "#7B2D8B"),
    ("Johannesburg", "Durban",   "#3A86FF"),
    ("Dakar",        "Abidjan",  "#44BBA4"),
]

PSI_VALUES  = [0.0, 0.7, 1.4]
PSI_LABELS  = ["ψ=0 (no growth)", "ψ=0.7 (cons scenario)", "ψ=1.4 (high scenario)"]
PSI_ALPHAS  = [1.0, 0.65, 0.35]
PSI_HATCHES = [None, None, "///"]

# ════════════════════════════════════════════════════════════════════════════
# 1. Load data & build graph
# ════════════════════════════════════════════════════════════════════════════
print("Loading data...")
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

# urban edge mask
city_set  = set(nodes.loc[nodes["Pop2015"]>0, "name"].astype(str))
is_urban  = (edges["from"].astype(str).isin(city_set) |
             edges["to"].astype(str).isin(city_set)).values
time_orig = edges["time"].values.copy()

# precompute time weights for each psi
time_weights = {}
for psi in PSI_VALUES:
    t = time_orig.copy()
    t[is_urban] *= (1 + psi)
    key = f"t_{psi:.1f}"
    G.es[key] = t.tolist()
    time_weights[psi] = key

# ════════════════════════════════════════════════════════════════════════════
# 2. Betweenness per corridor per psi (use precomputed delta where possible)
#    Here we compute per-edge betweenness contribution along the path
# ════════════════════════════════════════════════════════════════════════════
print("Computing corridor betweenness profiles...")

# load betweenness from sweep results
eb = pd.read_csv(os.path.join(OUTPUT_DIR, "edges_betweenness_delta.csv"))
bet_psi0 = eb["betweenness_psi0"].values
bet_psi1 = eb["betweenness_psi1"].values   # psi=1.0 approx


def get_path_info(G, orig_name, dest_name, weight_key):
    """Returns (edge_ids, cumulative_distances, is_urban_flags) along path."""
    o = name_to_node.get(orig_name)
    d = name_to_node.get(dest_name)
    if o is None or d is None:
        return None, None, None

    epath = G.get_shortest_paths(v=o, to=d, weights=weight_key, output="epath")[0]
    if not epath:
        return None, None, None

    wts   = G.es[weight_key]
    lens  = [G.es[e]["l"] for e in epath]
    urban = [is_urban[e] for e in epath]

    cum = [0]
    for ln in lens:
        cum.append(cum[-1] + ln)

    return epath, cum, urban


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Corridor bar profiles  (rows = corridors, cols = psi values)
# ════════════════════════════════════════════════════════════════════════════
print("Drawing corridor profiles...")

n_corr = len(CORRIDORS)
fig, axes = plt.subplots(n_corr, 1, figsize=(12, n_corr * 2.2))
fig.patch.set_facecolor("#0F1117")

YMAX = max(bet_psi0.max(), bet_psi1.max()) * 1.2 or 10

for ax_i, (orig, dest, color) in enumerate(CORRIDORS):
    ax = axes[ax_i]
    ax.set_facecolor("#0F1117")

    psi_drawn = False
    total_times = []

    for pi, (psi, label, alpha) in enumerate(zip(PSI_VALUES, PSI_LABELS, PSI_ALPHAS)):
        wkey = time_weights[psi]
        eids, cum_dist, urban_flags = get_path_info(G, orig, dest, wkey)
        if eids is None:
            continue

        # betweenness values: use psi0 and psi1 (closest available)
        if psi == 0.0:
            bvals = bet_psi0
        elif psi <= 0.7:
            bvals = bet_psi0 * (1 - psi/0.7) + bet_psi1 * (psi/0.7)
        else:
            bvals = bet_psi1

        total_t = sum(G.es[wkey][e] for e in eids)
        total_times.append(total_t)

        for i, eid in enumerate(eids):
            x0, x1 = cum_dist[i], cum_dist[i+1]
            bval = max(bvals[eid], 0.001)
            ec = color if urban_flags[i] else "#888888"

            ax.fill_between([x0, x1], [0, 0], [bval, bval],
                            color=color if pi == 0 else "#AAAAAA",
                            alpha=alpha * (0.9 if urban_flags[i] else 0.4),
                            linewidth=0)

        psi_drawn = True

    # path total distance (km)
    eids0, cum0, _ = get_path_info(G, orig, dest, time_weights[0.0])
    max_dist = cum0[-1] if cum0 else 1

    ax.set_xlim(0, max_dist)
    ax.set_ylim(0, max(YMAX, 0.01))
    ax.set_yscale("symlog", linthresh=0.01)

    # city label
    ax.text(max_dist * 0.98, YMAX * 0.5,
            f"{orig} → {dest}",
            ha="right", va="center", fontsize=9, color=color,
            fontweight="bold", fontfamily="monospace")

    # time annotations
    if total_times:
        t_diff = total_times[-1] - total_times[0] if len(total_times) > 1 else 0
        ax.text(max_dist * 0.02, YMAX * 0.5,
                f"+{t_diff:.0f} min (+{100*t_diff/total_times[0]:.0f}%)" if total_times[0] > 0 else "",
                ha="left", va="center", fontsize=8, color="#FFCC44")

    ax.spines[["top","right","bottom","left"]].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # x-axis: distance ticks
    for tick_km in range(0, int(max_dist)+1, 200):
        ax.axvline(tick_km, color="#333333", linewidth=0.5, zorder=0)

# Legend
legend_elements = [
    mpatches.Patch(color="#E84855", alpha=1.0, label="ψ=0 (baseline, urban edge)"),
    mpatches.Patch(color="#AAAAAA", alpha=0.65, label="ψ=0.7 (cons scenario)"),
    mpatches.Patch(color="#AAAAAA", alpha=0.35, label="ψ=1.4 (high scenario)"),
    mpatches.Patch(color="#888888", alpha=0.5, label="Rural edge"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4,
           facecolor="#0F1117", edgecolor="none",
           labelcolor="white", fontsize=8, bbox_to_anchor=(0.5, 0))

fig.suptitle("Corridor Congestion Profiles: Betweenness under Urban Growth Scenarios",
             color="white", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout(rect=[0, 0.04, 1, 1])
out1 = os.path.join(OUTPUT_DIR, "corridor_profiles.png")
fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.close()
print(f"Saved: {out1}")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Africa map — top betweenness-delta edges + corridors
# ════════════════════════════════════════════════════════════════════════════
print("Drawing Africa map...")

coords = nodes[["x","y"]].values

fig, ax = plt.subplots(figsize=(14, 14))
fig.patch.set_facecolor("#0A0E1A")
ax.set_facecolor("#0A0E1A")
ax.set_aspect("equal")
ax.axis("off")

# --- draw all edges (faint baseline) ---
for e in G.es:
    s, t = e.source, e.target
    xs = [coords[s,0], coords[t,0]]
    ys = [coords[s,1], coords[t,1]]
    ax.plot(xs, ys, color="#1A2035", linewidth=0.4, alpha=0.8, zorder=1)

# --- color edges by betweenness delta ---
delta = eb["betweenness_delta"].values
delta_norm = delta / (np.abs(delta).max() + 1e-9)

cmap_pos = plt.cm.YlOrRd
cmap_neg = plt.cm.Blues

for i, e in enumerate(G.es):
    d = delta_norm[i]
    if abs(d) < 0.05:
        continue
    s, t = e.source, e.target
    xs = [coords[s,0], coords[t,0]]
    ys = [coords[s,1], coords[t,1]]
    if d > 0:
        col = cmap_pos(0.3 + 0.7 * d)
        lw  = 0.5 + 3.0 * d
    else:
        col = cmap_neg(0.3 + 0.7 * abs(d))
        lw  = 0.5 + 2.0 * abs(d)
    ax.plot(xs, ys, color=col, linewidth=lw, alpha=0.85, zorder=2)

# --- draw corridor paths ---
for orig, dest, color in CORRIDORS:
    eids, cum, urban = get_path_info(G, orig, dest, time_weights[0.0])
    if eids is None:
        continue
    path_nodes = G.get_shortest_paths(
        v=name_to_node[orig], to=name_to_node[dest],
        weights=time_weights[0.0], output="vpath")[0]
    px = [coords[n,0] for n in path_nodes]
    py = [coords[n,1] for n in path_nodes]
    ax.plot(px, py, color=color, linewidth=1.8, alpha=0.7, zorder=3, linestyle="--")

# --- city nodes ---
city_mask = nodes["Pop2015"].values > 0
pop15 = nodes["Pop2015"].values
sizes = np.where(city_mask, np.log1p(pop15) * 0.8, 0.3)
colors_node = np.where(city_mask, "#FFFFFF", "#2A3050")
ax.scatter(coords[~city_mask,0], coords[~city_mask,1],
           s=0.3, color="#2A3050", zorder=4)
ax.scatter(coords[city_mask,0], coords[city_mask,1],
           s=sizes[city_mask], color="#FFFFFF", alpha=0.7,
           linewidths=0.2, edgecolors="#AAAAAA", zorder=5)

# --- label corridor endpoints ---
labeled = set()
for orig, dest, color in CORRIDORS:
    for city in [orig, dest]:
        if city in labeled:
            continue
        idx = name_to_node.get(city)
        if idx is not None:
            ax.annotate(city,
                        xy=(coords[idx,0], coords[idx,1]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7.5, color=color, fontweight="bold",
                        fontfamily="monospace", zorder=6)
            labeled.add(city)

# --- colorbars ---
sm_pos = plt.cm.ScalarMappable(cmap=cmap_pos,
    norm=mcolors.Normalize(vmin=0, vmax=delta.max()))
sm_neg = plt.cm.ScalarMappable(cmap=cmap_neg,
    norm=mcolors.Normalize(vmin=0, vmax=abs(delta.min())))
sm_pos.set_array([])
sm_neg.set_array([])

cb1 = fig.colorbar(sm_pos, ax=ax, fraction=0.018, pad=0.01, shrink=0.35,
                   location="right")
cb1.set_label("Betweenness increase\n(ψ=0 → ψ=1)", color="white", fontsize=8)
cb1.ax.yaxis.set_tick_params(color="white")
plt.setp(cb1.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

cb2 = fig.colorbar(sm_neg, ax=ax, fraction=0.018, pad=0.06, shrink=0.35,
                   location="right")
cb2.set_label("Betweenness decrease\n(ψ=0 → ψ=1)", color="white", fontsize=8)
cb2.ax.yaxis.set_tick_params(color="white")
plt.setp(cb2.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

# legend for corridors
legend_corr = [mpatches.Patch(color=c, label=f"{o}→{d}", alpha=0.8)
               for o, d, c in CORRIDORS]
ax.legend(handles=legend_corr, loc="lower left",
          facecolor="#0A0E1A", edgecolor="#333333",
          labelcolor="white", fontsize=8, title="Corridors",
          title_fontsize=8)

ax.set_title("Africa Road Network: Betweenness Change under Urban Growth (ψ=0 → ψ=1)\n"
             "Red = load increase | Blue = load decrease | Dashed = corridor paths",
             color="white", fontsize=11, pad=12)

out2 = os.path.join(OUTPUT_DIR, "corridor_map.png")
fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor="#0A0E1A")
plt.close()
print(f"Saved: {out2}")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Corridor time comparison bar chart
# ════════════════════════════════════════════════════════════════════════════
print("Drawing corridor time comparison...")

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor("#0F1117")
ax.set_facecolor("#0F1117")

corr_labels = [f"{o}→{d}" for o,d,_ in CORRIDORS]
x = np.arange(len(CORRIDORS))
width = 0.25

for pi, (psi, label, alpha) in enumerate(zip(PSI_VALUES, PSI_LABELS, PSI_ALPHAS)):
    wkey = time_weights[psi]
    times = []
    for orig, dest, _ in CORRIDORS:
        o = name_to_node.get(orig)
        d = name_to_node.get(dest)
        if o and d:
            t = G.distances(source=o, target=d, weights=wkey)[0][0]
            times.append(t / 60)  # hours
        else:
            times.append(0)

    bars = ax.bar(x + (pi-1)*width, times, width,
                  label=label,
                  color=["#4477FF","#FF8C42","#FF3366"][pi],
                  alpha=alpha, edgecolor="#333333", linewidth=0.5)

    # value labels on psi=0 bars
    if pi == 0:
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{t:.1f}h", ha="center", va="bottom",
                    fontsize=7, color="#AAAAAA")

ax.set_xticks(x)
ax.set_xticklabels(corr_labels, rotation=25, ha="right",
                   color="white", fontsize=9, fontfamily="monospace")
ax.set_ylabel("Travel time (hours)", color="white", fontsize=10)
ax.set_title("Corridor Travel Times under Urban Growth Scenarios",
             color="white", fontsize=12, fontweight="bold")
ax.tick_params(colors="white")
ax.spines[["top","right"]].set_visible(False)
ax.spines[["bottom","left"]].set_color("#444444")
ax.yaxis.set_tick_params(colors="white")
ax.grid(axis="y", color="#1E2535", linewidth=0.8)
ax.legend(facecolor="#0F1117", edgecolor="#333333",
          labelcolor="white", fontsize=9)

out3 = os.path.join(OUTPUT_DIR, "corridor_times.png")
plt.tight_layout()
fig.savefig(out3, dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.close()
print(f"Saved: {out3}")

print("\n✓ 完成！输出：")
print(f"  {out1}  — 逐路段拥堵profile")
print(f"  {out2}  — 非洲地图（betweenness变化 + corridor路径）")
print(f"  {out3}  — corridor时间对比柱状图")
