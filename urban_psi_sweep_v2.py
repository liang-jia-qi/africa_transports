"""
urban_psi_sweep.py
==================
扫描城市半径增长系数 psi (0 -> 2)，计算2050年非洲城市间引力加权平均travel time。

物理模型：
  每条边有道路等级，对应固定速度v（km/h）：
    motorway/motorway_link: 100 km/h
    trunk/trunk_link:        60 km/h
    primary/primary_link:    40 km/h
    added:                   15 km/h
  城市内部速度 v_slow = 15 km/h（added道路速度，作为城市内行驶速度）

  r_2015     = sqrt(Built_up_2015)               城市2015年半径（km）
  r_sweep    = r_2015 × (1 + ψ)                  ψ=城市半径增长比例
  urban_frac = min(r_sweep_from + r_sweep_to, L) / L

  time_new   = urban_frac × L/v_slow + (1 - urban_frac) × L/v   (分钟)

  ψ=0  → r_sweep=r_2015，对应2015年城市半径
  ψ>0  → 城市扩张，更多路段被城市覆盖
  基准  → time（无任何城市惩罚的原始行驶时间）

  当拿到3种scenario面积数据后：
    ψ_scenario = median((r_2050_scenario - r_2015) / r_2015)
    直接标注在折线图上

输出：
  results/psi_sweep_results.csv
  results/psi_sweep_chart.png
  results/city_psi_values.csv
  results/edges_betweenness_delta.csv
"""

import os
import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm

# ── 路径配置 ─────────────────────────────────────────────────────────────────
DATA_DIR   = "input"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NODES_FILE  = os.path.join(DATA_DIR, "AfricaNetworkNodes.csv")
EDGES_FILE  = os.path.join(DATA_DIR, "AfricaNetworkEdges.csv")
AFRICAPOLIS = os.path.join(DATA_DIR, "Africapolis_2050.csv")

# ── 参数 ─────────────────────────────────────────────────────────────────────
GRAVITY_BETA = 2.8
N_TOP_PAIRS  = 25_000
PSI_VALUES   = np.arange(0, 2.1, 0.2)

# 道路等级速度表（km/h）
SPEED_TABLE = {
    "motorway":      100.0,
    "motorway_link": 100.0,
    "trunk":          60.0,
    "trunk_link":     60.0,
    "primary":        40.0,
    "primary_link":   40.0,
    "added":          15.0,
}
V_SLOW = 15.0   # 城市内部速度（km/h），对应added道路

# ════════════════════════════════════════════════════════════════════════════
# 1. 读取数据
# ════════════════════════════════════════════════════════════════════════════
print("读取数据...")
nodes = pd.read_csv(NODES_FILE)
edges = pd.read_csv(EDGES_FILE)
afr   = pd.read_csv(AFRICAPOLIS)

# ════════════════════════════════════════════════════════════════════════════
# 2. 合并Pop2050
# ════════════════════════════════════════════════════════════════════════════
print("合并2050年人口数据...")
afp = afr[["Agglomeration_ID","Population_2050"]].rename(
    columns={"Agglomeration_ID":"name","Population_2050":"Pop2050"})
nodes = nodes.merge(afp, on="name", how="left")
nodes["Pop2050"] = nodes["Pop2050"].fillna(0)

# ════════════════════════════════════════════════════════════════════════════
# 3. 计算城市半径和psi_city
# ════════════════════════════════════════════════════════════════════════════
print("计算城市半径...")
city_mask  = nodes["Pop2015"] > 0
city_nodes = nodes[city_mask].copy().reset_index(drop=True)

tree = cKDTree(afr[["Longitude","Latitude"]].values)
_, idx = tree.query(city_nodes[["x","y"]].values, k=1)

city_nodes["Built_up_2015"]     = afr.iloc[idx]["Built up_2015"].values
city_nodes["Built_up_2045"]     = afr.iloc[idx]["Built up_2045"].values
city_nodes["Built_up_2050_raw"] = afr.iloc[idx]["Built up_2050"].values

# 修复Built_up_2050异常值（>2×Built_up_2045用2045替代）
anomaly = city_nodes["Built_up_2050_raw"] > 2 * city_nodes["Built_up_2045"]
city_nodes["Built_up_2050"] = city_nodes["Built_up_2050_raw"].copy()
city_nodes.loc[anomaly, "Built_up_2050"] = city_nodes.loc[anomaly, "Built_up_2045"]
print(f"  Built_up_2050异常城市（用2045替代）: {anomaly.sum()}")

city_nodes["r_2015"] = np.sqrt(city_nodes["Built_up_2015"].clip(lower=0))
city_nodes["r_2050"] = np.sqrt(city_nodes["Built_up_2050"].clip(lower=0))

# psi_city参考值（当有scenario面积数据时用于标定）
valid = city_nodes["r_2015"] > 0
city_nodes.loc[valid, "psi_city"] = (
    (city_nodes.loc[valid,"r_2050"] - city_nodes.loc[valid,"r_2015"])
    / city_nodes.loc[valid,"r_2015"]
)
city_nodes.loc[~valid, "psi_city"] = np.nan

psi_ref = city_nodes["psi_city"].median()
print(f"  psi_city中位数（Africapolis默认2050）: {psi_ref:.3f}")
print(f"  城市半径r_2015: 中位数={city_nodes['r_2015'].median():.1f}km, 最大={city_nodes['r_2015'].max():.1f}km")
print(f"  城市半径r_2050: 中位数={city_nodes['r_2050'].median():.1f}km, 最大={city_nodes['r_2050'].max():.1f}km")

city_nodes[["name","agglosName","ISO3","Pop2015","Pop2050",
            "Built_up_2015","Built_up_2050","r_2015","r_2050","psi_city"]].to_csv(
    os.path.join(OUTPUT_DIR,"city_psi_values.csv"), index=False)

# 半径查找表
city_r2015 = dict(zip(city_nodes["name"].astype(str), city_nodes["r_2015"].values))

# ════════════════════════════════════════════════════════════════════════════
# 4. 建图结构
# ════════════════════════════════════════════════════════════════════════════
print("建图...")
name_to_idx  = {str(n): i for i, n in enumerate(nodes["name"])}
from_idx_arr = edges["from"].astype(str).map(name_to_idx).values.astype(int)
to_idx_arr   = edges["to"].astype(str).map(name_to_idx).values.astype(int)
edge_tuples  = list(zip(from_idx_arr, to_idx_arr))

# 每条边的固定属性
edge_l   = edges["l"].values                                    # km
edge_v   = edges["h"].map(SPEED_TABLE).fillna(40.0).values      # km/h（未知等级默认40）
time_origin = edges["time"].values.copy()                        # 原始时间（分钟，用于验证）

# r_2015（每条边两端城市的半径）
r_from_2015 = edges["from"].astype(str).map(city_r2015).fillna(0).values
r_to_2015   = edges["to"].astype(str).map(city_r2015).fillna(0).values

print(f"  总节点: {len(nodes)}, 城市节点: {city_mask.sum()}")
print(f"  总边: {len(edges)}")

# 验证：用速度表重建time，与原始time对比
time_reconstructed = edge_l / edge_v * 60
print(f"  速度表验证: reconstructed vs original time 相关系数 = "
      f"{np.corrcoef(time_reconstructed, time_origin)[0,1]:.6f}")
print(f"  最大误差: {np.abs(time_reconstructed - time_origin).max():.4f} min")

# ════════════════════════════════════════════════════════════════════════════
# 5. 核心函数：给定psi，计算time_new
# ════════════════════════════════════════════════════════════════════════════
def compute_time_new(psi):
    """
    r_sweep = r_2015 * (1 + psi)
    urban_frac = min(r_sweep_from + r_sweep_to, L) / L
    time_new = urban_frac * L/v_slow + (1-urban_frac) * L/v   (分钟)
    """
    r_sweep_from = r_from_2015 * (1.0 + psi)
    r_sweep_to   = r_to_2015   * (1.0 + psi)
    urban_len    = np.minimum(r_sweep_from + r_sweep_to, edge_l)
    urban_frac   = np.where(edge_l > 0, urban_len / edge_l, 0.0)

    time_new = (urban_frac * edge_l / V_SLOW +
                (1.0 - urban_frac) * edge_l / edge_v) * 60.0
    return time_new, urban_frac

# ════════════════════════════════════════════════════════════════════════════
# 6. 引力对（baseline time排序，固定第25001-50000对）
# ════════════════════════════════════════════════════════════════════════════
print("\n计算引力对...")
city_idx_arr = np.where(nodes["Pop2015"].values > 0)[0]
pop2015 = nodes["Pop2015"].values.astype(float)
pop2050 = nodes["Pop2050"].values.astype(float)

G_base = ig.Graph(n=len(nodes), directed=False)
G_base.add_edges(edge_tuples)
G_base.es["time"] = time_origin.tolist()

print("  计算baseline距离矩阵...")
D_full = np.array(G_base.distances(source=city_idx_arr.tolist(), weights="time"))
D      = D_full[:, city_idx_arr]

P15 = pop2015[city_idx_arr] / 1000.0
P50 = pop2050[city_idx_arr] / 1000.0

Grav15 = np.outer(P15, P15) / (D ** GRAVITY_BETA + 1)
Grav50 = np.outer(P50, P50) / (D ** GRAVITY_BETA + 1)

ri, ci = np.tril_indices(len(city_idx_arr), k=-1)
GM = pd.DataFrame({
    "from":    city_idx_arr[ci],
    "to":      city_idx_arr[ri],
    "GravM":   Grav15[ri, ci],
    "GravM50": Grav50[ri, ci],
}).sort_values("GravM", ascending=False).reset_index(drop=True)

GM_top = GM.iloc[N_TOP_PAIRS : 2*N_TOP_PAIRS].copy()
print(f"  总城市对: {len(GM):,} | 跳过前{N_TOP_PAIRS:,}对 | 使用第{N_TOP_PAIRS+1:,}~{2*N_TOP_PAIRS:,}对")

# ════════════════════════════════════════════════════════════════════════════
# 7. 核心计算：给定psi，返回全部指标
# ════════════════════════════════════════════════════════════════════════════
def compute_metrics(psi):
    time_new, urban_frac = compute_time_new(psi)

    G = ig.Graph(n=len(nodes), directed=False)
    G.add_edges(edge_tuples)
    G.es["time_new"] = time_new.tolist()

    # 引力加权平均travel time
    # 用与betweenness相同的城市对（GM_top，第25001-50000对）
    # 固定引力权重（GM_top["GravM50"]，基于baseline距离），只有D_new随psi变化
    from_arr  = GM_top["from"].values
    to_arr    = GM_top["to"].values
    gm50_arr  = GM_top["GravM50"].values

    # 批量计算这些城市对的最短路径时间
    # 用distances矩阵（只算GM_top涉及的城市）
    gm_cities_from = np.unique(from_arr)
    D_new_sub = np.array(G.distances(source=gm_cities_from.tolist(),
                                     weights="time_new"))
    city_to_row = {c: i for i, c in enumerate(gm_cities_from)}

    d_pairs = np.array([
        D_new_sub[city_to_row[int(from_arr[k])],
                  int(to_arr[k])]
        for k in range(N_TOP_PAIRS)
    ])

    reach = np.isfinite(d_pairs) & (d_pairs > 0)
    gw_avg_time = np.average(d_pairs[reach], weights=gm50_arr[reach])

    # 引力加权边betweenness
    n_edges   = len(edges)
    between50 = np.zeros(n_edges)
    # from_arr, to_arr, gm50_arr already defined above

    for k in range(N_TOP_PAIRS):
        paths = G.get_shortest_paths(
            v=int(from_arr[k]), to=int(to_arr[k]),
            weights="time_new", output="epath")
        eids = paths[0]
        if not eids:
            continue
        between50[np.array(eids, dtype=int)] += gm50_arr[k]

    total_eb   = between50.sum()
    nonzero_eb = between50[between50 > 0]
    threshold  = np.percentile(nonzero_eb, 90) if len(nonzero_eb) else 0
    vuln_index = between50[between50 >= threshold].sum() / total_eb if total_eb > 0 else 0

    return gw_avg_time, total_eb, vuln_index, between50

# ════════════════════════════════════════════════════════════════════════════
# 8. 扫描 psi
# ════════════════════════════════════════════════════════════════════════════
# 快速sanity check
t_psi0, uf_psi0 = compute_time_new(0.0)
t_psi_ref, _    = compute_time_new(psi_ref)
print(f"\nSanity check:")
print(f"  psi=0 → time_new均值={t_psi0.mean():.2f} (time均值={time_origin.mean():.2f})")
print(f"  psi={psi_ref:.2f}(ref) → time_new均值={t_psi_ref.mean():.2f}")
print(f"  urban_frac(psi=0) 中位数(城市边)={np.median(uf_psi0[uf_psi0>0]):.3f}")

print(f"\n开始扫描 psi = {[round(p,2) for p in PSI_VALUES]}...")
print(f"每个psi约20-30秒，全程约5分钟\n")

records = []
betweenness_by_psi = {}

for psi in tqdm(PSI_VALUES, desc="psi sweep"):
    psi_r = round(psi, 2)
    gw_time, total_eb, vuln, eb_arr = compute_metrics(psi)
    records.append({
        "psi":                 psi_r,
        "gw_avg_time":         gw_time,
        "total_betweenness":   total_eb,
        "vulnerability_index": vuln,
    })
    betweenness_by_psi[psi_r] = eb_arr
    tqdm.write(f"  psi={psi:.1f} | gw_avg_time={gw_time:.4f} | "
               f"total_eb={total_eb:.3e} | vuln={vuln:.4f}")

results = pd.DataFrame(records)
results.to_csv(os.path.join(OUTPUT_DIR,"psi_sweep_results.csv"), index=False)

# ════════════════════════════════════════════════════════════════════════════
# 9. 折线图
# ════════════════════════════════════════════════════════════════════════════
print("生成折线图...")
psi_ticks = results["psi"].values

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Africa Transport Network: Urban Expansion (ψ) Impact on Travel Burden (2050 Population)\n"
    f"Baseline = time (no urban penalty) | v_slow={V_SLOW}km/h | "
    f"ψ = city radius growth ratio | ref ψ (Africapolis 2050 median) = {psi_ref:.2f}",
    fontsize=10)

for ax, col, color, ylabel, title in zip(
    axes,
    ["gw_avg_time", "total_betweenness", "vulnerability_index"],
    ["#2166ac", "#4dac26", "#d6604d"],
    ["Gravity-weighted avg travel time (min)",
     "Total edge betweenness (×10⁶)",
     "Vulnerability index (top 10% share)"],
    ["Main Indicator", "Total Network Load", "Network Vulnerability"]
):
    y = results[col] / (1e6 if col == "total_betweenness" else 1)
    ax.plot(results["psi"], y, color=color, marker="o",
            linewidth=2, markersize=6)
    ax.axvline(x=psi_ref, color="orange", linestyle="--", alpha=0.8,
               label=f"Africapolis 2050 ref (ψ={psi_ref:.2f})")
    ax.set_xlabel("ψ (city radius growth ratio beyond 2015)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(psi_ticks)
    ax.set_xticklabels([f"{p:.1f}" for p in psi_ticks], rotation=45, fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
out_chart = os.path.join(OUTPUT_DIR, "psi_sweep_chart.png")
fig.savefig(out_chart, dpi=150, bbox_inches="tight")
plt.close()
print(f"图表已保存: {out_chart}")

# ════════════════════════════════════════════════════════════════════════════
# 10. 空间分析：betweenness增量（psi=0 vs psi=1.0）
# ════════════════════════════════════════════════════════════════════════════
print("计算betweenness增量...")
eb0 = betweenness_by_psi.get(0.0, betweenness_by_psi[list(betweenness_by_psi.keys())[0]])
eb1 = betweenness_by_psi.get(1.0, betweenness_by_psi[list(betweenness_by_psi.keys())[-1]])

edges_out = edges.copy()
edges_out["betweenness_psi0"]    = eb0
edges_out["betweenness_psi1"]    = eb1
edges_out["betweenness_delta"]   = eb1 - eb0
edges_out["betweenness_pct_chg"] = (eb1 - eb0) / (eb0 + 1e-9) * 100
edges_out.to_csv(os.path.join(OUTPUT_DIR,"edges_betweenness_delta.csv"), index=False)

print("\n✓ 全部完成！输出：")
print(f"  results/psi_sweep_results.csv")
print(f"  results/psi_sweep_chart.png")
print(f"  results/city_psi_values.csv")
print(f"  results/edges_betweenness_delta.csv")
