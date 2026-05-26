"""
urban_psi_sweep.py
==================
扫描全局城市扩张系数 psi (0 -> 2)，计算2050年非洲城市间引力加权平均travel time。

与 africa_transport.py 保持一致的逻辑：
  - 引力公式：Pop_i/1000 * Pop_j/1000 / (D^2.8 + 1)
  - betweenness：对top-25000引力对逐对计算最短路径，累加GravM50权重
  - 距离矩阵：用time列（无urban penalty）作为baseline路径选择依据
    → 每个psi下重建time_new，重算距离矩阵，重跑betweenness

公式（框架C）：
  r_2015 = sqrt(Built_up_2015)
  r_2050 = sqrt(Built_up_2050)   异常值（>2×Built_up_2045）用Built_up_2045替代

  urban_frac(edge) = min(r_2050_from + r_2050_to, l) / l
    两端城市的半径之和，不超过边长l；非城市端点r=0

  time_new = time * (1 + urban_frac * psi)

  ψ的物理含义：城市半径增长比例 = (r_2050 - r_2015) / r_2015
  ψ=0 → time_new = time（无任何城市惩罚）
  ψ>0 → 城市段按半径比例减速

  当拿到3种scenario面积数据后：
    ψ_low/cons/high = 全大陆城市的(r_2050_scenario - r_2015)/r_2015 中位数
    直接标注在折线图上对应OECD预测点

输出：
  results/psi_sweep_results.csv        — 每个psi值对应的全大陆指标
  results/psi_sweep_chart.png          — 折线图
  results/edges_betweenness_delta.csv  — psi=0 vs psi=1的边负荷增量
"""

import os
import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm

# ── 路径配置 ────────────────────────────────────────────────────────────────
DATA_DIR   = "input"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NODES_FILE  = os.path.join(DATA_DIR, "AfricaNetworkNodes.csv")
EDGES_FILE  = os.path.join(DATA_DIR, "AfricaNetworkEdges.csv")
AFRICAPOLIS = os.path.join(DATA_DIR, "Africapolis_2050.csv")

# ── 参数 ────────────────────────────────────────────────────────────────────
GRAVITY_BETA = 2.8
N_TOP_PAIRS  = 25_000   # 与R脚本一致
PSI_VALUES   = np.arange(0, 2.1, 0.2)   # 完整11个点   # 0, 0.2, 0.4, ... 2.0

# ════════════════════════════════════════════════════════════════════════════
# 1. 读取数据
# ════════════════════════════════════════════════════════════════════════════
print("读取数据...")
nodes = pd.read_csv(NODES_FILE)
edges = pd.read_csv(EDGES_FILE)
afr   = pd.read_csv(AFRICAPOLIS)

# ════════════════════════════════════════════════════════════════════════════
# 2. 合并Pop2050到nodes（与africa_transport.py一致）
# ════════════════════════════════════════════════════════════════════════════
print("合并2050年人口数据...")
afp = afr[["Agglomeration_ID", "Population_2050"]].rename(
    columns={"Agglomeration_ID": "name", "Population_2050": "Pop2050"})
nodes = nodes.merge(afp, on="name", how="left")
nodes["Pop2050"] = nodes["Pop2050"].fillna(0)

# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
print("匹配Africapolis面积数据，计算psi_city...")
city_mask  = nodes["Pop2015"] > 0
city_nodes = nodes[city_mask].copy().reset_index(drop=True)

tree = cKDTree(afr[["Longitude", "Latitude"]].values)
dist, idx = tree.query(city_nodes[["x", "y"]].values, k=1)

city_nodes["Built_up_2015"]     = afr.iloc[idx]["Built up_2015"].values
city_nodes["Built_up_2045"]     = afr.iloc[idx]["Built up_2045"].values
city_nodes["Built_up_2050_raw"] = afr.iloc[idx]["Built up_2050"].values
city_nodes["match_dist"]        = dist

# 修复Built_up_2050异常值：>2×Built_up_2045时用Built_up_2045替代
# Nairobi等467个城市存在此问题，需向coauthor和OECD确认
anomaly = city_nodes["Built_up_2050_raw"] > 2 * city_nodes["Built_up_2045"]
city_nodes["Built_up_2050"] = city_nodes["Built_up_2050_raw"].copy()
city_nodes.loc[anomaly, "Built_up_2050"] = city_nodes.loc[anomaly, "Built_up_2045"]
print(f"  Built_up_2050异常城市（用2045替代）: {anomaly.sum()}")

# 城市半径（km）= sqrt(面积 km²)
city_nodes["r_2015"] = np.sqrt(city_nodes["Built_up_2015"].clip(lower=0))
city_nodes["r_2050"] = np.sqrt(city_nodes["Built_up_2050"].clip(lower=0))

# psi_city：每个城市的半径增长比例，供后续scenario标定用
valid = city_nodes["r_2015"] > 0
city_nodes.loc[valid,  "psi_city"] = (
    (city_nodes.loc[valid, "r_2050"] - city_nodes.loc[valid, "r_2015"])
    / city_nodes.loc[valid, "r_2015"]
)
city_nodes.loc[~valid, "psi_city"] = np.nan

psi_ref_median = city_nodes["psi_city"].median()
psi_ref_mean   = city_nodes["psi_city"].mean()
print(f"  psi_city（Africapolis默认2050）: 中位数={psi_ref_median:.3f}, 均值={psi_ref_mean:.3f}")
print(f"  城市半径统计（km）: 中位数={city_nodes['r_2050'].median():.1f}, 最大={city_nodes['r_2050'].max():.1f}")

city_nodes[["name", "agglosName", "ISO3", "Pop2015", "Pop2050",
            "Built_up_2015", "Built_up_2050", "r_2015", "r_2050", "psi_city"]].to_csv(
    os.path.join(OUTPUT_DIR, "city_psi_values.csv"), index=False)

# ════════════════════════════════════════════════════════════════════════════
# 4. 建基础图结构（与africa_transport.py的build_graph一致）
# ════════════════════════════════════════════════════════════════════════════
print("建图...")
node_names   = nodes["name"].astype(str).tolist()
name_to_idx  = {str(n): i for i, n in enumerate(nodes["name"])}

# 建城市半径查找表 name→r_2050（km）
city_radius = dict(zip(city_nodes["name"].astype(str), city_nodes["r_2050"].values))

# 每条边的urban_frac：
#   受城市影响的路段长度 = min(r_from + r_to, l)
#   非城市端点r=0，所以非城市边urban_frac=0
edge_l  = edges["l"].values
r_from  = edges["from"].astype(str).map(city_radius).fillna(0).values
r_to    = edges["to"].astype(str).map(city_radius).fillna(0).values
urban_len  = np.minimum(r_from + r_to, edge_l)
urban_frac = np.where(edge_l > 0, urban_len / edge_l, 0.0)

has_urban = urban_frac > 0
time_origin = edges["time"].values.copy()

from_idx_arr = edges["from"].astype(str).map(name_to_idx).values.astype(int)
to_idx_arr   = edges["to"].astype(str).map(name_to_idx).values.astype(int)
edge_tuples  = list(zip(from_idx_arr, to_idx_arr))

print(f"  总节点: {len(nodes)}, 城市节点: {city_mask.sum()}")
print(f"  总边: {len(edges)}, 含城市路段的边: {has_urban.sum()} ({has_urban.mean():.1%})")
print(f"  urban_frac（含城市路段的边）: 中位数={np.median(urban_frac[has_urban]):.3f}, "
      f"均值={urban_frac[has_urban].mean():.3f}")

# ════════════════════════════════════════════════════════════════════════════
# 5. 计算引力对（与africa_transport.py的compute_gravity一致）
#    用 psi=0 时的距离矩阵排序，固定top-25000对
#    注意：重新计算gravity对时需要基于当前psi的distance，
#    但为了和R脚本一致，GM的排序固定用baseline time
# ════════════════════════════════════════════════════════════════════════════
print("\n计算引力对（基于baseline time排序）...")

city_idx_arr = np.where(nodes["Pop2015"].values > 0)[0]
pop2015 = nodes["Pop2015"].values.astype(float)
pop2050 = nodes["Pop2050"].values.astype(float)

# 建baseline图（psi=0）
G_base = ig.Graph(n=len(nodes), directed=False)
G_base.add_edges(edge_tuples)
G_base.es["time"] = time_origin.tolist()

print("  计算baseline距离矩阵...")
D_full = np.array(G_base.distances(source=city_idx_arr.tolist(), weights="time"))
D      = D_full[:, city_idx_arr]   # shape (n_cities, n_cities)

P15 = pop2015[city_idx_arr] / 1000.0
P50 = pop2050[city_idx_arr] / 1000.0

# 引力矩阵（与R一致：+1防止除零）
Grav15 = np.outer(P15, P15) / (D ** GRAVITY_BETA + 1)
Grav50 = np.outer(P50, P50) / (D ** GRAVITY_BETA + 1)

# 下三角（无向，无对角线）
rows_i, cols_j = np.tril_indices(len(city_idx_arr), k=-1)
GM = pd.DataFrame({
    "from":    city_idx_arr[cols_j],
    "to":      city_idx_arr[rows_i],
    "GravM":   Grav15[rows_i, cols_j],
    "GravM50": Grav50[rows_i, cols_j],
}).sort_values("GravM", ascending=False).reset_index(drop=True)

GM_top = GM.iloc[N_TOP_PAIRS : 2*N_TOP_PAIRS].copy()
print(f"  总城市对: {len(GM):,} | 跳过前{N_TOP_PAIRS:,}对（极近邻）| 使用第{N_TOP_PAIRS+1:,}~{2*N_TOP_PAIRS:,}对")

# ════════════════════════════════════════════════════════════════════════════
# 6. 核心函数：给定psi，计算全部指标
# ════════════════════════════════════════════════════════════════════════════

def compute_metrics(psi):
    """
    给定psi，重建图，计算：
      - gravity_weighted_avg_time（引力加权平均travel time，主指标）
      - total_betweenness
      - vulnerability_index
      - edge betweenness数组（用于空间分析）
    """
    # time_new = time * (1 + urban_frac * psi)
    # ψ=0 → time_new = time（无任何城市惩罚）
    # ψ>0 → 城市路段按半径比例减速，非城市段不变
    time_new = time_origin * (1.0 + urban_frac * psi)

    # 建新图
    G = ig.Graph(n=len(nodes), directed=False)
    G.add_edges(edge_tuples)
    G.es["time_new"] = time_new.tolist()

    # ── 引力加权平均travel time ──────────────────────────────────────────
    # 重算城市间距离矩阵（用新时间权重）
    D_new_full = np.array(G.distances(source=city_idx_arr.tolist(), weights="time_new"))
    D_new      = D_new_full[:, city_idx_arr]

    # 引力权重（分母+1，与R一致，P50用2050人口）
    Grav50_new = np.outer(P50, P50) / (D_new ** GRAVITY_BETA + 1)

    rows_i_t, cols_j_t = np.tril_indices(len(city_idx_arr), k=-1)
    g_weights = Grav50_new[rows_i_t, cols_j_t]
    d_vals    = D_new[rows_i_t, cols_j_t]

    reachable = np.isfinite(d_vals) & (d_vals > 0)
    gw_avg_time = np.average(d_vals[reachable], weights=g_weights[reachable])

    # ── 引力加权边betweenness（与africa_transport.py的compute_betweenness一致）
    n_edges   = len(edges)
    between50 = np.zeros(n_edges)

    from_arr   = GM_top["from"].values
    to_arr     = GM_top["to"].values
    gm50_arr   = GM_top["GravM50"].values

    for k in range(N_TOP_PAIRS):
        paths = G.get_shortest_paths(
            v       = int(from_arr[k]),
            to      = int(to_arr[k]),
            weights = "time_new",
            output  = "epath",
        )
        edge_ids = paths[0]
        if not edge_ids:
            continue
        between50[np.array(edge_ids, dtype=int)] += gm50_arr[k]

    total_eb = between50.sum()
    threshold = np.percentile(between50[between50 > 0], 90) if (between50 > 0).any() else 0
    vuln_index = between50[between50 >= threshold].sum() / total_eb if total_eb > 0 else 0

    return gw_avg_time, total_eb, vuln_index, between50


# ════════════════════════════════════════════════════════════════════════════
# 7. 扫描 psi
# ════════════════════════════════════════════════════════════════════════════
print(f"\n开始扫描 psi = {[round(p,2) for p in PSI_VALUES]}...")
print(f"每个psi需要跑{N_TOP_PAIRS:,}对最短路径，预计每个psi约20-30秒，全程约5分钟\n")

records = []
betweenness_by_psi = {}

for psi in tqdm(PSI_VALUES, desc="psi sweep"):
    psi_r = round(psi, 2)
    gw_time, total_eb, vuln, eb_arr = compute_metrics(psi)
    records.append({
        "psi":               psi_r,
        "gw_avg_time":       gw_time,
        "total_betweenness": total_eb,
        "vulnerability_index": vuln,
    })
    betweenness_by_psi[psi_r] = eb_arr
    tqdm.write(f"  psi={psi:.1f} | gw_avg_time={gw_time:.4f} | "
               f"total_eb={total_eb:.3e} | vuln={vuln:.4f}")

results = pd.DataFrame(records)
results.to_csv(os.path.join(OUTPUT_DIR, "psi_sweep_results.csv"), index=False)
print(f"\n结果已保存: {OUTPUT_DIR}/psi_sweep_results.csv")

# ════════════════════════════════════════════════════════════════════════════
# 8. 折线图
# ════════════════════════════════════════════════════════════════════════════
print("生成折线图...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Africa Transport Network: Urban Expansion (ψ) Impact on Travel Burden (2050 Population)",
    fontsize=12)

psi_ticks = results["psi"].values

# 主指标
ax = axes[0]
ax.plot(results["psi"], results["gw_avg_time"], "b-o", linewidth=2, markersize=6)
ax.axvline(x=0, color="gray", linestyle="--", alpha=0.6, label="ψ=0 (2015 timeU baseline)")
ax.set_xlabel("ψ (urban penalty multiplier beyond 2015)")
ax.set_ylabel("Gravity-weighted avg travel time")
ax.set_title("Main Indicator")
ax.set_xticks(psi_ticks)
ax.set_xticklabels([f"{p:.1f}" for p in psi_ticks], rotation=45, fontsize=7)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 总betweenness
ax = axes[1]
ax.plot(results["psi"], results["total_betweenness"] / 1e6, "g-o", linewidth=2, markersize=6)
ax.set_xlabel("ψ (urban penalty multiplier beyond 2015)")
ax.set_ylabel("Total edge betweenness (×10⁶)")
ax.set_title("Total Network Load")
ax.set_xticks(psi_ticks)
ax.set_xticklabels([f"{p:.1f}" for p in psi_ticks], rotation=45, fontsize=7)
ax.grid(True, alpha=0.3)

# 脆弱性
ax = axes[2]
ax.plot(results["psi"], results["vulnerability_index"], "r-o", linewidth=2, markersize=6)
ax.set_xlabel("ψ (urban penalty multiplier beyond 2015)")
ax.set_ylabel("Vulnerability index (top 10% share)")
ax.set_title("Network Vulnerability")
ax.set_xticks(psi_ticks)
ax.set_xticklabels([f"{p:.1f}" for p in psi_ticks], rotation=45, fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "psi_sweep_chart.png"), dpi=150, bbox_inches="tight")
print(f"图表已保存: {OUTPUT_DIR}/psi_sweep_chart.png")

# ════════════════════════════════════════════════════════════════════════════
# 9. 空间分析：betweenness增量（psi=0 vs psi=1.0）
# ════════════════════════════════════════════════════════════════════════════
print("计算betweenness增量（psi=0 vs psi=1.0）...")

eb0  = betweenness_by_psi.get(0.0,  betweenness_by_psi[list(betweenness_by_psi.keys())[0]])
eb1  = betweenness_by_psi.get(1.0, betweenness_by_psi[list(betweenness_by_psi.keys())[-1]])

edges_out = edges.copy()
edges_out["betweenness_psi0"]    = eb0
edges_out["betweenness_psi1"]    = eb1
edges_out["betweenness_delta"]   = eb1 - eb0
edges_out["betweenness_pct_chg"] = (eb1 - eb0) / (eb0 + 1e-9) * 100

edges_out.to_csv(os.path.join(OUTPUT_DIR, "edges_betweenness_delta.csv"), index=False)
print(f"边增量数据已保存: {OUTPUT_DIR}/edges_betweenness_delta.csv")

print("\n✓ 全部完成！")
print("\n输出文件：")
print(f"  {OUTPUT_DIR}/psi_sweep_results.csv        — 折线图数据")
print(f"  {OUTPUT_DIR}/psi_sweep_chart.png          — 折线图（3个指标）")
print(f"  {OUTPUT_DIR}/city_psi_values.csv          — 每城市psi_city")
print(f"  {OUTPUT_DIR}/edges_betweenness_delta.csv  — 边负荷增量")
