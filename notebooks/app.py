"""
Africa Transport Network — Interactive Dashboard  v3
====================================================
Run:   python app.py
Open:  http://127.0.0.1:8050

五个scenario（年份语义固定，不再有year切换控件）：
  baseline        — 2015人口 + 2015城区  → Between
  pop2050         — 2050人口 + 2015城区  → Between50  (同一个baseline pkl)
  density_low     — 2050人口 + 低密度扩张城区  → Between50
  density_cons    — 2050人口 + 中等密度扩张城区 → Between50
  density_high    — 2050人口 + 紧凑密度扩张城区 → Between50

所有scenario都用Between50，只有baseline用Between。
对比逻辑：baseline vs 其他四个。

修改pkl文件名：
  在下方 SCENARIO_REGISTRY 里把 "path" 改成你本地实际的文件名。
"""

import os, glob, pickle
import numpy as np
import pandas as pd
import igraph as ig

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 路径 ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
INPUT_DIR  = "input"

# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 定义
# 修改 path 为你本地实际的pkl文件名
# year_attr: 从该pkl里读哪个属性作为拥堵值
# color: 在对比图里用什么颜色
# ══════════════════════════════════════════════════════════════════════════════
SCENARIO_REGISTRY = {
    "baseline": {
        "label":     "Baseline — 2015 pop, 2015 urban extent",
        "path":      "output/Network_baseline.pkl",
        "year_attr": "Between",       # 2015人口
        "color":     "#555555",
        "dash":      "solid",
    },
    "pop2050": {
        "label":     "Pop growth only — 2050 pop, 2015 urban extent",
        "path":      "output/Network_baseline.pkl",   # 同一个pkl，不同属性
        "year_attr": "Between50",     # 2050人口
        "color":     "#3498db",
        "dash":      "dash",
    },
    "density_low": {
        "label":     "Low density 2050 — sprawl + 2050 pop",
        "path":      "output/Network_urban_low_2050.pkl",   # ← 改成你的文件名
        "year_attr": "Between50",
        "color":     "#2ecc71",
        "dash":      "solid",
    },
    "density_cons": {
        "label":     "Consolidated density 2050 — medium + 2050 pop",
        "path":      "output/Network_urban_cons_2050.pkl",  # ← 改成你的文件名
        "year_attr": "Between50",
        "color":     "#f39c12",
        "dash":      "solid",
    },
    "density_high": {
        "label":     "High density 2050 — compact + 2050 pop",
        "path":      "output/Network_urban_high_2050.pkl",  # ← 改成你的文件名
        "year_attr": "Between50",
        "color":     "#e74c3c",
        "dash":      "solid",
    },
}

# 只保留pkl文件实际存在的scenario
SCENARIOS = {
    k: v for k, v in SCENARIO_REGISTRY.items()
    if os.path.exists(v["path"])
}
print(f"可用scenario ({len(SCENARIOS)}/{len(SCENARIO_REGISTRY)}):")
for k, v in SCENARIOS.items():
    print(f"  {k:20s} → {v['path']}  [{v['year_attr']}]")

missing = [k for k in SCENARIO_REGISTRY if k not in SCENARIOS]
if missing:
    print(f"缺少pkl（跑完urban_penalty后补充）: {missing}")

SC_KEYS    = list(SCENARIOS.keys())
SC_OPTIONS = [{"label": v["label"], "value": k} for k, v in SCENARIOS.items()]

# ── 静态数据 ──────────────────────────────────────────────────────────────────
edges_df  = pd.read_csv(os.path.join(OUTPUT_DIR, "edges_for_app.csv"))
cities_df = pd.read_csv(os.path.join(OUTPUT_DIR, "cities_for_app.csv"))

MAJOR_CITIES = sorted(
    cities_df[cities_df["Pop2015"] > 300_000]["agglosName"].dropna().unique()
)

# ── graph 缓存 ────────────────────────────────────────────────────────────────
_GCACHE: dict = {}

def get_G(key: str) -> ig.Graph:
    # baseline 和 pop2050 共用同一个pkl，只需加载一次
    path = SCENARIOS[key]["path"]
    if path not in _GCACHE:
        print(f"  加载 {path} ...")
        _GCACHE[path] = pickle.load(open(path, "rb"))
    return _GCACHE[path]

# 最短路径固定用baseline
G_base   = get_G("baseline")
vdf_base = G_base.get_vertex_dataframe()
NAME2IDX = {
    r["agglosName"]: i
    for i, r in vdf_base.iterrows()
    if pd.notna(r.get("agglosName"))
}

# ── 取拥堵值 ──────────────────────────────────────────────────────────────────
def get_between(key: str) -> np.ndarray:
    """每个scenario自己决定用Between还是Between50。"""
    attr = SCENARIOS[key]["year_attr"]
    G    = get_G(key)
    return np.array(G.es[attr], dtype=float)

# ── 视觉常量 ──────────────────────────────────────────────────────────────────
REGION_COLORS = {
    "South": "#FFC107", "East": "#7A98D6",
    "Central": "#E05C5C", "West": "#00CC99", "North": "#CC90C1",
}
GEO_LAYOUT = dict(
    scope="africa", showland=True, landcolor="#f0ece4",
    showocean=True, oceancolor="#daeef8",
    showcoastlines=True, coastlinecolor="#bbb",
    showcountries=True, countrycolor="#ddd",
    projection_type="mercator",
    lonaxis_range=[-20, 55], lataxis_range=[-36, 38],
)
CARD  = {"background":"#fafafa","borderRadius":"8px","padding":"10px 14px",
         "boxShadow":"0 1px 3px rgba(0,0,0,.1)","marginBottom":"8px"}
FLEX  = {"display":"flex","alignItems":"flex-end","flexWrap":"wrap","gap":"10px"}
LBL   = {"fontWeight":"bold","fontSize":"0.78rem","marginBottom":"3px"}

# ── 边宽渲染 ──────────────────────────────────────────────────────────────────
def log_w(vals: np.ndarray, hi: float = 7.0) -> np.ndarray:
    mx = vals.max()
    return np.log1p(np.clip(vals, 0, None)) / np.log1p(mx + 1) * hi if mx > 0 \
           else np.zeros(len(vals))

def edge_traces(key: str, min_log: float) -> list:
    b    = get_between(key)
    mv   = 10 ** min_log if min_log > 0 else 0
    mask = b > mv
    b_s  = b[mask]
    e_s  = edges_df[mask].reset_index(drop=True)
    ws   = log_w(b_s)
    traces, prev = [], 0.0
    for q, lw in zip(np.nanquantile(ws, [.2,.4,.6,.8,1.]), [1.,1.8,2.8,4.2,6.5]):
        sel  = (ws > prev) & (ws <= q)
        rows = e_s[sel]
        if len(rows):
            lo, la = [], []
            for _, r in rows.iterrows():
                lo += [r.x0, r.x1, None]
                la += [r.y0, r.y1, None]
            traces.append(go.Scattergeo(
                lon=lo, lat=la, mode="lines",
                line=dict(width=lw, color="rgba(60,60,60,0.55)"),
                hoverinfo="skip", showlegend=False,
            ))
        prev = q
    return traces

def city_traces(show: bool) -> list:
    if not show:
        return []
    traces = []
    # baseline은 항상 2015人口
    for region, grp in cities_df.groupby("Region"):
        sz = np.log1p(grp["Pop2015"]) * 0.55
        traces.append(go.Scattergeo(
            lon=grp["x"], lat=grp["y"], mode="markers",
            marker=dict(size=sz, color=REGION_COLORS.get(region,"#999"),
                        opacity=0.85, line=dict(width=0.4, color="black")),
            text=grp["agglosName"],
            hovertemplate="<b>%{text}</b><extra></extra>",
            name=region, legendgroup=region,
        ))
    return traces

def base_map_layout(uirev="map"):
    return dict(
        geo=GEO_LAYOUT,
        margin=dict(l=0,r=0,t=0,b=0),
        legend=dict(orientation="v", x=0.01, y=0.99,
                    bgcolor="rgba(255,255,255,0.75)", font=dict(size=10)),
        uirevision=uirev,
        paper_bgcolor="white",
    )

# ── 走廊最短路径 ──────────────────────────────────────────────────────────────
def corridor_xy(origin: str, dest: str, key: str):
    oi = NAME2IDX.get(origin)
    di = NAME2IDX.get(dest)
    if oi is None or di is None:
        return None, None
    vpath = G_base.get_shortest_paths(oi, di, weights=G_base.es["l"],
                                       output="vpath")[0]
    if len(vpath) < 2:
        return None, None
    G_sc = get_G(key)
    attr = SCENARIOS[key]["year_attr"]
    xs, ys, cum = [], [], 0.0
    for i in range(len(vpath)-1):
        eid   = G_base.get_eid(vpath[i], vpath[i+1])
        l_val = G_base.es[eid]["l"]
        b_val = max(G_sc.es[eid][attr], 0)
        xs   += [cum, cum+l_val]
        ys   += [b_val+1, b_val+1]
        cum  += l_val
    return xs, ys


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
def sc_drop(id_, default=None):
    return dcc.Dropdown(id=id_, options=SC_OPTIONS,
                        value=default or SC_KEYS[0], clearable=False,
                        style={"width":"310px","fontSize":"0.85rem"})

def sl_ctrl(id_):
    return dcc.Slider(id=id_, min=0, max=7, step=0.5, value=0,
                      marks={i: f"10^{i}" for i in range(8)},
                      tooltip={"placement":"bottom"})

app = dash.Dash(__name__, title="Africa Transport")
server = app.server

app.layout = html.Div([

    # 顶栏
    html.Div([
        html.Span("🌍", style={"fontSize":"1.5rem","marginRight":"10px"}),
        html.Div([
            html.H2("Africa Road Network — Congestion Dashboard",
                    style={"margin":0,"color":"white","fontSize":"1.2rem"}),
            html.P(
                "Baseline (2015 pop + 2015 urban)  vs  "
                "Pop growth only  vs  Low / Consolidated / High density 2050",
                style={"margin":"2px 0 0","color":"#9ab","fontSize":"0.78rem"}),
        ]),
    ], style={"background":"linear-gradient(90deg,#1a1a2e,#16213e)",
              "padding":"10px 18px","display":"flex","alignItems":"center"}),

    dcc.Tabs(value="map", children=[

        # ── TAB 1: 单scenario地图 ─────────────────────────────────────────────
        dcc.Tab(label="🗺  Map", value="map", children=[
            html.Div([
                html.Div([
                    html.Div([html.Label("Scenario", style=LBL),
                              sc_drop("sc-map")], style={"marginRight":"18px"}),
                    html.Div([html.Label("Min congestion", style=LBL),
                              sl_ctrl("sl-map")], style={"width":"220px","marginRight":"18px"}),
                    html.Div([
                        html.Label("Cities", style=LBL),
                        dcc.Checklist(id="cities-map",
                            options=[{"label":" show (sized by pop)","value":"yes"}],
                            value=["yes"], labelStyle={"fontSize":"0.9rem"}),
                    ]),
                ], style={**CARD, **FLEX}),
                dcc.Graph(id="map-fig", style={"height":"72vh"},
                          config={"scrollZoom":True}),
            ], style={"padding":"10px 14px"}),
        ]),

        # ── TAB 2: 两个scenario并排对比 ──────────────────────────────────────
        dcc.Tab(label="⚖  Compare", value="compare", children=[
            html.Div([
                html.Div([
                    html.Div([html.Label("Left", style=LBL),
                              sc_drop("sc-left", SC_KEYS[0])],
                             style={"marginRight":"18px"}),
                    html.Div([html.Label("Right", style=LBL),
                              sc_drop("sc-right", SC_KEYS[min(1,len(SC_KEYS)-1)])],
                             style={"marginRight":"18px"}),
                    html.Div([html.Label("Min congestion", style=LBL),
                              sl_ctrl("sl-cmp")], style={"width":"220px"}),
                ], style={**CARD, **FLEX}),
                dcc.Graph(id="cmp-fig", style={"height":"72vh"},
                          config={"scrollZoom":True}),
            ], style={"padding":"10px 14px"}),
        ]),

        # ── TAB 3: 散点图（baseline 2015 vs 选定scenario）─────────────────────
        dcc.Tab(label="📈  Baseline vs Scenario", value="scatter", children=[
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Compare baseline against", style=LBL),
                        sc_drop("sc-scatter", SC_KEYS[min(1,len(SC_KEYS)-1)]),
                    ], style={"marginRight":"18px"}),
                    html.P(
                        "X轴 = baseline拥堵（2015人口+2015城区）  "
                        "Y轴 = 所选scenario拥堵  "
                        "红线 = 无变化参照",
                        style={"color":"#666","fontSize":"0.82rem","margin":"0"}),
                ], style={**CARD, **FLEX}),
                dcc.Graph(id="scatter-fig"),
            ], style={"padding":"10px 14px"}),
        ]),

        # ── TAB 4: 走廊剖面 ───────────────────────────────────────────────────
        dcc.Tab(label="🛣  Corridor", value="corridor", children=[
            html.Div([
                html.Div([
                    html.Div([html.Label("Origin", style=LBL),
                              dcc.Dropdown(id="orig-drop",
                                  options=[{"label":c,"value":c} for c in MAJOR_CITIES],
                                  value="Lagos", clearable=False,
                                  style={"width":"190px","fontSize":"0.85rem"})],
                             style={"marginRight":"14px"}),
                    html.Div([html.Label("Destination", style=LBL),
                              dcc.Dropdown(id="dest-drop",
                                  options=[{"label":c,"value":c} for c in MAJOR_CITIES],
                                  value="Abidjan", clearable=False,
                                  style={"width":"190px","fontSize":"0.85rem"})],
                             style={"marginRight":"14px"}),
                    html.Div([html.Label("Single scenario", style=LBL),
                              sc_drop("sc-corr")], style={"marginRight":"14px"}),
                    html.Div([
                        html.Label("Overlay", style=LBL),
                        dcc.Checklist(id="overlay-all",
                            options=[{"label":" all scenarios","value":"yes"}],
                            value=[],
                            labelStyle={"fontSize":"0.9rem"}),
                    ]),
                ], style={**CARD, **FLEX}),

                # 说明文字
                html.P([
                    "路径按 baseline 最短距离计算。",
                    html.Br(),
                    "Y轴 = 拥堵值（log），代表该路段承载的潜在交通压力（不是真实流量）。",
                ], style={"color":"#666","fontSize":"0.82rem","margin":"0 0 8px 4px"}),

                dcc.Graph(id="corr-fig"),
            ], style={"padding":"10px 14px"}),
        ]),

    ]),
], style={"fontFamily":"sans-serif"})


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(Output("map-fig","figure"),
              Input("sc-map","value"),
              Input("cities-map","value"),
              Input("sl-map","value"))
def cb_map(sc, cities, sl):
    fig = go.Figure()
    for t in edge_traces(sc, sl):
        fig.add_trace(t)
    for t in city_traces("yes" in (cities or [])):
        fig.add_trace(t)
    fig.update_layout(**base_map_layout("map"))
    return fig


@app.callback(Output("cmp-fig","figure"),
              Input("sc-left","value"),
              Input("sc-right","value"),
              Input("sl-cmp","value"))
def cb_compare(sc_l, sc_r, sl):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type":"geo"},{"type":"geo"}]],
        subplot_titles=[SCENARIOS[sc_l]["label"], SCENARIOS[sc_r]["label"]],
    )
    for t in edge_traces(sc_l, sl):
        fig.add_trace(t, row=1, col=1)
    for t in edge_traces(sc_r, sl):
        fig.add_trace(t, row=1, col=2)
    fig.update_geos(GEO_LAYOUT)
    fig.update_layout(
        height=650, margin=dict(l=0,r=0,t=35,b=0),
        paper_bgcolor="white", uirevision="cmp",
    )
    return fig


@app.callback(Output("scatter-fig","figure"),
              Input("sc-scatter","value"))
def cb_scatter(sc_other):
    # X轴固定是baseline
    b_base  = get_between("baseline") + 1
    b_other = get_between(sc_other)   + 1
    mask = (b_base > 1) & (b_other > 1)

    htypes = edges_df["h"].values[mask]
    palette = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c","#e67e22"]
    fig = go.Figure()
    for i, rt in enumerate(sorted(set(htypes))):
        sel = htypes == rt
        fig.add_trace(go.Scattergl(
            x=b_base[mask][sel], y=b_other[mask][sel],
            mode="markers",
            marker=dict(size=4, color=palette[i%len(palette)], opacity=0.5),
            name=rt,
            hovertemplate=f"<b>{rt}</b><br>baseline: %{{x:.0f}}<br>{sc_other}: %{{y:.0f}}<extra></extra>",
        ))
    lim = max(b_base[mask].max(), b_other[mask].max()) * 1.3
    fig.add_trace(go.Scatter(
        x=[1,lim], y=[1,lim], mode="lines",
        line=dict(color="red", width=1.2, dash="dash"),
        showlegend=False,
    ))
    other_label = SCENARIOS[sc_other]["label"]
    fig.update_layout(
        xaxis=dict(type="log", title="Baseline congestion (2015 pop + 2015 urban)"),
        yaxis=dict(type="log", title=f"{other_label}"),
        legend=dict(title="Road type", font=dict(size=10)),
        margin=dict(l=55,r=10,t=10,b=55),
        height=530, paper_bgcolor="white",
    )
    return fig


@app.callback(Output("corr-fig","figure"),
              Input("orig-drop","value"),
              Input("dest-drop","value"),
              Input("sc-corr","value"),
              Input("overlay-all","value"))
def cb_corridor(origin, dest, sc, overlay):
    if not origin or not dest:
        return go.Figure()

    fig = go.Figure()

    if overlay and "yes" in overlay:
        # 所有scenario叠加
        for key, info in SCENARIOS.items():
            xs, ys = corridor_xy(origin, dest, key)
            if xs is None:
                continue
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=info["color"], width=2.2, dash=info["dash"]),
                name=info["label"],
                hovertemplate="dist %{x:.0f} km | congestion %{y:.0f}<extra></extra>",
            ))
        title = f"{origin} → {dest}  ·  All scenarios"
    else:
        xs, ys = corridor_xy(origin, dest, sc)
        if xs is None:
            return go.Figure()
        info = SCENARIOS[sc]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="tozeroy", mode="lines",
            line=dict(color=info["color"], width=1.0),
            fillcolor=info["color"].replace(")", ",0.25)").replace("rgb","rgba")
                if info["color"].startswith("rgb")
                else info["color"] + "40",
            hovertemplate="dist %{x:.0f} km | congestion %{y:.0f}<extra></extra>",
            showlegend=False,
        ))
        title = f"{origin} → {dest}  ·  {info['label']}"

    fig.update_layout(
        xaxis_title=f"Distance from {origin} (km)",
        yaxis=dict(title="Congestion + 1 (log scale)", type="log"),
        title=dict(text=title, font=dict(size=12)),
        legend=dict(font=dict(size=10)),
        margin=dict(l=55,r=10,t=42,b=50),
        height=440, paper_bgcolor="white",
    )
    return fig


# ── 启动 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*52}")
    print(f"  Africa Transport Dashboard  v3")
    print(f"  可用scenario: {', '.join(SC_KEYS)}")
    print(f"  → http://127.0.0.1:8050")
    print(f"{'='*52}\n")
    app.run(debug=False, port=8050)
