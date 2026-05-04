"""
Africa Transport Network — Interactive Dashboard  v2
====================================================
Run:   python app.py
Open:  http://127.0.0.1:8050

Tabs
----
1. Network Map   — full Africa map, edge width = congestion
2. Compare       — side-by-side any two scenarios
3. 2015 vs 2050  — scatter, coloured by road type
4. Corridor      — origin → destination road profile (+ overlay all scenarios)

Auto-detects all Network_*.pkl in output/ — drop new ones and restart.
"""

import os, glob, pickle
import numpy as np
import pandas as pd
import igraph as ig

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "../output"
INPUT_DIR  = "../input"

# ── auto-detect available scenarios ──────────────────────────────────────────
LABEL_MAP = {
    "baseline":          "Baseline (intercity speeds only)",
    "urban_2015":        "Urban penalty — 2015 footprint",
    "urban_CB_2015":     "Urban penalty + cross-border",
    "density_high2050":  "High density 2050 (compact)",
    "density_cons2050":  "Consolidated density 2050",
    "density_low2050":   "Low density 2050 (sprawl)",
}

SCENARIOS = {}
for pkl in sorted(glob.glob(os.path.join(OUTPUT_DIR, "Network_*.pkl"))):
    key = os.path.basename(pkl).replace("Network_", "").replace(".pkl", "")
    SCENARIOS[key] = {"label": LABEL_MAP.get(key, key), "path": pkl}

print(f"Detected scenarios: {list(SCENARIOS.keys())}")

SC_OPTIONS = [{"label": v["label"], "value": k} for k, v in SCENARIOS.items()]
SC_KEYS    = list(SCENARIOS.keys())

# ── load pre-exported edge/city CSVs ─────────────────────────────────────────
edges_df  = pd.read_csv(os.path.join(OUTPUT_DIR, "edges_for_app.csv"))
cities_df = pd.read_csv(os.path.join(OUTPUT_DIR, "cities_for_app.csv"))

# ── graph cache ───────────────────────────────────────────────────────────────
_GCACHE: dict = {}

def get_G(key: str) -> ig.Graph:
    if key not in _GCACHE:
        _GCACHE[key] = pickle.load(open(SCENARIOS[key]["path"], "rb"))
    return _GCACHE[key]

G_base  = get_G("baseline")
vdf_b   = G_base.get_vertex_dataframe()
NAME2IDX = {
    r["agglosName"]: i
    for i, r in vdf_b.iterrows()
    if pd.notna(r.get("agglosName"))
}
MAJOR_CITIES = sorted(
    cities_df[cities_df["Pop2015"] > 300_000]["agglosName"].dropna().unique()
)

# ── CSV column map for pre-computed scenarios ─────────────────────────────────
CSV_COL = {
    ("baseline",      "2015"): "between_baseline_2015",
    ("baseline",      "2050"): "between_baseline_2050",
    ("urban_2015",    "2015"): "between_urban_2015_2015",
    ("urban_2015",    "2050"): "between_urban_2015_2050",
    ("urban_CB_2015", "2015"): "between_urban_CB_2015_2015",
    ("urban_CB_2015", "2050"): "between_urban_CB_2015_2050",
}

def get_between(key: str, year: str) -> np.ndarray:
    col = CSV_COL.get((key, year))
    if col and col in edges_df.columns:
        return edges_df[col].values.astype(float)
    attr = "Between" if year == "2015" else "Between50"
    return np.array(get_G(key).es[attr], dtype=float)

# ── visual helpers ─────────────────────────────────────────────────────────────
REGION_COLORS = {
    "South": "#FFC107", "East": "#7A98D6",
    "Central": "#E05C5C", "West": "#00CC99", "North": "#CC90C1",
}
PALETTE = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c","#e67e22"]

def log_w(vals: np.ndarray, hi: float = 7.0) -> np.ndarray:
    mx = vals.max()
    return np.log1p(np.clip(vals, 0, None)) / np.log1p(mx + 1) * hi if mx > 0 \
           else np.zeros(len(vals))

def edge_traces(key: str, year: str, min_log: float) -> list:
    """Return Scattergeo traces for road edges (bucketed widths)."""
    b = get_between(key, year)
    mv = 10 ** min_log if min_log > 0 else 0
    mask = b > mv
    b_s  = b[mask];  e_s = edges_df[mask].reset_index(drop=True)
    ws   = log_w(b_s)
    traces = []
    if len(e_s) == 0:
        return traces
    for q, lw in zip(np.nanquantile(ws, [.2,.4,.6,.8,1.]), [1.,1.8,2.8,4.2,6.5]):
        sel = (ws <= q) & (ws > (traces[-1][0] if traces else -1))
        # simpler: re-bucket
    prev = 0.0
    for q, lw in zip(np.nanquantile(ws, [.2,.4,.6,.8,1.]), [1.,1.8,2.8,4.2,6.5]):
        sel  = (ws > prev) & (ws <= q)
        rows = e_s[sel]
        if len(rows) == 0:
            prev = q; continue
        lo, la = [], []
        for _, r in rows.iterrows():
            lo += [r.x0, r.x1, None]; la += [r.y0, r.y1, None]
        traces.append(go.Scattergeo(
            lon=lo, lat=la, mode="lines",
            line=dict(width=lw, color="rgba(60,60,60,0.55)"),
            hoverinfo="skip", showlegend=False,
        ))
        prev = q
    return traces

GEO_COMMON = dict(
    scope="africa", showland=True, landcolor="#f0ece4",
    showocean=True, oceancolor="#daeef8",
    showcoastlines=True, coastlinecolor="#bbb",
    showcountries=True, countrycolor="#ddd",
    projection_type="mercator",
    lonaxis_range=[-20, 55], lataxis_range=[-36, 38],
)

# ── style constants ───────────────────────────────────────────────────────────
CARD  = {"background":"#fafafa","borderRadius":"8px","padding":"10px 14px",
         "boxShadow":"0 1px 4px rgba(0,0,0,.1)","marginBottom":"8px"}
LBSTYLE = {"fontWeight":"bold","fontSize":"0.78rem","marginBottom":"3px"}
FLEX  = {"display":"flex","alignItems":"flex-end","flexWrap":"wrap","gap":"10px"}

def sc_ctrl(id_, default=None):
    return dcc.Dropdown(id=id_, options=SC_OPTIONS,
                        value=default or SC_KEYS[0], clearable=False,
                        style={"width":"270px","fontSize":"0.85rem"})

def yr_ctrl(id_):
    return dcc.RadioItems(id=id_,
        options=[{"label":"2015","value":"2015"},{"label":"2050","value":"2050"}],
        value="2015", inline=True,
        inputStyle={"marginRight":"4px"},
        labelStyle={"marginRight":"14px","fontSize":"0.9rem"})

def sl_ctrl(id_):
    return dcc.Slider(id=id_, min=0, max=6, step=0.5, value=0,
                      marks={i:f"10^{i}" for i in range(7)},
                      tooltip={"placement":"bottom"})


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(__name__, title="Africa Transport")
server = app.server

app.layout = html.Div([

    # header
    html.Div([
        html.Span("🌍", style={"fontSize":"1.5rem","marginRight":"10px"}),
        html.Div([
            html.H2("Africa Road Network — Congestion Dashboard",
                    style={"margin":0,"color":"white","fontSize":"1.2rem"}),
            html.P("Gravity-weighted betweenness · 2015 vs 2050 · Multiple density scenarios",
                   style={"margin":"1px 0 0","color":"#9ab","fontSize":"0.78rem"}),
        ]),
    ], style={"background":"linear-gradient(90deg,#1a1a2e,#16213e)",
              "padding":"10px 18px","display":"flex","alignItems":"center"}),

    dcc.Tabs(value="map", children=[

        # ── TAB 1: Map ────────────────────────────────────────────────────────
        dcc.Tab(label="🗺  Map", value="map", children=[
            html.Div([
                html.Div([
                    html.Div([html.Label("Scenario",style=LBSTYLE), sc_ctrl("sc-map")],
                             style={"marginRight":"18px"}),
                    html.Div([html.Label("Year",style=LBSTYLE), yr_ctrl("yr-map")],
                             style={"marginRight":"18px"}),
                    html.Div([html.Label("Min congestion",style=LBSTYLE), sl_ctrl("sl-map")],
                             style={"width":"200px","marginRight":"18px"}),
                    html.Div([
                        html.Label("Cities",style=LBSTYLE),
                        dcc.Checklist(id="cities-map",
                            options=[{"label":" show (sized by pop)","value":"yes"}],
                            value=["yes"],
                            labelStyle={"fontSize":"0.9rem"}),
                    ]),
                ], style={**CARD, **FLEX}),
                dcc.Graph(id="map-fig", style={"height":"72vh"},
                          config={"scrollZoom":True}),
            ], style={"padding":"10px 14px"}),
        ]),

        # ── TAB 2: Compare ────────────────────────────────────────────────────
        dcc.Tab(label="⚖  Compare", value="compare", children=[
            html.Div([
                html.Div([
                    html.Div([html.Label("Left scenario",style=LBSTYLE),
                              sc_ctrl("sc-left", SC_KEYS[0])],
                             style={"marginRight":"18px"}),
                    html.Div([html.Label("Right scenario",style=LBSTYLE),
                              sc_ctrl("sc-right", SC_KEYS[min(1,len(SC_KEYS)-1)])],
                             style={"marginRight":"18px"}),
                    html.Div([html.Label("Year",style=LBSTYLE), yr_ctrl("yr-cmp")],
                             style={"marginRight":"18px"}),
                    html.Div([html.Label("Min congestion",style=LBSTYLE), sl_ctrl("sl-cmp")],
                             style={"width":"200px"}),
                ], style={**CARD, **FLEX}),
                dcc.Graph(id="cmp-fig", style={"height":"72vh"},
                          config={"scrollZoom":True}),
            ], style={"padding":"10px 14px"}),
        ]),

        # ── TAB 3: Scatter ────────────────────────────────────────────────────
        dcc.Tab(label="📈  2015 vs 2050", value="scatter", children=[
            html.Div([
                html.Div([
                    html.Div([html.Label("Scenario",style=LBSTYLE), sc_ctrl("sc-scatter")],
                             style={"marginRight":"18px"}),
                    html.P("Each dot = one road edge. Above the red line = more congested in 2050.",
                           style={"color":"#666","fontSize":"0.85rem","margin":"0"}),
                ], style={**CARD, **FLEX}),
                dcc.Graph(id="scatter-fig"),
            ], style={"padding":"10px 14px"}),
        ]),

        # ── TAB 4: Corridor ───────────────────────────────────────────────────
        dcc.Tab(label="🛣  Corridor", value="corridor", children=[
            html.Div([
                html.Div([
                    html.Div([html.Label("Origin",style=LBSTYLE),
                              dcc.Dropdown(id="orig-drop",
                                  options=[{"label":c,"value":c} for c in MAJOR_CITIES],
                                  value="Lagos", clearable=False,
                                  style={"width":"200px","fontSize":"0.85rem"})],
                             style={"marginRight":"16px"}),
                    html.Div([html.Label("Destination",style=LBSTYLE),
                              dcc.Dropdown(id="dest-drop",
                                  options=[{"label":c,"value":c} for c in MAJOR_CITIES],
                                  value="Abidjan", clearable=False,
                                  style={"width":"200px","fontSize":"0.85rem"})],
                             style={"marginRight":"16px"}),
                    html.Div([html.Label("Scenario",style=LBSTYLE), sc_ctrl("sc-corr")],
                             style={"marginRight":"16px"}),
                    html.Div([html.Label("Year",style=LBSTYLE), yr_ctrl("yr-corr")],
                             style={"marginRight":"16px"}),
                    html.Div([
                        html.Label("Overlay",style=LBSTYLE),
                        dcc.Checklist(id="overlay-all",
                            options=[{"label":" all scenarios","value":"yes"}],
                            value=[],
                            labelStyle={"fontSize":"0.9rem"}),
                    ]),
                ], style={**CARD, **FLEX}),
                dcc.Graph(id="corr-fig"),
            ], style={"padding":"10px 14px"}),
        ]),

    ], style={"fontFamily":"sans-serif"}),

], style={"fontFamily":"sans-serif"})


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(Output("map-fig","figure"),
              Input("sc-map","value"), Input("yr-map","value"),
              Input("cities-map","value"), Input("sl-map","value"))
def cb_map(sc, yr, cities, sl):
    fig = go.Figure()
    for t in edge_traces(sc, yr, sl):
        fig.add_trace(t)

    if "yes" in (cities or []):
        pop_col = "Pop2015" if yr == "2015" else ("Pop2050" if "Pop2050" in cities_df.columns else "Pop2015")
        cdf = cities_df[cities_df.get(pop_col, cities_df["Pop2015"]) > 0].copy()
        for region, grp in cdf.groupby("Region"):
            sz = np.log1p(grp[pop_col]) * 0.55
            fig.add_trace(go.Scattergeo(
                lon=grp["x"], lat=grp["y"], mode="markers",
                marker=dict(size=sz, color=REGION_COLORS.get(region,"#999"),
                            opacity=0.85, line=dict(width=0.4,color="black")),
                text=grp["agglosName"],
                hovertemplate="<b>%{text}</b><extra></extra>",
                name=region, legendgroup=region,
            ))

    fig.update_layout(
        geo=GEO_COMMON,
        margin=dict(l=0,r=0,t=0,b=0),
        legend=dict(orientation="v",x=0.01,y=0.99,
                    bgcolor="rgba(255,255,255,0.75)",font=dict(size=10)),
        uirevision="map", paper_bgcolor="white",
    )
    return fig


@app.callback(Output("cmp-fig","figure"),
              Input("sc-left","value"), Input("sc-right","value"),
              Input("yr-cmp","value"), Input("sl-cmp","value"))
def cb_compare(sc_l, sc_r, yr, sl):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type":"geo"},{"type":"geo"}]],
        subplot_titles=[SCENARIOS[sc_l]["label"], SCENARIOS[sc_r]["label"]],
    )
    for col_idx, sc in enumerate([sc_l, sc_r], 1):
        for t in edge_traces(sc, yr, sl):
            fig.add_trace(t, row=1, col=col_idx)
    fig.update_geos(GEO_COMMON)
    fig.update_layout(
        height=640, margin=dict(l=0,r=0,t=35,b=0),
        title=dict(text=f"Scenario comparison — {yr}", font=dict(size=13)),
        paper_bgcolor="white",
    )
    return fig


@app.callback(Output("scatter-fig","figure"), Input("sc-scatter","value"))
def cb_scatter(sc):
    b15 = get_between(sc,"2015") + 1
    b50 = get_between(sc,"2050") + 1
    mask = (b15 > 1) & (b50 > 1)
    htypes = edges_df["h"].values[mask]
    fig = go.Figure()
    for i, rt in enumerate(sorted(set(htypes))):
        sel = htypes == rt
        fig.add_trace(go.Scattergl(
            x=b15[mask][sel], y=b50[mask][sel], mode="markers",
            marker=dict(size=4,color=PALETTE[i%len(PALETTE)],opacity=0.5),
            name=rt,
            hovertemplate=f"<b>{rt}</b><br>2015: %{{x:.0f}}<br>2050: %{{y:.0f}}<extra></extra>",
        ))
    lim = max(b15[mask].max(), b50[mask].max()) * 1.3
    fig.add_trace(go.Scatter(x=[1,lim],y=[1,lim],mode="lines",
        line=dict(color="red",width=1.2,dash="dash"),showlegend=False))
    fig.update_layout(
        xaxis=dict(type="log",title="Congestion 2015+1"),
        yaxis=dict(type="log",title="Congestion 2050+1"),
        legend=dict(title="Road type",font=dict(size=10)),
        margin=dict(l=50,r=10,t=10,b=50), height=520,
        paper_bgcolor="white",
    )
    return fig


def _shortest_path_data(origin, dest, sc_key, yr):
    """Return (xs, ys) for corridor plot, or (None, None) if no path."""
    oi, di = NAME2IDX.get(origin), NAME2IDX.get(dest)
    if oi is None or di is None:
        return None, None
    vpath = G_base.get_shortest_paths(oi, di, weights=G_base.es["l"], output="vpath")[0]
    if len(vpath) < 2:
        return None, None
    G_sc = get_G(sc_key)
    attr = "Between" if yr == "2015" else "Between50"
    xs, ys, cum = [], [], 0.0
    for i in range(len(vpath)-1):
        eid   = G_base.get_eid(vpath[i], vpath[i+1])
        l_val = G_base.es[eid]["l"]
        b_val = max(G_sc.es[eid][attr], 0)
        xs   += [cum, cum+l_val]; ys += [b_val+1, b_val+1]
        cum  += l_val
    return xs, ys


@app.callback(Output("corr-fig","figure"),
              Input("orig-drop","value"), Input("dest-drop","value"),
              Input("sc-corr","value"), Input("yr-corr","value"),
              Input("overlay-all","value"))
def cb_corridor(origin, dest, sc, yr, overlay):
    if not origin or not dest:
        return go.Figure()

    fig = go.Figure()

    if overlay and "yes" in overlay:
        for i, (k, info) in enumerate(SCENARIOS.items()):
            xs, ys = _shortest_path_data(origin, dest, k, yr)
            if xs is None: continue
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=PALETTE[i%len(PALETTE)], width=2.2),
                name=info["label"],
                hovertemplate="dist %{x:.0f} km | congestion %{y:.0f}<extra></extra>",
            ))
        title_txt = f"{origin} → {dest}  ·  All scenarios  ·  {yr}"
    else:
        xs, ys = _shortest_path_data(origin, dest, sc, yr)
        if xs is None:
            return go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="tozeroy", mode="lines",
            line=dict(color="#e74c3c", width=0.8),
            fillcolor="rgba(231,76,60,0.28)",
            hovertemplate="dist %{x:.0f} km | congestion %{y:.0f}<extra></extra>",
            showlegend=False,
        ))
        title_txt = f"{origin} → {dest}  ·  {SCENARIOS[sc]['label']}  ·  {yr}"

    fig.update_layout(
        xaxis_title=f"Distance from {origin} (km)",
        yaxis=dict(title="Congestion + 1 (log)", type="log"),
        title=dict(text=title_txt, font=dict(size=12)),
        legend=dict(font=dict(size=10)),
        margin=dict(l=50,r=10,t=42,b=50), height=430,
        paper_bgcolor="white",
    )
    return fig


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*52)
    print("  Africa Transport Dashboard  v2")
    print(f"  Scenarios: {', '.join(SC_KEYS)}")
    print("  → http://127.0.0.1:8050")
    print("="*52 + "\n")
    app.run(debug=False, port=8050)
