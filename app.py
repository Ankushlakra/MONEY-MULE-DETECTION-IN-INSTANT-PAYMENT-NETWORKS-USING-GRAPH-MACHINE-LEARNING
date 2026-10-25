import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import logic

st.set_page_config(page_title="AML Screening Console", page_icon="◼", layout="wide")

# ------------------------------------------------------------------ look and feel
BG, PANEL, LINE = "#0E1013", "#161A1F", "#262B33"
TEXT, MUTED = "#E7E4DC", "#8B919A"
AMBER, RED, GREEN, SLATE = "#E3A008", "#E5484D", "#3FB27F", "#7C8591"
TIER_COLOR = {"CRITICAL": RED, "HIGH": "#F2711C", "ELEVATED": AMBER, "LOW": SLATE}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, .stApp, [class*="css"] { font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 1.1rem; max-width: 1440px; }
h1, h2, h3 { letter-spacing: -0.01em; font-weight: 600; }
h3 { font-size: 1.05rem; text-transform: uppercase; letter-spacing: .08em; color: #8B919A; border-bottom: 1px solid #262B33; padding-bottom: .4rem; }
[data-testid="stMetric"] { background: #161A1F; border: 1px solid #262B33; border-left: 3px solid #E3A008; border-radius: 3px; padding: 12px 14px; }
[data-testid="stMetricLabel"] p { text-transform: uppercase; letter-spacing: .07em; font-size: .68rem; color: #8B919A; }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-size: 1.55rem; }
button[data-baseweb="tab"] { text-transform: uppercase; letter-spacing: .08em; font-size: .78rem; }
[data-testid="stDataFrame"] { border: 1px solid #262B33; border-radius: 3px; }
[data-testid="stSidebar"] { border-right: 1px solid #262B33; }
.topbar { display: flex; justify-content: space-between; align-items: center; border: 1px solid #262B33; background: #161A1F; border-left: 4px solid #E3A008; padding: 14px 18px; border-radius: 3px; margin-bottom: 14px; }
.topbar .t { font-size: 1.25rem; font-weight: 600; letter-spacing: .04em; }
.topbar .s { color: #8B919A; font-size: .8rem; font-family: 'IBM Plex Mono', monospace; margin-top: 2px; }
.pill { display: inline-block; padding: 3px 9px; margin-left: 6px; border: 1px solid #3a4049; border-radius: 2px; font-family: 'IBM Plex Mono', monospace; font-size: .68rem; letter-spacing: .08em; color: #E7E4DC; }
.pill.warn { border-color: #E3A008; color: #E3A008; }
.pill.bad { border-color: #E5484D; color: #E5484D; }
.tier { display: inline-block; padding: 4px 12px; border-radius: 2px; font-family: 'IBM Plex Mono', monospace; font-weight: 500; letter-spacing: .1em; color: #0E1013; }
.note { color: #8B919A; font-size: .8rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def style(fig, height=380):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color=TEXT, size=12),
        margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2),
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE), yaxis=dict(gridcolor=LINE, zerolinecolor=LINE),
    )
    return fig


def tier_badge(tier):
    return f"<span class='tier' style='background:{TIER_COLOR[tier]}'>{tier}</span>"


# Recorded notebook results (validation_results.json / final_test_results.json)
RECORDED = {"gnn_test_ap": 0.143055, "base_test_ap": 0.009234}
ABLATION = pd.DataFrame({
    "Model (validation, Sep-Oct 2025)": [
        "Baseline: transaction + running-history features",
        "GNN, self-loops only (no neighbour information)",
        "GNN, shuffled edges (control)",
        "GNN, real transaction graph",
    ],
    "Average precision": [0.0180, 0.1105, 0.1222, 0.1449],
})

# ------------------------------------------------------------------ data
st.sidebar.markdown("### Data source")
default = "dashboard_test_scores.csv.gz"
if not os.path.exists(default) and os.path.exists("sample_scores.csv.gz"):
    default = "sample_scores.csv.gz"
path = st.sidebar.text_input("Scores file", default)
upload = st.sidebar.file_uploader("...or upload", type=["csv", "gz"])


@st.cache_data(show_spinner="Loading scores...")
def load_path(p):
    return logic.load_scores(p)


@st.cache_data(show_spinner="Loading scores...")
def load_upload(data, name):
    import io
    return logic.load_scores(io.BytesIO(data), name=name)


try:
    if upload is not None:
        df, source_name = load_upload(upload.getvalue(), upload.name), upload.name
    else:
        df, source_name = load_path(path), path
except Exception as e:
    st.title("AML Screening Console")
    st.error(f"Could not load scores: {e}")
    st.info("Run `colab_rebuild_and_export.py` in Colab, download `dashboard_test_scores.csv.gz` "
            "from Drive and put it next to app.py. To just try the layout, run "
            "`python make_sample_data.py` (fake data).")
    st.stop()

is_sample = "sample" in os.path.basename(source_name).lower()


@st.cache_data
def get_metrics(_df, key):
    return logic.headline_metrics(_df)


@st.cache_data
def get_degrees(_df, key):
    return logic.degree_tables(_df)


m = get_metrics(df, source_name)
out_deg, in_deg = get_degrees(df, source_name)

pills = "<span class='pill'>TEST PERIOD NOV-DEC 2025</span><span class='pill'>OFFLINE REPLAY</span>"
pills += ("<span class='pill bad'>FAKE DEMO DATA</span>" if is_sample
          else "<span class='pill warn'>SYNTHETIC DATASET</span>")
st.markdown(
    f"<div class='topbar'><div><div class='t'>AML SCREENING CONSOLE</div>"
    f"<div class='s'>TransXion | GraphSAGE vs transaction baseline | alerts are for review, not proof</div>"
    f"</div><div>{pills}</div></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Alert queue", "Investigate", "Evidence and limits"])

# ------------------------------------------------------------------ Overview
with tab1:
    c = st.columns(5)
    c[0].metric("Transactions", f"{m['n_tx']:,}")
    c[1].metric("Accounts", f"{m['n_accounts']:,}")
    c[2].metric("Labelled laundering", f"{m['n_pos']:,}", f"{m['random_ap']:.3%} of all", delta_color="off")
    c[3].metric("GNN avg precision", f"{m['gnn_ap']:.4f}")
    c[4].metric("Baseline avg precision", f"{m['base_ap']:.4f}")
    st.markdown(f"<div class='note'>A random ranking scores about {m['random_ap']:.4f}. "
                "Average precision rewards ranking the rare positives near the top.</div>",
                unsafe_allow_html=True)

    if not is_sample:
        ok = (abs(m["gnn_ap"] - RECORDED["gnn_test_ap"]) < 1e-3
              and abs(m["base_ap"] - RECORDED["base_test_ap"]) < 1e-3)
        if ok:
            st.success("Reproduces the notebook's recorded test results (GNN 0.1431, baseline 0.0092).")
        else:
            st.error("These scores do NOT match the notebook's recorded test results. "
                     "Check the export came from the same run as Step 26.")

    st.markdown("### Review capacity: what if only a fixed share can be investigated?")
    bt = logic.budget_table(df)
    st.dataframe(bt, width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for (label, col), colr in zip(logic.MODELS.items(), (AMBER, SLATE)):
            r, p = logic.pr_points(df["Is Laundering"].to_numpy(), df[col].to_numpy())
            fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name=label, line=dict(color=colr, width=2.5)))
        fig.update_layout(title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(style(fig), width="stretch")
    with right:
        fig = go.Figure()
        for short, colr in (("GNN", AMBER), ("Baseline", SLATE)):
            fig.add_trace(go.Bar(x=bt["Alert budget"], y=bt[f"{short} caught"], name=short, marker_color=colr))
        fig.update_layout(title=f"Laundering transactions caught (of {m['n_pos']})", barmode="group",
                          xaxis_title="Share of transactions reviewed")
        st.plotly_chart(style(fig), width="stretch")

# ------------------------------------------------------------------ Alert queue
with tab2:
    st.markdown("### Alert queue, ranked by GNN risk score")
    k = st.slider("Alerts to show", 20, 2000, 200, step=20)
    reveal = st.toggle("Reveal true labels (not available in production)", value=True)
    q = logic.alert_queue(df, k)
    counts = q["Tier"].value_counts()
    tc = st.columns(4)
    for col_, name in zip(tc, ("CRITICAL", "HIGH", "ELEVATED", "LOW")):
        col_.metric(name.title(), int(counts.get(name, 0)))
    if reveal:
        hits = int(q["Is Laundering"].sum())
        st.metric(f"Labelled laundering within top {k}", f"{hits}", f"precision {hits / k:.1%}", delta_color="off")
    show = q if reveal else q.drop(columns=["Is Laundering"])
    st.dataframe(show, width="stretch", hide_index=True, height=480)
    st.download_button("Download this queue (CSV)", q.to_csv(index=False).encode(), "alert_queue.csv", "text/csv")
    st.markdown("<div class='note'>Tier is rank-based: CRITICAL = top 0.1% of transactions by score, "
                "HIGH = top 0.5%, ELEVATED = top 1%. It is not a calibrated probability. "
                "rank_base shows where the baseline ranked the same transaction.</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ Investigate
with tab3:
    mode = st.radio("Start from", ["An alert from the queue", "An account ID"], horizontal=True)
    center, row = None, None
    if mode == "An alert from the queue":
        top = logic.alert_queue(df, 300)
        pick = st.selectbox(
            "Alert (top 300 by GNN score)", top["tx_id"].tolist(),
            format_func=lambda t: (
                f"rank {int(top.loc[top.tx_id == t, 'rank_gnn'].iloc[0])} | tx {t} | score "
                f"{top.loc[top.tx_id == t, 'gnn_score'].iloc[0]:.3f}"))
        row = df[df["tx_id"] == pick].iloc[0]
        who = st.radio("Centre the network on", ["Sender", "Receiver"], horizontal=True)
        center = row[who]
    else:
        typed = st.text_input("Account ID (format bank_account, e.g. 3295718_A009852)")
        if typed.strip():
            center = typed.strip()
            if not ((df["Sender"] == center) | (df["Receiver"] == center)).any():
                st.warning("No transactions for that account in the test period.")
                center = None

    if row is not None:
        tier = logic.risk_tier(int(row["rank_gnn"]), len(df))
        st.markdown(f"{tier_badge(tier)} &nbsp; <span class='note'>rank {int(row['rank_gnn']):,} "
                    f"of {len(df):,} transactions</span>", unsafe_allow_html=True)
        a, b, c3, d = st.columns(4)
        a.metric("GNN score", f"{row['gnn_score']:.3f}")
        b.metric("Baseline score", f"{row['baseline_score']:.3f}", f"rank {int(row['rank_base']):,}", delta_color="off")
        c3.metric("Amount", f"{row['Amount Paid']:,.2f} {row['Payment Currency']}")
        d.metric("Actual label", "Laundering" if row["Is Laundering"] == 1 else "Not labelled")
        st.markdown(f"`{row['Sender']}` &rarr; `{row['Receiver']}` &nbsp; via {row['Payment Format']} "
                    f"&nbsp; at {row['Timestamp']}", unsafe_allow_html=True)
        win = st.select_slider("Look-around window (hours)", [1, 6, 24, 72, 168], value=24)
        st.markdown("### Context signals")
        st.markdown("<div class='note'>Simple counts for the investigator. They are not the model's "
                    "internal reasoning.</div>", unsafe_allow_html=True)
        st.dataframe(logic.evidence(df, row, out_deg, in_deg, win), width="stretch", hide_index=True)

    if center is not None:
        st.markdown(f"### Account summary: {center}")
        summ = logic.account_summary(df, center)
        st.dataframe(pd.DataFrame(summ.items(), columns=["Measure", "Value"]).astype(str),
                     width="stretch", hide_index=True)

        st.markdown("### Transfer network")
        hops = st.radio("Network depth", [1, 2], horizontal=True)
        edges = logic.ego_edges(df, center, hops=hops)
        nodes, agg = logic.build_network(edges, center)
        if nodes.empty:
            st.info("No edges to draw.")
        else:
            fig = go.Figure()
            for _, e in agg.iterrows():
                color = RED if e["laundering"] == 1 else "rgba(139,145,154,0.40)"
                fig.add_trace(go.Scatter(
                    x=[e["x0"], e["x1"]], y=[e["y0"], e["y1"]], mode="lines",
                    line=dict(color=color, width=1 + min(4, e["n_tx"] * 0.6)), hoverinfo="text",
                    text=f"{e['Sender']} -> {e['Receiver']}<br>{int(e['n_tx'])} tx | max GNN score {e['max_gnn']:.3f}",
                    showlegend=False))
            is_c = nodes["node"] == center
            fig.add_trace(go.Scatter(
                x=nodes["x"], y=nodes["y"], mode="markers",
                marker=dict(size=[24 if x else 9 + min(10, d) for x, d in zip(is_c, nodes["degree"])],
                            color=[AMBER if x else SLATE for x in is_c], line=dict(width=1, color=BG)),
                text=nodes["node"], hoverinfo="text", showlegend=False))
            fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(style(fig, 560), width="stretch")
            st.markdown(f"<div class='note'>Amber = centre account. Red edges include a labelled laundering "
                        f"transaction. Edges come from the Nov-Dec test period only; large neighbourhoods are "
                        f"trimmed to {len(edges)} edges (centre's edges first, then highest GNN score).</div>",
                        unsafe_allow_html=True)

# ------------------------------------------------------------------ Evidence and limits
with tab4:
    st.markdown("### Does the graph itself help? Controlled comparison")
    st.dataframe(ABLATION, width="stretch", hide_index=True)
    fig = go.Figure(go.Bar(x=ABLATION["Average precision"], y=ABLATION["Model (validation, Sep-Oct 2025)"],
                           orientation="h", marker_color=[SLATE, SLATE, SLATE, AMBER]))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(style(fig, 300), width="stretch")
    st.markdown("""
**How to read this.** Most of the improvement over the baseline comes from giving each account its
Jan-Apr history. The real graph adds a smaller gain on top (about +0.03 over self-loops, +0.02 over
shuffled edges). Each variant was trained once, so small gaps are not conclusive.

""")
