"""Pure data logic for the AML dashboard (no Streamlit imports, so it can be tested)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

REQUIRED = [
    "tx_id", "Timestamp", "Sender", "Receiver", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
    "baseline_score", "gnn_score",
]
MODELS = {"GNN (GraphSAGE)": "gnn_score", "Baseline (HistGradientBoosting)": "baseline_score"}
BUDGETS = (0.001, 0.0025, 0.005, 0.01)


def load_scores(src, name=None):
    comp = "gzip" if (name or str(src)).endswith(".gz") else "infer"
    df = pd.read_csv(src, compression=comp, parse_dates=["Timestamp"])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    df["Is Laundering"] = df["Is Laundering"].astype(int)
    df["rank_gnn"] = df["gnn_score"].rank(ascending=False, method="first").astype(int)
    df["rank_base"] = df["baseline_score"].rank(ascending=False, method="first").astype(int)
    return df


def headline_metrics(df):
    y = df["Is Laundering"].to_numpy()
    return {
        "n_tx": len(df),
        "n_pos": int(y.sum()),
        "n_accounts": int(pd.concat([df["Sender"], df["Receiver"]]).nunique()),
        "random_ap": float(y.mean()),
        "gnn_ap": float(average_precision_score(y, df["gnn_score"])),
        "base_ap": float(average_precision_score(y, df["baseline_score"])),
    }


def budget_table(df, budgets=BUDGETS):
    """Precision/recall when the bank can only review the top x% of transactions."""
    y = df["Is Laundering"].to_numpy()
    total_pos = max(1, int(y.sum()))
    rows = []
    for b in budgets:
        k = max(1, int(len(df) * b))
        row = {"Alert budget": f"{b:.2%}", "Alerts reviewed": k}
        for label, col in MODELS.items():
            s = df[col].to_numpy()
            top = np.argpartition(s, -k)[-k:]
            tp = int(y[top].sum())
            short = "GNN" if col == "gnn_score" else "Baseline"
            row[f"{short} caught"] = tp
            row[f"{short} precision"] = round(tp / k, 4)
            row[f"{short} recall"] = round(tp / total_pos, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def pr_points(y, s, max_points=400):
    p, r, _ = precision_recall_curve(y, s)
    if len(p) > max_points:
        idx = np.linspace(0, len(p) - 1, max_points).astype(int)
        p, r = p[idx], r[idx]
    return r, p


TIERS = [(0.001, "CRITICAL"), (0.005, "HIGH"), (0.01, "ELEVATED")]


def risk_tier(rank, n):
    """Rank-based tier (share of transactions ahead of this one). Not a calibrated probability."""
    share = rank / n
    for cut, name in TIERS:
        if share <= cut:
            return name
    return "LOW"


def alert_queue(df, k=200):
    q = df.nsmallest(k, "rank_gnn").copy()
    q["Tier"] = [risk_tier(r, len(df)) for r in q["rank_gnn"]]
    cols = ["rank_gnn", "Tier", "tx_id", "Timestamp", "Sender", "Receiver", "Amount Paid",
            "Payment Currency", "Payment Format", "gnn_score", "baseline_score",
            "rank_base", "Is Laundering"]
    return q[cols].reset_index(drop=True)


def degree_tables(df):
    return df["Sender"].value_counts(), df["Receiver"].value_counts()


def account_summary(df, acct):
    out = df[df["Sender"] == acct]
    inc = df[df["Receiver"] == acct]
    both = pd.concat([out, inc]).drop_duplicates("tx_id")
    top_cut = df["gnn_score"].quantile(0.999)
    return {
        "Sent (count)": len(out),
        "Received (count)": len(inc),
        "Unique recipients": out["Receiver"].nunique(),
        "Unique senders": inc["Sender"].nunique(),
        "First seen": both["Timestamp"].min() if len(both) else None,
        "Last seen": both["Timestamp"].max() if len(both) else None,
        "Highest GNN score": float(both["gnn_score"].max()) if len(both) else 0.0,
        "Txs in top 0.1% GNN scores": int((both["gnn_score"] >= top_cut).sum()),
        "Labelled laundering txs": int(both["Is Laundering"].sum()),
    }


def evidence(df, row, out_deg, in_deg, window_hours=24):
    """Context signals for one transaction. These are NOT model explanations."""
    t = row["Timestamp"]
    w = pd.Timedelta(hours=window_hours)
    s, r = row["Sender"], row["Receiver"]
    prior = df[df["Timestamp"] < t]
    s_in_recent = prior[(prior["Receiver"] == s) & (prior["Timestamp"] >= t - w)]
    r_prior_in = prior[prior["Receiver"] == r]
    s_prior_out = prior[prior["Sender"] == s]
    after = df[(df["Timestamp"] > t) & (df["Timestamp"] <= t + w) & (df["Sender"] == r)]

    pct_out = float((out_deg < out_deg.get(s, 0)).mean() * 100)
    pct_in = float((in_deg < in_deg.get(r, 0)).mean() * 100)
    rows = [
        ("Before this tx", "Sender received funds in prior window",
         f"{len(s_in_recent)} txs in last {window_hours}h", "Possible pass-through if high"),
        ("Before this tx", "Sender's earlier sends (test period)",
         f"{len(s_prior_out)} txs to {s_prior_out['Receiver'].nunique()} accounts", "Fan-out if many recipients"),
        ("Before this tx", "Receiver's earlier receipts (test period)",
         f"{len(r_prior_in)} txs from {r_prior_in['Sender'].nunique()} accounts", "Fan-in if many senders"),
        ("Activity level", "Sender send-count percentile", f"{pct_out:.0f}th", "vs all accounts in test period"),
        ("Activity level", "Receiver receive-count percentile", f"{pct_in:.0f}th", "vs all accounts in test period"),
        ("After this tx (retrospective)", f"Receiver sent onward within {window_hours}h",
         f"{len(after)} txs to {after['Receiver'].nunique()} accounts",
         "Not available at scoring time; shown for investigators"),
    ]
    return pd.DataFrame(rows, columns=["When", "Signal", "Value", "Why it matters"])


def ego_edges(df, center, hops=1, max_edges=120):
    nodes, frontier, parts = {center}, {center}, []
    for _ in range(hops):
        m = df[df["Sender"].isin(frontier) | df["Receiver"].isin(frontier)]
        parts.append(m)
        new = set(m["Sender"]) | set(m["Receiver"])
        frontier = new - nodes
        nodes |= new
    if not parts:
        return df.iloc[0:0]
    e = pd.concat(parts).drop_duplicates("tx_id")
    touches = ((e["Sender"] == center) | (e["Receiver"] == center)).astype(int)
    e = e.assign(_t=touches).sort_values(["_t", "gnn_score"], ascending=False).head(max_edges)
    return e.drop(columns="_t")


def build_network(edges, center):
    """Aggregate parallel edges and compute a radial layout (centre, hop-1 ring, outer ring)."""
    if edges.empty:
        return pd.DataFrame(), pd.DataFrame()
    agg = (edges.groupby(["Sender", "Receiver"])
           .agg(n_tx=("tx_id", "size"), max_gnn=("gnn_score", "max"),
                laundering=("Is Laundering", "max"), total_paid=("Amount Paid", "sum"))
           .reset_index())
    adj = {}
    for a, b in zip(agg["Sender"], agg["Receiver"]):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    dist, q = {center: 0}, [center]
    while q:
        cur = q.pop(0)
        for nb in adj.get(cur, ()):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    all_nodes = sorted(set(agg["Sender"]) | set(agg["Receiver"]))
    for n in all_nodes:
        dist.setdefault(n, 3)
    pos = {center: (0.0, 0.0)}
    for ring in sorted({d for d in dist.values() if d > 0}):
        members = [n for n in all_nodes if dist[n] == ring]
        for i, n in enumerate(members):
            ang = 2 * np.pi * i / max(1, len(members)) + ring * 0.3
            pos[n] = (ring * np.cos(ang), ring * np.sin(ang))
    nodes = pd.DataFrame({
        "node": all_nodes,
        "x": [pos[n][0] for n in all_nodes],
        "y": [pos[n][1] for n in all_nodes],
        "ring": [dist[n] for n in all_nodes],
        "degree": [len(adj.get(n, ())) for n in all_nodes],
    })
    agg["x0"] = agg["Sender"].map(lambda n: pos[n][0])
    agg["y0"] = agg["Sender"].map(lambda n: pos[n][1])
    agg["x1"] = agg["Receiver"].map(lambda n: pos[n][0])
    agg["y1"] = agg["Receiver"].map(lambda n: pos[n][1])
    return nodes, agg
