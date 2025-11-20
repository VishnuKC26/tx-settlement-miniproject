import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="SettleUp — Visual Settlements", layout="wide")

# ---------- Helper functions ----------

def parse_csv_input(text_or_file):
    try:
        if hasattr(text_or_file, "read"):
            df = pd.read_csv(text_or_file)
        else:
            df = pd.read_csv(StringIO(text_or_file))
        required = {"payer", "payee", "amount"}
        if not required.issubset(set(map(str.lower, df.columns))):
            st.error("CSV must have columns: payer, payee, amount (case-insensitive).")
            return None
        # normalize columns
        cols = {c.lower(): c for c in df.columns}
        df = df[[cols["payer"], cols["payee"], cols["amount"]]]
        df.columns = ["payer", "payee", "amount"]
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        if df["amount"].isnull().any():
            st.error("Some amounts are not valid numbers.")
            return None
        return df
    except Exception as e:
        st.error(f"Failed to parse CSV input: {e}")
        return None


def compute_net_balances(transactions_df: pd.DataFrame) -> pd.Series:
    net = pd.Series(dtype=float)
    for _, row in transactions_df.iterrows():
        payer = row["payer"]
        payee = row["payee"]
        amt = float(row["amount"])
        net.loc[payer] = net.get(payer, 0.0) - amt
        net.loc[payee] = net.get(payee, 0.0) + amt
    net = net.round(2)
    net = net[net.abs() > 1e-9]
    return net


def minimize_transactions_steps(net_balances: pd.Series):
    debtors = []
    creditors = []
    for name, amt in net_balances.items():
        if amt < 0:
            debtors.append([name, -amt])
        elif amt > 0:
            creditors.append([name, amt])
    debtors.sort(key=lambda x: x[1], reverse=True)
    creditors.sort(key=lambda x: x[1], reverse=True)

    steps = []
    di = 0
    ci = 0
    while di < len(debtors) and ci < len(creditors):
        dname, damt = debtors[di]
        cname, camt = creditors[ci]
        transfer = round(min(damt, camt), 2)
        step = {
            "from": dname,
            "to": cname,
            "amount": transfer,
            "debtors": [(x[0], round(x[1],2)) for x in debtors],
            "creditors": [(x[0], round(x[1],2)) for x in creditors],
        }
        steps.append(step)
        debtors[di][1] -= transfer
        creditors[ci][1] -= transfer
        if abs(debtors[di][1]) < 1e-9:
            di += 1
        if abs(creditors[ci][1]) < 1e-9:
            ci += 1
    return steps


def build_network_graph(net_balances: pd.Series, highlight_transfer=None):
    G = nx.DiGraph()
    names = list(net_balances.index)
    for n in names:
        G.add_node(n, net=net_balances[n])
    # initial proportional edges for visualization clarity
    debtors = [(n, -net_balances[n]) for n in names if net_balances[n] < 0]
    creditors = [(n, net_balances[n]) for n in names if net_balances[n] > 0]
    total_credit = sum([c for _, c in creditors])
    for dname, damt in debtors:
        for cname, camt in creditors:
            if total_credit > 0:
                share = round(damt * (camt / total_credit), 2)
                if share > 0:
                    G.add_edge(dname, cname, amount=share)
    if highlight_transfer:
        frm, to, amt = highlight_transfer
        G.add_edge(frm, to, amount=amt, settle=True)
    return G


def networkx_to_plotly(G, highlight_transfer=None):
    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []
    edge_text = []
    edge_colors = []
    widths = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        amt = data.get('amount', 0)
        edge_text.append(f"{u} → {v}: ₹{amt}")
        if data.get('settle'):
            edge_colors.append('black')
            widths.append(4)
        else:
            edge_colors.append('rgba(0,0,0,0.2)')
            widths.append(2)

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        net = d.get('net', 0)
        sign = 'receives' if net > 0 else 'owes' if net < 0 else 'settled'
        node_text.append(f"{n}<br>Net: ₹{round(net,2)} ({sign})")
        node_color.append(net)

    fig = go.Figure()
    # draw edges with per-edge widths/colors
    # we create multiple traces to allow varying widths/colors
    for (u, v, data), color, w in zip(G.edges(data=True), edge_colors, widths):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=w, color=color), hoverinfo='text', text=[f"{u} → {v}: ₹{data.get('amount',0)}"], showlegend=False))

    fig.add_trace(go.Scatter(x=node_x, y=node_y,
                             mode='markers+text',
                             marker=dict(size=[20 + abs(v)*2 for v in node_color], color=node_color, colorscale='RdYlGn', showscale=True),
                             text=[n for n in G.nodes()],
                             hoverinfo='text',
                             textposition='bottom center'))

    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(showgrid=False, zeroline=False, visible=False),
                      yaxis=dict(showgrid=False, zeroline=False, visible=False),
                      height=450)
    return fig

# ---------- Streamlit UI (with slider, fixed display) ----------

st.title("SettleUp — Visual Settlement Planner")
st.write("A friendly visual tool that computes net balances from transactions and **animates** the greedy settlement steps so you can *see* how debts are simplified.")

with st.sidebar:
    st.header("Input")
    input_mode = st.radio("Input mode", ["Example (quick)", "Manual entry", "Upload CSV"], index=0)
    if input_mode == "Manual entry":
        rows = st.number_input("How many transactions to add", min_value=1, max_value=30, value=3)
        manual_rows = []
        for i in range(rows):
            payer = st.text_input(f"Payer #{i+1}", key=f"mpayer_{i}")
            payee = st.text_input(f"Payee #{i+1}", key=f"mpayee_{i}")
            amount = st.number_input(f"Amount #{i+1}", min_value=0.0, format="%f", key=f"mamt_{i}")
            manual_rows.append((payer.strip(), payee.strip(), float(amount)))
    elif input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
        pasted = st.text_area("Or paste CSV here (payer,payee,amount)")
    else:
        st.markdown("Quick example will be loaded when you press 'Load' below.")

    if st.button("Load"):
        st.session_state.load = True

if not st.session_state.get('load'):
    st.info("Press **Load** in the sidebar after choosing an input mode.")
    st.stop()

# Build dataframe from chosen input
if input_mode == "Example (quick)":
    df = pd.DataFrame([
        {"payer": "A", "payee": "B", "amount": 10},
        {"payer": "B", "payee": "C", "amount": 15},
        {"payer": "C", "payee": "A", "amount": 5},
    ])
elif input_mode == "Manual entry":
    df = pd.DataFrame([{"payer": p, "payee": q, "amount": r} for (p,q,r) in manual_rows if p and q and r>0])
else:
    if uploaded is not None:
        df = parse_csv_input(uploaded)
    elif pasted.strip():
        df = parse_csv_input(pasted)
    else:
        st.error("No CSV provided")
        st.stop()

if df is None or df.empty:
    st.error("No valid transactions. Please provide at least one transaction.")
    st.stop()

st.subheader("Transactions")
st.dataframe(df)

# compute nets and steps
net = compute_net_balances(df)
steps = minimize_transactions_steps(net)

left_col, right_col = st.columns([2, 3])
with left_col:
    st.subheader("Net balances")
    net_df = net.reset_index()
    net_df.columns = ["person", "net_amount"]
    st.dataframe(net_df)
    st.download_button("Download net balances CSV", net_df.to_csv(index=False).encode('utf-8'), file_name='net_balances.csv')

    st.markdown("---")
    st.subheader("Algorithm steps")
    st.write("Use the slider to step through how the greedy algorithm picks transfers and reduces debts.")
    if steps:
        step_index = st.slider("Step", min_value=0, max_value=len(steps), value=0)
        if step_index == 0:
            st.info("Step 0 — initial net balances before any settlement.")
        else:
            # show the executed transfer
            st.write(f"Step {step_index}: {steps[step_index-1]['from']} → {steps[step_index-1]['to']} : ₹{steps[step_index-1]['amount']}")
            st.write("Remaining debtors and creditors at this moment:")
            # --- FIX: properly construct side-by-side tables without length mismatch ---
            debtors = steps[step_index-1]['debtors']
            creditors = steps[step_index-1]['creditors']
            # convert to DataFrames
            debtors_df = pd.DataFrame(debtors, columns=['person','amount']) if debtors else pd.DataFrame(columns=['person','amount'])
            creditors_df = pd.DataFrame(creditors, columns=['person','amount']) if creditors else pd.DataFrame(columns=['person','amount'])
            # display them side-by-side for clarity
            c1, c2 = st.columns(2)
            with c1:
                st.write("Debtors (name, remaining owes)")
                st.dataframe(debtors_df)
            with c2:
                st.write("Creditors (name, remaining to receive)")
                st.dataframe(creditors_df)
    else:
        st.success("No settlements needed — all balances are zero or already settled.")

with right_col:
    st.subheader("Visualisation")
    if steps:
        if step_index == 0:
            G = build_network_graph(net)
            fig = networkx_to_plotly(G)
            st.plotly_chart(fig, use_container_width=True)
        else:
            executed = steps[step_index-1]
            highlight = (executed['from'], executed['to'], executed['amount'])
            temp_net = net.copy()
            for i in range(step_index):
                s = steps[i]
                temp_net[s['from']] = round(temp_net[s['from']] + s['amount'], 2)
                temp_net[s['to']] = round(temp_net[s['to']] - s['amount'], 2)
            G = build_network_graph(temp_net, highlight_transfer=highlight)
            fig = networkx_to_plotly(G, highlight_transfer=highlight)
            st.plotly_chart(fig, use_container_width=True)
    else:
        G = build_network_graph(net)
        fig = networkx_to_plotly(G)
        st.plotly_chart(fig, use_container_width=True)

# Show final settlements list and download
st.markdown("---")
st.subheader("Result — minimized settlement transactions")
if steps:
    settlements = [{"from": s['from'], "to": s['to'], "amount": s['amount']} for s in steps]
    sett_df = pd.DataFrame(settlements)
    st.write(f"Number of transactions: {len(settlements)}")
    st.table(sett_df)
    st.download_button("Download settlements CSV", sett_df.to_csv(index=False).encode('utf-8'), file_name='settlements.csv')
else:
    st.success("No transactions needed.")

st.markdown("---")
st.caption("Tip: Try larger random testcases by creating multiple manual rows or uploading a CSV. If you want a prettier layout or a web deployment, I can convert this to a React + D3 front-end or add step-by-step animation controls.")
