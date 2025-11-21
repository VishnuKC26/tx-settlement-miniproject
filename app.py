# import streamlit as st
# import pandas as pd
# import numpy as np
# import networkx as nx
# import plotly.graph_objects as go
# from io import StringIO

# st.set_page_config(page_title="SettleUp — Visual Settlements", layout="wide")

# # ---------- Helper functions ----------

# def parse_csv_input(text_or_file):
#     try:
#         if hasattr(text_or_file, "read"):
#             df = pd.read_csv(text_or_file)
#         else:
#             df = pd.read_csv(StringIO(text_or_file))
#         required = {"payer", "payee", "amount"}
#         if not required.issubset(set(map(str.lower, df.columns))):
#             st.error("CSV must have columns: payer, payee, amount (case-insensitive).")
#             return None
#         # normalize columns
#         cols = {c.lower(): c for c in df.columns}
#         df = df[[cols["payer"], cols["payee"], cols["amount"]]]
#         df.columns = ["payer", "payee", "amount"]
#         df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
#         if df["amount"].isnull().any():
#             st.error("Some amounts are not valid numbers.")
#             return None
#         return df
#     except Exception as e:
#         st.error(f"Failed to parse CSV input: {e}")
#         return None


# def compute_net_balances(transactions_df: pd.DataFrame) -> pd.Series:
#     net = pd.Series(dtype=float)
#     for _, row in transactions_df.iterrows():
#         payer = row["payer"]
#         payee = row["payee"]
#         amt = float(row["amount"])
#         net.loc[payer] = net.get(payer, 0.0) - amt
#         net.loc[payee] = net.get(payee, 0.0) + amt
#     net = net.round(2)
#     net = net[net.abs() > 1e-9]
#     return net


# def minimize_transactions_steps(net_balances: pd.Series):
#     debtors = []
#     creditors = []
#     for name, amt in net_balances.items():
#         if amt < 0:
#             debtors.append([name, -amt])
#         elif amt > 0:
#             creditors.append([name, amt])
#     debtors.sort(key=lambda x: x[1], reverse=True)
#     creditors.sort(key=lambda x: x[1], reverse=True)

#     steps = []
#     di = 0
#     ci = 0
#     while di < len(debtors) and ci < len(creditors):
#         dname, damt = debtors[di]
#         cname, camt = creditors[ci]
#         transfer = round(min(damt, camt), 2)
#         step = {
#             "from": dname,
#             "to": cname,
#             "amount": transfer,
#             "debtors": [(x[0], round(x[1],2)) for x in debtors],
#             "creditors": [(x[0], round(x[1],2)) for x in creditors],
#         }
#         steps.append(step)
#         debtors[di][1] -= transfer
#         creditors[ci][1] -= transfer
#         if abs(debtors[di][1]) < 1e-9:
#             di += 1
#         if abs(creditors[ci][1]) < 1e-9:
#             ci += 1
#     return steps


# def build_network_graph(net_balances: pd.Series, highlight_transfer=None):
#     G = nx.DiGraph()
#     names = list(net_balances.index)
#     for n in names:
#         G.add_node(n, net=net_balances[n])
#     # initial proportional edges for visualization clarity
#     debtors = [(n, -net_balances[n]) for n in names if net_balances[n] < 0]
#     creditors = [(n, net_balances[n]) for n in names if net_balances[n] > 0]
#     total_credit = sum([c for _, c in creditors])
#     for dname, damt in debtors:
#         for cname, camt in creditors:
#             if total_credit > 0:
#                 share = round(damt * (camt / total_credit), 2)
#                 if share > 0:
#                     G.add_edge(dname, cname, amount=share)
#     if highlight_transfer:
#         frm, to, amt = highlight_transfer
#         G.add_edge(frm, to, amount=amt, settle=True)
#     return G


# def networkx_to_plotly(G, highlight_transfer=None):
#     pos = nx.spring_layout(G, seed=42)
#     edge_x = []
#     edge_y = []
#     edge_text = []
#     edge_colors = []
#     widths = []
#     for u, v, data in G.edges(data=True):
#         x0, y0 = pos[u]
#         x1, y1 = pos[v]
#         edge_x += [x0, x1, None]
#         edge_y += [y0, y1, None]
#         amt = data.get('amount', 0)
#         edge_text.append(f"{u} → {v}: ₹{amt}")
#         if data.get('settle'):
#             edge_colors.append('black')
#             widths.append(4)
#         else:
#             edge_colors.append('rgba(0,0,0,0.2)')
#             widths.append(2)

#     node_x = []
#     node_y = []
#     node_text = []
#     node_color = []
#     for n, d in G.nodes(data=True):
#         x, y = pos[n]
#         node_x.append(x)
#         node_y.append(y)
#         net = d.get('net', 0)
#         sign = 'receives' if net > 0 else 'owes' if net < 0 else 'settled'
#         node_text.append(f"{n}<br>Net: ₹{round(net,2)} ({sign})")
#         node_color.append(net)

#     fig = go.Figure()
#     # draw edges with per-edge widths/colors
#     # we create multiple traces to allow varying widths/colors
#     for (u, v, data), color, w in zip(G.edges(data=True), edge_colors, widths):
#         x0, y0 = pos[u]
#         x1, y1 = pos[v]
#         fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=w, color=color), hoverinfo='text', text=[f"{u} → {v}: ₹{data.get('amount',0)}"], showlegend=False))

#     fig.add_trace(go.Scatter(x=node_x, y=node_y,
#                              mode='markers+text',
#                              marker=dict(size=[20 + abs(v)*2 for v in node_color], color=node_color, colorscale='RdYlGn', showscale=True),
#                              text=[n for n in G.nodes()],
#                              hoverinfo='text',
#                              textposition='bottom center'))

#     fig.update_layout(margin=dict(l=10, r=10, t=30, b=10),
#                       xaxis=dict(showgrid=False, zeroline=False, visible=False),
#                       yaxis=dict(showgrid=False, zeroline=False, visible=False),
#                       height=450)
#     return fig

# # ---------- Streamlit UI (with slider, fixed display) ----------

# st.title("SettleUp — Visual Settlement Planner")
# st.write("A friendly visual tool that computes net balances from transactions and **animates** the greedy settlement steps so you can *see* how debts are simplified.")

# with st.sidebar:
#     st.header("Input")
#     input_mode = st.radio("Input mode", ["Example (quick)", "Manual entry", "Upload CSV"], index=0)
#     if input_mode == "Manual entry":
#         rows = st.number_input("How many transactions to add", min_value=1, max_value=30, value=3)
#         manual_rows = []
#         for i in range(rows):
#             payer = st.text_input(f"Payer #{i+1}", key=f"mpayer_{i}")
#             payee = st.text_input(f"Payee #{i+1}", key=f"mpayee_{i}")
#             amount = st.number_input(f"Amount #{i+1}", min_value=0.0, format="%f", key=f"mamt_{i}")
#             manual_rows.append((payer.strip(), payee.strip(), float(amount)))
#     elif input_mode == "Upload CSV":
#         uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
#         pasted = st.text_area("Or paste CSV here (payer,payee,amount)")
#     else:
#         st.markdown("Quick example will be loaded when you press 'Load' below.")

#     if st.button("Load"):
#         st.session_state.load = True

# if not st.session_state.get('load'):
#     st.info("Press **Load** in the sidebar after choosing an input mode.")
#     st.stop()

# # Build dataframe from chosen input
# if input_mode == "Example (quick)":
#     df = pd.DataFrame([
#         {"payer": "A", "payee": "B", "amount": 10},
#         {"payer": "B", "payee": "C", "amount": 15},
#         {"payer": "C", "payee": "A", "amount": 5},
#     ])
# elif input_mode == "Manual entry":
#     df = pd.DataFrame([{"payer": p, "payee": q, "amount": r} for (p,q,r) in manual_rows if p and q and r>0])
# else:
#     if uploaded is not None:
#         df = parse_csv_input(uploaded)
#     elif pasted.strip():
#         df = parse_csv_input(pasted)
#     else:
#         st.error("No CSV provided")
#         st.stop()

# if df is None or df.empty:
#     st.error("No valid transactions. Please provide at least one transaction.")
#     st.stop()

# st.subheader("Transactions")
# st.dataframe(df)

# # compute nets and steps
# net = compute_net_balances(df)
# steps = minimize_transactions_steps(net)

# left_col, right_col = st.columns([2, 3])
# with left_col:
#     st.subheader("Net balances")
#     net_df = net.reset_index()
#     net_df.columns = ["person", "net_amount"]
#     st.dataframe(net_df)
#     st.download_button("Download net balances CSV", net_df.to_csv(index=False).encode('utf-8'), file_name='net_balances.csv')

#     st.markdown("---")
#     st.subheader("Algorithm steps")
#     st.write("Use the slider to step through how the greedy algorithm picks transfers and reduces debts.")
#     if steps:
#         step_index = st.slider("Step", min_value=0, max_value=len(steps), value=0)
#         if step_index == 0:
#             st.info("Step 0 — initial net balances before any settlement.")
#         else:
#             # show the executed transfer
#             st.write(f"Step {step_index}: {steps[step_index-1]['from']} → {steps[step_index-1]['to']} : ₹{steps[step_index-1]['amount']}")
#             st.write("Remaining debtors and creditors at this moment:")
#             # --- FIX: properly construct side-by-side tables without length mismatch ---
#             debtors = steps[step_index-1]['debtors']
#             creditors = steps[step_index-1]['creditors']
#             # convert to DataFrames
#             debtors_df = pd.DataFrame(debtors, columns=['person','amount']) if debtors else pd.DataFrame(columns=['person','amount'])
#             creditors_df = pd.DataFrame(creditors, columns=['person','amount']) if creditors else pd.DataFrame(columns=['person','amount'])
#             # display them side-by-side for clarity
#             c1, c2 = st.columns(2)
#             with c1:
#                 st.write("Debtors (name, remaining owes)")
#                 st.dataframe(debtors_df)
#             with c2:
#                 st.write("Creditors (name, remaining to receive)")
#                 st.dataframe(creditors_df)
#     else:
#         st.success("No settlements needed — all balances are zero or already settled.")

# with right_col:
#     st.subheader("Visualisation")
#     if steps:
#         if step_index == 0:
#             G = build_network_graph(net)
#             fig = networkx_to_plotly(G)
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             executed = steps[step_index-1]
#             highlight = (executed['from'], executed['to'], executed['amount'])
#             temp_net = net.copy()
#             for i in range(step_index):
#                 s = steps[i]
#                 temp_net[s['from']] = round(temp_net[s['from']] + s['amount'], 2)
#                 temp_net[s['to']] = round(temp_net[s['to']] - s['amount'], 2)
#             G = build_network_graph(temp_net, highlight_transfer=highlight)
#             fig = networkx_to_plotly(G, highlight_transfer=highlight)
#             st.plotly_chart(fig, use_container_width=True)
#     else:
#         G = build_network_graph(net)
#         fig = networkx_to_plotly(G)
#         st.plotly_chart(fig, use_container_width=True)

# # Show final settlements list and download
# st.markdown("---")
# st.subheader("Result — minimized settlement transactions")
# if steps:
#     settlements = [{"from": s['from'], "to": s['to'], "amount": s['amount']} for s in steps]
#     sett_df = pd.DataFrame(settlements)
#     st.write(f"Number of transactions: {len(settlements)}")
#     st.table(sett_df)
#     st.download_button("Download settlements CSV", sett_df.to_csv(index=False).encode('utf-8'), file_name='settlements.csv')
# else:
#     st.success("No transactions needed.")

# st.markdown("---")
# st.caption("Tip: Try larger random testcases by creating multiple manual rows or uploading a CSV. If you want a prettier layout or a web deployment, I can convert this to a React + D3 front-end or add step-by-step animation controls.")
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="SettleUp — Smart Settlements", layout="wide", initial_sidebar_state="expanded")

# Clean, professional CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .reduction-banner {
        text-align: center;
        padding: 2rem;
        background: #f0fdf4;
        border-radius: 12px;
        border: 2px solid #10b981;
        margin: 2rem 0;
    }
    .graph-header {
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background: #f9fafb;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

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
        }
        steps.append(step)
        debtors[di][1] -= transfer
        creditors[ci][1] -= transfer
        if abs(debtors[di][1]) < 1e-9:
            di += 1
        if abs(creditors[ci][1]) < 1e-9:
            ci += 1
    return steps


def create_enhanced_graph(df, source_col, target_col, title, edge_color, is_optimized=False):
    """Create an enhanced graph visualization"""
    
    G = nx.DiGraph()
    
    # Add nodes and edges
    all_people = set(list(df[source_col]) + list(df[target_col]))
    for person in all_people:
        G.add_node(person)
    
    edge_data = []
    for _, row in df.iterrows():
        source = row[source_col]
        target = row[target_col]
        amount = row["amount"]
        edge_data.append((source, target, amount))
        G.add_edge(source, target)
    
    # Use hierarchical layout for better visualization
    try:
        pos = nx.planar_layout(G)
    except:
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    
    # Normalize positions to fit better
    pos_array = np.array(list(pos.values()))
    pos_min = pos_array.min(axis=0)
    pos_max = pos_array.max(axis=0)
    pos_range = pos_max - pos_min
    pos_range[pos_range == 0] = 1
    
    for node in pos:
        pos[node] = (pos[node] - pos_min) / pos_range
    
    fig = go.Figure()
    
    # Draw edges with curves for better visibility
    for source, target, amount in edge_data:
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        # Create curved path
        dx = x1 - x0
        dy = y1 - y0
        
        # Perpendicular offset for curve
        curve_offset = 0.15
        perp_x = -dy * curve_offset
        perp_y = dx * curve_offset
        
        # Control point for quadratic curve
        ctrl_x = (x0 + x1) / 2 + perp_x
        ctrl_y = (y0 + y1) / 2 + perp_y
        
        # Generate curve points
        t = np.linspace(0, 1, 50)
        curve_x = (1-t)**2 * x0 + 2*(1-t)*t * ctrl_x + t**2 * x1
        curve_y = (1-t)**2 * y0 + 2*(1-t)*t * ctrl_y + t**2 * y1
        
        # Draw curved edge
        fig.add_trace(go.Scatter(
            x=curve_x,
            y=curve_y,
            mode='lines',
            line=dict(
                width=2.5 if is_optimized else 1.5,
                color=edge_color
            ),
            hoverinfo='skip',
            showlegend=False,
            opacity=0.8
        ))
        
        # Add arrowhead at the end
        arrow_t = 0.85
        arrow_x = (1-arrow_t)**2 * x0 + 2*(1-arrow_t)*arrow_t * ctrl_x + arrow_t**2 * x1
        arrow_y = (1-arrow_t)**2 * y0 + 2*(1-arrow_t)*arrow_t * ctrl_y + arrow_t**2 * y1
        
        fig.add_annotation(
            x=x1, y=y1,
            ax=arrow_x, ay=arrow_y,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.8,
            arrowwidth=2.5 if is_optimized else 1.5,
            arrowcolor=edge_color,
            opacity=0.8
        )
        
        # Add amount label with background
        label_t = 0.5
        label_x = (1-label_t)**2 * x0 + 2*(1-label_t)*label_t * ctrl_x + label_t**2 * x1
        label_y = (1-label_t)**2 * y0 + 2*(1-label_t)*label_t * ctrl_y + label_t**2 * y1
        
        fig.add_trace(go.Scatter(
            x=[label_x],
            y=[label_y],
            mode='markers+text',
            marker=dict(
                size=28,
                color='white',
                line=dict(width=2, color=edge_color)
            ),
            text=[f"₹{amount:.0f}"],
            textfont=dict(
                size=10,
                color='#1a1a1a',
                family='Arial'
            ),
            textposition='middle center',
            showlegend=False,
            hoverinfo='text',
            hovertext=f"{source} → {target}: ₹{amount}"
        ))
    
    # Draw nodes with better styling
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_sizes = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        # Calculate degree for sizing
        out_degree = G.out_degree(node)
        in_degree = G.in_degree(node)
        total_degree = out_degree + in_degree
        
        node_sizes.append(60 + total_degree * 8)
        
        if out_degree > 0 and in_degree > 0:
            hover = f"<b>{node}</b><br>Pays: {out_degree} | Receives: {in_degree}"
        elif out_degree > 0:
            hover = f"<b>{node}</b><br>Pays: {out_degree}"
        elif in_degree > 0:
            hover = f"<b>{node}</b><br>Receives: {in_degree}"
        else:
            hover = f"<b>{node}</b>"
        
        node_hover.append(hover)
    
    # Main nodes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color='#3b82f6',
            line=dict(width=3, color='white'),
            opacity=1
        ),
        text=node_text,
        textposition='middle center',
        textfont=dict(
            size=13,
            color='white',
            family='Arial',
            weight='bold'
        ),
        hovertext=node_hover,
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
            range=[-0.1, 1.1]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
            range=[-0.1, 1.1]
        ),
        height=600,
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=20, color='#1a1a1a'),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        )
    )
    
    return fig


# ---------- Main UI ----------

st.markdown('<h1 class="main-header">SettleUp</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Visualize how complex transactions get simplified</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Data Input")
    
    input_mode = st.radio(
        "Choose input method:",
        ["Quick Example", "Manual Entry", "Upload CSV"],
        index=0
    )
    
    st.markdown("---")
    
    if input_mode == "Manual Entry":
        st.markdown("#### Add Transactions")
        rows = st.number_input("Number of transactions", min_value=1, max_value=30, value=3)
        manual_rows = []
        
        with st.expander("Enter transaction details", expanded=True):
            for i in range(rows):
                col1, col2, col3 = st.columns([3, 3, 2])
                with col1:
                    payer = st.text_input(f"From", key=f"mpayer_{i}", placeholder="Person A")
                with col2:
                    payee = st.text_input(f"To", key=f"mpayee_{i}", placeholder="Person B")
                with col3:
                    amount = st.number_input(f"₹", min_value=0.0, format="%.2f", key=f"mamt_{i}")
                if payer.strip() and payee.strip() and amount > 0:
                    manual_rows.append((payer.strip(), payee.strip(), float(amount)))
                    
    elif input_mode == "Upload CSV":
        st.markdown("#### Upload or Paste")
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        st.markdown("**Or paste CSV data:**")
        pasted = st.text_area("Format: payer,payee,amount", height=100, placeholder="Alice,Bob,100\nBob,Charlie,50")
    
    st.markdown("---")
    load_btn = st.button("Load & Analyze", type="primary", use_container_width=True)
    
    if load_btn:
        st.session_state.load = True

# Main content
if not st.session_state.get('load'):
    st.info("Choose an input method and press **Load & Analyze** to get started")
    st.stop()

# Build dataframe from chosen input
if input_mode == "Quick Example":
    df = pd.DataFrame([
        {"payer": "Alice", "payee": "Bob", "amount": 100},
        {"payer": "Bob", "payee": "Charlie", "amount": 150},
        {"payer": "Charlie", "payee": "Alice", "amount": 50},
        {"payer": "David", "payee": "Alice", "amount": 75},
        {"payer": "Alice", "payee": "Eve", "amount": 200},
        {"payer": "Eve", "payee": "Bob", "amount": 100},
        {"payer": "Charlie", "payee": "David", "amount": 80},
        {"payer": "Bob", "payee": "David", "amount": 60},
    ])
elif input_mode == "Manual Entry":
    if not manual_rows:
        st.error("No valid transactions entered. Please add at least one transaction.")
        st.stop()
    df = pd.DataFrame([{"payer": p, "payee": q, "amount": r} for (p,q,r) in manual_rows])
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

# Compute settlements
net = compute_net_balances(df)
steps = minimize_transactions_steps(net)

# Summary metrics
st.markdown("### Transaction Reduction Summary")

if steps:
    reduction_pct = ((len(df) - len(steps)) / len(df) * 100) if len(df) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original", len(df), help="Number of original transactions")
    
    with col2:
        st.metric("Optimized", len(steps), help="Minimum settlements needed")
    
    with col3:
        st.metric("Saved", len(df) - len(steps), help="Transactions eliminated")
    
    with col4:
        st.metric("Reduction", f"{reduction_pct:.1f}%", help="Percentage reduction")
    
    st.markdown("---")
    
    # Original Transactions Graph
    st.markdown('<div class="graph-header">Original Transactions Network</div>', unsafe_allow_html=True)
    fig_original = create_enhanced_graph(df, "payer", "payee", 
                                        f"{len(df)} Transactions", 
                                        "#94a3b8", 
                                        is_optimized=False)
    st.plotly_chart(fig_original, use_container_width=True)
    
    st.markdown("---")
    
    # Show reduction banner
    st.markdown(f"""
    <div class="reduction-banner">
        <div style="font-size: 2.5rem; font-weight: 700; color: #10b981; margin-bottom: 0.5rem;">
            ↓ {len(df) - len(steps)} Transactions Eliminated
        </div>
        <div style="font-size: 1.2rem; color: #059669;">
            {reduction_pct:.1f}% Reduction Achieved
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Optimized Settlements Graph
    st.markdown('<div class="graph-header">Optimized Settlement Network</div>', unsafe_allow_html=True)
    settlements_df = pd.DataFrame([
        {"from": s['from'], "to": s['to'], "amount": s['amount']} 
        for s in steps
    ])
    fig_optimized = create_enhanced_graph(settlements_df, "from", "to", 
                                         f"{len(steps)} Settlements", 
                                         "#10b981", 
                                         is_optimized=True)
    st.plotly_chart(fig_optimized, use_container_width=True)

else:
    st.success("Perfect! All balances are already settled. No transactions needed.")
    st.stop()

st.markdown("---")

# Details section
with st.expander("View Transaction Details"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Transactions")
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Optimized Settlements")
        settlements_display = pd.DataFrame([
            {"From": s['from'], "To": s['to'], "Amount": f"₹{s['amount']}"} 
            for s in steps
        ])
        st.dataframe(settlements_display, use_container_width=True, hide_index=True)

# Download section
st.markdown("### Export Data")
col1, col2 = st.columns(2)

with col1:
    st.download_button(
        "Download Original Transactions",
        df.to_csv(index=False).encode('utf-8'),
        file_name='original_transactions.csv',
        mime='text/csv',
        use_container_width=True
    )

with col2:
    st.download_button(
        "Download Optimized Settlements",
        settlements_df.to_csv(index=False).encode('utf-8'),
        file_name='optimized_settlements.csv',
        mime='text/csv',
        use_container_width=True
    )

st.markdown("---")
st.caption("Algorithm: Computes net balances and progressively settles largest debts to minimize total transactions.")