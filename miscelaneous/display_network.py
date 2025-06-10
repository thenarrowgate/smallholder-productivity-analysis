import pandas as pd
import networkx as nx
import numpy as np
import ast

from dash import Dash, html, dcc, Input, Output, callback_context, exceptions
import plotly.graph_objects as go

# ─── CONFIG ────────────────────────────────────────────────────────────────
# Path to your CSV; it must have columns: X, Y, Type, Estimate
csv_path = "results.csv"

# ─── LOAD AND PARSE ─────────────────────────────────────────────────────────
# Read in the CSV
associations = pd.read_csv(csv_path)

# Convert the 'Estimate' column:
#  - LIN → float
#  - NL  → dict parsed from its string representation
def parse_estimate(row):
    if row["Type"] == "LIN":
        return float(row["Estimate"])
    else:
        # safely evaluate the Python‐literal dict in the CSV string :contentReference[oaicite:0]{index=0}
        return ast.literal_eval(row["Estimate"])

associations["Estimate"] = associations.apply(parse_estimate, axis=1)

# ─── BUILD GRAPH + PLOTLY FIGURE ─────────────────────────────────────────────
# Create directed graph and edge‐labels
G = nx.DiGraph()
for _, row in associations.iterrows():
    if row["Type"] == "LIN":
        label = f"LIN: {row['Estimate']:.2f}"
    else:
        lines = [f"{interval}: {slope:.2f}"
                 for interval, slope in row["Estimate"].items()]
        label = "NL:<br>" + "<br>".join(lines)
    G.add_edge(row["X"], row["Y"], label=label, type=row["Type"])

# Compute layout
pos = nx.spring_layout(G, seed=42, k=0.5/np.sqrt(len(G.nodes())))

# Edge trace
edge_x, edge_y = [], []
for u, v in G.edges():
    x0, y0 = pos[u]; x1, y1 = pos[v]
    edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    mode="lines",
    line=dict(width=1, color="gray"),
    hoverinfo="none"
)

# ─── NODE TRACES WITH HIGHLIGHT ─────────────────────────────────────────────
target = "Q0__AGR_PROD__continuous"

# Separate regular nodes and the highlighted node
regular_nodes = [n for n in G.nodes() if n != target]
highlight_nodes = [target] if target in G.nodes() else []

# Regular node trace
node_x_reg = [pos[n][0] for n in regular_nodes]
node_y_reg = [pos[n][1] for n in regular_nodes]
node_trace_reg = go.Scatter(
    x=node_x_reg, y=node_y_reg,
    mode="markers+text",
    text=regular_nodes,
    textposition="top center",
    marker=dict(color="lightblue", size=10, line=dict(width=1, color="darkblue")),
    hoverinfo="text"
)

# Highlighted node trace
if highlight_nodes:
    node_x_hl = [pos[target][0]]
    node_y_hl = [pos[target][1]]
    node_trace_hl = go.Scatter(
        x=node_x_hl, y=node_y_hl,
        mode="markers+text",
        text=highlight_nodes,
        textposition="top center",
        marker=dict(color="red", size=14, line=dict(width=2, color="darkred")),
        textfont=dict(color="red", size=12, family="Arial Black"),
        hoverinfo="text"
    )
    data_traces = [edge_trace, node_trace_reg, node_trace_hl]
else:
    data_traces = [edge_trace, node_trace_reg]

fig = go.Figure(data=data_traces,
    layout=go.Layout(
        title="Association Network",
        hovermode="closest",
        clickmode="event+select",
        showlegend=False,
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
)
# ─── DASH APP ─────────────────────────────────────────────────────────────────
app = Dash(__name__)
app.layout = html.Div(style={"height": "100vh", "display": "flex", "flexDirection": "column"}, children=[
    html.Div("Association Network", style={"textAlign": "center", "fontSize": "24px", "padding": "10px"}),
    dcc.Graph(id="graph", figure=fig, style={"flex": "1", "width": "100%"}),
    html.Div(id="modal", style={
        "display": "none", "position": "fixed", "top": "10%", "left": "10%",
        "width": "80%", "height": "80%", "overflowY": "auto",
        "backgroundColor": "white", "border": "2px solid black", "padding": "20px", "zIndex": "1000"
    }, children=[
        html.Button("Close", id="close", style={"float": "right", "marginBottom": "10px"}),
        html.Div(id="modal-content")
    ]),
])

@app.callback(
    [Output("modal-content", "children"), Output("modal", "style")],
    [Input("graph", "clickData"), Input("close", "n_clicks")],
    prevent_initial_call=True
)
def toggle_modal(clickData, close_click):
    ctx = callback_context
    if not ctx.triggered:
        raise exceptions.PreventUpdate

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger == "close":
        return "", {"display": "none"}

    pt = clickData["points"][0]
    if pt.get("curveNumber") != 1:
        raise exceptions.PreventUpdate

    node = pt["text"]
    parents  = associations[associations["Y"] == node]
    children = associations[associations["X"] == node]

    content = []
    if not parents.empty:
        content.append(html.H4(f"{node} Explained by:"))
        for _, r in parents.iterrows():
            content.append(html.Div(f"{r['X']} → {r['Y']} ({r['Type']}): {r['Estimate']}"))

    if not children.empty:
        content.append(html.H4(f"{node} Explains:"))
        for _, r in children.iterrows():
            content.append(html.Div(f"{r['X']} → {r['Y']} ({r['Type']}): {r['Estimate']}"))

    if not content:
        content = [html.Div(f"No connections for {node}.")]

    return content, {"display": "block"}

if __name__ == "__main__":
    app.run(debug=True)
