import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import simulate

st.set_page_config(page_title="T2DM Complication Trajectories", page_icon="🧬", layout="wide")
st.title("🧬 Type 2 Diabetes – Complication Trajectories")
st.caption("Graph analysis of the order in which complications appear after diagnosis · simulated cohort (4,000 patients, 15-year follow-up)")


@st.cache_data
def load():
    return simulate()


df = load()
with st.sidebar:
    st.header("Cohort filters")
    sex = st.multiselect("Sex", ["Female", "Male"], default=["Female", "Male"])
    age = st.slider("Age at diagnosis", 30, 85, (30, 85))
    min_w = st.slider("Minimum transitions shown in network", 5, 150, 40, 5)
    st.markdown("[Portfolio](https://noe2019.github.io/DA-Portfolio-Website/)")

f = df[df.sex.isin(sex) & df.age_at_diagnosis.between(*age)]
if f.empty:
    st.warning("No patients match the filters.")
    st.stop()
f = f.sort_values(["patient_id", "order"])
f["next"] = f.groupby("patient_id").complication.shift(-1)
f["gap"] = f.groupby("patient_id").years_since_diagnosis.shift(-1) - f.years_since_diagnosis
tr = f.dropna(subset=["next"]).groupby(["complication", "next"]).agg(n=("patient_id", "size"), median_gap=("gap", "median")).reset_index()

k = st.columns(4)
k[0].metric("Patients with ≥ 1 complication", f"{f.patient_id.nunique():,}")
k[1].metric("Complication events", f"{len(f):,}")
k[2].metric("Mean complications / patient", f"{f.groupby('patient_id').size().mean():.2f}")
k[3].metric("Median time to 1st complication", f"{f[f.order == 1].years_since_diagnosis.median():.1f} y")

t1, t2, t3 = st.tabs(["Transition network", "Sankey (first 3)", "Trajectories & timing"])
with t1:
    G = nx.DiGraph()
    for _, e in tr[tr.n >= min_w].iterrows():
        G.add_edge(e.complication, e.next, weight=e.n)
    if G.number_of_edges() == 0:
        st.info("Lower the minimum number of transitions.")
    else:
        pos = nx.spring_layout(G, seed=3, k=1.2, weight=None)
        pr = nx.pagerank(G, weight="weight")
        fig = go.Figure()
        wmax = max(d["weight"] for *_, d in G.edges(data=True))
        for u, v, d in G.edges(data=True):
            fig.add_annotation(x=pos[v][0], y=pos[v][1], ax=pos[u][0], ay=pos[u][1], xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1 + 6 * d["weight"] / wmax, arrowcolor="rgba(46,117,182,.55)", standoff=14)
        nodes = list(G.nodes)
        fig.add_scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text", text=nodes, textposition="top center",
                        marker=dict(size=[20 + 300 * pr[n] for n in nodes], color=[pr[n] for n in nodes], colorscale="Reds", line=dict(width=1, color="#333")),
                        hovertext=[f"{n}<br>PageRank {pr[n]:.3f}<br>in {G.in_degree(n, weight='weight'):.0f} · out {G.out_degree(n, weight='weight'):.0f}" for n in nodes],
                        hoverinfo="text")
        fig.update_layout(height=620, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), title="Directed complication network (edge width = patients)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame({"PageRank": pr, "Betweenness": nx.betweenness_centrality(G)}).sort_values("PageRank", ascending=False)
                     .style.format("{:.3f}"), use_container_width=True)
with t2:
    s = f[f.order <= 3].pivot(index="patient_id", columns="order", values="complication")
    links = []
    for a, b in [(1, 2), (2, 3)]:
        if a in s and b in s:
            g = s.dropna(subset=[a, b]).groupby([a, b]).size().reset_index(name="n")
            links += [(f"{x} ({a})", f"{y} ({b})", n) for x, y, n in g.itertuples(index=False)]
    labels = sorted({l for x, y, _ in links for l in (x, y)})
    idx = {l: i for i, l in enumerate(labels)}
    st.plotly_chart(go.Figure(go.Sankey(node=dict(label=labels, pad=12), link=dict(source=[idx[x] for x, _, _ in links],
                                                                               target=[idx[y] for _, y, _ in links], value=[n for *_, n in links])))
                    .update_layout(height=650, title="Flow from 1st → 2nd → 3rd complication"), use_container_width=True)
with t3:
    paths = f.groupby("patient_id").complication.apply(lambda s: " → ".join(s.head(3)))
    top = paths.value_counts().head(15).rename_axis("trajectory (first 3)").reset_index(name="patients")
    a, b = st.columns(2)
    a.dataframe(top, hide_index=True, use_container_width=True)
    first = f[f.order == 1].complication.value_counts()
    b.plotly_chart(px.bar(first.sort_values(), orientation="h", title="First complication", labels={"value": "Patients", "index": ""})
                   .update_layout(showlegend=False), use_container_width=True)
    st.plotly_chart(px.box(f, x="complication", y="years_since_diagnosis", title="Years from diagnosis to complication onset")
                    .update_xaxes(tickangle=-35), use_container_width=True)
    st.dataframe(tr.sort_values("n", ascending=False).head(20).rename(columns={"complication": "from", "next": "to", "n": "patients",
                                                                              "median_gap": "median years between"}).round(2),
                 hide_index=True, use_container_width=True)
