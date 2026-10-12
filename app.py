import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats

import model as m

st.set_page_config(page_title="RL Maintenance under Imperfect Inspections", page_icon=":material/build:",
                   layout="wide")

PALETTE = {"STP": "#94A3B8", "ABR": "#F59E0B", "CBT-FS": "#06B6D4", "CBT-DS": "#0EA5E9",
           "CBT-R-DS": "#10B981", "DQN-II-FS": "#A78BFA", "DQN-RZ-DS": "#F472B6",
           "DQN-PI-DS": "#1F2937", "DQN-II-DS": "#4F46E5"}
ACTION_STYLE = {1: ("triangle-down", "#10B981"), 2: ("triangle-down", "#84CC16"),
                3: ("triangle-up", "#A855F7"), 4: ("triangle-up", "#EF4444")}

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
.hero {background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #0EA5E9 100%);
       padding: 1.6rem 2rem; border-radius: 18px; color: white; margin-bottom: 1.4rem;
       box-shadow: 0 10px 30px rgba(79,70,229,.25);}
.hero h1 {color: white; font-size: 2.0rem; margin: 0 0 .3rem 0; font-weight: 700;}
.hero p {color: rgba(255,255,255,.9); margin: 0; font-size: 1.02rem;}
div[data-testid="stMetric"] {background: #F4F6FB; border: 1px solid #E5E7EB; border-radius: 14px;
       padding: .9rem 1rem; box-shadow: 0 2px 6px rgba(15,23,42,.04);}
div[data-testid="stMetricValue"] {color: #4F46E5;}
.card {background: #F4F6FB; border: 1px solid #E5E7EB; border-radius: 14px; padding: 1.1rem 1.2rem;
       height: 100%;}
.card h4 {margin: 0 0 .4rem 0; color: #4F46E5;}
.card p {margin: 0; color: #374151; font-size: .95rem;}
.reco {background: #ECFDF5; border-left: 6px solid #10B981; border-radius: 10px; padding: 1rem 1.2rem;
       font-size: 1.05rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #1E1B4B 0%, #312E81 55%, #3730A3 100%);}
section[data-testid="stSidebar"] * {color: #E0E7FF;}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {border-radius: 10px; margin: 2px 6px;
       padding: .45rem .7rem; transition: all .15s ease;}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {background: rgba(255,255,255,.10);
       transform: translateX(3px);}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
       background: linear-gradient(90deg, rgba(165,180,252,.35), rgba(34,211,238,.20));
       box-shadow: inset 3px 0 0 #22D3EE;}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a span {font-weight: 500;}
.sb-title {font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: #A5B4FC !important;
       margin: 1.1rem 0 .5rem .2rem; font-weight: 700;}
.sb-grid {display: grid; grid-template-columns: 1fr 1fr; gap: .5rem;}
.sb-stat {background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.12); border-radius: 12px;
       padding: .6rem .7rem;}
.sb-stat .v {font-size: 1.25rem; font-weight: 700; color: #FFFFFF !important; line-height: 1.2;}
.sb-stat .l {font-size: .72rem; color: #C7D2FE !important;}
.sb-step {display: flex; gap: .6rem; align-items: flex-start; margin: .45rem 0; font-size: .85rem;}
.sb-step .n {flex: 0 0 22px; height: 22px; border-radius: 50%; background: linear-gradient(135deg,#A5B4FC,#22D3EE);
       color: #1E1B4B !important; font-weight: 700; font-size: .75rem; display: flex; align-items: center;
       justify-content: center;}
.sb-foot {margin-top: 1.2rem; padding-top: .8rem; border-top: 1px solid rgba(255,255,255,.15);
       font-size: .75rem; color: #A5B4FC !important; line-height: 1.5;}
</style>
""", unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def card(title, text):
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)


def short(name):
    return name.split(" ")[0]


def style(fig, height=420):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), font=dict(size=13))
    return fig


@st.cache_resource
def resources():
    return m.load_dqn_weights(), m.load_paper_results()


W, R = resources()


# ── About ────────────────────────────────────────────────────────────────────
def page_about():
    hero("Deep Reinforcement Learning for Multi-Component Maintenance",
         "Interactive companion to the paper: imperfect inspections, Bayesian degradation estimates, "
         "adaptive inspection intervals and learned maintenance policies.")
    b, ii, pi = R["base_cost_CBT-R-DS"], R["base_cost_DQN-II-DS"], R["base_cost_DQN-PI-DS"]
    k = st.columns(4)
    k[0].metric("Best implementable policy", "CBT-R-DS", f"{100 * (R['base_cost_STP'] - b) / R['base_cost_STP']:.1f}% vs static threshold",
                delta_color="normal")
    k[1].metric("Proposed DQN cost", f"{ii:.2f}", f"{100 * (ii - b) / b:+.1f}% vs CBT-R-DS", delta_color="inverse")
    k[2].metric("DQN with true state", f"{pi:.2f}", f"{100 * (pi - b) / b:+.1f}% vs CBT-R-DS", delta_color="inverse")
    k[3].metric("Posterior check (MC error)", f"{R['val_rel_err_pct']:.2f}%")

    st.write("")
    c = st.columns(3)
    with c[0]:
        card("Gamma degradation", "Each component wears according to a gamma process and fails when it reaches "
             "its threshold. Five actions are available, from doing nothing to replacement; repeated "
             "imperfect repairs become less effective.")
    with c[1]:
        card("Imperfect inspections", "A reading is unbiased but its error grows with the true degradation. "
             "Combined with the component's virtual age, it gives a closed-form gamma posterior.")
    with c[2]:
        card("Adaptive intervals", "All components are inspected together. The next inspection is brought "
             "forward as the most degraded component approaches its threshold.")
    st.write("")
    c = st.columns(3)
    with c[0]:
        card("Inspection calculator", "From one reading to an estimate, failure risk per "
             "action and a recommended action.")
    with c[1]:
        card("Simulate one history", "Follow a system over time under any policy, "
             "including the trained DQN agents.")
    with c[2]:
        card("Compare policies", "Monte Carlo cost comparison with a breakdown into "
             "inspection, maintenance and failure costs.")

    st.markdown("### Who is this app for?")
    c = st.columns(4)
    with c[0]:
        card("Maintenance engineers", "Turn a noisy inspection reading into a degradation estimate, see the "
             "failure risk of each action and get a recommended action and next inspection date.")
    with c[1]:
        card("Reliability and asset managers", "Compare maintenance strategies and see how inspection cost, "
             "failure cost and inspection quality change the total cost of a policy.")
    with c[2]:
        card("Researchers and students", "Explore the gamma degradation model, the Bayesian update and the "
             "trained reinforcement learning agents, and reproduce the results of the paper.")
    with c[3]:
        card("Reviewers and readers", "Check the paper's results interactively and test the policies on "
             "simulated histories of their own choosing.")
    st.write("")
    st.caption("All costs are in normalised cost units per unit time.")


# ── Inspection calculator ────────────────────────────────────────────────────
def page_calculator():
    hero("Inspection calculator", "Turn one noisy reading into a degradation estimate, a failure risk and a decision.")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Component**")
            eta = st.number_input("Shape rate η", 0.01, 5.0, 0.20, 0.01, format="%.3f")
            theta = st.number_input("Scale θ", 0.01, 10.0, 1.00, 0.05, format="%.3f")
            H = st.number_input("Failure threshold H", 0.1, 100.0, 10.0, 0.5)
            gn = st.number_input("Inspection quality γ_noise", 0.1, 100.0, 3.0, 0.5,
                                 help="Larger values mean more accurate readings (std of a reading = x/√γ).")
        with c2:
            st.markdown("**History and reading**")
            v = st.number_input("Virtual age v", 0.0, 1000.0, 25.0, 1.0,
                                help="Time since renewal, reduced by the effective factor of each repair.")
            mrep = st.number_input("Imperfect actions since last replacement m", 0, 50, 0, 1)
            z = st.number_input("Latest reading z", 0.001, 100.0, 4.0, 0.1)
            kappa = st.number_input("Repair decay κ", 0.01, 1.0, 0.5, 0.05)
        with c3:
            st.markdown("**Inspection rule and policy**")
            dt_max = st.number_input("Δt_max", 0.1, 100.0, 15.0, 0.5)
            dt_min = st.number_input("Δt_min", 0.01, 10.0, 0.5, 0.1)
            gs = st.number_input("Scheduling sensitivity γ", 0.1, 10.0, 1.0, 0.1)
            fr0, f0, M0 = R["base_CBT-R-DS_params"]
            f_r, f = st.slider("Repair and replacement limits (fraction of H)", 0.05, 1.2,
                               (float(fr0), float(f0)), 0.05)
            M = st.number_input("Maximum repairs before replacement M", 0, 100, int(M0), 1)

    shape, rate = m.posterior_params(np.array(z), np.array(v), eta, theta, gn)
    post = stats.gamma(a=shape, scale=1.0 / rate)
    x_hat = post.mean(); lo, hi = post.ppf([0.05, 0.95])

    k = st.columns(4)
    k[0].metric("Reading z", f"{z:.2f}")
    k[1].metric("Posterior mean x̂", f"{x_hat:.2f}", f"{x_hat - z:+.2f} vs reading", delta_color="off")
    k[2].metric("Posterior std", f"{post.std():.2f}")
    k[3].metric("90% credible interval", f"{lo:.2f} – {hi:.2f}")

    xs = np.linspace(0, max(H * 1.3, hi * 1.2, z * 1.2), 400)
    fig = go.Figure()
    if eta * v > 0:
        fig.add_trace(go.Scatter(x=xs, y=stats.gamma.pdf(xs, a=eta * v, scale=theta), name="Prior (virtual age)",
                                 line=dict(color="#94A3B8", dash="dash")))
    fig.add_trace(go.Scatter(x=xs, y=post.pdf(xs), name="Posterior", fill="tozeroy",
                             line=dict(color="#4F46E5", width=3), fillcolor="rgba(79,70,229,.15)"))
    fig.add_vline(x=z, line_color="#F59E0B", line_width=2, annotation_text="reading")
    fig.add_vline(x=H, line_color="#EF4444", line_dash="dot", annotation_text="threshold")
    fig.update_layout(xaxis_title="degradation", yaxis_title="density")

    rows = []
    for a in range(5):
        mu_eff = 1.0 - (1.0 - m.MU[a]) * kappa ** mrep if 1 <= a <= 3 else m.MU[a]
        after = mu_eff * x_hat
        dt = max(dt_min, dt_max * max(0.0, 1.0 - after / H) ** gs)
        pf = m.failure_probability(shape, rate, mu_eff, eta, theta, H, dt)
        rows.append({"Action": f"{a}: {m.ACTION_NAMES[a]}", "Effective factor": mu_eff,
                     "Estimate after action": after, "Next interval": dt, "Failure probability": pf})
    df = pd.DataFrame(rows)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(style(fig, 380), width="stretch")
    with c2:
        bar = go.Figure(go.Bar(y=df["Action"], x=df["Failure probability"], orientation="h",
                               marker_color=["#94A3B8", "#10B981", "#84CC16", "#A855F7", "#EF4444"],
                               text=[f"{p:.1%}" for p in df["Failure probability"]], textposition="outside"))
        bar.update_layout(title="Failure risk before the next inspection", xaxis_tickformat=".0%",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(style(bar, 380), width="stretch")

    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "Effective factor": st.column_config.NumberColumn(format="%.3f"),
        "Estimate after action": st.column_config.NumberColumn(format="%.2f"),
        "Next interval": st.column_config.NumberColumn(format="%.2f"),
        "Failure probability": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})

    rec = 4 if (x_hat >= f * H or (x_hat >= f_r * H and mrep >= M)) else (1 if x_hat >= f_r * H else 0)
    st.markdown(f'<div class="reco">Recommended by the repair/replace rule: <b>{rec}: {m.ACTION_NAMES[rec]}</b>'
                f' &nbsp;·&nbsp; next inspection in <b>{df["Next interval"][rec]:.1f}</b> time units</div>',
                unsafe_allow_html=True)
    st.caption("Single component. In a system, the common interval is set by the most degraded component.")


# ── Simulate one history ─────────────────────────────────────────────────────
def page_simulate():
    hero("Simulate one history", "Watch a system degrade, get inspected and maintained under a policy of your choice.")
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        sys_name = c1.selectbox("System", list(m.SYSTEMS))
        key, P = m.SYSTEMS[sys_name]
        policies = m.paper_policies(key, W, R)
        pol_name = c2.selectbox("Policy", list(policies), index=list(policies).index("DQN-II-DS (proposed)"))
        seed = c3.number_input("Random seed", 0, 10**6, 1, 1)
    fn, info, sched, dt_fixed, insp = policies[pol_name]
    res, trace = m.run_episode(P, fn, info, sched, dt_fixed, int(seed), insp, log=True)

    k = st.columns(5)
    k[0].metric("Cost per unit time", f"{res['cost']:.2f}")
    k[1].metric("Failures", res["failures"])
    k[2].metric("Inspections", res["inspections"])
    k[3].metric("Maintenance cost", f"{res['maint_cost']:.2f}")
    k[4].metric("Failure cost", f"{res['fail_cost']:.2f}")

    tt = np.array([s["t"] for s in trace]); X = np.array([s["x"] for s in trace])
    Z = np.array([s["z"] for s in trace]); E = np.array([s["est"] for s in trace])
    A = np.array([s["a"] for s in trace]); DT = np.array([s["dt"] for s in trace])
    names = m.COMPONENT_NAMES[key]; H = np.broadcast_to(np.asarray(P["H"], float), (3,))

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.29, 0.29, 0.29, 0.13], subplot_titles=names + ["Inspection interval"])
    for i in range(3):
        show = i == 0
        fig.add_trace(go.Scatter(x=tt, y=X[:, i], name="True degradation", legendgroup="x", showlegend=show,
                                 line=dict(color="#111827", width=1.6)), row=i + 1, col=1)
        if info != "pi":
            fig.add_trace(go.Scatter(x=tt, y=Z[:, i], name="Reading", mode="markers", legendgroup="z",
                                     showlegend=show, marker=dict(color="#F59E0B", size=5)), row=i + 1, col=1)
        if info == "ii":
            fig.add_trace(go.Scatter(x=tt, y=E[:, i], name="Posterior mean", mode="markers", legendgroup="e",
                                     showlegend=show, marker=dict(color="#4F46E5", size=6, symbol="x")),
                          row=i + 1, col=1)
        fig.add_hline(y=H[i], line_color="#EF4444", line_dash="dot", row=i + 1, col=1)
        for a, (sym, col) in ACTION_STYLE.items():
            sel = A[:, i] == a
            fig.add_trace(go.Scatter(x=tt[sel], y=np.full(sel.sum(), -0.6), mode="markers", name=m.ACTION_NAMES[a],
                                     legendgroup=f"a{a}", showlegend=show,
                                     marker=dict(symbol=sym, color=col, size=10)), row=i + 1, col=1)
        fails = [s["t"] + s["dt"] for s in trace if "failed" in s and s["failed"][i]]
        fig.add_trace(go.Scatter(x=fails, y=[H[i]] * len(fails), mode="markers", name="Failure", legendgroup="f",
                                 showlegend=show, marker=dict(symbol="star", color="#EF4444", size=14)),
                      row=i + 1, col=1)
        fig.update_yaxes(range=[-1.2, H[i] * 1.5], row=i + 1, col=1)
    fig.add_trace(go.Scatter(x=tt, y=DT, line_shape="hv", name="Δt", showlegend=False,
                             line=dict(color="#0EA5E9")), row=4, col=1)
    fig.update_xaxes(title_text="time", row=4, col=1)
    st.plotly_chart(style(fig, 820), width="stretch")
    if pol_name.startswith("ABR"):
        st.info("ABR performs no inspections; the readings are shown for reference only.")


# ── Compare policies ─────────────────────────────────────────────────────────
def page_compare():
    hero("Compare policies", "Monte Carlo comparison on common simulated histories, with a cost breakdown.")
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        sys_name = c1.selectbox("System", list(m.SYSTEMS))
        n_runs = c2.slider("Histories per policy", 50, 1000, 200, 50)
        key, P0 = m.SYSTEMS[sys_name]
        policies = m.paper_policies(key, W, R)
        chosen = st.multiselect("Policies", list(policies),
                                default=[p for p in policies if not p.startswith("DQN-II-FS")])
        with st.expander("Change parameters (policies keep the settings tuned or trained in the paper)"):
            c = st.columns(4)
            c_D = c[0].number_input("Failure penalty c_D", 0.0, 10_000.0, float(P0["c_D"]), 10.0)
            c_I = c[1].number_input("Inspection cost c_I", 0.0, 1000.0, float(P0["c_I"]), 0.5)
            gn_scale = c[2].number_input("Multiply γ_noise by", 0.1, 20.0, 1.0, 0.1,
                                         help=f"Inspection quality; paper value(s): {P0['gamma_noise']}")
            kappa = c[3].number_input("Repair decay κ", 0.01, 1.0, float(P0["delta"]), 0.05)
        run = st.button("Run comparison", type="primary", icon=":material/play_arrow:")
    P = dict(P0, c_D=c_D, c_I=c_I, delta=kappa, gamma_noise=(np.asarray(P0["gamma_noise"], float) * gn_scale).tolist())
    if (c_D, c_I, gn_scale, kappa) != (P0["c_D"], P0["c_I"], 1.0, P0["delta"]):
        st.warning("Parameters differ from the paper: heuristic thresholds and DQN agents are not re-optimised here.")
    if not (run and chosen):
        st.info("Choose policies and press **Run comparison**.")
        return

    rows, costs = [], {}
    bar = st.progress(0.0, text="Simulating...")
    for j, name in enumerate(chosen):
        fn, info, sched, dt_fixed, insp = policies[name]
        out = pd.DataFrame([m.run_episode(P, fn, info, sched, dt_fixed, 1_000_000 + s, insp) for s in range(n_runs)])
        costs[name] = out.cost.values
        rows.append({"Policy": name, "Cost": out.cost.mean(), "95% CI": 1.96 * out.cost.std() / np.sqrt(n_runs),
                     "Inspection": out.insp_cost.mean(), "Maintenance": out.maint_cost.mean(),
                     "Failures (cost)": out.fail_cost.mean(), "Failures per history": out.failures.mean(),
                     "Inspections per history": out.inspections.mean()})
        bar.progress((j + 1) / len(chosen), text=f"Simulated {name}")
    bar.empty()
    df = pd.DataFrame(rows).sort_values("Cost").reset_index(drop=True)
    best = df.iloc[0]
    k = st.columns(3)
    k[0].metric("Cheapest policy", short(best.Policy), f"{best.Cost:.3f} per unit time", delta_color="off")
    k[1].metric("Most expensive", short(df.iloc[-1].Policy), f"{df.iloc[-1].Cost:.3f} per unit time", delta_color="off")
    k[2].metric("Spread", f"{100 * (df.iloc[-1].Cost - best.Cost) / best.Cost:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for col, colr in [("Inspection", "#0EA5E9"), ("Maintenance", "#10B981"), ("Failures (cost)", "#EF4444")]:
            fig.add_trace(go.Bar(y=df.Policy.map(short), x=df[col], name=col, orientation="h", marker_color=colr))
        fig.update_layout(barmode="stack", title="Cost breakdown", yaxis=dict(autorange="reversed"),
                          xaxis_title="cost per unit time")
        st.plotly_chart(style(fig, 420), width="stretch")
    with c2:
        fig = go.Figure()
        for name in df.Policy:
            fig.add_trace(go.Box(x=costs[name], name=short(name), marker_color=PALETTE.get(short(name), "#4F46E5"),
                                 boxmean=True))
        fig.update_layout(title="Distribution of cost per history", showlegend=False,
                          yaxis=dict(autorange="reversed"), xaxis_title="cost per unit time")
        st.plotly_chart(style(fig, 420), width="stretch")
    st.dataframe(df.round(3), hide_index=True, width="stretch", column_config={
        "Cost": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=float(df.Cost.max()))})
    st.caption("Common random numbers: every policy is evaluated on the same simulated histories.")


# ── Paper results ────────────────────────────────────────────────────────────
def page_results():
    hero("Results reported in the paper", "2,000 evaluation histories per policy; DQN results pooled over training seeds.")
    labels = ["STP", "ABR", "CBT-FS", "CBT-DS", "CBT-R-DS", "DQN-RZ-DS", "DQN-II-FS", "DQN-PI-DS", "DQN-II-DS"]
    t1, t2, t3, t4 = st.tabs([":material/table: Main results", ":material/precision_manufacturing: Case study",
                              ":material/visibility: Inspection quality & size", ":material/schedule: Scheduling"])
    for tab, key in [(t1, "base"), (t2, "case")]:
        with tab:
            df = pd.DataFrame([{"Policy": k, "Cost": R[f"{key}_cost_{k}"], "Failures": R[f"{key}_fail_{k}"],
                                "Inspections": R[f"{key}_insp_{k}"]} for k in labels if f"{key}_cost_{k}" in R])
            df = df.sort_values("Cost")
            fig = go.Figure(go.Bar(x=df.Cost, y=df.Policy, orientation="h",
                                   marker_color=[PALETTE[p] for p in df.Policy],
                                   text=[f"{c:.2f}" for c in df.Cost], textposition="outside"))
            fig.update_layout(xaxis_title="cost per unit time", yaxis=dict(autorange="reversed"))
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(style(fig, 420), width="stretch")
            c2.dataframe(df, hide_index=True, width="stretch")
    with t3:
        c1, c2 = st.columns(2)
        f2 = pd.DataFrame(R["fig2"])
        fig = go.Figure()
        for col in ["STP", "CBT-DS", "CBT-R-DS", "DQN-II-FS", "DQN-II-DS", "DQN-PI-DS"]:
            fig.add_trace(go.Scatter(x=f2.gamma_noise, y=f2[col], name=col, mode="lines+markers",
                                     line=dict(color=PALETTE[col], width=3 if col == "DQN-II-DS" else 2,
                                               dash="dot" if col == "DQN-PI-DS" else "solid")))
        fig.update_layout(title="Cost vs inspection quality", xaxis_type="log",
                          xaxis_title="γ_noise (larger = more accurate)", yaxis_title="cost per unit time")
        c1.plotly_chart(style(fig, 440), width="stretch")
        t5 = pd.DataFrame(R["tab5"])
        fig = go.Figure()
        for col in ["STP", "CBT-R-DS", "DQN-II-DS"]:
            fig.add_trace(go.Scatter(x=t5.n, y=t5[col], name=col, mode="lines+markers",
                                     line=dict(color=PALETTE[col], width=3)))
        fig.update_layout(title="Cost vs number of components", xaxis_title="n", yaxis_title="cost per unit time")
        c2.plotly_chart(style(fig, 440), width="stretch")
    with t4:
        f3 = pd.DataFrame(R["fig3"])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=f3.gamma, y=f3["DQN-II-DS cost"], name="DQN-II-DS cost", mode="lines+markers",
                                 line=dict(color=PALETTE["DQN-II-DS"], width=3)))
        fig.add_trace(go.Scatter(x=f3.gamma, y=f3["CBT-DS cost"], name="CBT-DS cost", mode="lines+markers",
                                 line=dict(color=PALETTE["CBT-DS"], width=2)))
        fig.add_trace(go.Bar(x=f3.gamma, y=f3["DQN-II-DS inspections"], name="DQN-II-DS inspections",
                             marker_color="rgba(148,163,184,.35)"), secondary_y=True)
        fig.update_layout(title="Effect of scheduling sensitivity γ", xaxis_title="γ")
        fig.update_yaxes(title_text="cost per unit time", secondary_y=False)
        fig.update_yaxes(title_text="inspections per history", secondary_y=True)
        st.plotly_chart(style(fig, 440), width="stretch")


st.logo(str(m.DATA / "logo.svg"), size="large", icon_image=str(m.DATA / "logo_icon.svg"))
nav = st.navigation([
    st.Page(page_about, title="Overview", icon=":material/home:", default=True),
    st.Page(page_calculator, title="Inspection calculator", icon=":material/calculate:"),
    st.Page(page_simulate, title="Simulate one history", icon=":material/timeline:"),
    st.Page(page_compare, title="Compare policies", icon=":material/leaderboard:"),
    st.Page(page_results, title="Paper results", icon=":material/insights:"),
])
rule_saving = 100 * (R["base_cost_STP"] - R["base_cost_CBT-R-DS"]) / R["base_cost_STP"]
with st.sidebar:
    st.markdown(f"""
<div class="sb-title">Key results</div>
<div class="sb-grid">
  <div class="sb-stat"><div class="v">{rule_saving:.1f}%</div><div class="l">Bayesian rule vs static threshold</div></div>
  <div class="sb-stat"><div class="v">{R['base_saving_vs_DQN-II-FS']:.1f}%</div><div class="l">dynamic vs fixed interval (DQN)</div></div>
  <div class="sb-stat"><div class="v">{R['base_saving_vs_DQN-RZ-DS']:.1f}%</div><div class="l">posterior vs raw reading (DQN)</div></div>
  <div class="sb-stat"><div class="v">{R['val_rel_err_pct']:.2f}%</div><div class="l">posterior error vs Monte Carlo</div></div>
</div>
<div class="sb-title">How it works</div>
<div class="sb-step"><div class="n">1</div><div>Inspect all components and get noisy readings</div></div>
<div class="sb-step"><div class="n">2</div><div>Combine each reading with the virtual age into a Bayesian estimate</div></div>
<div class="sb-step"><div class="n">3</div><div>Choose one of five actions per component</div></div>
<div class="sb-step"><div class="n">4</div><div>Set the next inspection from the most degraded component</div></div>
<div class="sb-foot">Companion app of the paper<br><b>Deep Reinforcement Learning for Multi-Component
Maintenance under Imperfect Inspections</b></div>
""", unsafe_allow_html=True)
nav.run()
