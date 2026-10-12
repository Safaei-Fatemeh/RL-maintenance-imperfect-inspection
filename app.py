import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

import model as m

st.set_page_config(page_title="Maintenance under imperfect inspections", layout="wide")


@st.cache_resource
def resources():
    return m.load_dqn_weights(), m.load_paper_results()


W, R = resources()

PAGES = ["About", "Inspection calculator", "Simulate one history", "Compare policies", "Paper results"]
page = st.sidebar.radio("Page", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Companion app of the paper *Deep Reinforcement Learning for Multi-Component "
                   "Maintenance under Imperfect Inspections*.")


# ── About ────────────────────────────────────────────────────────────────────
if page == "About":
    st.title("Multi-component maintenance under imperfect inspections")
    st.markdown("""
This app accompanies the paper and lets you work with its model directly.

* Each component degrades according to a **gamma process**. An inspection gives a **noisy reading**
  whose error grows with the true degradation.
* The reading is combined with the component's **virtual age** (time since renewal, reduced by repairs)
  into a **closed-form Bayesian estimate** of the true degradation.
* Five maintenance actions are available per component: do nothing, imperfect repair, repair,
  imperfect replacement and replacement. Repeated imperfect actions lose effectiveness.
* All components are inspected together; the next inspection is brought forward as the most degraded
  component approaches its failure threshold.
* Policies compared: tuned heuristic rules and deep Q-networks with action branching trained in the paper.

**Pages**

1. *Inspection calculator*: from one reading to a degradation estimate, failure risk and a recommended action.
2. *Simulate one history*: follow a system over time under a chosen policy.
3. *Compare policies*: Monte Carlo cost comparison, with optional changes to costs and inspection quality.
4. *Paper results*: the main numbers reported in the paper.

All costs are in normalised cost units per unit time.
""")


# ── Inspection calculator ────────────────────────────────────────────────────
elif page == "Inspection calculator":
    st.title("Inspection calculator")
    st.write("Enter one component's parameters, its maintenance history and the latest reading.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Component")
        eta = st.number_input("Shape rate η", 0.01, 5.0, 0.20, 0.01, format="%.3f")
        theta = st.number_input("Scale θ", 0.01, 10.0, 1.00, 0.05, format="%.3f")
        H = st.number_input("Failure threshold H", 0.1, 100.0, 10.0, 0.5)
        gn = st.number_input("Inspection quality γ_noise (larger = more accurate)", 0.1, 100.0, 3.0, 0.5)
    with c2:
        st.subheader("History and reading")
        v = st.number_input("Virtual age v (time since renewal, adjusted for repairs)", 0.0, 1000.0, 25.0, 1.0)
        mrep = st.number_input("Imperfect actions since last replacement m", 0, 50, 0, 1)
        z = st.number_input("Latest reading z", 0.001, 100.0, 4.0, 0.1)
    with c3:
        st.subheader("Inspection rule and policy")
        dt_max = st.number_input("Δt_max", 0.1, 100.0, 15.0, 0.5)
        dt_min = st.number_input("Δt_min", 0.01, 10.0, 0.5, 0.1)
        gs = st.number_input("Scheduling sensitivity γ", 0.1, 10.0, 1.0, 0.1)
        kappa = st.number_input("Repair decay κ", 0.01, 1.0, 0.5, 0.05)
        fr0, f0, M0 = R["base_CBT-R-DS_params"]
        f_r = st.slider("Repair limit f_r (fraction of H)", 0.05, 1.0, float(fr0), 0.05)
        f = st.slider("Replacement limit f (fraction of H)", 0.05, 1.2, float(f0), 0.05)
        M = st.number_input("Maximum repairs before replacement M", 0, 100, int(M0), 1)

    shape, rate = m.posterior_params(np.array(z), np.array(v), eta, theta, gn)
    post = stats.gamma(a=shape, scale=1.0 / rate)
    x_hat = post.mean()

    st.subheader("Degradation estimate")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Reading z", f"{z:.2f}")
    k2.metric("Posterior mean x̂", f"{x_hat:.2f}")
    k3.metric("Posterior std", f"{post.std():.2f}")
    lo, hi = post.ppf([0.05, 0.95])
    k4.metric("90% credible interval", f"{lo:.2f} – {hi:.2f}")

    fig, ax = plt.subplots(figsize=(8, 3))
    xs = np.linspace(0, max(H * 1.3, hi * 1.2, z * 1.2), 400)
    if eta * v > 0:
        ax.plot(xs, stats.gamma.pdf(xs, a=eta * v, scale=theta), "--", color="grey", label="prior (from virtual age)")
    ax.plot(xs, post.pdf(xs), color="tab:blue", label="posterior (prior + reading)")
    ax.axvline(z, color="tab:orange", label="reading z")
    ax.axvline(H, color="red", ls=":", label="threshold H")
    ax.set_xlabel("degradation"); ax.set_yticks([]); ax.legend(fontsize=8)
    st.pyplot(fig)

    st.subheader("Actions")
    rows = []
    for k in range(5):
        mu_eff = 1.0 - (1.0 - m.MU[k]) * kappa ** mrep if 1 <= k <= 3 else m.MU[k]
        after = mu_eff * x_hat
        dt = max(dt_min, dt_max * max(0.0, 1.0 - after / H) ** gs)
        pf = m.failure_probability(shape, rate, mu_eff, eta, theta, H, dt)
        rows.append({"action": f"{k}: {m.ACTION_NAMES[k]}", "effective factor μ": round(mu_eff, 3),
                     "estimate after action": round(after, 2), "next interval Δt": round(dt, 2),
                     "P(failure before next inspection)": f"{pf:.1%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("For a single component. In a system, the common interval is set by the most degraded component.")

    if x_hat >= f * H or (x_hat >= f_r * H and mrep >= M):
        rec = 4
    elif x_hat >= f_r * H:
        rec = 1
    else:
        rec = 0
    st.success(f"Repair/replace rule (CBT-R-DS) recommends: **{rec}: {m.ACTION_NAMES[rec]}**")


# ── Simulate one history ─────────────────────────────────────────────────────
elif page == "Simulate one history":
    st.title("Simulate one history")
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
    k[3].metric("Maintenance cost / time", f"{res['maint_cost']:.2f}")
    k[4].metric("Failure cost / time", f"{res['fail_cost']:.2f}")

    tt = np.array([s["t"] for s in trace]); X = np.array([s["x"] for s in trace])
    Z = np.array([s["z"] for s in trace]); E = np.array([s["est"] for s in trace])
    A = np.array([s["a"] for s in trace]); DT = np.array([s["dt"] for s in trace])
    names = m.COMPONENT_NAMES[key]; H = np.broadcast_to(np.asarray(P["H"], float), (3,))
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 0.6]})
    for i, ax in enumerate(axes[:3]):
        ax.plot(tt, X[:, i], color="k", lw=1, label="true degradation")
        if info != "pi":
            ax.plot(tt, Z[:, i], ".", color="tab:orange", ms=4, label="reading")
        if info == "ii":
            ax.plot(tt, E[:, i], "x", color="tab:blue", ms=4, label="posterior mean")
        ax.axhline(H[i], color="red", ls="--", lw=1)
        for a, mk, col in [(1, "v", "tab:green"), (2, "v", "tab:olive"), (3, "^", "tab:purple"), (4, "^", "tab:red")]:
            sel = A[:, i] == a
            ax.plot(tt[sel], np.full(sel.sum(), -0.6), mk, color=col, ms=5,
                    label=m.ACTION_NAMES[a] if i == 0 else None)
        fails = [s["t"] + s["dt"] for s in trace if "failed" in s and s["failed"][i]]
        ax.plot(fails, [H[i]] * len(fails), "*", color="red", ms=10, label="failure" if i == 0 else None)
        ax.set_ylabel(names[i]); ax.set_ylim(-1.2, H[i] * 1.5); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=4, loc="upper left")
    axes[3].step(tt, DT, where="post"); axes[3].set_ylabel("interval Δt"); axes[3].set_xlabel("time")
    axes[3].grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    if pol_name.startswith("ABR"):
        st.info("ABR performs no inspections; the readings shown are not used by the policy.")


# ── Compare policies ─────────────────────────────────────────────────────────
elif page == "Compare policies":
    st.title("Compare policies")
    sys_name = st.selectbox("System", list(m.SYSTEMS))
    key, P0 = m.SYSTEMS[sys_name]
    policies = m.paper_policies(key, W, R)
    chosen = st.multiselect("Policies", list(policies), default=[p for p in policies if not p.startswith("DQN-II-FS")])
    n_runs = st.slider("Number of simulated histories per policy", 50, 1000, 200, 50)

    with st.expander("Change parameters (policies keep the settings tuned or trained in the paper)"):
        c1, c2, c3, c4 = st.columns(4)
        c_D = c1.number_input("Failure penalty c_D", 0.0, 10_000.0, float(P0["c_D"]), 10.0)
        c_I = c2.number_input("Inspection cost c_I", 0.0, 1000.0, float(P0["c_I"]), 0.5)
        gn_scale = c3.number_input("Multiply γ_noise by", 0.1, 20.0, 1.0, 0.1,
                                   help=f"Inspection quality; paper value(s): {P0['gamma_noise']}")
        kappa = c4.number_input("Repair decay κ", 0.01, 1.0, float(P0["delta"]), 0.05)
    P = dict(P0, c_D=c_D, c_I=c_I, delta=kappa,
             gamma_noise=(np.asarray(P0["gamma_noise"], float) * gn_scale).tolist())
    changed = (c_D, c_I, gn_scale, kappa) != (P0["c_D"], P0["c_I"], 1.0, P0["delta"])
    if changed:
        st.warning("Parameters differ from the paper: the heuristic thresholds and DQN agents were tuned or trained "
                   "for the paper settings and are not re-optimised here.")

    if st.button("Run comparison", type="primary") and chosen:
        rows = []
        bar = st.progress(0.0)
        for j, name in enumerate(chosen):
            fn, info, sched, dt_fixed, insp = policies[name]
            out = pd.DataFrame([m.run_episode(P, fn, info, sched, dt_fixed, 1_000_000 + s, insp) for s in range(n_runs)])
            rows.append({"policy": name, "cost per unit time": out.cost.mean(),
                         "± 95% CI": 1.96 * out.cost.std() / np.sqrt(n_runs),
                         "inspection": out.insp_cost.mean(), "maintenance": out.maint_cost.mean(),
                         "failures (cost)": out.fail_cost.mean(), "failures per history": out.failures.mean(),
                         "inspections per history": out.inspections.mean()})
            bar.progress((j + 1) / len(chosen))
        df = pd.DataFrame(rows).sort_values("cost per unit time")
        st.dataframe(df.round(3), hide_index=True, width="stretch")

        fig, ax = plt.subplots(figsize=(9, 0.5 * len(df) + 1))
        left = np.zeros(len(df))
        for col, colr in [("inspection", "tab:blue"), ("maintenance", "tab:green"), ("failures (cost)", "tab:red")]:
            ax.barh(df.policy, df[col], left=left, color=colr, label=col)
            left += df[col].values
        ax.invert_yaxis(); ax.set_xlabel("cost per unit time"); ax.legend(fontsize=8)
        st.pyplot(fig)
        st.caption("Common random numbers: every policy is evaluated on the same simulated histories.")


# ── Paper results ────────────────────────────────────────────────────────────
elif page == "Paper results":
    st.title("Results reported in the paper")
    st.write("Evaluated on 2,000 independent histories per policy (DQN results pooled over training seeds).")

    names = {"STP": "STP", "ABR": "ABR", "CBT-FS": "CBT-FS", "CBT-DS": "CBT-DS", "CBT-R-DS": "CBT-R-DS",
             "DQN-RZ-DS": "DQN-RZ-DS (raw reading)", "DQN-II-FS": "DQN-II-FS",
             "DQN-PI-DS": "DQN-PI-DS (true state)", "DQN-II-DS": "DQN-II-DS (proposed)"}
    for key, title in [("base", "Numerical study (Section 4)"), ("case", "Case study (Section 5)")]:
        st.subheader(title)
        rows = [{"policy": lab, "cost per unit time": R[f"{key}_cost_{k}"], "failures per history": R[f"{key}_fail_{k}"],
                 "inspections per history": R[f"{key}_insp_{k}"]}
                for k, lab in names.items() if f"{key}_cost_{k}" in R]
        st.dataframe(pd.DataFrame(rows).sort_values("cost per unit time"), hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Effect of inspection quality")
        f2 = pd.DataFrame(R["fig2"]).set_index("gamma_noise")
        st.line_chart(f2[["STP", "CBT-DS", "CBT-R-DS", "DQN-II-FS", "DQN-II-DS", "DQN-PI-DS"]])
        st.caption("x-axis: γ_noise (larger = more accurate inspections)")
    with c2:
        st.subheader("Effect of the number of components")
        t5 = pd.DataFrame(R["tab5"]).set_index("n")
        st.line_chart(t5[["STP", "CBT-R-DS", "DQN-II-DS"]])
        st.caption("x-axis: number of identical components n")
    st.subheader("Effect of the scheduling sensitivity γ")
    f3 = pd.DataFrame(R["fig3"]).set_index("gamma")
    st.dataframe(f3.round(3), width="stretch")
