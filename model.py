"""Simulation model of the paper: gamma degradation, imperfect inspections, maintenance policies."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

DATA = Path(__file__).parent / "data"
MU = np.array([1.0, 0.7, 0.5, 0.2, 0.0])
ACTION_NAMES = ["Do nothing", "Imperfect repair", "Repair", "Imperfect replacement", "Replacement"]

BASE = dict(eta=[0.20, 0.16, 0.24], theta=[1.00, 1.25, 0.875], H=[10.0, 10.0, 10.0],
            c_act=[0, 2, 5, 8, 12], c_D=300.0, c_I=4.0, gamma_noise=3.0,
            T=500.0, dt_max=15.0, dt_min=0.5, gamma_sched=1.0, delta=0.5)
CASE = dict(eta=[0.18, 0.24, 0.32], theta=[1.25, 1.00, 0.75], H=[12.0, 10.0, 8.0],
            c_act=[[0, 3, 7, 14, 20], [0, 4, 9, 18, 28], [0, 2, 4, 9, 14]],
            c_D=400.0, c_I=6.0, gamma_noise=[6.0, 2.0, 4.0], T=260.0,
            dt_max=12.0, dt_min=0.5, gamma_sched=1.0, delta=0.5)
SYSTEMS = {"Numerical study (Section 4)": ("base", BASE),
           "Case study: conveyor-belt drive (Section 5)": ("case", CASE)}
COMPONENT_NAMES = {"base": ["Motor", "Pump", "Conveyor"], "case": ["Drive motor", "Gearbox", "Belt"]}


class MaintEnv:
    def __init__(self, eta, theta, H, c_act, c_D, c_I, gamma_noise, T,
                 dt_max, dt_min, gamma_sched, delta, mu=MU):
        self.eta = np.asarray(eta, float); self.theta = np.asarray(theta, float)
        self.H = np.asarray(H, float); self.n = len(self.eta)
        self.c_act = np.broadcast_to(np.asarray(c_act, float), (self.n, 5)).copy()
        self.gn = np.broadcast_to(np.asarray(gamma_noise, float), (self.n,)).copy()
        self.c_D, self.c_I, self.T = c_D, c_I, T
        self.dt_max, self.dt_min, self.gs, self.delta = dt_max, dt_min, gamma_sched, delta
        self.mu = np.asarray(mu, float); self.idx = np.arange(self.n)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.x = np.zeros(self.n); self.v = np.zeros(self.n); self.m = np.zeros(self.n); self.t = 0.0
        self.inspect()

    def inspect(self):
        x = np.maximum(self.x, 1e-12)
        self.z = 1.0 / self.rng.gamma(self.gn + 2.0, 1.0 / ((self.gn + 1.0) * x))

    def posterior_mean(self):
        shape, rate = posterior_params(self.z, self.v, self.eta, self.theta, self.gn)
        return shape / rate

    def estimate(self, info):
        return {"pi": self.x, "ii": self.posterior_mean(), "rz": self.z}[info].copy()

    def eff_mu(self, a):
        imperfect = (a >= 1) & (a <= 3)
        return np.where(imperfect, 1.0 - (1.0 - self.mu[a]) * self.delta ** self.m, self.mu[a])

    def schedule(self, level_after):
        r = np.max(level_after / self.H)
        return max(self.dt_min, self.dt_max * max(0.0, 1.0 - r) ** self.gs)

    def step(self, a, dt, inspected=True):
        cost = (self.c_I if inspected else 0.0) + self.c_act[self.idx, a].sum()
        m = self.eff_mu(a)
        self.x = m * self.x + self.rng.gamma(self.eta * dt, self.theta)
        self.v = m * self.v + dt
        self.m = np.where(a == 4, 0.0, self.m + ((a >= 1) & (a <= 3)))
        failed = self.x >= self.H
        nf = int(failed.sum())
        if nf:
            cost += nf * self.c_D + self.c_act[failed, 4].sum()
            self.x[failed] = 0.0; self.v[failed] = 0.0; self.m[failed] = 0.0
        self.t += dt
        done = self.t >= self.T - 1e-9
        if not done and inspected:
            self.inspect()
        return cost, nf, done, failed


def posterior_params(z, v, eta, theta, gn):
    """Gamma posterior (shape, rate) of the true degradation given reading z and virtual age v."""
    shape = eta * v + gn + 2.0
    rate = (z + (gn + 1.0) * theta) / (z * theta)
    return shape, rate


def failure_probability(shape, rate, mu_eff, eta, theta, H, dt, n_samples=20_000, seed=0):
    """P(failure before the next inspection), averaging over the posterior of the current degradation."""
    rng = np.random.default_rng(seed)
    x = rng.gamma(shape, 1.0 / rate, n_samples)
    return float(np.mean(stats.gamma.sf(H - mu_eff * x, a=eta * dt, scale=theta) * (mu_eff * x < H)
                         + (mu_eff * x >= H)))


# ── Policies ─────────────────────────────────────────────────────────────────
def threshold_policy(frac):
    return lambda env, est: np.where(est >= frac * env.H, 4, 0)


def age_policy(tau):
    return lambda env, est: np.where(env.v >= tau, 4, 0)


def repair_replace_policy(f_r, f, max_rep):
    def pol(env, est):
        replace = (est >= f * env.H) | ((est >= f_r * env.H) & (env.m >= max_rep))
        return np.where(replace, 4, np.where(est >= f_r * env.H, 1, 0))
    return pol


class NumpyBranchingQ:
    """Forward pass of the branching dueling Q-network trained in the paper."""

    def __init__(self, weights, prefix):
        g = lambda k: weights[f"{prefix}/{k}"]
        self.layers = {k: (g(f"{k}.weight"), g(f"{k}.bias"))
                       for k in ["torso.0", "torso.2", "value.0", "value.2", "adv.0", "adv.2"]}
        self.n = self.layers["adv.2"][0].shape[0] // 5

    def _lin(self, k, x):
        w, b = self.layers[k]
        return x @ w.T + b

    def act(self, s):
        h = np.maximum(self._lin("torso.2", np.maximum(self._lin("torso.0", s), 0)), 0)
        v = self._lin("value.2", np.maximum(self._lin("value.0", h), 0))
        a = self._lin("adv.2", np.maximum(self._lin("adv.0", h), 0)).reshape(self.n, 5)
        return np.argmax(v + a - a.mean(-1, keepdims=True), axis=-1)


def dqn_features(env, info):
    est = env.estimate(info)
    prior = env.eta * env.v * env.theta / env.H
    return np.concatenate([est / env.H, prior, env.m / 5.0]).astype(np.float32), est


def load_paper_results():
    with open(DATA / "paper_results.json") as fh:
        return json.load(fh)


def load_dqn_weights():
    return dict(np.load(DATA / "dqn_weights.npz"))


def paper_policies(system_key, weights, results):
    """Policies with the parameters tuned in the paper: name -> (act_fn(env, est), info, sched, dt_fixed, inspect)."""
    p = lambda k: results[f"{system_key}_{k}_params"]
    fs_dt = results[f"{system_key}_fs_dt"]
    pol = {
        "STP (static threshold, raw reading)": (threshold_policy(p("STP")[0]), "rz", "fs", p("STP")[1], True),
        "ABR (age-based replacement)": (age_policy(p("ABR")[0]), "pi", "fs", 1.0, False),
        "CBT-FS (Bayesian threshold, fixed interval)": (threshold_policy(p("CBT-FS")[0]), "ii", "fs", p("CBT-FS")[1], True),
        "CBT-DS (Bayesian threshold, dynamic interval)": (threshold_policy(p("CBT-DS")[0]), "ii", "ds", None, True),
        "CBT-R-DS (Bayesian repair/replace, dynamic interval)": (repair_replace_policy(*p("CBT-R-DS")), "ii", "ds", None, True),
    }
    for name, info, sched in [("DQN-II-DS", "ii", "ds"), ("DQN-II-FS", "ii", "fs"),
                              ("DQN-PI-DS (true state, not implementable)", "pi", "ds")]:
        q = NumpyBranchingQ(weights, f"{system_key}_{info}_{sched}")
        pol[name] = ((lambda q, info: lambda env, est: q.act(dqn_features(env, info)[0]))(q, info),
                     info, sched, fs_dt if sched == "fs" else None, True)
    return pol


def run_episode(P, act_fn, info, sched, dt_fixed=None, seed=0, inspect=True, log=False):
    env = MaintEnv(**P); env.reset(seed)
    tot = 0.0; nf = 0; ni = 0; done = False; trace = []
    parts = np.zeros(3)
    while not done:
        est = env.estimate(info)
        a = np.asarray(act_fn(env, est))
        dt = dt_fixed if sched == "fs" else env.schedule(env.eff_mu(a) * est)
        if log:
            trace.append(dict(t=env.t, x=env.x.copy(), z=env.z.copy(), est=est.copy(), a=a.copy(), dt=dt))
        maint = env.c_act[env.idx, a].sum()
        c, f, done, failed = env.step(a, dt, inspected=inspect)
        insp = env.c_I if inspect else 0.0
        parts += [insp, maint, c - insp - maint]
        if log and f:
            trace[-1]["failed"] = failed.copy()
        tot += c; nf += f; ni += int(inspect)
    out = dict(cost=tot / env.T, failures=nf, inspections=ni,
               insp_cost=parts[0] / env.T, maint_cost=parts[1] / env.T, fail_cost=parts[2] / env.T)
    return (out, trace) if log else out
