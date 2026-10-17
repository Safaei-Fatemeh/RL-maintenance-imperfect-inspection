# RL Maintenance under Imperfect Inspections

Interactive companion app for the paper **"Maintenance under Imperfect Inspections Using Deep Reinforcement Learning, with a Web-Based Decision Tool"**.

**Live app:** https://rl-maintenance.streamlit.app/

The app lets you work directly with the model of the paper: estimate the true condition of a component from a noisy inspection reading, see the failure risk of each maintenance action, simulate a multi-component system over time, and compare maintenance policies, including the deep Q-network agents trained in the paper.

---

## The model in brief

- **Degradation.** Each component degrades according to a gamma process and fails when its degradation reaches a threshold. A failure stops the system and is followed by a corrective replacement; in the model, the replaced component re-enters the system as new at the next inspection.
- **Imperfect inspections.** An inspection returns a reading that is unbiased but whose error grows with the true degradation. Combined with the component's *virtual age* (time since renewal, reduced by repairs), the reading gives a closed-form gamma posterior for the true degradation.
- **Maintenance actions.** Five actions per component: do nothing, imperfect repair, repair, imperfect replacement and replacement. Repeated imperfect actions become less effective.
- **Adaptive inspection intervals.** All components are inspected together, and the next inspection is brought forward as the most degraded component approaches its threshold.
- **Policies.** Tuned heuristic rules (static threshold, age-based replacement, Bayesian control limits, a Bayesian repair/replace rule) and branching deep Q-networks trained with imperfect information, with the raw reading, with a fixed interval and with the true state.

## What you can do in the app

| Page | Description |
|---|---|
| **Overview** | Who the app is for, the main results and a short description of the model. |
| **Inspection calculator** | Enter a component's parameters, history and latest reading. Get the posterior estimate and credible interval, the failure risk and next inspection interval for each action, and the recommended action. |
| **Simulate one history** | Choose a system and a policy and follow it over time: true degradation, readings, estimates, maintenance actions, failures and inspection intervals. |
| **Compare policies** | Monte Carlo comparison of any set of policies on common simulated histories, with a breakdown into inspection, maintenance and failure costs. Costs, inspection quality and repair effectiveness can be changed. |
| **Paper results** | The main tables and sensitivity results reported in the paper. |

All costs are in normalised cost units per unit time.

## Run it on your computer

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Safaei-Fatemeh/RL-maintenance-imperfect-inspection.git
cd RL-maintenance-imperfect-inspection
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Repository structure

```
app.py                    Streamlit interface (pages, charts, layout)
model.py                  Simulation model: degradation, inspections, Bayesian update, policies
requirements.txt          Python dependencies
.streamlit/config.toml    Colour theme
data/
  dqn_weights.npz         Weights of the deep Q-networks trained in the paper
  paper_results.json      Results reported in the paper (costs, sensitivity analyses)
  logo.svg, logo_icon.svg App logo
```

The trained networks are stored as NumPy arrays and evaluated with NumPy, so the app does not need PyTorch. The networks were trained in the paper's full experiment code; the app reproduces their reported costs up to Monte Carlo error.

## Notes and limitations

- The DQN agents were trained for the parameter settings of the paper (the three-component numerical study and the conveyor-belt case study). On the *Compare policies* page you can change costs and inspection quality, but the agents and the tuned heuristic thresholds are not re-trained or re-tuned for the new values.
- The case-study parameters are illustrative and were not estimated from plant data.
- The app is intended for research and teaching. Decisions on real equipment should be based on a model calibrated to that equipment.

## Citation

The paper *Maintenance under Imperfect Inspections Using Deep Reinforcement Learning, with a Web-Based Decision Tool* is
currently under review. Citation details will be added here after publication.

## Contact

For questions or suggestions, please open an issue in this repository.
