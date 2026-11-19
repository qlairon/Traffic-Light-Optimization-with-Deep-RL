# 🚦 Traffic Light Optimization with Deep RL

A Deep Q-Network (DQN) agent that learns to control the traffic light at a single
four-way intersection. It comes with a live Streamlit dashboard for watching the
agent train.

## Features

- **Custom environment**: a 4-lane intersection (N, S, E, W) with stochastic
  arrivals, yellow transitions, and a minimum green time.
- **Four traffic patterns**: `uniform`, `rush_hour`, `heavy`, `imbalanced`.
- **DQN agent** in PyTorch with experience replay, a target network, Double-DQN
  targets, Huber loss, and gradient clipping.
- **Interactive dashboard**: animated intersection view, lane queues,
  reward/wait/loss curves, and live Q-values. You can tune hyperparameters from
  the sidebar.

## Quick start

Requires **Python 3.10+**.

```bash
git clone https://github.com/qlairon/Traffic-Light-Optimization-with-Deep-RL.git
cd Traffic-Light-Optimization-with-Deep-RL

python -m venv .venv
# Windows:      .\.venv\Scripts\Activate.ps1
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

On Windows you can run `.\scripts\setup_env.ps1` instead, which does the same thing.

> The default `torch` wheel is large. For a CPU-only install, see the
> [PyTorch install selector](https://pytorch.org/get-started/locally/).

### Train from the command line

```bash
python -m traffic_dqn                                # 5000 episodes, uniform traffic
python -m traffic_dqn --episodes 300 --mode rush_hour --seed 0
python -m traffic_dqn --eval-only                    # evaluate the bundled checkpoint
python -m traffic_dqn --help                         # all options
```

Training saves the model to `models/dqn_traffic.pth`, which overwrites the
bundled one unless you pass `--model`. It also prints evaluation stats and
writes `results/training_curves.png`. Add `--no-show` to skip opening the plot
window.

### Run the dashboard

```bash
streamlit run app.py
```

Pick a traffic mode and hyperparameters in the sidebar, then press **▶ Start**.
The dashboard trains a fresh agent in the browser session.

## How it works

| | |
|---|---|
| **State** (8 values, normalized) | queue length per lane ×4, current phase, time in phase, N/S and E/W arrival rates |
| **Actions** | 0 = NS green, 1 = NS yellow, 2 = EW green, 3 = EW yellow. Any action other than the current phase starts the next transition. |
| **Dynamics** | Phases cycle NS green → NS yellow → EW green → EW yellow. Yellow lasts one step. Green lasts at least 5 steps. A green lane discharges up to 2 vehicles per step. Each lane holds at most 12 vehicles. |
| **Reward** | `+2 × departed − 1 × total queue − 1 (during yellow)` |
| **Network** | MLP 8 → 64 → 64 → 4 (ReLU) |

## Project structure

```
├── traffic_dqn/            # Python package
│   ├── env.py              #   TrafficIntersection environment
│   ├── agent.py            #   QNetwork, ReplayBuffer, DQNAgent
│   ├── train.py            #   train / evaluate / plot + command-line interface
│   └── __main__.py         #   enables `python -m traffic_dqn`
├── models/
│   └── dqn_traffic.pth     # pre-trained checkpoint (uniform traffic)
├── scripts/
│   └── setup_env.ps1       # Windows setup helper
├── app.py                  # Streamlit dashboard
├── requirements.txt
└── LICENSE
```

Training plots go to `results/`, which git ignores.

## License

[MIT](LICENSE)
