"""
Training, evaluation and plotting CLI.

Usage:
    python -m traffic_dqn                          # train with defaults
    python -m traffic_dqn --episodes 300 --mode rush_hour
    python -m traffic_dqn --eval-only              # evaluate models/dqn_traffic.pth

Run `python -m traffic_dqn --help` for all options.
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .agent import DQNAgent
from .env import TrafficIntersection

DEFAULT_MODEL = "models/dqn_traffic.pth"
DEFAULT_PLOT  = "results/training_curves.png"


# ─────────────────────────────────────────────
#  5. TRAINING LOOP
# ─────────────────────────────────────────────

def train(
    n_episodes:    int   = 5000,
    steps_per_ep:  int   = 500,
    traffic_mode:  str   = "uniform",
    render_every:  int   = 50,
    save_path:     str   = DEFAULT_MODEL,
) -> tuple[DQNAgent, dict]:
    """
    Full training loop.

    Returns the trained agent and a history dict with
    reward, wait time, and loss curves.
    """
    env   = TrafficIntersection(traffic_mode=traffic_mode)
    agent = DQNAgent()

    history = {
        "rewards":   [],
        "avg_wait":  [],
        "losses":    [],
    }

    print(f"Training DQN — {n_episodes} episodes × {steps_per_ep} steps  "
          f"[traffic: {traffic_mode}]")
    print(f"Device: {agent.device}\n")
    print(f"{'Episode':>8} {'Reward':>10} {'AvgWait':>10} {'ε':>7} {'Loss':>10}")
    print("─" * 52)

    for ep in range(1, n_episodes + 1):
        state        = env.reset()
        ep_reward    = 0.0
        ep_wait      = 0.0
        ep_loss      = []

        for _ in range(steps_per_ep):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store(state, action, reward, next_state, float(done))
            loss = agent.learn()
            if loss is not None:
                ep_loss.append(loss)

            ep_reward += reward
            ep_wait   += info["total_queue"] / 4.0   # mean queue ≈ avg wait proxy
            state      = next_state

        avg_wait = ep_wait / steps_per_ep
        avg_loss = np.mean(ep_loss) if ep_loss else 0.0

        history["rewards"].append(ep_reward)
        history["avg_wait"].append(avg_wait)
        history["losses"].append(avg_loss)

        if ep % render_every == 0 or ep == 1:
            avg_r = np.mean(history["rewards"][-20:])
            print(f"{ep:>8} {ep_reward:>10.1f} {avg_wait:>10.2f} "
                  f"{agent.epsilon:>7.3f} {avg_loss:>10.4f}  "
                  f"(20-ep avg reward: {avg_r:.1f})")

    agent.save(save_path)
    return agent, history


# ─────────────────────────────────────────────
#  6. EVALUATION
# ─────────────────────────────────────────────

def evaluate(
    agent:         DQNAgent,
    traffic_mode:  str = "uniform",
    n_episodes:    int = 20,
    steps_per_ep:  int = 500,
) -> dict:
    """Run the trained agent (greedy policy) and collect stats."""
    env = TrafficIntersection(traffic_mode=traffic_mode)
    old_eps = agent.epsilon
    agent.epsilon = 0.0   # greedy

    total_departed = 0
    total_wait     = 0.0
    total_reward   = 0.0

    for _ in range(n_episodes):
        state = env.reset()
        for _ in range(steps_per_ep):
            action = agent.select_action(state)
            state, reward, _, info = env.step(action)
            total_departed += info["departed"]
            total_wait     += info["total_queue"] / 4.0
            total_reward   += reward

    agent.epsilon = old_eps
    n = n_episodes * steps_per_ep
    stats = {
        "avg_reward":    total_reward   / n_episodes,
        "avg_wait_time": total_wait     / n,
        "avg_throughput": total_departed / n,
    }
    print("\n── Evaluation results ──────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<22}: {v:.4f}")
    return stats


# ─────────────────────────────────────────────
#  7. PLOTTING
# ─────────────────────────────────────────────

def plot_training(history: dict, save_path: str = DEFAULT_PLOT, show: bool = True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("DQN Traffic Light Training", fontsize=14, fontweight="bold")

    rewards = history["rewards"]
    window  = 20
    avg_r   = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]

    axes[0].plot(rewards, alpha=0.4, color="#378ADD", label="Episode reward")
    axes[0].plot(avg_r,   color="#1D9E75", linewidth=2, label=f"{window}-ep moving avg")
    axes[0].set_title("Reward")
    axes[0].set_xlabel("Episode")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["avg_wait"], color="#D85A30", linewidth=1.5)
    axes[1].set_title("Avg Queue Length (wait proxy)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Vehicles waiting (mean)")
    axes[1].grid(alpha=0.3)

    losses = [l for l in history["losses"] if l > 0]
    if losses:
        axes[2].plot(losses, color="#BA7517", linewidth=1, alpha=0.7)
        axes[2].set_title("Huber Loss")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Loss")
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────
#  8. ENTRY POINT
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train or evaluate a DQN traffic-light controller.")
    p.add_argument("--mode", default="uniform", choices=list(TrafficIntersection.TRAFFIC_MODES),
                   help="traffic arrival pattern (default: uniform)")
    p.add_argument("--episodes", type=int, default=5000, help="training episodes (default: 5000)")
    p.add_argument("--steps", type=int, default=500, help="steps per episode (default: 500)")
    p.add_argument("--log-every", type=int, default=50, help="print progress every N episodes")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"checkpoint path to save/load (default: {DEFAULT_MODEL})")
    p.add_argument("--plot", default=DEFAULT_PLOT,
                   help=f"where to save training curves (default: {DEFAULT_PLOT})")
    p.add_argument("--eval-only", action="store_true",
                   help="skip training; load --model and evaluate it")
    p.add_argument("--eval-episodes", type=int, default=20, help="evaluation episodes (default: 20)")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    p.add_argument("--no-show", action="store_true",
                   help="save the training plot without opening a window")
    return p.parse_args()


def main() -> None:
    # Avoid UnicodeEncodeError (ε, →) on consoles/pipes with legacy encodings.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.eval_only:
        agent = DQNAgent()
        agent.load(args.model)
        evaluate(agent, traffic_mode=args.mode,
                 n_episodes=args.eval_episodes, steps_per_ep=args.steps)
    else:
        agent, history = train(
            n_episodes   = args.episodes,
            steps_per_ep = args.steps,
            traffic_mode = args.mode,
            render_every = args.log_every,
            save_path    = args.model,
        )
        evaluate(agent, traffic_mode=args.mode,
                 n_episodes=args.eval_episodes, steps_per_ep=args.steps)
        plot_training(history, save_path=args.plot, show=not args.no_show)


if __name__ == "__main__":
    main()
