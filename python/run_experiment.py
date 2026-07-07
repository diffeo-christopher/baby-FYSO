"""
run_experiment.py — FYSO RL experiment, Case 1
================================================
Compares three approaches on the 2024 INFORMS RAS Case 1 instance:

  1. Λ-greedy baseline   Greedy policy maximising deferred-link reduction.
                         Approximates the paper's SAN-R without a MIP solver.

  2. Plain Q-learning    Tabular Q-learning, reward = −switching_time.

  3. Shaped Q-learning   Same, plus Λ-based reward shaping for a denser
                         training signal.

Paper benchmarks (Han et al. 2025, Table 7):
  SAN-R (greedy MIP):    5931.84 s
  Tree search (optimal): 4426.18 s  (−25.4%)

Usage
-----
    python run_experiment.py              # default settings
    python run_experiment.py --episodes 30000 --seed 0
"""

import argparse
import collections
import random
import time

import numpy as np

from fyso_env import FlatYard, switching_time, DESIRED_ORDER
from fyso_agent import QLearningAgent, ShapedQLearningAgent


# ---------------------------------------------------------------------------
# Λ-greedy baseline
# ---------------------------------------------------------------------------

def lambda_greedy_episode(max_steps: int = 60) -> tuple[float, bool, list]:
    """
    Single episode of the Λ-greedy policy.
    At each step, pick the action that most reduces deferred links,
    breaking ties by minimum switching time.
    Includes cycle detection to prevent infinite oscillation.
    """
    yard = FlatYard()
    total_time = 0.0
    path: list = []
    visited: set = set()

    for _ in range(max_steps):
        if yard.is_done():
            break
        actions = yard.valid_actions()
        if not actions:
            break

        best_gain = -999
        best_cost = float("inf")
        best_action = None

        for a in actions:
            y2 = yard.copy()
            y2.step(a)
            gain = yard.deferred_links() - y2.deferred_links()
            cost = switching_time(*a)
            if gain > best_gain or (gain == best_gain and cost < best_cost):
                best_gain, best_cost, best_action = gain, cost, a

        total_time += switching_time(*best_action)
        path.append(best_action)
        _, _, _ = yard.step(best_action)

        state = yard.state()
        if state in visited:
            break  # cycle — stop rather than oscillate
        visited.add(state)

    return total_time, yard.is_done(), path


def run_lambda_greedy(n_runs: int = 100, verbose: bool = True) -> list[float]:
    times = [t for _ in range(n_runs) for t, s, _ in [lambda_greedy_episode()] if s]
    if verbose:
        if times:
            print(f"  Λ-greedy: {len(times)}/{n_runs} solved")
            print(f"    mean = {np.mean(times):.1f}s   best = {min(times):.1f}s")
        else:
            print(f"  Λ-greedy: 0/{n_runs} solved (all hit cycles)")
            t0, _, p0 = lambda_greedy_episode()
            print(f"    Sample trace ({len(p0)} steps, {t0:.1f}s, unsolved):")
            yard = FlatYard()
            for i, a in enumerate(p0):
                src, n, dst = a
                t = switching_time(src, n, dst)
                yard.step(a)
                print(f"      {i+1:2d}: T{src}→T{dst} n={n}  "
                      f"t={t:.0f}s  Λ={yard.deferred_links()}")
    return times


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main(n_episodes: int = 20_000, seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 65)
    print("baby-FYSO  ·  Q-Learning Experiment  ·  Case 1")
    print("11 railcars · 4 blocks · 5 tracks · stub yard")
    print("=" * 65)
    print()

    # ── Paper benchmarks ──
    print("Paper benchmarks (Han et al. 2025):")
    print("  SAN-R greedy (MIP):    5931.84 s  ← our comparison target")
    print("  Tree search (optimal): 4426.18 s  (−25.4%)")
    print()

    # ── Λ-greedy baseline ──
    print("── 1. Λ-greedy baseline ──────────────────────────────────")
    greedy_times = run_lambda_greedy(n_runs=100, verbose=True)
    print()

    # ── Plain Q-learning ──
    print("── 2. Plain Q-learning  ──────────────────────────────────")
    agent_plain = QLearningAgent(
        alpha=0.25, gamma=0.97,
        epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9996,
    )
    print(f"  Training for {n_episodes:,} episodes...")
    _, best_plain = agent_plain.train(
        n_episodes=n_episodes, max_steps=50, verbose_every=n_episodes // 4
    )
    eval_plain = agent_plain.evaluate(n_episodes=500)
    print(f"  Greedy eval (500 ep): {len(eval_plain)}/500 solved")
    if eval_plain:
        print(f"    mean = {np.mean(eval_plain):.1f}s   best = {min(eval_plain):.1f}s")
    print()

    # ── Shaped Q-learning ──
    print("── 3. Shaped Q-learning  (Λ reward shaping) ─────────────")
    agent_shaped = ShapedQLearningAgent(
        shaping_weight=200.0,
        alpha=0.25, gamma=0.97,
        epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9996,
    )
    print(f"  Training for {n_episodes:,} episodes...")
    ep_times_shaped, best_shaped = agent_shaped.train(
        n_episodes=n_episodes, max_steps=50, verbose_every=n_episodes // 4
    )
    eval_shaped = agent_shaped.evaluate(n_episodes=500)
    print(f"  Greedy eval (500 ep): {len(eval_shaped)}/500 solved")
    if eval_shaped:
        print(f"    mean = {np.mean(eval_shaped):.1f}s   best = {min(eval_shaped):.1f}s")
    print()

    # ── Learning curve ──
    if ep_times_shaped:
        print("── Shaped Q-learning curve ───────────────────────────────")
        chunk = n_episodes // 4
        for i in range(4):
            window = ep_times_shaped[i * chunk : (i + 1) * chunk]
            if window:
                print(
                    f"  ep {i*chunk+1:6d}–{(i+1)*chunk:6d}: "
                    f"mean={np.mean(window):.0f}s  "
                    f"best={min(window):.0f}s  "
                    f"solved={len(window)}"
                )
        print()

    # ── Summary table ──
    paper_san_r = 5931.84
    paper_optimal = 4426.18
    base = np.mean(greedy_times) if greedy_times else float("nan")

    print("=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    rows = [
        ("Paper SAN-R (greedy MIP)",  paper_san_r,  None),
        ("Paper tree search (optimal)", paper_optimal,
         f"−{100*(paper_san_r-paper_optimal)/paper_san_r:.1f}%"),
        ("Our Λ-greedy (mean)",        base,
         f"{100*(paper_san_r-base)/paper_san_r:+.1f}% vs paper" if base == base else "N/A"),
    ]
    if eval_plain:
        m = np.mean(eval_plain)
        rows.append(("Plain Q-learn (mean)",    m,
                     f"{100*(paper_san_r-m)/paper_san_r:+.1f}% vs paper"))
        rows.append(("Plain Q-learn (best)",    min(eval_plain), None))
    if eval_shaped:
        m = np.mean(eval_shaped)
        rows.append(("Shaped Q-learn (mean)",   m,
                     f"{100*(paper_san_r-m)/paper_san_r:+.1f}% vs paper"))
        rows.append(("Shaped Q-learn (best)",   min(eval_shaped), None))

    for label, value, note in rows:
        val_str = f"{value:.1f}s" if value == value else "N/A"
        note_str = f"  ({note})" if note else ""
        print(f"  {label:<30s} {val_str:>10s}{note_str}")

    # ── Honest interpretation ──
    print()
    print("Notes:")
    print("  * The Λ-greedy baseline hits oscillation cycles that the paper's")
    print("    SAN-R avoids via the MIP solver's global consistency enforcement.")
    print("  * Tabular Q-learning struggles here because the state space is too")
    print("    large for Q-values to propagate reliably from terminal states.")
    print("  * The natural next step is a DQN with Λ + block-composition features,")
    print("    trained on randomised initial layouts — as suggested in Han et al. §8.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FYSO RL experiment — Case 1")
    parser.add_argument("--episodes", type=int, default=20_000,
                        help="number of training episodes (default: 20000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    args = parser.parse_args()
    main(n_episodes=args.episodes, seed=args.seed)
