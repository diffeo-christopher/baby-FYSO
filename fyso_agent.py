"""
fyso_agent.py — Q-learning agents for FYSO
============================================
Two tabular Q-learning variants:

  QLearningAgent      Plain negative-time reward.
  ShapedQLearningAgent  Adds deferred-link (Λ) reward shaping
                        as a dense cost-to-go signal.

Both agents use epsilon-greedy exploration and a defaultdict Q-table
keyed by (state_hash, action_index), where actions are sorted
deterministically at each step for stable indexing.

Design notes
------------
* Tabular Q-learning is appropriate for Case 1 (11 cars) but will
  not scale to Cases 2/3 without function approximation (DQN).
* The shaped agent converges faster because Λ decreases monotonically
  toward the terminal state in expectation, providing a dense signal
  where the time reward is very sparse.
* The cycle-breaking heuristic (revisit detection within 3 steps)
  prevents the agent from getting trapped in oscillations — a failure
  mode the paper's SAN-R greedy also exhibits without the MIP solver's
  global consistency enforcement.

Reference:
    Han et al. (2025) SSRN:6346981, Section 8 (future work).
"""

import collections
import random
import time
from typing import Optional

import numpy as np

from fyso_env import FlatYard, switching_time


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Tabular Q-learning for FYSO.

    Parameters
    ----------
    alpha       : learning rate
    gamma       : discount factor
    epsilon     : initial exploration rate (annealed during training)
    epsilon_min : floor for epsilon
    epsilon_decay : multiplicative decay applied after each episode
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 0.97,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9996,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q: dict = collections.defaultdict(float)

    # ------------------------------------------------------------------
    # Core RL methods
    # ------------------------------------------------------------------

    def select_action(
        self, state: tuple, actions: list
    ) -> tuple[int, tuple]:
        """Epsilon-greedy action selection."""
        if not actions:
            raise ValueError("No valid actions available.")
        if random.random() < self.epsilon:
            idx = random.randrange(len(actions))
        else:
            q_vals = [self.Q[(state, i)] for i in range(len(actions))]
            idx = int(np.argmax(q_vals))
        return idx, actions[idx]

    def update(
        self,
        state: tuple,
        action_idx: int,
        reward: float,
        next_state: tuple,
        next_actions: list,
        done: bool,
    ) -> None:
        """Standard Q-learning (off-policy TD) update."""
        current_q = self.Q[(state, action_idx)]
        if done or not next_actions:
            target = reward
        else:
            best_next = max(
                self.Q[(next_state, j)] for j in range(len(next_actions))
            )
            target = reward + self.gamma * best_next
        self.Q[(state, action_idx)] += self.alpha * (target - current_q)

    def _shaped_reward(
        self, base_reward: float, lam_before: int, lam_after: int
    ) -> float:
        """Override in subclass to add reward shaping."""
        return base_reward

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        n_episodes: int = 20_000,
        max_steps: int = 50,
        verbose_every: int = 5_000,
    ) -> tuple[list[float], float]:
        """
        Train the agent.

        Returns
        -------
        episode_times : switching times for completed episodes (inf if not solved)
        best_time     : minimum switching time seen during training
        """
        best_time = float("inf")
        episode_times: list[float] = []
        recent: collections.deque = collections.deque(maxlen=1_000)
        t0 = time.time()

        for ep in range(n_episodes):
            yard = FlatYard()
            state = yard.state()
            ep_time = 0.0
            visited: dict = {}   # state → step index, for cycle detection

            for step in range(max_steps):
                actions = sorted(yard.valid_actions())
                if not actions:
                    break

                lam_before = yard.deferred_links()
                action_idx, action = self.select_action(state, actions)
                src, n, dst = action
                ep_time += switching_time(src, n, dst)

                next_state, base_reward, done = yard.step(action)
                lam_after = yard.deferred_links()

                reward = self._shaped_reward(base_reward, lam_before, lam_after)
                next_actions = sorted(yard.valid_actions()) if not done else []
                self.update(state, action_idx, reward, next_state, next_actions, done)
                state = next_state

                if done:
                    break
                # Break out of short-cycle oscillations
                if next_state in visited and step - visited[next_state] < 3:
                    break
                visited[next_state] = step

            if yard.is_done():
                episode_times.append(ep_time)
                recent.append(ep_time)
                if ep_time < best_time:
                    best_time = ep_time
            else:
                recent.append(float("inf"))

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if verbose_every and (ep + 1) % verbose_every == 0:
                solved = [x for x in recent if x < float("inf")]
                avg = f"{np.mean(solved):.0f}s" if solved else "N/A"
                pct = f"{100 * len(solved) / len(recent):.0f}%"
                print(
                    f"  ep {ep+1:6d}/{n_episodes}"
                    f"  ε={self.epsilon:.3f}"
                    f"  best={best_time:.0f}s"
                    f"  avg={avg}"
                    f"  completion={pct}"
                    f"  |Q|={len(self.Q)}"
                    f"  wall={time.time()-t0:.0f}s"
                )

        return episode_times, best_time

    # ------------------------------------------------------------------
    # Evaluation (greedy, no exploration)
    # ------------------------------------------------------------------

    def evaluate(
        self, n_episodes: int = 500, max_steps: int = 50
    ) -> list[float]:
        """
        Run greedy evaluation (epsilon=0) and return switching times
        for completed episodes.
        """
        saved_eps = self.epsilon
        self.epsilon = 0.0
        times: list[float] = []

        for _ in range(n_episodes):
            yard = FlatYard()
            state = yard.state()
            ep_time = 0.0
            seen: set = set()

            for _ in range(max_steps):
                actions = sorted(yard.valid_actions())
                if not actions:
                    break
                _, action = self.select_action(state, actions)
                src, n, dst = action
                ep_time += switching_time(src, n, dst)
                yard.step(action)
                state = yard.state()
                if yard.is_done():
                    break
                if state in seen:
                    break
                seen.add(state)

            if yard.is_done():
                times.append(ep_time)

        self.epsilon = saved_eps
        return times


# ---------------------------------------------------------------------------
# Shaped agent (deferred-link reward shaping)
# ---------------------------------------------------------------------------

class ShapedQLearningAgent(QLearningAgent):
    """
    Q-learning with potential-based reward shaping using the deferred
    link count Λ from Han et al. Section 6.3.2.

    Shaped reward = base_reward + shaping_weight * (Λ_before - Λ_after)

    A decrease in Λ (same-block cars moving closer together) earns a
    bonus; an increase earns a penalty.  This provides a dense training
    signal grounded in the paper's own cost-to-go approximation.

    Parameters
    ----------
    shaping_weight : scale factor balancing time penalty against Λ signal.
                     Tune relative to typical switching_time values (~1000s).
    """

    def __init__(self, shaping_weight: float = 200.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.shaping_weight = shaping_weight

    def _shaped_reward(
        self, base_reward: float, lam_before: int, lam_after: int
    ) -> float:
        return base_reward + self.shaping_weight * (lam_before - lam_after)
