"""
fyso_env.py — Flat Yard Switching Optimization environment
===========================================================
Implements the stub-yard simulator for Case 1 of the
2024 INFORMS RAS Problem Solving Competition.

Reference:
    Han et al. (2025) "Optimizing Railroad Flat Yard Switching with
    Reward-Driven Integer Programming." SSRN:6346981.

Case 1 configuration (Table 5):
    11 railcars · 4 blocks · 5 tracks
    Track lengths: [2000, 1900, 1800, 1700, 500] m
    Desired block order: [0, 2, 1, 3]
    All railcars: 15 m uniform length
    Yard type: stub (LIFO per track)

State:
    Tuple-of-tuples representation of all 5 tracks.
    Track 0 = lead (always empty of cargo cars).
    Index 0 in each track list = lead-side (accessible) end.

Action:
    (src_track, n_cars, dst_track)
    Pull n_cars from the lead end of src_track,
    push them to the lead end of dst_track.

Switching time formula (Appendix B of Han et al.):
    Total = ladder_time + pull_time + push_time
    where pull/push times depend on track length, wagon count,
    and acceleration/deceleration parameters.
"""

import collections
from typing import Optional

# ---------------------------------------------------------------------------
# Instance constants
# ---------------------------------------------------------------------------

CAR_LEN: float = 15.0                          # metres, uniform
TRACK_LENGTHS: list[float] = [2000, 1900, 1800, 1700, 500]  # metres, tracks 0-4
DESIRED_ORDER: list[int] = [0, 2, 1, 3]        # departure block order

# Initial layout — reproduced from paper Table 5 / Figure 1 for Case 1.
# Each sub-list is a track; index 0 = lead-accessible end.
# Track 0 = lead (always empty).
INITIAL_LAYOUT: list[list[int]] = [
    [],          # track 0: lead
    [2, 1, 0],   # track 1
    [0, 3, 1],   # track 2
    [2, 3],      # track 3
    [1, 3, 0],   # track 4
]                # total: 11 cars, blocks {0:3, 1:3, 2:2, 3:3}

# ---------------------------------------------------------------------------
# Switching time formula (Appendix B)
# ---------------------------------------------------------------------------

_S_MAX    = 16 / 3600      # km/s  (max shunter speed)
_ALPHA_A  = 6750           # acceleration constant
_BETA_A   = 292.5          # wagon-count coefficient for acceleration
_ALPHA_D  = 6750           # deceleration constant
_BETA_D   = 225            # wagon-count coefficient for deceleration
_LADDER_V = 10 / 3600      # km/s  (ladder traversal speed)
_SPACING  = 3.0            # metres between adjacent tracks on ladder


def switching_time(src: int, n: int, dst: int) -> float:
    """
    Total time (seconds) for one shunting action:
    pull *n* cars from *src* track, push to *dst* track.

    Parameters
    ----------
    src : source track index (1-4)
    n   : number of cars pulled
    dst : destination track index (1-4)
    """
    # Ladder traversal
    t_ladder = abs(src - dst) * _SPACING / 1000.0 / _LADDER_V

    # Acceleration/deceleration coefficients for n wagons
    a_n = 1.0 / (_ALPHA_A + _BETA_A * n)
    d_n = 1.0 / (_ALPHA_D + _BETA_D * n)

    # Pull: shunter enters src track and exits with n cars
    lk_src = TRACK_LENGTHS[src] / 1000.0
    t_pull = lk_src / _S_MAX + 0.5 * _S_MAX * (1 / a_n + 1 / d_n) * 2

    # Push: shunter enters dst track and exits without cars
    lk_dst = TRACK_LENGTHS[dst] / 1000.0
    t_push = lk_dst / _S_MAX + 0.5 * _S_MAX * (1 / a_n + 1 / d_n) * 2

    return t_ladder + t_pull + t_push


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class FlatYard:
    """
    Stub flat-yard simulator for Case 1.

    Tracks behave as LIFO stacks: only the car at index 0
    (closest to the lead) is accessible for pulling.

    Parameters
    ----------
    layout : optional initial track layout (list of lists).
              Defaults to INITIAL_LAYOUT.
    """

    def __init__(self, layout: Optional[list[list[int]]] = None) -> None:
        self.tracks = [list(t) for t in (layout or INITIAL_LAYOUT)]
        # Total car counts per block — fixed at construction, used for is_done.
        self._all_counts = collections.Counter(
            c for t in self.tracks for c in t
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def state(self) -> tuple:
        """Hashable representation of the current yard layout."""
        return tuple(tuple(t) for t in self.tracks)

    def n_cars(self) -> int:
        return sum(len(t) for t in self.tracks)

    def is_done(self) -> bool:
        """
        Return True iff all cars are assembled on a single track
        in the correct block-group order (DESIRED_ORDER).
        """
        n = self.n_cars()
        for t in self.tracks[1:]:
            if len(t) != n:
                continue
            if collections.Counter(t) != self._all_counts:
                continue
            # Check run-length encoding matches desired order
            blocks_seen: list[int] = []
            for c in t:
                if not blocks_seen or blocks_seen[-1] != c:
                    blocks_seen.append(c)
            if blocks_seen == DESIRED_ORDER:
                return True
        return False

    def valid_actions(self) -> list[tuple[int, int, int]]:
        """
        Enumerate all feasible (src, n, dst) triples given current layout
        and track-length constraints.
        """
        actions = []
        for src in range(1, 5):
            if not self.tracks[src]:
                continue
            for n in range(1, len(self.tracks[src]) + 1):
                for dst in range(1, 5):
                    if dst == src:
                        continue
                    # Track-length feasibility check
                    if (len(self.tracks[dst]) + n) * CAR_LEN > TRACK_LENGTHS[dst]:
                        continue
                    actions.append((src, n, dst))
        return actions

    def step(self, action: tuple[int, int, int]) -> tuple[tuple, float, bool]:
        """
        Execute *action* = (src, n, dst).

        Returns
        -------
        state   : new yard state (hashable)
        reward  : negative switching time (minimising time ≡ maximising reward)
        done    : whether the terminal assembly is reached
        """
        src, n, dst = action
        pulled = self.tracks[src][:n]
        self.tracks[src] = self.tracks[src][n:]
        self.tracks[dst] = pulled + self.tracks[dst]
        reward = -switching_time(src, n, dst)
        done = self.is_done()
        return self.state(), reward, done

    # ------------------------------------------------------------------
    # Cost-to-go proxy (Section 6.3.2 of Han et al.)
    # ------------------------------------------------------------------

    def deferred_links(self) -> int:
        """
        Compute the deferred link count Λ — the number of ordered pairs
        (i, j) of same-block cars that are NOT adjacent on the same track.
        Used as a cost-to-go approximation in the BB&R tree search and
        as a reward-shaping signal in the RL agent.

        Λ = 0 iff all same-block cars are already contiguous on one track
        (a necessary, though not sufficient, condition for the terminal state).
        """
        lam = 0
        all_cars = [
            (ti, pi, c)
            for ti, t in enumerate(self.tracks)
            for pi, c in enumerate(t)
        ]
        for i, (ti, pi, ci) in enumerate(all_cars):
            for j, (tj, pj, cj) in enumerate(all_cars):
                if i >= j or ci != cj:
                    continue
                if ti != tj or abs(pi - pj) > 1:
                    lam += 2   # directed: count both (i→j) and (j→i)
        return lam

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def copy(self) -> "FlatYard":
        return FlatYard(self.tracks)

    def render(self) -> str:
        """Human-readable track layout."""
        lines = []
        for i, t in enumerate(self.tracks):
            label = "lead  " if i == 0 else f"track {i}"
            cars = " ".join(f"[{c}]" for c in t) if t else "(empty)"
            lines.append(f"  {label}: {cars}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"FlatYard(Λ={self.deferred_links()}, done={self.is_done()})\n{self.render()}"
