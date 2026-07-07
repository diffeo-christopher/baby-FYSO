# baby-FYSO

Let's learn about railroad optimization! This is a minimal exploration of the **Flat Yard Switching Optimization (FYSO)** problem from the 2024 INFORMS Railway Applications Section Problem Solving Competition.

Everything here is extremely naive. I'm still learning!

This repo contains:
- A Python simulator + tabular Q-learning agents for Case 1
- A Julia/JuMP scaffold of the Shunting Action Network (SAN) integer programming model

It accompanies an ongoing study of Han et al. (2025) while getting up to speed with some operations research tooling from scratch.

---

## The problem

A flat yard is a railroad classification facility where a shunter (yard locomotive) sorts scattered railcars into a single outbound train in the correct block order. Unlike a hump yard, there's no gravity assist — every move is a manual pull-and-push. The yard's tracks behave as LIFO stacks, making this structurally equivalent to multi-stack sorting (NP-hard for ≥ 4 stacks).

**Case 1:** 11 railcars · 4 blocks · 5 tracks · desired order [0, 2, 1, 3]

**Objective:** minimise total switching time (seconds), which depends on track length, number of wagons, and distance between tracks on the ladder.

**Paper baseline (SAN-R greedy MIP):** 5931.84 s  
**Paper tree search optimum:** 4426.18 s (−25%)

---

## Repository layout

```
baby-FYSO/
├── python/
│   ├── fyso_env.py          # Yard simulator, time formula, deferred-link proxy
│   ├── fyso_agent.py        # Tabular Q-learning + Λ-shaped variant
│   └── run_experiment.py    # Comparison experiment (Λ-greedy vs Q-learning)
├── julia/
│   ├── fyso_san.jl          # SAN 0-1 IP model in JuMP (one stage)
│   └── Project.toml         # Julia dependency manifest
├── requirements.txt
└── README.md
```

---

## Quickstart

### Python

```bash
pip install -r requirements.txt
cd python
python run_experiment.py                    # default: 20,000 episodes
python run_experiment.py --episodes 50000   # longer training
```

### Julia

```julia
# From the julia/ directory
using Pkg
Pkg.activate(".")
Pkg.instantiate()       # installs JuMP and HiGHS

include("fyso_san.jl")
case1_example()         # runs the SAN model on Figure 1 of the paper
```

---

## What the Python RL experiment shows

Three policies are compared:

| Method | Notes |
|---|---|
| **Λ-greedy** | Greedy on deferred-link reduction. Approximates the paper's SAN-R without a MIP solver. Hits oscillation cycles the paper avoids via global MIP consistency. |
| **Plain Q-learning** | Reward = −switching_time. Q-table too sparse for this state space; rarely reaches the terminal. |
| **Shaped Q-learning** | Adds Λ-reduction bonus as a dense signal. Converges faster but still undershoots the paper's SAN-R. |

The honest takeaway: beating SAN-R without the MIP solver is hard precisely because the MIP enforces global consistency across all arc decisions simultaneously. A natural extension (suggested in §8 of the paper) is a **DQN with Λ and block-composition features**, trained on randomised initial layouts.

---

## Julia scaffold status

`fyso_san.jl` implements one SAN stage (Model 1–10 from the paper, minus the lazy track-length constraints of Section 4.2). The full sequential loop (Algorithm 1) and the branch-and-bound tree (Section 6) are future work.

---

## Reference

Han, P., Meng, L., Luan, X., Bešinović, N., Miao, J., Liao, Z., & Zheng, R. (2025).  
*Optimizing Railroad Flat Yard Switching with Reward-Driven Integer Programming: A Sequential Decision Framework with Branch-and-Bound and Remember Algorithm.*  
SSRN preprint. https://ssrn.com/abstract=6346981

Competition instances: https://github.com/MarcMeketonVillanova/2024_RAS_PSC
