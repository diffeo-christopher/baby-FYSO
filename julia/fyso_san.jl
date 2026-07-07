# fyso_san.jl — SAN model scaffold for FYSO Case 1
# =======================================================
# Julia/JuMP implementation of the Shunting Action Network (SAN)
# 0-1 integer programming model from Han et al. (2025).
#
# This file scaffolds ONE stage of the sequential decision framework
# (Algorithm 1). In the full implementation, this would be called
# in a loop, updating the yard layout after each stage.
#
# Reference:
#   Han et al. (2025) SSRN:6346981, Sections 3-5.
#
# Requirements:
#   julia> using Pkg
#   julia> Pkg.add(["JuMP", "HiGHS"])
#
# Usage:
#   julia> include("fyso_san.jl")
#   julia> model, x, t = build_san_model(yard)
#   julia> optimize!(model)

using JuMP
using HiGHS

# ── Data structures ──────────────────────────────────────────────────────────

"""
A railcar in the yard.

Fields
------
- `id`          : unique integer identifier
- `destination` : block ID this car belongs to
- `length`      : physical length (metres)
- `track`       : current track index (1-based, track 0 = lead)
- `position`    : position on track (1 = closest to lead)
"""
struct Railcar
    id::Int
    destination::Int
    length::Float64
    track::Int
    position::Int
end

"""
An arc in the Shunting Action Network.

Fields
------
- `i`        : source railcar id
- `j`        : destination railcar id (or virtual node id, negative)
- `reward`   : contribution to the objective (Table 2 of Han et al.)
- `arc_type` : one of :shunting1, :shunting2, :shunting3,
                       :skipping1, :skipping2,
                       :holding1,  :holding2,
                       :virtual1,  :virtual2
"""
struct Arc
    i::Int
    j::Int
    reward::Float64
    arc_type::Symbol
end

"""
One stage of the flat yard, ready for SAN construction.

Fields
------
- `railcars`      : all cars currently in the yard
- `tracks`        : track indices (typically 1:5 for Case 1)
- `track_lengths` : Dict mapping track index → length in metres
- `arcs`          : all candidate arcs for this stage
- `desired_order` : block IDs in required departure order
"""
struct FlatYardStage
    railcars::Vector{Railcar}
    tracks::Vector{Int}
    track_lengths::Dict{Int,Float64}
    arcs::Vector{Arc}
    desired_order::Vector{Int}
end

# ── Reward constants (Table 2 of Han et al.) ─────────────────────────────────

const REWARDS = Dict(
    :shunting1 => 1.0,   # cross-track, correct order connection
    :shunting2 => 2.0,   # cross-track, same destination (clustering)
    :shunting3 => 0.0,   # cross-track, no useful connection
    :skipping1 => 1.0,   # same-track non-adjacent, correct order
    :skipping2 => 2.0,   # same-track non-adjacent, same destination
    :holding1  => 0.0,   # adjacent cars move together
    :holding2  => 0.1,   # adjacent same-destination cars move together
    :virtual1  => 0.5,   # move to empty track
    :virtual2  => 0.0,   # stay on same track's virtual slot
)

# ── SAN model (Section 4.1) ──────────────────────────────────────────────────

"""
    build_san_model(yard::FlatYardStage)

Construct the 0-1 integer programming model for one SAN stage.

Decision variables
------------------
- `x[(i,j)]`  : 1 if arc (i→j) is selected, 0 otherwise
- `t[k]`      : 1 if track k is the source track for this action

Constraints (equation numbers from Han et al. 2025)
----------------------------------------------------
- (2)  In-degree ≤ 1 per node
- (3)  Out-degree ≤ 1 per node
- (4)  Arc selection consistent with track selection
- (5)  Exactly one source track
- (6,7) All cars on selected track must have an outgoing arc
- (8)  Symmetry-breaking by position index
- (9)  Track-length capacity (simplified; lazy version in Section 4.2)
- (10) At least one effective shunting arc

Returns `(model, x, t)`.
"""
function build_san_model(yard::FlatYardStage)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    cars      = yard.railcars
    K         = yard.tracks
    A         = yard.arcs
    car_ids   = [c.id for c in cars]
    car_track = Dict(c.id => c.track    for c in cars)
    car_pos   = Dict(c.id => c.position for c in cars)
    car_len   = Dict(c.id => c.length   for c in cars)
    tlen      = yard.track_lengths

    # Cars grouped by track
    cars_on = Dict(k => [c.id for c in cars if c.track == k] for k in K)

    arc_pairs = [(a.i, a.j) for a in A]

    # ── Decision variables ────────────────────────────────────────────
    @variable(model, x[arc_pairs], Bin)
    @variable(model, t[K],         Bin)

    # ── Objective: maximise total arc reward (eq. 1) ─────────────────
    @objective(model, Max,
        sum(a.reward * x[(a.i, a.j)] for a in A)
    )

    # ── Constraints ───────────────────────────────────────────────────

    # (2) In-degree ≤ 1
    for j in car_ids
        incoming = [(a.i, a.j) for a in A if a.j == j]
        isempty(incoming) && continue
        @constraint(model, sum(x[p] for p in incoming) <= 1)
    end

    # (3) Out-degree ≤ 1
    for i in car_ids
        outgoing = [(a.i, a.j) for a in A if a.i == i]
        isempty(outgoing) && continue
        @constraint(model, sum(x[p] for p in outgoing) <= 1)
    end

    # (4) Arc selection implies track selection
    for k in K, a in A
        get(car_track, a.i, -1) == k || continue
        get(car_track, a.j, -1) == k || continue
        @constraint(model, x[(a.i, a.j)] <= t[k])
    end

    # (5) Exactly one source track
    @constraint(model, sum(t[k] for k in K) == 1)

    # (6,7) If track k selected, all its cars must have exactly one outgoing arc
    for k in K, i in cars_on[k]
        outgoing = [(a.i, a.j) for a in A if a.i == i]
        isempty(outgoing) && continue
        out_sum = sum(x[p] for p in outgoing)
        @constraint(model, out_sum == t[k])
    end

    # (8) Symmetry breaking: enforce position-based ordering within each track
    # (car closer to lead must have outgoing arc before car further from lead)
    for k in K
        sorted_cars = sort(cars_on[k]; by = i -> car_pos[i])
        for idx in 1:length(sorted_cars)-1
            i_near = sorted_cars[idx]      # closer to lead
            i_far  = sorted_cars[idx+1]    # further from lead
            out_near = [(a.i, a.j) for a in A if a.i == i_near]
            out_far  = [(a.i, a.j) for a in A if a.i == i_far]
            (isempty(out_near) || isempty(out_far)) && continue
            @constraint(model,
                sum(x[p] for p in out_far) <= sum(x[p] for p in out_near)
            )
        end
    end

    # (9) Track length capacity (simplified; paper Section 4.2 adds lazy version)
    for k in K, a in A
        get(car_track, a.j, -1) == k || continue
        n_on_track = length(cars_on[k])
        @constraint(model,
            x[(a.i, a.j)] * car_len[a.i] + n_on_track * get(car_len, a.j, 0.0) <= tlen[k]
        )
    end

    # (10) At least one effective (non-holding) shunting arc must be selected
    effective = Set([:shunting1, :shunting2, :virtual1])
    eff_arcs  = [(a.i, a.j) for a in A if a.arc_type in effective]
    if !isempty(eff_arcs)
        @constraint(model, sum(x[p] for p in eff_arcs) >= 1)
    end

    return model, x, t
end

# ── Solve and extract ─────────────────────────────────────────────────────────

"""
    solve_san(yard::FlatYardStage; verbose=false)

Build, solve, and print the SAN model for one stage.
Returns `(selected_arcs, selected_track)`.
"""
function solve_san(yard::FlatYardStage; verbose::Bool = false)
    model, x, t = build_san_model(yard)
    optimize!(model)

    status = termination_status(model)
    verbose && println("Solver status: $status")

    if status in (MOI.OPTIMAL, MOI.FEASIBLE_POINT)
        obj = objective_value(model)
        verbose && println("Objective (total reward): $obj")

        selected_arcs  = [(p, value(x[p])) for p in keys(x) if value(x[p]) > 0.5]
        selected_track = [k for k in keys(t) if value(t[k]) > 0.5]

        if verbose
            println("Selected track: $selected_track")
            println("Selected arcs:")
            for (p, _) in selected_arcs
                println("  $(p[1]) → $(p[2])")
            end
        end

        return selected_arcs, selected_track
    else
        @warn "No feasible solution found. Status: $status"
        return nothing, nothing
    end
end

# ── Case 1 — minimal example (Figure 1 of Han et al.) ───────────────────────

"""
    case1_example()

Run the SAN model on a 4-car example matching Figure 1 of Han et al.
Desired order: block 1 → block 2 → block 3.
Initial layout:
  Track I:  [block-1, block-3]   (block-1 at lead end)
  Track II: [block-2, block-1]   (block-2 at lead end)
"""
function case1_example()
    println("=" ^ 55)
    println("SAN Example — Figure 1 of Han et al. (2025)")
    println("=" ^ 55)

    cars = [
        Railcar(1, 1, 15.0, 1, 1),   # block-1, Track I,  pos 1 (lead)
        Railcar(2, 3, 15.0, 1, 2),   # block-3, Track I,  pos 2
        Railcar(3, 2, 15.0, 2, 1),   # block-2, Track II, pos 1 (lead)
        Railcar(4, 1, 15.0, 2, 2),   # block-1, Track II, pos 2
    ]

    # Arc set for initial state (first SAN stage):
    # block-2 (car 3) → block-1 (car 1): shunting arc 1 (correct order)
    # block-1 (car 4) → block-1 (car 1): shunting arc 2 (same destination)
    # remaining cars → virtual nodes
    arcs = [
        Arc(3, 1,  1.0, :shunting1),   # car3 → car1 (2 follows 1 in order)
        Arc(4, 1,  2.0, :shunting2),   # car4 → car1 (same block)
        Arc(1, -1, 0.5, :virtual1),    # car1 → virtual (empty track)
        Arc(2, -1, 0.5, :virtual1),    # car2 → virtual
        Arc(3, -2, 0.0, :virtual2),    # car3 → virtual (stay)
        Arc(4, -2, 0.0, :virtual2),    # car4 → virtual (stay)
    ]

    yard = FlatYardStage(
        cars,
        [1, 2, 3],
        Dict(1 => 2000.0, 2 => 1900.0, 3 => 1800.0),
        arcs,
        [1, 2, 3],
    )

    println("Running SAN model...")
    selected_arcs, selected_track = solve_san(yard; verbose = true)
    println()
    println("Next step: use selected arcs to update layout and build SAN for stage 2.")
    println("See Algorithm 1 in Han et al. for the full sequential loop.")

    return selected_arcs, selected_track
end

# ── Entry point ───────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    case1_example()
end
