#!/usr/bin/env python3
"""
single_item_cg_dp.py

Single-item Column Generation with DP pricing (NO enumeration, NO target columns).
- RMP: convex combination over ZIO columns
- Node constraints (as in your reproduction):
    * match production at t in {3,4,6,7} to given RHS
    * require setups y_4 = 1 and y_7 = 1
- Dummy-setup mechanism:
    * pricing generates a "pure" ZIO plan (setups only where production occurs),
    * then we "upgrade" the column to include required setups (y_4=y_7=1) by
      adding dummy setup cost for missing ones (X unchanged).

Key point: pricing is DP shortest-path on ZIO blocks (contiguous segments), NOT enumeration.

Run:
  source /Users/egecanaktan/github_repositories/MultiItemPerishability/.venv/bin/activate
  /Users/egecanaktan/github_repositories/MultiItemPerishability/.venv/bin/python \
    /Users/egecanaktan/github_repositories/MultiItemPerishability/bnp_solver/single_item_cg_dp.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import gurobipy as gp
from gurobipy import GRB


EPS = 1e-9
RC_EPS = 1e-7


# =============================================================================
# Instance guard (exact fixed instance)
# =============================================================================


def assert_same_instance(demand, c_var, setup, h, shelf_seq):
    exp_demand = [0, 0, 68, 49, 66, 38, 17, 17, 41, 43]
    exp_c_var = [
        1.7616423189662318,
        1.714378675341894,
        2.495224634136382,
        1.778983838209823,
        1.6742623735990734,
        2.1097890590592483,
        1.3744368033291616,
        1.2393047580737573,
        2.680699649170001,
        1.8054001149306065,
    ]
    exp_setup = [
        80,
        81.66329352654208,
        83.2538931446064,
        84.70228201833979,
        85.94515860381915,
        86.9282032302755,
        87.60845213036123,
        87.9561751629462,
        87.9561751629462,
        87.60845213036123,
    ]
    exp_h = [
        0.4,
        0.4083164676327104,
        0.41626946572303203,
        0.423511410091699,
        0.42972579301909575,
        0.43464101615137757,
        0.4380422606518062,
        0.439780875814731,
        0.439780875814731,
        0.4380422606518062,
    ]
    exp_shelf = [24, 18, 22, 6, 16, 9, 21, 23, 9, 7]

    if (
        demand != exp_demand
        or c_var != exp_c_var
        or setup != exp_setup
        or h != exp_h
        or shelf_seq != exp_shelf
    ):
        raise SystemExit(
            "[FAIL] Instance differs from the verified one. "
            "This script intentionally supports only the single fixed test instance."
        )


# =============================================================================
# Column structure
# =============================================================================


@dataclass(frozen=True)
class Column:
    name: str
    blocks: Tuple[Tuple[int, int], ...]  # (s,e) blocks covering demand
    x: Tuple[float, ...]  # production per period
    setups: Tuple[int, ...]  # y=1 periods after dummy-setup upgrade
    base_setups: Tuple[int, ...]  # real setups (block starts)
    cost: float  # cost after dummy-setup upgrade
    base_cost: float  # cost before dummy-setup upgrade


def holding_sum_prefix(h: List[float]) -> List[float]:
    """prefix[k] = sum_{i=0..k-1} h[i]"""
    pref = [0.0]
    s = 0.0
    for v in h:
        s += float(v)
        pref.append(s)
    return pref


def holding_sum(pref: List[float], s: int, u: int) -> float:
    """Sum_{k=s..u-1} h[k]. If u<=s -> 0."""
    if u <= s:
        return 0.0
    return pref[u] - pref[s]


def production_vector_from_blocks(
    demand: List[int], T: int, blocks: List[Tuple[int, int]]
) -> List[float]:
    x = [0.0] * T
    for s, e in blocks:
        q = float(sum(demand[u] for u in range(s, e + 1)))
        if q > EPS:
            x[s] += q
    return x


def column_cost_from_blocks(
    demand: List[int],
    c_var: List[float],
    setup_cost: List[float],
    h_pref: List[float],
    blocks: List[Tuple[int, int]],
) -> float:
    """
    ZIO block interpretation:
      block (s,e) covers demand u=s..e from production at s.
      qty at s = sum_{u=s..e} d[u]
      cost = setup[s] + sum_{u=s..e} d[u]*(c_var[s] + holding(s,u))
    """
    total = 0.0
    for s, e in blocks:
        q = sum(float(demand[u]) for u in range(s, e + 1))
        if q <= EPS:
            continue
        total += float(setup_cost[s])
        for u in range(s, e + 1):
            du = float(demand[u])
            total += du * (float(c_var[s]) + holding_sum(h_pref, s, u))
    return total


def upgrade_required_setups(
    setup_cost: List[float],
    base_setups: Set[int],
    required_setups: List[int],
) -> Tuple[Tuple[int, ...], float, Tuple[int, ...]]:
    """
    Ensure y_t=1 for all required_setups by adding dummy setups if missing.
    Returns:
      upgraded_setups (sorted tuple),
      extra_cost (sum setup[t] for missing),
      missing (sorted tuple)
    """
    missing = sorted(set(required_setups) - set(base_setups))
    extra = sum(float(setup_cost[t]) for t in missing)
    upgraded = sorted(set(base_setups) | set(required_setups))
    return tuple(upgraded), extra, tuple(missing)


def column_signature(
    x: Tuple[float, ...], setups: Tuple[int, ...]
) -> Tuple[Tuple[int, int], Tuple[int, ...]]:
    """
    Signature to avoid duplicates:
      - sparse x as list of (t, int(q)) pairs
      - setups tuple
    """
    sparse = tuple((t, int(round(q))) for t, q in enumerate(x) if abs(q) > 1e-6)
    return sparse, setups


# =============================================================================
# DP pricing (NO enumeration)
# =============================================================================


def dp_best_zio_blocks(
    demand: List[int],
    c_var: List[float],
    setup_cost: List[float],
    h_pref: List[float],
    shelf_seq: List[int],
    alpha: Dict[int, float],
    t_start: int,
    t_end: int,
    tie_penalty_on_starts: Optional[Dict[int, float]] = None,
) -> List[Tuple[int, int]]:
    """
    Find the minimum (cost - alpha*production) ZIO partition covering all periods t_start..t_end.
    Because in this instance d[t]>0 for all t in [2..9], ZIO plans correspond to partitions into
    contiguous segments.

    DP state:
      F[t] = min reduced segment-sum to cover [t..t_end]
      choose end e in [t..max_feasible_end(t)] then recurse to e+1

    tie_penalty_on_starts:
      only used if we get duplicate columns; adds a tiny epsilon to discourage identical starts.
      This is NOT a "target column"; it's just a deterministic tie-break / diversification.
    """
    T = len(demand)
    INF = 1e100

    F = [INF] * (T + 1)
    nxt = [-1] * (T + 1)

    F[t_end + 1] = 0.0

    for t in range(t_end, t_start - 1, -1):
        # max end allowed by shelf life
        L = int(shelf_seq[t])
        max_end = min(t_end, t + L - 1)

        best = INF
        best_e = -1

        # precompute q(t,e) cumulative
        q = 0.0
        seg_cost = 0.0  # we will rebuild for each e (T small anyway)

        for e in range(t, max_end + 1):
            # segment cost for block (t,e)
            # q(t,e) = sum d[u]
            q = float(sum(demand[u] for u in range(t, e + 1)))

            # base segment cost
            seg_cost = float(setup_cost[t])
            for u in range(t, e + 1):
                du = float(demand[u])
                seg_cost += du * (float(c_var[t]) + holding_sum(h_pref, t, u))

            # subtract alpha[t] * x_t where x_t = q produced at start t
            seg_cost -= float(alpha.get(t, 0.0)) * q

            # tiny tie-break/diversification if requested
            if tie_penalty_on_starts and t in tie_penalty_on_starts:
                seg_cost += float(tie_penalty_on_starts[t])

            cand = seg_cost + F[e + 1]
            if cand < best - 1e-12:
                best = cand
                best_e = e

        F[t] = best
        nxt[t] = best_e

    # reconstruct blocks
    blocks: List[Tuple[int, int]] = []
    t = t_start
    while t <= t_end:
        e = nxt[t]
        if e < t:
            raise RuntimeError("DP reconstruction failed (no feasible next end).")
        blocks.append((t, e))
        t = e + 1

    return blocks


def price_column_dp(
    demand: List[int],
    c_var: List[float],
    setup_cost: List[float],
    h_pref: List[float],
    shelf_seq: List[int],
    alpha: Dict[int, float],
    mu: float,
    beta: Dict[int, float],
    required_setups: List[int],
    name: str,
    dup_avoid_signatures: Set[Tuple[Tuple[Tuple[int, int], ...], Tuple[int, ...]]],
) -> Tuple[Optional[Column], float]:
    """
    Returns (best_column, reduced_cost) using DP pricing.
    If a duplicate is produced, apply tiny diversification penalties and retry a few times.
    """
    # demand positive periods: here 2..9
    t_start = next(t for t, d in enumerate(demand) if d > 0)
    t_end = max(t for t, d in enumerate(demand) if d > 0)

    # Try a few times if we hit duplicates
    tie_penalty: Dict[int, float] = {}
    for attempt in range(30):
        blocks = dp_best_zio_blocks(
            demand=demand,
            c_var=c_var,
            setup_cost=setup_cost,
            h_pref=h_pref,
            shelf_seq=shelf_seq,
            alpha=alpha,
            t_start=t_start,
            t_end=t_end,
            tie_penalty_on_starts=tie_penalty if tie_penalty else None,
        )

        T = len(demand)
        x = tuple(production_vector_from_blocks(demand, T, blocks))
        base_setups = sorted(
            {s for (s, e) in blocks if sum(demand[u] for u in range(s, e + 1)) > 0}
        )
        base_setups_t = tuple(base_setups)

        base_cost = column_cost_from_blocks(demand, c_var, setup_cost, h_pref, blocks)

        # upgrade to satisfy required setups via dummy setups
        upgraded_setups, extra_cost, missing = upgrade_required_setups(
            setup_cost=setup_cost,
            base_setups=set(base_setups),
            required_setups=required_setups,
        )
        cost = base_cost + extra_cost

        # reduced cost w.r.t. current duals
        # rc = cost - mu - sum_t alpha[t]*x[t] - sum_r beta[r]*y[r]
        rc = cost - float(mu)
        rc -= sum(float(alpha.get(t, 0.0)) * float(x[t]) for t in range(T))
        rc -= sum(
            float(beta.get(r, 0.0)) * 1.0 for r in required_setups
        )  # y[r]=1 after upgrade

        sig = column_signature(x, upgraded_setups)
        if sig not in dup_avoid_signatures:
            col = Column(
                name=name,
                blocks=tuple(blocks),
                x=x,
                setups=upgraded_setups,
                base_setups=base_setups_t,
                cost=float(cost),
                base_cost=float(base_cost),
            )
            return col, float(rc)

        # duplicate -> add tiny penalty to one of its starts to force a different DP solution next try
        # (still DP; no enumeration; just avoiding infinite loops on identical best column)
        starts = [s for (s, e) in blocks]
        if not starts:
            break
        bump_s = starts[attempt % len(starts)]
        tie_penalty[bump_s] = tie_penalty.get(bump_s, 0.0) + 1e-4

    return None, 0.0


# =============================================================================
# RMP build/solve with artificials (to start from scratch)
# =============================================================================


@dataclass
class RMP:
    model: gp.Model
    lam: List[gp.Var]
    # artificials for match constraints: p_t - n_t
    p_match: Dict[int, gp.Var]
    n_match: Dict[int, gp.Var]
    # artificials for force setup constraints: p_y - n_y
    p_y: Dict[int, gp.Var]
    n_y: Dict[int, gp.Var]
    # constraints
    cons_conv: gp.Constr
    cons_match: Dict[int, gp.Constr]
    cons_y: Dict[int, gp.Constr]


def build_rmp(
    T: int,
    rhs_x: Dict[int, float],
    match_periods: List[int],
    required_setups: List[int],
    columns: List[Column],
    bigM: float = 1e6,
) -> RMP:
    m = gp.Model("RMP_single_item")
    m.Params.OutputFlag = 0
    m.Params.LogToConsole = 0

    lam = []
    for k, col in enumerate(columns):
        lam.append(m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"lam_{k}"))

    # artificials for match_x equalities
    p_match: Dict[int, gp.Var] = {}
    n_match: Dict[int, gp.Var] = {}
    for t in match_periods:
        p_match[t] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"p_match_{t}")
        n_match[t] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"n_match_{t}")

    # artificials for required y equalities (kept for completeness)
    p_y: Dict[int, gp.Var] = {}
    n_y: Dict[int, gp.Var] = {}
    for t in required_setups:
        p_y[t] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"p_y_{t}")
        n_y[t] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"n_y_{t}")

    # convexity
    cons_conv = m.addConstr(gp.quicksum(lam) == 1.0, name="conv")

    # match production constraints
    cons_match: Dict[int, gp.Constr] = {}
    for t in match_periods:
        cons_match[t] = m.addConstr(
            gp.quicksum(lam[k] * float(columns[k].x[t]) for k in range(len(columns)))
            + p_match[t]
            - n_match[t]
            == float(rhs_x[t]),
            name=f"match_x_{t}",
        )

    # required setup constraints (after upgrade, every column should have y[t]=1; still keep constraint)
    cons_y: Dict[int, gp.Constr] = {}
    for t in required_setups:
        cons_y[t] = m.addConstr(
            gp.quicksum(
                lam[k] * (1.0 if t in columns[k].setups else 0.0)
                for k in range(len(columns))
            )
            + p_y[t]
            - n_y[t]
            == 1.0,
            name=f"force_y_{t}",
        )

    # objective: min sum lam*cost + bigM * sum artificials
    obj = gp.quicksum(lam[k] * float(columns[k].cost) for k in range(len(columns)))
    obj += bigM * gp.quicksum(p_match[t] + n_match[t] for t in match_periods)
    obj += bigM * gp.quicksum(p_y[t] + n_y[t] for t in required_setups)

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    return RMP(
        model=m,
        lam=lam,
        p_match=p_match,
        n_match=n_match,
        p_y=p_y,
        n_y=n_y,
        cons_conv=cons_conv,
        cons_match=cons_match,
        cons_y=cons_y,
    )


def extract_duals(
    rmp: RMP, match_periods: List[int], required_setups: List[int]
) -> Tuple[Dict[int, float], float, Dict[int, float]]:
    m = rmp.model
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"RMP not optimal. Status={m.Status}")

    mu = float(rmp.cons_conv.Pi)

    alpha: Dict[int, float] = {}
    for t in match_periods:
        alpha[t] = float(rmp.cons_match[t].Pi)

    beta: Dict[int, float] = {}
    for t in required_setups:
        beta[t] = float(rmp.cons_y[t].Pi)

    return alpha, mu, beta


def rmp_artificial_violation(
    rmp: RMP, match_periods: List[int], required_setups: List[int]
) -> float:
    s = 0.0
    for t in match_periods:
        s += float(rmp.p_match[t].X) + float(rmp.n_match[t].X)
    for t in required_setups:
        s += float(rmp.p_y[t].X) + float(rmp.n_y[t].X)
    return s


# =============================================================================
# Main CG driver
# =============================================================================


def main():
    # --- Verified instance data (single item) ---
    T = 10
    demand = [0, 0, 68, 49, 66, 38, 17, 17, 41, 43]
    h = [
        0.4,
        0.4083164676327104,
        0.41626946572303203,
        0.423511410091699,
        0.42972579301909575,
        0.43464101615137757,
        0.4380422606518062,
        0.439780875814731,
        0.439780875814731,
        0.4380422606518062,
    ]
    c_var = [
        1.7616423189662318,
        1.714378675341894,
        2.495224634136382,
        1.778983838209823,
        1.6742623735990734,
        2.1097890590592483,
        1.3744368033291616,
        1.2393047580737573,
        2.680699649170001,
        1.8054001149306065,
    ]
    setup = [
        80,
        81.66329352654208,
        83.2538931446064,
        84.70228201833979,
        85.94515860381915,
        86.9282032302755,
        87.60845213036123,
        87.9561751629462,
        87.9561751629462,
        87.60845213036123,
    ]
    shelf_seq = [24, 18, 22, 6, 16, 9, 21, 23, 9, 7]

    assert_same_instance(demand, c_var, setup, h, shelf_seq)

    # write instance json
    inst = {
        "T": T,
        "demand": demand,
        "h": h,
        "c_var": c_var,
        "setup": setup,
        "shelf_seq": shelf_seq,
    }
    with open("single_item_instance.json", "w", encoding="utf-8") as f:
        json.dump(inst, f, indent=2)

    # Node constraints (same as your reproduction)
    match_periods = [3, 4, 6, 7]
    rhs_x = {3: 73.0, 4: 80.0, 6: 38.0, 7: 80.0}
    required_setups = [4, 7]

    print(f"[Instance] T={T}")
    print(
        f"[Node] match_periods={match_periods}, rhs_x={rhs_x}, required_setups={required_setups}"
    )
    print("[Info] Wrote single_item_instance.json")

    h_pref = holding_sum_prefix(h)

    # Column pool (start empty; CG must generate)
    columns: List[Column] = []
    signatures: Set[Tuple[Tuple[Tuple[int, int], ...], Tuple[int, ...]]] = set()

    # We still need an initial RMP to get duals.
    # Create one "first" column by DP with zero duals (this is not a target; it is just a feasible column candidate).
    alpha0 = {t: 0.0 for t in match_periods}
    beta0 = {t: 0.0 for t in required_setups}
    mu0 = 0.0

    col0, rc0 = price_column_dp(
        demand=demand,
        c_var=c_var,
        setup_cost=setup,
        h_pref=h_pref,
        shelf_seq=shelf_seq,
        alpha=alpha0,
        mu=mu0,
        beta=beta0,
        required_setups=required_setups,
        name="K0",
        dup_avoid_signatures=signatures,
    )
    if col0 is None:
        raise SystemExit("[FAIL] Could not generate initial column by DP.")
    columns.append(col0)
    signatures.add(column_signature(col0.x, col0.setups))

    print("\n[Init column from DP(duals=0)]")
    print(
        f"  {col0.name}: blocks={col0.blocks} base_setups={col0.base_setups} setups(upgraded)={col0.setups}"
    )
    print(f"  x(nonzero)={[ (t,int(col0.x[t])) for t in range(T) if col0.x[t]>1e-6 ]}")
    print(f"  base_cost={col0.base_cost:.6f} cost(upgraded)={col0.cost:.6f}")

    # CG loop
    max_iters = 100
    bigM = 1e6

    for it in range(max_iters):
        rmp = build_rmp(
            T=T,
            rhs_x=rhs_x,
            match_periods=match_periods,
            required_setups=required_setups,
            columns=columns,
            bigM=bigM,
        )

        viol = rmp_artificial_violation(rmp, match_periods, required_setups)
        alpha, mu, beta = extract_duals(rmp, match_periods, required_setups)

        print(
            f"\n[CG iter {it}] columns={len(columns)}  artificial_violation={viol:.6e}"
        )
        print(f"  dual mu(conv)={mu:.6f}")
        for t in match_periods:
            print(f"  dual alpha[{t}]={alpha[t]:.6f}")
        for t in required_setups:
            print(f"  dual beta[{t}]={beta[t]:.6f}")

        new_col, new_rc = price_column_dp(
            demand=demand,
            c_var=c_var,
            setup_cost=setup,
            h_pref=h_pref,
            shelf_seq=shelf_seq,
            alpha=alpha,
            mu=mu,
            beta=beta,
            required_setups=required_setups,
            name=f"K{len(columns)}",
            dup_avoid_signatures=signatures,
        )

        if new_col is None:
            print("  [STOP] Pricing only produced duplicates; ending.")
            break

        print(
            f"  pricing: rc={new_rc:.8f}  blocks={new_col.blocks}  base_setups={new_col.base_setups}"
        )

        if new_rc >= -RC_EPS:
            print("  [STOP] No negative reduced cost column.")
            break

        # add column
        columns.append(new_col)
        signatures.add(column_signature(new_col.x, new_col.setups))

    # Final solve (same RMP; by now artificials should be ~0)
    rmp_final = build_rmp(
        T=T,
        rhs_x=rhs_x,
        match_periods=match_periods,
        required_setups=required_setups,
        columns=columns,
        bigM=bigM,
    )
    viol_final = rmp_artificial_violation(rmp_final, match_periods, required_setups)
    print(f"\n[Final] columns={len(columns)}  artificial_violation={viol_final:.6e}")

    # Report lambdas
    lam_vals = [float(v.X) for v in rmp_final.lam]
    print("\n[Final lambdas (nonzeros)]")
    for k, val in enumerate(lam_vals):
        if abs(val) > 1e-9:
            print(
                f"  lam[{k}] ({columns[k].name}) = {val:.9f}  blocks={columns[k].blocks}"
            )

    # Implied x on match periods
    print("\n[Reconstruction check on match periods]")
    for t in match_periods:
        x_imp = sum(lam_vals[k] * float(columns[k].x[t]) for k in range(len(columns)))
        print(f"  t={t}:  {x_imp:.6f}  (rhs={rhs_x[t]:.6f})")

    # Try to locate your 4 “screenshot columns” by block pattern
    target_blocks = {
        ((2, 2), (3, 3), (4, 5), (6, 6), (7, 9)): "C1",
        ((2, 2), (3, 3), (4, 5), (6, 9)): "C2",
        ((2, 2), (3, 5), (6, 6), (7, 9)): "C3",
        ((2, 2), (3, 5), (6, 9)): "C4",
    }

    found = {}
    for k, col in enumerate(columns):
        b = tuple(col.blocks)
        if b in target_blocks:
            found[target_blocks[b]] = (k, col)

    print("\n[Detected screenshot columns in generated pool]")
    for tag in ["C1", "C2", "C3", "C4"]:
        if tag in found:
            k, col = found[tag]
            print(
                f"  {tag}: at k={k} name={col.name} blocks={col.blocks} x@starts={[ (t,int(col.x[t])) for t in range(T) if col.x[t]>1e-6 ]}"
            )
        else:
            print(f"  {tag}: NOT FOUND")

    print("\nDone.")


if __name__ == "__main__":
    main()
