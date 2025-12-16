#!/usr/bin/env python3
"""
single_item_dw_lp.py

Dantzig-Wolfe LP for SINGLE-ITEM capacitated lot-sizing via column generation.

- Subproblem (pricing): generates UNCAPACITATED ZIO plans (blocks/partitions).
- Master (RMP): chooses lambdas over plans to satisfy CAPACITY by convex combination.
- Supports "dummy setups": y_t=1 with x_t=0 inside a column (adds setup cost).
  These appear when you FORCE y_t=1 in the master but the ZIO plan doesn't start at t.

This is EXACTLY the dashed-line behavior you drew.

Requirements:
  pip install gurobipy  (and a working Gurobi license)

Run:
  python single_item_dw_lp.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Any

import gurobipy as gp
from gurobipy import GRB

EPS = 1e-9
RC_EPS = 1e-7
BIGM = 1e6
MAX_CG_ITERS = 400


# =============================================================================
# INSTANCE (use your demo)
# =============================================================================


def demo_instance() -> Dict[str, Any]:
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
    cap = [0, 0, 80, 80, 80, 80, 80, 80, 100, 80]

    return {
        "T": T,
        "demand": demand,
        "h": h,
        "c_var": c_var,
        "setup": setup,
        "shelf_seq": shelf_seq,
        "cap": cap,
    }


# =============================================================================
# COLUMN STRUCTURE
# =============================================================================


@dataclass(frozen=True)
class Column:
    # ZIO blocks (s,e) partition of demands (including zeros ok)
    blocks: Tuple[Tuple[int, int], ...]
    # production in each period (only at starts)
    x: Tuple[float, ...]
    # real setup starts (where x>0)
    y_starts: Tuple[int, ...]
    # all setups in the column (real + dummy)
    y_all: Tuple[int, ...]
    # total cost including dummy setup costs
    cost: float


def holding_prefix(h: List[float]) -> List[float]:
    pref = [0.0]
    s = 0.0
    for v in h:
        s += float(v)
        pref.append(s)
    return pref


def holding_sum(pref: List[float], s: int, u: int) -> float:
    # sum h[k] for k in [s, u-1]
    return 0.0 if u <= s else (pref[u] - pref[s])


def build_column_from_blocks(
    T: int,
    demand: List[int],
    setup: List[float],
    c_var: List[float],
    h_pref: List[float],
    blocks: List[Tuple[int, int]],
    forced_setups: Optional[List[int]] = None,
) -> Column:
    """
    Build a full uncapacitated ZIO plan from blocks.
    Then add dummy setups for forced_setups not in y_starts:
      - adds setup cost
      - x[t]=0 at dummy setup times
    """
    forced_setups = forced_setups or []

    x = [0.0] * T
    y_starts: List[int] = []
    cost = 0.0

    for s, e in blocks:
        q = float(sum(demand[u] for u in range(s, e + 1)))
        if q <= EPS:
            # nothing to produce here
            continue

        # produce at s
        x[s] += q
        y_starts.append(s)
        cost += float(setup[s])

        # variable + holding costs to serve each demand u from production at s
        for u in range(s, e + 1):
            du = float(demand[u])
            if du <= 0:
                continue
            cost += du * (float(c_var[s]) + holding_sum(h_pref, s, u))

    # dummy setups (y=1, x=0) for forced ones not present
    missing = sorted(set(forced_setups) - set(y_starts))
    for t in missing:
        cost += float(setup[t])

    y_all = sorted(set(y_starts) | set(forced_setups))

    return Column(
        blocks=tuple(blocks),
        x=tuple(x),
        y_starts=tuple(sorted(y_starts)),
        y_all=tuple(y_all),
        cost=float(cost),
    )


def col_sig(col: Column) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[int, ...]]:
    # blocks + all setups (dummy matters)
    return col.blocks, col.y_all


# =============================================================================
# PRICING (DP shortest path over ZIO partitions)
# =============================================================================


def price_column_dp(
    T: int,
    demand: List[int],
    setup: List[float],
    c_var: List[float],
    h_pref: List[float],
    shelf_seq: List[int],
    # duals
    mu: float,
    pi_cap: Dict[int, float],
    sigma_yfix: Dict[int, float],
    # forced dummy setups (from y-fix = 1)
    forced_setups: List[int],
) -> Tuple[Optional[Column], float]:
    """
    DP chooses a partition into blocks (s,e).
    State: F[t] = min reduced cost to cover demands from t..last
    Transition: choose end e for block starting at t.

    IMPORTANT:
      - You may SKIP t only if demand[t]==0 AND t is not forced setup.
      - If demand[t]>0, you must cover it: start a block at t in this state.
        (If demand[t] is to be served earlier, then this state would not be reached.)
      - To allow earlier production for later demand, DP must start at t=0 (not first demand).
    """
    last = max((i for i, d in enumerate(demand) if d > 0), default=0)

    INF = 1e100
    F = [INF] * (T + 2)
    nxt = [-1] * (T + 2)
    F[last + 1] = 0.0

    forced = set(forced_setups)

    for t in range(last, -1, -1):
        L = int(shelf_seq[t])
        max_end = min(last, t + L - 1)

        best = INF
        best_e = -1

        # option: skip t (only if no demand at t and not forced)
        if demand[t] == 0 and t not in forced:
            if F[t + 1] < best:
                best = F[t + 1]
                best_e = t  # skip marker

        # option: start block at t and end at e
        for e in range(t, max_end + 1):
            seg = float(setup[t])
            q = 0.0
            for u in range(t, e + 1):
                du = float(demand[u])
                q += du
                if du > 0:
                    seg += du * (float(c_var[t]) + holding_sum(h_pref, t, u))

            # reduced cost subtract dual*coeff:
            # capacity: coeff is x_t (=q produced at t)
            seg -= float(pi_cap.get(t, 0.0)) * q

            # y-fix constraints (only exist if you set them): coeff is 1 if y_t=1 in column
            seg -= float(sigma_yfix.get(t, 0.0))  # since start at t => y_t=1

            cand = seg + F[e + 1]
            if cand < best - 1e-12:
                best = cand
                best_e = e

        F[t] = best
        nxt[t] = best_e

    if F[0] >= INF / 2:
        return None, 0.0

    # reconstruct blocks
    blocks: List[Tuple[int, int]] = []
    t = 0
    while t <= last:
        e = nxt[t]
        if e == -1:
            return None, 0.0
        if e == t and demand[t] == 0 and t not in forced:
            t += 1
            continue
        blocks.append((t, e))
        t = e + 1

    col = build_column_from_blocks(
        T=T,
        demand=demand,
        setup=setup,
        c_var=c_var,
        h_pref=h_pref,
        blocks=blocks,
        forced_setups=forced_setups,
    )

    # dummy setup reduced-cost contribution (for forced setups not real starts):
    # each dummy contributes (setup[t] - sigma_yfix[t]) to reduced cost
    dummy_rc = 0.0
    missing = sorted(set(forced_setups) - set(col.y_starts))
    for tt in missing:
        dummy_rc += float(setup[tt]) - float(sigma_yfix.get(tt, 0.0))

    rc = float(F[0]) + float(dummy_rc) - float(mu)
    return col, float(rc)


# =============================================================================
# MASTER (RMP) LP
# =============================================================================


@dataclass
class RMP:
    m: gp.Model
    lam: List[gp.Var]
    conv: gp.Constr
    cap: Dict[int, gp.Constr]
    yfix: Dict[int, gp.Constr]
    vcap: Dict[int, gp.Var]  # slack to keep feasibility during CG


def build_rmp(
    T: int,
    cols: List[Column],
    cap: List[float],
    y_fix: Dict[
        int, int
    ],  # optional: force y_t = 0/1 at the aggregate (this is where dummy matters)
) -> RMP:
    m = gp.Model("RMP")
    m.Params.OutputFlag = 0

    lam = [
        m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"lam_{k}")
        for k in range(len(cols))
    ]
    conv = m.addConstr(gp.quicksum(lam) == 1.0, name="conv")

    # capacity with feasibility slack
    cap_con: Dict[int, gp.Constr] = {}
    vcap: Dict[int, gp.Var] = {}
    for t in range(T):
        vcap[t] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"v_cap_{t}")
        lhs = gp.quicksum(lam[k] * float(cols[k].x[t]) for k in range(len(cols)))
        cap_con[t] = m.addConstr(lhs <= float(cap[t]) + vcap[t], name=f"cap_{t}")

    # optional y-fix equalities: sum_k lam_k * y_t^k == val
    yfix_con: Dict[int, gp.Constr] = {}
    for t, val in y_fix.items():
        lhs = gp.quicksum(
            lam[k] * (1.0 if t in cols[k].y_all else 0.0) for k in range(len(cols))
        )
        yfix_con[t] = m.addConstr(lhs == float(val), name=f"yfix_{t}")

    # objective: sum lam_k * col_cost + BIGM * sum vcap (force cap violation to 0)
    obj = gp.quicksum(lam[k] * float(cols[k].cost) for k in range(len(cols)))
    obj += BIGM * gp.quicksum(vcap.values())
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    return RMP(m=m, lam=lam, conv=conv, cap=cap_con, yfix=yfix_con, vcap=vcap)


def extract_duals(rmp: RMP) -> Tuple[float, Dict[int, float], Dict[int, float]]:
    if rmp.m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"RMP not optimal, status={rmp.m.Status}")

    mu = float(rmp.conv.Pi)
    pi = {t: float(con.Pi) for t, con in rmp.cap.items()}
    sigma = {t: float(con.Pi) for t, con in rmp.yfix.items()}
    return mu, pi, sigma


def cap_violation_amount(rmp: RMP) -> float:
    return sum(float(v.X) for v in rmp.vcap.values())


# =============================================================================
# INITIAL COLUMNS
# =============================================================================


def initial_columns(
    T: int,
    demand: List[int],
    setup: List[float],
    c_var: List[float],
    h: List[float],
    shelf_seq: List[int],
) -> Tuple[List[Column], Set[Tuple]]:
    h_pref = holding_prefix(h)
    cols: List[Column] = []
    sigs: Set[Tuple] = set()

    # a few heuristic partitions: singleton, all-in-one starting at 0, and some staggered
    last = max((i for i, d in enumerate(demand) if d > 0), default=0)

    candidates = []

    # (1) all-in-one block starting at 0
    candidates.append([(0, last)])

    # (2) singletons on demand-positive indices, but allow starting from 0 by grouping leading zeros
    blocks = []
    t = 0
    while t <= last:
        if demand[t] == 0:
            t += 1
            continue
        blocks.append((t, t))
        t += 1
    if blocks:
        candidates.append(blocks)

    # (3) simple split at a few cut points
    for cut in range(1, min(last, 6)):
        candidates.append([(0, cut), (cut + 1, last)])

    for blocks in candidates:
        col = build_column_from_blocks(
            T, demand, setup, c_var, h_pref, blocks, forced_setups=[]
        )
        s = col_sig(col)
        if s not in sigs:
            cols.append(col)
            sigs.add(s)

    return cols, sigs


# =============================================================================
# MAIN: COLUMN GENERATION LOOP
# =============================================================================


def solve_dw_lp(
    instance: Dict[str, Any], y_fix: Optional[Dict[int, int]] = None
) -> Dict[str, Any]:
    T = int(instance["T"])
    demand = list(map(int, instance["demand"]))
    setup = list(map(float, instance["setup"]))
    c_var = list(map(float, instance["c_var"]))
    h = list(map(float, instance["h"]))
    cap = list(map(float, instance["cap"]))
    shelf_seq = list(map(int, instance["shelf_seq"]))

    y_fix = y_fix or {}
    forced_setups = sorted([t for t, v in y_fix.items() if v == 1])

    h_pref = holding_prefix(h)
    cols, sigs = initial_columns(T, demand, setup, c_var, h, shelf_seq)

    t0 = time.time()
    for it in range(MAX_CG_ITERS):
        rmp = build_rmp(T, cols, cap, y_fix)
        mu, pi, sigma = extract_duals(rmp)

        col, rc = price_column_dp(
            T=T,
            demand=demand,
            setup=setup,
            c_var=c_var,
            h_pref=h_pref,
            shelf_seq=shelf_seq,
            mu=mu,
            pi_cap=pi,
            sigma_yfix=sigma,
            forced_setups=forced_setups,
        )

        if col is None or rc >= -RC_EPS:
            # stop CG
            if cap_violation_amount(rmp) > 1e-6:
                return {
                    "status": "infeasible_under_cap",
                    "reason": "RMP still needs capacity slacks > 0 (not enough columns to satisfy cap).",
                    "iterations": it + 1,
                    "columns": len(cols),
                }

            lam_vals = [float(v.X) for v in rmp.lam]
            x_agg = [0.0] * T
            y_agg = {t: 0.0 for t in range(T)}
            for k, lam in enumerate(lam_vals):
                if abs(lam) < 1e-12:
                    continue
                for t in range(T):
                    x_agg[t] += lam * float(cols[k].x[t])
                for t in cols[k].y_all:
                    y_agg[t] += lam

            return {
                "status": "optimal_lp",
                "objective": float(rmp.m.ObjVal),
                "iterations": it + 1,
                "columns": len(cols),
                "lambda": lam_vals,
                "x_agg": x_agg,
                "y_agg": y_agg,
                "y_fix": dict(y_fix),
                "active_cols": [
                    {
                        "k": k,
                        "lambda": lam_vals[k],
                        "blocks": cols[k].blocks,
                        "y_starts": cols[k].y_starts,
                        "y_all": cols[k].y_all,
                        "x_nonzero": [
                            (t, cols[k].x[t])
                            for t in range(T)
                            if abs(cols[k].x[t]) > 1e-9
                        ],
                        "cost": cols[k].cost,
                    }
                    for k in range(len(cols))
                    if abs(lam_vals[k]) > 1e-9
                ],
                "time_sec": time.time() - t0,
            }

        # add new column if new signature
        s = col_sig(col)
        if s not in sigs:
            cols.append(col)
            sigs.add(s)
        else:
            # rare: pricing returns duplicate; nudge by just stopping (simple script)
            # (if you want diversification, I can add it, but you asked simple)
            continue

    return {
        "status": "max_iters",
        "iterations": MAX_CG_ITERS,
        "columns": len(cols),
        "time_sec": time.time() - t0,
    }


# =============================================================================
# CLICK-RUN
# =============================================================================

if __name__ == "__main__":
    inst = demo_instance()

    # --------------------------
    # OPTIONAL: Force dashed dummy setups in the convex combination.
    # This is EXACTLY where dummy y=1,x=0 shows up.
    #
    # Example: force y[5]=1 even if a plan doesn't produce at 5.
    # Then pricing will create columns with a dummy setup at 5 (adds setup cost, x[5]=0).
    # --------------------------
    Y_FIX = {
        # 5: 1,
        # 4: 0,
    }

    res = solve_dw_lp(inst, y_fix=Y_FIX)

    print("\n================= RESULT =================")
    print("status:", res["status"])
    if res["status"] == "optimal_lp":
        print("objective:", res["objective"])
        print(
            "iters:",
            res["iterations"],
            "cols:",
            res["columns"],
            "time:",
            f"{res['time_sec']:.2f}s",
        )
        print("\nAGG x (capacity-feasible):")
        for t, v in enumerate(res["x_agg"]):
            if abs(v) > 1e-6:
                print(f"  t={t}: {v:.6f}   cap={inst['cap'][t]}")
        print("\nAGG y (can be fractional):")
        y_ones = [t for t, v in res["y_agg"].items() if v > 1e-6]
        print("  support:", y_ones)
        for t in y_ones:
            print(f"  t={t}: y={res['y_agg'][t]:.6f}")
        print("\nACTIVE COLUMNS:")
        for col in res["active_cols"]:
            print(f"  k={col['k']}  lam={col['lambda']:.9f}  cost={col['cost']:.3f}")
            print(f"    blocks={col['blocks']}")
            print(f"    y_starts={col['y_starts']}")
            print(f"    y_all   ={col['y_all']}   (includes dummy if any)")
            print(f"    x_nonzero={col['x_nonzero']}")
    else:
        print(res)
