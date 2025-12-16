#!/usr/bin/env python3
"""
Block-Based Column Generation for Capacitated Lot-Sizing.

Each column represents a BLOCK - production at period t serving demands from t to e.
Multiple blocks are combined to cover all demands.
This decomposition allows demand splitting via fractional block combinations.

Professor's columns are blocks like:
- Block (2, 5): production at t=2 serves demands 2,3,4,5
- Block (3, 3): production at t=3 serves demand 3 only

The master combines blocks to cover each demand, enabling fractional coverage
that matches the compact LP bound.
"""

import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional
import math
import heapq

EPS = 1e-6


def build_instance():
    return {
        "T": 10,
        "demand": [0, 0, 68, 49, 66, 38, 17, 17, 41, 43],
        "h": [
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
        ],
        "c_var": [
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
        ],
        "setup": [
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
        ],
        "shelf_seq": [24, 18, 22, 6, 16, 9, 21, 23, 9, 7],
        "cap": [0, 0, 80, 80, 80, 80, 80, 80, 100, 80],
    }


def holding_cost(h, t, u):
    if u <= t:
        return 0.0
    return sum(h[r] for r in range(t, u))


def arc_flow_cost(inst, t, u):
    """Cost per unit flow on arc (t,u)."""
    return inst["c_var"][t] + holding_cost(inst["h"], t, u)


@dataclass
class Block:
    """
    A production block: production at t can serve demands from t to e.

    - t: production period
    - e: end period (last demand that can be served)
    - coverage[u] = d[u] for each u in [t, e] that this block covers
    - cost = setup + flow costs
    """

    t: int
    e: int
    setup_cost: float
    flow_cost: float  # cost per unit total flow
    x: float  # total production

    @property
    def cost(self):
        return self.setup_cost + self.flow_cost * self.x


def enumerate_blocks(inst: Dict) -> List[Tuple[int, int, float]]:
    """
    Enumerate all valid blocks (t, e) with their setup costs.

    A block (t, e) means:
    - Production at period t
    - Can serve demands in periods [t, e]
    - Cost = setup[t] (setup cost only, flow costs added in master)
    """
    T = inst["T"]
    demand = inst["demand"]
    setup = inst["setup"]
    cap = inst["cap"]
    shelf = inst["shelf_seq"]

    demand_periods = [u for u in range(T) if demand[u] > 0]
    prod_periods = [t for t in range(T) if cap[t] > 0]

    blocks = []
    for t in prod_periods:
        L = shelf[t]
        max_e = min(T - 1, t + L - 1)

        # Block ends must be at demand periods
        for e in range(t, max_e + 1):
            if any(u >= t and u <= e and demand[u] > 0 for u in range(t, e + 1)):
                blocks.append((t, e, setup[t]))

    return blocks


def solve_block_master(
    inst: Dict,
    blocks: List[Tuple[int, int, float]],
    forced_y: Set[int],
    forbidden_y: Set[int],
    forced_z: Set[Tuple[int, int]],
    forbidden_z: Set[Tuple[int, int]],
):
    """
    Solve master problem with block variables.

    Variables:
    - λ[t,e] >= 0: extent of block (t,e) used
    - f[t,u] >= 0: flow on arc (t,u)

    Constraints:
    - Demand: Σ f[t,u] = d[u]
    - Capacity: Σ f[t,u] <= cap[t]
    - Block-flow linking: f[t,u] <= d[u] * Σ_{e >= u} λ[t,e]
      (flow on arc (t,u) only if some block from t covering u is active)
    - Convex combination of setups: Σ_{e} λ[t,e] = y[t]
    - Forced/forbidden y constraints
    """
    T = inst["T"]
    demand = inst["demand"]
    cap = inst["cap"]
    shelf = inst["shelf_seq"]
    setup = inst["setup"]

    demand_periods = [u for u in range(T) if demand[u] > 0]
    prod_periods = [t for t in range(T) if cap[t] > 0]

    # Valid arcs
    arcs = []
    for t in prod_periods:
        L = shelf[t]
        for u in demand_periods:
            if t <= u <= t + L - 1:
                if (t, u) not in forbidden_z:
                    arcs.append((t, u))

    # Filter blocks
    valid_blocks = []
    for t, e, s in blocks:
        if t in forbidden_y:
            continue
        # Check if any arc in this block is forbidden
        block_ok = True
        for u in demand_periods:
            if t <= u <= e and (t, u) in forbidden_z:
                block_ok = False
                break
        if block_ok:
            valid_blocks.append((t, e, s))

    m = gp.Model()
    m.Params.OutputFlag = 0

    # Block variables
    lam = {
        (t, e): m.addVar(lb=0.0, ub=1.0, name=f"lam_{t}_{e}")
        for t, e, s in valid_blocks
    }

    # Flow variables
    f = {arc: m.addVar(lb=0.0, name=f"f_{arc}") for arc in arcs}

    # Aggregate setup variables (auxiliary)
    y = {t: m.addVar(lb=0.0, ub=1.0, name=f"y_{t}") for t in prod_periods}

    # Y = sum of blocks starting at t
    for t in prod_periods:
        blocks_at_t = [lam[t, e] for (tt, e) in lam if tt == t]
        if blocks_at_t:
            m.addConstr(y[t] == gp.quicksum(blocks_at_t), f"y_def_{t}")
        else:
            m.addConstr(y[t] == 0, f"y_def_{t}")

    # Demand satisfaction
    for u in demand_periods:
        m.addConstr(
            gp.quicksum(f[t, u] for t in prod_periods if (t, u) in f) == demand[u],
            f"demand_{u}",
        )

    # Capacity
    cap_con = {}
    for t in prod_periods:
        cap_con[t] = m.addConstr(
            gp.quicksum(f[t, u] for u in demand_periods if (t, u) in f) <= cap[t],
            f"cap_{t}",
        )

    # Block-flow linking: f[t,u] <= d[u] * Σ_{e >= u} λ[t,e]
    for t, u in arcs:
        covering_blocks = [lam[tt, e] for (tt, e) in lam if tt == t and e >= u]
        if covering_blocks:
            m.addConstr(
                f[t, u] <= demand[u] * gp.quicksum(covering_blocks), f"link_{t}_{u}"
            )
        else:
            m.addConstr(f[t, u] == 0, f"link_{t}_{u}")

    # Forced Y
    y_con = {}
    for t in forced_y:
        if t in y:
            y_con[t] = m.addConstr(y[t] == 1.0, f"force_y_{t}")

    # Forced Z: Σ_{e >= u} λ[t,e] >= 1 for arc (t,u)
    z_con = {}
    for t, u in forced_z:
        covering = [lam[tt, e] for (tt, e) in lam if tt == t and e >= u]
        if covering:
            z_con[(t, u)] = m.addConstr(
                gp.quicksum(covering) >= 1.0, f"force_z_{t}_{u}"
            )

    # Objective: setup costs + flow costs
    obj = gp.quicksum(setup[t] * y[t] for t in prod_periods)
    for t, u in arcs:
        obj += arc_flow_cost(inst, t, u) * f[t, u]
    m.setObjective(obj, GRB.MINIMIZE)

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return math.inf, {"status": "infeasible"}

    # Extract solution
    y_vals = {t: y[t].X for t in y}
    f_vals = {arc: f[arc].X for arc in f if f[arc].X > EPS}
    lam_vals = {k: lam[k].X for k in lam if lam[k].X > EPS}

    x_agg = [0.0] * T
    for (t, u), val in f_vals.items():
        x_agg[t] += val

    return m.ObjVal, {
        "status": "optimal",
        "y": y_vals,
        "f": f_vals,
        "lam": lam_vals,
        "x_agg": x_agg,
    }


def solve_branch_and_price(inst, max_nodes=2000, verbose=True):
    """Branch on Y."""
    T = inst["T"]

    blocks = enumerate_blocks(inst)
    print(f"Enumerated {len(blocks)} blocks")

    # Root LP
    root_obj, root_info = solve_block_master(inst, blocks, set(), set(), set(), set())

    if not math.isfinite(root_obj):
        print("Root infeasible")
        return None

    y_vals = root_info["y"]

    print(f"\nROOT LP = {root_obj:.4f}")
    print(f"  Y: {[(t, round(v,3)) for t, v in y_vals.items() if v > EPS]}")

    def frac_y(y_vals):
        return [(t, v) for t, v in y_vals.items() if EPS < v < 1 - EPS]

    fy = frac_y(y_vals)

    if not fy:
        print("Integer at root!")
        return {"obj": root_obj, "info": root_info}

    best_ub = math.inf
    best_sol = None

    queue = [(root_obj, 0, frozenset(), frozenset())]
    heapq.heapify(queue)
    node_id = 0
    explored = 0

    print(f"\n{'='*60}")
    print("BRANCH AND PRICE ON Y")
    print(f"{'='*60}")

    while queue and explored < max_nodes:
        lb, nid, forced_y, forbidden_y = heapq.heappop(queue)

        if lb >= best_ub - EPS:
            continue

        explored += 1

        obj, info = solve_block_master(
            inst, blocks, set(forced_y), set(forbidden_y), set(), set()
        )

        if not math.isfinite(obj):
            if verbose:
                print(f"[{explored}] Node {nid}: INFEASIBLE")
            continue

        if obj >= best_ub - EPS:
            if verbose:
                print(f"[{explored}] Node {nid}: PRUNED")
            continue

        y_vals = info["y"]
        fy_list = frac_y(y_vals)

        if not fy_list:
            if obj < best_ub:
                best_ub = obj
                best_sol = {"obj": obj, "info": info}
                print(f"[{explored}] Node {nid}: ★ INCUMBENT = {obj:.4f}")
                print(f"    Y = {[t for t, v in y_vals.items() if v > 0.5]}")
            continue

        # Branch
        t_br, v_br = max(fy_list, key=lambda x: min(x[1], 1 - x[1]))
        if verbose:
            print(f"[{explored}] Node {nid}: LB={obj:.2f}, branch Y[{t_br}]={v_br:.3f}")

        node_id += 1
        heapq.heappush(queue, (obj, node_id, forced_y, forbidden_y | {t_br}))
        node_id += 1
        heapq.heappush(queue, (obj, node_id, forced_y | {t_br}, forbidden_y))

    print(f"\n{'='*60}")
    print(f"Explored {explored} nodes")
    print(f"Best: {best_ub:.4f}" if best_ub < math.inf else "No solution")

    return best_sol


def print_solution(sol, inst):
    if sol is None:
        print("No solution")
        return

    T = inst["T"]
    info = sol["info"]

    print(f"\n{'='*60}")
    print(f"SOLUTION: {sol['obj']:.4f}")
    print(f"{'='*60}")

    print(f"\nY: {[(t, round(v,3)) for t, v in info['y'].items() if v > 0.5]}")
    print(f"X: {[(t, round(v,2)) for t, v in enumerate(info['x_agg']) if v > EPS]}")

    print(f"\nFlows:")
    for (t, u), val in sorted(info["f"].items()):
        print(f"  f[{t},{u}] = {val:.2f}")

    if "lam" in info:
        print(f"\nActive blocks:")
        for (t, e), val in sorted(info["lam"].items()):
            print(f"  λ[{t},{e}] = {val:.4f}")


if __name__ == "__main__":
    inst = build_instance()

    print("=" * 60)
    print("PURE BRANCH-AND-PRICE (NO MIP)")
    print("=" * 60)
    print(f"T={inst['T']}, demand={sum(inst['demand'])}")

    sol = solve_branch_and_price(inst, max_nodes=2000, verbose=True)
    print_solution(sol, inst)
