"""
Block-Based Branch-and-Price for Multi-Item Capacitated Lot-Sizing.

Each "column" is a block (t, e): production at period t serving demands up to e.
The master problem combines blocks via flow variables to cover all demands.
Branching is done on setup variables Y[item, t].

Features:
- Column generation with DP pricing
- CSV logging of nodes and CG iterations
- Proper gap calculation (0% when proven optimal)
"""

from __future__ import annotations
import csv
import json
import math
import time
import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import gurobipy as gp
from gurobipy import GRB

EPS = 1e-6
ADD_ALL_LEFO_CUTS_UPFRONT = False  # Add all LEFO cuts at root (exact)
MAX_LEFO_CUTS_PER_ITER = 10  # max cuts per CG iter if lazy cutting (still will add all neccesary cuts but slower)

# Safe column pruning settings
ENABLE_COLUMN_PRUNING = True
PRUNE_AFTER_UNUSED_ITERS = 15  # Only prune if unused for this many consecutive CG iters
MIN_COLUMNS_PER_ITEM = 10  # Always keep at least this many columns per item


@dataclass
class BranchNode:
    """Node in the branch-and-bound tree."""

    node_id: int
    parent_id: Optional[int]
    depth: int
    forced_y: Dict[int, Set[int]] = field(default_factory=dict)  # item -> {periods}
    forbidden_y: Dict[int, Set[int]] = field(default_factory=dict)  # item -> {periods}
    # LEFO cuts accumulated at this node: (item_id, t1, u, t2, up) means Z[t1,u] + Z[t2,up] <= 1
    lefo_cuts: List[Tuple[int, int, int, int, int]] = field(default_factory=list)
    # X bounds for production quantities: (item_id, t) -> (lower, upper)
    x_bounds: Dict[Tuple[int, int], Tuple[float, float]] = field(default_factory=dict)
    lp_bound: float = math.inf

    def __lt__(self, other: "BranchNode") -> bool:
        return self.lp_bound < other.lp_bound


@dataclass
class NodeLogEntry:
    """Log entry for a B&P node."""

    node_id: int
    depth: int
    lp_bound: float
    incumbent: float
    branch_item: Optional[int]
    branch_t: Optional[int]
    direction: Optional[str]
    status: str
    cg_iters: int
    columns_added: int


@dataclass
class CGIterLogEntry:
    """Log entry for a column generation iteration."""

    node_id: int
    cg_iter: int
    rmp_obj: float
    num_columns: int
    min_rc: float
    columns_added: int


# =============================================================================
# CSV LOGGING
# =============================================================================


class BnPLogger:
    """Logger for Branch-and-Price solver."""

    def __init__(self, out_dir: Path, enabled: bool = True):
        self.enabled = enabled
        self.out_dir = out_dir
        self.node_entries: List[NodeLogEntry] = []
        self.cg_entries: List[CGIterLogEntry] = []

    def log_node(self, entry: NodeLogEntry):
        if self.enabled:
            self.node_entries.append(entry)

    def log_cg_iter(self, entry: CGIterLogEntry):
        if self.enabled:
            self.cg_entries.append(entry)

    def write_csvs(self):
        if not self.enabled:
            return

        # Write node log
        node_path = self.out_dir / "bnp_nodes.csv"
        with open(node_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "node_id",
                    "depth",
                    "lp_bound",
                    "incumbent",
                    "branch_item",
                    "branch_t",
                    "direction",
                    "status",
                    "cg_iters",
                    "columns_added",
                ]
            )
            for e in self.node_entries:
                writer.writerow(
                    [
                        e.node_id,
                        e.depth,
                        f"{e.lp_bound:.6f}" if math.isfinite(e.lp_bound) else "inf",
                        f"{e.incumbent:.6f}" if math.isfinite(e.incumbent) else "inf",
                        e.branch_item if e.branch_item is not None else "",
                        e.branch_t if e.branch_t is not None else "",
                        e.direction if e.direction else "",
                        e.status,
                        e.cg_iters,
                        e.columns_added,
                    ]
                )

        # Write CG iteration log
        cg_path = self.out_dir / "bnp_cg_iters.csv"
        with open(cg_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "node_id",
                    "cg_iter",
                    "rmp_obj",
                    "num_columns",
                    "min_rc",
                    "columns_added",
                ]
            )
            for e in self.cg_entries:
                writer.writerow(
                    [
                        e.node_id,
                        e.cg_iter,
                        f"{e.rmp_obj:.6f}",
                        e.num_columns,
                        f"{e.min_rc:.6f}",
                        e.columns_added,
                    ]
                )


# =============================================================================
# BLOCK ENUMERATION AND COSTS
# =============================================================================


def list_blocks_for_item(
    item_id: int,
    item_data: dict,
    T: int,
    Gamma: Dict[int, List[int]],
) -> List[Tuple[int, int, float]]:
    """
    Enumerate all valid blocks (t, e) for an item.
    A block (t, e) means production at t can serve demands in [t, e].
    Returns list of (t, e, setup_cost).
    """
    demand = item_data["demand"]
    setup = item_data["setup"]

    def setup_at(t: int) -> float:
        if isinstance(setup, list):
            return float(setup[t]) if t < len(setup) else 0.0
        return float(setup)

    blocks = []
    for t in range(T):
        reachable = Gamma.get(t, [])
        if not reachable:
            continue

        max_e = max(reachable)

        for e in range(t, max_e + 1):
            has_demand = any(u in reachable and demand[u] > 0 for u in range(t, e + 1))
            if has_demand:
                blocks.append((t, e, setup_at(t)))

    return blocks


def arc_cost_for_item(item_data: dict, t: int, u: int) -> float:
    """Cost per unit flow on arc (t, u): variable cost + holding cost."""
    c_var = item_data["c_var"]
    h = item_data.get("h", [0.0])

    if isinstance(c_var, list):
        var_cost = float(c_var[t]) if t < len(c_var) else 0.0
    else:
        var_cost = float(c_var)

    hold_cost = 0.0
    if u > t:
        for r in range(t, u):
            if isinstance(h, list):
                hold_cost += float(h[r]) if r < len(h) else 0.0
            else:
                hold_cost += float(h)

    return var_cost + hold_cost


def generate_minimal_initial_columns(
    item_id: int,
    item_data: dict,
    T: int,
    Gamma: Dict[int, List[int]],
) -> List[Tuple[int, int, float]]:
    """
    Generate minimal initial columns for feasibility.

    Creates just enough blocks to cover all demands - one block per demand period
    using the earliest feasible production period.
    """
    demand = item_data["demand"]
    setup = item_data["setup"]

    def setup_at(t: int) -> float:
        if isinstance(setup, list):
            return float(setup[t]) if t < len(setup) else 0.0
        return float(setup)

    demand_periods = [u for u in range(T) if demand[u] > 0]
    if not demand_periods:
        return []

    # Build reverse mapping: for each demand period u, which production periods can reach it?
    can_serve = {u: [] for u in demand_periods}
    for t in range(T):
        reachable = Gamma.get(t, [])
        for u in reachable:
            if u in can_serve:
                can_serve[u].append(t)

    # Greedily select blocks to cover all demands
    initial_blocks = set()
    for u in demand_periods:
        if not can_serve[u]:
            continue
        # Pick the production period that can reach this demand
        # Prefer periods that can also reach later demands (larger blocks)
        best_t = None
        best_e = -1
        for t in can_serve[u]:
            reachable = Gamma.get(t, [])
            max_e = (
                max(r for r in reachable if r in demand_periods)
                if any(r in demand_periods for r in reachable)
                else t
            )
            if max_e > best_e:
                best_e = max_e
                best_t = t

        if best_t is not None:
            initial_blocks.add((best_t, best_e, setup_at(best_t)))

    return list(initial_blocks)


# =============================================================================
# COLUMN GENERATION - RMP WITH DUALS
# =============================================================================


BIG_M_ARTIFICIAL = 1e6  # Penalty for artificial variables


def solve_rmp_with_duals(
    items: Dict[int, dict],
    T: int,
    capacity: List[float],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    columns_by_item: Dict[int, List[Tuple[int, int, float]]],
    forced_y: Dict[int, Set[int]],
    forbidden_y: Dict[int, Set[int]],
    lefo_cuts: List[Tuple[int, int, int, int, int]],
    x_bounds: Dict[Tuple[int, int], Tuple[float, float]] = None,
    use_artificial: bool = True,
) -> Tuple[float, Dict, Dict]:
    """
    Solve RMP and return objective, solution, and dual values.

    Uses artificial variables to ensure feasibility during column generation.

    Args:
        x_bounds: Optional dict of (item_id, t) -> (lower, upper) bounds on production.

    Returns:
        (obj, solution_dict, duals_dict)

    duals_dict contains:
        - pi_demand[(item_id, u)]: dual for demand constraint
        - rho_cap[t]: dual for capacity constraint
        - sigma_link[(item_id, t, u)]: dual for linking constraint
        - mu_y_def[(item_id, t)]: dual for Y definition constraint
        - tau_force_y[(item_id, t)]: dual for forced Y constraint
        - pi_lefo[idx]: dual for LEFO cut constraint
    """
    m = gp.Model("RMP")
    m.Params.OutputFlag = 0
    m.Params.LogToConsole = 0
    m.Params.Method = 1

    demand_periods_by_item = {}
    for item_id, item_data in items.items():
        demand = item_data["demand"]
        demand_periods_by_item[item_id] = [u for u in range(T) if demand[u] > 0]

    # Filter blocks by forbidden Y
    valid_columns_by_item = {}
    for item_id, cols in columns_by_item.items():
        forbidden_periods = forbidden_y.get(item_id, set())
        valid_cols = []
        for t, e, s in cols:
            if t in forbidden_periods:
                continue
            valid_cols.append((t, e, s))
        valid_columns_by_item[item_id] = valid_cols

    # Block variables λ[i,t,e]
    lam = {}
    for item_id, cols in valid_columns_by_item.items():
        for t, e, s in cols:
            key = (item_id, t, e)
            if key not in lam:
                lam[key] = m.addVar(lb=0.0, ub=1.0, name=f"lam_{item_id}_{t}_{e}")

    # Flow variables f[i,t,u]
    f = {}
    for item_id, item_data in items.items():
        demand = item_data["demand"]
        Gamma = Gamma_by_item.get(item_id, {})
        forbidden_periods = forbidden_y.get(item_id, set())

        for t in range(T):
            if t in forbidden_periods:
                continue
            reachable = Gamma.get(t, [])
            for u in reachable:
                if demand[u] > 0:
                    f[(item_id, t, u)] = m.addVar(lb=0.0, name=f"f_{item_id}_{t}_{u}")

    # Aggregate setup variables y[i,t]
    y = {}
    for item_id in items:
        for t in range(T):
            y[(item_id, t)] = m.addVar(lb=0.0, ub=1.0, name=f"y_{item_id}_{t}")

    # Aggregate arc usage variables z[i,t,u] (indicator for flow on arc)
    z = {}
    for item_id, t, u in f:
        z[(item_id, t, u)] = m.addVar(lb=0.0, ub=1.0, name=f"z_{item_id}_{t}_{u}")

    m.update()

    # Constraints with names for dual extraction
    y_def_con = {}
    z_def_con = {}
    demand_con = {}
    cap_con = {}
    link_con = {}
    force_y_con = {}
    lefo_cut_con = {}

    # Y definition: y[i,t] = Σ_e λ[i,t,e]
    for item_id in items:
        for t in range(T):
            blocks_at_t = [
                lam[(i, tt, e)] for (i, tt, e) in lam if i == item_id and tt == t
            ]
            if blocks_at_t:
                y_def_con[(item_id, t)] = m.addConstr(
                    y[(item_id, t)] == gp.quicksum(blocks_at_t), f"y_def_{item_id}_{t}"
                )
            else:
                y_def_con[(item_id, t)] = m.addConstr(
                    y[(item_id, t)] == 0.0, f"y_def_{item_id}_{t}"
                )

    # Artificial variables for demand satisfaction (to ensure initial feasibility)
    art = {}
    if use_artificial:
        for item_id, item_data in items.items():
            for u in demand_periods_by_item[item_id]:
                art[(item_id, u)] = m.addVar(lb=0.0, name=f"art_{item_id}_{u}")
        m.update()

    # Demand satisfaction: Σ_t f[i,t,u] + art[i,u] = d[i,u]
    for item_id, item_data in items.items():
        demand = item_data["demand"]
        for u in demand_periods_by_item[item_id]:
            flow_to_u = [f[(i, t, uu)] for (i, t, uu) in f if i == item_id and uu == u]
            lhs = gp.quicksum(flow_to_u) if flow_to_u else gp.LinExpr()
            if use_artificial and (item_id, u) in art:
                lhs += art[(item_id, u)]
            demand_con[(item_id, u)] = m.addConstr(
                lhs == demand[u], f"demand_{item_id}_{u}"
            )

    # Capacity: Σ_i Σ_u f[i,t,u] <= cap[t]
    for t in range(T):
        flow_at_t = [f[(i, tt, u)] for (i, tt, u) in f if tt == t]
        if flow_at_t:
            cap_con[t] = m.addConstr(gp.quicksum(flow_at_t) <= capacity[t], f"cap_{t}")

    # Block-flow linking: f[i,t,u] <= d[i,u] * Σ_{e>=u} λ[i,t,e]
    for (item_id, t, u), f_var in f.items():
        demand = items[item_id]["demand"][u]
        covering_blocks = [
            lam[(i, tt, e)] for (i, tt, e) in lam if i == item_id and tt == t and e >= u
        ]
        if covering_blocks:
            link_con[(item_id, t, u)] = m.addConstr(
                f_var <= demand * gp.quicksum(covering_blocks),
                f"link_{item_id}_{t}_{u}",
            )
        else:
            link_con[(item_id, t, u)] = m.addConstr(
                f_var == 0.0, f"link_{item_id}_{t}_{u}"
            )

    # Z definition: z[i,t,u] >= f[i,t,u] / d[i,u] (z=1 if any flow on arc)
    for (item_id, t, u), z_var in z.items():
        demand_u = items[item_id]["demand"][u]
        if demand_u > 0 and (item_id, t, u) in f:
            z_def_con[(item_id, t, u)] = m.addConstr(
                z_var >= f[(item_id, t, u)] / demand_u, f"z_def_{item_id}_{t}_{u}"
            )

    # Forced Y: y[i,t] = 1
    for item_id, periods in forced_y.items():
        for t in periods:
            if (item_id, t) in y:
                force_y_con[(item_id, t)] = m.addConstr(
                    y[(item_id, t)] == 1.0, f"force_y_{item_id}_{t}"
                )

    # LEFO cuts: Z[t1,u] + Z[t2,up] <= 1 for crossing pairs
    for idx, (item_id, t1, u, t2, up) in enumerate(lefo_cuts):
        if (item_id, t1, u) in z and (item_id, t2, up) in z:
            lefo_cut_con[idx] = m.addConstr(
                z[(item_id, t1, u)] + z[(item_id, t2, up)] <= 1,
                f"lefo_cut_{idx}",
            )

    # X bounds: production quantity bounds from branching
    if x_bounds:
        for (item_id, t), (lb, ub) in x_bounds.items():
            # Total production at (item, t) = Σ_u f[item, t, u]
            flow_from_t = [
                f[(i, tt, u)] for (i, tt, u) in f if i == item_id and tt == t
            ]
            if flow_from_t:
                total_prod = gp.quicksum(flow_from_t)
                if lb > 0:
                    m.addConstr(total_prod >= lb, f"x_lb_{item_id}_{t}")
                if ub < math.inf:
                    m.addConstr(total_prod <= ub, f"x_ub_{item_id}_{t}")

    # Objective: setup costs + flow costs + artificial penalties
    obj = gp.LinExpr()
    for item_id, item_data in items.items():
        setup = item_data["setup"]
        for t in range(T):
            if isinstance(setup, list):
                s_cost = float(setup[t]) if t < len(setup) else 0.0
            else:
                s_cost = float(setup)
            obj += s_cost * y[(item_id, t)]

    for (item_id, t, u), f_var in f.items():
        item_data = items[item_id]
        arc_cost = arc_cost_for_item(item_data, t, u)
        obj += arc_cost * f_var

    # Penalty for artificial variables
    if use_artificial:
        for art_var in art.values():
            obj += BIG_M_ARTIFICIAL * art_var

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return math.inf, {"status": "infeasible"}, {}

    # Check if artificial variables are used (indicates true infeasibility)
    art_total = 0.0
    if use_artificial:
        for art_var in art.values():
            art_total += art_var.X

    # Extract solution
    y_vals = {}
    for (item_id, t), y_var in y.items():
        val = y_var.X
        if val > EPS:
            y_vals[(item_id, t)] = val

    f_vals = {}
    for key, f_var in f.items():
        val = f_var.X
        if val > EPS:
            f_vals[key] = val

    lam_vals = {}
    for key, lam_var in lam.items():
        val = lam_var.X
        if val > EPS:
            lam_vals[key] = val

    x_agg = {item_id: {} for item_id in items}
    for (item_id, t, u), val in f_vals.items():
        x_agg[item_id][t] = x_agg[item_id].get(t, 0.0) + val

    # Extract z values
    z_vals = {}
    for key, z_var in z.items():
        val = z_var.X
        if val > EPS:
            z_vals[key] = val

    solution = {
        "status": "optimal",
        "y": y_vals,
        "z": z_vals,
        "f": f_vals,
        "lam": lam_vals,
        "x_agg": x_agg,
        "art_total": art_total,
    }

    # Extract duals
    duals = {
        "pi_demand": {},
        "rho_cap": {},
        "sigma_link": {},
        "mu_y_def": {},
        "mu_z_def": {},
        "tau_force_y": {},
        "pi_lefo": {},
    }

    for key, con in demand_con.items():
        duals["pi_demand"][key] = con.Pi

    for key, con in cap_con.items():
        duals["rho_cap"][key] = con.Pi

    for key, con in link_con.items():
        duals["sigma_link"][key] = con.Pi

    for key, con in y_def_con.items():
        duals["mu_y_def"][key] = con.Pi

    for key, con in z_def_con.items():
        duals["mu_z_def"][key] = con.Pi

    for key, con in force_y_con.items():
        duals["tau_force_y"][key] = con.Pi

    for key, con in lefo_cut_con.items():
        duals["pi_lefo"][key] = con.Pi

    return m.ObjVal, solution, duals


# COLUMN GENERATION - DP PRICING


def price_block_dp(
    item_id: int,
    item_data: dict,
    T: int,
    Gamma: Dict[int, List[int]],
    duals: Dict,
    forced_y: Set[int],
    forbidden_y: Set[int],
    existing_blocks: Set[Tuple[int, int, float]],
    max_columns_per_item: int = 5,
) -> Tuple[List[Tuple[Tuple[int, int, float], float]], float]:
    """
    DP pricing to find negative reduced cost blocks for an item.

    NOTE: Column generation is INDEPENDENT of LEFO cuts. We generate all
    cost-effective columns; the RMP's LEFO constraints select which to use.

    Reduced cost for block (t, e):
        rc = setup[t] - μ_y_def[(item_id, t)]
             - Σ_{u in [t,e]} σ_link[(item_id, t, u)] * d[u]
             - τ_force_y[(item_id, t)]  (if Y[i,t]=1 forced)

    Returns:
        - list of top max_columns_per_item ((t, e, setup_cost), reduced_cost) with rc < -EPS
        - true_min_rc: the minimum reduced cost across ALL negative RC blocks found
    """
    demand = item_data["demand"]
    setup = item_data["setup"]

    def setup_at(t: int) -> float:
        if isinstance(setup, list):
            return float(setup[t]) if t < len(setup) else 0.0
        return float(setup)

    sigma_link = duals.get("sigma_link", {})
    mu_y_def = duals.get("mu_y_def", {})
    tau_force_y = duals.get("tau_force_y", {})
    # NOTE: LEFO duals (pi_lefo) are NOT used in pricing.
    # Column generation is independent of LEFO cuts - we generate all cost-effective
    # columns, and the RMP's LEFO constraints will select which ones to use.

    negative_rc_blocks = []

    for t in range(T):
        if t in forbidden_y:
            continue

        reachable = Gamma.get(t, [])
        if not reachable:
            continue

        max_e = max(reachable)

        for e in range(t, max_e + 1):
            # Check if block has any demand
            has_demand = any(u in reachable and demand[u] > 0 for u in range(t, e + 1))
            if not has_demand:
                continue

            s_cost = setup_at(t)
            block = (t, e, s_cost)

            # Skip if already in RMP
            if block in existing_blocks:
                continue

            # Compute reduced cost
            # μ_y_def is the dual of y[i,t] = Σ_e λ[i,t,e] (free dual)
            mu = mu_y_def.get((item_id, t), 0.0)

            # σ_link is the dual of f[t,u] <= d[u] * Σ_{e>=u} λ[t,e]
            # This is <= 0 (constraint has <=)
            # When block (t,e) is added, RHS of link constraints for u <= e increases
            # So contribution is -σ * d[u] for each u in [t, e]
            sigma_contrib = 0.0
            for u in range(t, e + 1):
                if u in reachable and demand[u] > 0:
                    sigma = sigma_link.get((item_id, t, u), 0.0)
                    # σ <= 0 for <= constraint, contribution is -σ * d[u] >= 0
                    sigma_contrib += (-sigma) * demand[u]

            # τ_force_y: dual of forced Y constraint (y[i,t] = 1)
            tau_y = tau_force_y.get((item_id, t), 0.0)

            # RC = setup_cost - μ - σ_contrib - τ_y
            # NOTE: LEFO duals NOT included - CG is independent of LEFO cuts
            rc = s_cost - mu - sigma_contrib - tau_y

            if rc < -EPS:
                # Store with score: (rc, -(e-t)) to prefer larger blocks at same RC
                block_size = e - t
                negative_rc_blocks.append((block, rc, block_size))

    # Sort by (rc, -block_size): most negative RC first, then larger blocks
    negative_rc_blocks.sort(key=lambda x: (x[1], -x[2]))

    # True minimum RC across ALL negative RC blocks
    true_min_rc = negative_rc_blocks[0][1] if negative_rc_blocks else 0.0

    # Return top k columns (block, rc) and the true minimum RC
    top_blocks = [
        (block, rc) for block, rc, _ in negative_rc_blocks[:max_columns_per_item]
    ]
    return top_blocks, true_min_rc


def price_all_items(
    items: Dict[int, dict],
    T: int,
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    duals: Dict,
    forced_y: Dict[int, Set[int]],
    forbidden_y: Dict[int, Set[int]],
    columns_by_item: Dict[int, List[Tuple[int, int, float]]],
    max_columns_per_item: int = 20,
) -> Tuple[Dict[int, List[Tuple[Tuple[int, int, float], float]]], float]:
    """
    Price blocks for all items, returning top max_columns_per_item per item.

    NOTE: Column generation is INDEPENDENT of LEFO cuts. We generate all
    cost-effective columns; the RMP's LEFO constraints select which to use.

    Returns:
        - result: dict mapping item_id -> list of (block, rc) tuples
        - true_min_rc: minimum reduced cost across ALL items and ALL negative RC blocks
    """
    result = {}
    true_min_rc = 0.0
    for item_id, item_data in items.items():
        Gamma = Gamma_by_item.get(item_id, {})
        forced_periods = forced_y.get(item_id, set())
        forbidden_periods = forbidden_y.get(item_id, set())
        existing = set(columns_by_item.get(item_id, []))
        cols, item_min_rc = price_block_dp(
            item_id,
            item_data,
            T,
            Gamma,
            duals,
            forced_periods,
            forbidden_periods,
            existing,
            max_columns_per_item,
        )
        result[item_id] = cols
        if item_min_rc < true_min_rc:
            true_min_rc = item_min_rc
    return result, true_min_rc


# CG


def column_generation_loop(
    items: Dict[int, dict],
    T: int,
    capacity: List[float],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    initial_columns: Dict[int, List[Tuple[int, int, float]]],
    forced_y: Dict[int, Set[int]],
    forbidden_y: Dict[int, Set[int]],
    lefo_cuts: List[Tuple[int, int, int, int, int]],
    x_bounds: Dict[Tuple[int, int], Tuple[float, float]],
    node_id: int,
    logger: Optional[BnPLogger],
    max_cg_iters: int = 20000,
    verbose: bool = False,
) -> Tuple[float, Dict, Dict[int, List[Tuple[int, int, float]]], int, int]:
    """
    Run column generation loop.

    Returns:
        (obj, solution, columns_by_item, total_cg_iters, total_columns_added)
    """
    columns_by_item = {i: list(cols) for i, cols in initial_columns.items()}
    total_columns_added = 0
    cg_iter = 0

    # Track consecutive iterations with zero/positive reduced cost for each column
    # (item_id, block) -> count of consecutive non-negative RC iterations
    column_zero_rc_count: Dict[Tuple[int, Tuple[int, int, float]], int] = {}
    for item_id, cols in columns_by_item.items():
        for block in cols:
            column_zero_rc_count[(item_id, block)] = 0

    while cg_iter < max_cg_iters:
        # Solve RMP
        obj, solution, duals = solve_rmp_with_duals(
            items,
            T,
            capacity,
            Gamma_by_item,
            columns_by_item,
            forced_y,
            forbidden_y,
            lefo_cuts,
            x_bounds,
        )

        if not math.isfinite(obj):
            return math.inf, {"status": "infeasible"}, columns_by_item, cg_iter, 0

        # Price new columns (returns top 5 per item + true min RC across all)
        # NOTE: Pricing is INDEPENDENT of LEFO cuts - we generate all cost-effective columns
        new_columns_by_item, min_rc = price_all_items(
            items,
            T,
            Gamma_by_item,
            duals,
            forced_y,
            forbidden_y,
            columns_by_item,
        )

        # Add new columns and track their reduced costs
        columns_added_this_iter = 0

        # First, update RC tracking for existing columns based on pricing results
        # Columns that were priced and have negative RC: reset counter
        # Columns not found in pricing (RC >= 0): increment counter
        priced_blocks_this_iter: Dict[int, Set[Tuple[int, int, float]]] = {
            i: set() for i in items
        }
        for item_id, new_cols in new_columns_by_item.items():
            for block, rc in new_cols:
                priced_blocks_this_iter[item_id].add(block)
                # This column has negative RC, reset its zero-RC counter
                if (item_id, block) in column_zero_rc_count:
                    column_zero_rc_count[(item_id, block)] = 0

        # Increment counter for columns NOT in pricing results (they have RC >= 0)
        for item_id, cols in columns_by_item.items():
            for block in cols:
                if block not in priced_blocks_this_iter.get(item_id, set()):
                    column_zero_rc_count[(item_id, block)] = (
                        column_zero_rc_count.get((item_id, block), 0) + 1
                    )

        # Add new columns
        for item_id, new_cols in new_columns_by_item.items():
            existing = set(columns_by_item.get(item_id, []))
            for block, rc in new_cols:
                if block not in existing:
                    columns_by_item[item_id].append(block)
                    column_zero_rc_count[(item_id, block)] = (
                        0  # New column starts fresh
                    )
                    columns_added_this_iter += 1

        # Count total columns
        total_cols = sum(len(cols) for cols in columns_by_item.values())

        # Safe column pruning: remove columns unused for many iterations
        # Key safety: NEVER prune active columns (those with λ > 0 in current solution)
        if ENABLE_COLUMN_PRUNING and cg_iter > 0:
            # Get active columns from current solution
            lam_vals = solution.get("lam", {}) if isinstance(solution, dict) else {}
            active_columns: Dict[int, Set[Tuple[int, int, float]]] = {
                i: set() for i in items
            }
            for (item_id, t, e), val in lam_vals.items():
                if val > EPS:
                    # Find matching block in pool
                    for block in columns_by_item.get(item_id, []):
                        if block[0] == t and block[1] == e:
                            active_columns[item_id].add(block)
                            break

            # Prune inactive columns that have been unused for too long
            for item_id in columns_by_item:
                if len(columns_by_item[item_id]) <= MIN_COLUMNS_PER_ITEM:
                    continue  # Keep minimum

                columns_to_keep = []
                for col in columns_by_item[item_id]:
                    # NEVER prune active columns
                    if col in active_columns[item_id]:
                        columns_to_keep.append(col)
                        column_zero_rc_count[(item_id, col)] = 0  # Reset counter
                        continue

                    # Check if unused for too long
                    unused_count = column_zero_rc_count.get((item_id, col), 0)
                    if unused_count < PRUNE_AFTER_UNUSED_ITERS:
                        columns_to_keep.append(col)
                    # else: prune this column (don't add to keep list)

                # Ensure we keep at least MIN_COLUMNS_PER_ITEM
                if len(columns_to_keep) >= MIN_COLUMNS_PER_ITEM:
                    pruned_count = len(columns_by_item[item_id]) - len(columns_to_keep)
                    if pruned_count > 0:
                        # Clean up tracking for pruned columns
                        old_cols = set(columns_by_item[item_id])
                        new_cols = set(columns_to_keep)
                        for col in old_cols - new_cols:
                            column_zero_rc_count.pop((item_id, col), None)
                        columns_by_item[item_id] = columns_to_keep
                        if verbose:
                            print(
                                f"      Pruned {pruned_count} inactive columns from item {item_id}"
                            )

            # Recount after pruning
            total_cols = sum(len(cols) for cols in columns_by_item.values())

        # Log CG iteration
        if logger:
            logger.log_cg_iter(
                CGIterLogEntry(
                    node_id=node_id,
                    cg_iter=cg_iter,
                    rmp_obj=obj,
                    num_columns=total_cols,
                    min_rc=min_rc,
                    columns_added=columns_added_this_iter,
                )
            )

        if verbose:
            print(
                f"    CG iter {cg_iter}: obj={obj:.4f}, cols={total_cols}, "
                f"min_rc={min_rc:.6f}, added={columns_added_this_iter}"
            )

        total_columns_added += columns_added_this_iter

        # Convergence check
        if columns_added_this_iter == 0 or min_rc >= -EPS:
            cg_iter += 1
            break

        cg_iter += 1

    # Final solve
    obj, solution, _ = solve_rmp_with_duals(
        items,
        T,
        capacity,
        Gamma_by_item,
        columns_by_item,
        forced_y,
        forbidden_y,
        lefo_cuts,
        x_bounds,
    )

    return obj, solution, columns_by_item, cg_iter, total_columns_added


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def is_y_integer(y_vals: Dict[Tuple[int, int], float]) -> bool:
    """Check if all Y values are integer."""
    for val in y_vals.values():
        if EPS < val < 1.0 - EPS:
            return False
    return True


def find_lefo_violation(
    z_vals: Dict[Tuple[int, int, int], float],
    items: Dict[int, dict],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    existing_cuts: List[Tuple[int, int, int, int, int]],
) -> Optional[Tuple[int, int, int, int, int]]:
    """
    Find a LEFO violation: Z[t1,u] + Z[t2,up] > 1 for a crossing pair.

    LEFO constraint: For production periods t1 < t2 with expiry v_t1 < v_t2,
    if t2 can serve u' and t1 can serve u where t2 <= u <= u'-1,
    then Z[t1,u] + Z[t2,u'] <= 1.

    Returns: (item_id, t1, u, t2, up) if violation found, else None.
    """
    existing_set = set(existing_cuts)

    for item_id, item_data in items.items():
        Gamma = Gamma_by_item.get(item_id, {})
        shelf_seq = item_data.get("shelf_seq", [])
        demand = item_data.get("demand", [])
        T = len(demand)

        # Get production periods that have reachable demands
        prods = [t for t in range(T) if Gamma.get(t)]
        if len(prods) < 2:
            continue

        # Compute expiry for each production period: v_t = t + m_t
        expiry = {}
        for t in prods:
            m_t = int(shelf_seq[t]) if t < len(shelf_seq) else 0
            expiry[t] = t + m_t

        # Sort production periods by expiry (ascending)
        prods_sorted = sorted(prods, key=lambda t: expiry[t])

        # Check pairs (t1, t2) where v_t1 < v_t2
        for a in range(len(prods_sorted)):
            t1 = prods_sorted[a]
            v1 = expiry[t1]
            reachable_t1 = set(Gamma.get(t1, []))

            for b in range(a + 1, len(prods_sorted)):
                t2 = prods_sorted[b]
                v2 = expiry[t2]

                if v1 >= v2:
                    continue  # We only care about v1 < v2

                reachable_t2 = Gamma.get(t2, [])

                # For each arc (t2, up) where t2 can serve up
                for up in reachable_t2:
                    if demand[up] <= 0:
                        continue
                    z_t2_up = z_vals.get((item_id, t2, up), 0.0)
                    if z_t2_up < EPS:
                        continue

                    # For each arc (t1, u) where t2 <= u <= up-1 and t1 can serve u
                    for u in range(t2, up):
                        if u not in reachable_t1 or demand[u] <= 0:
                            continue

                        z_t1_u = z_vals.get((item_id, t1, u), 0.0)
                        if z_t1_u < EPS:
                            continue

                        # Check if violation: Z[t1,u] + Z[t2,up] > 1
                        if z_t1_u + z_t2_up > 1.0 + EPS:
                            cut = (item_id, t1, u, t2, up)
                            if cut not in existing_set:
                                return cut

    return None


def find_all_lefo_violations(
    z_vals: Dict[Tuple[int, int, int], float],
    items: Dict[int, dict],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    existing_cuts: List[Tuple[int, int, int, int, int]],
) -> List[Tuple[int, int, int, int, int]]:
    """
    Find ALL LEFO violations: Z[t1,u] + Z[t2,up] > 1 for crossing pairs.

    Returns: List of (item_id, t1, u, t2, up) tuples for all violations.
    """
    existing_set = set(existing_cuts)
    violations = []

    for item_id, item_data in items.items():
        Gamma = Gamma_by_item.get(item_id, {})
        shelf_seq = item_data.get("shelf_seq", [])
        demand = item_data.get("demand", [])
        T = len(demand)

        prods = [t for t in range(T) if Gamma.get(t)]
        if len(prods) < 2:
            continue

        expiry = {}
        for t in prods:
            m_t = int(shelf_seq[t]) if t < len(shelf_seq) else 0
            expiry[t] = t + m_t

        prods_sorted = sorted(prods, key=lambda t: expiry[t])

        for a in range(len(prods_sorted)):
            t1 = prods_sorted[a]
            v1 = expiry[t1]
            reachable_t1 = set(Gamma.get(t1, []))

            for b in range(a + 1, len(prods_sorted)):
                t2 = prods_sorted[b]
                v2 = expiry[t2]

                if v1 >= v2:
                    continue

                reachable_t2 = Gamma.get(t2, [])

                for up in reachable_t2:
                    if demand[up] <= 0:
                        continue
                    z_t2_up = z_vals.get((item_id, t2, up), 0.0)
                    if z_t2_up < EPS:
                        continue

                    for u in range(t2, up):
                        if u not in reachable_t1 or demand[u] <= 0:
                            continue

                        z_t1_u = z_vals.get((item_id, t1, u), 0.0)
                        if z_t1_u < EPS:
                            continue

                        if z_t1_u + z_t2_up > 1.0 + EPS:
                            cut = (item_id, t1, u, t2, up)
                            if cut not in existing_set:
                                violations.append(cut)
                                existing_set.add(cut)

    return violations


def generate_all_lefo_cuts(
    items: Dict[int, dict],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
) -> List[Tuple[int, int, int, int, int]]:
    """
    Generate ALL possible LEFO cuts upfront (same as MIP no-cross constraints).

    For each pair of production periods t1 < t2 with expiry v1 < v2,
    for each arc (t2, up) and (t1, u) where t2 <= u <= up-1,
    add constraint Z[t1,u] + Z[t2,up] <= 1.

    Returns: List of (item_id, t1, u, t2, up) tuples for all LEFO cuts.
    """
    all_cuts = []

    for item_id, item_data in items.items():
        Gamma = Gamma_by_item.get(item_id, {})
        shelf_seq = item_data.get("shelf_seq", [])
        demand = item_data.get("demand", [])
        T = len(demand)

        prods = [t for t in range(T) if Gamma.get(t)]
        if len(prods) < 2:
            continue

        expiry = {}
        for t in prods:
            m_t = int(shelf_seq[t]) if t < len(shelf_seq) else 0
            expiry[t] = t + m_t

        prods_sorted = sorted(prods, key=lambda t: expiry[t])

        for a in range(len(prods_sorted)):
            t1 = prods_sorted[a]
            v1 = expiry[t1]
            reachable_t1 = set(Gamma.get(t1, []))

            for b in range(a + 1, len(prods_sorted)):
                t2 = prods_sorted[b]
                v2 = expiry[t2]

                if v1 >= v2:
                    continue

                reachable_t2 = Gamma.get(t2, [])

                for up in reachable_t2:
                    if demand[up] <= 0:
                        continue

                    for u in range(t2, up):
                        if u not in reachable_t1 or demand[u] <= 0:
                            continue

                        cut = (item_id, t1, u, t2, up)
                        all_cuts.append(cut)

    return all_cuts


def find_best_branching_y(
    y_vals: Dict[Tuple[int, int], float],
    forced_y: Dict[int, Set[int]],
    forbidden_y: Dict[int, Set[int]],
    items: Dict[int, dict],
) -> Optional[Tuple[int, int, float]]:
    """
    Find the best Y variable to branch on using strong branching heuristics.

    Score = fractionality * setup_cost_weight
    - Fractionality: 4 * min(val, 1-val) gives score 0-1, max at val=0.5
    - Setup cost weight: normalized setup cost (higher cost = higher priority)

    This prioritizes branching on high-impact fractional variables.
    """
    candidates = []

    # Collect all fractional Y candidates
    for (item_id, t), val in y_vals.items():
        if t in forced_y.get(item_id, set()):
            continue
        if t in forbidden_y.get(item_id, set()):
            continue

        frac = min(val, 1.0 - val)
        if frac > EPS:
            # Get setup cost for this item at period t
            setup = items[item_id].get("setup", [0])
            if isinstance(setup, list):
                s_cost = float(setup[t]) if t < len(setup) else 0.0
            else:
                s_cost = float(setup)
            candidates.append((item_id, t, val, frac, s_cost))

    if not candidates:
        return None

    # Normalize setup costs to [0, 1] range
    max_setup = max(c[4] for c in candidates) if candidates else 1.0
    if max_setup < EPS:
        max_setup = 1.0

    # Score each candidate: fractionality_score * (0.3 + 0.7 * normalized_setup_cost)
    # This gives 30% weight to fractionality alone and 70% to setup cost
    best_score = -1.0
    best = None

    for item_id, t, val, frac, s_cost in candidates:
        frac_score = 4.0 * frac  # 0 to 1, max at val=0.5
        setup_weight = 0.3 + 0.7 * (s_cost / max_setup)
        score = frac_score * setup_weight

        if score > best_score:
            best_score = score
            best = (item_id, t, val)

    return best


def find_fractional_x(
    x_agg: Dict[int, Dict[int, float]],
    x_bounds: Dict[Tuple[int, int], Tuple[float, float]],
) -> Optional[Tuple[int, int, float]]:
    """
    Find a fractional X (production quantity) to branch on.

    x_agg: dict of item_id -> {t: total_production_at_t}

    Returns: (item_id, t, value) for the most fractional X, or None if all integer.
    """
    best_frac = 0.0
    best = None

    for item_id, prod_by_t in x_agg.items():
        for t, val in prod_by_t.items():
            # Check if this (item, t) already has bounds that make it integer
            if (item_id, t) in x_bounds:
                lb, ub = x_bounds[(item_id, t)]
                if abs(ub - lb) < EPS:
                    continue  # Already fixed

            # Check fractionality
            frac_part = val - math.floor(val)
            if frac_part < EPS or frac_part > 1 - EPS:
                continue  # Integer

            # Score by how close to 0.5 the fractional part is
            frac_score = min(frac_part, 1.0 - frac_part)
            if frac_score > best_frac:
                best_frac = frac_score
                best = (item_id, t, val)

    return best


# =============================================================================
# DIVE HEURISTIC
# =============================================================================


def _try_dive_with_threshold(
    items: Dict[int, dict],
    T: int,
    capacity: List[float],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    columns_pool: Dict[int, List[Tuple[int, int, float]]],
    y_vals: Dict[Tuple[int, int], float],
    base_forced_y: Dict[int, Set[int]],
    base_forbidden_y: Dict[int, Set[int]],
    lefo_cuts: List[Tuple[int, int, int, int, int]],
    threshold: float,
    force_all_positive: bool = False,
) -> Tuple[Optional[float], Optional[Dict]]:
    """
    Try dive with a specific threshold.
    If force_all_positive=True, force Y=1 for ALL Y > 0.
    Otherwise force Y=1 for Y >= threshold, Y=0 for Y < threshold.
    """
    forced_y = {i: set(s) for i, s in base_forced_y.items()}
    forbidden_y = {i: set(s) for i, s in base_forbidden_y.items()}

    for (item_id, t), val in y_vals.items():
        if t in forced_y.get(item_id, set()) or t in forbidden_y.get(item_id, set()):
            continue

        if force_all_positive:
            # Force ALL positive Y to 1 (most likely to find feasible)
            if val > EPS:
                if item_id not in forced_y:
                    forced_y[item_id] = set()
                forced_y[item_id].add(t)
        else:
            # Use threshold
            if val >= threshold:
                if item_id not in forced_y:
                    forced_y[item_id] = set()
                forced_y[item_id].add(t)
            else:
                if item_id not in forbidden_y:
                    forbidden_y[item_id] = set()
                forbidden_y[item_id].add(t)

    obj, solution, _ = solve_rmp_with_duals(
        items,
        T,
        capacity,
        Gamma_by_item,
        columns_pool,
        forced_y,
        forbidden_y,
        lefo_cuts,
        x_bounds=None,
        use_artificial=True,
    )

    if not math.isfinite(obj):
        return None, None

    # Check if truly feasible (no artificials)
    art_total = solution.get("art_total", 0.0)
    if art_total > EPS:
        return None, None

    # Check if Y is integer
    y_dive = solution.get("y", {})
    if not is_y_integer(y_dive):
        return None, None

    # Check if X is integer
    x_agg = solution.get("x_agg", {})
    if find_fractional_x(x_agg, {}) is not None:
        return None, None  # X is fractional

    return obj, solution


def try_dive(
    items: Dict[int, dict],
    T: int,
    capacity: List[float],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    columns_pool: Dict[int, List[Tuple[int, int, float]]],
    y_vals: Dict[Tuple[int, int], float],
    base_forced_y: Dict[int, Set[int]],
    base_forbidden_y: Dict[int, Set[int]],
    lefo_cuts: List[Tuple[int, int, int, int, int]],
) -> Tuple[Optional[float], Optional[Dict]]:
    """
    Smart dive heuristic - tries multiple strategies to find feasible solution.

    Strategy 1: Force Y=1 for ALL periods with any positive Y value
    Strategy 2-4: Try progressively lower thresholds (0.5, 0.3, 0.1)

    Returns (objective, solution) if feasible integer found, else (None, None).
    """
    # Strategy 1: Force ALL positive Y to 1
    result = _try_dive_with_threshold(
        items,
        T,
        capacity,
        Gamma_by_item,
        columns_pool,
        y_vals,
        base_forced_y,
        base_forbidden_y,
        lefo_cuts,
        threshold=0.0,
        force_all_positive=True,
    )
    if result[0] is not None:
        return result

    # Strategy 2-4: Try different thresholds
    for threshold in [0.5, 0.3, 0.1]:
        result = _try_dive_with_threshold(
            items,
            T,
            capacity,
            Gamma_by_item,
            columns_pool,
            y_vals,
            base_forced_y,
            base_forbidden_y,
            lefo_cuts,
            threshold=threshold,
            force_all_positive=False,
        )
        if result[0] is not None:
            return result

    return None, None


# =============================================================================
# BRANCH-AND-PRICE
# =============================================================================


def solve_branch_and_price(
    items: Dict[int, dict],
    T: int,
    capacity: List[float],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    initial_blocks_by_item: Dict[int, List[Tuple[int, int, float]]],
    max_time: float = 3600000,
    max_nodes: int = 100000000,
    verbose: bool = False,
    logger: Optional[BnPLogger] = None,
    add_all_lefo_cuts_upfront: bool = ADD_ALL_LEFO_CUTS_UPFRONT,
) -> Tuple[float, Optional[Dict], float, float]:
    """
    Solve using block-based Branch-and-Price with column generation.

    Args:
        add_all_lefo_cuts_upfront: If True, add ALL LEFO (no-crossing) constraints
            at the root node (same as MIP). If False, add them lazily when violations
            are detected. Default True for exactness.

    Returns: (best_obj, best_solution, root_bound, final_gap)

    The final_gap is 0% when proven optimal (queue empty and all nodes processed).
    """
    start_time = time.time()

    # Generate LEFO cuts upfront if enabled
    if add_all_lefo_cuts_upfront:
        initial_lefo_cuts = generate_all_lefo_cuts(items, Gamma_by_item)
        if verbose:
            print(f"  Adding {len(initial_lefo_cuts)} LEFO cuts upfront")
    else:
        initial_lefo_cuts = []

    root = BranchNode(
        node_id=0,
        parent_id=None,
        depth=0,
        forced_y={i: set() for i in items},
        forbidden_y={i: set() for i in items},
        lefo_cuts=initial_lefo_cuts,
    )

    if verbose:
        print("\n>>> ROOT NODE <<<")

    # Column generation at root
    root_obj, root_info, root_columns, root_cg_iters, root_cols_added = (
        column_generation_loop(
            items,
            T,
            capacity,
            Gamma_by_item,
            initial_blocks_by_item,
            root.forced_y,
            root.forbidden_y,
            root.lefo_cuts,
            root.x_bounds,
            node_id=0,
            logger=logger,
            verbose=verbose,
        )
    )

    if not math.isfinite(root_obj):
        if verbose:
            print("  Root infeasible!")
        if logger:
            logger.log_node(
                NodeLogEntry(
                    node_id=0,
                    depth=0,
                    lp_bound=math.inf,
                    incumbent=math.inf,
                    branch_item=None,
                    branch_t=None,
                    direction=None,
                    status="INFEASIBLE",
                    cg_iters=root_cg_iters,
                    columns_added=root_cols_added,
                )
            )
        return math.inf, None, math.inf, 0.0

    root.lp_bound = root_obj
    root_bound = root_obj

    if verbose:
        print(f"  Root LP bound: {root_obj:.4f}")
        print(f"  CG iterations: {root_cg_iters}, columns added: {root_cols_added}")

    y_vals = root_info.get("y", {})

    # Check if root is already Y-integer (we don't branch on Z anymore)
    if is_y_integer(y_vals):
        if verbose:
            print("  ✓ Root is Y-INTEGER - OPTIMAL!")
        if logger:
            logger.log_node(
                NodeLogEntry(
                    node_id=0,
                    depth=0,
                    lp_bound=root_obj,
                    incumbent=root_obj,
                    branch_item=None,
                    branch_t=None,
                    direction=None,
                    status="OPTIMAL_AT_ROOT",
                    cg_iters=root_cg_iters,
                    columns_added=root_cols_added,
                )
            )
        return root_obj, root_info, root_bound, 0.0

    # Initialize B&B
    best_ub: Optional[float] = None
    best_solution: Optional[Dict] = None
    columns_pool = {i: list(cols) for i, cols in root_columns.items()}

    # Track column usage: (item_id, block) -> last depth used
    # This allows pruning columns unused for 2 consecutive depth levels
    column_last_used: Dict[Tuple[int, Tuple[int, int, float]], int] = {}
    for item_id, cols in columns_pool.items():
        for block in cols:
            column_last_used[(item_id, block)] = 0  # Used at root

    # Try dive at root to find initial incumbent
    if verbose:
        print("  Trying dive at root...")
    dive_obj, dive_sol = try_dive(
        items,
        T,
        capacity,
        Gamma_by_item,
        columns_pool,
        y_vals,
        root.forced_y,
        root.forbidden_y,
        root.lefo_cuts,
    )
    if dive_obj is not None:
        # Check for LEFO violations before accepting as incumbent
        dive_z = dive_sol.get("z", {})
        lefo_violations = find_all_lefo_violations(dive_z, items, Gamma_by_item, [])
        if not lefo_violations:
            best_ub = dive_obj
            best_solution = dive_sol
            if verbose:
                print(f"  🎯 Dive found incumbent: {best_ub:.4f}")
        else:
            if verbose:
                print(
                    f"  Dive solution has {len(lefo_violations)} LEFO violations, rejected"
                )
    else:
        if verbose:
            print("  Dive did not find feasible solution")

    # Priority queue: (-depth, -node_id, node) for DFS (depth-first search)
    # Negative depth means deeper nodes have smaller values → processed first
    # Negative node_id gives LIFO behavior for same-depth nodes
    queue: List[Tuple[int, int, BranchNode]] = []
    heapq.heappush(queue, (-root.depth, -root.node_id, root))

    node_counter = 1
    nodes_explored = 0
    max_depth = 0

    # Track global lower bound
    global_lb = root_bound

    while queue:
        # Time check
        if time.time() - start_time > max_time:
            if verbose:
                print(f"\n⏱ Time limit reached ({max_time}s)")
            break

        if nodes_explored >= max_nodes:
            if verbose:
                print(f"\n⚠ Node limit reached ({max_nodes})")
            break

        _, _, node = heapq.heappop(queue)
        nodes_explored += 1

        # Update global lower bound (min LP bound of all nodes in queue)
        if queue:
            global_lb = min(q[2].lp_bound for q in queue)
        else:
            global_lb = best_ub if best_ub is not None else root_bound

        # Pruning by bound
        if best_ub is not None and node.lp_bound >= best_ub - EPS:
            if logger:
                logger.log_node(
                    NodeLogEntry(
                        node_id=node.node_id,
                        depth=node.depth,
                        lp_bound=node.lp_bound,
                        incumbent=best_ub,
                        branch_item=None,
                        branch_t=None,
                        direction=None,
                        status="PRUNED_BY_BOUND",
                        cg_iters=0,
                        columns_added=0,
                    )
                )
            continue

        # Column generation with LEFO cut-and-resolve loop
        total_cg_iters = 0
        total_cols_added = 0
        while True:
            obj, info, node_columns, cg_iters, cols_added = column_generation_loop(
                items,
                T,
                capacity,
                Gamma_by_item,
                columns_pool,
                node.forced_y,
                node.forbidden_y,
                node.lefo_cuts,
                node.x_bounds,
                node_id=node.node_id,
                logger=logger,
                verbose=False,  # Suppress CG iteration details
            )
            total_cg_iters += cg_iters
            total_cols_added += cols_added

            if not math.isfinite(obj):
                break  # Infeasible

            # Check for ALL LEFO violations and add cuts (up to MAX_LEFO_CUTS_PER_ITER)
            z_vals = info.get("z", {})
            violations = find_all_lefo_violations(
                z_vals, items, Gamma_by_item, node.lefo_cuts
            )
            if not violations:
                break  # No violations, proceed

            # Add up to MAX_LEFO_CUTS_PER_ITER cuts, sorted by violation severity
            violations_sorted = sorted(
                violations,
                key=lambda v: z_vals.get((v[0], v[1], v[2]), 0)
                + z_vals.get((v[0], v[3], v[4]), 0),
                reverse=True,
            )
            cuts_to_add = violations_sorted[:MAX_LEFO_CUTS_PER_ITER]
            for v in cuts_to_add:
                node.lefo_cuts.append(v)
            if verbose:
                print(
                    f"    Added {len(cuts_to_add)} LEFO cuts "
                    f"(of {len(violations)} violations), restarting CG"
                )

        cg_iters = total_cg_iters
        cols_added = total_cols_added

        # Merge columns into pool and track usage
        for item_id, cols in node_columns.items():
            existing = set(columns_pool.get(item_id, []))
            for col in cols:
                if col not in existing:
                    columns_pool[item_id].append(col)
                    column_last_used[(item_id, col)] = node.depth

        # Update column usage based on active lambdas
        # lam key is (item_id, t, e), block in pool is (t, e, setup_cost)
        lam_vals = info.get("lam", {}) if isinstance(info, dict) else {}
        for (item_id, t, e), val in lam_vals.items():
            if val > EPS:
                # Find the matching block in columns_pool
                for block in columns_pool.get(item_id, []):
                    if block[0] == t and block[1] == e:
                        column_last_used[(item_id, block)] = node.depth
                        break

        # Prune columns not used for 2+ consecutive depth levels
        # Only prune if we have enough columns (keep at least 20 per item)
        if node.depth >= 2:
            prune_depth_threshold = node.depth - 2
            for item_id in columns_pool:
                if len(columns_pool[item_id]) <= 20:
                    continue  # Keep minimum columns per item
                original_count = len(columns_pool[item_id])
                columns_pool[item_id] = [
                    col
                    for col in columns_pool[item_id]
                    if column_last_used.get((item_id, col), 0) >= prune_depth_threshold
                ]
                # Ensure we keep at least 20 columns (restore if pruned too many)
                if len(columns_pool[item_id]) < 20:
                    # This shouldn't happen often, but safety fallback
                    continue
                pruned = original_count - len(columns_pool[item_id])
                if pruned > 0:
                    # Clean up tracking dict for pruned columns
                    keys_to_remove = [
                        k
                        for k in column_last_used
                        if k[0] == item_id and k[1] not in set(columns_pool[item_id])
                    ]
                    for k in keys_to_remove:
                        del column_last_used[k]

        if not math.isfinite(obj):
            if verbose:
                print(
                    f"  [{nodes_explored}] Node {node.node_id} (d={node.depth}): INFEASIBLE"
                )
            if logger:
                logger.log_node(
                    NodeLogEntry(
                        node_id=node.node_id,
                        depth=node.depth,
                        lp_bound=math.inf,
                        incumbent=best_ub if best_ub else math.inf,
                        branch_item=None,
                        branch_t=None,
                        direction=None,
                        status="INFEASIBLE",
                        cg_iters=cg_iters,
                        columns_added=cols_added,
                    )
                )
            continue

        node.lp_bound = obj

        # Pruning by bound
        if best_ub is not None and obj >= best_ub - EPS:
            if verbose:
                print(
                    f"  [{nodes_explored}] Node {node.node_id} (d={node.depth}): LP={obj:.2f} >= UB={best_ub:.2f} → PRUNED"
                )
            if logger:
                logger.log_node(
                    NodeLogEntry(
                        node_id=node.node_id,
                        depth=node.depth,
                        lp_bound=obj,
                        incumbent=best_ub,
                        branch_item=None,
                        branch_t=None,
                        direction=None,
                        status="PRUNED_BY_BOUND",
                        cg_iters=cg_iters,
                        columns_added=cols_added,
                    )
                )
            continue

        # Dive heuristic: every 50 nodes if no incumbent, every 150 if we have one
        dive_interval = 50 if best_ub is None else 150
        if nodes_explored % dive_interval == 0 and nodes_explored > 0:
            dive_obj, dive_sol = try_dive(
                items,
                T,
                capacity,
                Gamma_by_item,
                columns_pool,
                info.get("y", {}),
                node.forced_y,
                node.forbidden_y,
                node.lefo_cuts,
            )
            if dive_obj is not None and (best_ub is None or dive_obj < best_ub - EPS):
                # Check for LEFO violations before accepting as incumbent
                dive_z = dive_sol.get("z", {})
                lefo_violations = find_all_lefo_violations(
                    dive_z, items, Gamma_by_item, []
                )
                if not lefo_violations:
                    best_ub = dive_obj
                    best_solution = dive_sol
                    if verbose:
                        print(f"  🎯 Dive found better incumbent: {best_ub:.4f}")
                elif verbose:
                    print(
                        f"  Dive solution has {len(lefo_violations)} LEFO violations, rejected"
                    )

        y_vals = info.get("y", {})

        # Check Y integrality (we don't branch on Z - we use LEFO cuts instead)
        y_int = is_y_integer(y_vals)

        if y_int:
            # Y is integer - check if X is also integer
            x_agg = info.get("x_agg", {})
            branch_var_x = find_fractional_x(x_agg, node.x_bounds)

            if branch_var_x is None:
                # Both Y and X are integer - valid solution!
                if best_ub is None or obj < best_ub - EPS:
                    old_ub = best_ub
                    best_ub = obj
                    best_solution = info
                    if verbose:
                        if old_ub is None:
                            print(
                                f"  [{nodes_explored}] Node {node.node_id} (d={node.depth}): LP={obj:.2f} → ★ FIRST INCUMBENT"
                            )
                        else:
                            print(
                                f"  [{nodes_explored}] Node {node.node_id} (d={node.depth}): LP={obj:.2f} → ★ IMPROVED {old_ub:.2f} → {best_ub:.2f}"
                            )
                    if logger:
                        logger.log_node(
                            NodeLogEntry(
                                node_id=node.node_id,
                                depth=node.depth,
                                lp_bound=obj,
                                incumbent=obj,
                                branch_item=None,
                                branch_t=None,
                                direction=None,
                                status="NEW_INCUMBENT",
                                cg_iters=cg_iters,
                                columns_added=cols_added,
                            )
                        )
                continue

            # Y is integer but X is fractional - branch on X
            item_id, t, val = branch_var_x
            floor_val = math.floor(val)
            ceil_val = math.ceil(val)

            if verbose:
                ub_str = f"{best_ub:.2f}" if best_ub else "N/A"
                print(
                    f"  [{nodes_explored}] Node {node.node_id} (d={node.depth}): "
                    f"LP={obj:.2f}, UB={ub_str} → branch X[{item_id},{t}]={val:.2f}"
                )

            if logger:
                logger.log_node(
                    NodeLogEntry(
                        node_id=node.node_id,
                        depth=node.depth,
                        lp_bound=obj,
                        incumbent=best_ub if best_ub else math.inf,
                        branch_item=item_id,
                        branch_t=t,
                        direction="X",
                        status="BRANCHED_X",
                        cg_iters=cg_iters,
                        columns_added=cols_added,
                    )
                )

            # Create two children: X <= floor, X >= ceil
            for direction in [0, 1]:
                child = BranchNode(
                    node_id=node_counter,
                    parent_id=node.node_id,
                    depth=node.depth + 1,
                    forced_y={i: set(s) for i, s in node.forced_y.items()},
                    forbidden_y={i: set(s) for i, s in node.forbidden_y.items()},
                    lefo_cuts=list(node.lefo_cuts),
                    x_bounds=dict(node.x_bounds),  # Inherit parent's X bounds
                    lp_bound=obj,
                )

                if direction == 0:
                    # X <= floor_val
                    old_lb, old_ub_bound = child.x_bounds.get(
                        (item_id, t), (0.0, math.inf)
                    )
                    child.x_bounds[(item_id, t)] = (
                        old_lb,
                        min(old_ub_bound, floor_val),
                    )
                else:
                    # X >= ceil_val
                    old_lb, old_ub_bound = child.x_bounds.get(
                        (item_id, t), (0.0, math.inf)
                    )
                    child.x_bounds[(item_id, t)] = (max(old_lb, ceil_val), old_ub_bound)

                heapq.heappush(queue, (-child.depth, -child.node_id, child))
                node_counter += 1

            max_depth = max(max_depth, node.depth + 1)
            continue

        # Branch on best Y variable (setup-cost weighted)
        branch_var_y = find_best_branching_y(
            y_vals, node.forced_y, node.forbidden_y, items
        )

        if branch_var_y is not None:
            # Branch on Y
            item_id, t, val = branch_var_y

            if verbose:
                ub_str = f"{best_ub:.2f}" if best_ub else "N/A"
                print(
                    f"  [{nodes_explored}] Node {node.node_id} (d={node.depth}): "
                    f"LP={obj:.2f}, UB={ub_str} → branch Y[{item_id},{t}]={val:.3f}"
                )

            if logger:
                logger.log_node(
                    NodeLogEntry(
                        node_id=node.node_id,
                        depth=node.depth,
                        lp_bound=obj,
                        incumbent=best_ub if best_ub else math.inf,
                        branch_item=item_id,
                        branch_t=t,
                        direction="Y",
                        status="BRANCHED_Y",
                        cg_iters=cg_iters,
                        columns_added=cols_added,
                    )
                )

            # Create children for Y branching (inherit LEFO cuts and X bounds)
            for direction in [0, 1]:
                child = BranchNode(
                    node_id=node_counter,
                    parent_id=node.node_id,
                    depth=node.depth + 1,
                    forced_y={i: set(s) for i, s in node.forced_y.items()},
                    forbidden_y={i: set(s) for i, s in node.forbidden_y.items()},
                    lefo_cuts=list(node.lefo_cuts),  # Inherit parent's cuts
                    x_bounds=dict(node.x_bounds),  # Inherit parent's X bounds
                    lp_bound=obj,
                )

                if direction == 0:
                    child.forbidden_y[item_id].add(t)
                else:
                    child.forced_y[item_id].add(t)

                heapq.heappush(queue, (-child.depth, -child.node_id, child))
                node_counter += 1

            max_depth = max(max_depth, node.depth + 1)
            continue

        # Y is fractional but no branching candidate found - this shouldn't happen
        if verbose:
            print(
                f"  [{nodes_explored}] Node {node.node_id}: No branching candidate found"
            )

        # Progress
        if verbose and nodes_explored % 50 == 0:
            ub_str = f"{best_ub:.2f}" if best_ub else "N/A"
            print(f"  [{nodes_explored}] queue={len(queue)}, best={ub_str}")

    # Compute final gap
    # When queue is empty and we have a solution, it's proven optimal -> gap = 0%
    if best_ub is not None:
        if not queue:
            # All nodes processed -> proven optimal
            final_gap = 0.0
            global_lb = best_ub
        else:
            # Time/node limit -> gap is (UB - LB) / UB
            global_lb = min(q[2].lp_bound for q in queue) if queue else root_bound
            final_gap = (best_ub - global_lb) / max(abs(best_ub), 1e-10)
    else:
        final_gap = math.inf

    if verbose:
        print(f"\n=== B&P Summary ===")
        print(f"  Nodes explored: {nodes_explored}")
        print(f"  Max depth: {max_depth}")
        print(f"  Root bound: {root_bound:.4f}")
        print(f"  Global LB: {global_lb:.4f}")
        if best_ub is not None:
            print(f"  Best solution: {best_ub:.4f}")
            print(f"  Gap: {final_gap * 100:.4f}%")

    return (
        best_ub if best_ub is not None else math.inf,
        best_solution,
        root_bound,
        final_gap,
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def _as_len_T_vector(val, T: int) -> List[float]:
    """Convert scalar or list to length-T vector."""
    if val is None:
        return []
    if isinstance(val, (int, float)):
        return [float(val)] * T
    if isinstance(val, list):
        if len(val) != T:
            raise ValueError(f"Expected length-{T} list, got {len(val)}")
        return [float(x) for x in val]
    raise TypeError("Value must be a number or a list")


def _cap_global_from_dem(items: Dict[int, dict], T: int) -> List[float]:
    """Generate default capacity from total demand."""
    cap_raw = [0.0] * T
    for it in items.values():
        dem = it["demand"]
        for t in range(T):
            cap_raw[t] += float(dem[t])
    buf = max(5.0, 0.2 * max(cap_raw) if cap_raw else 0.0)
    return [c + buf for c in cap_raw]


def solve_instance(
    instance_path: str | Path = "last_instance.json",
    time_limit: int = 0,
    out_dir: str | Path = "bnp_results",
    verbose: bool = True,  # Default True to show progress in terminal
    mip_gap: float = 0.0,  # Accepted for compatibility, not used (pure B&P)
) -> Tuple[Dict, List[str]]:
    """
    Solve the perishable lot-sizing problem using Block-Based Branch-and-Price.

    Args:
        instance_path: Path to instance JSON file
        time_limit: Time limit in seconds (0 = unlimited)
        out_dir: Output directory
        verbose: If True, print detailed logs and write CSV files
        mip_gap: Ignored (kept for Streamlit compatibility)

    Returns:
        (summary_dict, orders_list)
    """
    _ = mip_gap  # Not used in pure B&P solver
    start_time = time.time()

    data = json.loads(Path(instance_path).read_text())
    T = int(data["period"])
    items: Dict[int, dict] = {int(k): v for k, v in data["items"].items()}

    prod_cap = data.get("production_capacity") or data.get("manual_capacity")
    capacity = (
        _as_len_T_vector(prod_cap, T)
        if prod_cap is not None
        else _cap_global_from_dem(items, T)
    )

    max_time = int(time_limit) if time_limit > 0 else 36000

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = BnPLogger(out_path, enabled=verbose) if verbose else None

    if verbose:
        header = [
            "╠" + "═" * 68 + "╣",
            f"║  Items:     {len(items):<55d} ║",
            f"║  Periods:   {T:<55d} ║",
            f"║  Capacity:  {str(capacity[:min(6, T)]) + ('...' if T > 6 else ''):<55s} ║",
            "╚" + "═" * 68 + "╝",
        ]
        for line in header:
            print(line)

    # Build Gamma (reachable demands) for each item
    Gamma_by_item: Dict[int, Dict[int, List[int]]] = {}
    for item_id, item_data in items.items():
        shelf_seq = list(item_data["shelf_seq"])
        if len(shelf_seq) != T:
            raise ValueError(f"items[{item_id}]['shelf_seq'] must have length {T}")

        Gamma: Dict[int, List[int]] = {}
        for t in range(T):
            m_it = int(shelf_seq[t])
            if m_it <= 0:
                Gamma[t] = []
            else:
                u_max = min(T - 1, t + m_it)
                Gamma[t] = list(range(t, u_max + 1))
        Gamma_by_item[item_id] = Gamma

    # Generate minimal initial columns (not all blocks!)
    initial_blocks_by_item: Dict[int, List[Tuple[int, int, float]]] = {}
    total_initial = 0
    total_possible = 0
    for item_id, item_data in items.items():
        # Minimal columns for feasibility
        initial_cols = generate_minimal_initial_columns(
            item_id, item_data, T, Gamma_by_item[item_id]
        )
        initial_blocks_by_item[item_id] = initial_cols
        total_initial += len(initial_cols)

        # Count total possible blocks (for info only)
        all_blocks = list_blocks_for_item(item_id, item_data, T, Gamma_by_item[item_id])
        total_possible += len(all_blocks)

        if verbose:
            print(
                f"  Item {item_id}: {len(initial_cols)} initial / {len(all_blocks)} possible blocks"
            )

    if verbose:
        print(
            f"  Total: {total_initial} initial columns (out of {total_possible} possible)"
        )

    # Solve
    best_obj, best_solution, root_bound, final_gap = solve_branch_and_price(
        items=items,
        T=T,
        capacity=capacity,
        Gamma_by_item=Gamma_by_item,
        initial_blocks_by_item=initial_blocks_by_item,
        max_time=max_time,
        verbose=verbose,
        logger=logger,
    )

    runtime = time.time() - start_time

    # Write CSVs if verbose
    if logger:
        logger.write_csvs()

    # Build output
    if best_solution is None or not math.isfinite(best_obj):
        summary = {
            "status": int(GRB.INFEASIBLE),
            "objective": None,
            "best_bound": root_bound if math.isfinite(root_bound) else None,
            "gap": None,
            "runtime_sec": runtime,
            "solver_version": "block_branch_and_price_cg",
            "n_items": len(items),
            "T": T,
        }
        orders_txt = []
    else:
        # Determine status based on gap
        if final_gap < EPS:
            status = int(GRB.OPTIMAL)
        elif runtime >= max_time:
            status = int(GRB.TIME_LIMIT)
        else:
            status = int(GRB.OPTIMAL)

        summary = {
            "status": status,
            "objective": float(best_obj),
            "best_bound": float(best_obj) if final_gap < EPS else float(root_bound),
            "gap": final_gap,
            "runtime_sec": runtime,
            "solver_version": "block_branch_and_price_cg",
            "n_items": len(items),
            "T": T,
        }

        x_agg = best_solution.get("x_agg", {})
        y_vals = best_solution.get("y", {})
        lam_vals = best_solution.get("lam", {})
        orders_txt = []
        for item_id in sorted(items.keys()):
            # Collect Y values for this item
            y_periods = [
                t for (i, t), val in y_vals.items() if i == item_id and val > 0.5
            ]
            y_periods.sort()
            y_str = ", ".join(str(t) for t in y_periods) if y_periods else "none"

            orders_txt.append(f"Item {item_id}")
            orders_txt.append(f"  Y (setups at periods): [{y_str}]")

            # Collect lambda (combination coefficients) for this item
            item_lambdas = [
                (t, e, val) for (i, t, e), val in lam_vals.items() if i == item_id
            ]
            item_lambdas.sort(key=lambda x: (x[0], x[1]))  # Sort by (t, e)
            if item_lambdas:
                orders_txt.append(f"  λ (block coefficients):")
                for t, e, val in item_lambdas:
                    orders_txt.append(f"    λ[{t},{e}] = {val:.6f}")

            orders_txt.append(f"  Production (t → qty):")
            production = x_agg.get(item_id, {})
            for t in sorted(production.keys()):
                qty = production[t]
                if qty > EPS:
                    orders_txt.append(f"    {t:2d} → {qty:8.3f}")
            orders_txt.append("")

    (out_path / "orders.txt").write_text("\n".join(orders_txt), encoding="utf-8")
    (out_path / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return summary, orders_txt


# =============================================================================
# DEMO MAIN
# =============================================================================


if __name__ == "__main__":
    # 3-item instance for testing stronger branching and column pruning:
    # instance = {
    #     "period": 12,
    #     "manual_capacity": [0, 0, 100, 120, 110, 100, 90, 100, 110, 120, 100, 80],
    #     "items": {
    #         "0": {
    #             "h": [
    #                 0.5,
    #                 0.52,
    #                 0.54,
    #                 0.56,
    #                 0.58,
    #                 0.6,
    #                 0.58,
    #                 0.56,
    #                 0.54,
    #                 0.52,
    #                 0.5,
    #                 0.48,
    #             ],
    #             "c_var": [2.0, 1.8, 2.5, 1.9, 1.7, 2.2, 1.5, 1.3, 2.7, 1.9, 2.1, 1.6],
    #             "setup": [85, 86, 87, 88, 89, 90, 91, 92, 91, 90, 89, 88],
    #             "demand": [0, 0, 45, 32, 55, 28, 15, 20, 38, 42, 30, 25],
    #             "shelf_seq": [12, 10, 8, 6, 8, 5, 7, 9, 6, 5, 4, 3],
    #         },
    #         "1": {
    #             "h": [
    #                 0.3,
    #                 0.32,
    #                 0.34,
    #                 0.36,
    #                 0.38,
    #                 0.4,
    #                 0.38,
    #                 0.36,
    #                 0.34,
    #                 0.32,
    #                 0.3,
    #                 0.28,
    #             ],
    #             "c_var": [1.5, 1.4, 2.0, 1.6, 1.3, 1.8, 1.2, 1.1, 2.2, 1.5, 1.7, 1.3],
    #             "setup": [70, 71, 72, 73, 74, 75, 76, 77, 76, 75, 74, 73],
    #             "demand": [0, 0, 30, 25, 40, 20, 12, 18, 35, 28, 22, 18],
    #             "shelf_seq": [10, 8, 7, 5, 6, 4, 6, 8, 5, 4, 3, 2],
    #         },
    #         "2": {
    #             "h": [
    #                 0.6,
    #                 0.62,
    #                 0.64,
    #                 0.66,
    #                 0.68,
    #                 0.7,
    #                 0.68,
    #                 0.66,
    #                 0.64,
    #                 0.62,
    #                 0.6,
    #                 0.58,
    #             ],
    #             "c_var": [2.5, 2.3, 3.0, 2.4, 2.1, 2.8, 1.9, 1.7, 3.2, 2.4, 2.6, 2.0],
    #             "setup": [95, 96, 97, 98, 99, 100, 101, 102, 101, 100, 99, 98],
    #             "demand": [0, 0, 25, 18, 35, 15, 10, 12, 28, 22, 16, 14],
    #             "shelf_seq": [8, 6, 5, 4, 5, 3, 5, 6, 4, 3, 2, 2],
    #         },
    #     },
    # }
    # 1105 case - single item instance (T=10, expected optimal ~ 1105):
    instance = {
        "period": 10,
        "manual_capacity": [0, 0, 80, 80, 80, 80, 80, 80, 100, 80],
        "items": {
            "0": {
                "h": [
                    0.4,
                    0.408,
                    0.416,
                    0.424,
                    0.430,
                    0.435,
                    0.438,
                    0.440,
                    0.440,
                    0.438,
                ],
                "c_var": [
                    1.76,
                    1.71,
                    2.50,
                    1.78,
                    1.67,
                    2.11,
                    1.37,
                    1.24,
                    2.68,
                    1.81,
                ],
                "setup": [
                    80,
                    81.66,
                    83.25,
                    84.70,
                    85.95,
                    86.93,
                    87.61,
                    87.96,
                    87.96,
                    87.61,
                ],
                "demand": [0, 0, 68, 49, 66, 38, 17, 17, 41, 43],
                "shelf_seq": [24, 18, 22, 6, 16, 9, 21, 23, 9, 7],
            }
        },
    }

    out_dir = Path("bnp_dp_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    instance_path = out_dir / "test_instance.json"
    instance_path.write_text(json.dumps(instance, indent=2))

    print("=" * 70)
    print("BLOCK-BASED BRANCH-AND-PRICE WITH COLUMN GENERATION")
    print("=" * 70)

    # Run with verbose=True for demo
    summary, orders = solve_instance(
        instance_path=str(instance_path),
        time_limit=60000,
        out_dir=str(out_dir),
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    status_map = {2: "OPTIMAL", 3: "INFEASIBLE", 9: "TIME_LIMIT", 13: "SUBOPTIMAL"}
    status_str = status_map.get(summary["status"], f"STATUS_{summary['status']}")
    print(f"  Status:     {status_str}")
    print(f"  Objective:  {summary.get('objective', 'N/A')}")
    print(f"  Best bound: {summary.get('best_bound', 'N/A')}")
    gap = summary.get("gap", 0) or 0
    print(f"  Gap:        {gap * 100:.4f}%")
    print(f"  Runtime:    {summary['runtime_sec']:.2f}s")

    print("\nOrders:")
    for line in orders:
        print(f"  {line}")

    # Show CSV files if created
    nodes_csv = out_dir / "bnp_nodes.csv"
    cg_csv = out_dir / "bnp_cg_iters.csv"
    if nodes_csv.exists():
        print(f"\n📄 Node log: {nodes_csv}")
    if cg_csv.exists():
        print(f"📄 CG iterations log: {cg_csv}")
