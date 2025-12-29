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


@dataclass
class BranchNode:
    """Node in the branch-and-bound tree."""

    node_id: int
    parent_id: Optional[int]
    depth: int
    forced_y: Dict[int, Set[int]] = field(default_factory=dict)  # item -> {periods}
    forbidden_y: Dict[int, Set[int]] = field(default_factory=dict)  # item -> {periods}
    forced_z: Dict[int, Set[Tuple[int, int]]] = field(
        default_factory=dict
    )  # item -> {(t,u)}
    forbidden_z: Dict[int, Set[Tuple[int, int]]] = field(
        default_factory=dict
    )  # item -> {(t,u)}
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
    forced_z: Dict[int, Set[Tuple[int, int]]],
    forbidden_z: Dict[int, Set[Tuple[int, int]]],
    use_artificial: bool = True,
) -> Tuple[float, Dict, Dict]:
    """
    Solve RMP and return objective, solution, and dual values.

    Uses artificial variables to ensure feasibility during column generation.

    Returns:
        (obj, solution_dict, duals_dict)

    duals_dict contains:
        - pi_demand[(item_id, u)]: dual for demand constraint
        - rho_cap[t]: dual for capacity constraint
        - sigma_link[(item_id, t, u)]: dual for linking constraint
        - mu_y_def[(item_id, t)]: dual for Y definition constraint
        - tau_force_y[(item_id, t)]: dual for forced Y constraint
        - tau_force_z[(item_id, t, u)]: dual for forced Z constraint
    """
    m = gp.Model("RMP")
    m.Params.OutputFlag = 0
    m.Params.LogToConsole = 0
    m.Params.Method = 1

    demand_periods_by_item = {}
    for item_id, item_data in items.items():
        demand = item_data["demand"]
        demand_periods_by_item[item_id] = [u for u in range(T) if demand[u] > 0]

    # Filter blocks by forbidden Y and forbidden Z
    valid_columns_by_item = {}
    for item_id, cols in columns_by_item.items():
        forbidden_periods = forbidden_y.get(item_id, set())
        forbidden_arcs = forbidden_z.get(item_id, set())
        valid_cols = []
        for t, e, s in cols:
            if t in forbidden_periods:
                continue
            # Check if block uses any forbidden arc
            has_forbidden_arc = False
            Gamma = Gamma_by_item.get(item_id, {})
            reachable = Gamma.get(t, [])
            for u in range(t, e + 1):
                if u in reachable and (t, u) in forbidden_arcs:
                    has_forbidden_arc = True
                    break
            if not has_forbidden_arc:
                valid_cols.append((t, e, s))
        valid_columns_by_item[item_id] = valid_cols

    # Block variables λ[i,t,e]
    lam = {}
    for item_id, cols in valid_columns_by_item.items():
        for t, e, s in cols:
            key = (item_id, t, e)
            if key not in lam:
                lam[key] = m.addVar(lb=0.0, ub=1.0, name=f"lam_{item_id}_{t}_{e}")

    # Flow variables f[i,t,u] - exclude forbidden arcs
    f = {}
    for item_id, item_data in items.items():
        demand = item_data["demand"]
        Gamma = Gamma_by_item.get(item_id, {})
        forbidden_periods = forbidden_y.get(item_id, set())
        forbidden_arcs = forbidden_z.get(item_id, set())

        for t in range(T):
            if t in forbidden_periods:
                continue
            reachable = Gamma.get(t, [])
            for u in reachable:
                if demand[u] > 0 and (t, u) not in forbidden_arcs:
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
    force_z_con = {}

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

    # Forced Z: z[i,t,u] = 1
    for item_id, arcs in forced_z.items():
        for t, u in arcs:
            if (item_id, t, u) in z:
                force_z_con[(item_id, t, u)] = m.addConstr(
                    z[(item_id, t, u)] == 1.0, f"force_z_{item_id}_{t}_{u}"
                )

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
        "tau_force_z": {},
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

    for key, con in force_z_con.items():
        duals["tau_force_z"][key] = con.Pi

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
    forced_z: Set[Tuple[int, int]],
    forbidden_z: Set[Tuple[int, int]],
    existing_blocks: Set[Tuple[int, int, float]],
    max_columns_per_item: int = 5,
) -> Tuple[List[Tuple[Tuple[int, int, float], float]], float]:
    """
    DP pricing to find negative reduced cost blocks for an item.

    Reduced cost for block (t, e):
        rc = setup[t] - μ_y_def[(item_id, t)]
             - Σ_{u in [t,e]} σ_link[(item_id, t, u)] * d[u]
             - sig_force_y[(item_id, t)]  (if Y[i,t]=1 forced)
             - Σ τ_force_z[(item_id, t, u)]  (for forced arcs in block)

    The sig and τ duals give a "discount" to blocks that satisfy forced constraints,
    ensuring we generate all necessary columns when branching.

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
    tau_force_z = duals.get("tau_force_z", {})

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

            # Check if block uses any forbidden arc
            has_forbidden_arc = False
            for u in range(t, e + 1):
                if u in reachable and demand[u] > 0 and (t, u) in forbidden_z:
                    has_forbidden_arc = True
                    break
            if has_forbidden_arc:
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
            # When Y[i,t]=1 is forced, blocks at period t get a discount
            # The dual can be positive or negative; we subtract it from RC
            tau_y = tau_force_y.get((item_id, t), 0.0)

            # τ_force_z: dual of forced Z constraints (z[i,t,u] = 1)
            # For each forced arc (t,u) that this block covers, we get a discount
            tau_z_contrib = 0.0
            for u in range(t, e + 1):
                if u in reachable and demand[u] > 0 and (t, u) in forced_z:
                    tau_z = tau_force_z.get((item_id, t, u), 0.0)
                    tau_z_contrib += tau_z

            # RC = setup_cost - μ - σ_contrib - τ_y - τ_z_contrib
            rc = s_cost - mu - sigma_contrib - tau_y - tau_z_contrib

            if rc < -EPS:
                negative_rc_blocks.append((block, rc))

    # Sort by reduced cost (most negative first)
    negative_rc_blocks.sort(key=lambda x: x[1])

    # True minimum RC across ALL negative RC blocks
    true_min_rc = negative_rc_blocks[0][1] if negative_rc_blocks else 0.0

    # Return top k columns and the true minimum RC
    return negative_rc_blocks[:max_columns_per_item], true_min_rc


def price_all_items(
    items: Dict[int, dict],
    T: int,
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    duals: Dict,
    forced_y: Dict[int, Set[int]],
    forbidden_y: Dict[int, Set[int]],
    forced_z: Dict[int, Set[Tuple[int, int]]],
    forbidden_z: Dict[int, Set[Tuple[int, int]]],
    columns_by_item: Dict[int, List[Tuple[int, int, float]]],
    max_columns_per_item: int = 20,
) -> Tuple[Dict[int, List[Tuple[Tuple[int, int, float], float]]], float]:
    """
    Price blocks for all items, returning top max_columns_per_item per item.

    Uses forced_y and forced_z duals to give "discounts" to columns that
    satisfy forced branching constraints, ensuring we generate all needed columns.

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
        forced_arcs = forced_z.get(item_id, set())
        forbidden_arcs = forbidden_z.get(item_id, set())
        existing = set(columns_by_item.get(item_id, []))
        cols, item_min_rc = price_block_dp(
            item_id,
            item_data,
            T,
            Gamma,
            duals,
            forced_periods,
            forbidden_periods,
            forced_arcs,
            forbidden_arcs,
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
    forced_z: Dict[int, Set[Tuple[int, int]]],
    forbidden_z: Dict[int, Set[Tuple[int, int]]],
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
            forced_z,
            forbidden_z,
        )

        if not math.isfinite(obj):
            return math.inf, {"status": "infeasible"}, columns_by_item, cg_iter, 0

        # Price new columns (returns top 5 per item + true min RC across all)
        # Pass forced_y/forced_z so their duals give "discounts" to relevant blocks
        new_columns_by_item, min_rc = price_all_items(
            items,
            T,
            Gamma_by_item,
            duals,
            forced_y,
            forbidden_y,
            forced_z,
            forbidden_z,
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

        # Pruning disabled - causes cyclic instability when pruned columns
        # are needed for feasibility (even if they have zero reduced cost)
        # The LP may need columns for constraint satisfaction even if they
        # don't improve the objective
        pass

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
        forced_z,
        forbidden_z,
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


def is_z_integer(z_vals: Dict[Tuple[int, int, int], float]) -> bool:
    """Check if all Z values are integer."""
    for val in z_vals.values():
        if EPS < val < 1.0 - EPS:
            return False
    return True


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


def find_most_fractional_z(
    z_vals: Dict[Tuple[int, int, int], float],
    forced_z: Dict[int, Set[Tuple[int, int]]],
    forbidden_z: Dict[int, Set[Tuple[int, int]]],
) -> Optional[Tuple[int, int, int, float]]:
    """Find the most fractional Z variable to branch on."""
    best_frac = 0.0
    best = None

    for (item_id, t, u), val in z_vals.items():
        if (t, u) in forced_z.get(item_id, set()):
            continue
        if (t, u) in forbidden_z.get(item_id, set()):
            continue

        frac = min(val, 1.0 - val)
        if frac > EPS and frac > best_frac:
            best_frac = frac
            best = (item_id, t, u, val)

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

    forced_z: Dict[int, Set[Tuple[int, int]]] = {i: set() for i in items}
    forbidden_z: Dict[int, Set[Tuple[int, int]]] = {i: set() for i in items}

    obj, solution, _ = solve_rmp_with_duals(
        items,
        T,
        capacity,
        Gamma_by_item,
        columns_pool,
        forced_y,
        forbidden_y,
        forced_z,
        forbidden_z,
        use_artificial=True,
    )

    if not math.isfinite(obj):
        return None, None

    # Check if truly feasible (no artificials)
    art_total = solution.get("art_total", 0.0)
    if art_total > EPS:
        return None, None

    # Check if integer
    y_dive = solution.get("y", {})
    z_dive = solution.get("z", {})
    if is_y_integer(y_dive) and is_z_integer(z_dive):
        return obj, solution

    return None, None


def try_dive(
    items: Dict[int, dict],
    T: int,
    capacity: List[float],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    columns_pool: Dict[int, List[Tuple[int, int, float]]],
    y_vals: Dict[Tuple[int, int], float],
    base_forced_y: Dict[int, Set[int]],
    base_forbidden_y: Dict[int, Set[int]],
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
) -> Tuple[float, Optional[Dict], float, float]:
    """
    Solve using block-based Branch-and-Price with column generation.

    Returns: (best_obj, best_solution, root_bound, final_gap)

    The final_gap is 0% when proven optimal (queue empty and all nodes processed).
    """
    start_time = time.time()

    root = BranchNode(
        node_id=0,
        parent_id=None,
        depth=0,
        forced_y={i: set() for i in items},
        forbidden_y={i: set() for i in items},
        forced_z={i: set() for i in items},
        forbidden_z={i: set() for i in items},
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
            root.forced_z,
            root.forbidden_z,
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
    z_vals = root_info.get("z", {})

    # Check if root is already integer (both Y and Z)
    if is_y_integer(y_vals) and is_z_integer(z_vals):
        if verbose:
            print("  ✓ Root is INTEGER (Y and Z) - OPTIMAL!")
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
    )
    if dive_obj is not None:
        best_ub = dive_obj
        best_solution = dive_sol
        if verbose:
            print(f"  🎯 Dive found incumbent: {best_ub:.4f}")
    else:
        if verbose:
            print("  Dive did not find feasible solution")

    # Priority queue: (bound, node_id, node)
    queue: List[Tuple[float, int, BranchNode]] = []
    heapq.heappush(queue, (root.lp_bound, root.node_id, root))

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

        # Update global lower bound (min of queue + best_ub)
        if queue:
            global_lb = min(q[0] for q in queue)
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

        # Column generation at this node
        obj, info, node_columns, cg_iters, cols_added = column_generation_loop(
            items,
            T,
            capacity,
            Gamma_by_item,
            columns_pool,
            node.forced_y,
            node.forbidden_y,
            node.forced_z,
            node.forbidden_z,
            node_id=node.node_id,
            logger=logger,
            verbose=False,  # Suppress CG iteration details
        )

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
            )
            if dive_obj is not None and (best_ub is None or dive_obj < best_ub - EPS):
                best_ub = dive_obj
                best_solution = dive_sol
                if verbose:
                    print(f"  🎯 Dive found better incumbent: {best_ub:.4f}")

        y_vals = info.get("y", {})
        z_vals = info.get("z", {})

        # Check integrality (both Y and Z)
        y_int = is_y_integer(y_vals)
        z_int = is_z_integer(z_vals)

        if y_int and z_int:
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

        # Branch on best Y variable (setup-cost weighted)
        branch_var_y = (
            find_best_branching_y(y_vals, node.forced_y, node.forbidden_y, items)
            if not y_int
            else None
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

            # Create children for Y branching
            for direction in [0, 1]:
                child = BranchNode(
                    node_id=node_counter,
                    parent_id=node.node_id,
                    depth=node.depth + 1,
                    forced_y={i: set(s) for i, s in node.forced_y.items()},
                    forbidden_y={i: set(s) for i, s in node.forbidden_y.items()},
                    forced_z={i: set(s) for i, s in node.forced_z.items()},
                    forbidden_z={i: set(s) for i, s in node.forbidden_z.items()},
                    lp_bound=obj,
                )

                if direction == 0:
                    child.forbidden_y[item_id].add(t)
                else:
                    child.forced_y[item_id].add(t)

                heapq.heappush(queue, (obj, child.node_id, child))
                node_counter += 1

            max_depth = max(max_depth, node.depth + 1)
            continue

        # Y is integer, branch on Z if fractional
        branch_var_z = (
            find_most_fractional_z(z_vals, node.forced_z, node.forbidden_z)
            if not z_int
            else None
        )

        if branch_var_z is not None:
            # Branch on Z
            item_id, t, u, val = branch_var_z

            if verbose:
                ub_str = f"{best_ub:.2f}" if best_ub else "N/A"
                print(
                    f"  [{nodes_explored}] Node {node.node_id} (d={node.depth}): "
                    f"LP={obj:.2f}, UB={ub_str} → branch Z[{item_id},{t},{u}]={val:.3f}"
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
                        direction=f"Z({u})",
                        status="BRANCHED_Z",
                        cg_iters=cg_iters,
                        columns_added=cols_added,
                    )
                )

            # Create children for Z branching
            for direction in [0, 1]:
                child = BranchNode(
                    node_id=node_counter,
                    parent_id=node.node_id,
                    depth=node.depth + 1,
                    forced_y={i: set(s) for i, s in node.forced_y.items()},
                    forbidden_y={i: set(s) for i, s in node.forbidden_y.items()},
                    forced_z={i: set(s) for i, s in node.forced_z.items()},
                    forbidden_z={i: set(s) for i, s in node.forbidden_z.items()},
                    lp_bound=obj,
                )

                if direction == 0:
                    child.forbidden_z[item_id].add((t, u))
                else:
                    child.forced_z[item_id].add((t, u))

                heapq.heappush(queue, (obj, child.node_id, child))
                node_counter += 1

            max_depth = max(max_depth, node.depth + 1)
            continue

        # Both Y and Z are integer but we didn't catch it above - treat as incumbent
        if best_ub is None or obj < best_ub - EPS:
            best_ub = obj
            best_solution = info
            if verbose:
                print(f"  ★ New incumbent (YZ-int): {best_ub:.4f} (depth={node.depth})")
        if logger:
            logger.log_node(
                NodeLogEntry(
                    node_id=node.node_id,
                    depth=node.depth,
                    lp_bound=obj,
                    incumbent=obj if best_ub is None or obj < best_ub else best_ub,
                    branch_item=None,
                    branch_t=None,
                    direction=None,
                    status="NEW_INCUMBENT",
                    cg_iters=cg_iters,
                    columns_added=cols_added,
                )
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
            global_lb = min(q[0] for q in queue) if queue else root_bound
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
        orders_txt = []
        for item_id in sorted(items.keys()):
            orders_txt.append(f"Item {item_id} — orders (t → qty)")
            production = x_agg.get(item_id, {})
            for t in sorted(production.keys()):
                qty = production[t]
                if qty > EPS:
                    orders_txt.append(f" {t:2d} → {qty:8.3f}")
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
    instance = {
        "period": 12,
        "manual_capacity": [0, 0, 100, 120, 110, 100, 90, 100, 110, 120, 100, 80],
        "items": {
            "0": {
                "h": [
                    0.5,
                    0.52,
                    0.54,
                    0.56,
                    0.58,
                    0.6,
                    0.58,
                    0.56,
                    0.54,
                    0.52,
                    0.5,
                    0.48,
                ],
                "c_var": [2.0, 1.8, 2.5, 1.9, 1.7, 2.2, 1.5, 1.3, 2.7, 1.9, 2.1, 1.6],
                "setup": [85, 86, 87, 88, 89, 90, 91, 92, 91, 90, 89, 88],
                "demand": [0, 0, 45, 32, 55, 28, 15, 20, 38, 42, 30, 25],
                "shelf_seq": [12, 10, 8, 6, 8, 5, 7, 9, 6, 5, 4, 3],
            },
            "1": {
                "h": [
                    0.3,
                    0.32,
                    0.34,
                    0.36,
                    0.38,
                    0.4,
                    0.38,
                    0.36,
                    0.34,
                    0.32,
                    0.3,
                    0.28,
                ],
                "c_var": [1.5, 1.4, 2.0, 1.6, 1.3, 1.8, 1.2, 1.1, 2.2, 1.5, 1.7, 1.3],
                "setup": [70, 71, 72, 73, 74, 75, 76, 77, 76, 75, 74, 73],
                "demand": [0, 0, 30, 25, 40, 20, 12, 18, 35, 28, 22, 18],
                "shelf_seq": [10, 8, 7, 5, 6, 4, 6, 8, 5, 4, 3, 2],
            },
            "2": {
                "h": [
                    0.6,
                    0.62,
                    0.64,
                    0.66,
                    0.68,
                    0.7,
                    0.68,
                    0.66,
                    0.64,
                    0.62,
                    0.6,
                    0.58,
                ],
                "c_var": [2.5, 2.3, 3.0, 2.4, 2.1, 2.8, 1.9, 1.7, 3.2, 2.4, 2.6, 2.0],
                "setup": [95, 96, 97, 98, 99, 100, 101, 102, 101, 100, 99, 98],
                "demand": [0, 0, 25, 18, 35, 15, 10, 12, 28, 22, 16, 14],
                "shelf_seq": [8, 6, 5, 4, 5, 3, 5, 6, 4, 3, 2, 2],
            },
        },
    }
    # Small instance (1 item, 10 periods):
    # instance = {
    #     "period": 10,
    #     "manual_capacity": [0, 0, 80, 80, 80, 80, 80, 80, 100, 80],
    #     "items": {
    #         "0": {
    #             "h": [
    #                 0.4,
    #                 0.408,
    #                 0.416,
    #                 0.424,
    #                 0.430,
    #                 0.435,
    #                 0.438,
    #                 0.440,
    #                 0.440,
    #                 0.438,
    #             ],
    #             "c_var": [1.76, 1.71, 2.50, 1.78, 1.67, 2.11, 1.37, 1.24, 2.68, 1.81],
    #             "setup": [
    #                 80,
    #                 81.66,
    #                 83.25,
    #                 84.70,
    #                 85.95,
    #                 86.93,
    #                 87.61,
    #                 87.96,
    #                 87.96,
    #                 87.61,
    #             ],
    #             "demand": [0, 0, 68, 49, 66, 38, 17, 17, 41, 43],
    #             "shelf_seq": [24, 18, 22, 6, 16, 9, 21, 23, 9, 7],
    #         }
    #     },
    # }

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
