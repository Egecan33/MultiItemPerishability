"""
Branch-and-Price for Perishable Lot-Sizing with Heterogeneous Shelf Lives and LEFO
Version 10: Memory-optimized with Best-First Search

Features:
- Best-first search using heap ordered by LP bound (lowest first)
- Memory optimization: queue stores only (bound, node_id, node, rmp)
- Column inheritance for faster convergence at branch nodes
- σ (sigma) and τ (tau) duals for forbidden branches to guide pricing
- Dummy column detection to prevent false integer solutions
- Branching on Z (arc) and Y (setup) variables
- DP-based pricing for ZIO columns
- Comprehensive logging of all solver details
"""

from __future__ import annotations
import csv
import heapq
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, IO
import gurobipy as gp
from gurobipy import GRB

# Global output directory for convergence tracking
_convergence_out_dir: Optional[Path] = None


@dataclass
class ProductionPlanColumn:
    """Represents a single production plan (column) for one item."""

    item_id: int
    total_plan_cost: float
    capacity_usage_by_period: List[float]
    setup_by_period: List[float]
    arc_usage: Dict[Tuple[int, int], float]

    def violates_branching_constraints(
        self,
        theta_0: Set[Tuple[int, int]],
        theta_1: Set[Tuple[int, int]],
        upsilon_0: Set[int],
        upsilon_1: Set[int],
        eps: float = 1e-6,
    ) -> bool:
        """Check if column violates branching constraints."""
        for t, u in theta_0:
            if (t, u) in self.arc_usage and self.arc_usage[(t, u)] > eps:
                return True
        for t, u in theta_1:
            if (t, u) not in self.arc_usage or self.arc_usage[(t, u)] < 1.0 - eps:
                return True
        for t in upsilon_0:
            if t < len(self.setup_by_period) and self.setup_by_period[t] > eps:
                return True
        for t in upsilon_1:
            if t >= len(self.setup_by_period) or self.setup_by_period[t] < 1.0 - eps:
                return True
        return False

    def get_signature(self) -> str:
        """Generate a unique signature for duplicate detection."""
        setup_str = ",".join(
            f"{t}" for t, v in enumerate(self.setup_by_period) if v > 0.5
        )
        arc_str = ",".join(
            f"{t}-{u}"
            for (t, u) in sorted(self.arc_usage.keys())
            if self.arc_usage[(t, u)] > 0.5
        )
        return f"I{self.item_id}_S[{setup_str}]_A[{arc_str}]"


@dataclass
class BranchNode:
    """Represents a node in the branch-and-bound tree."""

    node_id: int
    parent_id: Optional[int]
    depth: int
    theta_0_by_item: Dict[int, Set[Tuple[int, int]]] = field(default_factory=dict)
    theta_1_by_item: Dict[int, Set[Tuple[int, int]]] = field(default_factory=dict)
    upsilon_0_by_item: Dict[int, Set[int]] = field(default_factory=dict)
    upsilon_1_by_item: Dict[int, Set[int]] = field(default_factory=dict)
    lp_bound: float = math.inf
    is_integer: bool = False
    is_pruned: bool = False
    prune_reason: Optional[str] = None
    branch_variable: Optional[Tuple] = None
    branch_direction: Optional[str] = None

    def __lt__(self, other: "BranchNode") -> bool:
        """For heap comparison - lower bound = higher priority."""
        return self.lp_bound < other.lp_bound


@dataclass
class SearchStatistics:
    """Tracks search statistics for the branch-and-price algorithm."""

    nodes_created: int = 0
    nodes_explored: int = 0
    nodes_integer: int = 0
    nodes_fathomed_by_bound: int = 0
    nodes_fathomed_by_infeasible: int = 0
    nodes_fathomed_integer: int = 0
    nodes_fathomed_on_incumbent: int = 0
    max_depth: int = 0
    total_columns_generated: int = 0
    total_cg_iterations: int = 0
    start_time: float = field(default_factory=time.time)

    def print_summary(
        self, best_lb: float, best_ub: Optional[float], eps: float, log_file=None
    ):
        elapsed = time.time() - self.start_time

        lines = []
        lines.append("\n" + "=" * 70)
        lines.append(" " * 28 + "FINAL RESULTS")
        lines.append("=" * 70)
        lines.append(f"  Time elapsed:          {elapsed:.2f} seconds")
        lines.append(f"  Nodes created:         {self.nodes_created}")
        lines.append(f"  Nodes explored:        {self.nodes_explored}")
        lines.append(f"  Integer solutions:     {self.nodes_integer}")
        lines.append(f"  Max depth:             {self.max_depth}")
        lines.append(f"  Total CG iterations:   {self.total_cg_iterations}")
        lines.append(f"  Total columns added:   {self.total_columns_generated}")
        lines.append("")
        lines.append(f"  Fathomed by bound:     {self.nodes_fathomed_by_bound}")
        lines.append(f"  Fathomed infeasible:   {self.nodes_fathomed_by_infeasible}")
        lines.append(f"  Fathomed on incumbent: {self.nodes_fathomed_on_incumbent}")
        lines.append("")
        lines.append(f"  Best lower bound:      {best_lb:.4f}")

        if best_ub is not None:
            lines.append(f"  Best upper bound:      {best_ub:.4f}")
            gap = best_ub - best_lb
            gap_pct = 100 * gap / max(abs(best_ub), 1e-10)
            lines.append(f"  Gap:                   {gap:.4f} ({gap_pct:.2f}%)")
            if gap < eps:
                lines.append(f"\n  ★★★ PROVEN OPTIMAL! ★★★")
        else:
            lines.append(f"  Best upper bound:      Not found")
        lines.append("=" * 70)

        for line in lines:
            print(line)
            if log_file:
                log_file.write(line + "\n")


def _cap_global_from_dem(items: Dict[int, dict], T: int) -> List[float]:
    """Generate default capacity from total demand with buffer."""
    cap_raw = [0.0] * T
    for it in items.values():
        dem = it["demand"]
        for t in range(T):
            cap_raw[t] += float(dem[t])
    buf = max(5.0, 0.2 * max(cap_raw) if cap_raw else 0.0)
    return [c + buf for c in cap_raw]


def _as_len_T_vector(val, T: int) -> List[float]:
    """Convert a scalar or list to a length-T vector."""
    if val is None:
        return []
    if isinstance(val, (int, float)):
        return [float(val)] * T
    if isinstance(val, list):
        if len(val) != T:
            raise ValueError(f"Expected length-{T} list, got {len(val)}")
        return [float(x) for x in val]
    raise TypeError("Value must be a number or a list")


def node_signature(node: BranchNode) -> str:
    """Generate a unique signature for a branch node to detect duplicates."""
    sig_parts = []
    for item_id in sorted(node.theta_0_by_item.keys()):
        arcs = sorted(node.theta_0_by_item[item_id])
        if arcs:
            sig_parts.append(f"I{item_id}_Z0:{','.join(f'{t}-{u}' for t, u in arcs)}")
    for item_id in sorted(node.theta_1_by_item.keys()):
        arcs = sorted(node.theta_1_by_item[item_id])
        if arcs:
            sig_parts.append(f"I{item_id}_Z1:{','.join(f'{t}-{u}' for t, u in arcs)}")
    for item_id in sorted(node.upsilon_0_by_item.keys()):
        periods = sorted(node.upsilon_0_by_item[item_id])
        if periods:
            sig_parts.append(f"I{item_id}_Y0:{','.join(map(str, periods))}")
    for item_id in sorted(node.upsilon_1_by_item.keys()):
        periods = sorted(node.upsilon_1_by_item[item_id])
        if periods:
            sig_parts.append(f"I{item_id}_Y1:{','.join(map(str, periods))}")
    return "|".join(sig_parts)


def inherit_columns_from_parent(
    parent_rmp: Optional["RestrictedMasterProblem"],
    child_node: BranchNode,
    items: Dict[int, dict],
    eps: float = 1e-6,
) -> Dict[int, List[ProductionPlanColumn]]:
    """Filter parent's columns to get feasible columns for child."""
    if parent_rmp is None:
        return {i: [] for i in items}

    inherited = {i: [] for i in items}
    for item_id in items:
        theta_0 = child_node.theta_0_by_item.get(item_id, set())
        theta_1 = child_node.theta_1_by_item.get(item_id, set())
        upsilon_0 = child_node.upsilon_0_by_item.get(item_id, set())
        upsilon_1 = child_node.upsilon_1_by_item.get(item_id, set())

        for col in parent_rmp.columns[item_id][1:]:  # Skip dummy column
            if not col.violates_branching_constraints(
                theta_0, theta_1, upsilon_0, upsilon_1, eps
            ):
                inherited[item_id].append(col)

    return inherited


def solve_pricing_subproblem(
    item_id: int,
    item_data: dict,
    T: int,
    Gamma: Dict[int, List[int]],
    Expiry: Dict[int, int],
    capacity_duals: List[float],
    convexity_dual: float,
    theta_0: Set[Tuple[int, int]],
    theta_1: Set[Tuple[int, int]],
    upsilon_0: Set[int],
    upsilon_1: Set[int],
    sigma: Optional[Dict[Tuple[int, int], float]] = None,
    tau: Optional[Dict[Tuple[int, int, int], float]] = None,
    eps: float = 1e-6,
    use_mip: bool = False,
) -> Tuple[float, Optional[ProductionPlanColumn]]:
    """
    DP-based pricing subproblem for ZIO columns with COMPLETE Wagner-Whitin coverage.
    """
    sigma = sigma or {}
    tau = tau or {}

    demand = item_data["demand"]
    c_var = item_data["c_var"]
    h = item_data["h"]
    setup = item_data["setup"]

    def c_at(t: int) -> float:
        return float(c_var[t]) if isinstance(c_var, list) else float(c_var)

    def h_at(t: int) -> float:
        return float(h[t]) if isinstance(h, list) else float(h)

    def s_at(t: int) -> float:
        return float(setup[t]) if isinstance(setup, list) else float(setup)

    h_prefix = [0.0] * (T + 1)
    for k in range(T):
        h_prefix[k + 1] = h_prefix[k] + h_at(k)

    def h_sum(t: int, u: int) -> float:
        return h_prefix[u] - h_prefix[t]

    INF = float("inf")

    for t in upsilon_1:
        if t in upsilon_0:
            return INF, None

    forced_source: Dict[int, int] = {}
    for t, u in theta_1:
        if u in forced_source and forced_source[u] != t:
            return INF, None
        forced_source[u] = t
        if t in upsilon_0:
            return INF, None

    must_produce_at: Set[int] = set()
    for t, u in theta_1:
        must_produce_at.add(t)

    min_end_for_forced: Dict[int, int] = {}
    for t, u in theta_1:
        if t not in min_end_for_forced:
            min_end_for_forced[t] = u
        else:
            min_end_for_forced[t] = max(min_end_for_forced[t], u)

    def can_produce_at(t: int) -> bool:
        if t in upsilon_0:
            return False
        if t not in Gamma:
            return False
        return len(Gamma.get(t, [])) > 0

    def get_valid_ends(t: int) -> List[int]:
        if not can_produce_at(t):
            return []
        valid_ends = Gamma.get(t, [])
        if not valid_ends:
            return []
        result = []
        min_s = min_end_for_forced.get(t, t)
        for s in valid_ends:
            if s < min_s:
                continue
            arc_valid = True
            for u in range(t, s + 1):
                if (t, u) in theta_0:
                    arc_valid = False
                    break
            if not arc_valid:
                continue
            source_conflict = False
            for u in range(t, s + 1):
                if u in forced_source and forced_source[u] != t:
                    source_conflict = True
                    break
            if source_conflict:
                continue
            result.append(s)
        return result

    def run_reduced_cost(t: int, s: int) -> float:
        cost = s_at(t)
        cost -= sigma.get((item_id, t), 0.0)
        for u in range(t, s + 1):
            d_u = float(demand[u])
            if d_u > 0:
                unit_cost = c_at(t) + h_sum(t, u) - capacity_duals[t]
                cost += unit_cost * d_u
                cost -= tau.get((item_id, t, u), 0.0)
        return cost

    dp = [INF] * (T + 1)
    decision = [(-1, -1)] * T
    dp[T] = 0

    for t in range(T - 1, -1, -1):
        if t in forced_source and forced_source[t] != t:
            dp[t] = dp[t + 1]
            decision[t] = (-1, -1)
            continue

        best_cost = INF
        best_action = (-1, -1)

        can_skip = (
            (t not in must_produce_at)
            and (t not in upsilon_1)
            and float(demand[t]) <= 0
        )
        if can_skip:
            skip_cost = dp[t + 1]
            if skip_cost < best_cost:
                best_cost = skip_cost
                best_action = (-1, -1)

        can_setup_only = (
            (t not in upsilon_0)
            and (t not in must_produce_at)
            and float(demand[t]) <= 0
        )
        if can_setup_only:
            setup_cost = s_at(t) - sigma.get((item_id, t), 0.0) + dp[t + 1]
            if setup_cost < best_cost:
                best_cost = setup_cost
                best_action = (0, -1)

        if can_produce_at(t):
            valid_ends = get_valid_ends(t)
            for s in valid_ends:
                if s + 1 <= T and dp[s + 1] < INF:
                    cost = run_reduced_cost(t, s) + dp[s + 1]
                    if cost < best_cost:
                        best_cost = cost
                        best_action = (1, s)

        if t in must_produce_at and best_action[0] != 1:
            dp[t] = INF
            continue

        if t in upsilon_1 and best_action[0] == -1:
            if can_setup_only:
                setup_cost = s_at(t) - sigma.get((item_id, t), 0.0) + dp[t + 1]
                if setup_cost < INF:
                    best_cost = setup_cost
                    best_action = (0, -1)
            else:
                dp[t] = INF
                continue

        dp[t] = best_cost
        decision[t] = best_action

    if dp[0] >= INF:
        return INF, None

    reduced_cost = dp[0] - convexity_dual

    if reduced_cost >= -eps:
        return reduced_cost, None

    cap_usage = [0.0] * T
    setup_usage = [0.0] * T
    arc_usage: Dict[Tuple[int, int], float] = {}
    total_cost = 0.0

    t = 0
    while t < T:
        action, s = decision[t]
        if action == -1:
            t += 1
            continue
        elif action == 0:
            setup_usage[t] = 1.0
            total_cost += s_at(t)
            t += 1
        elif action == 1:
            setup_usage[t] = 1.0
            total_cost += s_at(t)
            for u in range(t, s + 1):
                d_u = float(demand[u])
                if d_u > 0:
                    cap_usage[t] += d_u
                    arc_usage[(t, u)] = 1.0
                    total_cost += (c_at(t) + h_sum(t, u)) * d_u
            t = s + 1

    for t_forced in upsilon_1:
        if t_forced < T and setup_usage[t_forced] < 0.5:
            setup_usage[t_forced] = 1.0
            total_cost += s_at(t_forced)

    covered = [False] * T
    for (t_prod, u), val in arc_usage.items():
        if val > 0.5 and u < T:
            covered[u] = True

    for u in range(T):
        if demand[u] > 0 and not covered[u]:
            return INF, None

    for t_force, u_force in theta_1:
        if arc_usage.get((t_force, u_force), 0.0) < 0.5:
            return INF, None

    column = ProductionPlanColumn(
        item_id=item_id,
        total_plan_cost=total_cost,
        capacity_usage_by_period=cap_usage,
        setup_by_period=setup_usage,
        arc_usage=arc_usage,
    )

    return reduced_cost, column


class RestrictedMasterProblem:
    """Restricted Master Problem for the Dantzig-Wolfe decomposition."""

    def __init__(
        self,
        items: Dict[int, dict],
        T: int,
        capacity: List[float],
        Gamma_by_item: Dict[int, Dict[int, List[int]]],
        theta_0_by_item: Optional[Dict[int, Set[Tuple[int, int]]]] = None,
        upsilon_0_by_item: Optional[Dict[int, Set[int]]] = None,
        initial_columns: Optional[Dict[int, List[ProductionPlanColumn]]] = None,
    ):
        self.items = items
        self.T = T
        self.capacity = capacity
        self.Gamma_by_item = Gamma_by_item
        self.theta_0_by_item = theta_0_by_item or {i: set() for i in items}
        self.upsilon_0_by_item = upsilon_0_by_item or {i: set() for i in items}

        self.model = gp.Model("RMP")
        self.model.Params.OutputFlag = 0
        self.model.Params.LogToConsole = 0
        self.model.Params.Method = 1

        self.columns: Dict[int, List[ProductionPlanColumn]] = {i: [] for i in items}
        self.lambdas: Dict[Tuple[int, int], gp.Var] = {}

        self.convex_expr: Dict[int, gp.LinExpr] = {i: gp.LinExpr(0.0) for i in items}
        self.cap_expr: List[gp.LinExpr] = [gp.LinExpr(0.0) for _ in range(T)]

        self.y_link_expr: Dict[Tuple[int, int], gp.LinExpr] = {}
        for item_id in items:
            for t in self.upsilon_0_by_item.get(item_id, set()):
                self.y_link_expr[(item_id, t)] = gp.LinExpr(0.0)

        self.z_link_expr: Dict[Tuple[int, int, int], gp.LinExpr] = {}
        for item_id in items:
            for t, u in self.theta_0_by_item.get(item_id, set()):
                self.z_link_expr[(item_id, t, u)] = gp.LinExpr(0.0)

        self.convex_con: Dict[int, gp.Constr] = {}
        self.cap_con: List[gp.Constr] = []
        self.y_link_con: Dict[Tuple[int, int], gp.Constr] = {}
        self.z_link_con: Dict[Tuple[int, int, int], gp.Constr] = {}

        for item_id in items:
            self.convex_con[item_id] = self.model.addConstr(
                self.convex_expr[item_id] == 1.0, name=f"conv_{item_id}"
            )

        for t in range(T):
            self.cap_con.append(
                self.model.addConstr(
                    self.cap_expr[t] <= self.capacity[t], name=f"cap_{t}"
                )
            )

        for (item_id, t), expr in self.y_link_expr.items():
            self.y_link_con[(item_id, t)] = self.model.addConstr(
                expr == 0.0, name=f"y_link_{item_id}_{t}"
            )

        for (item_id, t, u), expr in self.z_link_expr.items():
            self.z_link_con[(item_id, t, u)] = self.model.addConstr(
                expr == 0.0, name=f"z_link_{item_id}_{t}_{u}"
            )

        if initial_columns:
            for item_id, cols in initial_columns.items():
                for col in cols:
                    self.add_column(col)

    def _rebuild(self):
        for item_id in self.items:
            self.model.remove(self.convex_con[item_id])
            self.convex_con[item_id] = self.model.addConstr(
                self.convex_expr[item_id] == 1.0, name=f"conv_{item_id}"
            )
        for t in range(self.T):
            self.model.remove(self.cap_con[t])
            self.cap_con[t] = self.model.addConstr(
                self.cap_expr[t] <= self.capacity[t], name=f"cap_{t}"
            )
        for key, expr in self.y_link_expr.items():
            self.model.remove(self.y_link_con[key])
            self.y_link_con[key] = self.model.addConstr(
                expr == 0.0, name=f"y_link_{key[0]}_{key[1]}"
            )
        for key, expr in self.z_link_expr.items():
            self.model.remove(self.z_link_con[key])
            self.z_link_con[key] = self.model.addConstr(
                expr == 0.0, name=f"z_link_{key[0]}_{key[1]}_{key[2]}"
            )

    def add_column(self, col: ProductionPlanColumn):
        item_id = col.item_id
        idx = len(self.columns[item_id])

        lam = self.model.addVar(
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            obj=col.total_plan_cost,
            name=f"lam_{item_id}_{idx}",
        )

        self.lambdas[(item_id, idx)] = lam
        self.columns[item_id].append(col)
        self.convex_expr[item_id] += lam

        for t in range(self.T):
            if col.capacity_usage_by_period[t] != 0.0:
                self.cap_expr[t] += col.capacity_usage_by_period[t] * lam

        for t in self.upsilon_0_by_item.get(item_id, set()):
            if t < len(col.setup_by_period):
                y_val = col.setup_by_period[t]
                if y_val != 0.0:
                    self.y_link_expr[(item_id, t)] += y_val * lam

        for t, u in self.theta_0_by_item.get(item_id, set()):
            z_val = col.arc_usage.get((t, u), 0.0)
            if z_val != 0.0:
                self.z_link_expr[(item_id, t, u)] += z_val * lam

        self._rebuild()

    def solve(self) -> Tuple[
        float,
        Dict[int, float],
        List[float],
        Dict[Tuple[int, int], float],
        Dict[Tuple[int, int, int], float],
    ]:
        self.model.optimize()

        if self.model.status != GRB.OPTIMAL:
            return math.inf, {}, [], {}, {}

        mu = {i: self.convex_con[i].Pi for i in self.items}
        pi = [self.cap_con[t].Pi for t in range(self.T)]

        sigma: Dict[Tuple[int, int], float] = {}
        for (item_id, t), con in self.y_link_con.items():
            sigma[(item_id, t)] = con.Pi

        tau: Dict[Tuple[int, int, int], float] = {}
        for (item_id, t, u), con in self.z_link_con.items():
            tau[(item_id, t, u)] = con.Pi

        return self.model.ObjVal, mu, pi, sigma, tau

    def get_column_count(self) -> int:
        return sum(len(cols) for cols in self.columns.values())


def solve_node_with_column_generation(
    items: Dict[int, dict],
    T: int,
    capacity: List[float],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    Expiry_by_item: Dict[int, Dict[int, int]],
    node: BranchNode,
    parent_rmp: Optional[RestrictedMasterProblem] = None,
    max_iter: int = 500,
    eps: float = 1e-6,
    verbose: bool = False,
    log_file=None,
    stats: Optional[SearchStatistics] = None,
    convergence_dir: Optional[Path] = None,
) -> Tuple[
    float,
    Optional[RestrictedMasterProblem],
    bool,
    Dict[int, Dict[Tuple[int, int], float]],
    Dict[int, Dict[int, float]],
    Dict[int, Dict[int, float]],
]:
    """Solve a branch node using column generation with column inheritance."""
    inherited_cols = inherit_columns_from_parent(parent_rmp, node, items, eps)
    total_inherited = sum(len(cols) for cols in inherited_cols.values())

    rmp = RestrictedMasterProblem(
        items=items,
        T=T,
        capacity=capacity,
        Gamma_by_item=Gamma_by_item,
        theta_0_by_item=node.theta_0_by_item,
        upsilon_0_by_item=node.upsilon_0_by_item,
        initial_columns=inherited_cols,
    )

    # Add dummy columns
    for item_id, item_data in items.items():
        demand = item_data["demand"]
        total_demand = sum(demand)
        dummy_cost = 10000.0 * (total_demand + 1.0)

        col = ProductionPlanColumn(
            item_id=item_id,
            total_plan_cost=dummy_cost,
            capacity_usage_by_period=[0.0] * T,
            setup_by_period=[0.0] * T,
            arc_usage={},
        )
        rmp.add_column(col)

    if verbose:
        msg = f"  └─ CG[inherited={total_inherited}]: "
        print(msg, end="", flush=True)
        if log_file:
            log_file.write(msg)

    columns_added_this_node = 0
    iterations_this_node = 0

    # Setup convergence CSV tracking
    csv_file = None
    csv_writer = None
    item_ids_sorted = sorted(items.keys())
    if convergence_dir is not None:
        csv_path = convergence_dir / f"convergence_node_{node.node_id}.csv"
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        # Header includes lambda sum for each item
        header = [
            "iteration",
            "rmp_obj",
            "worst_rc",
            "total_rc",
            "cols_added",
            "total_cols",
        ]
        for item_id in item_ids_sorted:
            header.append(f"lambda_sum_i{item_id}")
        csv_writer.writerow(header)

    try:
        for iteration in range(1, max_iter + 1):
            lb, mu, pi, sigma, tau = rmp.solve()
            iterations_this_node += 1

            # Calculate lambda sums right after solving (before any modifications)
            lambda_sums = []
            if csv_writer and math.isfinite(lb):
                for item_id in item_ids_sorted:
                    lam_sum = 0.0
                    for (iid, idx), lam_var in rmp.lambdas.items():
                        if iid == item_id:
                            lam_sum += lam_var.X
                    lambda_sums.append(f"{lam_sum:.6f}")

            if not math.isfinite(lb):
                if csv_writer:
                    row = [iteration, "inf", "N/A", "N/A", 0, rmp.get_column_count()]
                    row += ["N/A"] * len(item_ids_sorted)
                    csv_writer.writerow(row)
                if verbose:
                    print("INFEASIBLE")
                    if log_file:
                        log_file.write("INFEASIBLE\n")
                return math.inf, rmp, False, {}, {}, {}

            any_added = False
            cols_this_iter = 0
            worst_rc = 0.0
            total_rc = 0.0

            for item_id, item_data in items.items():
                theta_0 = node.theta_0_by_item.get(item_id, set())
                theta_1 = node.theta_1_by_item.get(item_id, set())
                upsilon_0 = node.upsilon_0_by_item.get(item_id, set())
                upsilon_1 = node.upsilon_1_by_item.get(item_id, set())

                Gamma = Gamma_by_item[item_id]
                Expiry = Expiry_by_item[item_id]

                rc, col = solve_pricing_subproblem(
                    item_id=item_id,
                    item_data=item_data,
                    T=T,
                    Gamma=Gamma,
                    Expiry=Expiry,
                    capacity_duals=pi,
                    convexity_dual=mu[item_id],
                    theta_0=theta_0,
                    theta_1=theta_1,
                    upsilon_0=upsilon_0,
                    upsilon_1=upsilon_1,
                    sigma=sigma,
                    tau=tau,
                    eps=eps,
                )

                if math.isfinite(rc):
                    total_rc += rc
                    worst_rc = min(worst_rc, rc)

                if col is not None and rc < -eps:
                    rmp.add_column(col)
                    any_added = True
                    columns_added_this_node += 1
                    cols_this_iter += 1

            # Write convergence data (using lambda_sums captured earlier)
            if csv_writer:
                row = [
                    iteration,
                    f"{lb:.6f}",
                    f"{worst_rc:.10f}",
                    f"{total_rc:.10f}",
                    cols_this_iter,
                    rmp.get_column_count(),
                ] + lambda_sums
                csv_writer.writerow(row)

            if not any_added:
                if verbose:
                    msg = f"LB={lb:.2f} (iter={iteration}, cols={columns_added_this_node})"
                    print(msg)
                    if log_file:
                        log_file.write(msg + "\n")

                if stats:
                    stats.total_cg_iterations += iterations_this_node
                    stats.total_columns_generated += columns_added_this_node

                z_vals = extract_z_values(rmp, items, eps)
                y_vals = extract_y_values(rmp, items, eps)
                x_vals = extract_x_values(rmp, items, eps)
                return lb, rmp, True, z_vals, y_vals, x_vals

        lb, _, _, _, _ = rmp.solve()

        if stats:
            stats.total_cg_iterations += iterations_this_node
            stats.total_columns_generated += columns_added_this_node

        z_vals = extract_z_values(rmp, items, eps)
        y_vals = extract_y_values(rmp, items, eps)
        x_vals = extract_x_values(rmp, items, eps)

        if verbose:
            msg = f"LB={lb:.2f} (max iter, cols={columns_added_this_node})"
            print(msg)
            if log_file:
                log_file.write(msg + "\n")

        return lb, rmp, False, z_vals, y_vals, x_vals

    finally:
        if csv_file:
            csv_file.close()


def extract_z_values(
    rmp: RestrictedMasterProblem,
    items: Dict[int, dict],
    eps: float,
) -> Dict[int, Dict[Tuple[int, int], float]]:
    z_vals = {i: {} for i in items}
    for (item_id, idx), lam in rmp.lambdas.items():
        lam_val = lam.X
        if lam_val < eps:
            continue
        col = rmp.columns[item_id][idx]
        for (t, u), z_val in col.arc_usage.items():
            if z_val > eps:
                z_vals[item_id][(t, u)] = (
                    z_vals[item_id].get((t, u), 0.0) + lam_val * z_val
                )
    return z_vals


def extract_y_values(
    rmp: RestrictedMasterProblem,
    items: Dict[int, dict],
    eps: float,
) -> Dict[int, Dict[int, float]]:
    y_vals = {i: {} for i in items}
    for (item_id, idx), lam in rmp.lambdas.items():
        lam_val = lam.X
        if lam_val < eps:
            continue
        col = rmp.columns[item_id][idx]
        for t, y_val in enumerate(col.setup_by_period):
            if y_val > eps:
                y_vals[item_id][t] = y_vals[item_id].get(t, 0.0) + lam_val * y_val
    return y_vals


def extract_x_values(
    rmp: RestrictedMasterProblem,
    items: Dict[int, dict],
    eps: float,
) -> Dict[int, Dict[int, float]]:
    x_vals = {i: {} for i in items}
    for (item_id, idx), lam in rmp.lambdas.items():
        lam_val = lam.X
        if lam_val < eps:
            continue
        col = rmp.columns[item_id][idx]
        for t, qty in enumerate(col.capacity_usage_by_period):
            if qty > eps:
                x_vals[item_id][t] = x_vals[item_id].get(t, 0.0) + lam_val * qty
    return x_vals


def extract_active_columns(
    rmp: RestrictedMasterProblem,
    items: Dict[int, dict],
    eps: float = 1e-6,
) -> List[Dict]:
    """Extract active (non-dummy) columns with their lambda values."""
    active_cols = []
    for (item_id, idx), lam in rmp.lambdas.items():
        lam_val = lam.X
        if lam_val < eps:
            continue

        col = rmp.columns[item_id][idx]

        # Check if this is a dummy column by its properties (not index!)
        item_demand = sum(items[item_id]["demand"])
        dummy_cost_threshold = 5000.0 * (item_demand + 1)
        is_dummy = (
            col.total_plan_cost > dummy_cost_threshold
            and all(x == 0.0 for x in col.capacity_usage_by_period)
            and len(col.arc_usage) == 0
        )
        if is_dummy:
            continue

        setups = [t for t, v in enumerate(col.setup_by_period) if v > 0.5]
        arcs = [(t, u) for (t, u), v in col.arc_usage.items() if v > 0.5]
        active_cols.append(
            {
                "item_id": item_id,
                "col_idx": idx,
                "lambda_val": lam_val,
                "cost": col.total_plan_cost,
                "setups": setups,
                "arcs": arcs,
                "capacity_usage": [q for q in col.capacity_usage_by_period],
            }
        )
    return active_cols


def is_integer(
    z_vals: Dict[int, Dict[Tuple[int, int], float]],
    y_vals: Dict[int, Dict[int, float]],
    eps: float = 1e-6,
) -> bool:
    for item_arcs in z_vals.values():
        for val in item_arcs.values():
            if eps < val < 1.0 - eps:
                return False
    for item_setups in y_vals.values():
        for val in item_setups.values():
            if eps < val < 1.0 - eps:
                return False
    return True


def solution_uses_dummy(
    rmp: RestrictedMasterProblem,
    items: Dict[int, dict],
    eps: float = 1e-6,
) -> bool:
    for item_id in items:
        item_demand = sum(items[item_id]["demand"])
        dummy_cost_threshold = 5000.0 * (item_demand + 1)
        for idx, col in enumerate(rmp.columns[item_id]):
            is_dummy = (
                col.total_plan_cost > dummy_cost_threshold
                and all(x == 0.0 for x in col.capacity_usage_by_period)
                and len(col.arc_usage) == 0
            )
            if is_dummy:
                lam_key = (item_id, idx)
                if lam_key in rmp.lambdas:
                    lam_val = rmp.lambdas[lam_key].X
                    if lam_val > eps:
                        return True
    return False


def is_valid_integer_solution(
    z_vals: Dict[int, Dict[Tuple[int, int], float]],
    y_vals: Dict[int, Dict[int, float]],
    rmp: RestrictedMasterProblem,
    items: Dict[int, dict],
    eps: float = 1e-6,
) -> bool:
    if not is_integer(z_vals, y_vals, eps):
        return False
    if solution_uses_dummy(rmp, items, eps):
        return False
    all_z_empty = all(len(arcs) == 0 for arcs in z_vals.values())
    all_y_empty = all(len(setups) == 0 for setups in y_vals.values())
    if all_z_empty and all_y_empty:
        return False
    return True


def find_most_fractional_z(
    z_vals: Dict[int, Dict[Tuple[int, int], float]],
    node: BranchNode,
    eps: float = 1e-6,
) -> Optional[Tuple[int, int, int, float]]:
    best_frac = 0.0
    best = None
    for item_id, arcs in z_vals.items():
        theta_0 = node.theta_0_by_item.get(item_id, set())
        theta_1 = node.theta_1_by_item.get(item_id, set())
        for (t, u), val in arcs.items():
            if (t, u) in theta_0 or (t, u) in theta_1:
                continue
            frac = min(val, 1.0 - val)
            if frac > eps and frac > best_frac:
                best_frac = frac
                best = (item_id, t, u, val)
    return best


def find_most_fractional_y(
    y_vals: Dict[int, Dict[int, float]],
    node: BranchNode,
    eps: float = 1e-6,
) -> Optional[Tuple[int, int, float]]:
    best_frac = 0.0
    best = None
    for item_id, setups in y_vals.items():
        upsilon_0 = node.upsilon_0_by_item.get(item_id, set())
        upsilon_1 = node.upsilon_1_by_item.get(item_id, set())
        for t, val in setups.items():
            if t in upsilon_0 or t in upsilon_1:
                continue
            frac = min(val, 1.0 - val)
            if frac > eps and frac > best_frac:
                best_frac = frac
                best = (item_id, t, val)
    return best


def fathom_heap_by_incumbent(
    heap: List[Tuple[float, int, BranchNode, RestrictedMasterProblem]],
    incumbent: float,
    eps: float,
) -> int:
    """Remove nodes from heap that can be fathomed by bound."""
    original_size = len(heap)
    new_heap = [
        (bound, node_id, node, rmp)
        for bound, node_id, node, rmp in heap
        if bound < incumbent - eps
    ]
    heap.clear()
    for item in new_heap:
        heapq.heappush(heap, item)
    return original_size - len(heap)


def solve_instance(
    instance_path: str | Path = "last_instance.json",
    time_limit: int = 0,
    mip_gap: float = 0.0,
    out_dir: str | Path = "bnp_results",
    log_file=None,
) -> Tuple[Dict, List[str], List[Dict]]:
    """Solve the perishable lot-sizing problem using Branch-and-Price with Best-First Search."""
    start_time = time.time()

    data = json.loads(Path(instance_path).read_text())
    T = int(data["period"])
    Periods = list(range(T))
    items: Dict[int, dict] = {int(k): v for k, v in data["items"].items()}

    prod_cap = data.get("production_capacity") or data.get("manual_capacity")
    capacity = (
        _as_len_T_vector(prod_cap, T)
        if prod_cap is not None
        else _cap_global_from_dem(items, T)
    )

    eps = 1e-6
    max_time = int(time_limit) if time_limit > 0 else 60000
    max_nodes = 1000000
    print_frequency = 50

    # Setup convergence tracking directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    convergence_dir = out_path / "convergence"
    convergence_dir.mkdir(parents=True, exist_ok=True)

    header = [
        "\n" + "╔" + "═" * 68 + "╗",
        f"║ {'BRANCH-AND-PRICE: PERISHABLE LOT-SIZING WITH LEFO':^66s} ║",
        "╠" + "═" * 68 + "╣",
        f"║  Version:   {'v10 - Best-First Search + Memory Optimization':<55s} ║",
        f"║  Items:     {len(items):<55d} ║",
        f"║  Periods:   {T:<55d} ║",
        f"║  Capacity:  {str(capacity[:min(6, T)]) + ('...' if T > 6 else ''):<55s} ║",
        "╚" + "═" * 68 + "╝",
    ]
    for line in header:
        print(line)
        if log_file:
            log_file.write(line + "\n")

    Gamma_by_item: Dict[int, Dict[int, List[int]]] = {}
    Expiry_by_item: Dict[int, Dict[int, int]] = {}

    for item_id, item_data in items.items():
        shelf_seq = list(item_data["shelf_seq"])
        if len(shelf_seq) != T:
            raise ValueError(f"items[{item_id}]['shelf_seq'] must have length {T}")
        Gamma: Dict[int, List[int]] = {}
        Expiry: Dict[int, int] = {}
        for t in Periods:
            m_it = int(shelf_seq[t])
            v_it = t + m_it
            Expiry[t] = v_it
            if m_it <= 0:
                Gamma[t] = []
                continue
            u_max = min(T - 1, v_it - 1)
            Gamma[t] = list(range(t, u_max + 1))
        Gamma_by_item[item_id] = Gamma
        Expiry_by_item[item_id] = Expiry

    stats = SearchStatistics()
    stats.start_time = start_time

    root = BranchNode(
        node_id=0,
        parent_id=None,
        depth=0,
        theta_0_by_item={i: set() for i in items},
        theta_1_by_item={i: set() for i in items},
        upsilon_0_by_item={i: set() for i in items},
        upsilon_1_by_item={i: set() for i in items},
    )

    msg = "\n>>> ROOT NODE <<<"
    print(msg)
    if log_file:
        log_file.write(msg + "\n")

    root_lb, rmp, converged, z_vals, y_vals, x_vals = solve_node_with_column_generation(
        items=items,
        T=T,
        capacity=capacity,
        Gamma_by_item=Gamma_by_item,
        Expiry_by_item=Expiry_by_item,
        node=root,
        parent_rmp=None,
        verbose=True,
        log_file=log_file,
        stats=stats,
        convergence_dir=convergence_dir,
    )

    if not math.isfinite(root_lb):
        msg = "\n✗ Root infeasible!"
        print(msg)
        if log_file:
            log_file.write(msg + "\n")
        summary = {
            "status": int(GRB.INFEASIBLE),
            "objective": None,
            "best_bound": None,
            "gap": None,
            "runtime_sec": time.time() - start_time,
            "solver_version": "branch_and_price_v10_bestfirst",
            "n_items": len(items),
            "T": T,
        }
        return summary, [], None

    root.lp_bound = root_lb

    msgs = [
        f"  Root LB:      {root_lb:.4f}",
        f"  Integer?      {is_integer(z_vals, y_vals, eps)}",
        f"  Uses dummy?   {solution_uses_dummy(rmp, items, eps)}",
    ]
    for m in msgs:
        print(m)
        if log_file:
            log_file.write(m + "\n")

    best_lb = root_lb
    best_ub: Optional[float] = None
    best_rmp: Optional[RestrictedMasterProblem] = None
    best_z: Dict[int, Dict[Tuple[int, int], float]] = {}
    best_y: Dict[int, Dict[int, float]] = {}
    best_x: Dict[int, Dict[int, float]] = {}
    best_active_cols: List[Dict] = []

    node_counter = 1
    seen_signatures = {node_signature(root)}

    # Best-first search heap: (bound, node_id, node, parent_rmp)
    # Memory optimized: no z_vals, y_vals, x_vals stored
    heap: List[Tuple[float, int, BranchNode, RestrictedMasterProblem]] = []

    stats.nodes_created = 1
    stats.nodes_explored = 1

    # Check if root is already a valid integer solution
    if is_valid_integer_solution(z_vals, y_vals, rmp, items, eps):
        best_ub = root_lb
        best_lb = root_lb
        best_rmp = rmp
        best_z = z_vals
        best_y = y_vals
        best_x = x_vals
        best_active_cols = extract_active_columns(rmp, items, eps)
        root.is_integer = True
        stats.nodes_integer = 1

        msg = "\n✓ Root is INTEGER - OPTIMAL!"
        print(msg)
        if log_file:
            log_file.write(msg + "\n")

        stats.print_summary(best_lb, best_ub, eps, log_file)
        orders_txt = generate_orders_txt(items, best_x, eps)

        summary = {
            "status": int(GRB.OPTIMAL),
            "objective": float(best_ub),
            "best_bound": float(best_lb),
            "gap": 0.0,
            "runtime_sec": time.time() - start_time,
            "solver_version": "branch_and_price_v10_bestfirst",
            "n_items": len(items),
            "T": T,
        }

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "orders.txt").write_text("\n".join(orders_txt), encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        return summary, orders_txt, best_active_cols

    # Branch from root - add children to heap
    branch_var_z = find_most_fractional_z(z_vals, root, eps)
    if branch_var_z is not None:
        item_id, t_br, u_br, z_val = branch_var_z

        left = BranchNode(
            node_id=node_counter,
            parent_id=root.node_id,
            depth=1,
            theta_0_by_item={i: s.copy() for i, s in root.theta_0_by_item.items()},
            theta_1_by_item={i: s.copy() for i, s in root.theta_1_by_item.items()},
            upsilon_0_by_item={i: s.copy() for i, s in root.upsilon_0_by_item.items()},
            upsilon_1_by_item={i: s.copy() for i, s in root.upsilon_1_by_item.items()},
            branch_variable=("Z", item_id, t_br, u_br, z_val),
            branch_direction="Z=0",
            lp_bound=root.lp_bound,
        )
        left.theta_0_by_item[item_id].add((t_br, u_br))
        sig_left = node_signature(left)
        if sig_left not in seen_signatures:
            seen_signatures.add(sig_left)
            node_counter += 1
            stats.nodes_created += 1
            heapq.heappush(heap, (left.lp_bound, left.node_id, left, rmp))

        right = BranchNode(
            node_id=node_counter,
            parent_id=root.node_id,
            depth=1,
            theta_0_by_item={i: s.copy() for i, s in root.theta_0_by_item.items()},
            theta_1_by_item={i: s.copy() for i, s in root.theta_1_by_item.items()},
            upsilon_0_by_item={i: s.copy() for i, s in root.upsilon_0_by_item.items()},
            upsilon_1_by_item={i: s.copy() for i, s in root.upsilon_1_by_item.items()},
            branch_variable=("Z", item_id, t_br, u_br, z_val),
            branch_direction="Z=1",
            lp_bound=root.lp_bound,
        )
        right.theta_1_by_item[item_id].add((t_br, u_br))
        sig_right = node_signature(right)
        if sig_right not in seen_signatures:
            seen_signatures.add(sig_right)
            node_counter += 1
            stats.nodes_created += 1
            heapq.heappush(heap, (right.lp_bound, right.node_id, right, rmp))

        msg = f"  Branch Z[{item_id},{t_br},{u_br}]={z_val:.3f}"
        print(msg)
        if log_file:
            log_file.write(msg + "\n")
    else:
        branch_var_y = find_most_fractional_y(y_vals, root, eps)
        if branch_var_y is not None:
            item_id, t_br, y_val = branch_var_y

            left = BranchNode(
                node_id=node_counter,
                parent_id=root.node_id,
                depth=1,
                theta_0_by_item={i: s.copy() for i, s in root.theta_0_by_item.items()},
                theta_1_by_item={i: s.copy() for i, s in root.theta_1_by_item.items()},
                upsilon_0_by_item={
                    i: s.copy() for i, s in root.upsilon_0_by_item.items()
                },
                upsilon_1_by_item={
                    i: s.copy() for i, s in root.upsilon_1_by_item.items()
                },
                branch_variable=("Y", item_id, t_br, y_val),
                branch_direction="Y=0",
                lp_bound=root.lp_bound,
            )
            left.upsilon_0_by_item[item_id].add(t_br)
            sig_left = node_signature(left)
            if sig_left not in seen_signatures:
                seen_signatures.add(sig_left)
                node_counter += 1
                stats.nodes_created += 1
                heapq.heappush(heap, (left.lp_bound, left.node_id, left, rmp))

            right = BranchNode(
                node_id=node_counter,
                parent_id=root.node_id,
                depth=1,
                theta_0_by_item={i: s.copy() for i, s in root.theta_0_by_item.items()},
                theta_1_by_item={i: s.copy() for i, s in root.theta_1_by_item.items()},
                upsilon_0_by_item={
                    i: s.copy() for i, s in root.upsilon_0_by_item.items()
                },
                upsilon_1_by_item={
                    i: s.copy() for i, s in root.upsilon_1_by_item.items()
                },
                branch_variable=("Y", item_id, t_br, y_val),
                branch_direction="Y=1",
                lp_bound=root.lp_bound,
            )
            right.upsilon_1_by_item[item_id].add(t_br)
            sig_right = node_signature(right)
            if sig_right not in seen_signatures:
                seen_signatures.add(sig_right)
                node_counter += 1
                stats.nodes_created += 1
                heapq.heappush(heap, (right.lp_bound, right.node_id, right, rmp))

            msg = f"  Branch Y[{item_id},{t_br}]={y_val:.3f}"
            print(msg)
            if log_file:
                log_file.write(msg + "\n")

    msg = f"\n{'=' * 70}\nBEST-FIRST SEARCH (Heap-based)\n{'=' * 70}\n"
    print(msg)
    if log_file:
        log_file.write(msg)

    while heap and stats.nodes_explored < max_nodes:
        if time.time() - start_time > max_time:
            msg = "\n⏱ Time limit reached"
            print(msg)
            if log_file:
                log_file.write(msg + "\n")
            break

        # Pop node with lowest bound (best-first)
        bound, node_id, node, parent_rmp = heapq.heappop(heap)

        # Prune by bound
        if best_ub is not None and bound >= best_ub - eps:
            stats.nodes_fathomed_by_bound += 1
            continue

        if stats.nodes_explored % print_frequency == 0:
            gap_str = "N/A"
            if best_ub is not None:
                gap = best_ub - best_lb
                gap_pct = 100 * gap / max(abs(best_ub), 1e-10)
                gap_str = f"{gap_pct:.2f}%"
            msg = (
                f"[Progress: N={stats.nodes_explored:4d}, Heap={len(heap):4d}, "
                f"LB={best_lb:.2f}, UB={best_ub if best_ub is not None else 'N/A'}, Gap={gap_str}]"
            )
            print(msg)
            if log_file:
                log_file.write(msg + "\n")

        msg = f"N{node.node_id:4d} D{node.depth:2d} "
        print(msg, end="", flush=True)
        if log_file:
            log_file.write(msg)

        lb, rmp, converged, z_vals, y_vals, x_vals = solve_node_with_column_generation(
            items=items,
            T=T,
            capacity=capacity,
            Gamma_by_item=Gamma_by_item,
            Expiry_by_item=Expiry_by_item,
            node=node,
            parent_rmp=parent_rmp,
            verbose=True,
            log_file=log_file,
            stats=stats,
            convergence_dir=convergence_dir,
        )

        stats.nodes_explored += 1
        stats.max_depth = max(stats.max_depth, node.depth)
        node.lp_bound = lb

        if not math.isfinite(lb):
            msg = "  FATHOMED: Infeasible"
            print(msg)
            if log_file:
                log_file.write(msg + "\n")
            node.is_pruned = True
            node.prune_reason = "infeasible"
            stats.nodes_fathomed_by_infeasible += 1
            continue

        # Update global lower bound (minimum of unexplored nodes)
        if heap:
            best_lb = min(lb, min(h[0] for h in heap))
        else:
            best_lb = lb

        if best_ub is not None and lb >= best_ub - eps:
            msg = f"  FATHOMED: {lb:.2f} >= {best_ub:.2f}"
            print(msg)
            if log_file:
                log_file.write(msg + "\n")
            node.is_pruned = True
            node.prune_reason = "bound"
            stats.nodes_fathomed_by_bound += 1
            continue

        # Check if valid integer solution
        if is_valid_integer_solution(z_vals, y_vals, rmp, items, eps):
            msg = f"  INTEGER: {lb:.2f}"
            print(msg, end="")
            if log_file:
                log_file.write(msg)

            node.is_integer = True
            stats.nodes_integer += 1

            if best_ub is None or lb < best_ub - eps:
                best_ub = lb
                best_rmp = rmp
                best_z = z_vals
                best_y = y_vals
                best_x = x_vals
                best_active_cols = extract_active_columns(rmp, items, eps)
                msg = " ★ NEW INCUMBENT!"
                print(msg)
                if log_file:
                    log_file.write(msg + "\n")

                num_fathomed = fathom_heap_by_incumbent(heap, best_ub, eps)
                stats.nodes_fathomed_on_incumbent += num_fathomed

                if num_fathomed > 0:
                    msg = f"  Fathomed {num_fathomed} nodes from heap"
                    print(msg)
                    if log_file:
                        log_file.write(msg + "\n")
            else:
                print()
                if log_file:
                    log_file.write("\n")

            if heap:
                best_lb = min(h[0] for h in heap)
            else:
                best_lb = best_ub if best_ub is not None else lb

            continue

        # Check if solution only uses dummy
        if solution_uses_dummy(rmp, items, eps) and is_integer(z_vals, y_vals, eps):
            msg = "  FATHOMED: Dummy-only solution"
            print(msg)
            if log_file:
                log_file.write(msg + "\n")
            node.is_pruned = True
            node.prune_reason = "dummy_infeasible"
            stats.nodes_fathomed_by_infeasible += 1
            continue

        # Branch
        branch_var_z = find_most_fractional_z(z_vals, node, eps)
        if branch_var_z is not None:
            item_id, t_br, u_br, z_val = branch_var_z

            left = BranchNode(
                node_id=node_counter,
                parent_id=node.node_id,
                depth=node.depth + 1,
                theta_0_by_item={i: s.copy() for i, s in node.theta_0_by_item.items()},
                theta_1_by_item={i: s.copy() for i, s in node.theta_1_by_item.items()},
                upsilon_0_by_item={
                    i: s.copy() for i, s in node.upsilon_0_by_item.items()
                },
                upsilon_1_by_item={
                    i: s.copy() for i, s in node.upsilon_1_by_item.items()
                },
                branch_variable=("Z", item_id, t_br, u_br, z_val),
                branch_direction="Z=0",
                lp_bound=lb,
            )
            left.theta_0_by_item[item_id].add((t_br, u_br))
            sig_left = node_signature(left)
            if sig_left not in seen_signatures:
                seen_signatures.add(sig_left)
                node_counter += 1
                stats.nodes_created += 1
                heapq.heappush(heap, (lb, left.node_id, left, rmp))

            right = BranchNode(
                node_id=node_counter,
                parent_id=node.node_id,
                depth=node.depth + 1,
                theta_0_by_item={i: s.copy() for i, s in node.theta_0_by_item.items()},
                theta_1_by_item={i: s.copy() for i, s in node.theta_1_by_item.items()},
                upsilon_0_by_item={
                    i: s.copy() for i, s in node.upsilon_0_by_item.items()
                },
                upsilon_1_by_item={
                    i: s.copy() for i, s in node.upsilon_1_by_item.items()
                },
                branch_variable=("Z", item_id, t_br, u_br, z_val),
                branch_direction="Z=1",
                lp_bound=lb,
            )
            right.theta_1_by_item[item_id].add((t_br, u_br))
            sig_right = node_signature(right)
            if sig_right not in seen_signatures:
                seen_signatures.add(sig_right)
                node_counter += 1
                stats.nodes_created += 1
                heapq.heappush(heap, (lb, right.node_id, right, rmp))

            msg = f"  Branch Z[{item_id},{t_br},{u_br}]={z_val:.3f}"
            print(msg)
            if log_file:
                log_file.write(msg + "\n")
            continue

        branch_var_y = find_most_fractional_y(y_vals, node, eps)
        if branch_var_y is not None:
            item_id, t_br, y_val = branch_var_y

            left = BranchNode(
                node_id=node_counter,
                parent_id=node.node_id,
                depth=node.depth + 1,
                theta_0_by_item={i: s.copy() for i, s in node.theta_0_by_item.items()},
                theta_1_by_item={i: s.copy() for i, s in node.theta_1_by_item.items()},
                upsilon_0_by_item={
                    i: s.copy() for i, s in node.upsilon_0_by_item.items()
                },
                upsilon_1_by_item={
                    i: s.copy() for i, s in node.upsilon_1_by_item.items()
                },
                branch_variable=("Y", item_id, t_br, y_val),
                branch_direction="Y=0",
                lp_bound=lb,
            )
            left.upsilon_0_by_item[item_id].add(t_br)
            sig_left = node_signature(left)
            if sig_left not in seen_signatures:
                seen_signatures.add(sig_left)
                node_counter += 1
                stats.nodes_created += 1
                heapq.heappush(heap, (lb, left.node_id, left, rmp))

            right = BranchNode(
                node_id=node_counter,
                parent_id=node.node_id,
                depth=node.depth + 1,
                theta_0_by_item={i: s.copy() for i, s in node.theta_0_by_item.items()},
                theta_1_by_item={i: s.copy() for i, s in node.theta_1_by_item.items()},
                upsilon_0_by_item={
                    i: s.copy() for i, s in node.upsilon_0_by_item.items()
                },
                upsilon_1_by_item={
                    i: s.copy() for i, s in node.upsilon_1_by_item.items()
                },
                branch_variable=("Y", item_id, t_br, y_val),
                branch_direction="Y=1",
                lp_bound=lb,
            )
            right.upsilon_1_by_item[item_id].add(t_br)
            sig_right = node_signature(right)
            if sig_right not in seen_signatures:
                seen_signatures.add(sig_right)
                node_counter += 1
                stats.nodes_created += 1
                heapq.heappush(heap, (lb, right.node_id, right, rmp))

            msg = f"  Branch Y[{item_id},{t_br}]={y_val:.3f}"
            print(msg)
            if log_file:
                log_file.write(msg + "\n")
            continue

        msg = "  No fractional variable - solution is integral"
        print(msg)
        if log_file:
            log_file.write(msg + "\n")

    if best_ub is not None:
        best_lb = best_ub

    stats.print_summary(best_lb, best_ub, eps, log_file)

    orders_txt = generate_orders_txt(items, best_x, eps)

    summary = {
        "status": int(
            GRB.OPTIMAL
            if best_ub is not None and abs(best_ub - best_lb) < eps
            else GRB.SUBOPTIMAL if best_ub is not None else GRB.INFEASIBLE
        ),
        "objective": float(best_ub) if best_ub is not None else None,
        "best_bound": float(best_lb),
        "gap": ((best_ub - best_lb) / max(abs(best_ub), 1e-10) if best_ub else None),
        "runtime_sec": float(time.time() - start_time),
        "solver_version": "branch_and_price_v10_bestfirst",
        "n_items": len(items),
        "T": T,
        "nodes_explored": stats.nodes_explored,
        "nodes_created": stats.nodes_created,
        "total_columns": stats.total_columns_generated,
        "total_cg_iterations": stats.total_cg_iterations,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "orders.txt").write_text("\n".join(orders_txt), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return summary, orders_txt, best_active_cols


def generate_orders_txt(
    items: Dict[int, dict],
    x_vals: Dict[int, Dict[int, float]],
    eps: float,
) -> List[str]:
    orders_txt: List[str] = []
    for item_id in sorted(items.keys()):
        orders_txt.append(f"Item {item_id} — orders (t → qty)")
        production = x_vals.get(item_id, {})
        for t in sorted(production.keys()):
            qty = production[t]
            if qty > eps:
                orders_txt.append(f" {t:2d} → {qty:8.3f}")
        orders_txt.append("")
    return orders_txt


def log_optimal_solution_details(
    summary: Dict,
    orders: List[str],
    active_cols: List[Dict],
    items: Dict[int, dict],
    log_file,
    eps: float = 1e-6,
):
    """Write comprehensive details about the optimal solution to log file."""
    log_file.write("\n" + "=" * 90 + "\n")
    log_file.write("OPTIMAL SOLUTION DETAILS\n")
    log_file.write("=" * 90 + "\n\n")

    # Summary statistics
    log_file.write("SOLUTION SUMMARY\n")
    log_file.write("-" * 60 + "\n")
    log_file.write(f"  Objective value:       {summary.get('objective', 'N/A')}\n")
    log_file.write(f"  Best bound:            {summary.get('best_bound', 'N/A'):.4f}\n")
    gap = summary.get("gap", 0) or 0
    log_file.write(f"  Optimality gap:        {gap * 100:.4f}%\n")
    log_file.write(
        f"  Runtime:               {summary.get('runtime_sec', 0):.2f} seconds\n"
    )
    log_file.write(f"  Nodes explored:        {summary.get('nodes_explored', 0)}\n")
    log_file.write(f"  Nodes created:         {summary.get('nodes_created', 0)}\n")
    log_file.write(
        f"  Total CG iterations:   {summary.get('total_cg_iterations', 0)}\n"
    )
    log_file.write(f"  Total columns added:   {summary.get('total_columns', 0)}\n")
    log_file.write("\n")

    # Production plan
    log_file.write("PRODUCTION PLAN\n")
    log_file.write("-" * 60 + "\n")
    for line in orders:
        log_file.write(line + "\n")
    log_file.write("\n")

    # Active columns in optimal basis
    if active_cols:
        log_file.write("ACTIVE COLUMNS IN OPTIMAL BASIS\n")
        log_file.write("-" * 100 + "\n")
        log_file.write(
            f"{'Column':<15} {'Item':<6} {'λ value':<12} {'Cost':<14} {'Setups':<20} {'Arcs'}\n"
        )
        log_file.write("-" * 100 + "\n")

        # Sort by item_id, then col_idx
        sorted_cols = sorted(active_cols, key=lambda x: (x["item_id"], x["col_idx"]))

        for col_info in sorted_cols:
            item_id = col_info["item_id"]
            idx = col_info["col_idx"]
            lam_val = col_info["lambda_val"]
            cost = col_info["cost"]
            setups = col_info["setups"]
            arcs = col_info["arcs"]

            setup_str = ",".join(map(str, setups)) if setups else "none"
            arc_str = ",".join(f"({t},{u})" for t, u in arcs) if arcs else "none"
            log_file.write(
                f"λ[{item_id},{idx}]     {item_id:<6} {lam_val:<12.6f} {cost:<14.2f} {setup_str:<20} {arc_str}\n"
            )

        log_file.write("-" * 100 + "\n")
        log_file.write(f"Total active columns: {len(active_cols)}\n\n")

        # Per-item breakdown
        log_file.write("PER-ITEM SOLUTION BREAKDOWN\n")
        log_file.write("-" * 60 + "\n")

        for item_id in sorted(items.keys()):
            item_data = items[item_id]
            log_file.write(f"\nItem {item_id}:\n")
            log_file.write(f"  Demand:    {item_data['demand']}\n")
            log_file.write(f"  Shelf seq: {item_data['shelf_seq']}\n")
            log_file.write(f"  Setup:     {item_data['setup']}\n")
            log_file.write(f"  Var cost:  {item_data['c_var']}\n")
            log_file.write(f"  Holding:   {item_data['h']}\n")

            # Get columns for this item
            item_cols = [c for c in active_cols if c["item_id"] == item_id]

            if item_cols:
                log_file.write(f"  Active columns: {len(item_cols)}\n")
                for col_info in item_cols:
                    idx = col_info["col_idx"]
                    lam_val = col_info["lambda_val"]
                    cost = col_info["cost"]
                    cap_usage = col_info["capacity_usage"]
                    setups = col_info["setups"]

                    log_file.write(
                        f"    λ[{item_id},{idx}] = {lam_val:.6f}, cost = {cost:.2f}\n"
                    )
                    for t in setups:
                        prod = cap_usage[t] if t < len(cap_usage) else 0
                        log_file.write(f"      t={t}: setup=1, prod={prod:.2f}\n")
    else:
        log_file.write("ACTIVE COLUMNS IN OPTIMAL BASIS\n")
        log_file.write("-" * 60 + "\n")
        log_file.write("No active columns captured (solution may be infeasible)\n\n")

    log_file.write("\n" + "=" * 90 + "\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    from datetime import datetime

    instance = {
        "period": 7,
        "production_capacity": 32,
        "items": {
            "1": {
                "demand": [10, 12, 8, 15, 10, 9, 11],
                "c_var": 5.0,
                "h": 0.5,
                "setup": 80.0,
                "shelf_seq": [2, 2, 5, 1, 2, 2, 3],
            },
            "2": {
                "demand": [5, 7, 6, 8, 9, 7, 6],
                "c_var": 8.0,
                "h": 0.8,
                "setup": 120.0,
                "shelf_seq": [1, 4, 5, 1, 3, 3, 5],
            },
            "3": {
                "demand": [2, 7, 6, 8, 9, 17, 6],
                "c_var": 8.5,
                "h": 0.7,
                "setup": 110.0,
                "shelf_seq": [2, 1, 2, 3, 4, 5, 6],
            },
        },
    }

    out_dir = Path("bnp_v10_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "solver_log.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write("=" * 90 + "\n")
    log_file.write(f"Branch-and-Price Solver v10 — Best-First Search\n")
    log_file.write(f"Execution started: {timestamp}\n")
    log_file.write("=" * 90 + "\n\n")

    # Save test instance
    instance_path = out_dir / "test_instance.json"
    instance_path.write_text(json.dumps(instance, indent=2))
    log_file.write(f"Instance saved to: {instance_path}\n\n")

    # Log instance details
    log_file.write("INSTANCE DETAILS\n")
    log_file.write("-" * 60 + "\n")
    log_file.write(f"  Periods: {instance['period']}\n")
    log_file.write(f"  Capacity: {instance['production_capacity']}\n")
    log_file.write(f"  Items: {len(instance['items'])}\n\n")

    for item_id, item_data in instance["items"].items():
        log_file.write(f"  Item {item_id}:\n")
        log_file.write(f"    Demand:    {item_data['demand']}\n")
        log_file.write(f"    Shelf seq: {item_data['shelf_seq']}\n")
        log_file.write(f"    Setup:     {item_data['setup']}\n")
        log_file.write(f"    Var cost:  {item_data['c_var']}\n")
        log_file.write(f"    Holding:   {item_data['h']}\n")
    log_file.write("\n")
    # ------------------------------------------------------------------
    print("=" * 70)
    print("BRANCH-AND-PRICE SOLVER v10")
    print("Best-First Search + Memory Optimization")
    print("=" * 70)

    summary, orders, best_active_cols = solve_instance(
        instance_path=str(instance_path),
        time_limit=600,
        out_dir=out_dir,
        log_file=log_file,
    )

    # ------------------------------------------------------------------
    # Log optimal solution details
    # ------------------------------------------------------------------
    items = {int(k): v for k, v in instance["items"].items()}
    log_optimal_solution_details(summary, orders, best_active_cols, items, log_file)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SOLUTION SUMMARY")
    print("=" * 70)

    status_map = {2: "OPTIMAL", 3: "INFEASIBLE", 13: "SUBOPTIMAL"}
    status_str = status_map.get(summary["status"], f"STATUS_{summary['status']}")

    print(f"  Status:          {status_str}")
    print(f"  Objective:       {summary.get('objective', 'N/A')}")
    print(f"  Best bound:      {summary['best_bound']:.4f}")
    gap = summary.get("gap", 0) or 0
    print(f"  Gap:             {gap * 100:.4f}%")
    print(f"  Runtime:         {summary['runtime_sec']:.2f}s")
    print(f"  Nodes explored:  {summary.get('nodes_explored', 0)}")
    print(f"  Nodes created:   {summary.get('nodes_created', 0)}")
    print(f"  CG iterations:   {summary.get('total_cg_iterations', 0)}")
    print(f"  Columns added:   {summary.get('total_columns', 0)}")

    if best_active_cols:
        print(f"  Active columns:  {len(best_active_cols)}")

    print("\n" + "-" * 70)
    print("PRODUCTION PLAN")
    print("-" * 70)
    for line in orders:
        print(line)

    print("\n" + "-" * 70)
    print("OUTPUT FILES")
    print("-" * 70)
    print(f"  Log file:     {log_path}")
    print(f"  Summary:      {out_dir / 'summary.json'}")
    print(f"  Orders:       {out_dir / 'orders.txt'}")
    print("=" * 70)

    log_file.write("\nExecution completed successfully.\n")
    log_file.close()

    print("\nDone.")
