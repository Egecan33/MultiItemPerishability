"""
Branch-and-Price for Perishable Lot-Sizing with Heterogeneous Shelf Lives and LEFO
Version 11: Branching via RMP Linking Constraints (not subproblem restrictions)

Key Changes from v10:
- Branching enforcement moved from pricing subproblem to RMP:
  * Υ^0, Θ^0 (forbidden): Enforced by FILTERING columns before adding to RMP
  * Υ^1, Θ^1 (forced): Enforced by RMP linking constraints (Σ Y λ = 1, Σ Z λ = 1)
- Pricing subproblem is UNRESTRICTED - generates any feasible ZIO column
- Duals from = 1 linking constraints guide pricing toward required features

Features:
- Best-first search using heap ordered by LP bound (lowest first)
- Memory optimization: queue stores only (bound, node_id, node, rmp)
- Column inheritance for faster convergence at branch nodes
- σ (sigma) and τ (tau) duals from Υ^1/Θ^1 linking constraints guide pricing
- Dummy column detection to prevent false integer solutions
- Branching on Y (setup) first, then Z (arc) variables
- DP-based pricing for ZIO columns

"""

from __future__ import annotations
import heapq
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import gurobipy as gp
from gurobipy import GRB


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

    def print_summary(self, best_lb: float, best_ub: Optional[float], eps: float):
        elapsed = time.time() - self.start_time

        print("\n" + "=" * 70)
        print(" " * 28 + "FINAL RESULTS")
        print("=" * 70)
        print(f"  Time elapsed:          {elapsed:.2f} seconds")
        print(f"  Nodes created:         {self.nodes_created}")
        print(f"  Nodes explored:        {self.nodes_explored}")
        print(f"  Integer solutions:     {self.nodes_integer}")
        print(f"  Max depth:             {self.max_depth}")
        print(f"  Total CG iterations:   {self.total_cg_iterations}")
        print(f"  Total columns added:   {self.total_columns_generated}")
        print()
        print(f"  Fathomed by bound:     {self.nodes_fathomed_by_bound}")
        print(f"  Fathomed infeasible:   {self.nodes_fathomed_by_infeasible}")
        print(f"  Fathomed on incumbent: {self.nodes_fathomed_on_incumbent}")
        print()
        print(f"  Best lower bound:      {best_lb:.4f}")

        if best_ub is not None:
            print(f"  Best upper bound:      {best_ub:.4f}")
            gap = best_ub - best_lb
            gap_pct = 100 * gap / max(abs(best_ub), 1e-10)
            print(f"  Gap:                   {gap:.4f} ({gap_pct:.2f}%)")
            if gap < eps:
                print("  DONE")
        else:
            print(f"  Best upper bound:      Not found")
        print("=" * 70)


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


def column_violates_lefo(
    arc_usage: Dict[Tuple[int, int], float],
    Gamma: Dict[int, List[int]],
    Expiry: Dict[int, int],
    eps: float = 1e-6,
) -> bool:

    if not arc_usage:
        return False

    # production periods that actually appear in this column
    prods = sorted({t for (t, u) in arc_usage.keys()}, key=lambda t: Expiry[t])

    for a in range(len(prods)):
        t1 = prods[a]
        v1 = Expiry[t1]
        for b in range(a + 1, len(prods)):
            t2 = prods[b]
            v2 = Expiry[t2]
            if v1 >= v2:
                # we only care about v1 < v2 (earlier expiry first)
                continue

            # up = demand periods served from t2
            for up in Gamma.get(t2, []):
                if arc_usage.get((t2, up), 0.0) <= eps:
                    continue  # this arc not actually used

                # u = demand periods served from t1 in [t2, up-1]
                for u in Gamma.get(t1, []):
                    if u < t2 or u > up - 1:
                        continue
                    if arc_usage.get((t1, u), 0.0) > eps:
                        # we have both Z[t1,u] = 1 and Z[t2,up] = 1 in this column
                        return True
    return False


def solve_pricing_subproblem(
    item_id: int,
    item_data: dict,
    T: int,
    Gamma: Dict[int, List[int]],
    Expiry: Dict[int, int],
    capacity_duals: List[float],
    convexity_dual: float,
    theta_0: Set[Tuple[int, int]],
    upsilon_0: Set[int],
    sigma: Optional[Dict[Tuple[int, int], float]] = None,
    tau: Optional[Dict[Tuple[int, int, int], float]] = None,
    eps: float = 1e-6,
) -> Tuple[float, Optional[ProductionPlanColumn]]:

    sigma = sigma or {}
    tau = tau or {}

    # Extract item parameters
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

    # Cumulative holding cost prefix sums
    h_prefix = [0.0] * (T + 1)
    for k in range(T):
        h_prefix[k + 1] = h_prefix[k] + h_at(k)

    def h_sum(t: int, u: int) -> float:
        return h_prefix[u] - h_prefix[t]

    INF = float("inf")

    # HELPER FUNCTIONS - Enforce Υ^0 and Θ^0 (forbidden constraints)

    def can_produce_at(t: int) -> bool:
        """
        Check if production is allowed at period t.
        Returns False if:
            - t ∈ Υ^0 (setup forbidden)
            - Γ_t^i = ∅ (no reachable demand periods)
        """
        if t in upsilon_0:
            return False
        if t not in Gamma:
            return False
        return len(Gamma.get(t, [])) > 0

    def get_valid_ends(t: int) -> List[int]:
        """
        Get valid end periods for production run starting at t.
        Filters by:
            1. s ∈ Γ_t^i (shelf-life feasibility)
            2. ∀u ∈ [t,s]: (t,u) ∉ Θ^0 (no forbidden arcs)
        """
        if not can_produce_at(t):
            return []

        valid_ends = Gamma.get(t, [])
        if not valid_ends:
            return []

        result = []
        for s in valid_ends:
            # Check no arc in [t,s] is forbidden
            arc_valid = True
            for u in range(t, s + 1):
                if (t, u) in theta_0:
                    arc_valid = False
                    break
            if arc_valid:
                result.append(s)

        return result

    def run_reduced_cost(t: int, s: int) -> float:
        """
        Compute reduced cost for production run [t, s].
        Includes duals from Υ^1/Θ^1 linking constraints.
        """
        cost = s_at(t)  # Setup cost
        cost -= sigma.get((item_id, t), 0.0)  # Setup dual from Υ^1 constraints

        for u in range(t, s + 1):
            d_u = float(demand[u])
            if d_u > 0:
                unit_cost = c_at(t) + h_sum(t, u) - capacity_duals[t]
                cost += unit_cost * d_u
                cost -= tau.get((item_id, t, u), 0.0)  # Arc dual from Θ^1 constraints

        return cost

    # DYNAMIC PROGRAMMING: BACKWARD RECURSION

    dp = [INF] * (T + 1)
    decision = [(-1, -1)] * T
    dp[T] = 0

    for t in range(T - 1, -1, -1):
        best_cost = INF
        best_action = (-1, -1)

        # ACTION 1: SKIP (only if no demand)
        if float(demand[t]) <= 0:
            skip_cost = dp[t + 1]
            if skip_cost < best_cost:
                best_cost = skip_cost
                best_action = (-1, -1)

        # ACTION 2: SETUP ONLY (only if no demand and setup not forbidden)
        if float(demand[t]) <= 0 and t not in upsilon_0:
            setup_cost = s_at(t) - sigma.get((item_id, t), 0.0) + dp[t + 1]
            if setup_cost < best_cost:
                best_cost = setup_cost
                best_action = (0, -1)

        # ACTION 3: PRODUCE from t to s (respecting Υ^0 and Θ^0)
        if can_produce_at(t):
            for s in get_valid_ends(t):
                if s + 1 <= T and dp[s + 1] < INF:
                    cost = run_reduced_cost(t, s) + dp[s + 1]
                    if cost < best_cost:
                        best_cost = cost
                        best_action = (1, s)

        dp[t] = best_cost
        decision[t] = best_action

    # Check feasibility
    if dp[0] >= INF:
        return INF, None

    reduced_cost = dp[0] - convexity_dual

    if reduced_cost >= -eps:
        return reduced_cost, None

    # SOLUTION RECONSTRUCTION

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

    # Validate demand coverage
    covered = [False] * T
    for (t_prod, u), val in arc_usage.items():
        if val > 0.5 and u < T:
            covered[u] = True

    for u in range(T):
        if demand[u] > 0 and not covered[u]:
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

    def __init__(
        self,
        items: Dict[int, dict],
        T: int,
        capacity: List[float],
        Gamma_by_item: Dict[int, Dict[int, List[int]]],
        theta_0_by_item: Optional[Dict[int, Set[Tuple[int, int]]]] = None,
        theta_1_by_item: Optional[Dict[int, Set[Tuple[int, int]]]] = None,
        upsilon_0_by_item: Optional[Dict[int, Set[int]]] = None,
        upsilon_1_by_item: Optional[Dict[int, Set[int]]] = None,
        initial_columns: Optional[Dict[int, List[ProductionPlanColumn]]] = None,
    ):
        self.items = items
        self.T = T
        self.capacity = capacity
        self.Gamma_by_item = Gamma_by_item
        # Store = 0 sets for filtering (used externally, not in RMP constraints)
        self.theta_0_by_item = theta_0_by_item or {i: set() for i in items}
        self.upsilon_0_by_item = upsilon_0_by_item or {i: set() for i in items}
        # Store = 1 sets for linking constraints
        self.theta_1_by_item = theta_1_by_item or {i: set() for i in items}
        self.upsilon_1_by_item = upsilon_1_by_item or {i: set() for i in items}

        self.model = gp.Model("RMP")
        self.model.Params.OutputFlag = 0
        self.model.Params.LogToConsole = 0
        self.model.Params.Method = 1

        self.columns: Dict[int, List[ProductionPlanColumn]] = {i: [] for i in items}
        self.lambdas: Dict[Tuple[int, int], gp.Var] = {}

        self.convex_expr: Dict[int, gp.LinExpr] = {i: gp.LinExpr(0.0) for i in items}
        self.cap_expr: List[gp.LinExpr] = [gp.LinExpr(0.0) for _ in range(T)]

        # Linking expressions for Υ^1 (forced setups): Σ Y λ = 1
        self.y_link_expr: Dict[Tuple[int, int], gp.LinExpr] = {}
        for item_id in items:
            for t in self.upsilon_1_by_item.get(item_id, set()):
                self.y_link_expr[(item_id, t)] = gp.LinExpr(0.0)

        # Linking expressions for Θ^1 (forced arcs): Σ Z λ = 1
        self.z_link_expr: Dict[Tuple[int, int, int], gp.LinExpr] = {}
        for item_id in items:
            for t, u in self.theta_1_by_item.get(item_id, set()):
                self.z_link_expr[(item_id, t, u)] = gp.LinExpr(0.0)

        # LEFO  constraints in the RMP
        # key = (item_id, t1, t2, u, up)
        self.lefo_expr: Dict[Tuple[int, int, int, int, int], gp.LinExpr] = {}
        self.lefo_con: Dict[Tuple[int, int, int, int, int], gp.Constr] = {}
        # index to know which LEFO constraints each arc (i,t,u) appears in
        self.lefo_index_by_arc: Dict[
            Tuple[int, int, int], List[Tuple[int, int, int, int, int]]
        ] = {}

        # Build LEFO rows per item using the same logic as MIP C5
        for item_id in items:
            Gamma = self.Gamma_by_item[item_id]

            # compute expiry v_it from shelf_seq
            shelf_seq = list(self.items[item_id]["shelf_seq"])
            Expiry = {t: t + int(shelf_seq[t]) for t in range(self.T)}

            prods = [t for t in range(self.T) if Gamma.get(t)]
            prods.sort(key=lambda t: Expiry[t])  # ascending by v_it

            for a in range(len(prods)):
                t1 = prods[a]
                v1 = Expiry[t1]
                for b in range(a + 1, len(prods)):
                    t2 = prods[b]
                    v2 = Expiry[t2]
                    if v1 >= v2:  # only v1 < v2
                        continue

                    for up in Gamma.get(t2, []):  # u' for t2
                        for u in [
                            uu for uu in Gamma.get(t1, []) if t2 <= uu < up
                        ]:  # u for t1
                            key = (item_id, t1, t2, u, up)

                            expr = gp.LinExpr(0.0)
                            self.lefo_expr[key] = expr
                            self.lefo_con[key] = self.model.addConstr(
                                expr <= 1.0,
                                name=f"lefo_{item_id}_{t1}_{t2}_{u}_{up}",
                            )

                            # index arcs → LEFO rows
                            self.lefo_index_by_arc.setdefault(
                                (item_id, t1, u), []
                            ).append(key)
                            self.lefo_index_by_arc.setdefault(
                                (item_id, t2, up), []
                            ).append(key)

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

        # Setup linking constraints for Υ^1 (= 1)
        for (item_id, t), expr in self.y_link_expr.items():
            self.y_link_con[(item_id, t)] = self.model.addConstr(
                expr == 1.0, name=f"y_link_{item_id}_{t}"
            )

        # Arc linking constraints for Θ^1 (= 1)
        for (item_id, t, u), expr in self.z_link_expr.items():
            self.z_link_con[(item_id, t, u)] = self.model.addConstr(
                expr == 1.0, name=f"z_link_{item_id}_{t}_{u}"
            )

        if initial_columns:
            for item_id, cols in initial_columns.items():
                for col in cols:
                    self.add_column(col)

    def _rebuild(self):
        """Rebuild constraints after adding columns."""
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
        # Rebuild Υ^1 constraints
        for key, expr in self.y_link_expr.items():
            self.model.remove(self.y_link_con[key])
            self.y_link_con[key] = self.model.addConstr(
                expr == 1.0, name=f"y_link_{key[0]}_{key[1]}"
            )
        # Rebuild Θ^1 constraints
        for key, expr in self.z_link_expr.items():
            self.model.remove(self.z_link_con[key])
            self.z_link_con[key] = self.model.addConstr(
                expr == 1.0, name=f"z_link_{key[0]}_{key[1]}_{key[2]}"
            )

        # Rebuild LEFO (C5) constraints
        for key, expr in self.lefo_expr.items():
            self.model.remove(self.lefo_con[key])
            i, t1, t2, u, up = key
            self.lefo_con[key] = self.model.addConstr(
                expr <= 1.0,
                name=f"lefo_{i}_{t1}_{t2}_{u}_{up}",
            )

    def add_column(self, col: ProductionPlanColumn):
        """Add a column to the RMP."""
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

        # Update Υ^1 linking expressions
        for t in self.upsilon_1_by_item.get(item_id, set()):
            if t < len(col.setup_by_period):
                y_val = col.setup_by_period[t]
                if y_val != 0.0:
                    self.y_link_expr[(item_id, t)] += y_val * lam

        # Update Θ^1 linking expressions
        for t, u in self.theta_1_by_item.get(item_id, set()):
            z_val = col.arc_usage.get((t, u), 0.0)
            if z_val != 0.0:
                self.z_link_expr[(item_id, t, u)] += z_val * lam

        # Update LEFO constraints
        for (t, u), z_val in col.arc_usage.items():
            if z_val == 0.0:
                continue
            for key in self.lefo_index_by_arc.get((item_id, t, u), []):
                self.lefo_expr[key] += z_val * lam

        self._rebuild()

    def solve(self) -> Tuple[
        float,
        Dict[int, float],
        List[float],
        Dict[Tuple[int, int], float],
        Dict[Tuple[int, int, int], float],
    ]:
        """
        Solve the RMP and return duals.

        Returns:
            (obj_val, mu, pi, sigma, tau) where:
            - obj_val: Objective value
            - mu: Convexity duals {item_id: μ_i}
            - pi: Capacity duals [π_t for t in T]
            - sigma: Setup linking duals {(item_id, t): σ_{it}} from Υ^1 constraints
            - tau: Arc linking duals {(item_id, t, u): τ_{itu}} from Θ^1 constraints
        """
        self.model.optimize()

        if self.model.status != GRB.OPTIMAL:
            return math.inf, {}, [], {}, {}

        mu = {i: self.convex_con[i].Pi for i in self.items}
        pi = [self.cap_con[t].Pi for t in range(self.T)]

        # Collect setup linking duals from Υ^1
        sigma: Dict[Tuple[int, int], float] = {}
        for (item_id, t), con in self.y_link_con.items():
            sigma[(item_id, t)] = con.Pi

        # Collect arc linking duals from Θ^1
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
    stats: Optional[SearchStatistics] = None,
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

    # Create RMP with linking constraints for = 1 branching (Υ^1, Θ^1)
    # = 0 constraints (Υ^0, Θ^0) are enforced by filtering columns before adding
    rmp = RestrictedMasterProblem(
        items=items,
        T=T,
        capacity=capacity,
        Gamma_by_item=Gamma_by_item,
        theta_0_by_item=node.theta_0_by_item,
        theta_1_by_item=node.theta_1_by_item,
        upsilon_0_by_item=node.upsilon_0_by_item,
        upsilon_1_by_item=node.upsilon_1_by_item,
        initial_columns=inherited_cols,
    )

    # Add dummy columns that satisfy = 1 branching constraints (Υ^1, Θ^1)
    # This ensures RMP feasibility at branch nodes
    for item_id, item_data in items.items():
        demand = item_data["demand"]
        total_demand = sum(demand)
        dummy_cost = 10000.0 * (total_demand + 1.0)

        # Create setup vector: 1 for all forced setups (Υ^1), 0 otherwise
        dummy_setup = [0.0] * T
        for t in node.upsilon_1_by_item.get(item_id, set()):
            if t < T:
                dummy_setup[t] = 1.0

        # Create arc usage: 1 for all forced arcs (Θ^1)
        dummy_arcs: Dict[Tuple[int, int], float] = {}
        for t, u in node.theta_1_by_item.get(item_id, set()):
            dummy_arcs[(t, u)] = 1.0

        col = ProductionPlanColumn(
            item_id=item_id,
            total_plan_cost=dummy_cost,
            capacity_usage_by_period=[0.0] * T,
            setup_by_period=dummy_setup,
            arc_usage=dummy_arcs,
        )
        rmp.add_column(col)

    print(f"  └─ CG[inherited={total_inherited}]: ", end="", flush=True)

    columns_added_this_node = 0
    iterations_this_node = 0

    for iteration in range(1, max_iter + 1):
        lb, mu, pi, sigma, tau = rmp.solve()
        iterations_this_node += 1

        if not math.isfinite(lb):
            print("INFEASIBLE")
            return math.inf, rmp, False, {}, {}, {}

        any_added = False
        cols_this_iter = 0

        for item_id, item_data in items.items():
            # Get forbidden branching sets for pricing
            theta_0 = node.theta_0_by_item.get(item_id, set())
            upsilon_0 = node.upsilon_0_by_item.get(item_id, set())

            Gamma = Gamma_by_item[item_id]
            Expiry = Expiry_by_item[item_id]

            # Pricing subproblem:
            # - Enforces Υ^0 and Θ^0 (forbidden) as hard constraints
            # - Uses σ and τ duals from Υ^1/Θ^1 linking constraints
            rc, col = solve_pricing_subproblem(
                item_id=item_id,
                item_data=item_data,
                T=T,
                Gamma=Gamma,
                Expiry=Expiry,
                capacity_duals=pi,
                convexity_dual=mu[item_id],
                theta_0=theta_0,
                upsilon_0=upsilon_0,
                sigma=sigma,
                tau=tau,
                eps=eps,
            )

            if col is not None and rc < -eps:
                rmp.add_column(col)
                any_added = True
                columns_added_this_node += 1
                cols_this_iter += 1

        if not any_added:
            print(f"LB={lb:.2f} (iter={iteration}, cols={columns_added_this_node})")

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

    print(f"LB={lb:.2f} (max iter, cols={columns_added_this_node})")

    return lb, rmp, False, z_vals, y_vals, x_vals


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
) -> Tuple[Dict, List[str]]:
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

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    header = [
        "╠" + "═" * 68 + "╣",
        f"║  Items:     {len(items):<55d} ║",
        f"║  Periods:   {T:<55d} ║",
        f"║  Capacity:  {str(capacity[:min(6, T)]) + ('...' if T > 6 else ''):<55s} ║",
        "╚" + "═" * 68 + "╝",
    ]
    for line in header:
        print(line)

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

    print("\n>>> ROOT NODE <<<")

    root_lb, rmp, converged, z_vals, y_vals, x_vals = solve_node_with_column_generation(
        items=items,
        T=T,
        capacity=capacity,
        Gamma_by_item=Gamma_by_item,
        Expiry_by_item=Expiry_by_item,
        node=root,
        parent_rmp=None,
        stats=stats,
    )

    if not math.isfinite(root_lb):
        print("\n✗ Root infeasible!")
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
        return summary, []

    root.lp_bound = root_lb

    print(f"  Root LB:      {root_lb:.4f}")
    print(f"  Integer?      {is_integer(z_vals, y_vals, eps)}")
    print(f"  Uses dummy?   {solution_uses_dummy(rmp, items, eps)}")

    best_lb = root_lb
    best_ub: Optional[float] = None
    best_rmp: Optional[RestrictedMasterProblem] = None
    best_z: Dict[int, Dict[Tuple[int, int], float]] = {}
    best_y: Dict[int, Dict[int, float]] = {}
    best_x: Dict[int, Dict[int, float]] = {}

    node_counter = 1
    seen_signatures = {node_signature(root)}

    # Best-first search heap: (bound, node_id, node, parent_rmp)
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
        root.is_integer = True
        stats.nodes_integer = 1

        print("\n✓ Root is INTEGER - OPTIMAL!")

        stats.print_summary(best_lb, best_ub, eps)
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

        return summary, orders_txt

    # Branch from root - add children to heap
    # Branch on Y (setup) variables first, then Z (arc) variables
    branch_var_y = find_most_fractional_y(y_vals, root, eps)
    if branch_var_y is not None:
        item_id, t_br, y_val = branch_var_y

        left = BranchNode(
            node_id=node_counter,
            parent_id=root.node_id,
            depth=1,
            theta_0_by_item={i: s.copy() for i, s in root.theta_0_by_item.items()},
            theta_1_by_item={i: s.copy() for i, s in root.theta_1_by_item.items()},
            upsilon_0_by_item={i: s.copy() for i, s in root.upsilon_0_by_item.items()},
            upsilon_1_by_item={i: s.copy() for i, s in root.upsilon_1_by_item.items()},
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
            upsilon_0_by_item={i: s.copy() for i, s in root.upsilon_0_by_item.items()},
            upsilon_1_by_item={i: s.copy() for i, s in root.upsilon_1_by_item.items()},
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

        print(f"  Branch Y[{item_id},{t_br}]={y_val:.3f}")
    else:
        # If no fractional Y, branch on Z (arc) variables
        branch_var_z = find_most_fractional_z(z_vals, root, eps)
        if branch_var_z is not None:
            item_id, t_br, u_br, z_val = branch_var_z

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
                upsilon_0_by_item={
                    i: s.copy() for i, s in root.upsilon_0_by_item.items()
                },
                upsilon_1_by_item={
                    i: s.copy() for i, s in root.upsilon_1_by_item.items()
                },
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

            print(f"  Branch Z[{item_id},{t_br},{u_br}]={z_val:.3f}")

    print(f"\n{'=' * 70}\nBEST-FIRST SEARCH)\n{'=' * 70}\n")

    while heap and stats.nodes_explored < max_nodes:
        if time.time() - start_time > max_time:
            print("\n⏱ Time limit reached")
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
            print(
                f"[Progress: N={stats.nodes_explored:4d}, Heap={len(heap):4d}, "
                f"LB={best_lb:.2f}, UB={best_ub if best_ub is not None else 'N/A'}, Gap={gap_str}]"
            )

        print(f"N{node.node_id:4d} D{node.depth:2d} ", end="", flush=True)

        lb, rmp, converged, z_vals, y_vals, x_vals = solve_node_with_column_generation(
            items=items,
            T=T,
            capacity=capacity,
            Gamma_by_item=Gamma_by_item,
            Expiry_by_item=Expiry_by_item,
            node=node,
            parent_rmp=parent_rmp,
            stats=stats,
        )

        stats.nodes_explored += 1
        stats.max_depth = max(stats.max_depth, node.depth)
        node.lp_bound = lb

        if not math.isfinite(lb):
            print("  FATHOMED: Infeasible")
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
            print(f"  FATHOMED: {lb:.2f} >= {best_ub:.2f}")
            node.is_pruned = True
            node.prune_reason = "bound"
            stats.nodes_fathomed_by_bound += 1
            continue

        # Check if valid integer solution
        if is_valid_integer_solution(z_vals, y_vals, rmp, items, eps):
            print(f"  INTEGER: {lb:.2f}", end="")

            node.is_integer = True
            stats.nodes_integer += 1

            if best_ub is None or lb < best_ub - eps:
                best_ub = lb
                best_rmp = rmp
                best_z = z_vals
                best_y = y_vals
                best_x = x_vals
                print(" ★ NEW INCUMBENT!")

                num_fathomed = fathom_heap_by_incumbent(heap, best_ub, eps)
                stats.nodes_fathomed_on_incumbent += num_fathomed

                if num_fathomed > 0:
                    print(f"  Fathomed {num_fathomed} nodes from heap")
            else:
                print()

            if heap:
                best_lb = min(h[0] for h in heap)
            else:
                best_lb = best_ub if best_ub is not None else lb

            continue

        # Check if solution only uses dummy
        if solution_uses_dummy(rmp, items, eps) and is_integer(z_vals, y_vals, eps):
            print("  FATHOMED: Dummy-only solution")
            node.is_pruned = True
            node.prune_reason = "dummy_infeasible"
            stats.nodes_fathomed_by_infeasible += 1
            continue

        # Branch on Y (setup) variables first, then Z (arc) variables
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

            print(f"  Branch Y[{item_id},{t_br}]={y_val:.3f}")
            continue

        # If no fractional Y, branch on Z (arc) variables
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

            print(f"  Branch Z[{item_id},{t_br},{u_br}]={z_val:.3f}")
            continue

        print("  No fractional variable - solution is integral")

    if best_ub is not None:
        best_lb = best_ub

    stats.print_summary(best_lb, best_ub, eps)

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

    return summary, orders_txt


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
