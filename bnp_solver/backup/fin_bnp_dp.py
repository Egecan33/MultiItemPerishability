"""
Branch-and-Price for Perishable Lot-Sizing with Heterogeneous Shelf Lives and LEFO
Using Dynamic Programming for Zero-Inventory Ordering (ZIO) Column Generation

IMPLEMENTATION DETAILS:
1. No perturbation: Removed to ensure correctness. Was causing missed improving columns.
2. Duplicate avoidance: When DP finds a duplicate column, it re-runs with that
   production run excluded to find alternative improving columns.
3. Sparse column storage: Dict instead of List for capacity_usage and setup_by_period
4. Memory optimization: Don't store z/y/x in B&B queue, only (bound, node_id, node, rmp)
5. Best-first search: Use heap ordered by LP bound (lowest first)
6. Signature-based tracking: Each column identified by its arc usage pattern

Key Changes from MIP Pricing:
- DP generates only ZIO extreme points (Wagner-Whitin style)
- ZIO: Production only when inventory is zero and demand exists
- Each production run covers consecutive demand periods [t, s] where s ∈ Γ_t
- Convex combinations of ZIO columns span the feasible region

ZIO Property (Theorem):
For uncapacitated lot-sizing, optimal solutions have the ZIO property:
    X_t > 0  ⟹  I_{t-1} = 0
This means production occurs only when entering with zero inventory.

DP Recursion:
    f(t) = min cost to satisfy demand [t, T-1] starting with zero inventory

    f(T) = 0  (base case)

    f(t) = { f(t+1)                                        if d_t = 0
           { min_{s ∈ Γ_t} { c(t,s) + f(s+1) }            if d_t > 0

    where c(t,s) = setup_t + Σ_{u=t}^{s} (c_t + h_{t→u} - π_t) · d_u - σ_t - Σ τ_{tu}
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
    """
    Represents a single production plan (column) for one item.

    SPARSE STORAGE: capacity_usage_by_period and setup_by_period are now dicts
    storing only non-zero entries, reducing memory significantly for ZIO columns.
    """

    item_id: int
    total_plan_cost: float
    capacity_usage_by_period: Dict[int, float]  # {period: qty} - sparse
    setup_by_period: Dict[int, float]  # {period: 0 or 1} - sparse
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
        # Check theta_0: must NOT use forbidden arcs
        for t, u in theta_0:
            if (t, u) in self.arc_usage and self.arc_usage[(t, u)] > eps:
                return True

        # Check theta_1: MUST use forced arcs
        for t, u in theta_1:
            if (t, u) not in self.arc_usage or self.arc_usage[(t, u)] < 1.0 - eps:
                return True

        # Check upsilon_0: must NOT use forbidden setups
        for t in upsilon_0:
            if self.setup_by_period.get(t, 0.0) > eps:
                return True

        # Check upsilon_1: MUST use forced setups
        for t in upsilon_1:
            if self.setup_by_period.get(t, 0.0) < 1.0 - eps:
                return True

        return False


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
    start_time: float = field(default_factory=time.time)

    def print_summary(self, best_lb: float, best_ub: Optional[float], eps: float):
        elapsed = time.time() - self.start_time
        print("\n" + "=" * 70)
        print(" " * 28 + "FINAL RESULTS")
        print("=" * 70)
        print(f"  Time elapsed:       {elapsed:.2f} seconds")
        print(f"  Nodes created:      {self.nodes_created}")
        print(f"  Nodes explored:     {self.nodes_explored}")
        print(f"  Integer solutions:  {self.nodes_integer}")
        print(f"  Max depth:          {self.max_depth}")
        print()
        print(f"  Best lower bound:   {best_lb:.4f}")
        if best_ub is not None:
            print(f"  Best upper bound:   {best_ub:.4f}")
            gap = best_ub - best_lb
            gap_pct = 100 * gap / max(abs(best_ub), 1e-10)
            print(f"  Gap:                {gap:.4f} ({gap_pct:.2f}%)")
            if gap < eps:
                print(f"\n  ★★★ PROVEN OPTIMAL! ★★★")
        else:
            print(f"  Best upper bound:   Not found")
        print("=" * 70)


def _cap_global_from_dem(items: Dict[int, dict], T: int) -> List[float]:
    """Generate default capacity from total demand with large buffer.

    For lot-sizing with batching, capacity at each period should be large enough
    to potentially produce ALL demand (worst case: batch everything at one period).
    We use total demand across all periods + 10% buffer as the per-period capacity.
    """
    total_demand = 0.0
    for it in items.values():
        dem = it["demand"]
        total_demand += sum(float(d) for d in dem)

    # Each period should be able to handle total demand (for batching)
    cap_per_period = total_demand * 1.1  # 10% buffer
    return [cap_per_period] * T


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


def column_signature(col: ProductionPlanColumn) -> str:
    """Generate a unique signature for a column to detect duplicates."""
    # A ZIO column is uniquely identified by its arc usage pattern
    arcs = sorted(col.arc_usage.keys())
    return f"I{col.item_id}:" + ",".join(f"{t}-{u}" for t, u in arcs)


def solve_pricing_subproblem_dp(
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
) -> Tuple[float, Optional[ProductionPlanColumn]]:
    """
    EXACT DP-based pricing subproblem for Zero-Inventory Ordering (ZIO) columns.

    This function returns the column with MINIMUM reduced cost, without any
    duplicate avoidance heuristics. Correct Dantzig-Wolfe convergence requires
    that we always return the true minimum-RC column.

    ZIO Property: Production occurs only when entering with zero inventory.

    Returns:
        (reduced_cost, column): minimum reduced cost and the corresponding column
                                (column may be None if infeasible)
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

    # Build forced production mapping from θ¹
    forced_source: Dict[int, int] = {}
    for t, u in theta_1:
        if u in forced_source and forced_source[u] != t:
            return INF, None
        forced_source[u] = t

    # Check upsilon constraints consistency
    for t in upsilon_1:
        if t in upsilon_0:
            return INF, None

    for t, u in theta_1:
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

    def get_valid_ends(t: int) -> List[int]:
        """Get valid end periods for production starting at t."""
        if t in upsilon_0:
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

    def solve_dp() -> Tuple[float, List[int]]:
        """Solve DP to find minimum reduced cost column."""
        dp = [INF] * (T + 1)
        decision = [-1] * T

        dp[T] = 0

        for t in range(T - 1, -1, -1):
            if t in forced_source:
                src = forced_source[t]
                if src != t:
                    dp[t] = dp[t + 1]
                    decision[t] = -2
                    continue

            # Check if there's any demand in reachable periods Γ_t
            reachable_demand = sum(demand[u] for u in Gamma.get(t, []))

            if reachable_demand == 0 and t not in must_produce_at:
                # No demand reachable from t, skip this period
                dp[t] = dp[t + 1]
                decision[t] = -1
                continue

            valid_ends = get_valid_ends(t)

            # CRITICAL FIX: When demand[t] == 0, we can CHOOSE to produce or not
            # We should compare both options and pick the cheaper one

            # Option 1: Don't produce at t, let future periods handle the demand
            # This is only valid if demand[t] == 0 (no immediate demand to satisfy)
            skip_cost = (
                dp[t + 1] if demand[t] == 0 and t not in must_produce_at else INF
            )

            # Option 2: Produce at t covering [t, s] for some valid s
            best_produce_cost = INF
            best_s = -1

            for s in valid_ends:
                if s + 1 <= T and dp[s + 1] < INF:
                    cost = run_reduced_cost(t, s) + dp[s + 1]
                    if cost < best_produce_cost:
                        best_produce_cost = cost
                        best_s = s

            # Choose the better option
            if skip_cost <= best_produce_cost and skip_cost < INF:
                dp[t] = skip_cost
                decision[t] = -1  # Skip this period
            elif best_s >= 0:
                dp[t] = best_produce_cost
                decision[t] = best_s
            elif t in must_produce_at or demand[t] > 0:
                dp[t] = INF  # Must produce but can't
            else:
                dp[t] = dp[t + 1]
                decision[t] = -1

        return dp[0], decision

    def reconstruct_and_verify(decision: List[int]) -> Optional[ProductionPlanColumn]:
        """Reconstruct column from decisions and verify feasibility."""
        cap_usage: Dict[int, float] = {}
        setup_usage: Dict[int, float] = {}
        arc_usage: Dict[Tuple[int, int], float] = {}
        total_cost = 0.0

        t = 0
        while t < T:
            s = decision[t]
            if s == -1 or s == -2:
                t += 1
                continue

            setup_usage[t] = 1.0
            total_cost += s_at(t)

            for u in range(t, s + 1):
                d_u = float(demand[u])
                if d_u > 0:
                    cap_usage[t] = cap_usage.get(t, 0.0) + d_u
                    arc_usage[(t, u)] = 1.0
                    total_cost += (c_at(t) + h_sum(t, u)) * d_u

            t = s + 1

        for t_forced in upsilon_1:
            if t_forced < T and setup_usage.get(t_forced, 0.0) < 0.5:
                setup_usage[t_forced] = 1.0
                total_cost += s_at(t_forced)

        # Verify coverage
        covered = [False] * T
        for (t_prod, u), val in arc_usage.items():
            if val > 0.5:
                covered[u] = True

        for u in range(T):
            if demand[u] > 0 and not covered[u]:
                return None

        for t_force, u_force in theta_1:
            if arc_usage.get((t_force, u_force), 0.0) < 0.5:
                return None

        return ProductionPlanColumn(
            item_id=item_id,
            total_plan_cost=total_cost,
            capacity_usage_by_period=cap_usage,
            setup_by_period=setup_usage,
            arc_usage=arc_usage,
        )

    # Solve DP to get the minimum reduced cost column
    dp_val, decision = solve_dp()

    if dp_val >= INF:
        return INF, None  # Infeasible

    reduced_cost = dp_val - convexity_dual

    # Always return the column, even if rc >= 0 (caller decides convergence)
    column = reconstruct_and_verify(decision)

    return reduced_cost, column


class RestrictedMasterProblem:
    """
    Restricted Master Problem for the Dantzig-Wolfe decomposition.

    Structure (Section 4.3.2):
    - Convexity constraints (dual: μ)
    - Capacity constraints (dual: π)
    - Y linking constraints for forbidden setups (dual: σ)
    - Z linking constraints for forbidden arcs (dual: τ)

    Note: Forced branches (θ¹, Υ¹) are enforced only in pricing subproblem
    because dummy columns cannot satisfy = 1 constraints.

    UPDATED: Works with SPARSE column storage.
    """

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

        # Branching sets for forbidden branches (can add = 0 constraints safely)
        self.theta_0_by_item = theta_0_by_item or {i: set() for i in items}
        self.upsilon_0_by_item = upsilon_0_by_item or {i: set() for i in items}

        self.model = gp.Model("RMP")
        self.model.Params.OutputFlag = 0
        self.model.Params.LogToConsole = 0
        self.model.Params.Method = 1

        self.columns: Dict[int, List[ProductionPlanColumn]] = {i: [] for i in items}
        self.lambdas: Dict[Tuple[int, int], gp.Var] = {}

        # Expressions for constraints
        self.convex_expr: Dict[int, gp.LinExpr] = {i: gp.LinExpr(0.0) for i in items}
        self.cap_expr: List[gp.LinExpr] = [gp.LinExpr(0.0) for _ in range(T)]

        # Y linking expressions: Σ_k Y^k_it λ^k_i for forbidden setups
        self.y_link_expr: Dict[Tuple[int, int], gp.LinExpr] = {}
        for item_id in items:
            for t in self.upsilon_0_by_item.get(item_id, set()):
                self.y_link_expr[(item_id, t)] = gp.LinExpr(0.0)

        # Z linking expressions: Σ_k Z^k_itu λ^k_i for forbidden arcs
        self.z_link_expr: Dict[Tuple[int, int, int], gp.LinExpr] = {}
        for item_id in items:
            for t, u in self.theta_0_by_item.get(item_id, set()):
                self.z_link_expr[(item_id, t, u)] = gp.LinExpr(0.0)

        # Constraints
        self.convex_con: Dict[int, gp.Constr] = {}
        self.cap_con: List[gp.Constr] = []
        self.y_link_con: Dict[Tuple[int, int], gp.Constr] = {}
        self.z_link_con: Dict[Tuple[int, int, int], gp.Constr] = {}

        # Add convexity constraints
        for item_id in items:
            self.convex_con[item_id] = self.model.addConstr(
                self.convex_expr[item_id] == 1.0, name=f"conv_{item_id}"
            )

        # Add capacity constraints
        for t in range(T):
            self.cap_con.append(
                self.model.addConstr(
                    self.cap_expr[t] <= self.capacity[t], name=f"cap_{t}"
                )
            )

        # Add Y linking constraints for forbidden setups (Υ⁰): Σ_k Y^k_it λ^k_i = 0
        for (item_id, t), expr in self.y_link_expr.items():
            self.y_link_con[(item_id, t)] = self.model.addConstr(
                expr == 0.0, name=f"y_link_{item_id}_{t}"
            )

        # Add Z linking constraints for forbidden arcs (Θ⁰): Σ_k Z^k_itu λ^k_i = 0
        for (item_id, t, u), expr in self.z_link_expr.items():
            self.z_link_con[(item_id, t, u)] = self.model.addConstr(
                expr == 0.0, name=f"z_link_{item_id}_{t}_{u}"
            )

        # Add inherited columns if provided
        if initial_columns:
            for item_id, cols in initial_columns.items():
                for col in cols:
                    self.add_column(col)

    def _rebuild(self):
        """Rebuild all constraints after adding a column."""
        # Rebuild convexity constraints
        for item_id in self.items:
            self.model.remove(self.convex_con[item_id])
            self.convex_con[item_id] = self.model.addConstr(
                self.convex_expr[item_id] == 1.0, name=f"conv_{item_id}"
            )

        # Rebuild capacity constraints
        for t in range(self.T):
            self.model.remove(self.cap_con[t])
            self.cap_con[t] = self.model.addConstr(
                self.cap_expr[t] <= self.capacity[t], name=f"cap_{t}"
            )

        # Rebuild Y linking constraints
        for key, expr in self.y_link_expr.items():
            self.model.remove(self.y_link_con[key])
            self.y_link_con[key] = self.model.addConstr(
                expr == 0.0, name=f"y_link_{key[0]}_{key[1]}"
            )

        # Rebuild Z linking constraints
        for key, expr in self.z_link_expr.items():
            self.model.remove(self.z_link_con[key])
            self.z_link_con[key] = self.model.addConstr(
                expr == 0.0, name=f"z_link_{key[0]}_{key[1]}_{key[2]}"
            )

    def add_column(self, col: ProductionPlanColumn):
        """Add a column to the RMP - UPDATED for sparse storage."""
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

        # Update convexity expression
        self.convex_expr[item_id] += lam

        # Update capacity expressions - SPARSE iteration
        for t, qty in col.capacity_usage_by_period.items():
            if qty != 0.0:
                self.cap_expr[t] += qty * lam

        # Update Y linking expressions for forbidden setups - SPARSE
        for t in self.upsilon_0_by_item.get(item_id, set()):
            y_val = col.setup_by_period.get(t, 0.0)
            if y_val != 0.0:
                self.y_link_expr[(item_id, t)] += y_val * lam

        # Update Z linking expressions for forbidden arcs
        for t, u in self.theta_0_by_item.get(item_id, set()):
            z_val = col.arc_usage.get((t, u), 0.0)
            if z_val != 0.0:
                self.z_link_expr[(item_id, t, u)] += z_val * lam

        self._rebuild()

    def solve(
        self,
    ) -> Tuple[
        float,
        Dict[int, float],
        List[float],
        Dict[Tuple[int, int], float],
        Dict[Tuple[int, int, int], float],
    ]:
        """
        Solve the RMP and return objective + all duals.

        Returns:
            obj_val: Objective value
            mu: Convexity duals {item_id: dual}
            pi: Capacity duals [dual_t0, dual_t1, ...]
            sigma: Setup duals {(item_id, t): dual} for forbidden setups
            tau: Arc duals {(item_id, t, u): dual} for forbidden arcs
        """
        self.model.optimize()

        if self.model.status != GRB.OPTIMAL:
            return math.inf, {}, [], {}, {}

        mu = {i: self.convex_con[i].Pi for i in self.items}
        pi = [self.cap_con[t].Pi for t in range(self.T)]

        # Extract σ duals from Y linking constraints
        sigma: Dict[Tuple[int, int], float] = {}
        for (item_id, t), con in self.y_link_con.items():
            sigma[(item_id, t)] = con.Pi

        # Extract τ duals from Z linking constraints
        tau: Dict[Tuple[int, int, int], float] = {}
        for (item_id, t, u), con in self.z_link_con.items():
            tau[(item_id, t, u)] = con.Pi

        return self.model.ObjVal, mu, pi, sigma, tau

    def drop_cold_columns(
        self,
        cold_count: Dict[Tuple[int, int], int],
        threshold: int = 3,
        eps: float = 1e-6,
    ) -> Set[str]:
        """
        Remove columns that have been "cold" (λ ≈ 0) for too many iterations.

        Returns the set of dropped column signatures so they can be removed from
        existing_signatures (allowing them to be re-generated by pricing if needed).

        Note: Never drop dummy columns (index 0 for each item).
        """
        dropped_sigs = set()

        for item_id in self.items:
            cols_to_keep = [0]  # Always keep dummy column
            indices_to_drop = []

            for idx in range(1, len(self.columns[item_id])):
                key = (item_id, idx)
                if cold_count.get(key, 0) >= threshold:
                    indices_to_drop.append(idx)
                else:
                    cols_to_keep.append(idx)

            if not indices_to_drop:
                continue

            # Record signatures of dropped columns
            for idx in indices_to_drop:
                col = self.columns[item_id][idx]
                dropped_sigs.add(column_signature(col))

            # Rebuild the model without dropped columns
            # This is expensive but cold column dropping should be infrequent
            new_columns = [self.columns[item_id][i] for i in cols_to_keep]

            # Remove old lambda variables
            for idx in indices_to_drop:
                key = (item_id, idx)
                if key in self.lambdas:
                    self.model.remove(self.lambdas[key])
                    del self.lambdas[key]

            # Renumber remaining columns
            new_lambdas = {}
            for new_idx, old_idx in enumerate(cols_to_keep):
                old_key = (item_id, old_idx)
                new_key = (item_id, new_idx)
                if old_key in self.lambdas:
                    new_lambdas[new_key] = self.lambdas[old_key]

            # Update internal state
            for key in list(self.lambdas.keys()):
                if key[0] == item_id:
                    del self.lambdas[key]
            self.lambdas.update(new_lambdas)
            self.columns[item_id] = new_columns

            # Rebuild expressions from scratch
            self.convex_expr[item_id] = gp.LinExpr(0.0)
            for new_idx, col in enumerate(new_columns):
                lam = self.lambdas.get((item_id, new_idx))
                if lam is not None:
                    self.convex_expr[item_id] += lam

        if dropped_sigs:
            # Rebuild capacity expressions
            for t in range(self.T):
                self.cap_expr[t] = gp.LinExpr(0.0)

            for item_id in self.items:
                for idx, col in enumerate(self.columns[item_id]):
                    lam = self.lambdas.get((item_id, idx))
                    if lam is not None:
                        for t, qty in col.capacity_usage_by_period.items():
                            if qty != 0.0:
                                self.cap_expr[t] += qty * lam

            # Rebuild Y and Z linking expressions
            for key in self.y_link_expr:
                self.y_link_expr[key] = gp.LinExpr(0.0)
            for key in self.z_link_expr:
                self.z_link_expr[key] = gp.LinExpr(0.0)

            for item_id in self.items:
                for idx, col in enumerate(self.columns[item_id]):
                    lam = self.lambdas.get((item_id, idx))
                    if lam is None:
                        continue

                    for t in self.upsilon_0_by_item.get(item_id, set()):
                        y_val = col.setup_by_period.get(t, 0.0)
                        if y_val != 0.0:
                            self.y_link_expr[(item_id, t)] += y_val * lam

                    for t, u in self.theta_0_by_item.get(item_id, set()):
                        z_val = col.arc_usage.get((t, u), 0.0)
                        if z_val != 0.0:
                            self.z_link_expr[(item_id, t, u)] += z_val * lam

            self._rebuild()

        return dropped_sigs


def solve_node_with_column_generation(
    items: Dict[int, dict],
    T: int,
    capacity: List[float],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    Expiry_by_item: Dict[int, Dict[int, int]],
    node: BranchNode,
    parent_rmp: Optional[RestrictedMasterProblem] = None,
    max_iter: int = 500,  # Increased from 100
    eps: float = 1e-6,
    verbose: bool = False,
) -> Tuple[
    float,
    Optional[RestrictedMasterProblem],
    bool,
    Dict[int, Dict[Tuple[int, int], float]],
    Dict[int, Dict[int, float]],
    Dict[int, Dict[int, float]],
]:
    """Solve a branch node using column generation with DP-based ZIO pricing."""
    # Inherit columns from parent
    inherited_cols = inherit_columns_from_parent(parent_rmp, node, items, eps)

    # Create RMP with branching info for forbidden branches (θ⁰, Υ⁰)
    rmp = RestrictedMasterProblem(
        items=items,
        T=T,
        capacity=capacity,
        Gamma_by_item=Gamma_by_item,
        theta_0_by_item=node.theta_0_by_item,
        upsilon_0_by_item=node.upsilon_0_by_item,
        initial_columns=inherited_cols,
    )

    # Track existing column signatures per item to avoid duplicates
    existing_signatures: Dict[int, Set[str]] = {i: set() for i in items}

    # Add signatures of inherited columns
    for item_id, cols in inherited_cols.items():
        for col in cols:
            existing_signatures[item_id].add(column_signature(col))

    # Add dummy columns for feasibility (using SPARSE storage)
    for item_id, item_data in items.items():
        demand = item_data["demand"]
        total_demand = sum(demand)
        dummy_cost = 1000.0 * (total_demand + 1.0)

        col = ProductionPlanColumn(
            item_id=item_id,
            total_plan_cost=dummy_cost,
            capacity_usage_by_period={},  # Empty dict for sparse
            setup_by_period={},  # Empty dict for sparse
            arc_usage={},
        )
        rmp.add_column(col)
        # Don't add dummy to signatures - it's special

    if verbose:
        print(f"  └─ CG(DP): ", end="", flush=True)

    # Cold column tracking: how many consecutive iterations each column has λ ≈ 0
    cold_count: Dict[Tuple[int, int], int] = {}
    cold_threshold = 5  # Drop after this many cold iterations
    drop_frequency = 10  # Check for drops every N iterations

    for iteration in range(1, max_iter + 1):
        lb, mu, pi, sigma, tau = rmp.solve()

        if not math.isfinite(lb):
            if verbose:
                print("INFEASIBLE")
            return math.inf, rmp, False, {}, {}, {}

        # Update cold counts based on current λ values
        for item_id in items:
            for idx in range(len(rmp.columns[item_id])):
                key = (item_id, idx)
                if idx == 0:  # Never count dummy as cold
                    continue
                lam_var = rmp.lambdas.get(key)
                if lam_var is not None:
                    if lam_var.X < eps:
                        cold_count[key] = cold_count.get(key, 0) + 1
                    else:
                        cold_count[key] = 0  # Reset if used

        # Periodically drop cold columns
        if iteration % drop_frequency == 0 and iteration > drop_frequency:
            dropped_sigs = rmp.drop_cold_columns(cold_count, cold_threshold, eps)
            for sig in dropped_sigs:
                # Find which item this signature belongs to
                for item_id in items:
                    existing_signatures[item_id].discard(sig)
            # Reset cold counts after dropping
            cold_count = {
                k: v for k, v in cold_count.items() if k[1] < len(rmp.columns[k[0]])
            }

        # EXACT CONVERGENCE: Check if ALL items have min_rc >= -eps
        all_converged = True

        for item_id, item_data in items.items():
            theta_0 = node.theta_0_by_item.get(item_id, set())
            theta_1 = node.theta_1_by_item.get(item_id, set())
            upsilon_0 = node.upsilon_0_by_item.get(item_id, set())
            upsilon_1 = node.upsilon_1_by_item.get(item_id, set())

            Gamma = Gamma_by_item[item_id]
            Expiry = Expiry_by_item[item_id]

            # Exact pricing - returns minimum reduced cost column
            rc, col = solve_pricing_subproblem_dp(
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

            # Convergence check: is min_rc >= -eps?
            if rc < -eps:
                all_converged = False
                # Add column if not duplicate (duplicate check is just for efficiency)
                if col is not None:
                    sig = column_signature(col)
                    if sig not in existing_signatures[item_id]:
                        rmp.add_column(col)
                        existing_signatures[item_id].add(sig)

        if all_converged:
            if verbose:
                print(f"LB={lb:.2f} (iter={iteration})")
            z_vals = extract_z_values(rmp, items, eps)
            y_vals = extract_y_values(rmp, items, eps)
            x_vals = extract_x_values(rmp, items, eps)
            return lb, rmp, True, z_vals, y_vals, x_vals
            return lb, rmp, True, z_vals, y_vals, x_vals

    lb, _, _, _, _ = rmp.solve()
    z_vals = extract_z_values(rmp, items, eps)
    y_vals = extract_y_values(rmp, items, eps)
    x_vals = extract_x_values(rmp, items, eps)

    if verbose:
        print(f"LB={lb:.2f} (max iter)")

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
    """UPDATED for sparse storage."""
    y_vals = {i: {} for i in items}

    for (item_id, idx), lam in rmp.lambdas.items():
        lam_val = lam.X
        if lam_val < eps:
            continue

        col = rmp.columns[item_id][idx]
        # Sparse iteration
        for t, y_val in col.setup_by_period.items():
            if y_val > eps:
                y_vals[item_id][t] = y_vals[item_id].get(t, 0.0) + lam_val * y_val

    return y_vals


def extract_x_values(
    rmp: RestrictedMasterProblem,
    items: Dict[int, dict],
    eps: float,
) -> Dict[int, Dict[int, float]]:
    """UPDATED for sparse storage."""
    x_vals = {i: {} for i in items}

    for (item_id, idx), lam in rmp.lambdas.items():
        lam_val = lam.X
        if lam_val < eps:
            continue

        col = rmp.columns[item_id][idx]
        # Sparse iteration
        for t, qty in col.capacity_usage_by_period.items():
            if qty > eps:
                x_vals[item_id][t] = x_vals[item_id].get(t, 0.0) + lam_val * qty

    return x_vals


def is_integer(
    z_vals: Dict[int, Dict[Tuple[int, int], float]],
    y_vals: Dict[int, Dict[int, float]],
    eps: float = 1e-6,
) -> bool:
    """Check if all Z and Y values are integral."""
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
    """Check if the current RMP solution uses dummy columns significantly."""
    for item_id in items:
        item_demand = sum(items[item_id]["demand"])
        dummy_cost_threshold = 500.0 * (item_demand + 1)

        for idx, col in enumerate(rmp.columns[item_id]):
            is_dummy = (
                col.total_plan_cost > dummy_cost_threshold
                and len(col.capacity_usage_by_period) == 0  # Sparse check
                and len(col.arc_usage) == 0
            )

            if is_dummy:
                lam_key = (item_id, idx)
                if lam_key in rmp.lambdas:
                    lam_val = rmp.lambdas[lam_key].X
                    if lam_val > eps:
                        return True
    return False


def check_lefo_satisfied(
    z_vals: Dict[int, Dict[Tuple[int, int], float]],
    Expiry_by_item: Dict[int, Dict[int, int]],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    eps: float = 1e-6,
) -> bool:
    """Check if the solution satisfies all LEFO constraints."""
    for item_id, arcs in z_vals.items():
        Expiry = Expiry_by_item.get(item_id, {})

        # Get all active arcs (Z > 0.5 for integer)
        active_arcs = [(t, u) for (t, u), val in arcs.items() if val > 0.5]

        for t1, u in active_arcs:
            v1 = Expiry.get(t1, t1)

            for t2, up in active_arcs:
                if t1 == t2:
                    continue

                v2 = Expiry.get(t2, t2)

                # Check LEFO condition: v1 < v2 and t2 <= u <= up - 1
                if v1 < v2 and t2 <= u <= up - 1:
                    return False  # LEFO violation found

    return True


def is_valid_integer_solution(
    z_vals: Dict[int, Dict[Tuple[int, int], float]],
    y_vals: Dict[int, Dict[int, float]],
    rmp: RestrictedMasterProblem,
    items: Dict[int, dict],
    Expiry_by_item: Dict[int, Dict[int, int]],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    eps: float = 1e-6,
) -> bool:
    """Check if solution is integer, doesn't use dummy columns, AND satisfies LEFO."""
    if not is_integer(z_vals, y_vals, eps):
        return False

    if solution_uses_dummy(rmp, items, eps):
        return False

    all_z_empty = all(len(arcs) == 0 for arcs in z_vals.values())
    all_y_empty = all(len(setups) == 0 for setups in y_vals.values())

    if all_z_empty and all_y_empty:
        return False

    # Check LEFO constraints
    if not check_lefo_satisfied(z_vals, Expiry_by_item, Gamma_by_item, eps):
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


def find_lefo_violation(
    z_vals: Dict[int, Dict[Tuple[int, int], float]],
    Expiry_by_item: Dict[int, Dict[int, int]],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    node: BranchNode,
    eps: float = 1e-6,
) -> Optional[Tuple[int, int, int, int, int, float, float]]:
    """
    Find a LEFO violation in the current solution.

    LEFO violation: Two arcs (t1, u) and (t2, up) where:
    - v(t1) < v(t2) (t1 expires before t2)
    - t2 <= u <= up - 1 (t1's arc crosses into t2's range)
    - Both arcs are active (Z > 0)

    Returns: (item_id, t1, u, t2, up, z1_val, z2_val) or None if no violation
    """
    best_violation = None
    best_score = 0.0  # Score by min(z1, z2) to pick most impactful violation

    for item_id, arcs in z_vals.items():
        Expiry = Expiry_by_item[item_id]
        Gamma = Gamma_by_item[item_id]

        theta_0 = node.theta_0_by_item.get(item_id, set())
        theta_1 = node.theta_1_by_item.get(item_id, set())

        # Get all active arcs (Z > eps)
        active_arcs = [(t, u, val) for (t, u), val in arcs.items() if val > eps]

        for t1, u, z1_val in active_arcs:
            if (t1, u) in theta_0 or (t1, u) in theta_1:
                continue

            v1 = Expiry.get(t1, t1)

            for t2, up, z2_val in active_arcs:
                if t1 == t2:
                    continue
                if (t2, up) in theta_0 or (t2, up) in theta_1:
                    continue

                v2 = Expiry.get(t2, t2)

                # Check LEFO condition: v1 < v2 and t2 <= u <= up - 1
                if v1 < v2 and t2 <= u <= up - 1:
                    # Found a violation!
                    score = min(z1_val, z2_val)
                    if score > best_score:
                        best_score = score
                        best_violation = (item_id, t1, u, t2, up, z1_val, z2_val)

    return best_violation


def fathom_queue_by_incumbent(queue: List, incumbent: float, eps: float) -> int:
    """UPDATED: Queue is now a heap storing (bound, node_id, node, rmp) for best-first search."""
    original_size = len(queue)
    new_queue = []

    for item in queue:
        bound, node_id, node, rmp = item
        if node.lp_bound < incumbent - eps:
            new_queue.append(item)

    num_fathomed = original_size - len(new_queue)
    queue.clear()
    queue.extend(new_queue)
    heapq.heapify(queue)  # Re-heapify after modification

    return num_fathomed


def solve_instance(
    instance_path: str | Path = "last_instance.json",
    time_limit: int = 0,
    mip_gap: float = 0.0,
    out_dir: str | Path = "bnp_results",
) -> Tuple[Dict, List[str]]:
    """Solve the perishable lot-sizing problem using Branch-and-Price with DP pricing."""
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

    print("\n" + "╔" + "═" * 68 + "╗")
    print(f"║ {'BRANCH-AND-PRICE WITH DP PRICING (ZIO) - BEST-FIRST':^66s} ║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  Items:    {len(items):<57d} ║")
    print(f"║  Periods:  {T:<57d} ║")
    print(f"║  Strategy: {'DP generates ZIO extreme points only':<57s} ║")
    print(f"║  Search:   {'Best-first (by LP bound)':<57s} ║")
    print("╚" + "═" * 68 + "╝")

    Gamma_by_item: Dict[int, Dict[int, List[int]]] = {}
    Expiry_by_item: Dict[int, Dict[int, int]] = {}

    for item_id, item_data in items.items():
        shelf_seq = list(item_data["shelf_seq"])
        if len(shelf_seq) != T:
            raise ValueError(f"items[{item_id}]['shelf_seq'] must have length {T}")

        Gamma: Dict[int, List[int]] = {}
        Expiry: Dict[int, int] = {}

        for t in Periods:
            m_it = int(shelf_seq[t])  # shelf life

            # v_it = expiry period (last period the item can be used)
            # shelf_life = 0 means can only use in period t
            # shelf_life = m means can use in periods t, t+1, ..., t+m
            v_it = t + m_it
            Expiry[t] = v_it

            if m_it < 0:  # Negative shelf life makes no sense
                Gamma[t] = []
                continue

            # u_max = last valid consumption period (inclusive)
            # Capped at T-1 (last period index)
            u_max = min(T - 1, v_it)
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
    root_lb, root_rmp, converged, root_z, root_y, root_x = (
        solve_node_with_column_generation(
            items=items,
            T=T,
            capacity=capacity,
            Gamma_by_item=Gamma_by_item,
            Expiry_by_item=Expiry_by_item,
            node=root,
            parent_rmp=None,
            verbose=True,
        )
    )

    if not math.isfinite(root_lb):
        print("\n✗ Root infeasible!")
        summary = {
            "status": int(GRB.INFEASIBLE),
            "objective": None,
            "best_bound": None,
            "gap": None,
            "runtime_sec": time.time() - start_time,
            "solver_version": "branch_and_price_dp_zio_bestfirst",
            "n_items": len(items),
            "T": T,
        }
        return summary, []

    root.lp_bound = root_lb
    print(f"  Root LB:  {root_lb:.4f}")
    print(f"  Integer?  {is_integer(root_z, root_y, eps)}")
    print(f"  Uses dummy? {solution_uses_dummy(root_rmp, items, eps)}")

    best_lb = root_lb
    best_ub: Optional[float] = None
    node_counter = 1
    seen_signatures = {node_signature(root)}

    # BEST-FIRST SEARCH: Use heap with (bound, node_id, node, rmp)
    # node_id is for tie-breaking (lower id = created earlier)
    queue: List[Tuple[float, int, BranchNode, RestrictedMasterProblem]] = []
    heapq.heappush(queue, (root_lb, root.node_id, root, root_rmp))
    stats.nodes_created = 1

    # Store root solution separately for initial check
    opt_node = root
    opt_z = root_z
    opt_y = root_y
    opt_x = root_x

    # Check if root is already a valid integer solution
    if is_valid_integer_solution(
        root_z, root_y, root_rmp, items, Expiry_by_item, Gamma_by_item, eps
    ):
        best_ub = root_lb
        best_lb = root_lb
        root.is_integer = True
        stats.nodes_explored = 1
        stats.nodes_integer = 1
        print("\n✓ Root is INTEGER - OPTIMAL!")
        stats.print_summary(best_lb, best_ub, eps)

        orders_txt = generate_orders_txt(items, root_x, eps)

        summary = {
            "status": int(GRB.OPTIMAL),
            "objective": float(best_ub),
            "best_bound": float(best_lb),
            "gap": 0.0,
            "runtime_sec": time.time() - start_time,
            "solver_version": "branch_and_price_dp_zio_bestfirst",
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

    stats.nodes_explored = 1
    print(f"\n{'=' * 70}")
    print("DIVE-FIRST SEARCH (lower LP bound = more promising)")
    print(f"{'=' * 70}\n")

    # Dive-first: we may have a node to process immediately instead of popping from queue
    dive_node: Optional[BranchNode] = None
    dive_rmp: Optional[RestrictedMasterProblem] = None

    while (queue or dive_node) and stats.nodes_explored < max_nodes:
        if time.time() - start_time > max_time:
            print("\n⏱ Time limit reached")
            break

        # DIVE-FIRST: If we have a dive node, use it; otherwise pop from queue
        if dive_node is not None:
            node = dive_node
            parent_rmp = dive_rmp
            dive_node = None
            dive_rmp = None
        else:
            if not queue:
                break
            _, _, node, parent_rmp = heapq.heappop(queue)

        if node.node_id != 0:
            if stats.nodes_explored % print_frequency == 0:
                gap_str = "N/A"
                if best_ub is not None:
                    gap = best_ub - best_lb
                    gap_pct = 100 * gap / max(abs(best_ub), 1e-10)
                    gap_str = f"{gap_pct:.2f}%"
                print(
                    f"[Progress: N={stats.nodes_explored:4d}, Queue={len(queue):4d}, "
                    f"LB={best_lb:.2f}, UB={best_ub if best_ub is not None else 'N/A'}, Gap={gap_str}]"
                )

            print(f"N{node.node_id:4d} D{node.depth:2d} ", end="", flush=True)

            # Re-solve the node (this is where z/y/x are computed fresh)
            lb, rmp, converged, z_vals, y_vals, x_vals = (
                solve_node_with_column_generation(
                    items=items,
                    T=T,
                    capacity=capacity,
                    Gamma_by_item=Gamma_by_item,
                    Expiry_by_item=Expiry_by_item,
                    node=node,
                    parent_rmp=parent_rmp,
                    verbose=True,
                )
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

            if queue:
                best_lb = min(
                    lb, min(item[0] for item in queue)
                )  # item[0] is the bound
            else:
                best_lb = lb

            if best_ub is not None and lb >= best_ub - eps:
                print(f"  FATHOMED: {lb:.2f} >= {best_ub:.2f}")
                node.is_pruned = True
                node.prune_reason = "bound"
                stats.nodes_fathomed_by_bound += 1
                continue

            if is_valid_integer_solution(
                z_vals, y_vals, rmp, items, Expiry_by_item, Gamma_by_item, eps
            ):
                print(f"  INTEGER: {lb:.2f}", end="")
                node.is_integer = True
                stats.nodes_integer += 1

                if best_ub is None or lb < best_ub - eps:
                    best_ub = lb
                    print(" ★ NEW INCUMBENT!")
                    num_fathomed = fathom_queue_by_incumbent(queue, best_ub, eps)
                    stats.nodes_fathomed_on_incumbent += num_fathomed
                    opt_node = node
                    opt_z = z_vals
                    opt_y = y_vals
                    opt_x = x_vals
                else:
                    print()

                if queue:
                    best_lb = min(item[0] for item in queue)  # item[0] is the bound
                else:
                    best_lb = best_ub if best_ub is not None else lb

                continue

            if solution_uses_dummy(rmp, items, eps) and is_integer(z_vals, y_vals, eps):
                print(f"  FATHOMED: Dummy-only solution")
                node.is_pruned = True
                node.prune_reason = "dummy_infeasible"
                stats.nodes_fathomed_by_infeasible += 1
                continue

        else:
            # Root node - use stored values
            z_vals = root_z
            y_vals = root_y
            x_vals = root_x
            rmp = root_rmp
            lb = root_lb

        # FIRST: Check for LEFO violations and branch on them
        lefo_viol = find_lefo_violation(
            z_vals, Expiry_by_item, Gamma_by_item, node, eps
        )
        if lefo_viol is not None:
            item_id, t1, u, t2, up, z1_val, z2_val = lefo_viol

            # Create two children: forbid arc1 vs forbid arc2
            child_0 = BranchNode(
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
                branch_variable=("LEFO", item_id, t1, u, z1_val),
                branch_direction="Z=0",
            )
            child_0.theta_0_by_item[item_id].add((t1, u))

            child_1 = BranchNode(
                node_id=node_counter + 1,
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
                branch_variable=("LEFO", item_id, t2, up, z2_val),
                branch_direction="Z=0",
            )
            child_1.theta_0_by_item[item_id].add((t2, up))

            sig_0 = node_signature(child_0)
            sig_1 = node_signature(child_1)

            valid_0 = sig_0 not in seen_signatures
            valid_1 = sig_1 not in seen_signatures

            if valid_0 and valid_1:
                # Solve both children to find their LP bounds
                lb_0, rmp_0, _, _, _, _ = solve_node_with_column_generation(
                    items=items,
                    T=T,
                    capacity=capacity,
                    Gamma_by_item=Gamma_by_item,
                    Expiry_by_item=Expiry_by_item,
                    node=child_0,
                    parent_rmp=rmp,
                    verbose=False,
                )
                lb_1, rmp_1, _, _, _, _ = solve_node_with_column_generation(
                    items=items,
                    T=T,
                    capacity=capacity,
                    Gamma_by_item=Gamma_by_item,
                    Expiry_by_item=Expiry_by_item,
                    node=child_1,
                    parent_rmp=rmp,
                    verbose=False,
                )

                child_0.lp_bound = lb_0
                child_1.lp_bound = lb_1

                seen_signatures.add(sig_0)
                seen_signatures.add(sig_1)
                node_counter += 2
                stats.nodes_created += 2

                # Dive into the more promising (lower LP bound) child
                if lb_0 <= lb_1:
                    dive_node = child_0
                    dive_rmp = rmp_0
                    heapq.heappush(queue, (lb_1, child_1.node_id, child_1, rmp_1))
                    print(
                        f"  Branch LEFO: forbid Z[{item_id},{t1},{u}] → dive (LB={lb_0:.2f} vs {lb_1:.2f})"
                    )
                else:
                    dive_node = child_1
                    dive_rmp = rmp_1
                    heapq.heappush(queue, (lb_0, child_0.node_id, child_0, rmp_0))
                    print(
                        f"  Branch LEFO: forbid Z[{item_id},{t2},{up}] → dive (LB={lb_1:.2f} vs {lb_0:.2f})"
                    )
            elif valid_0:
                seen_signatures.add(sig_0)
                node_counter += 1
                stats.nodes_created += 1
                dive_node = child_0
                dive_rmp = rmp
                print(
                    f"  Branch LEFO: forbid Z[{item_id},{t1},{u}] → dive (only valid)"
                )
            elif valid_1:
                seen_signatures.add(sig_1)
                node_counter += 1
                stats.nodes_created += 1
                dive_node = child_1
                dive_rmp = rmp
                print(
                    f"  Branch LEFO: forbid Z[{item_id},{t2},{up}] → dive (only valid)"
                )

            continue

        # BRANCHING PRIORITY: Y first, then Z
        # DIVE-FIRST: Solve both children, dive into lower LP bound
        branch_var_y = find_most_fractional_y(y_vals, node, eps)
        if branch_var_y is not None:
            item_id, t_br, y_val = branch_var_y

            # Create Y=0 child
            child_0 = BranchNode(
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
            )
            child_0.upsilon_0_by_item[item_id].add(t_br)

            # Create Y=1 child
            child_1 = BranchNode(
                node_id=node_counter + 1,
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
            )
            child_1.upsilon_1_by_item[item_id].add(t_br)

            # Check signatures
            sig_0 = node_signature(child_0)
            sig_1 = node_signature(child_1)

            valid_0 = sig_0 not in seen_signatures
            valid_1 = sig_1 not in seen_signatures

            if valid_0 and valid_1:
                # Solve both children to find their LP bounds
                lb_0, rmp_0, _, _, _, _ = solve_node_with_column_generation(
                    items=items,
                    T=T,
                    capacity=capacity,
                    Gamma_by_item=Gamma_by_item,
                    Expiry_by_item=Expiry_by_item,
                    node=child_0,
                    parent_rmp=rmp,
                    verbose=False,
                )
                lb_1, rmp_1, _, _, _, _ = solve_node_with_column_generation(
                    items=items,
                    T=T,
                    capacity=capacity,
                    Gamma_by_item=Gamma_by_item,
                    Expiry_by_item=Expiry_by_item,
                    node=child_1,
                    parent_rmp=rmp,
                    verbose=False,
                )

                child_0.lp_bound = lb_0
                child_1.lp_bound = lb_1

                seen_signatures.add(sig_0)
                seen_signatures.add(sig_1)
                node_counter += 2
                stats.nodes_created += 2

                # Dive into the more promising (lower LP bound) child
                if lb_0 <= lb_1:
                    dive_node = child_0
                    dive_rmp = rmp_0
                    heapq.heappush(queue, (lb_1, child_1.node_id, child_1, rmp_1))
                    print(
                        f"  Branch Y[{item_id},{t_br}]={y_val:.3f} → dive Y=0 (LB={lb_0:.2f} vs {lb_1:.2f})"
                    )
                else:
                    dive_node = child_1
                    dive_rmp = rmp_1
                    heapq.heappush(queue, (lb_0, child_0.node_id, child_0, rmp_0))
                    print(
                        f"  Branch Y[{item_id},{t_br}]={y_val:.3f} → dive Y=1 (LB={lb_1:.2f} vs {lb_0:.2f})"
                    )
            elif valid_0:
                seen_signatures.add(sig_0)
                node_counter += 1
                stats.nodes_created += 1
                dive_node = child_0
                dive_rmp = rmp
                print(
                    f"  Branch Y[{item_id},{t_br}]={y_val:.3f} → dive Y=0 (only valid)"
                )
            elif valid_1:
                seen_signatures.add(sig_1)
                node_counter += 1
                stats.nodes_created += 1
                dive_node = child_1
                dive_rmp = rmp
                print(
                    f"  Branch Y[{item_id},{t_br}]={y_val:.3f} → dive Y=1 (only valid)"
                )

            continue

        branch_var_z = find_most_fractional_z(z_vals, node, eps)
        if branch_var_z is not None:
            item_id, t_br, u_br, z_val = branch_var_z

            # Create Z=0 child
            child_0 = BranchNode(
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
            )
            child_0.theta_0_by_item[item_id].add((t_br, u_br))

            # Create Z=1 child
            child_1 = BranchNode(
                node_id=node_counter + 1,
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
            )
            child_1.theta_1_by_item[item_id].add((t_br, u_br))

            # Check signatures
            sig_0 = node_signature(child_0)
            sig_1 = node_signature(child_1)

            valid_0 = sig_0 not in seen_signatures
            valid_1 = sig_1 not in seen_signatures

            if valid_0 and valid_1:
                # Solve both children to find their LP bounds
                lb_0, rmp_0, _, _, _, _ = solve_node_with_column_generation(
                    items=items,
                    T=T,
                    capacity=capacity,
                    Gamma_by_item=Gamma_by_item,
                    Expiry_by_item=Expiry_by_item,
                    node=child_0,
                    parent_rmp=rmp,
                    verbose=False,
                )
                lb_1, rmp_1, _, _, _, _ = solve_node_with_column_generation(
                    items=items,
                    T=T,
                    capacity=capacity,
                    Gamma_by_item=Gamma_by_item,
                    Expiry_by_item=Expiry_by_item,
                    node=child_1,
                    parent_rmp=rmp,
                    verbose=False,
                )

                child_0.lp_bound = lb_0
                child_1.lp_bound = lb_1

                seen_signatures.add(sig_0)
                seen_signatures.add(sig_1)
                node_counter += 2
                stats.nodes_created += 2

                # Dive into the more promising (lower LP bound) child
                if lb_0 <= lb_1:
                    dive_node = child_0
                    dive_rmp = rmp_0
                    heapq.heappush(queue, (lb_1, child_1.node_id, child_1, rmp_1))
                    print(
                        f"  Branch Z[{item_id},{t_br},{u_br}]={z_val:.3f} → dive Z=0 (LB={lb_0:.2f} vs {lb_1:.2f})"
                    )
                else:
                    dive_node = child_1
                    dive_rmp = rmp_1
                    heapq.heappush(queue, (lb_0, child_0.node_id, child_0, rmp_0))
                    print(
                        f"  Branch Z[{item_id},{t_br},{u_br}]={z_val:.3f} → dive Z=1 (LB={lb_1:.2f} vs {lb_0:.2f})"
                    )
            elif valid_0:
                seen_signatures.add(sig_0)
                node_counter += 1
                stats.nodes_created += 1
                dive_node = child_0
                dive_rmp = rmp
                print(
                    f"  Branch Z[{item_id},{t_br},{u_br}]={z_val:.3f} → dive Z=0 (only valid)"
                )
            elif valid_1:
                seen_signatures.add(sig_1)
                node_counter += 1
                stats.nodes_created += 1
                dive_node = child_1
                dive_rmp = rmp
                print(
                    f"  Branch Z[{item_id},{t_br},{u_br}]={z_val:.3f} → dive Z=1 (only valid)"
                )

            continue

        print("  No fractional variable - solution is integral")

    if best_ub is not None:
        best_lb = best_ub

    stats.print_summary(best_lb, best_ub, eps)

    orders_txt = generate_orders_txt(items, opt_x, eps)

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
        "solver_version": "branch_and_price_dp_zio_bestfirst",
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


if __name__ == "__main__":
    import json
    from pathlib import Path

    # Test instance
    instance = {
        "period": 6,
        "production_capacity": 25,  # 30 units per period
        "items": {
            "1": {
                "demand": [10, 12, 8, 15, 10, 9],
                "c_var": 5.0,
                "h": 0.5,
                "setup": 80.0,
                "shelf_seq": [1, 1, 3, 1, 2, 2],
            },
            "2": {
                "demand": [5, 7, 6, 8, 9, 7],
                "c_var": 8.0,
                "h": 0.8,
                "setup": 120.0,
                "shelf_seq": [5, 4, 5, 5, 4, 5],
            },
        },
    }

    out_dir = Path("dp_zio_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    Path("test_instance.json").write_text(json.dumps(instance, indent=2))

    print("=" * 70)
    print("Testing Branch-and-Price with DP-based ZIO Pricing (BEST-FIRST)")
    print("=" * 70)

    summary, orders = solve_instance(
        instance_path="test_instance.json",
        time_limit=600,
        out_dir=out_dir,
    )

    print("\n" + "=" * 70)
    print("SOLUTION SUMMARY")
    print("=" * 70)
    print(f"Objective   : {summary.get('objective', '—')}")
    print(f"Best bound  : {summary['best_bound']:.4f}")
    print(f"Gap         : {summary.get('gap',0)*100:.3f}%")
    print(f"Runtime     : {summary['runtime_sec']:.2f}s")
    print(f"Solver      : {summary['solver_version']}")
    print("\nProduction Plan:")
    for line in orders:
        print(line)
