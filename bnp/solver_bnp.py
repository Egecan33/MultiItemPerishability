"""
Branch-and-Price for Perishable Lot-Sizing with Heterogeneous Shelf Lives and LEFO
Optimized with column inheritance (warm-starting child nodes from parent).

Features:
- Column inheritance for faster convergence at branch nodes
- σ (sigma) and τ (tau) duals for forbidden branches to guide pricing (Section 4.3.2)
- Dummy column detection to prevent false integer solutions
- Branching on Z (arc) and Y (setup) variables

Note on dual variable handling:
- At root node: σ = τ = 0 (no branching constraints)
- At branch nodes with forbidden constraints (θ⁰, Υ⁰): Extract σ, τ from RMP
- At branch nodes with forced constraints (θ¹, Υ¹): Cannot add to RMP (dummy incompatible),
  enforce only in pricing subproblem
"""

from __future__ import annotations
import json
import math
import time
from collections import deque
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
            if t < len(self.setup_by_period) and self.setup_by_period[t] > eps:
                return True

        # Check upsilon_1: MUST use forced setups
        for t in upsilon_1:
            if t >= len(self.setup_by_period) or self.setup_by_period[t] < 1.0 - eps:
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


def column_signature(col: ProductionPlanColumn) -> str:
    """Generate a unique signature for a column to detect duplicates."""
    # A column is uniquely identified by its arc usage pattern
    arcs = sorted((t, u) for (t, u), val in col.arc_usage.items() if val > 0.5)
    return f"I{col.item_id}:" + ",".join(f"{t}-{u}" for t, u in arcs)


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
    existing_signatures: Optional[Set[str]] = None,
    arc_usage_counts: Optional[Dict[Tuple[int, int], int]] = None,
    perturbation_eps: float = 1e-5,
) -> Tuple[float, Optional[ProductionPlanColumn]]:
    """
    Solve the pricing subproblem for a single item.

    Reduced cost formula (Section 4.3.2, Equation 7):
        rc = c^k_i - μ_i - Σ_t π_t X^k_it - Σ_t σ_it Y^k_it - Σ_{t,u} τ_itu Z^k_itu

    Where:
        c^k_i: Total cost of column k for item i
        μ_i: Convexity dual
        π_t: Capacity dual for period t
        σ_it: Setup linking dual (for forbidden setups)
        τ_itu: Arc linking dual (for forbidden arcs)
    """
    sigma = sigma or {}
    tau = tau or {}
    existing_signatures = existing_signatures or set()
    arc_usage_counts = arc_usage_counts or {}

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

    Triples: List[Tuple[int, int]] = []
    for t in range(T):
        for u in Gamma.get(t, []):
            Triples.append((t, u))

    mu_t: Dict[int, float] = {}
    for t in range(T):
        mu_t[t] = sum(float(demand[u]) for u in Gamma.get(t, []))

    model = gp.Model(f"pricing_item_{item_id}")
    model.Params.OutputFlag = 0
    model.Params.LogToConsole = 0

    X: Dict[Tuple[int, int], gp.Var] = {}
    Z: Dict[Tuple[int, int], gp.Var] = {}
    for t, u in Triples:
        X[t, u] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"X_{t}_{u}")
        Z[t, u] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f"Z_{t}_{u}")

    Y: Dict[int, gp.Var] = {}
    for t in range(T):
        Y[t] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f"Y_{t}")

    model.update()

    for u in range(T):
        if demand[u] <= 0:
            continue
        expr = gp.LinExpr()
        for t in range(u + 1):
            if (t, u) in X:
                expr += X[t, u]
        model.addConstr(expr == float(demand[u]), name=f"demand_{u}")

    for t in range(T):
        if not Gamma.get(t):
            model.addConstr(Y[t] == 0.0, name=f"setup_zero_{t}")
            continue
        expr = gp.LinExpr()
        for u in Gamma[t]:
            if (t, u) in X:
                expr += X[t, u]
        model.addConstr(expr <= mu_t[t] * Y[t], name=f"setupLink_{t}")

    for t, u in Triples:
        C_u = float(demand[u])
        model.addConstr(X[t, u] <= C_u * Z[t, u], name=f"arc_on_{t}_{u}")

    prods = [t for t in range(T) if Gamma.get(t)]
    prods.sort(key=lambda t: Expiry.get(t, t))

    for idx1 in range(len(prods)):
        t1 = prods[idx1]
        v1 = Expiry.get(t1, t1)
        for idx2 in range(idx1 + 1, len(prods)):
            t2 = prods[idx2]
            v2 = Expiry.get(t2, t2)
            if v1 >= v2:
                continue
            for up in Gamma.get(t2, []):
                for u in [uu for uu in Gamma.get(t1, []) if t2 <= uu <= up - 1]:
                    if (t1, u) in Z and (t2, up) in Z:
                        model.addConstr(
                            Z[t1, u] + Z[t2, up] <= 1,
                            name=f"nocross_{t1}_{t2}_{u}_{up}",
                        )

    # Branching constraints (enforced in pricing, not RMP)
    for t_forb, u_forb in theta_0:
        if (t_forb, u_forb) in Z:
            model.addConstr(
                Z[t_forb, u_forb] == 0.0, name=f"branch_Z0_{t_forb}_{u_forb}"
            )

    for t_force, u_force in theta_1:
        if (t_force, u_force) in Z:
            model.addConstr(
                Z[t_force, u_force] == 1.0, name=f"branch_Z1_{t_force}_{u_force}"
            )
        else:
            model.addConstr(0.0 == 1.0, name=f"branch_impossible_{t_force}_{u_force}")

    for t_forb in upsilon_0:
        if t_forb in Y:
            model.addConstr(Y[t_forb] == 0.0, name=f"branch_Y0_{t_forb}")

    for t_force in upsilon_1:
        if t_force in Y:
            model.addConstr(Y[t_force] == 1.0, name=f"branch_Y1_{t_force}")

    # Objective: reduced cost with all duals (Section 4.3.2, Equation 7)
    # rc = c^k_i - μ_i - Σ π_t X - Σ σ_it Y - Σ τ_itu Z
    obj = gp.LinExpr()

    # Production and holding costs minus capacity duals
    for t, u in Triples:
        unit_cost = c_at(t) + h_sum(t, u)
        obj += unit_cost * X[t, u]
        obj -= capacity_duals[t] * X[t, u]

    # Setup costs minus sigma duals
    for t in range(T):
        obj += s_at(t) * Y[t]
        # Subtract sigma dual if we have one for this (item, period)
        sigma_val = sigma.get((item_id, t), 0.0)
        if sigma_val != 0.0:
            obj -= sigma_val * Y[t]

    # Subtract tau duals for arcs
    for t, u in Triples:
        tau_val = tau.get((item_id, t, u), 0.0)
        if tau_val != 0.0:
            obj -= tau_val * Z[t, u]
        # Add perturbation based on arc usage count to encourage diversification
        arc_count = arc_usage_counts.get((t, u), 0)
        if arc_count > 0:
            obj += perturbation_eps * arc_count * Z[t, u]

    # Subtract convexity dual
    obj -= convexity_dual

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        return math.inf, None

    reduced_cost = model.ObjVal
    if reduced_cost >= -eps:
        return reduced_cost, None

    cap_usage = [0.0] * T
    setup_usage = [0.0] * T
    arc_usage: Dict[Tuple[int, int], float] = {}
    total_cost = 0.0

    for t, u in Triples:
        x_val = X[t, u].X
        if x_val > eps:
            cap_usage[t] += x_val
            total_cost += (c_at(t) + h_sum(t, u)) * x_val
        arc_usage[(t, u)] = Z[t, u].X

    for t in range(T):
        y_val = Y[t].X
        if y_val > 0.5:
            setup_usage[t] = y_val
            total_cost += s_at(t) * y_val

    column = ProductionPlanColumn(
        item_id=item_id,
        total_plan_cost=total_cost,
        capacity_usage_by_period=cap_usage,
        setup_by_period=setup_usage,
        arc_usage=arc_usage,
    )

    # Check for duplicate
    sig = column_signature(column)
    if sig in existing_signatures:
        return reduced_cost, None  # Don't return duplicate

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

        # Update capacity expressions
        for t in range(self.T):
            if col.capacity_usage_by_period[t] != 0.0:
                self.cap_expr[t] += col.capacity_usage_by_period[t] * lam

        # Update Y linking expressions for forbidden setups
        for t in self.upsilon_0_by_item.get(item_id, set()):
            if t < len(col.setup_by_period):
                y_val = col.setup_by_period[t]
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


def solve_node_with_column_generation(
    items: Dict[int, dict],
    T: int,
    capacity: List[float],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    Expiry_by_item: Dict[int, Dict[int, int]],
    node: BranchNode,
    parent_rmp: Optional[RestrictedMasterProblem] = None,
    max_iter: int = 100,
    eps: float = 1e-6,
    verbose: bool = False,
    use_mip_pricing: bool = True,
) -> Tuple[
    float,
    Optional[RestrictedMasterProblem],
    bool,
    Dict[int, Dict[Tuple[int, int], float]],
    Dict[int, Dict[int, float]],
    Dict[int, Dict[int, float]],
]:
    """Solve a branch node using column generation with column inheritance."""
    # Inherit columns from parent
    inherited_cols = inherit_columns_from_parent(parent_rmp, node, items, eps)

    # Create RMP with branching info for forbidden branches (θ⁰, Υ⁰)
    # This allows extracting σ and τ duals to guide pricing
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

    # Track arc usage counts for perturbation (diversification)
    arc_usage_counts: Dict[int, Dict[Tuple[int, int], int]] = {i: {} for i in items}

    # Add signatures of inherited columns and count their arc usage
    for item_id, cols in inherited_cols.items():
        for col in cols:
            existing_signatures[item_id].add(column_signature(col))
            for (t, u), val in col.arc_usage.items():
                if val > 0.5:
                    arc_usage_counts[item_id][(t, u)] = (
                        arc_usage_counts[item_id].get((t, u), 0) + 1
                    )

    # Add dummy columns
    for item_id, item_data in items.items():
        demand = item_data["demand"]
        total_demand = sum(demand)
        dummy_cost = 1000.0 * (total_demand + 1.0)

        col = ProductionPlanColumn(
            item_id=item_id,
            total_plan_cost=dummy_cost,
            capacity_usage_by_period=[0.0] * T,
            setup_by_period=[0.0] * T,
            arc_usage={},
        )
        rmp.add_column(col)
        # Don't add dummy to signatures - it's special

    if verbose:
        print(f"  └─ CG: ", end="", flush=True)

    for iteration in range(1, max_iter + 1):
        lb, mu, pi, sigma, tau = rmp.solve()

        if not math.isfinite(lb):
            if verbose:
                print("INFEASIBLE")
            return math.inf, rmp, False, {}, {}, {}

        any_added = False
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
                use_mip=use_mip_pricing,
                existing_signatures=existing_signatures[item_id],
                arc_usage_counts=arc_usage_counts[item_id],
                perturbation_eps=1e-5,
            )

            if col is not None and rc < -eps:
                sig = column_signature(col)
                if sig not in existing_signatures[item_id]:
                    rmp.add_column(col)
                    existing_signatures[item_id].add(sig)
                    # Update arc usage counts
                    for (t, u), val in col.arc_usage.items():
                        if val > 0.5:
                            arc_usage_counts[item_id][(t, u)] = (
                                arc_usage_counts[item_id].get((t, u), 0) + 1
                            )
                    any_added = True

        if not any_added:
            if verbose:
                print(f"LB={lb:.2f} (iter={iteration})")
            z_vals = extract_z_values(rmp, items, eps)
            y_vals = extract_y_values(rmp, items, eps)
            x_vals = extract_x_values(rmp, items, eps)
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
    """
    Check if the current RMP solution uses dummy columns significantly.

    Dummy columns are identified by their characteristics:
    - Very high cost (proportional to total demand)
    - Zero capacity usage
    - Empty arc usage
    """
    for item_id in items:
        item_demand = sum(items[item_id]["demand"])
        # Dummy cost is 10000 * (total_demand + 1), use 5000 as threshold
        dummy_cost_threshold = 5000.0 * (item_demand + 1)

        for idx, col in enumerate(rmp.columns[item_id]):
            # Check if this looks like a dummy column
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
    """
    Check if solution is both integer AND doesn't use dummy columns.

    A solution that is 'integer' but only uses dummy columns is actually
    infeasible (dummy columns don't satisfy demand).
    """
    # Must be integral
    if not is_integer(z_vals, y_vals, eps):
        return False

    # Must not use dummy columns
    if solution_uses_dummy(rmp, items, eps):
        return False

    # Check that solution actually produces something (not all empty)
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


def fathom_queue_by_incumbent(queue: deque, incumbent: float, eps: float) -> int:
    original_size = len(queue)
    new_queue = deque()

    for node, z_vals, y_vals, x_vals, rmp in queue:
        if node.lp_bound < incumbent - eps:
            new_queue.append((node, z_vals, y_vals, x_vals, rmp))

    num_fathomed = original_size - len(new_queue)
    queue.clear()
    queue.extend(new_queue)

    return num_fathomed


def solve_instance(
    instance_path: str | Path = "last_instance.json",
    time_limit: int = 0,
    mip_gap: float = 0.0,
    out_dir: str | Path = "bnp_results",
) -> Tuple[Dict, List[str]]:
    """Solve the perishable lot-sizing problem using Branch-and-Price."""
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
    use_mip_pricing = True

    print("\n" + "╔" + "═" * 68 + "╗")
    print(f"║ {'BRANCH-AND-PRICE: PERISHABLE LOT-SIZING WITH LEFO':^66s} ║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  Items:    {len(items):<57d} ║")
    print(f"║  Periods:  {T:<57d} ║")
    print(f"║  Strategy: {'Column Inheritance + σ/τ Dual Guidance':<57s} ║")
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
        verbose=True,
        use_mip_pricing=use_mip_pricing,
    )

    if not math.isfinite(root_lb):
        print("\n✗ Root infeasible!")
        summary = {
            "status": int(GRB.INFEASIBLE),
            "objective": None,
            "best_bound": None,
            "gap": None,
            "runtime_sec": time.time() - start_time,
            "solver_version": "branch_and_price_fixed",
            "n_items": len(items),
            "T": T,
        }
        return summary, []

    root.lp_bound = root_lb
    print(f"  Root LB:  {root_lb:.4f}")
    print(f"  Integer?  {is_integer(z_vals, y_vals, eps)}")
    print(f"  Uses dummy? {solution_uses_dummy(rmp, items, eps)}")

    best_lb = root_lb
    best_ub: Optional[float] = None
    node_counter = 1
    seen_signatures = {node_signature(root)}
    queue = deque([(root, z_vals, y_vals, x_vals, rmp)])
    stats.nodes_created = 1

    # Check if root is already a valid integer solution
    if is_valid_integer_solution(z_vals, y_vals, rmp, items, eps):
        best_ub = root_lb
        best_lb = root_lb
        root.is_integer = True
        stats.nodes_explored = 1
        stats.nodes_integer = 1
        print("\n✓ Root is INTEGER - OPTIMAL!")
        stats.print_summary(best_lb, best_ub, eps)

        orders_txt = generate_orders_txt(items, x_vals, eps)

        summary = {
            "status": int(GRB.OPTIMAL),
            "objective": float(best_ub),
            "best_bound": float(best_lb),
            "gap": 0.0,
            "runtime_sec": time.time() - start_time,
            "solver_version": "branch_and_price_fixed",
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
    print("DEPTH-FIRST SEARCH")
    print(f"{'=' * 70}\n")

    opt_node = root
    opt_z = z_vals
    opt_y = y_vals
    opt_x = x_vals

    while queue and stats.nodes_explored < max_nodes:
        if time.time() - start_time > max_time:
            print("\n⏱ Time limit reached")
            break

        node, parent_z, parent_y, parent_x, parent_rmp = queue.pop()

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
                    use_mip_pricing=use_mip_pricing,
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
                best_lb = min(lb, min(n.lp_bound for n, _, _, _, _ in queue))
            else:
                best_lb = lb

            if best_ub is not None and lb >= best_ub - eps:
                print(f"  FATHOMED: {lb:.2f} >= {best_ub:.2f}")
                node.is_pruned = True
                node.prune_reason = "bound"
                stats.nodes_fathomed_by_bound += 1
                continue

            # Check if it's a valid integer solution (integral AND not using dummy)
            if is_valid_integer_solution(z_vals, y_vals, rmp, items, eps):
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
                    best_lb = min(n.lp_bound for n, _, _, _, _ in queue)
                else:
                    best_lb = best_ub if best_ub is not None else lb

                continue

            # Check if solution only uses dummy (effectively infeasible)
            if solution_uses_dummy(rmp, items, eps) and is_integer(z_vals, y_vals, eps):
                print(f"  FATHOMED: Dummy-only solution")
                node.is_pruned = True
                node.prune_reason = "dummy_infeasible"
                stats.nodes_fathomed_by_infeasible += 1
                continue

            parent_z = z_vals
            parent_y = y_vals
            parent_x = x_vals
            parent_rmp = rmp
        else:
            z_vals = parent_z
            y_vals = parent_y
            x_vals = parent_x

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
            )
            left.theta_0_by_item[item_id].add((t_br, u_br))
            left.lp_bound = node.lp_bound

            sig_left = node_signature(left)
            if sig_left not in seen_signatures:
                seen_signatures.add(sig_left)
                node_counter += 1
                stats.nodes_created += 1
                queue.append((left, z_vals, y_vals, x_vals, parent_rmp))

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
            )
            right.theta_1_by_item[item_id].add((t_br, u_br))
            right.lp_bound = node.lp_bound

            sig_right = node_signature(right)
            if sig_right not in seen_signatures:
                seen_signatures.add(sig_right)
                node_counter += 1
                stats.nodes_created += 1
                queue.append((right, z_vals, y_vals, x_vals, parent_rmp))

            print(f"  Branch Z[{item_id},{t_br},{u_br}]={z_val:.3f}")
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
            )
            left.upsilon_0_by_item[item_id].add(t_br)
            left.lp_bound = node.lp_bound

            sig_left = node_signature(left)
            if sig_left not in seen_signatures:
                seen_signatures.add(sig_left)
                node_counter += 1
                stats.nodes_created += 1
                queue.append((left, z_vals, y_vals, x_vals, parent_rmp))

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
            )
            right.upsilon_1_by_item[item_id].add(t_br)
            right.lp_bound = node.lp_bound

            sig_right = node_signature(right)
            if sig_right not in seen_signatures:
                seen_signatures.add(sig_right)
                node_counter += 1
                stats.nodes_created += 1
                queue.append((right, z_vals, y_vals, x_vals, parent_rmp))

            print(f"  Branch Y[{item_id},{t_br}]={y_val:.3f}")
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
        "solver_version": "branch_and_price_fixed",
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
