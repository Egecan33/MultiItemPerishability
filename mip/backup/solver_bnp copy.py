#!/usr/bin/env python3
"""
Branch-and-Price for Perishable Lot-Sizing with Heterogeneous Shelf Lives and LEFO
This implementation follows the Dantzig-Wolfe decomposition described in the paper:
- Master Problem: Selects convex combinations of item-level production plans
- Pricing Subproblem: Solved via Gurobi (LP relaxation with MIP constraints)
- Branching: First on Z (arc) variables, then Y (setup) variables
- LEFO: Last-Expired-First-Out consumption policy enforced via no-crossing constraints
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


# ============================================================================
# DATA STRUCTURES
# ============================================================================
@dataclass
class ProductionPlanColumn:
    """Represents a single production plan (column) for one item."""

    item_id: int
    total_plan_cost: float
    capacity_usage_by_period: List[float]  # X_it: production quantity in period t
    setup_by_period: List[float]  # Y_it: setup indicator for period t
    arc_usage: Dict[Tuple[int, int], float]  # Z_{t,u}: arc activation indicators


@dataclass
class BranchNode:
    """Represents a node in the branch-and-bound tree."""

    node_id: int
    parent_id: Optional[int]
    depth: int
    # Branching decisions: theta_0 = forbidden arcs, theta_1 = forced arcs
    theta_0_by_item: Dict[int, Set[Tuple[int, int]]] = field(default_factory=dict)
    theta_1_by_item: Dict[int, Set[Tuple[int, int]]] = field(default_factory=dict)
    # upsilon_0 = forbidden setups, upsilon_1 = forced setups
    upsilon_0_by_item: Dict[int, Set[int]] = field(default_factory=dict)
    upsilon_1_by_item: Dict[int, Set[int]] = field(default_factory=dict)
    # Node state
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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
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


# ============================================================================
# PRICING SUBPROBLEM (Solved via Gurobi LP)
# ============================================================================
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
    eps: float = 1e-9,
    use_mip: bool = False,
) -> Tuple[float, Optional[ProductionPlanColumn]]:
    """
    Solve the pricing subproblem for a single item.

    This is a single-item lot-sizing problem with:
    - Demand satisfaction constraints
    - Setup-linking constraints
    - Arc activation constraints
    - LEFO no-crossing constraints
    - Branching constraints (theta_0/theta_1 for arcs, upsilon_0/upsilon_1 for setups)

    Args:
        item_id: Item identifier
        item_data: Item parameters (demand, costs, shelf_seq)
        T: Number of periods
        Gamma: Feasible consumption periods for each production period
        Expiry: Expiry period for each production period
        capacity_duals: Dual prices π_t for capacity constraints
        convexity_dual: Dual price μ_i for convexity constraint
        theta_0: Set of forbidden arcs (t, u)
        theta_1: Set of forced arcs (t, u)
        upsilon_0: Set of forbidden setup periods
        upsilon_1: Set of forced setup periods
        eps: Tolerance for reduced cost
        use_mip: If True, solve as MIP; if False, solve as LP relaxation

    Returns:
        (reduced_cost, column) where column is None if no improving column found
    """
    demand = item_data["demand"]

    # Extract cost parameters
    c_var = item_data["c_var"]
    h = item_data["h"]
    setup = item_data["setup"]

    def c_at(t: int) -> float:
        return float(c_var[t]) if isinstance(c_var, list) else float(c_var)

    def h_at(t: int) -> float:
        return float(h[t]) if isinstance(h, list) else float(h)

    def s_at(t: int) -> float:
        return float(setup[t]) if isinstance(setup, list) else float(setup)

    # Precompute holding cost prefix sums for efficient h_sum calculation
    h_prefix = [0.0] * (T + 1)
    for k in range(T):
        h_prefix[k + 1] = h_prefix[k] + h_at(k)

    def h_sum(t: int, u: int) -> float:
        """Total holding cost for item produced at t, consumed at u."""
        return h_prefix[u] - h_prefix[t]

    # Build list of feasible arcs for this item
    Triples: List[Tuple[int, int]] = []
    for t in range(T):
        for u in Gamma.get(t, []):
            Triples.append((t, u))

    # Compute μ_t: tight setup-linking upper bound
    mu_t: Dict[int, float] = {}
    for t in range(T):
        mu_t[t] = sum(float(demand[u]) for u in Gamma.get(t, []))

    # Create pricing model
    model = gp.Model(f"pricing_item_{item_id}")
    model.Params.OutputFlag = 0

    # Decision variables
    vtype_z = GRB.BINARY if use_mip else GRB.CONTINUOUS
    vtype_y = GRB.BINARY if use_mip else GRB.CONTINUOUS

    X: Dict[Tuple[int, int], gp.Var] = {}
    Z: Dict[Tuple[int, int], gp.Var] = {}
    for t, u in Triples:
        X[t, u] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"X_{t}_{u}")
        Z[t, u] = model.addVar(lb=0.0, ub=1.0, vtype=vtype_z, name=f"Z_{t}_{u}")

    Y: Dict[int, gp.Var] = {}
    for t in range(T):
        Y[t] = model.addVar(lb=0.0, ub=1.0, vtype=vtype_y, name=f"Y_{t}")

    model.update()

    # (C3) Demand satisfaction: sum_t X_{t,u} = d_u
    for u in range(T):
        if demand[u] <= 0:
            continue
        expr = gp.LinExpr()
        for t in range(u + 1):
            if (t, u) in X:
                expr += X[t, u]
        model.addConstr(expr == float(demand[u]), name=f"demand_{u}")

    # (C2) Setup-linking: sum_u X_{t,u} <= μ_t * Y_t
    for t in range(T):
        if not Gamma.get(t):
            model.addConstr(Y[t] == 0.0, name=f"setup_zero_{t}")
            continue
        expr = gp.LinExpr()
        for u in Gamma[t]:
            if (t, u) in X:
                expr += X[t, u]
        model.addConstr(expr <= mu_t[t] * Y[t], name=f"setupLink_{t}")

    # (C4) Arc activation: X_{t,u} <= d_u * Z_{t,u}
    for t, u in Triples:
        C_u = float(demand[u])
        model.addConstr(X[t, u] <= C_u * Z[t, u], name=f"arc_on_{t}_{u}")

    # (C5) LEFO no-crossing constraints
    # For pairs (t1, t2) where v_{t1} < v_{t2}:
    #   For u' in Gamma[t2], for u in Gamma[t1] with t2 <= u <= u'-1:
    #     Z[t1,u] + Z[t2,u'] <= 1
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

    # Branching constraints: theta_0 (forbid arc), theta_1 (force arc)
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
            # Force on impossible arc -> infeasible
            model.addConstr(0.0 == 1.0, name=f"branch_impossible_{t_force}_{u_force}")

    # Branching constraints: upsilon_0 (forbid setup), upsilon_1 (force setup)
    for t_forb in upsilon_0:
        if t_forb in Y:
            model.addConstr(Y[t_forb] == 0.0, name=f"branch_Y0_{t_forb}")

    for t_force in upsilon_1:
        if t_force in Y:
            model.addConstr(Y[t_force] == 1.0, name=f"branch_Y1_{t_force}")

    # Objective: reduced cost
    # reduced_cost = physical_cost - sum_t π_t * X_t - μ_i
    obj = gp.LinExpr()

    # Physical costs: production + holding + setup
    for t, u in Triples:
        unit_cost = c_at(t) + h_sum(t, u)
        obj += unit_cost * X[t, u]
        # Subtract capacity dual
        obj -= capacity_duals[t] * X[t, u]

    for t in range(T):
        obj += s_at(t) * Y[t]

    # Subtract convexity dual
    obj -= convexity_dual

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        return math.inf, None

    reduced_cost = model.ObjVal
    if reduced_cost >= -eps:
        return reduced_cost, None

    # Reconstruct column
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

    return reduced_cost, column


# ============================================================================
# RESTRICTED MASTER PROBLEM
# ============================================================================
class RestrictedMasterProblem:
    """
    Restricted Master Problem for the Dantzig-Wolfe decomposition.

    Master problem structure:
    - Convexity constraints: sum_k λ^k_i = 1 for each item i
    - Capacity constraints: sum_i sum_k X^k_it * λ^k_i <= κ_t for each period t
    - Y-linking constraints: sum_k Y^k_it * λ^k_i = Y_it (for branching)
    - Z-linking constraints: sum_k Z^k_{itu} * λ^k_i = Z_{itu} (for branching)
    """

    def __init__(
        self,
        items: Dict[int, dict],
        T: int,
        capacity: List[float],
        Gamma_by_item: Dict[int, Dict[int, List[int]]],
    ):
        self.items = items
        self.T = T
        self.capacity = capacity
        self.Gamma_by_item = Gamma_by_item

        self.model = gp.Model("RMP")
        self.model.Params.OutputFlag = 0
        self.model.Params.Method = 1  # Dual Simplex

        # Column storage
        self.columns: Dict[int, List[ProductionPlanColumn]] = {i: [] for i in items}
        self.lambdas: Dict[Tuple[int, int], gp.Var] = {}

        # Constraint expressions
        self.convex_expr: Dict[int, gp.LinExpr] = {i: gp.LinExpr(0.0) for i in items}
        self.cap_expr: List[gp.LinExpr] = [gp.LinExpr(0.0) for _ in range(T)]

        # Constraint references
        self.convex_con: Dict[int, gp.Constr] = {}
        self.cap_con: List[gp.Constr] = []

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

    def add_column(self, col: ProductionPlanColumn):
        """Add a new column to the RMP."""
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

        # Update expressions
        self.convex_expr[item_id] += lam
        for t in range(self.T):
            if col.capacity_usage_by_period[t] != 0.0:
                self.cap_expr[t] += col.capacity_usage_by_period[t] * lam

        self._rebuild()

    def solve(self) -> Tuple[float, Dict[int, float], List[float]]:
        """
        Solve the RMP and return objective value and dual prices.

        Returns:
            (objective, mu, pi) where:
            - mu[i] = convexity dual for item i
            - pi[t] = capacity dual for period t
        """
        self.model.optimize()

        if self.model.status != GRB.OPTIMAL:
            return math.inf, {}, []

        mu = {i: self.convex_con[i].Pi for i in self.items}
        pi = [self.cap_con[t].Pi for t in range(self.T)]

        return self.model.ObjVal, mu, pi


# ============================================================================
# COLUMN GENERATION
# ============================================================================
def solve_node_with_column_generation(
    items: Dict[int, dict],
    T: int,
    capacity: List[float],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    Expiry_by_item: Dict[int, Dict[int, int]],
    node: BranchNode,
    max_iter: int = 500,
    eps: float = 1e-9,
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
    """
    Solve a branch node using column generation.

    Returns:
        (lb, rmp, converged, z_vals, y_vals, x_vals)
    """
    rmp = RestrictedMasterProblem(items, T, capacity, Gamma_by_item)

    # Add dummy (lost-sales) columns for feasibility
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
        print(f"  └─ CG: ", end="", flush=True)

    for iteration in range(1, max_iter + 1):
        lb, mu, pi = rmp.solve()

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
                eps=eps,
                use_mip=use_mip_pricing,
            )

            if col is not None and rc < -eps:
                rmp.add_column(col)
                any_added = True

        if not any_added:
            if verbose:
                print(f"LB={lb:.2f} (iter={iteration})")
            z_vals = extract_z_values(rmp, items, eps)
            y_vals = extract_y_values(rmp, items, eps)
            x_vals = extract_x_values(rmp, items, eps)
            return lb, rmp, True, z_vals, y_vals, x_vals

    lb, _, _ = rmp.solve()
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
    """Extract aggregated Z values from RMP solution."""
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
    """Extract aggregated Y values from RMP solution."""
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
    """Extract aggregated production quantities X by period."""
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


# ============================================================================
# BRANCHING
# ============================================================================
def is_integer(
    z_vals: Dict[int, Dict[Tuple[int, int], float]],
    y_vals: Dict[int, Dict[int, float]],
    eps: float = 1e-6,
) -> bool:
    """Check if the solution is integer in Z and Y."""
    for item_arcs in z_vals.values():
        for val in item_arcs.values():
            if eps < val < 1.0 - eps:
                return False

    for item_setups in y_vals.values():
        for val in item_setups.values():
            if eps < val < 1.0 - eps:
                return False

    return True


def find_most_fractional_z(
    z_vals: Dict[int, Dict[Tuple[int, int], float]],
    node: BranchNode,
    eps: float = 1e-9,
) -> Optional[Tuple[int, int, int, float]]:
    """Find the most fractional Z variable for branching."""
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
    eps: float = 1e-9,
) -> Optional[Tuple[int, int, float]]:
    """Find the most fractional Y variable for branching."""
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
    """Remove nodes from queue that cannot improve on incumbent."""
    original_size = len(queue)
    new_queue = deque()

    for node, z_vals, y_vals, x_vals in queue:
        if node.lp_bound < incumbent - eps:
            new_queue.append((node, z_vals, y_vals, x_vals))

    num_fathomed = original_size - len(new_queue)
    queue.clear()
    queue.extend(new_queue)

    return num_fathomed


# ============================================================================
# MAIN SOLVER FUNCTION (MATCHING MIP SOLVER I/O STRUCTURE)
# ============================================================================
def solve_instance(
    instance_path: str | Path = "last_instance.json",
    time_limit: int = 0,
    mip_gap: float = 0.0,
    out_dir: str | Path = "bnp_results",
) -> Tuple[Dict, List[str]]:
    """
    Solve the perishable lot-sizing problem using Branch-and-Price.

    Args:
        instance_path: Path to the instance JSON file
        time_limit: Maximum time in seconds (0 = no limit)
        mip_gap: Optimality gap tolerance (converted to absolute tolerance)
        out_dir: Output directory for results

    Returns:
        (summary, orders_txt) where:
        - summary: Dictionary with results
        - orders_txt: List of strings describing the solution
    """
    start_time = time.time()

    # Load instance
    data = json.loads(Path(instance_path).read_text())
    T = int(data["period"])
    Periods = list(range(T))
    items: Dict[int, dict] = {int(k): v for k, v in data["items"].items()}

    # Capacity
    prod_cap = data.get("production_capacity") or data.get("manual_capacity")
    capacity = (
        _as_len_T_vector(prod_cap, T)
        if prod_cap is not None
        else _cap_global_from_dem(items, T)
    )

    # Convert mip_gap to absolute tolerance (eps)
    eps = max(1e-6, float(mip_gap)) if mip_gap > 0 else 1e-6

    # Set time limit (default to 600 if not specified)
    max_time = int(time_limit) if time_limit > 0 else 600
    max_nodes = 1000000  # Large default
    print_frequency = 50
    use_mip_pricing = True

    print("\n" + "╔" + "═" * 68 + "╗")
    print(f"║ {'BRANCH-AND-PRICE: PERISHABLE LOT-SIZING WITH LEFO':^66s} ║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  Items:    {len(items):<57d} ║")
    print(f"║  Periods:  {T:<57d} ║")
    print(f"║  Pricing:  {'MIP' if use_mip_pricing else 'LP':<57s} ║")
    print("╚" + "═" * 68 + "╝")

    # Build Gamma and Expiry for each item
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
            v_it = t + m_it  # Expiry period (exclusive)
            Expiry[t] = v_it

            if m_it <= 0:
                Gamma[t] = []
                continue

            u_max = min(T - 1, v_it - 1)  # Last feasible consumption period
            Gamma[t] = list(range(t, u_max + 1))

        Gamma_by_item[item_id] = Gamma
        Expiry_by_item[item_id] = Expiry

    # Initialize statistics
    stats = SearchStatistics()
    stats.start_time = start_time

    # Create root node
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
            "solver_version": "branch_and_price_v1",
            "n_items": len(items),
            "T": T,
        }
        return summary, []

    root.lp_bound = root_lb
    print(f"  Root LB:  {root_lb:.4f}")
    print(f"  Integer?  {is_integer(z_vals, y_vals, eps)}")

    best_lb = root_lb
    best_ub: Optional[float] = None
    node_counter = 1
    seen_signatures = {node_signature(root)}
    queue = deque([(root, z_vals, y_vals, x_vals)])
    stats.nodes_created = 1

    if is_integer(z_vals, y_vals, eps):
        best_ub = root_lb
        best_lb = root_lb
        root.is_integer = True
        stats.nodes_explored = 1
        stats.nodes_integer = 1
        print("\n✓ Root is INTEGER - OPTIMAL!")
        stats.print_summary(best_lb, best_ub, eps)

        # Generate orders_txt
        orders_txt = generate_orders_txt(items, x_vals, eps)

        summary = {
            "status": int(GRB.OPTIMAL),
            "objective": float(best_ub),
            "best_bound": float(best_lb),
            "gap": 0.0,
            "runtime_sec": time.time() - start_time,
            "solver_version": "branch_and_price_v1",
            "n_items": len(items),
            "T": T,
        }

        # Save to output directory
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

        # DFS: pop from end of queue
        node, parent_z, parent_y, parent_x = queue.pop()

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

            # Update lower bound
            if queue:
                best_lb = min(lb, min(n.lp_bound for n, _, _, _ in queue))
            else:
                best_lb = lb

            if best_ub is not None and lb >= best_ub - eps:
                print(f"  FATHOMED: {lb:.2f} >= {best_ub:.2f}")
                node.is_pruned = True
                node.prune_reason = "bound"
                stats.nodes_fathomed_by_bound += 1
                continue

            if is_integer(z_vals, y_vals, eps):
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
                    best_lb = min(n.lp_bound for n, _, _, _ in queue)
                else:
                    best_lb = best_ub if best_ub is not None else lb

                continue

            parent_z = z_vals
            parent_y = y_vals
            parent_x = x_vals
        else:
            z_vals = parent_z
            y_vals = parent_y
            x_vals = parent_x

        # Branching: first on Z (arcs), then on Y (setups)
        branch_var_z = find_most_fractional_z(z_vals, node, eps)
        if branch_var_z is not None:
            item_id, t_br, u_br, z_val = branch_var_z

            # Create Z=0 child
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
                queue.append((left, z_vals, y_vals, x_vals))

            # Create Z=1 child
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
                queue.append((right, z_vals, y_vals, x_vals))

            print(f"  Branch Z[{item_id},{t_br},{u_br}]={z_val:.3f}")
            continue

        # No fractional Z, try Y
        branch_var_y = find_most_fractional_y(y_vals, node, eps)
        if branch_var_y is not None:
            item_id, t_br, y_val = branch_var_y

            # Create Y=0 child
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
                queue.append((left, z_vals, y_vals, x_vals))

            # Create Y=1 child
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
                queue.append((right, z_vals, y_vals, x_vals))

            print(f"  Branch Y[{item_id},{t_br}]={y_val:.3f}")
            continue

        print("  No fractional variable - solution is integral")

    # Final bounds
    if best_ub is not None:
        best_lb = best_ub

    stats.print_summary(best_lb, best_ub, eps)

    # Generate orders_txt
    orders_txt = generate_orders_txt(items, opt_x, eps)

    # Create summary
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
        "solver_version": "branch_and_price_v1",
        "n_items": len(items),
        "T": T,
    }

    # Save to output directory
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
    """
    Generate orders text output in the same format as the MIP solver.

    Args:
        items: Item data dictionary
        x_vals: Production quantities X[i,t] by item and period
        eps: Tolerance for numerical comparisons

    Returns:
        List of strings describing production orders
    """
    orders_txt: List[str] = []

    for item_id in sorted(items.keys()):
        orders_txt.append(f"Item {item_id} — orders (t → qty)")

        # Get production quantities for this item
        production = x_vals.get(item_id, {})

        for t in sorted(production.keys()):
            qty = production[t]
            if qty > eps:
                orders_txt.append(f" {t:2d} → {qty:8.3f}")

        orders_txt.append("")

    return orders_txt
