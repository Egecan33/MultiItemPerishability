#!/usr/bin/env python3
# Branch-and-Price with MIP-based Column Generation (block-based SILSP pricing)

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import math
import gurobipy as gurobi
from gurobipy import GRB
from collections import deque
import time


# Data classes


@dataclass
class ProductionItem:
    item_id: int
    number_of_periods: int
    demand_quantity_by_period: List[float]
    production_unit_cost_by_period: List[float]
    setup_cost_by_period: List[float]
    holding_unit_cost_by_period: List[float]
    # NOTE: In this code, we treat this as ABSOLUTE EXPIRY INDEX E_t = t + L_t (inclusive).
    perishability_horizon_by_start_period: List[int]
    lost_sales_penalty_per_unit: float


@dataclass
class ProductionPlanColumn:
    item_id: int
    total_plan_cost: float
    capacity_usage_by_period: List[float]
    setup_open_fraction_by_period: Optional[List[float]] = None
    lost_demand_quantity_by_period: Optional[List[float]] = None
    arc_usage: Optional[Dict[Tuple[int, int], float]] = None


@dataclass
class BranchNode:
    node_id: int
    parent_id: Optional[int]
    depth: int
    theta_0_by_item: Dict[int, Set[Tuple[int, int]]] = field(
        default_factory=dict
    )  # forbid arcs
    theta_1_by_item: Dict[int, Set[Tuple[int, int]]] = field(
        default_factory=dict
    )  # force arcs
    lp_bound: float = math.inf
    is_integer: bool = False
    is_pruned: bool = False
    prune_reason: Optional[str] = None
    branch_variable: Optional[Tuple[int, int, int, float]] = None
    branch_direction: Optional[str] = None


class SearchStatistics:
    def __init__(self):
        self.nodes_created = 0
        self.nodes_explored = 0
        self.nodes_integer = 0
        self.nodes_fathomed_by_bound = 0
        self.nodes_fathomed_by_infeasible = 0
        self.nodes_fathomed_integer = 0
        self.nodes_fathomed_on_incumbent = 0
        self.nodes_fathomed_duplicate = 0
        self.max_depth = 0
        self.incumbent_history = []
        self.start_time = time.time()

    def node_created(self):
        self.nodes_created += 1

    def node_explored(
        self, node: BranchNode, incumbent_improved: bool = False, num_fathomed: int = 0
    ):
        self.nodes_explored += 1
        self.max_depth = max(self.max_depth, node.depth)
        if node.is_integer:
            self.nodes_integer += 1
            self.nodes_fathomed_integer += 1
            if incumbent_improved:
                self.incumbent_history.append((node.node_id, node.lp_bound))
                self.nodes_fathomed_on_incumbent += num_fathomed
        if node.is_pruned:
            if node.prune_reason == "bound":
                self.nodes_fathomed_by_bound += 1
            elif node.prune_reason == "infeasible":
                self.nodes_fathomed_by_infeasible += 1
            elif node.prune_reason == "duplicate":
                self.nodes_fathomed_duplicate += 1

    def print_summary(self, best_lb: float, best_ub: Optional[float], eps: float):
        elapsed = time.time() - self.start_time
        total_fathomed = (
            self.nodes_fathomed_by_bound
            + self.nodes_fathomed_by_infeasible
            + self.nodes_fathomed_integer
            + self.nodes_fathomed_duplicate
        )
        print("\n" + "=" * 70)
        print(" " * 28 + "FINAL RESULTS")
        print("=" * 70)
        print(f"  Time elapsed:       {elapsed:.2f} seconds")
        print(f"  Nodes created:      {self.nodes_created}")
        print(f"  Nodes explored:     {self.nodes_explored}")
        print(f"  Integer solutions:  {self.nodes_integer}")
        print(f"  Max depth:          {self.max_depth}")
        print()
        print(f"  Fathomed nodes:     {total_fathomed}")
        print(f"    By bound:         {self.nodes_fathomed_by_bound}")
        print(f"    Infeasible:       {self.nodes_fathomed_by_infeasible}")
        print(f"    Integer:          {self.nodes_fathomed_integer}")
        print(f"    On incumbent:     {self.nodes_fathomed_on_incumbent}")
        print(f"    Duplicate:        {self.nodes_fathomed_duplicate}")
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


# Helpers


def node_signature(node: BranchNode) -> str:
    sig_parts = []
    for item_id in sorted(node.theta_0_by_item.keys()):
        arcs = sorted(node.theta_0_by_item[item_id])
        if arcs:
            sig_parts.append(f"I{item_id}_0:{','.join(f'{t}-{u}' for t, u in arcs)}")
    for item_id in sorted(node.theta_1_by_item.keys()):
        arcs = sorted(node.theta_1_by_item[item_id])
        if arcs:
            sig_parts.append(f"I{item_id}_1:{','.join(f'{t}-{u}' for t, u in arcs)}")
    return "|".join(sig_parts)


# MIP-based pricing for a single item


def mip_pricing_for_single_item(
    production_item: ProductionItem,
    capacity_dual_prices: List[float],
    convexity_dual_price: float,
    theta_0: Set[Tuple[int, int]],
    theta_1: Set[Tuple[int, int]],
    arc_dual_prices: Optional[Dict[Tuple[int, int], float]] = None,
    epsilon_tolerance: float = 1e-9,
) -> Tuple[float, Optional[ProductionPlanColumn]]:
    """
    MIP-based pricing for a single item, with explicit LEFO C5 constraints.

    This is a per-item analogue of the global MIP:
      - Variables: x_{t,u}, y_t, z_{t,u}
      - Constraints:
          * demand satisfaction (no lost sales here),
          * setup-linking,
          * arc activation,
          * LEFO no-crossing (C5) on z,
          * branching constraints θ_0 / θ_1 on arcs.
      - Objective: reduced cost
            physical_cost - sum_t π_t * prod_t - μ_i  (and - τ-part if used).
    """

    T = production_item.number_of_periods
    demand = production_item.demand_quantity_by_period
    setup_cost = production_item.setup_cost_by_period
    prod_cost = production_item.production_unit_cost_by_period
    holding_cost = production_item.holding_unit_cost_by_period
    expiry_abs = (
        production_item.perishability_horizon_by_start_period
    )  # E_t (inclusive)

    assert len(demand) == T
    assert len(setup_cost) == T
    assert len(prod_cost) == T
    assert len(holding_cost) == T
    assert len(expiry_abs) == T

    INF = 1e10

    # ---- Build Gamma and Expiry (single item) ----
    # Gamma[t] = {u : t <= u <= min(T-1, E_t), d_u > 0}
    Gamma: Dict[int, List[int]] = {}
    Expiry: Dict[int, int] = {}
    Triples: List[Tuple[int, int]] = []  # (t,u) arcs

    for t in range(T):
        E_t = int(expiry_abs[t])
        Expiry[t] = E_t
        if E_t <= t:
            Gamma[t] = []
            continue
        u_max = min(T - 1, E_t)  # E_t inclusive horizon
        us = [u for u in range(t, u_max + 1) if demand[u] > 0.0]
        Gamma[t] = us
        for u in us:
            Triples.append((t, u))

    # Tight setup-linking upper bounds μ_t = sum_{u in Gamma(t)} d_u
    mu_t: Dict[int, float] = {}
    for t in range(T):
        mu_t[t] = float(sum(demand[u] for u in Gamma.get(t, [])))

    # ---- Holding-cost prefix sums and helper hsum(t,u) ----
    # Hpref[k] = sum_{τ=0}^{k-1} h_τ
    Hpref = [0.0] * (T + 1)
    for k in range(1, T + 1):
        Hpref[k] = Hpref[k - 1] + float(holding_cost[k - 1])

    def hsum(t: int, u: int) -> float:
        """Total holding cost per unit produced at t and consumed at u (u >= t)."""
        # sum_{τ=t}^{u-1} h_τ
        return float(Hpref[u] - Hpref[t])

    # ---- Build pricing MIP ----
    model = gurobi.Model(f"pricing_item_{production_item.item_id}")
    model.Params.OutputFlag = 0

    # Variables: x_{t,u} >= 0, z_{t,u} ∈ {0,1}, y_t ∈ {0,1}
    x: Dict[Tuple[int, int], gurobi.Var] = {}
    z: Dict[Tuple[int, int], gurobi.Var] = {}
    for t, u in Triples:
        x[t, u] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x_{t}_{u}")
        z[t, u] = model.addVar(vtype=GRB.BINARY, name=f"z_{t}_{u}")

    y: Dict[int, gurobi.Var] = {
        t: model.addVar(vtype=GRB.BINARY, name=f"y_{t}") for t in range(T)
    }

    model.update()

    # ---- (C3) Demand satisfaction: sum_t x_{t,u} = d_u (no lost sales here) ----
    for u in range(T):
        if demand[u] <= 0.0:
            continue
        expr = gurobi.LinExpr()
        for t in range(u + 1):  # t <= u
            if (t, u) in x:
                expr += x[t, u]
        model.addConstr(expr == float(demand[u]), name=f"dem_{u}")

    # ---- (C2) Setup-linking: sum_u x_{t,u} ≤ μ_t * y_t ----
    for t in range(T):
        if not Gamma.get(t):
            # No feasible arcs → no setup
            model.addConstr(y[t] == 0.0, name=f"setup_zero_{t}")
            continue
        expr = gurobi.LinExpr()
        for u in Gamma[t]:
            expr += x[t, u]
        model.addConstr(expr <= mu_t[t] * y[t], name=f"setupLink_{t}")

    # ---- (C4) Arc activation: x_{t,u} ≤ d_u * z_{t,u} ----
    for t, u in Triples:
        C_u = float(demand[u])
        model.addConstr(x[t, u] <= C_u * z[t, u], name=f"arc_on_{t}_{u}")

    # ---- Branching constraints: θ_0 (forbid), θ_1 (force) on this item ----
    # θ_0, θ_1 are (t,u) sets for THIS item_id.
    for t_forb, u_forb in theta_0:
        if (t_forb, u_forb) in z:
            model.addConstr(
                z[t_forb, u_forb] == 0.0,
                name=f"branch_0_{t_forb}_{u_forb}",
            )
        # else: forbidden arc is already impossible because it's not in Gamma

    for t_force, u_force in theta_1:
        if (t_force, u_force) in z:
            model.addConstr(
                z[t_force, u_force] == 1.0,
                name=f"branch_1_{t_force}_{u_force}",
            )
        else:
            # Branch forces an impossible arc → infeasible pricing at this node
            model.addConstr(0.0 == 1.0, name=f"branch_impossible_{t_force}_{u_force}")

    # ---- (C5) LEFO no-crossing constraints on z (EXACT pattern) ----
    # This mirrors the big MIP logic, specialized to a single item:
    #
    # For any t1,t2 with Expiry[t1] < Expiry[t2]:
    #   For any up ∈ Gamma[t2]:
    #       For any u ∈ Gamma[t1] with t2 ≤ u ≤ up-1:
    #           z[t1,u] + z[t2,up] ≤ 1
    #
    prods = [t for t in range(T) if Gamma.get(t)]
    prods.sort(key=lambda t: Expiry[t])  # ascending by expiry E_t

    for idx1 in range(len(prods)):
        t1 = prods[idx1]
        v1 = Expiry[t1]
        for idx2 in range(idx1 + 1, len(prods)):
            t2 = prods[idx2]
            v2 = Expiry[t2]
            # only pairs where t1 has EARLIER expiry than t2
            if v1 >= v2:
                continue
            if not Gamma.get(t1) or not Gamma.get(t2):
                continue

            for up in Gamma[t2]:  # u' for t2
                # u for t1 in [t2, up-1]
                for u in [uu for uu in Gamma[t1] if t2 <= uu <= up - 1]:
                    if (t1, u) in z and (t2, up) in z:
                        model.addConstr(
                            z[t1, u] + z[t2, up] <= 1,
                            name=f"nocross_{t1}_{t2}_{u}_{up}",
                        )

    # ---- Objective: reduced cost ----
    obj = gurobi.LinExpr()

    # Physical cost: production + holding + setup
    for t, u in Triples:
        unit_phys = float(prod_cost[t]) + hsum(t, u)
        # Add physical cost
        obj += unit_phys * x[t, u]
        # Subtract capacity dual π_t * quantity produced at t
        obj += -float(capacity_dual_prices[t]) * x[t, u]
        # Optional arc dual τ_{t,u}
        if arc_dual_prices and (t, u) in arc_dual_prices:
            obj += -float(arc_dual_prices[(t, u)]) * x[t, u]

    for t in range(T):
        obj += float(setup_cost[t]) * y[t]

    # Subtract convexity dual μ_i
    obj += -float(convexity_dual_price)

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        return INF, None

    rc = model.ObjVal
    if rc >= -epsilon_tolerance:
        # No improving column
        return rc, None

    # ---- Reconstruct a column (physical plan) ----
    cap_usage = [0.0] * T
    setup_usage = [0.0] * T
    arc_usage: Dict[Tuple[int, int], float] = {}
    total_cost = 0.0  # physical cost (not reduced)

    # Production + holding cost
    for t, u in Triples:
        x_val = x[t, u].X
        if x_val <= 1e-9:
            continue
        cap_usage[t] += x_val
        unit_phys = float(prod_cost[t]) + hsum(t, u)
        total_cost += unit_phys * x_val
        # For branching we care about which arcs carry flow
        arc_usage[(t, u)] = 1.0

    # Setup cost
    for t in range(T):
        if y[t].X > 0.5:
            setup_usage[t] = 1.0
            total_cost += float(setup_cost[t])

    col = ProductionPlanColumn(
        item_id=production_item.item_id,
        total_plan_cost=total_cost,
        capacity_usage_by_period=cap_usage,
        setup_open_fraction_by_period=setup_usage,
        lost_demand_quantity_by_period=[0.0] * T,
        arc_usage=arc_usage,
    )

    return rc, col


# Restricted Master Problem


class RestrictedMasterProblem:
    def __init__(
        self,
        items: List[ProductionItem],
        capacity: List[float],
        arc_rows_spec: Optional[Dict[int, Dict[Tuple[int, int], float]]] = None,
    ):
        self.items = items
        self.T = len(capacity)
        self.capacity = capacity
        self.model = gurobi.Model("RMP")
        self.model.Params.OutputFlag = 0
        self.model.Params.Method = 1  # Dual Simplex

        self.columns: Dict[int, List[ProductionPlanColumn]] = {
            i.item_id: [] for i in items
        }
        self.lambdas: Dict[Tuple[int, int], gurobi.Var] = {}
        self.convex_expr = {i.item_id: gurobi.LinExpr(0.0) for i in items}
        self.cap_expr = [gurobi.LinExpr(0.0) for _ in range(self.T)]
        self.convex_con: Dict[int, gurobi.Constr] = {}
        self.cap_con: List[gurobi.Constr] = []

        for item in items:
            self.convex_con[item.item_id] = self.model.addConstr(
                self.convex_expr[item.item_id] == 1.0, name=f"conv_{item.item_id}"
            )
        for t in range(self.T):
            self.cap_con.append(
                self.model.addConstr(
                    self.cap_expr[t] <= self.capacity[t], name=f"cap_{t}"
                )
            )

    def _rebuild(self):
        # Re-attach constraints to updated expressions
        for item in self.items:
            self.model.remove(self.convex_con[item.item_id])
            self.convex_con[item.item_id] = self.model.addConstr(
                self.convex_expr[item.item_id] == 1.0, name=f"conv_{item.item_id}"
            )
        for t in range(self.T):
            self.model.remove(self.cap_con[t])
            self.cap_con[t] = self.model.addConstr(
                self.cap_expr[t] <= self.capacity[t], name=f"cap_{t}"
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

        self._rebuild()

    def solve(self):
        self.model.optimize()
        if self.model.status != GRB.OPTIMAL:
            return math.inf, {}, []
        mu = {i.item_id: self.convex_con[i.item_id].Pi for i in self.items}
        pi = [self.cap_con[t].Pi for t in range(self.T)]
        return self.model.ObjVal, mu, pi


# Column generation at a node (uses MIP pricing now)


def solve_node_with_column_generation(
    items: List[ProductionItem],
    capacity: List[float],
    node: BranchNode,
    max_iter: int = 300,
    eps: float = 1e-9,
    verbose: bool = False,
):
    rmp = RestrictedMasterProblem(items, capacity)

    # Add high-cost dummy columns (lost sales) for feasibility
    for item in items:
        T = item.number_of_periods
        total_demand = sum(item.demand_quantity_by_period)
        dummy_cost = 10000.0 * total_demand

        col = ProductionPlanColumn(
            item_id=item.item_id,
            total_plan_cost=dummy_cost,
            capacity_usage_by_period=[0.0] * T,
            setup_open_fraction_by_period=[0.0] * T,
            lost_demand_quantity_by_period=item.demand_quantity_by_period[:],
            arc_usage={},
        )
        rmp.add_column(col)

    if verbose:
        print(f"  └─ CG: ", end="", flush=True)

    for _ in range(1, max_iter + 1):
        lb, mu, pi = rmp.solve()
        if not math.isfinite(lb):
            if verbose:
                print("INFEASIBLE")
            return math.inf, rmp, False, {}

        any_added = False
        for item in items:
            theta_0 = node.theta_0_by_item.get(item.item_id, set())
            theta_1 = node.theta_1_by_item.get(item.item_id, set())

            # MIP-based pricing instead of DP
            rc, col = mip_pricing_for_single_item(
                item,
                capacity_dual_prices=pi,
                convexity_dual_price=mu[item.item_id],
                theta_0=theta_0,
                theta_1=theta_1,
                arc_dual_prices=None,  # no τ rows yet
                epsilon_tolerance=eps,
            )

            if col is not None and rc < -eps:
                rmp.add_column(col)
                any_added = True

        if not any_added:
            if verbose:
                print(f"LB={lb:.2f}")
            z_vals = extract_z_values(rmp, items, eps)
            return lb, rmp, True, z_vals

    lb, _, _ = rmp.solve()
    z_vals = extract_z_values(rmp, items, eps)
    if verbose:
        print(f"LB={lb:.2f} (max iter)")
    return lb, rmp, False, z_vals


# Z-value extraction, branching, and B&P driver


def extract_z_values(
    rmp: RestrictedMasterProblem, items: List[ProductionItem], eps: float
):
    z = {i.item_id: {} for i in items}
    for (item_id, idx), lam in rmp.lambdas.items():
        lam_val = lam.X
        if lam_val < eps:
            continue
        col = rmp.columns[item_id][idx]
        if col.arc_usage:
            for t, u in col.arc_usage:
                z[item_id][(t, u)] = z[item_id].get((t, u), 0.0) + lam_val
    return z


def is_integer(z_vals, eps=1e-6):
    for arcs in z_vals.values():
        for val in arcs.values():
            if eps < val < 1.0 - eps:
                return False
    return True


def find_most_fractional(z_vals, node: BranchNode, eps=1e-9):
    best_frac = 0.0
    best = None
    for item_id, arcs in z_vals.items():
        for (t, u), val in arcs.items():
            if (t, u) in node.theta_0_by_item.get(item_id, set()):
                continue
            if (t, u) in node.theta_1_by_item.get(item_id, set()):
                continue
            frac = min(val, 1.0 - val)
            if frac > eps and frac > best_frac:
                best_frac = frac
                best = (item_id, t, u, val)
    return best


def fathom_queue_by_incumbent(queue: deque, incumbent: float, eps: float) -> int:
    original_size = len(queue)
    new_queue = deque()
    for node, z_vals in queue:
        if node.lp_bound < incumbent - eps:
            new_queue.append((node, z_vals))
    num_fathomed = original_size - len(new_queue)
    queue.clear()
    queue.extend(new_queue)
    return num_fathomed


def solve_branch_and_price(
    items, capacity, max_nodes=5000, eps=1e-6, print_frequency=50
):
    print("\n" + "=" * 70)
    print(" " * 20 + "BRANCH-AND-PRICE WITH MIP PRICING (BFS)")
    print("=" * 70)

    stats = SearchStatistics()

    root = BranchNode(
        0, None, 0, {i.item_id: set() for i in items}, {i.item_id: set() for i in items}
    )

    print("\n>>> ROOT NODE <<<")
    root_lb, _, _, z_vals = solve_node_with_column_generation(
        items, capacity, root, verbose=True
    )

    if not math.isfinite(root_lb):
        print("\n✗ Root infeasible!")
        return math.inf, None, 1

    root.lp_bound = root_lb
    print(f"  Root LB:  {root_lb:.4f}")
    print(f"  Integer?  {is_integer(z_vals, eps)}")

    best_lb = root_lb
    best_ub: Optional[float] = None
    nodes_explored = 1
    node_counter = 1

    seen_signatures = {node_signature(root)}
    queue = deque([(root, z_vals)])  # BFS queue
    stats.node_created()

    if is_integer(z_vals, eps):
        best_ub = root_lb
        best_lb = root_lb
        root.is_integer = True
        stats.node_explored(root, incumbent_improved=True)
        print("\n✓ Root is INTEGER - OPTIMAL!")
        stats.print_summary(best_lb, best_ub, eps)
        return root_lb, root_lb, 1

    stats.node_explored(root)

    print(f"\n{'='*70}")
    print("BREADTH-FIRST SEARCH")
    print(f"{'='*70}\n")

    opt_node = root
    opt_z = z_vals
    opt_rmp = None

    while queue and nodes_explored < max_nodes:
        node, parent_z = queue.popleft()

        if node.node_id != 0:
            if nodes_explored % print_frequency == 0:
                gap_str = "N/A"
                if best_ub is not None:
                    gap = best_ub - best_lb
                    gap_pct = 100 * gap / max(abs(best_ub), 1e-10)
                    gap_str = f"{gap_pct:.2f}%"
                print(
                    f"[Progress: N={nodes_explored:4d}, Queue={len(queue):4d}, "
                    f"LB={best_lb:.2f}, UB={best_ub if best_ub is not None else 'N/A'}, Gap={gap_str}]"
                )

            print(f"N{node.node_id:4d} D{node.depth:2d}", end="", flush=True)

            lb, _, _, z_vals = solve_node_with_column_generation(
                items, capacity, node, verbose=True
            )
            nodes_explored += 1
            node.lp_bound = lb

            if not math.isfinite(lb):
                print("  FATHOMED: Infeasible")
                node.is_pruned = True
                node.prune_reason = "infeasible"
                stats.node_explored(node)
                if queue:
                    best_lb = min(min(n.lp_bound for n, _ in queue), best_lb)
                elif best_ub is not None:
                    best_lb = best_ub
                continue

            if queue:
                best_lb = min(lb, min(n.lp_bound for n, _ in queue))
            else:
                best_lb = lb

            if best_ub is not None and lb >= best_ub - eps:
                print(f"  FATHOMED: {lb:.2f} ≥ {best_ub:.2f}")
                node.is_pruned = True
                node.prune_reason = "bound"
                stats.node_explored(node)
                if queue:
                    best_lb = min(n.lp_bound for n, _ in queue)
                elif best_ub is not None:
                    best_lb = best_ub
                continue

            if is_integer(z_vals, eps):
                print(f"  INTEGER: {lb:.2f}", end="")
                node.is_integer = True
                if best_ub is None or lb < best_ub - eps:
                    best_ub = lb
                    print(" ★ NEW INCUMBENT!")
                    num_fathomed = fathom_queue_by_incumbent(queue, best_ub, eps)
                    if queue:
                        best_lb = min(n.lp_bound for n, _ in queue)
                    else:
                        best_lb = best_ub
                    stats.node_explored(
                        node, incumbent_improved=True, num_fathomed=num_fathomed
                    )
                    opt_node = node
                    opt_z = z_vals
                else:
                    print()
                    if queue:
                        best_lb = min(n.lp_bound for n, _ in queue)
                    else:
                        best_lb = best_ub
                    stats.node_explored(node)
                continue

            stats.node_explored(node)
            parent_z = z_vals
        else:
            z_vals = parent_z

        branch_var = find_most_fractional(z_vals, node, eps)
        if branch_var is None:
            print("  No fractional variable")
            node.is_pruned = True
            if queue:
                best_lb = min(n.lp_bound for n, _ in queue)
            elif best_ub is not None:
                best_lb = best_ub
            continue

        item_id, t_br, u_br, z_val = branch_var
        can_be_zero = (t_br, u_br) not in node.theta_1_by_item[item_id]
        can_be_one = (t_br, u_br) not in node.theta_0_by_item[item_id]
        if not can_be_zero and not can_be_one:
            print("  Conflict")
            continue

        children = 0

        # Z=0 branch
        if can_be_zero:
            left = BranchNode(
                node_counter,
                node.node_id,
                node.depth + 1,
                {i: node.theta_0_by_item[i].copy() for i in node.theta_0_by_item},
                {i: node.theta_1_by_item[i].copy() for i in node.theta_1_by_item},
                branch_variable=branch_var,
                branch_direction="Z=0",
            )
            left.theta_0_by_item[item_id].add((t_br, u_br))
            sig = node_signature(left)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                left.lp_bound = node.lp_bound
                node_counter += 1
                queue.append((left, z_vals))
                children += 1

        # Z=1 branch
        if can_be_one:
            right = BranchNode(
                node_counter,
                node.node_id,
                node.depth + 1,
                {i: node.theta_0_by_item[i].copy() for i in node.theta_0_by_item},
                {i: node.theta_1_by_item[i].copy() for i in node.theta_1_by_item},
                branch_variable=branch_var,
                branch_direction="Z=1",
            )
            right.theta_1_by_item[item_id].add((t_br, u_br))
            sig = node_signature(right)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                right.lp_bound = node.lp_bound
                node_counter += 1
                queue.append((right, z_vals))
                children += 1

        if children > 0:
            print(f"  Branch Z[{item_id},{t_br},{u_br}]={z_val:.3f}")

    # After search: try to recover RMP for the incumbent node
    if best_ub is not None:
        best_lb = best_ub
        _, opt_rmp, _, _ = solve_node_with_column_generation(items, capacity, opt_node)
    else:
        opt_rmp = None

    stats.print_summary(best_lb, best_ub, eps)
    return best_lb, best_ub, nodes_explored, opt_node, opt_z, opt_rmp


# Example instance and main


def build_small_example_instance():
    """Build the problematic test case instance from bnp_v10_results/test_instance.json"""
    T = 10

    # Item 0 data from test_instance.json
    demand = [0, 0, 68, 49, 66, 38, 17, 17, 41, 43]
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
        80.0,
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
    shelf_seq = [24, 18, 22, 6, 16, 9, 21, 23, 9, 7]

    # expiry_abs[t] = t + shelf_seq[t] (inclusive)
    expiry_abs = [t + shelf_seq[t] for t in range(T)]

    # Capacity from test_instance.json
    capacity = [0, 0, 80, 80, 80, 80, 80, 80, 100, 80]

    return (
        [
            ProductionItem(
                item_id=0,
                number_of_periods=T,
                demand_quantity_by_period=demand,
                production_unit_cost_by_period=c_var,
                setup_cost_by_period=setup,
                holding_unit_cost_by_period=h,
                perishability_horizon_by_start_period=expiry_abs,
                lost_sales_penalty_per_unit=5000.0,
            ),
        ],
        capacity,
    )


if __name__ == "__main__":
    items, cap = build_small_example_instance()
    print("\n╔" + "═" * 68 + "╗")
    print(f"║ {'CAPACITATED LOT SIZING WITH PERISHABILITY':^66s} ║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  Items:    {len(items):<57d} ║")
    print(f"║  Periods:  {len(cap):<57d} ║")
    print("╚" + "═" * 68 + "╝")

    lb, ub, nodes, opt_node, opt_z, opt_rmp = solve_branch_and_price(
        items, cap, max_nodes=10000, print_frequency=50
    )

    print("\n" + "=" * 70)
    print(" " * 25 + "OPTIMAL NODE ORDER POLICY")
    print("=" * 70)

    for item in items:
        print(f"\nItem {item.item_id}:")
        print("-" * 25)
        arcs = opt_z[item.item_id]
        if not arcs:
            print("  No production (lost sales only).")
            continue
        for (t, u), val in sorted(arcs.items()):
            if val > 0.9:
                print(
                    f"  Produce in period {t+1:>2d} → satisfies demand of period {u+1:>2d}"
                )

    if opt_rmp is not None:
        print("\n" + "=" * 70)
        print(" " * 30 + "FINAL RMP VARIABLE VALUES")
        print("=" * 70)

        for (item_id, idx), lam in opt_rmp.lambdas.items():
            lam_val = lam.X
            if lam_val < 1e-6:
                continue
            col = opt_rmp.columns[item_id][idx]
            print(
                f"\nλ[{item_id},{idx}] = {lam_val:.4f}, cost = {col.total_plan_cost:.2f}"
            )

            x = col.capacity_usage_by_period
            y = col.setup_open_fraction_by_period
            print("  x_it:", ["{:.2f}".format(v) for v in x])
            print("  y_it:", ["{:.2f}".format(v) for v in y])
