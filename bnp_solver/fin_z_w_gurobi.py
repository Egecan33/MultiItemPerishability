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
    MIP-based pricing for the single-item SILSP with ZIO structure.

    - Uses block variables B_{t,e} for 0 <= t <= e < T.
    - Each block (t,e) corresponds to a batch produced at t serving demands t..e.
    - Enforces:
        * shelf life: e <= E_t,
        * branching constraints (θ_0: forbidden arcs, θ_1: forced arcs),
        * coverage: every positive-demand period u is covered by exactly one block.
    - Objective minimizes the reduced cost directly:
        sum_{blocks} B_{t,e} * [setup + unit * Q_{t,e} + H_{t,e} - π_t * Q_{t,e} - τ-part]
        - μ_i.
    """

    T = production_item.number_of_periods
    demand = production_item.demand_quantity_by_period
    setup_cost = production_item.setup_cost_by_period
    prod_cost = production_item.production_unit_cost_by_period
    holding_cost = production_item.holding_unit_cost_by_period
    expiry_abs = (
        production_item.perishability_horizon_by_start_period
    )  # E_t (inclusive)

    INF = 1e10

    # ---- Prefix sums for O(1) block demand and holding cost ----
    # Dpref[k] = sum_{j=0}^{k-1} d_j
    Dpref = [0.0] * (T + 1)
    for k in range(1, T + 1):
        Dpref[k] = Dpref[k - 1] + demand[k - 1]

    # Hpref[k] = sum_{τ=0}^{k-1} h_τ
    Hpref = [0.0] * (T + 1)
    for k in range(1, T + 1):
        Hpref[k] = Hpref[k - 1] + holding_cost[k - 1]

    # DHpref[k] = sum_{j=0}^{k-1} d_j * Hpref[j]
    DHpref = [0.0] * (T + 1)
    for k in range(1, T + 1):
        DHpref[k] = DHpref[k - 1] + demand[k - 1] * Hpref[k - 1]

    def block_demand(t: int, e: int) -> float:
        """Q_{t,e} = sum_{j=t}^e d_j"""
        return Dpref[e + 1] - Dpref[t]

    def block_holding_cost(t: int, e: int) -> float:
        """
        H_{t,e} = sum_{τ=t}^{e-1} h_τ * sum_{j=τ+1}^e d_j
        Implemented via prefix sums in O(1), using u_excl = e+1:
        sum_{τ=t}^{u_excl-2} h_τ * sum_{j=τ+1}^{u_excl-1} d_j
        """
        u_excl = e + 1
        if u_excl <= t + 1:
            return 0.0
        return (DHpref[u_excl] - DHpref[t + 1]) - Hpref[t] * (
            Dpref[u_excl] - Dpref[t + 1]
        )

    # ---- Build feasible blocks and their costs ----
    blocks: List[Tuple[Tuple[int, int], float, float, float, float]] = (
        []
    )  # ((t,e), Q, H, physical_cost, reduced_block_cost)

    for t in range(T):
        for e in range(t, T):
            # Shelf-life: E_t >= e
            if expiry_abs[t] < e:
                continue

            q = block_demand(t, e)
            if q <= 0.0:
                continue  # no point in a block that serves no demand

            # Check branching constraints (θ_0, θ_1) for this block:
            violates = False

            # 1) Forbidden arcs: if (t,u) in θ_0 for some u in [t,e] with positive demand
            for u in range(t, e + 1):
                if demand[u] <= 0:
                    continue
                if (t, u) in theta_0:
                    violates = True
                    break
            if violates:
                continue

            # 2) Forced arcs: if any forced arc (t_f, u_f) has t_f != t but u_f in [t,e],
            #    then this block would steal u_f from the required t_f.
            if theta_1:
                for t_f, u_f in theta_1:
                    if t <= u_f <= e and t_f != t:
                        violates = True
                        break
                if violates:
                    continue

            # Physical block cost
            H_te = block_holding_cost(t, e)
            physical_cost = setup_cost[t] + prod_cost[t] * q + H_te

            # Reduced block cost: subtract capacity dual, and optionally arc duals (τ)
            reduced_cost_block = physical_cost - capacity_dual_prices[t] * q

            if arc_dual_prices:
                tau_sum = 0.0
                for u in range(t, e + 1):
                    if demand[u] > 0:
                        tau_sum += arc_dual_prices.get((t, u), 0.0)
                reduced_cost_block -= tau_sum

            blocks.append(((t, e), q, H_te, physical_cost, reduced_cost_block))

    if not blocks:
        # No feasible block under branching; subproblem infeasible
        return INF, None

    # ---- Build and solve MIP ----
    model = gurobi.Model(f"pricing_item_{production_item.item_id}")
    model.Params.OutputFlag = 0

    # Block variables B_{t,e} ∈ {0,1}
    b_vars: Dict[Tuple[int, int], gurobi.Var] = {}
    for (t, e), q, H_te, phys_cost, red_cost in blocks:
        b_vars[(t, e)] = model.addVar(vtype=GRB.BINARY, name=f"b_{t}_{e}")

    # Coverage constraints: for each period u with positive demand,
    # exactly one block that covers u must be chosen.
    for u in range(T):
        if demand[u] <= 0:
            continue
        expr = gurobi.LinExpr()
        for (t, e), q, H_te, phys_cost, red_cost in blocks:
            if t <= u <= e:
                expr += b_vars[(t, e)]
        model.addConstr(expr == 1.0, name=f"cover_{u}")

    # Objective: sum_{blocks} B_{t,e} * reduced_cost_block - μ_i
    obj = gurobi.LinExpr()
    for (t, e), q, H_te, phys_cost, red_cost in blocks:
        obj += red_cost * b_vars[(t, e)]
    obj += -convexity_dual_price
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        # Treat as no feasible improving column from the DP/MIP viewpoint
        return INF, None

    rc = model.ObjVal  # reduced cost of best block plan for this item

    if rc >= -epsilon_tolerance:
        # No improving column
        return rc, None

    # ---- Reconstruct a column (physical plan) from chosen blocks ----
    cap_usage = [0.0] * T
    setup_usage = [0.0] * T
    arc_usage: Dict[Tuple[int, int], float] = {}
    total_cost = 0.0

    for (t, e), q, H_te, phys_cost, red_cost in blocks:
        if b_vars[(t, e)].X > 0.5:
            total_cost += phys_cost
            cap_usage[t] += q
            setup_usage[t] = 1.0
            # arcs (t,u) for u=t..e serving positive demand
            for u in range(t, e + 1):
                if demand[u] > 0:
                    arc_usage[(t, u)] = 1.0

    return rc, ProductionPlanColumn(
        item_id=production_item.item_id,
        total_plan_cost=total_cost,
        capacity_usage_by_period=cap_usage,
        setup_open_fraction_by_period=setup_usage,
        lost_demand_quantity_by_period=[0.0] * T,
        arc_usage=arc_usage,
    )


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
    T = 8
    return (
        [
            ProductionItem(
                1,
                T,
                [6, 0, 5, 0, 3, 4, 0, 5],
                [3.0, 3.0, 3.3, 3.3, 3.6, 3.6, 3.8, 3.8],
                [22.0] * T,
                [1.0] * T,
                # expiry_abs[t] = t + L_t (inclusive)
                [t + l for t, l in enumerate([3, 3, 2, 2, 2, 2, 2, 2])],
                5000.0,
            ),
            ProductionItem(
                2,
                T,
                [0, 7, 0, 6, 0, 5, 4, 0],
                [2.8, 2.8, 3.0, 3.0, 3.2, 3.2, 3.5, 3.5],
                [18.0] * T,
                [0.8] * T,
                [t + l for t, l in enumerate([2, 2, 3, 3, 2, 2, 2, 2])],
                5000.0,
            ),
            ProductionItem(
                3,
                T,
                [4, 4, 4, 0, 3, 0, 6, 0],
                [3.2, 3.2, 3.2, 3.8, 3.8, 4.0, 4.0, 4.0],
                [24.0] * T,
                [1.2] * T,
                [t + l for t, l in enumerate([2, 3, 2, 2, 3, 2, 2, 2])],
                5000.0,
            ),
        ],
        [15, 11, 15, 8, 8, 8, 9, 9],
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
