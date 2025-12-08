#!/usr/bin/env python3
"""
Branch-and-Price with DP-based Column Generation
- Best-First Search (ordered by LP bound)
- Detailed CSV logging for CG convergence at each node
- Comprehensive output with lambdas and columns
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import math
import heapq
import gurobipy as gurobi
from gurobipy import GRB
import time
import csv
import os
from pathlib import Path


@dataclass
class ProductionItem:
    item_id: int
    number_of_periods: int
    demand_quantity_by_period: List[float]
    production_unit_cost_by_period: List[float]
    setup_cost_by_period: List[float]
    holding_unit_cost_by_period: List[float]
    perishability_horizon_by_start_period: List[int]  # E_t = t + L_t (inclusive)
    lost_sales_penalty_per_unit: float


@dataclass
class ProductionPlanColumn:
    item_id: int
    total_plan_cost: float
    capacity_usage_by_period: List[float]
    setup_open_fraction_by_period: Optional[List[float]] = None
    lost_demand_quantity_by_period: Optional[List[float]] = None
    arc_usage: Optional[Dict[Tuple[int, int], float]] = None

    def get_signature(self) -> str:
        """Unique signature for duplicate detection."""
        arcs = sorted(self.arc_usage.keys()) if self.arc_usage else []
        setups = [
            t for t, y in enumerate(self.setup_open_fraction_by_period or []) if y > 0.5
        ]
        return f"I{self.item_id}_Y{setups}_Z{arcs}"


@dataclass
class BranchNode:
    node_id: int
    parent_id: Optional[int]
    depth: int
    theta_0_by_item: Dict[int, Set[Tuple[int, int]]] = field(default_factory=dict)
    theta_1_by_item: Dict[int, Set[Tuple[int, int]]] = field(default_factory=dict)
    lp_bound: float = math.inf
    is_integer: bool = False
    is_pruned: bool = False
    prune_reason: Optional[str] = None
    branch_variable: Optional[Tuple[int, int, int, float]] = None
    branch_direction: Optional[str] = None

    def __lt__(self, other: "BranchNode") -> bool:
        return self.lp_bound < other.lp_bound


class CGLogger:
    """Logger for Column Generation convergence data."""

    def __init__(self, output_dir: str = "bp_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.node_logs: Dict[int, List[Dict]] = {}
        self.current_node_id = None
        self.global_log = []

    def start_node(self, node_id: int):
        self.current_node_id = node_id
        self.node_logs[node_id] = []

    def log_iteration(
        self,
        iteration: int,
        rmp_obj: float,
        reduced_costs: Dict[int, float],
        cols_added: int,
        total_cols: int,
        lambda_sums: Dict[int, float],
        pi: List[float],
        mu: Dict[int, float],
    ):
        """Log a single CG iteration."""
        if self.current_node_id is None:
            return

        worst_rc = min(reduced_costs.values()) if reduced_costs else 0.0
        total_rc = sum(reduced_costs.values()) if reduced_costs else 0.0

        log_entry = {
            "node_id": self.current_node_id,
            "iteration": iteration,
            "rmp_obj": rmp_obj,
            "worst_rc": worst_rc,
            "total_rc": total_rc,
            "cols_added": cols_added,
            "total_cols": total_cols,
        }

        # Add lambda sums per item
        for item_id, lam_sum in lambda_sums.items():
            log_entry[f"lambda_sum_i{item_id}"] = lam_sum

        # Add capacity duals
        for t, pi_t in enumerate(pi):
            log_entry[f"pi_{t}"] = pi_t

        # Add convexity duals
        for item_id, mu_i in mu.items():
            log_entry[f"mu_{item_id}"] = mu_i

        self.node_logs[self.current_node_id].append(log_entry)
        self.global_log.append(log_entry)

    def write_node_csv(self, node_id: int):
        """Write CSV for a specific node."""
        if node_id not in self.node_logs or not self.node_logs[node_id]:
            return

        logs = self.node_logs[node_id]
        filepath = self.output_dir / f"cg_node_{node_id}.csv"

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=logs[0].keys())
            writer.writeheader()
            writer.writerows(logs)

    def write_all_nodes_csv(self):
        """Write combined CSV for all nodes."""
        if not self.global_log:
            return

        filepath = self.output_dir / "cg_all_nodes.csv"

        # Get all possible keys
        all_keys = set()
        for entry in self.global_log:
            all_keys.update(entry.keys())
        all_keys = sorted(all_keys)

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for entry in self.global_log:
                # Fill missing keys with empty string
                row = {k: entry.get(k, "") for k in all_keys}
                writer.writerow(row)

    def write_summary_csv(self, nodes_data: List[Dict]):
        """Write summary CSV with one row per node."""
        if not nodes_data:
            return

        filepath = self.output_dir / "bp_node_summary.csv"

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=nodes_data[0].keys())
            writer.writeheader()
            writer.writerows(nodes_data)


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
        self.total_cg_iterations = 0
        self.total_columns_generated = 0

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

    def get_runtime(self) -> float:
        return time.time() - self.start_time

    def print_summary(self, best_lb: float, best_ub: Optional[float], eps: float):
        elapsed = self.get_runtime()
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
        print(f"  Total CG iters:     {self.total_cg_iterations}")
        print(f"  Total cols gen:     {self.total_columns_generated}")
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


def node_signature(node: BranchNode) -> str:
    sig_parts = []
    for item_id in sorted(node.theta_0_by_item.keys()):
        arcs = sorted(node.theta_0_by_item[item_id])
        if arcs:
            sig_parts.append(f"I{item_id}_0:{','.join(f'{t}-{u}' for t,u in arcs)}")
    for item_id in sorted(node.theta_1_by_item.keys()):
        arcs = sorted(node.theta_1_by_item[item_id])
        if arcs:
            sig_parts.append(f"I{item_id}_1:{','.join(f'{t}-{u}' for t,u in arcs)}")
    return "|".join(sig_parts)


def dp_pricing_for_single_item(
    production_item: ProductionItem,
    capacity_dual_prices: List[float],
    convexity_dual_price: float,
    theta_0: Set[Tuple[int, int]],
    theta_1: Set[Tuple[int, int]],
    arc_dual_prices: Optional[Dict[Tuple[int, int], float]] = None,
    epsilon_tolerance: float = 1e-9,
) -> Tuple[float, Optional[ProductionPlanColumn]]:
    """
    DP-based pricing for zero-inventory-ordering lot-sizing with perishability.
    """
    T = production_item.number_of_periods
    demand = production_item.demand_quantity_by_period
    setup_cost = production_item.setup_cost_by_period
    prod_cost = production_item.production_unit_cost_by_period
    holding_cost = production_item.holding_unit_cost_by_period
    expiry_abs = production_item.perishability_horizon_by_start_period

    INF = 1e10

    # Precomputation
    Dpref = [0.0] * (T + 1)
    for k in range(1, T + 1):
        Dpref[k] = Dpref[k - 1] + demand[k - 1]

    Hpref = [0.0] * (T + 1)
    for k in range(1, T + 1):
        Hpref[k] = Hpref[k - 1] + holding_cost[k - 1]

    DHpref = [0.0] * (T + 1)
    for k in range(1, T + 1):
        DHpref[k] = DHpref[k - 1] + demand[k - 1] * Hpref[k - 1]

    def block_demand_excl(t: int, u_excl: int) -> float:
        return Dpref[u_excl] - Dpref[t]

    def block_holding_cost_excl(t: int, u_excl: int) -> float:
        if u_excl <= t + 1:
            return 0.0
        return (DHpref[u_excl] - DHpref[t + 1]) - Hpref[t] * (
            Dpref[u_excl] - Dpref[t + 1]
        )

    # DP arrays
    F = [INF] * (T + 1)
    F[0] = 0.0
    pred: List[Tuple[int, int]] = [(-1, -1)] * (T + 1)

    for u_excl in range(1, T + 1):
        # Skip option when demand=0
        if demand[u_excl - 1] == 0 and F[u_excl - 1] < INF:
            if F[u_excl - 1] < F[u_excl]:
                F[u_excl] = F[u_excl - 1]
                pred[u_excl] = (-2, -2)

        # Production block option
        for t in range(u_excl):
            E_t = expiry_abs[t]
            if E_t < (u_excl - 1):
                continue

            violates = False

            # Forbidden arcs check
            for period in range(t, u_excl):
                if demand[period] > 0 and (t, period) in theta_0:
                    violates = True
                    break
            if violates:
                continue

            # Forced arcs consistency check
            for period in range(t, u_excl):
                if demand[period] > 0:
                    for t_f, u_f in theta_1:
                        if u_f == period and t_f != t:
                            violates = True
                            break
                if violates:
                    break
            if violates:
                continue

            q = block_demand_excl(t, u_excl)
            if q <= 0.0:
                continue

            hold_cost = block_holding_cost_excl(t, u_excl)
            cost = setup_cost[t] + prod_cost[t] * q + hold_cost
            cost -= capacity_dual_prices[t] * q

            if arc_dual_prices:
                tau_sum = 0.0
                for u in range(t, u_excl):
                    if demand[u] > 0:
                        tau_sum += arc_dual_prices.get((t, u), 0.0)
                cost -= tau_sum

            if F[t] + cost < F[u_excl]:
                F[u_excl] = F[t] + cost
                pred[u_excl] = (t, u_excl - 1)

    if F[T] >= INF:
        return INF, None

    # Reconstruct
    blocks: List[Tuple[int, int]] = []
    u = T
    while u > 0:
        if pred[u] == (-2, -2):
            u -= 1
            continue
        t, e = pred[u]
        if t < 0:
            break
        blocks.append((t, e))
        u = t
    if not blocks:
        return INF, None
    blocks.reverse()

    # Build column
    cap_usage = [0.0] * T
    setup_usage = [0.0] * T
    arc_usage: Dict[Tuple[int, int], float] = {}
    total_cost = 0.0

    for t, e in blocks:
        u_excl = e + 1
        q = Dpref[u_excl] - Dpref[t]
        if q <= 0.0:
            continue

        cap_usage[t] += q
        setup_usage[t] = 1.0

        for j in range(t, e + 1):
            if demand[j] > 0:
                arc_usage[(t, j)] = 1.0

        hold_cost = block_holding_cost_excl(t, u_excl)
        total_cost += setup_cost[t] + prod_cost[t] * q + hold_cost

    rc = total_cost - convexity_dual_price
    for t in range(T):
        if cap_usage[t] != 0.0:
            rc -= capacity_dual_prices[t] * cap_usage[t]

    if rc < -epsilon_tolerance:
        return rc, ProductionPlanColumn(
            item_id=production_item.item_id,
            total_plan_cost=total_cost,
            capacity_usage_by_period=cap_usage,
            setup_open_fraction_by_period=setup_usage,
            lost_demand_quantity_by_period=[0.0] * T,
            arc_usage=arc_usage,
        )
    else:
        return rc, None


class RestrictedMasterProblem:
    def __init__(self, items: List[ProductionItem], capacity: List[float]):
        self.items = items
        self.T = len(capacity)
        self.capacity = capacity
        self.model = gurobi.Model("RMP")
        self.model.Params.OutputFlag = 0
        self.model.Params.Method = 1

        self.columns: Dict[int, List[ProductionPlanColumn]] = {
            i.item_id: [] for i in items
        }
        self.lambdas: Dict[Tuple[int, int], gurobi.Var] = {}
        self.convex_expr = {i.item_id: gurobi.LinExpr(0.0) for i in items}
        self.cap_expr = [gurobi.LinExpr(0.0) for _ in range(self.T)]
        self.convex_con: Dict[int, gurobi.Constr] = {}
        self.cap_con: List[gurobi.Constr] = []
        self.column_signatures: Set[str] = set()

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

    def add_column(self, col: ProductionPlanColumn) -> bool:
        """Add column, return True if added (not duplicate)."""
        sig = col.get_signature()
        if sig in self.column_signatures:
            return False
        self.column_signatures.add(sig)

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
        return True

    def solve(self):
        self.model.optimize()
        if self.model.status != GRB.OPTIMAL:
            return math.inf, {}, []
        mu = {i.item_id: self.convex_con[i.item_id].Pi for i in self.items}
        pi = [self.cap_con[t].Pi for t in range(self.T)]
        return self.model.ObjVal, mu, pi

    def get_lambda_sums(self, eps: float = 1e-9) -> Dict[int, float]:
        """Get sum of lambda values per item."""
        sums = {i.item_id: 0.0 for i in self.items}
        # Only access .X if model has been solved
        if self.model.status != GRB.OPTIMAL:
            return sums
        try:
            for (item_id, idx), lam in self.lambdas.items():
                val = lam.X
                if val > eps:
                    sums[item_id] += val
        except Exception:
            pass  # Model not solved yet
        return sums

    def get_total_columns(self) -> int:
        return sum(len(cols) for cols in self.columns.values())


def solve_node_with_column_generation(
    items: List[ProductionItem],
    capacity: List[float],
    node: BranchNode,
    max_iter: int = 300,
    eps: float = 1e-9,
    verbose: bool = False,
    cg_logger: Optional[CGLogger] = None,
    stats: Optional[SearchStatistics] = None,
):
    """Solve node with column generation and optional detailed logging."""

    rmp = RestrictedMasterProblem(items, capacity)

    if cg_logger:
        cg_logger.start_node(node.node_id)

    # Add dummy columns
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

    for iteration in range(1, max_iter + 1):
        lb, mu, pi = rmp.solve()
        if not math.isfinite(lb):
            if verbose:
                print("INFEASIBLE")
            if cg_logger:
                cg_logger.write_node_csv(node.node_id)
            return math.inf, rmp, False, {}

        # Log iteration BEFORE adding new columns (while model is still solved)
        lambda_sums = rmp.get_lambda_sums(eps) if cg_logger else {}

        # Pricing
        reduced_costs = {}
        cols_added = 0

        for item in items:
            theta_0 = node.theta_0_by_item.get(item.item_id, set())
            theta_1 = node.theta_1_by_item.get(item.item_id, set())

            rc, col = dp_pricing_for_single_item(
                item,
                capacity_dual_prices=pi,
                convexity_dual_price=mu[item.item_id],
                theta_0=theta_0,
                theta_1=theta_1,
                arc_dual_prices=None,
                epsilon_tolerance=eps,
            )

            reduced_costs[item.item_id] = rc

            if col is not None and rc < -eps:
                if rmp.add_column(col):
                    cols_added += 1
                    if stats:
                        stats.total_columns_generated += 1

        # Log iteration (using lambda_sums computed before adding columns)
        if cg_logger:
            cg_logger.log_iteration(
                iteration=iteration,
                rmp_obj=lb,
                reduced_costs=reduced_costs,
                cols_added=cols_added,
                total_cols=rmp.get_total_columns(),
                lambda_sums=lambda_sums,
                pi=pi,
                mu=mu,
            )

        if stats:
            stats.total_cg_iterations += 1

        if cols_added == 0:
            break

    lb, mu, pi = rmp.solve()
    if verbose:
        print(f"LB={lb:.2f}")

    if cg_logger:
        cg_logger.write_node_csv(node.node_id)

    # Extract z values
    z_vals = {item.item_id: {} for item in items}
    for (item_id, idx), lam in rmp.lambdas.items():
        lam_val = lam.X
        if lam_val < eps:
            continue
        col = rmp.columns[item_id][idx]
        if col.arc_usage:
            for (t, u), arc_val in col.arc_usage.items():
                if arc_val > eps:
                    z_vals[item_id][(t, u)] = (
                        z_vals[item_id].get((t, u), 0.0) + lam_val * arc_val
                    )

    return lb, rmp, True, z_vals


def is_integer(z_vals: Dict[int, Dict[Tuple[int, int], float]], eps: float) -> bool:
    for item_id, arcs in z_vals.items():
        for (t, u), val in arcs.items():
            if eps < val < 1.0 - eps:
                return False
    return True


def find_most_fractional(
    z_vals: Dict[int, Dict[Tuple[int, int], float]], node: BranchNode, eps: float
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


def fathom_heap_by_incumbent(
    heap: List[Tuple[float, int, BranchNode, dict]], incumbent: float, eps: float
) -> int:
    original_size = len(heap)
    new_heap = [
        (bound, nid, node, z_vals)
        for bound, nid, node, z_vals in heap
        if bound < incumbent - eps
    ]
    heap.clear()
    for item in new_heap:
        heapq.heappush(heap, item)
    return original_size - len(heap)


def print_rmp_solution(
    rmp: RestrictedMasterProblem, items: List[ProductionItem], eps: float = 1e-6
):
    """Print detailed RMP solution with lambdas and column contents."""
    print("\n" + "=" * 70)
    print(" " * 30 + "FINAL RMP VARIABLE VALUES")
    print("=" * 70)

    for item in items:
        item_id = item.item_id
        print(f"\n--- Item {item_id} ---")

        active_cols = []
        for idx, col in enumerate(rmp.columns[item_id]):
            lam = rmp.lambdas.get((item_id, idx))
            if lam is not None and lam.X > eps:
                active_cols.append((idx, lam.X, col))

        if not active_cols:
            print("  No active columns")
            continue

        for idx, lam_val, col in active_cols:
            print(
                f"\nλ[{item_id},{idx}] = {lam_val:.6f}, cost = {col.total_plan_cost:.2f}"
            )

            x = col.capacity_usage_by_period
            y = col.setup_open_fraction_by_period
            print(f"  x_it (capacity usage): {[f'{v:.2f}' for v in x]}")
            print(f"  y_it (setup):          {[f'{v:.2f}' for v in y]}")

            if col.arc_usage:
                arcs_str = ", ".join(
                    f"({t},{u})" for t, u in sorted(col.arc_usage.keys())
                )
                print(f"  z_itu (arcs):          {arcs_str}")
            else:
                print(f"  z_itu (arcs):          (none - dummy)")


def print_optimal_policy(
    z_vals: Dict[int, Dict[Tuple[int, int], float]],
    items: List[ProductionItem],
    eps: float = 1e-6,
):
    """Print the optimal order policy from z values."""
    print("\n" + "=" * 70)
    print(" " * 25 + "OPTIMAL NODE ORDER POLICY")
    print("=" * 70)

    for item in items:
        print(f"\nItem {item.item_id}:")
        print("-" * 25)
        arcs = z_vals.get(item.item_id, {})
        if not arcs:
            print("  No production (lost sales only).")
            continue
        for (t, u), val in sorted(arcs.items()):
            if val > 0.9:
                demand_u = item.demand_quantity_by_period[u]
                print(
                    f"  Produce in period {t+1:>2d} → satisfies demand of period {u+1:>2d} (d={demand_u:.1f})"
                )


def solve_branch_and_price(
    items,
    capacity,
    max_nodes=5000,
    eps=1e-6,
    print_frequency=50,
    log_dir: str = "bp_logs",
):
    """Main Branch-and-Price solver with logging."""

    print("\n" + "=" * 70)
    print(" " * 18 + "BRANCH-AND-PRICE WITH BEST-FIRST SEARCH")
    print("=" * 70)

    stats = SearchStatistics()
    cg_logger = CGLogger(output_dir=log_dir)
    nodes_summary = []

    root = BranchNode(
        0, None, 0, {i.item_id: set() for i in items}, {i.item_id: set() for i in items}
    )

    print("\n>>> ROOT NODE <<<")
    root_lb, root_rmp, _, z_vals = solve_node_with_column_generation(
        items, capacity, root, verbose=True, cg_logger=cg_logger, stats=stats
    )

    if not math.isfinite(root_lb):
        print("\n✗ Root infeasible!")
        stats.print_summary(math.inf, None, eps)
        cg_logger.write_all_nodes_csv()
        return math.inf, None, 1, None, None, None

    root.lp_bound = root_lb
    print(f"  Root LB:  {root_lb:.4f}")
    print(f"  Integer?  {is_integer(z_vals, eps)}")

    # Log root node summary
    nodes_summary.append(
        {
            "node_id": 0,
            "parent_id": -1,
            "depth": 0,
            "lp_bound": root_lb,
            "is_integer": is_integer(z_vals, eps),
            "status": "integer" if is_integer(z_vals, eps) else "fractional",
            "branch_var": "",
            "branch_dir": "",
        }
    )

    best_lb = root_lb
    best_ub: Optional[float] = None
    best_z: Optional[Dict] = None
    best_rmp: Optional[RestrictedMasterProblem] = None
    best_node: Optional[BranchNode] = None

    nodes_explored = 1
    node_counter = 1

    seen_signatures = {node_signature(root)}
    stats.node_created()

    if is_integer(z_vals, eps):
        best_ub = root_lb
        best_lb = root_lb
        best_z = z_vals
        best_rmp = root_rmp
        best_node = root
        root.is_integer = True
        stats.node_explored(root, incumbent_improved=True)
        print("\n✓ Root is INTEGER - OPTIMAL!")
        cg_logger.write_all_nodes_csv()
        cg_logger.write_summary_csv(nodes_summary)
        stats.print_summary(best_lb, best_ub, eps)
        return root_lb, root_lb, 1, best_node, best_z, best_rmp

    stats.node_explored(root)

    heap: List[Tuple[float, int, BranchNode, dict]] = []

    print(f"\n{'='*70}")
    print("BEST-FIRST SEARCH")
    print(f"{'='*70}\n")

    # Branch from root
    branch_var = find_most_fractional(z_vals, root, eps)
    if branch_var:
        item_id, t_br, u_br, z_val = branch_var
        print(f"  Branch on Z[{item_id},{t_br},{u_br}]={z_val:.3f}")

        # Z=0 child
        left = BranchNode(
            node_counter,
            root.node_id,
            1,
            {i: root.theta_0_by_item[i].copy() for i in root.theta_0_by_item},
            {i: root.theta_1_by_item[i].copy() for i in root.theta_1_by_item},
            branch_variable=branch_var,
            branch_direction="Z=0",
        )
        left.theta_0_by_item[item_id].add((t_br, u_br))
        left.lp_bound = root_lb
        heapq.heappush(heap, (root_lb, node_counter, left, z_vals))
        seen_signatures.add(node_signature(left))
        stats.node_created()
        node_counter += 1

        # Z=1 child
        right = BranchNode(
            node_counter,
            root.node_id,
            1,
            {i: root.theta_0_by_item[i].copy() for i in root.theta_0_by_item},
            {i: root.theta_1_by_item[i].copy() for i in root.theta_1_by_item},
            branch_variable=branch_var,
            branch_direction="Z=1",
        )
        right.theta_1_by_item[item_id].add((t_br, u_br))
        right.lp_bound = root_lb
        heapq.heappush(heap, (root_lb, node_counter, right, z_vals))
        seen_signatures.add(node_signature(right))
        stats.node_created()
        node_counter += 1

    while heap and nodes_explored < max_nodes:
        bound, nid, node, parent_z = heapq.heappop(heap)

        if best_ub is not None and bound >= best_ub - eps:
            node.is_pruned = True
            node.prune_reason = "bound"
            stats.node_explored(node)
            nodes_summary.append(
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "depth": node.depth,
                    "lp_bound": bound,
                    "is_integer": False,
                    "status": "pruned_bound",
                    "branch_var": (
                        str(node.branch_variable) if node.branch_variable else ""
                    ),
                    "branch_dir": node.branch_direction or "",
                }
            )
            continue

        if nodes_explored % print_frequency == 0:
            gap_str = "N/A"
            if best_ub is not None:
                gap = best_ub - best_lb
                gap_pct = 100 * gap / max(abs(best_ub), 1e-10)
                gap_str = f"{gap_pct:.2f}%"
            print(
                f"[Progress: N={nodes_explored:4d}, Heap={len(heap):4d}, "
                f"LB={best_lb:.2f}, UB={best_ub if best_ub else 'N/A'}, Gap={gap_str}]"
            )

        print(f"N{node.node_id:4d} D{node.depth:2d}", end="", flush=True)

        lb, rmp, converged, z_vals = solve_node_with_column_generation(
            items, capacity, node, verbose=True, cg_logger=cg_logger, stats=stats
        )
        nodes_explored += 1
        node.lp_bound = lb

        if not math.isfinite(lb):
            print("  FATHOMED: Infeasible")
            node.is_pruned = True
            node.prune_reason = "infeasible"
            stats.node_explored(node)
            nodes_summary.append(
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "depth": node.depth,
                    "lp_bound": float("inf"),
                    "is_integer": False,
                    "status": "infeasible",
                    "branch_var": (
                        str(node.branch_variable) if node.branch_variable else ""
                    ),
                    "branch_dir": node.branch_direction or "",
                }
            )
            if heap:
                best_lb = heap[0][0]
            elif best_ub is not None:
                best_lb = best_ub
            continue

        if heap:
            best_lb = min(lb, heap[0][0])
        else:
            best_lb = lb

        if best_ub is not None and lb >= best_ub - eps:
            print(f"  FATHOMED: {lb:.2f} ≥ {best_ub:.2f}")
            node.is_pruned = True
            node.prune_reason = "bound"
            stats.node_explored(node)
            nodes_summary.append(
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "depth": node.depth,
                    "lp_bound": lb,
                    "is_integer": False,
                    "status": "pruned_bound",
                    "branch_var": (
                        str(node.branch_variable) if node.branch_variable else ""
                    ),
                    "branch_dir": node.branch_direction or "",
                }
            )
            if heap:
                best_lb = heap[0][0]
            elif best_ub is not None:
                best_lb = best_ub
            continue

        if is_integer(z_vals, eps):
            print(f"  INTEGER: {lb:.2f}", end="")
            node.is_integer = True
            status = "integer"
            if best_ub is None or lb < best_ub - eps:
                best_ub = lb
                best_z = z_vals
                best_rmp = rmp
                best_node = node
                print(" ★ NEW INCUMBENT!")
                status = "integer_incumbent"
                num_fathomed = fathom_heap_by_incumbent(heap, best_ub, eps)
                if heap:
                    best_lb = heap[0][0]
                else:
                    best_lb = best_ub
                stats.node_explored(
                    node, incumbent_improved=True, num_fathomed=num_fathomed
                )
            else:
                print()
                if heap:
                    best_lb = heap[0][0]
                else:
                    best_lb = best_ub
                stats.node_explored(node)
            nodes_summary.append(
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "depth": node.depth,
                    "lp_bound": lb,
                    "is_integer": True,
                    "status": status,
                    "branch_var": (
                        str(node.branch_variable) if node.branch_variable else ""
                    ),
                    "branch_dir": node.branch_direction or "",
                }
            )
            continue

        stats.node_explored(node)
        nodes_summary.append(
            {
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "depth": node.depth,
                "lp_bound": lb,
                "is_integer": False,
                "status": "fractional",
                "branch_var": str(node.branch_variable) if node.branch_variable else "",
                "branch_dir": node.branch_direction or "",
            }
        )

        # Branch
        branch_var = find_most_fractional(z_vals, node, eps)
        if branch_var is None:
            print("  No fractional variable")
            node.is_pruned = True
            if heap:
                best_lb = heap[0][0]
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
                left.lp_bound = lb
                heapq.heappush(heap, (lb, node_counter, left, z_vals))
                stats.node_created()
                node_counter += 1
                children += 1

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
                right.lp_bound = lb
                heapq.heappush(heap, (lb, node_counter, right, z_vals))
                stats.node_created()
                node_counter += 1
                children += 1

        if children > 0:
            print(f"  Branch Z[{item_id},{t_br},{u_br}]={z_val:.3f}")

    if not heap and best_ub is not None:
        best_lb = best_ub

    # Write final logs
    cg_logger.write_all_nodes_csv()
    cg_logger.write_summary_csv(nodes_summary)

    stats.print_summary(best_lb, best_ub, eps)

    print(f"\n  Logs written to: {cg_logger.output_dir}/")

    return best_lb, best_ub, nodes_explored, best_node, best_z, best_rmp


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
        items, cap, max_nodes=10000, print_frequency=50, log_dir="bp_logs"
    )

    if opt_z:
        print_optimal_policy(opt_z, items)

    if opt_rmp:
        print_rmp_solution(opt_rmp, items)
