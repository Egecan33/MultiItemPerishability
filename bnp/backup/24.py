"""
ZIO Full-Plans Branch-and-Price Solver.

Uses complete ZIO production plans as columns, where each column:
- Covers ALL demand for one item
- Is a feasible uncapacitated lot-sizing solution (Wagner-Whitin style)
- Can have "dummy setups" (Y=1, X=0) when branching forces Y[t]=1

The RMP combines columns via convex combination:
- Convexity: Σ λ_k = 1 per item
- Capacity: Σ_i Σ_k x_k[t] * λ_k ≤ capacity[t] (shared)

Branching on Y[i,t] forces/forbids setups at specific periods.
Through branching, the Y pattern becomes fixed,(all columns share same Y pattern
with dummy setups creating different X patterns).
"""

from __future__ import annotations
import csv
import json
import math
import time
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import gurobipy as gp
from gurobipy import GRB

EPS = 1e-6

# ============================================================================
# CONFIGURATION FLAGS - Easy on/off switches
# ============================================================================

# --- Search Strategy ---
USE_DFS_UNTIL_INCUMBENT = True  # Use DFS until incumbent found, then best-first
# When True: prioritizes deeper nodes to find integer solutions faster
# When False: uses best-first search (prioritizes lower LP bounds)

# --- Dive Heuristic (runs periodically during B&P) ---
ENABLE_DIVE_HEURISTIC = False  # Try to find integer solution by fixing Y values
ENABLE_COLUMN_SELECTION = False  # Try column selection heuristic when dive fails

# --- Column Generation Settings ---
MAX_COLUMNS_PER_ITEM = 10  # Max columns to add per item per CG iteration

# --- Other Settings ---
ENABLE_SHELF_LIFE_PRIORITIZATION = (
    False  # Prioritize shelf-life-0 periods in heuristics
)

# ============================================================================


# DATA STRUCTURES


@dataclass
class ZIOColumn:
    """A ZIO column representing a complete production plan for one item."""

    item_id: int
    y: Dict[int, int]  # Y[t] = 1 if setup at period t
    x: Dict[int, float]  # X[t] = production quantity at period t
    z: Dict[Tuple[int, int], int]  # Z[t,u] = 1 if production at t serves demand at u
    cost: float  # Total cost (setup + variable + holding)

    def __repr__(self):
        y_periods = sorted([t for t, v in self.y.items() if v == 1])
        x_vals = {t: self.x.get(t, 0) for t in y_periods if self.x.get(t, 0) > 0}
        dummy = [t for t in y_periods if self.x.get(t, 0) < EPS]
        return f"ZIOColumn(item={self.item_id}, cost={self.cost:.2f}, Y={y_periods}, dummy={dummy})"

    def signature(self) -> Tuple:
        """Unique signature for deduplication."""
        return (self.item_id, tuple(sorted(self.x.items())))


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
    )  # item -> {(t,u) arcs}
    forbidden_z: Dict[int, Set[Tuple[int, int]]] = field(
        default_factory=dict
    )  # item -> {(t,u) arcs}
    lp_bound: float = math.inf

    def __lt__(self, other: "BranchNode") -> bool:
        return self.lp_bound < other.lp_bound

    def copy_constraints(self):
        """Deep copy the branching constraints."""
        forced_y = {i: set(s) for i, s in self.forced_y.items()}
        forbidden_y = {i: set(s) for i, s in self.forbidden_y.items()}
        forced_z = {i: set(s) for i, s in self.forced_z.items()}
        forbidden_z = {i: set(s) for i, s in self.forbidden_z.items()}
        return forced_y, forbidden_y, forced_z, forbidden_z


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


# CSV LOGGING


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

        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Node log
        nodes_path = self.out_dir / "bnp_nodes.csv"
        with open(nodes_path, "w", newline="") as f:
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
                        e.lp_bound,
                        e.incumbent,
                        e.branch_item,
                        e.branch_t,
                        e.direction,
                        e.status,
                        e.cg_iters,
                        e.columns_added,
                    ]
                )

        # CG iterations log
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
                        e.rmp_obj,
                        e.num_columns,
                        e.min_rc,
                        e.columns_added,
                    ]
                )


# =============================================================================
# PRICING SUBPROBLEM (DP)
# =============================================================================


def generate_initial_column(
    item_id: int,
    demand: List[float],
    setup: List[float],
    h: List[float],
    c_var: List[float],
    shelf_seq: List[int],
    forbidden_y: Set[int],
    forced_y: Set[int],
) -> ZIOColumn:
    """
    Generate initial ZIO column (lot-for-lot style) respecting constraints.

    Uses shelf_seq[t] to determine how far production at t can reach.
    """
    T = len(demand)
    demand_periods = [t for t in range(T) if demand[t] > EPS]

    y = {}
    x = {}
    z = {}
    cost = 0.0

    i = 0
    while i < len(demand_periods):
        u = demand_periods[i]

        # Find latest allowed production period <= u that can reach u
        prod_t = None
        for t in range(u, -1, -1):
            if t not in forbidden_y:
                # If shelf_seq[t] <= 0, production at t cannot serve any demand
                if shelf_seq[t] <= 0:
                    continue
                # Check if t can reach u (t + shelf_seq[t] >= u)
                max_reach = min(T - 1, t + shelf_seq[t])
                if u <= max_reach:
                    prod_t = t
                    break

        if prod_t is None:
            raise ValueError(f"Item {item_id}: Cannot produce for demand at {u}")

        # Produce at prod_t, cover consecutive reachable demands
        max_reachable = min(T - 1, prod_t + shelf_seq[prod_t])
        y[prod_t] = 1
        total_prod = 0.0
        covered = []

        for j in range(i, len(demand_periods)):
            uj = demand_periods[j]
            if uj >= prod_t and uj <= max_reachable:
                total_prod += demand[uj]
                covered.append(uj)
                z[(prod_t, uj)] = 1
            else:
                break

            # Stop if next demand could have its own production
            if j + 1 < len(demand_periods):
                next_u = demand_periods[j + 1]
                has_allowed = any(
                    t not in forbidden_y for t in range(prod_t + 1, next_u + 1)
                )
                if has_allowed:
                    break

        x[prod_t] = total_prod

        # Cost
        s_cost = setup[prod_t]
        v_cost = sum(c_var[prod_t] * demand[uj] for uj in covered)
        # Holding cost: sum of h[r] for each period r from prod_t to uj-1, times demand[uj]
        h_cost = sum(
            sum(h[r] if isinstance(h, list) else h for r in range(prod_t, uj))
            * demand[uj]
            for uj in covered
        )
        cost += s_cost + v_cost + h_cost

        i += len(covered)

    # Add forced Y periods as dummy setups
    for t in forced_y:
        if t not in y and t not in forbidden_y:
            y[t] = 1
            x[t] = 0
            cost += setup[t]  # here we are adding the setup cost for the dummy setup

    return ZIOColumn(
        item_id=item_id, y=y, x=x, z=z, cost=cost
    )  # returns ZIO column with dummy setup


# DP-based pricing following Algorithm 3 from the paper (Wagner-Whitin on ZIO Blocks)
# Modified to return K-best columns for diversity needed in convex combinations
def price_zio_column_dp(
    item_id: int,
    demand: List[float],
    setup: List[float],
    h: List[float],
    c_var: List[float],
    mu: float,
    rho: Dict[int, float],
    shelf_seq: List[int],
    max_columns: int = 5,
    forced_y: Set[int] = None,
    forbidden_y: Set[int] = None,
    forced_z: Set[Tuple[int, int]] = None,
    forbidden_z: Set[Tuple[int, int]] = None,
    sig_y: Dict[int, float] = None,
    tau_z: Dict[Tuple[int, int], float] = None,
) -> List[Tuple[ZIOColumn, float]]:
    """
    DP-based K-best path pricing to find negative reduced cost ZIO columns.

    Following Algorithm 3 from the paper with K-best extension (currently set on 1 column per item per itereation, can be turned into k-best):
    - Forward DP tracking K-best paths at each state
    - Returns up to max_columns columns with negative reduced cost
    - Enables diversity needed for optimal convex combinations

    shelf_seq[t] defines how far forward production at t can serve (t to t+shelf_seq[t]).
    sig_y and tau_z duals give "discounts" to columns that satisfy forced constraints,
    ensuring we generate all necessary columns when branching.
    """
    T = len(demand)
    forced_y = forced_y or set()
    forbidden_y = forbidden_y or set()
    forced_z = forced_z or set()
    forbidden_z = forbidden_z or set()
    sig_y = sig_y or {}
    tau_z = tau_z or {}

    demand_periods = [u for u in range(T) if demand[u] > EPS]

    if (
        not demand_periods
    ):  # if there is no demand, we return a column with only forced dummy setups (edge case)
        # No demand - column with only forced dummy setups
        dummy_cost = sum(setup[t] for t in forced_y if t not in forbidden_y)
        # Add sig_y discounts for forced Y periods
        sig_y_discount = sum(
            sig_y.get(t, 0.0) for t in forced_y if t not in forbidden_y
        )
        rc = dummy_cost - mu - sig_y_discount

        if rc < -EPS:
            y = {t: 1 for t in forced_y if t not in forbidden_y}
            z = {}
            for t, u in forced_z:
                z[(t, u)] = 1
                if t not in y:
                    y[t] = 1

            cost = sum(setup[t] for t, v in y.items() if v == 1)
            return [(ZIOColumn(item_id=item_id, y=y, x={}, z=z, cost=cost), rc)]

        return []

    n = len(demand_periods)
    INF = float("inf")

    # Build required production periods from forced_z
    required_prod = {}
    for t, u in forced_z:
        if u in demand_periods:
            required_prod[u] = t

    # Precompute all valid blocks and their costs
    blocks_by_start = defaultdict(list)  # i -> list of (t, j, rc, actual)

    for i in range(n):
        u_start = demand_periods[i]

        for t in range(u_start + 1):
            if t in forbidden_y:
                continue

            # Per-period shelf life: production at t can serve t to t + shelf_seq[t]
            # If shelf_seq[t] <= 0, production at t cannot serve any demand
            if shelf_seq[t] <= 0:
                continue

            max_reachable = min(T - 1, t + shelf_seq[t])

            for j in range(i, n):
                u_end = demand_periods[j]

                if u_end > max_reachable:
                    break

                # Check forbidden arcs and forced_z constraints
                demands_in_block = demand_periods[i : j + 1]
                valid = True
                for u in demands_in_block:
                    if (t, u) in forbidden_z:
                        valid = False
                        break
                    if u in required_prod and required_prod[u] != t:
                        valid = False
                        break

                if not valid:
                    continue

                # Compute block costs
                total_prod = sum(demand[u] for u in demands_in_block)
                rho_t = rho.get(t, 0.0)

                # For forced Y periods: set setup cost to 0 in DP, add back later
                # This follows the standard approach: solve DP with S_i'=0, then add S_i' to F_T
                s_cost_dp = 0.0 if t in forced_y else setup[t]
                s_cost_actual = setup[t]  # Always use real cost for actual

                v_cost = sum(c_var[t] * demand[u] for u in demands_in_block)
                # Holding cost: sum of h[r] for each period r from t to u-1, times demand[u]
                h_cost = sum(
                    sum(h[r] if isinstance(h, list) else h for r in range(t, u))
                    * demand[u]
                    for u in demands_in_block
                )

                actual = s_cost_actual + v_cost + h_cost
                dp_cost = s_cost_dp + v_cost + h_cost

                # tau_z discount: for each forced arc (t,u) covered by block
                tau_z_discount = 0.0
                for u in demands_in_block:
                    if (t, u) in forced_z:
                        tau_z_discount += tau_z.get((t, u), 0.0)

                reduced = dp_cost - rho_t * total_prod - tau_z_discount

                blocks_by_start[i].append((t, j, reduced, actual))

    # K-best DP: F[idx] = list of (reduced_cost, actual_cost, path) for k-best paths
    K = max(
        max_columns * 10, 50
    )  # Keep many paths for diversity needed in convex combinations

    # Each entry: (reduced_cost, actual_cost, path as list of (t, i, j))
    F = [[] for _ in range(n + 1)]
    F[0] = [(0.0, 0.0, [])]

    for idx in range(n):
        if not F[idx]:
            continue

        # Extend each path with all valid blocks
        for rc_so_far, actual_so_far, path_so_far in F[idx]:
            for t, j, rc_block, actual_block in blocks_by_start[idx]:
                new_rc = rc_so_far + rc_block
                new_actual = actual_so_far + actual_block
                new_path = path_so_far + [(t, idx, j)]

                dest_idx = j + 1
                F[dest_idx].append((new_rc, new_actual, new_path))

        # Keep only K-best at each state
        for dest_idx in range(idx + 1, n + 1):
            if len(F[dest_idx]) > K:
                F[dest_idx].sort(key=lambda x: x[0])
                F[dest_idx] = F[dest_idx][:K]

    if not F[n]:
        return []

    # Build columns from all paths with negative reduced cost
    results = []
    seen_signatures = set()

    for rc_path, actual_path, path in F[n]:
        production_periods = set(p[0] for p in path)

        # Add back setup costs for forced Y periods that were set to 0 in DP
        # This follows: solve DP with S_i'=0, then compute F_T + S_i' for real reduced cost
        forced_setup_cost_in_path = 0.0
        for t in forced_y:
            if t in production_periods:
                forced_setup_cost_in_path += setup[t]  # Add back S_i' that was 0 in DP

        # Add forced_y dummy setup costs (for periods NOT in path)
        dummy_setup_cost = 0.0
        for t in forced_y:
            if t not in production_periods and t not in forbidden_y:
                dummy_setup_cost += setup[t]

        # Add tau_z discount for forced arcs not already covered
        arcs_covered = set()
        for prod_t, start_i, end_j in path:
            for u in demand_periods[start_i : end_j + 1]:
                arcs_covered.add((prod_t, u))
        dummy_tau_z_discount = 0.0
        for t, u in forced_z:
            if (t, u) not in arcs_covered:
                dummy_tau_z_discount += tau_z.get((t, u), 0.0)

        # Final reduced cost = F_T + S_i' (for forced periods in path) + dummy costs - mu
        final_rc = (
            rc_path
            + forced_setup_cost_in_path  # Add back real cost for forced Y in path
            + dummy_setup_cost  # Add cost for forced Y not in path (dummy setups)
            - mu
            - dummy_tau_z_discount
        )
        final_actual = actual_path + dummy_setup_cost

        if final_rc >= -EPS:
            continue

        # Build column
        y = {}
        x = {}
        z = {}

        for prod_t, start_i, end_j in path:
            y[prod_t] = 1
            demands_covered = demand_periods[start_i : end_j + 1]
            x[prod_t] = sum(demand[u] for u in demands_covered)
            for u in demands_covered:
                z[(prod_t, u)] = 1

        # Add forced_y as dummy setups
        for t in forced_y:
            if t not in y and t not in forbidden_y:
                y[t] = 1
                x[t] = 0

        # Add forced_z as dummy arcs
        for t, u in forced_z:
            if (t, u) not in z:
                z[(t, u)] = 1
                if t not in y:
                    y[t] = 1
                    x[t] = 0

        # Deduplicate by signature
        y_sig = tuple(sorted(y.keys()))
        x_sig = tuple(sorted((k, v) for k, v in x.items()))
        z_sig = tuple(sorted(z.keys()))
        sig = (y_sig, x_sig, z_sig)

        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)

        col = ZIOColumn(item_id=item_id, y=y, x=x, z=z, cost=final_actual)
        results.append((col, final_rc))

    results.sort(key=lambda x: x[1])
    return results[:max_columns]


# LEFO-COMPATIBLE Z COMPUTATION (Post-processing)


def compute_lefo_z_values(
    item_id: int,
    x_values: Dict[int, float],  # t -> production quantity
    demand: List[float],
    shelf_seq: List[int],
    T: int,
) -> Tuple[bool, Dict[Tuple[int, int], float]]:
    """
    Given X values (production quantities), find LEFO-compatible Z values using LP.

    Z[t,u] ∈ [0,1] CONTINUOUS - fraction of demand u served by production t

    Constraints:
    - Σ_t Z[t,u] = 1 for each u (demand coverage)
    - Σ_u Z[t,u] * demand[u] ≤ X[t] (production capacity)
    - Z[t1,u] + Z[t2,u'] ≤ 1 (no-crossing / LEFO)

    Returns (is_feasible, z_values) where z_values[(t,u)] is the arc flow fraction.
    """
    # Get production periods and demand periods
    prod_periods = [t for t, x in x_values.items() if x > EPS]
    demand_periods = [u for u in range(T) if demand[u] > EPS]

    if not prod_periods or not demand_periods:
        return True, {}

    # Compute expiry for each production period
    expiry = {}
    for t in prod_periods:
        m_t = shelf_seq[t] if t < len(shelf_seq) else T
        expiry[t] = t + m_t

    # Build valid arcs: (t, u) where t <= u < t + shelf_seq[t]
    valid_arcs = []
    for t in prod_periods:
        for u in demand_periods:
            if t <= u < expiry[t]:
                valid_arcs.append((t, u))

    if not valid_arcs:
        return False, {}

    # Check if each demand period has at least one arc
    for u in demand_periods:
        arcs_to_u = [(t, uu) for (t, uu) in valid_arcs if uu == u]
        if not arcs_to_u:
            return False, {}

    # Create Gurobi LP model with LEFO constraints
    m = gp.Model("LEFO_Z_LP")
    m.Params.OutputFlag = 0

    # Variables: Z[t,u] ∈ [0,1] CONTINUOUS
    Z = {}
    for t, u in valid_arcs:
        Z[t, u] = m.addVar(lb=0, ub=1, name=f"Z_{t}_{u}")

    # Constraint (1): Demand coverage - Σ_t Z[t,u] = 1
    for u in demand_periods:
        arcs_to_u = [(t, uu) for (t, uu) in valid_arcs if uu == u]
        if arcs_to_u:
            m.addConstr(
                gp.quicksum(Z[t, u] for t, uu in arcs_to_u) == 1,
                name=f"demand_{u}",
            )

    # Constraint (2): Production capacity - Σ_u Z[t,u] * demand[u] ≤ X[t]
    for t in prod_periods:
        arcs_from_t = [(tt, u) for (tt, u) in valid_arcs if tt == t]
        if arcs_from_t:
            m.addConstr(
                gp.quicksum(Z[t, u] * demand[u] for tt, u in arcs_from_t)
                <= x_values[t],
                name=f"prod_cap_{t}",
            )

    # Constraint (3): No-crossing (LEFO)
    # For t1, t2 where exp(t1) < exp(t2):
    # If t2 serves u', then t1 cannot serve any u in [t2, u'-1]
    # => Z[t1, u] + Z[t2, u'] <= 1 for such pairs
    prod_sorted = sorted(prod_periods, key=lambda t: expiry[t])

    for a in range(len(prod_sorted)):
        t1 = prod_sorted[a]
        v1 = expiry[t1]

        for b in range(a + 1, len(prod_sorted)):
            t2 = prod_sorted[b]
            v2 = expiry[t2]

            if v1 >= v2:
                continue  # Only when v1 < v2

            # Get demands that t2 can serve
            t2_demands = [u for u in demand_periods if (t2, u) in Z]

            for up in t2_demands:
                # Get demands in [t2, up-1] that t1 can serve
                for u in demand_periods:
                    if t2 <= u <= up - 1 and (t1, u) in Z:
                        # No-crossing: Z[t1, u] + Z[t2, up] <= 1
                        m.addConstr(
                            Z[t1, u] + Z[t2, up] <= 1,
                            name=f"nocross_{t1}_{t2}_{u}_{up}",
                        )

    # Objective: minimize total (just find any feasible solution)
    m.setObjective(0, GRB.MINIMIZE)
    m.optimize()

    if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        z_vals = {(t, u): Z[t, u].X for t, u in valid_arcs if Z[t, u].X > EPS}
        return True, z_vals
    else:
        return False, {}


def compute_all_lefo_z(
    items: Dict[int, dict],
    x_per_item: Dict[int, Dict[int, float]],
    T: int,
) -> Tuple[bool, Dict[int, Dict[Tuple[int, int], float]]]:
    """
    Compute LEFO-compatible Z values for all items.

    Returns (all_feasible, z_per_item) where z_per_item[i][(t,u)] is the arc flow.
    """
    z_per_item = {}
    all_feasible = True

    for i, item_data in items.items():
        demand = item_data["demand"]
        shelf_seq = item_data.get("shelf_seq", [T] * T)
        x_values = x_per_item.get(i, {})

        is_feas, z_vals = compute_lefo_z_values(i, x_values, demand, shelf_seq, T)
        z_per_item[i] = z_vals

        if not is_feas:
            all_feasible = False

    return all_feasible, z_per_item


# RESTRICTED MASTER PROBLEM


def solve_rmp_with_duals(
    items: Dict[int, dict],
    columns: Dict[int, List[ZIOColumn]],  # item_id -> columns
    capacity: List[float],
    T: int,
    forced_y: Dict[int, Set[int]] = None,
    forbidden_y: Dict[int, Set[int]] = None,
    forced_z: Dict[int, Set[Tuple[int, int]]] = None,
    forbidden_z: Dict[int, Set[Tuple[int, int]]] = None,
) -> Tuple[
    Optional[float],
    Dict[int, float],
    Dict[int, float],
    Dict[int, Dict[int, float]],
    Dict[Tuple[int, int], float],  # sig_y duals
    Dict[Tuple[int, int, int], float],  # tau_z duals
]:
    """
    Solve the RMP and return (objective, mu_duals, rho_duals, lambda_values, sig_y, tau_z).

    RMP:
        min  Σ_i Σ_k c_k^i λ_k^i
        s.t. Σ_k λ_k^i = 1                    ∀i  (convexity) → dual μ_i
             Σ_i Σ_k x_k^i[t] λ_k^i ≤ C_t    ∀t  (capacity)  → dual ρ_t
             Σ_k y_k[t] λ_k = 1               (forced Y)     → dual sig_y[i,t]
             Σ_k z_k[t,u] λ_k = 1             (forced Z)     → dual τ_z[i,t,u]
             λ_k^i ≥ 0

    Columns violating branching constraints are excluded.
    sig_y and τ_z duals give "discounts" for pricing to generate required columns.

    Returns None if infeasible.
    """
    forced_y = forced_y or {}
    forbidden_y = forbidden_y or {}
    forced_z = forced_z or {}
    forbidden_z = forbidden_z or {}

    def column_is_valid(col: ZIOColumn) -> bool:
        """Check if column satisfies branching constraints."""
        i = col.item_id

        # Check forced Y: column must have Y[t]=1 for all forced periods
        for t in forced_y.get(i, set()):
            if col.y.get(t, 0) != 1:
                return False

        # Check forbidden Y: column must have Y[t]=0 for all forbidden periods
        for t in forbidden_y.get(i, set()):
            if col.y.get(t, 0) == 1:
                return False

        # Check forced Z: column must have Z[t,u]=1 for all forced arcs
        for t, u in forced_z.get(i, set()):
            if col.z.get((t, u), 0) != 1:
                return False

        # Check forbidden Z: column must have Z[t,u]=0 for all forbidden arcs
        for t, u in forbidden_z.get(i, set()):
            if col.z.get((t, u), 0) == 1:
                return False

        return True

    m = gp.Model("RMP")
    m.Params.OutputFlag = 0

    BIG_M = 1e6  # Penalty for capacity violation (ensures initial feasibility)

    # Variables: λ[i][k] for each item i, valid column k
    lam = {}
    valid_col_indices = {}  # item -> list of valid column indices

    total_cols = 0
    valid_cols = 0
    for i, cols in columns.items():
        valid_indices = [k for k, col in enumerate(cols) if column_is_valid(col)]
        valid_col_indices[i] = valid_indices
        lam[i] = [m.addVar(lb=0, name=f"lam_{i}_{k}") for k in valid_indices]
        total_cols += len(cols)
        valid_cols += len(valid_indices)

    # Convexity constraints
    convex_cons = {}
    art_vars = {}  # Track artificial variables for objective
    items_with_no_cols = []
    for i in items:
        if i in lam and lam[i]:
            convex_cons[i] = m.addConstr(gp.quicksum(lam[i]) == 1, f"convex_{i}")
        else:
            items_with_no_cols.append(i)
            # No valid columns for this item - add artificial
            art_vars[i] = m.addVar(lb=0, name=f"art_{i}")
            convex_cons[i] = m.addConstr(art_vars[i] == 1, f"convex_{i}")

    # Capacity constraints with slack variables for initial feasibility
    cap_cons = {}
    cap_slack = {}
    for t in range(T):
        if capacity[t] > EPS:
            x_sum = gp.LinExpr()
            for i, cols in columns.items():
                valid_indices = valid_col_indices.get(i, [])
                for idx, orig_k in enumerate(valid_indices):
                    col = cols[orig_k]
                    x_sum += col.x.get(t, 0) * lam[i][idx]
            # Slack variable allows capacity violation with penalty
            cap_slack[t] = m.addVar(lb=0, name=f"cap_slack_{t}")
            cap_cons[t] = m.addConstr(x_sum <= capacity[t] + cap_slack[t], f"cap_{t}")

    # Y-aggregation constraints for forced Y (with slack for feasibility)
    y_fix_cons = {}
    y_fix_slack = {}
    for i, periods in forced_y.items():
        for t in periods:
            y_sum = gp.LinExpr()
            valid_indices = valid_col_indices.get(i, [])
            for idx, orig_k in enumerate(valid_indices):
                col = columns[i][orig_k]
                y_sum += col.y.get(t, 0) * lam[i][idx]
            # Slack for infeasibility (will be penalized)
            y_fix_slack[(i, t)] = m.addVar(lb=0, name=f"y_fix_slack_{i}_{t}")
            y_fix_cons[(i, t)] = m.addConstr(
                y_sum + y_fix_slack[(i, t)] == 1, f"y_fix_{i}_{t}"
            )

    # Z-aggregation constraints for forced Z (with slack for feasibility)
    z_fix_cons = {}
    z_fix_slack = {}
    for i, arcs in forced_z.items():
        for t, u in arcs:
            z_sum = gp.LinExpr()
            valid_indices = valid_col_indices.get(i, [])
            for idx, orig_k in enumerate(valid_indices):
                col = columns[i][orig_k]
                z_sum += col.z.get((t, u), 0) * lam[i][idx]
            # Slack for infeasibility
            z_fix_slack[(i, t, u)] = m.addVar(lb=0, name=f"z_fix_slack_{i}_{t}_{u}")
            z_fix_cons[(i, t, u)] = m.addConstr(
                z_sum + z_fix_slack[(i, t, u)] == 1, f"z_fix_{i}_{t}_{u}"
            )

    # LEFO No-Crossing Constraints
    # For t1, t2 where exp(t1) < exp(t2):
    # Z_agg[t1, u] + Z_agg[t2, u'] <= 1 for u in [t2, u'-1]
    # This ensures LEFO rule: earlier-expiring production must be used first
    lefo_cons = {}  # (i, t1, t2, u, up) -> constraint
    for i, item_data in items.items():
        shelf_seq = item_data.get("shelf_seq", [T] * T)
        demand = item_data.get("demand", [0] * T)
        demand_periods = [u for u in range(T) if demand[u] > EPS]

        if not demand_periods:
            continue

        # Collect all production periods used in columns for this item
        prod_periods = set()
        valid_indices = valid_col_indices.get(i, [])
        for idx, orig_k in enumerate(valid_indices):
            col = columns[i][orig_k]
            for t in col.y.keys():
                if col.y.get(t, 0) == 1:
                    prod_periods.add(t)

        if len(prod_periods) < 2:
            continue

        # Compute expiry for each production period
        expiry = {t: t + shelf_seq[t] for t in prod_periods}

        # Sort production periods by expiry
        prod_sorted = sorted(prod_periods, key=lambda t: expiry[t])

        # Add no-crossing constraints
        for a in range(len(prod_sorted)):
            t1 = prod_sorted[a]
            v1 = expiry[t1]

            for b in range(a + 1, len(prod_sorted)):
                t2 = prod_sorted[b]
                v2 = expiry[t2]

                if v1 >= v2:
                    continue  # Only when v1 < v2 (t1 expires before t2)

                # Get demands that t2 can serve
                t2_demands = [u for u in demand_periods if t2 <= u < v2]

                for up in t2_demands:
                    # Get demands in [t2, up-1] that t1 can serve
                    for u in demand_periods:
                        if t2 <= u <= up - 1 and t1 <= u < v1:
                            # Build Z_agg[t1, u] and Z_agg[t2, up]
                            z1_sum = gp.LinExpr()
                            z2_sum = gp.LinExpr()

                            for idx, orig_k in enumerate(valid_indices):
                                col = columns[i][orig_k]
                                z1_sum += col.z.get((t1, u), 0) * lam[i][idx]
                                z2_sum += col.z.get((t2, up), 0) * lam[i][idx]

                            # No-crossing: Z[t1, u] + Z[t2, up] <= 1
                            lefo_cons[(i, t1, t2, u, up)] = m.addConstr(
                                z1_sum + z2_sum <= 1,
                                name=f"lefo_{i}_{t1}_{t2}_{u}_{up}",
                            )

    # Objective: column costs + Big-M penalties for artificial/slack variables
    obj_expr = gp.LinExpr()
    for i, cols in columns.items():
        valid_indices = valid_col_indices.get(i, [])
        for idx, orig_k in enumerate(valid_indices):
            col = cols[orig_k]
            obj_expr += col.cost * lam[i][idx]
    # Add artificial variable penalties
    for art_var in art_vars.values():
        obj_expr += BIG_M * art_var
    # Add capacity slack penalties
    for slack_var in cap_slack.values():
        obj_expr += BIG_M * slack_var
    # Add Y-fix slack penalties
    for slack_var in y_fix_slack.values():
        obj_expr += BIG_M * slack_var
    # Add Z-fix slack penalties
    for slack_var in z_fix_slack.values():
        obj_expr += BIG_M * slack_var
    m.setObjective(obj_expr, GRB.MINIMIZE)

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return None, {}, {}, {}, {}, {}, {}, 0, 0, []

    # Extract duals
    mu = {i: convex_cons[i].Pi for i in convex_cons}
    rho = {t: cap_cons[t].Pi for t in cap_cons}
    sig_y = {key: con.Pi for key, con in y_fix_cons.items()}
    tau_z_fix = {key: con.Pi for key, con in z_fix_cons.items()}

    # Extract LEFO constraint duals
    # tau_lefo[(i, t1, t2, u, up)] is the dual of Z[t1,u] + Z[t2,up] <= 1
    # This dual provides incentive for pricing to generate LEFO-compliant columns
    tau_lefo = {key: con.Pi for key, con in lefo_cons.items()}

    # Extract lambda values (map back to original column indices)
    lam_vals = {}
    for i, cols in columns.items():
        lam_vals[i] = {}
        valid_indices = valid_col_indices.get(i, [])
        for idx, orig_k in enumerate(valid_indices):
            lam_vals[i][orig_k] = lam[i][idx].X

    return (
        m.ObjVal,
        mu,
        rho,
        lam_vals,
        sig_y,
        tau_z_fix,
        tau_lefo,
        valid_cols,
        total_cols,
        items_with_no_cols,
    )


def compute_y_aggregates(
    items: Dict[int, dict],
    columns: Dict[int, List[ZIOColumn]],
    lam_vals: Dict[int, Dict[int, float]],
    T: int,
    forced_y: Dict[int, Set[int]] = None,
    forbidden_y: Dict[int, Set[int]] = None,
) -> Dict[Tuple[int, int], float]:
    """
    Compute Y_agg[i, t] = Σ_k y_k[t] * λ_k for each item and period.

    Note: forced Y values are always 1, forbidden are always 0.
    """
    forced_y = forced_y or {}
    forbidden_y = forbidden_y or {}

    y_agg = {}
    for i in items:
        for t in range(T):
            # Check if fixed by branching
            if t in forced_y.get(i, set()):
                y_agg[(i, t)] = 1.0
                continue
            if t in forbidden_y.get(i, set()):
                y_agg[(i, t)] = 0.0
                continue

            # Compute from columns
            val = 0.0
            if i in columns and i in lam_vals:
                for k, col in enumerate(columns[i]):
                    lam_k = lam_vals[i].get(k, 0)
                    if lam_k > EPS:
                        val += col.y.get(t, 0) * lam_k
            y_agg[(i, t)] = val
    return y_agg


def compute_x_aggregates(
    items: Dict[int, dict],
    columns: Dict[int, List[ZIOColumn]],
    lam_vals: Dict[int, Dict[int, float]],
    T: int,
) -> Dict[int, float]:
    """
    Compute X_agg[t] = Σ_i Σ_k x_k[t] * λ_k for each period (total production).
    """
    x_agg = {}
    for t in range(T):
        val = 0.0
        for i, cols in columns.items():
            if i in lam_vals:
                for k, col in enumerate(cols):
                    val += col.x.get(t, 0) * lam_vals[i].get(k, 0)
        x_agg[t] = val
    return x_agg


def compute_true_objective_and_validate(
    items: Dict[int, dict],
    columns: Dict[int, List[ZIOColumn]],
    lam_vals: Dict[int, Dict[int, float]],
    capacity: List[float],
    T: int,
    forced_y: Dict[int, Set[int]] = None,
    forbidden_y: Dict[int, Set[int]] = None,
) -> Tuple[Optional[float], bool, str]:
    """
    Compute the true objective from X and Y aggregates and validate feasibility.

    Returns: (true_objective, is_feasible, error_message)
    - true_objective: Actual cost computed from X_agg and Y_agg (None if infeasible)
    - is_feasible: True if solution satisfies all constraints
    - error_message: Description of any infeasibility
    """
    forced_y = forced_y or {}
    forbidden_y = forbidden_y or {}

    # Compute aggregates
    y_agg = compute_y_aggregates(items, columns, lam_vals, T, forced_y, forbidden_y)
    x_agg_total = compute_x_aggregates(items, columns, lam_vals, T)

    # Validate capacity constraints
    for t in range(T):
        if x_agg_total.get(t, 0) > capacity[t] + EPS:
            return (
                None,
                False,
                f"Capacity violation at t={t}: {x_agg_total.get(t, 0):.2f} > {capacity[t]:.2f}",
            )

    # Compute true objective: setup costs + variable costs + holding costs
    total_cost = 0.0

    # Per-item costs
    for i, item_data in items.items():
        demand = item_data["demand"]
        setup = item_data["setup"]
        h = item_data["h"]
        c_var = item_data["c_var"]
        shelf_seq = item_data.get("shelf_seq", [T] * T)

        # Compute X_agg for this item
        x_agg_item = {}
        for t in range(T):
            x_val = 0.0
            if i in columns and i in lam_vals:
                for k, col in enumerate(columns[i]):
                    x_val += col.x.get(t, 0) * lam_vals[i].get(k, 0)
            if x_val > EPS:
                x_agg_item[t] = x_val

        # Setup costs: for each period with Y=1
        for t in range(T):
            y_val = y_agg.get((i, t), 0)
            if y_val > 0.5:  # Integer Y = 1
                total_cost += setup[t]

        # Variable and holding costs: for each production period
        for t, x_val in x_agg_item.items():
            # Variable cost
            total_cost += c_var[t] * x_val

            # Holding cost: need to determine which demands are served by production at t
            # This is approximate - we compute holding cost based on production quantity
            # For exact calculation, we'd need Z values, but this gives a lower bound
            # Actually, let's compute it more accurately from columns
            if i in columns and i in lam_vals:
                for k, col in enumerate(columns[i]):
                    lam_k = lam_vals[i].get(k, 0)
                    if lam_k < EPS:
                        continue
                    # Add column cost (which includes holding) weighted by lambda
                    total_cost += col.cost * lam_k
                    # But subtract setup and variable costs we already counted
                    for prod_t in col.y.keys():
                        if col.y.get(prod_t, 0) == 1:
                            total_cost -= setup[prod_t] * lam_k
                    for prod_t, prod_qty in col.x.items():
                        total_cost -= c_var[prod_t] * prod_qty * lam_k
                break  # Only count once per item

    return total_cost, True, ""


# COLUMN GENERATION LOOP


def column_generation_loop(
    items: Dict[int, dict],
    columns: Dict[int, List[ZIOColumn]],
    capacity: List[float],
    T: int,
    forced_y: Dict[int, Set[int]],
    forbidden_y: Dict[int, Set[int]],
    forced_z: Dict[int, Set[Tuple[int, int]]],
    forbidden_z: Dict[int, Set[Tuple[int, int]]],
    node_id: int,
    logger: Optional[BnPLogger],
    verbose: bool,
    max_cg_iters: int = 100,
) -> Tuple[Optional[float], Dict[int, Dict[int, float]], int, int]:
    """
    Run column generation until no negative reduced cost columns exist.

    Returns: (lp_bound, lam_vals, cg_iterations, columns_added)
    """
    total_cols_added = 0

    for cg_iter in range(max_cg_iters):
        # Solve RMP (with column filtering based on branching constraints)
        result = solve_rmp_with_duals(
            items, columns, capacity, T, forced_y, forbidden_y, forced_z, forbidden_z
        )

        if result[0] is None:
            return None, {}, cg_iter, total_cols_added

        (
            obj,
            mu,
            rho,
            lam_vals,
            sig_y,
            tau_z_fix,
            tau_lefo,
            valid_cols,
            total_pool_cols,
            items_no_cols,
        ) = result

        # Log column filtering info on first iteration
        if cg_iter == 0 and verbose and total_pool_cols > valid_cols:
            print(
                f"    Column filtering: {valid_cols}/{total_pool_cols} valid (filtered {total_pool_cols - valid_cols})"
            )
            if items_no_cols:
                print(f"    Items with NO valid columns: {items_no_cols}")

        # Pricing for each item
        min_rc = 0.0
        iter_cols_added = 0

        for i, item_data in items.items():
            demand = item_data["demand"]
            setup = item_data["setup"]
            h = item_data["h"]
            c_var = item_data["c_var"]
            shelf_seq = item_data.get("shelf_seq", [T] * T)

            forced = forced_y.get(i, set())
            forbidden = forbidden_y.get(i, set())
            forced_arcs = forced_z.get(i, set())
            forbidden_arcs = forbidden_z.get(i, set())

            # Extract tau duals for this item
            sig_y_item = {t: sig_y.get((i, t), 0.0) for t in forced}
            tau_z_item = {(t, u): tau_z_fix.get((i, t, u), 0.0) for t, u in forced_arcs}

            # Add LEFO duals to tau_z_item
            # tau_lefo[(i, t1, t2, u, up)] is dual of Z[t1,u] + Z[t2,up] <= 1
            # For arc (t, u), sum duals from all LEFO constraints involving this arc
            # LEFO duals are <= 0, so -dual is the discount for using the arc
            for key, dual in tau_lefo.items():
                if key[0] != i:
                    continue
                _, t1, t2, u, up = key
                # Dual for early arc (t1, u)
                if (t1, u) not in tau_z_item:
                    tau_z_item[(t1, u)] = 0.0
                tau_z_item[(t1, u)] -= dual  # -dual because <= constraint
                # Dual for later arc (t2, up)
                if (t2, up) not in tau_z_item:
                    tau_z_item[(t2, up)] = 0.0
                tau_z_item[(t2, up)] -= dual  # -dual because <= constraint

            new_cols = price_zio_column_dp(
                item_id=i,
                demand=demand,
                setup=setup,
                h=h,
                c_var=c_var,
                mu=mu.get(i, 0.0),
                rho=rho,
                shelf_seq=shelf_seq,
                sig_y=sig_y_item,
                tau_z=tau_z_item,
                max_columns=MAX_COLUMNS_PER_ITEM,
                forced_y=forced,
                forbidden_y=forbidden,
                forced_z=forced_arcs,
                forbidden_z=forbidden_arcs,
            )

            if new_cols:
                # Add new columns and track min_rc only for NON-duplicate columns
                existing_sigs = {col.signature() for col in columns.get(i, [])}
                for col, rc in new_cols:
                    if col.signature() not in existing_sigs:
                        # Only count reduced cost for new (non-duplicate) columns
                        if rc < min_rc:
                            min_rc = rc
                        if i not in columns:
                            columns[i] = []
                        columns[i].append(col)
                        existing_sigs.add(col.signature())
                        iter_cols_added += 1

        total_cols_added += iter_cols_added
        total_cols = sum(len(cols) for cols in columns.values())

        if logger:
            logger.log_cg_iter(
                CGIterLogEntry(
                    node_id=node_id,
                    cg_iter=cg_iter,
                    rmp_obj=obj,
                    num_columns=total_cols,
                    min_rc=min_rc,
                    columns_added=iter_cols_added,
                )
            )

        if verbose:
            print(
                f"    CG iter {cg_iter}: obj={obj:.2f}, cols={total_cols}, "
                f"added={iter_cols_added}, min_rc={min_rc:.4f}"
            )

        if min_rc >= -EPS:
            # Converged
            return obj, lam_vals, cg_iter + 1, total_cols_added

    # If Max iterations reached as fallback don't update tau and sig duals (this doesn't happen anymore after testing)
    result = solve_rmp_with_duals(
        items, columns, capacity, T, forced_y, forbidden_y, forced_z, forbidden_z
    )
    obj = result[0]
    lam_vals = result[3]
    return obj, lam_vals, max_cg_iters, total_cols_added


# BRANCHING


def is_y_integer(
    y_agg: Dict[Tuple[int, int], float],
    tol: float = 1e-4,
) -> bool:
    """Check if all Y_agg values are integer."""
    for (i, t), val in y_agg.items():
        if tol < val < 1 - tol:
            return False
    return True


def find_most_fractional_y(
    y_agg: Dict[Tuple[int, int], float],
    items: Dict[int, dict],
    forced_y: Dict[int, Set[int]],
    forbidden_y: Dict[int, Set[int]],
    tol: float = 1e-4,
) -> Optional[Tuple[int, int, float]]:
    """
    Find the most fractional Y[i, t] value.

    Returns (item_id, period, value) or None if all integer.
    Uses setup cost weighting for better branching.
    """
    best = None
    best_score = -1

    for (i, t), val in y_agg.items():
        # Skip already fixed
        if i in forced_y and t in forced_y[i]:
            continue
        if i in forbidden_y and t in forbidden_y[i]:
            continue

        # Skip integer values
        if val < tol or val > 1 - tol:
            continue

        # Score: fractionality * setup cost weight
        frac = 4 * min(val, 1 - val)  # Max at 0.5
        setup_cost = items[i]["setup"][t] if i in items else 0
        max_setup = max(items[i]["setup"]) if i in items else 1
        cost_weight = 0.3 + 0.7 * (setup_cost / max_setup)

        score = frac * cost_weight

        if score > best_score:
            best_score = score
            best = (i, t, val)

    return best


def compute_z_aggregates(
    items: Dict[int, dict],
    columns: Dict[int, List[ZIOColumn]],
    lam_vals: Dict[int, Dict[int, float]],
    T: int,
    forced_z: Dict[int, Set[Tuple[int, int]]] = None,
    forbidden_z: Dict[int, Set[Tuple[int, int]]] = None,
) -> Dict[Tuple[int, int, int], float]:
    """
    Compute Z_agg[i, t, u] = Σ_k z_k[t,u] * λ_k for each item and arc.

    Returns dict with (item, t, u) -> aggregate value.
    """
    forced_z = forced_z or {}
    forbidden_z = forbidden_z or {}

    z_agg = {}
    for i in items:
        # Collect all arcs across columns
        all_arcs = set()
        for col in columns.get(i, []):
            for t, u in col.z.keys():
                all_arcs.add((t, u))

        for t, u in all_arcs:
            # Check if fixed by branching
            if (t, u) in forced_z.get(i, set()):
                z_agg[(i, t, u)] = 1.0
                continue
            if (t, u) in forbidden_z.get(i, set()):
                z_agg[(i, t, u)] = 0.0
                continue

            # Compute aggregate
            val = 0.0
            if i in lam_vals:
                for k, lam in lam_vals[i].items():
                    if lam > EPS:
                        cols = columns.get(i, [])
                        if k < len(cols):
                            val += cols[k].z.get((t, u), 0) * lam
            z_agg[(i, t, u)] = val

    return z_agg


def is_z_integer(
    z_agg: Dict[Tuple[int, int, int], float],
    tol: float = 1e-4,
) -> bool:
    """Check if all Z_agg values are integer."""
    for (i, t, u), val in z_agg.items():
        if tol < val < 1 - tol:
            return False
    return True


def find_most_fractional_z(
    z_agg: Dict[Tuple[int, int, int], float],
    items: Dict[int, dict],
    forced_z: Dict[int, Set[Tuple[int, int]]],
    forbidden_z: Dict[int, Set[Tuple[int, int]]],
    tol: float = 1e-4,
) -> Optional[Tuple[int, int, int, float]]:
    """
    Find the most fractional Z[i, t, u] value.

    Returns (item_id, t, u, value) or None if all integer.
    """
    best = None
    best_score = -1

    for (i, t, u), val in z_agg.items():
        # Skip already fixed
        if (t, u) in forced_z.get(i, set()):
            continue
        if (t, u) in forbidden_z.get(i, set()):
            continue

        # Skip integer values
        if val < tol or val > 1 - tol:
            continue

        # Score: fractionality (max at 0.5)
        frac = 4 * min(val, 1 - val)
        score = frac

        if score > best_score:
            best_score = score
            best = (i, t, u, val)

    return best


# COLUMN SELECTION HEURISTIC (finds feasible integer solution from column pool)
# Selects one column per item that together satisfy capacity constraints


def column_selection_heuristic(
    items: Dict[int, dict],
    columns: Dict[int, List[ZIOColumn]],
    lam_vals: Dict[int, Dict[int, float]],
    capacity: List[float],
    T: int,
    forced_y: Dict[int, Set[int]],
    forbidden_y: Dict[int, Set[int]],
) -> Tuple[Optional[float], Optional[Dict[int, ZIOColumn]]]:
    """
    Select one column per item from the pool that together satisfy capacity.
    Uses a greedy approach: for each item, pick the column with highest lambda value
    that satisfies branching constraints, checking capacity as we go.
    """
    selected_columns = {}
    capacity_used = [0.0] * T
    total_cost = 0.0

    # Sort items by "difficulty" - items with fewer valid columns first
    def count_valid_columns(i):
        count = 0
        for col in columns.get(i, []):
            if _column_satisfies_branching(
                col, forced_y.get(i, set()), forbidden_y.get(i, set())
            ):
                count += 1
        return count

    item_order = sorted(items.keys(), key=count_valid_columns)

    for i in item_order:
        item_cols = columns.get(i, [])
        if not item_cols:
            return None, None  # No columns for this item

        # Get lambda values for this item
        item_lam = lam_vals.get(i, {})

        # Sort columns by lambda value (highest first)
        col_with_lam = []
        for k, col in enumerate(item_cols):
            if _column_satisfies_branching(
                col, forced_y.get(i, set()), forbidden_y.get(i, set())
            ):
                lam_val = item_lam.get(k, 0)
                col_with_lam.append((lam_val, k, col))

        col_with_lam.sort(key=lambda x: x[0], reverse=True)

        # Try columns in order of lambda value
        found = False
        for lam_val, k, col in col_with_lam:
            # Check if this column fits in remaining capacity
            fits = True
            for t in range(T):
                x_t = col.x.get(t, 0)
                if capacity_used[t] + x_t > capacity[t] + EPS:
                    fits = False
                    break

            if fits:
                # Use this column
                selected_columns[i] = col
                for t in range(T):
                    capacity_used[t] += col.x.get(t, 0)
                total_cost += col.cost
                found = True
                break

        if not found:
            return None, None  # No feasible column found for this item

    return total_cost, selected_columns


def _column_satisfies_branching(
    col: ZIOColumn,
    forced_y: Set[int],
    forbidden_y: Set[int],
) -> bool:
    """Check if a column satisfies branching constraints."""
    # Check forced Y: column must have Y=1 for forced periods
    for t in forced_y:
        if col.y.get(t, 0) != 1:
            return False

    # Check forbidden Y: column must have Y=0 for forbidden periods
    for t in forbidden_y:
        if col.y.get(t, 0) == 1:
            return False

    return True


# DIVE HEURISTIC (controlled by ENABLE_DIVE_HEURISTIC)
# This is critical for convergence - finds integer solutions by fixing Y values


def try_dive(
    items: Dict[int, dict],
    columns: Dict[int, List[ZIOColumn]],
    y_agg: Dict[Tuple[int, int], float],
    capacity: List[float],
    T: int,
    base_forced_y: Dict[int, Set[int]],
    base_forbidden_y: Dict[int, Set[int]],
    base_forced_z: Dict[int, Set[Tuple[int, int]]],
    base_forbidden_z: Dict[int, Set[Tuple[int, int]]],
) -> Tuple[Optional[float], Optional[Dict[int, Dict[int, float]]]]:
    """
    Try to find an integer solution by fixing Y values based on LP solution.
    This is critical for B&P convergence - provides upper bounds for pruning.

    Strategy: Try multiple threshold strategies to find feasible integer solution.
    Returns (objective, lambda_values) or (None, None) if no feasible solution found.
    """
    # Strategy 1: Force ALL positive Y to 1 (most likely to be feasible)
    result = _try_dive_with_threshold(
        items,
        columns,
        y_agg,
        capacity,
        T,
        base_forced_y,
        base_forbidden_y,
        base_forced_z,
        base_forbidden_z,
        threshold=0.0,
        force_all_positive=True,
    )
    if result[0] is not None:
        return result

    # Strategy 2-4: Try different thresholds
    for threshold in [0.5, 0.3, 0.1]:
        result = _try_dive_with_threshold(
            items,
            columns,
            y_agg,
            capacity,
            T,
            base_forced_y,
            base_forbidden_y,
            base_forced_z,
            base_forbidden_z,
            threshold=threshold,
            force_all_positive=False,
        )
        if result[0] is not None:
            return result

    return None, None


def _try_dive_with_threshold(
    items: Dict[int, dict],
    columns: Dict[int, List[ZIOColumn]],
    y_agg: Dict[Tuple[int, int], float],
    capacity: List[float],
    T: int,
    base_forced_y: Dict[int, Set[int]],
    base_forbidden_y: Dict[int, Set[int]],
    base_forced_z: Dict[int, Set[Tuple[int, int]]],
    base_forbidden_z: Dict[int, Set[Tuple[int, int]]],
    threshold: float,
    force_all_positive: bool = False,
) -> Tuple[Optional[float], Optional[Dict[int, Dict[int, float]]]]:
    """
    Try dive with a specific threshold.
    If force_all_positive=True, force Y=1 for ALL Y > 0.
    Otherwise force Y=1 for Y >= threshold, Y=0 for Y < threshold.
    """
    forced_y = {i: set(s) for i, s in base_forced_y.items()}
    forbidden_y = {i: set(s) for i, s in base_forbidden_y.items()}

    for (item_id, t), val in y_agg.items():
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

    # Solve RMP with fixed Y values
    result = solve_rmp_with_duals(
        items=items,
        columns=columns,
        capacity=capacity,
        T=T,
        forced_y=forced_y,
        forbidden_y=forbidden_y,
        forced_z=base_forced_z,
        forbidden_z=base_forbidden_z,
    )

    if result[0] is None:
        return None, None

    obj = result[0]
    mu = result[1]
    rho = result[2]
    lam_vals = result[3]
    sig_y = result[4]
    tau_z_fix = result[5]
    # tau_lefo = result[6]  # Not needed for dive

    if obj is None or not math.isfinite(obj):
        return None, None

    # Check if solution uses artificial variables (BIG_M penalty makes obj very large)
    # A reasonable cost bound: worst case is every item has setup + prod in every period
    # Max reasonable cost = items * T * (max_setup + max_demand * max_var_cost + max_h * max_demand * T)
    max_setup = max(max(it.get("setup", [0])) for it in items.values())
    max_demand = max(max(it.get("demand", [0])) for it in items.values())
    max_c_var = max(max(it.get("c_var", [1])) for it in items.values())
    max_h = max(max(it.get("h", [0.1])) for it in items.values())
    reasonable_bound = (
        len(items) * T * (max_setup + max_demand * (max_c_var + max_h * T))
    )

    if obj > reasonable_bound:
        # Solution uses artificial variables - not truly feasible
        return None, None

    # Check if truly feasible (compute Y aggregates and check integrality)
    y_dive = compute_y_aggregates(items, columns, lam_vals, T, forced_y, forbidden_y)

    if is_y_integer(y_dive):
        # Compute true objective from column costs
        true_obj = 0.0
        for i in items:
            if i in columns and i in lam_vals:
                for k, col in enumerate(columns[i]):
                    lam_k = lam_vals[i].get(k, 0)
                    if lam_k > EPS:
                        true_obj += col.cost * lam_k

        # Validate capacity constraints
        x_agg_total = compute_x_aggregates(items, columns, lam_vals, T)
        for t in range(T):
            if x_agg_total.get(t, 0) > capacity[t] + EPS:
                # Capacity violation - not feasible
                return None, None

        return true_obj, lam_vals

    return None, None


# PRIMAL HEURISTIC


def primal_heuristic_round_y(
    items: Dict[int, dict],
    columns: Dict[int, List[ZIOColumn]],
    lam_vals: Dict[int, Dict[int, float]],
    capacity: List[float],
    T: int,
    forced_y: Dict[int, Set[int]],
    forbidden_y: Dict[int, Set[int]],
    threshold: float = 0.5,
) -> Tuple[Optional[float], Optional[Dict[int, ZIOColumn]]]:
    """
    Improved primal heuristic: capacity-aware rounding with multi-pass strategy.

    Strategy:
    1. Compute Y aggregates and round with priority to high fractional values
    2. For each item, build production plan greedily considering capacity
    3. If capacity violated, try to shift production to less loaded periods
    4. Return feasible solution if found

    Returns (cost, columns_dict) or (None, None) if infeasible.
    """
    # Compute Y aggregates with priority scoring
    y_candidates = {}  # item -> list of (t, y_val, priority_score)

    for i in items:
        y_candidates[i] = []

        # Add forced Y periods with high priority
        for t in forced_y.get(i, set()):
            y_candidates[i].append((t, 1.0, 1000.0))

        # Compute aggregates and score
        for t in range(T):
            if t in forbidden_y.get(i, set()):
                continue
            if t in forced_y.get(i, set()):
                continue

            # Compute Y_agg[i,t]
            y_val = 0.0
            for k, col in enumerate(columns.get(i, [])):
                y_val += col.y.get(t, 0) * lam_vals.get(i, {}).get(k, 0)

            if y_val > EPS:
                # Priority: higher fractional values get higher priority
                # Also consider setup cost (cheaper setups preferred)
                setup_cost = items[i]["setup"][t]
                max_setup = max(items[i]["setup"]) if items[i]["setup"] else 1
                cost_factor = (
                    1.0 - (setup_cost / max_setup) * 0.3
                )  # Prefer cheaper setups
                priority = y_val * cost_factor
                y_candidates[i].append((t, y_val, priority))

        # Sort by priority (highest first)
        y_candidates[i].sort(key=lambda x: x[2], reverse=True)

    # Round Y values: take top candidates until threshold coverage
    y_rounded = {}
    for i in items:
        y_rounded[i] = set()
        # Always include forced Y
        for t in forced_y.get(i, set()):
            y_rounded[i].add(t)
        # Add candidates above threshold or top N
        for t, y_val, _ in y_candidates[i]:
            if t in y_rounded[i]:
                continue
            if y_val >= threshold:
                y_rounded[i].add(t)

    # Track capacity usage per period
    capacity_used = [0.0] * T

    # Build production plans for each item (capacity-aware)
    heur_columns = {}
    total_cost = 0.0

    for i, item_data in items.items():
        demand = item_data["demand"]
        setup = item_data["setup"]
        h = item_data["h"]
        c_var = item_data["c_var"]
        shelf_seq = item_data.get("shelf_seq", [T] * T)

        allowed_y = y_rounded[i].copy()
        y = {}
        x = {}
        z = {}
        cost = 0.0

        demand_periods = [u for u in range(T) if demand[u] > EPS]

        # CRITICAL: Handle shelf life 0 periods first (they are very restrictive)
        # Shelf life 0 means production can only be used in the same period
        # Sort demand periods: shelf life 0 periods first, then others
        if ENABLE_SHELF_LIFE_PRIORITIZATION:

            def get_priority(u):
                # Check if any production period can serve u with shelf life 0
                for t in range(u, -1, -1):
                    if t in forbidden_y.get(i, set()):
                        continue
                    if shelf_seq[t] == 0 and u == t:
                        return 0  # Highest priority: shelf life 0, same period
                    if shelf_seq[t] > 0 and u <= t + shelf_seq[t]:
                        return 1  # Normal priority
                return 2  # Low priority: might be infeasible

            demand_periods.sort(key=get_priority)

        # First pass: assign demands to production periods (capacity-aware)
        for u in demand_periods:
            best_t = None
            best_score = -1

            # Try allowed Y periods first
            for t in sorted(allowed_y):
                # CRITICAL: Shelf life 0 means production can only be used in same period
                if shelf_seq[t] == 0 and u != t:
                    continue
                if t > u or u > t + shelf_seq[t]:
                    continue
                # Check capacity
                if capacity_used[t] + demand[u] > capacity[t] + EPS:
                    continue
                # Score: prefer periods with more remaining capacity
                remaining_cap = capacity[t] - capacity_used[t]
                score = remaining_cap / max(capacity[t], 1.0)
                if score > best_score:
                    best_score = score
                    best_t = t

            # If no allowed Y works, try adding a new production period
            if best_t is None:
                for t in range(u, -1, -1):
                    if t in forbidden_y.get(i, set()):
                        continue
                    # CRITICAL: Shelf life 0 means production can only be used in same period
                    if shelf_seq[t] == 0 and u != t:
                        continue
                    if u > t + shelf_seq[t]:
                        continue
                    if capacity_used[t] + demand[u] > capacity[t] + EPS:
                        continue
                    remaining_cap = capacity[t] - capacity_used[t]
                    score = remaining_cap / max(capacity[t], 1.0)
                    if score > best_score:
                        best_score = score
                        best_t = t
                if best_t is not None:
                    allowed_y.add(best_t)

            if best_t is None:
                # Last resort: find any period that can serve u (even if over capacity)
                # We'll check feasibility at the end
                for t in range(u, -1, -1):
                    if t in forbidden_y.get(i, set()):
                        continue
                    # CRITICAL: Shelf life 0 means production can only be used in same period
                    if shelf_seq[t] == 0 and u != t:
                        continue
                    if u > t + shelf_seq[t]:
                        continue
                    best_t = t
                    if best_t not in allowed_y:
                        allowed_y.add(best_t)
                    break

                if best_t is None:
                    return None, None

            # Assign production
            if best_t not in y:
                y[best_t] = 1
                x[best_t] = 0
                cost += setup[best_t]

            x[best_t] = x.get(best_t, 0) + demand[u]
            z[(best_t, u)] = 1
            capacity_used[best_t] += demand[u]

            # Variable cost
            cost += c_var[best_t] * demand[u]

            # Holding cost
            for r in range(best_t, u):
                cost += h[r] * demand[u]

        heur_columns[i] = ZIOColumn(item_id=i, y=y, x=x, z=z, cost=cost)
        total_cost += cost

    # Final capacity check
    for t in range(T):
        if capacity_used[t] > capacity[t] + EPS:
            return None, None

    return total_cost, heur_columns


def greedy_heuristic_simple(
    items: Dict[int, dict],
    capacity: List[float],
    T: int,
    forced_y: Dict[int, Set[int]],
    forbidden_y: Dict[int, Set[int]],
) -> Tuple[Optional[float], Optional[Dict[int, ZIOColumn]]]:
    """
    Simple greedy heuristic: lot-for-lot with capacity awareness.
    More robust but potentially less optimal than rounding heuristic.
    """
    heur_columns = {}
    total_cost = 0.0
    capacity_used = [0.0] * T

    for i, item_data in items.items():
        demand = item_data["demand"]
        setup = item_data["setup"]
        h = item_data["h"]
        c_var = item_data["c_var"]
        shelf_seq = item_data.get("shelf_seq", [T] * T)

        y = {}
        x = {}
        z = {}
        cost = 0.0

        # Process demands: prioritize shelf life 0 periods (very restrictive)
        demand_periods = [u for u in range(T) if demand[u] > EPS]
        # Sort: shelf life 0 periods first (u where shelf_seq[u] == 0)
        if ENABLE_SHELF_LIFE_PRIORITIZATION:
            demand_periods.sort(key=lambda u: (0 if shelf_seq[u] == 0 else 1, u))

        for u in demand_periods:

            # Find best production period
            best_t = None
            best_cost = math.inf

            # Try forced Y periods first
            for t in forced_y.get(i, set()):
                # CRITICAL: Shelf life 0 means production can only be used in same period
                if shelf_seq[t] == 0 and u != t:
                    continue
                if t > u or u > t + shelf_seq[t]:
                    continue
                if capacity_used[t] + demand[u] > capacity[t] + EPS:
                    continue

                prod_cost = setup[t] + c_var[t] * demand[u]
                hold_cost = sum(h[r] for r in range(t, u)) * demand[u]
                total_prod_cost = prod_cost + hold_cost

                if total_prod_cost < best_cost:
                    best_cost = total_prod_cost
                    best_t = t

            # Try other periods
            if best_t is None:
                # First pass: try periods with available capacity
                for t in range(u, -1, -1):
                    if t in forbidden_y.get(i, set()):
                        continue
                    # CRITICAL: Shelf life 0 means production can only be used in same period
                    if shelf_seq[t] == 0 and u != t:
                        continue
                    if u > t + shelf_seq[t]:
                        continue
                    if capacity_used[t] + demand[u] > capacity[t] + EPS:
                        continue

                    prod_cost = setup[t] + c_var[t] * demand[u]
                    hold_cost = sum(h[r] for r in range(t, u)) * demand[u]
                    total_prod_cost = prod_cost + hold_cost

                    if total_prod_cost < best_cost:
                        best_cost = total_prod_cost
                        best_t = t

                # Second pass: if still no period found, allow capacity violation
                # (we'll check overall feasibility at the end)
                if best_t is None:
                    for t in range(u, -1, -1):
                        if t in forbidden_y.get(i, set()):
                            continue
                        # CRITICAL: Shelf life 0 means production can only be used in same period
                        if shelf_seq[t] == 0 and u != t:
                            continue
                        if u > t + shelf_seq[t]:
                            continue
                        # Allow capacity violation for now

                        prod_cost = setup[t] + c_var[t] * demand[u]
                        hold_cost = sum(h[r] for r in range(t, u)) * demand[u]
                        total_prod_cost = prod_cost + hold_cost

                        if total_prod_cost < best_cost:
                            best_cost = total_prod_cost
                            best_t = t

            if best_t is None:
                return None, None  # Infeasible

            # Assign production
            if best_t not in y:
                y[best_t] = 1
                x[best_t] = 0
                cost += setup[best_t]

            x[best_t] += demand[u]
            z[(best_t, u)] = 1
            capacity_used[best_t] += demand[u]
            cost += c_var[best_t] * demand[u]
            cost += sum(h[r] for r in range(best_t, u)) * demand[u]

        heur_columns[i] = ZIOColumn(item_id=i, y=y, x=x, z=z, cost=cost)
        total_cost += cost

    # Final capacity check
    # Allow small violations - they might be repairable later or acceptable
    # Only reject if violation is too large (more than 10% over capacity)
    max_violation = 0.0
    for t in range(T):
        if capacity[t] > EPS:
            violation_ratio = (capacity_used[t] - capacity[t]) / capacity[t]
            max_violation = max(max_violation, violation_ratio)
        elif capacity_used[t] > EPS:
            # Capacity is 0 but we're using it - definitely infeasible
            return None, None

    # If violation is too large, reject the solution
    if max_violation > 0.1:  # More than 10% over capacity
        return None, None

    return total_cost, heur_columns


# BRANCH-AND-PRICE MAIN LOOP


def solve_branch_and_price(
    items: Dict[int, dict],
    capacity: List[float],
    T: int,
    time_limit: float,
    logger: Optional[BnPLogger],
    verbose: bool,
) -> Tuple[float, float, Dict[int, List[ZIOColumn]], Dict[int, Dict[int, float]], bool]:
    """
    Solve using Branch-and-Price.

    Returns: (best_ub, best_lb, columns, lam_vals, time_limit_reached)

    Returns: (best_ub, best_lb, columns, lam_vals)
    """
    start_time = time.time()

    # Initialize columns with lot-for-lot for each item
    columns: Dict[int, List[ZIOColumn]] = {}

    for i, item_data in items.items():
        demand = item_data["demand"]
        setup = item_data["setup"]
        h = item_data["h"]
        c_var = item_data["c_var"]
        shelf_seq = item_data.get("shelf_seq", [T] * T)

        # Determine forbidden Y from zero capacity
        forbidden = {t for t in range(T) if capacity[t] <= EPS}

        col = generate_initial_column(
            item_id=i,
            demand=demand,
            setup=setup,
            h=h,
            c_var=c_var,
            shelf_seq=shelf_seq,
            forbidden_y=forbidden,
            forced_y=set(),
        )
        columns[i] = [col]

    # Create root node
    root = BranchNode(node_id=0, parent_id=None, depth=0)
    root.forbidden_y = {i: {t for t in range(T) if capacity[t] <= EPS} for i in items}

    # Priority queue: (priority, node_id, lb_estimate, node)
    # lb_estimate is the parent's LP bound (valid lower bound for the child)
    queue: List[Tuple[float, int, float, BranchNode]] = []
    heapq.heappush(queue, (0.0, 0, 0.0, root))

    node_counter = 1
    best_ub = math.inf
    best_lb = -math.inf
    best_lam = {}

    nodes_explored = 0
    time_limit_reached = False

    while queue:
        # Time limit check
        if time_limit > 0 and time.time() - start_time > time_limit:
            time_limit_reached = True
            if verbose:
                # Compute current gap before breaking
                current_lb = min(q[2] for q in queue) if queue else best_ub
                gap = (
                    (best_ub - current_lb) / max(abs(best_ub), 1e-10) * 100
                    if best_ub < math.inf
                    else float("inf")
                )
                print(f"\nTime limit reached after {nodes_explored} nodes")
                print(
                    f"  Current UB: {best_ub:.4f}, LB: {current_lb:.4f}, Gap: {gap:.2f}%"
                )
            break

        _, _, _, node = heapq.heappop(queue)
        nodes_explored += 1

        if verbose:
            print(
                f"\nNode {node.node_id} (depth={node.depth}, "
                f"incumbent={best_ub:.2f}, queue={len(queue)})"
            )

        # Run column generation
        lp_bound, lam_vals, cg_iters, cols_added = column_generation_loop(
            items=items,
            columns=columns,
            capacity=capacity,
            T=T,
            forced_y=node.forced_y,
            forbidden_y=node.forbidden_y,
            forced_z=node.forced_z,
            forbidden_z=node.forbidden_z,
            node_id=node.node_id,
            logger=logger,
            verbose=verbose,
        )

        if lp_bound is None:
            if verbose:
                print(f"  Node infeasible")
            if logger:
                logger.log_node(
                    NodeLogEntry(
                        node_id=node.node_id,
                        depth=node.depth,
                        lp_bound=math.inf,
                        incumbent=best_ub,
                        branch_item=None,
                        branch_t=None,
                        direction=None,
                        status="INFEASIBLE",
                        cg_iters=cg_iters,
                        columns_added=cols_added,
                    )
                )
            continue

        node.lp_bound = lp_bound

        # Update global lower bound
        if nodes_explored == 1:
            best_lb = lp_bound
            # DIVE at root node (controlled by ENABLE_DIVE_HEURISTIC)
            if best_ub >= math.inf and ENABLE_DIVE_HEURISTIC:
                if verbose:
                    print("  Trying dive at root...")
                y_agg_dive = compute_y_aggregates(
                    items, columns, lam_vals, T, node.forced_y, node.forbidden_y
                )
                dive_obj, dive_lam = try_dive(
                    items=items,
                    columns=columns,
                    y_agg=y_agg_dive,
                    capacity=capacity,
                    T=T,
                    base_forced_y=node.forced_y,
                    base_forbidden_y=node.forbidden_y,
                    base_forced_z=node.forced_z,
                    base_forbidden_z=node.forbidden_z,
                )
                if dive_obj is not None:
                    best_ub = dive_obj
                    best_lam = dive_lam
                    if verbose:
                        print(f"  *** DIVE found incumbent: {best_ub:.2f} ***")
                elif ENABLE_COLUMN_SELECTION:
                    # Dive failed - try column selection heuristic
                    if verbose:
                        print("  Dive failed, trying column selection...")
                    sel_cost, sel_cols = column_selection_heuristic(
                        items=items,
                        columns=columns,
                        lam_vals=lam_vals,
                        capacity=capacity,
                        T=T,
                        forced_y=node.forced_y,
                        forbidden_y=node.forbidden_y,
                    )
                    if sel_cost is not None:
                        best_ub = sel_cost
                        # Create lambda values for selected columns
                        best_lam = {}
                        for i, col in sel_cols.items():
                            for k, c in enumerate(columns[i]):
                                if c.signature() == col.signature():
                                    best_lam[i] = {k: 1.0}
                                    break
                        if verbose:
                            print(
                                f"  *** COLUMN SELECTION found incumbent: {best_ub:.2f} ***"
                            )
                    elif verbose:
                        print("  Column selection also failed")

        # Pruning by bound
        if lp_bound >= best_ub - EPS:
            if verbose:
                print(f"  Pruned by bound: {lp_bound:.2f} >= {best_ub:.2f}")
            if logger:
                logger.log_node(
                    NodeLogEntry(
                        node_id=node.node_id,
                        depth=node.depth,
                        lp_bound=lp_bound,
                        incumbent=best_ub,
                        branch_item=None,
                        branch_t=None,
                        direction=None,
                        status="PRUNED",
                        cg_iters=cg_iters,
                        columns_added=cols_added,
                    )
                )
            continue

        # Compute Y and Z aggregates (RAW - without forced/forbidden override for true integrality check)
        y_agg_raw = compute_y_aggregates(
            items, columns, lam_vals, T, {}, {}  # No forced/forbidden override
        )
        z_agg_raw = compute_z_aggregates(
            items, columns, lam_vals, T, {}, {}  # No forced/forbidden override
        )

        # Check integrality on RAW aggregates
        y_is_int = is_y_integer(y_agg_raw)
        z_is_int = is_z_integer(z_agg_raw)

        # =====================================================================
        # DIVE HEURISTIC (always runs - critical for convergence)
        # Run periodically to find integer solutions for pruning (controlled by flags)
        # =====================================================================
        if ENABLE_DIVE_HEURISTIC:
            dive_interval = 50 if best_ub >= math.inf else 150
            if (
                nodes_explored > 1
                and nodes_explored % dive_interval == 0
                and not y_is_int
            ):
                if verbose:
                    print(f"  Trying periodic dive at node {nodes_explored}...")
                dive_obj, dive_lam = try_dive(
                    items=items,
                    columns=columns,
                    y_agg=y_agg_raw,
                    capacity=capacity,
                    T=T,
                    base_forced_y=node.forced_y,
                    base_forbidden_y=node.forbidden_y,
                    base_forced_z=node.forced_z,
                    base_forbidden_z=node.forbidden_z,
                )
                if dive_obj is not None and dive_obj < best_ub:
                    best_ub = dive_obj
                    best_lam = dive_lam
                    if verbose:
                        print(f"  *** DIVE found new incumbent: {best_ub:.2f} ***")
                elif best_ub >= math.inf and ENABLE_COLUMN_SELECTION:
                    # Dive failed and no incumbent yet - try column selection
                    sel_cost, sel_cols = column_selection_heuristic(
                        items=items,
                        columns=columns,
                        lam_vals=lam_vals,
                        capacity=capacity,
                        T=T,
                        forced_y=node.forced_y,
                        forbidden_y=node.forbidden_y,
                    )
                    if sel_cost is not None and sel_cost < best_ub:
                        best_ub = sel_cost
                        best_lam = {}
                    for i, col in sel_cols.items():
                        for k, c in enumerate(columns[i]):
                            if c.signature() == col.signature():
                                best_lam[i] = {k: 1.0}
                                break
                    if verbose:
                        print(
                            f"  *** COLUMN SELECTION found incumbent: {best_ub:.2f} ***"
                        )

        # Integer solution: Y is integer (Z can be fractional in convex combination)
        # ZIO formulation: only Y needs to be binary
        if y_is_int:
            if verbose:
                print(f"  Integer Y solution found: {lp_bound:.2f}")

            # Validate feasibility and compute true objective
            # Check if solution uses artificial variables (BIG_M penalty)
            reasonable_bound = (
                len(items)
                * T
                * 1000000  # Very loose bound - if obj is reasonable, no artificial vars
            )

            if lp_bound > reasonable_bound:
                if verbose:
                    print(
                        f"  Rejecting solution: uses artificial variables (obj={lp_bound:.2f})"
                    )
            else:
                # Compute true objective from column costs
                true_obj = 0.0
                for i in items:
                    if i in columns and i in lam_vals:
                        for k, col in enumerate(columns[i]):
                            lam_k = lam_vals[i].get(k, 0)
                            if lam_k > EPS:
                                true_obj += col.cost * lam_k

                # Validate capacity constraints
                x_agg_total = compute_x_aggregates(items, columns, lam_vals, T)
                capacity_violated = False
                for t in range(T):
                    if x_agg_total.get(t, 0) > capacity[t] + EPS:
                        capacity_violated = True
                        if verbose:
                            print(f"  Rejecting solution: capacity violation at t={t}")
                        break

                if not capacity_violated and true_obj < best_ub:
                    best_ub = true_obj
                    best_lam = lam_vals
                    if verbose:
                        print(
                            f"  *** New incumbent: {best_ub:.2f} (LP bound was {lp_bound:.2f}) ***"
                        )

            if logger:
                logger.log_node(
                    NodeLogEntry(
                        node_id=node.node_id,
                        depth=node.depth,
                        lp_bound=lp_bound,
                        incumbent=best_ub,
                        branch_item=None,
                        branch_t=None,
                        direction=None,
                        status="INTEGER",
                        cg_iters=cg_iters,
                        columns_added=cols_added,
                    )
                )
            continue

        # Decide: branch on Y or Z (Y first to fix setup pattern, then Z)
        branch_on_z = False
        branch_var_y = None
        branch_var_z = None

        # Branch on Y first to fix the setup pattern
        # Only branch on Z if Y is already integer
        if not y_is_int:
            branch_var_y = find_most_fractional_y(
                y_agg_raw, items, node.forced_y, node.forbidden_y
            )

        if branch_var_y is None and not z_is_int:
            branch_var_z = find_most_fractional_z(
                z_agg_raw, items, node.forced_z, node.forbidden_z
            )
            if branch_var_z is not None:
                branch_on_z = True

        # Copy constraints from parent
        forced_y_copy, forbidden_y_copy, forced_z_copy, forbidden_z_copy = (
            node.copy_constraints()
        )

        if not branch_on_z and branch_var_y is not None:
            # Branch on Y
            branch_i, branch_t, branch_val = branch_var_y

            if verbose:
                print(f"  Branching on Y[{branch_i},{branch_t}] = {branch_val:.4f}")

            if logger:
                logger.log_node(
                    NodeLogEntry(
                        node_id=node.node_id,
                        depth=node.depth,
                        lp_bound=lp_bound,
                        incumbent=best_ub,
                        branch_item=branch_i,
                        branch_t=branch_t,
                        direction="BRANCH_Y",
                        status="BRANCHED",
                        cg_iters=cg_iters,
                        columns_added=cols_added,
                    )
                )

            # Child 1: Y[i,t] = 1 forced
            child1 = BranchNode(
                node_id=node_counter,
                parent_id=node.node_id,
                depth=node.depth + 1,
            )
            child1.forced_y = {i: set(s) for i, s in forced_y_copy.items()}
            child1.forbidden_y = {i: set(s) for i, s in forbidden_y_copy.items()}
            child1.forced_z = {i: set(s) for i, s in forced_z_copy.items()}
            child1.forbidden_z = {i: set(s) for i, s in forbidden_z_copy.items()}
            if branch_i not in child1.forced_y:
                child1.forced_y[branch_i] = set()
            child1.forced_y[branch_i].add(branch_t)
            node_counter += 1

            # Child 2: Y[i,t] = 0 forbidden
            child2 = BranchNode(
                node_id=node_counter,
                parent_id=node.node_id,
                depth=node.depth + 1,
            )
            child2.forced_y = {i: set(s) for i, s in forced_y_copy.items()}
            child2.forbidden_y = {i: set(s) for i, s in forbidden_y_copy.items()}
            child2.forced_z = {i: set(s) for i, s in forced_z_copy.items()}
            child2.forbidden_z = {i: set(s) for i, s in forbidden_z_copy.items()}
            if branch_i not in child2.forbidden_y:
                child2.forbidden_y[branch_i] = set()
            child2.forbidden_y[branch_i].add(branch_t)
            node_counter += 1

            # Add to queue
            # HYBRID SEARCH: Use DFS until incumbent found (controlled by USE_DFS_UNTIL_INCUMBENT)
            if USE_DFS_UNTIL_INCUMBENT and best_ub >= math.inf:
                # DFS mode: prioritize deeper nodes (negative depth)
                priority1 = -child1.depth
                priority2 = -child2.depth
            else:
                # Best-first mode: prioritize by LP bound
                priority1 = lp_bound
                priority2 = lp_bound
            heapq.heappush(queue, (priority1, child1.node_id, lp_bound, child1))
            heapq.heappush(queue, (priority2, child2.node_id, lp_bound, child2))

        elif branch_on_z and branch_var_z is not None:
            # Branch on Z
            branch_i, branch_t, branch_u, branch_val = branch_var_z

            if verbose:
                print(
                    f"  Branching on Z[{branch_i},{branch_t},{branch_u}] = {branch_val:.4f}"
                )

            if logger:
                logger.log_node(
                    NodeLogEntry(
                        node_id=node.node_id,
                        depth=node.depth,
                        lp_bound=lp_bound,
                        incumbent=best_ub,
                        branch_item=branch_i,
                        branch_t=branch_t,
                        direction="BRANCH_Z",
                        status="BRANCHED",
                        cg_iters=cg_iters,
                        columns_added=cols_added,
                    )
                )

            # Child 1: Z[i,t,u] = 1 forced
            child1 = BranchNode(
                node_id=node_counter,
                parent_id=node.node_id,
                depth=node.depth + 1,
            )
            child1.forced_y = {i: set(s) for i, s in forced_y_copy.items()}
            child1.forbidden_y = {i: set(s) for i, s in forbidden_y_copy.items()}
            child1.forced_z = {i: set(s) for i, s in forced_z_copy.items()}
            child1.forbidden_z = {i: set(s) for i, s in forbidden_z_copy.items()}
            if branch_i not in child1.forced_z:
                child1.forced_z[branch_i] = set()
            child1.forced_z[branch_i].add((branch_t, branch_u))
            node_counter += 1

            # Child 2: Z[i,t,u] = 0 forbidden
            child2 = BranchNode(
                node_id=node_counter,
                parent_id=node.node_id,
                depth=node.depth + 1,
            )
            child2.forced_y = {i: set(s) for i, s in forced_y_copy.items()}
            child2.forbidden_y = {i: set(s) for i, s in forbidden_y_copy.items()}
            child2.forced_z = {i: set(s) for i, s in forced_z_copy.items()}
            child2.forbidden_z = {i: set(s) for i, s in forbidden_z_copy.items()}
            if branch_i not in child2.forbidden_z:
                child2.forbidden_z[branch_i] = set()
            child2.forbidden_z[branch_i].add((branch_t, branch_u))
            node_counter += 1

            # Add to queue
            # HYBRID SEARCH: Use DFS until incumbent found (controlled by USE_DFS_UNTIL_INCUMBENT)
            if USE_DFS_UNTIL_INCUMBENT and best_ub >= math.inf:
                priority1 = -child1.depth
                priority2 = -child2.depth
            else:
                priority1 = lp_bound
                priority2 = lp_bound
            heapq.heappush(queue, (priority1, child1.node_id, lp_bound, child1))
            heapq.heappush(queue, (priority2, child2.node_id, lp_bound, child2))

        else:
            # No branching variable found - numerical issue
            if verbose:
                print(f"  No branching variable found (numerical issue)")
            continue

    # Compute final lower bound from unexplored nodes' lb_estimates
    if queue:
        global_lb = min(q[2] for q in queue)  # q[2] is lb_estimate
    else:
        global_lb = best_ub

    # Compute gap
    if best_ub < math.inf and global_lb > -math.inf:
        gap = (best_ub - global_lb) / max(abs(best_ub), 1e-10) * 100
    else:
        gap = float("inf")

    if verbose:
        print(f"\nBranch-and-Price completed:")
        print(f"  Nodes explored: {nodes_explored}")
        print(f"  Best UB: {best_ub:.4f}")
        print(f"  Best LB: {global_lb:.4f}")
        print(f"  Gap: {gap:.4f}%")
        if time_limit_reached:
            print(f"  Termination: TIME_LIMIT")

    return best_ub, global_lb, columns, best_lam, time_limit_reached


# =============================================================================
# INTERFACE FUNCTION'S


def _as_len_T_vector(val, T: int) -> List[float]:
    """Convert scalar or list to length-T list."""
    if isinstance(val, (int, float)):
        return [float(val)] * T
    return [float(v) for v in val]


def solve_instance(
    instance_path: str | Path = "last_instance.json",
    time_limit: int = 0,
    out_dir: str | Path = "bnp_results",
    verbose: bool = True,
    mip_gap: float = 0.0,  # For Streamlit compatibility, not used
) -> Tuple[Dict, List[str]]:
    """
    Solve the perishable lot-sizing problem using ZIO Full-Plans B&P.

    Args:
        instance_path: Path to instance JSON file
        time_limit: Time limit in seconds (0 = unlimited)
        out_dir: Output directory
        verbose: If True, print detailed logs
        mip_gap: Ignored (kept for Streamlit compatibility)

    Returns:
        (summary_dict, orders_list)
    """
    _ = mip_gap
    start_time = time.time()

    data = json.loads(Path(instance_path).read_text())
    T = int(data["period"])
    items_raw: Dict[str, dict] = data["items"]
    items: Dict[int, dict] = {}

    for k, v in items_raw.items():
        i = int(k)
        # Get shelf_seq per period (critical for correct reachability)
        shelf_seq = v.get("shelf_seq", [T] * T)
        if isinstance(shelf_seq, (int, float)):
            shelf_seq = [int(shelf_seq)] * T
        else:
            shelf_seq = [int(s) for s in shelf_seq]
        if len(shelf_seq) < T:
            shelf_seq = shelf_seq + [shelf_seq[-1]] * (T - len(shelf_seq))
        items[i] = {
            "demand": _as_len_T_vector(v["demand"], T),
            "setup": _as_len_T_vector(v["setup"], T),
            "h": _as_len_T_vector(v["h"], T),
            "c_var": _as_len_T_vector(v["c_var"], T),
            "shelf_seq": shelf_seq[:T],  # Per-period shelf life
        }

    prod_cap = data.get("production_capacity") or data.get("manual_capacity")
    capacity = _as_len_T_vector(prod_cap, T) if prod_cap else [math.inf] * T

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger = BnPLogger(out_path, enabled=verbose)

    if verbose:
        print("=" * 70)
        print("ZIO FULL-PLANS BRANCH-AND-PRICE")
        print("=" * 70)
        print(f"Items: {len(items)}, Periods: {T}")
        print(f"Capacity: {capacity[:min(10, T)]}{'...' if T > 10 else ''}")
        print()

    # Solve
    best_ub, best_lb, columns, lam_vals, time_limit_reached = solve_branch_and_price(
        items=items,
        capacity=capacity,
        T=T,
        time_limit=float(time_limit) if time_limit > 0 else math.inf,
        logger=logger,
        verbose=verbose,
    )

    runtime = time.time() - start_time

    # Write CSV logs
    logger.write_csvs()

    # Compute gap
    if best_ub < math.inf and best_lb > -math.inf:
        gap = (best_ub - best_lb) / max(abs(best_lb), 1e-10) if best_lb != 0 else 0
    else:
        gap = 1.0

    # Determine status
    if best_ub < math.inf:
        if gap < 1e-6:
            status = GRB.OPTIMAL
        else:
            status = GRB.SUBOPTIMAL
    elif time_limit_reached:
        # Time limit reached but no incumbent found - not proven infeasible
        status = GRB.TIME_LIMIT
    else:
        # Branching concluded with no feasible solution
        status = GRB.INFEASIBLE

    # Build orders output (format compatible with parse_orders_lines)
    orders = []
    if best_ub < math.inf and lam_vals:
        # Compute x_agg per item: x_agg[item_id][t] = production quantity
        x_agg_per_item: Dict[int, Dict[int, float]] = {i: {} for i in items}
        for i in sorted(items.keys()):
            if i not in columns or i not in lam_vals:
                continue
            for t in range(T):
                x_val = 0.0
                for k, col in enumerate(columns[i]):
                    x_val += col.x.get(t, 0) * lam_vals[i].get(k, 0)
                if x_val > EPS:
                    x_agg_per_item[i][t] = x_val

        # Compute column Z_agg (original from convex combination)
        z_agg_per_item: Dict[int, Dict[Tuple[int, int], float]] = {i: {} for i in items}
        for i in sorted(items.keys()):
            if i not in columns or i not in lam_vals:
                continue
            for k, col in enumerate(columns[i]):
                lam_k = lam_vals[i].get(k, 0)
                if lam_k < EPS:
                    continue
                for (t, u), z in col.z.items():
                    if z > EPS:
                        z_agg_per_item[i][(t, u)] = (
                            z_agg_per_item[i].get((t, u), 0) + z * lam_k
                        )
            # Filter small values
            z_agg_per_item[i] = {k: v for k, v in z_agg_per_item[i].items() if v > EPS}

        # Compute LEFO-compatible Z values via LP
        lefo_ok, lefo_z_per_item = compute_all_lefo_z(items, x_agg_per_item, T)

        # Format orders like solver_bnp_dp: "Item {id} — orders (t → qty)"
        for item_id in sorted(items.keys()):
            orders.append(f"Item {item_id} — orders (t → qty)")
            production = x_agg_per_item.get(item_id, {})
            for t in sorted(production.keys()):
                qty = production[t]
                if qty > EPS:
                    orders.append(f" {t:2d} → {qty:8.3f}")

            # Show original Z_agg from columns (use 'to' instead of arrow to avoid parser confusion)
            z_agg = z_agg_per_item.get(item_id, {})
            if z_agg:
                orders.append(f"  Column Z_agg (original):")
                for (t, u), z_val in sorted(z_agg.items()):
                    flow = z_val * items[item_id]["demand"][u]
                    orders.append(f"    ({t} to {u}): {z_val:.4f} (flow={flow:.2f})")

            # Add LEFO-compatible Z arcs - Z=1 means arc is open, show flow amount
            lefo_z = lefo_z_per_item.get(item_id, {})
            if lefo_z:
                orders.append(f"  LEFO Z (post-processed) - arc:Z (flow):")
                for (t, u), z_val in sorted(lefo_z.items()):
                    flow = z_val * items[item_id]["demand"][u]
                    orders.append(f"    ({t} to {u}): 1 (flow={flow:.2f})")
            orders.append("")

        if verbose:
            print(f"\nLEFO Z computation: {'OK' if lefo_ok else 'FAILED'}")
            print(
                f"Objective after LEFO: {best_ub:.4f} (UNCHANGED - depends only on X,Y)"
            )

    summary = {
        "status": status,
        "objective": best_ub if best_ub < math.inf else None,
        "best_bound": best_lb if best_lb > -math.inf else None,
        "gap": gap,
        "runtime_sec": runtime,
    }

    return summary, orders


# TESTING FUNCTIONS


def print_active_columns(
    columns: Dict[int, List[ZIOColumn]],
    lam_vals: Dict[int, Dict[int, float]],
    items: Dict[int, dict],
    T: int,
):
    """Print detailed info about active columns in the optimal solution."""
    print("\n" + "=" * 70)
    print("ACTIVE COLUMNS IN OPTIMAL SOLUTION")
    print("=" * 70)

    for i in sorted(items.keys()):
        print(f"\n--- Item {i} ---")
        if i not in columns or i not in lam_vals:
            print("  No columns")
            continue

        active_cols = []
        for k, col in enumerate(columns[i]):
            lam_k = lam_vals[i].get(k, 0)
            if lam_k > 1e-6:
                active_cols.append((k, col, lam_k))

        if not active_cols:
            print("  No active columns")
            continue

        print(f"  Active columns: {len(active_cols)}")
        total_lam = sum(lam for _, _, lam in active_cols)
        print(f"  Sum of λ: {total_lam:.6f}")

        for k, col, lam_k in active_cols:
            print(f"\n  Column {k}: λ = {lam_k:.6f} ({lam_k*100:.2f}%)")
            print(f"    Cost: {col.cost:.2f}")

            y_periods = sorted([t for t, v in col.y.items() if v == 1])
            print(f"    Y (setups): {y_periods}")

            # Show X values
            x_str = ", ".join([f"t{t}:{col.x.get(t, 0):.1f}" for t in y_periods])
            print(f"    X (production): {x_str}")

            # Identify dummy setups (Y=1, X=0)
            dummy = [t for t in y_periods if col.x.get(t, 0) < 1e-6]
            if dummy:
                print(f"    Dummy setups (Y=1, X=0): {dummy}")

            # Show arcs
            arcs = sorted(col.z.keys())
            if arcs:
                arc_str = ", ".join([f"({t}→{u})" for t, u in arcs])
                print(f"    Z (arcs): {arc_str}")

        # Show convex combination result
        print(f"\n  Convex combination for Item {i}:")
        x_agg = {}
        y_agg = {}
        z_agg = {}
        cost_agg = 0
        for k, col, lam_k in active_cols:
            cost_agg += col.cost * lam_k
            for t in range(T):
                x_agg[t] = x_agg.get(t, 0) + col.x.get(t, 0) * lam_k
                y_agg[t] = y_agg.get(t, 0) + col.y.get(t, 0) * lam_k
            for (t, u), val in col.z.items():
                z_agg[(t, u)] = z_agg.get((t, u), 0) + val * lam_k

        print(f"    Aggregated cost: {cost_agg:.2f}")
        x_periods = [(t, x_agg[t]) for t in range(T) if x_agg[t] > 1e-6]
        print(f"    X_agg: {', '.join([f't{t}:{x:.2f}' for t, x in x_periods])}")
        y_periods = [(t, y_agg[t]) for t in range(T) if y_agg[t] > 1e-6]
        print(f"    Y_agg: {', '.join([f't{t}:{y:.4f}' for t, y in y_periods])}")
        z_arcs = sorted([(t, u, v) for (t, u), v in z_agg.items() if v > 1e-6])
        print(
            f"    Z_agg (column): {', '.join([f'({t}→{u}):{v:.4f}' for t, u, v in z_arcs])}"
        )

        # Compute LEFO-compatible Z values
        x_values = {t: x for t, x in x_agg.items() if x > EPS}
        demand = items[i]["demand"]
        shelf_seq = items[i].get("shelf_seq", [T] * T)
        lefo_ok, lefo_z = compute_lefo_z_values(i, x_values, demand, shelf_seq, T)

        if lefo_ok and lefo_z:
            # Format: Z=1 (arc open) with flow amount
            lefo_arcs = sorted([(t, u, v) for (t, u), v in lefo_z.items()])
            arc_strs = []
            for t, u, v in lefo_arcs:
                flow = v * demand[u]
                arc_strs.append(f"({t}→{u}):1 (flow={flow:.2f})")
            print(f"    Z_LEFO: {', '.join(arc_strs)}")
            print(f"    LEFO Status: OK (objective unchanged: {cost_agg:.2f})")
        else:
            print(f"    LEFO Status: FAILED")


def _print_results(summary: Dict, orders: List[str], instance: dict):
    """Print formatted results."""
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

    if orders:
        print("\nProduction plan:")
        for line in orders:
            if line.strip():
                print(f"  {line}")


if __name__ == "__main__":
    import sys

    # Choose test: "single", "multi", or path to JSON file
    test_type = sys.argv[1] if len(sys.argv) > 1 else "single"

    out_dir = Path("bnp_full_plans_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if test_type is a JSON file path
    if test_type.endswith(".json") or "/" in test_type:
        # Load instance from JSON file
        json_path = Path(test_type)
        if not json_path.exists():
            print(f"Error: File not found: {json_path}")
            sys.exit(1)
        instance = json.loads(json_path.read_text())
        instance_path = out_dir / "loaded_instance.json"
        print(f"Loading instance from: {json_path}")
    elif test_type == "single":
        # Single-item test instance
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
                    "shelf_seq": [10, 10, 7, 10, 3, 10, 10, 10, 10, 10],
                }
            },
        }
        instance_path = out_dir / "single_item_test.json"
        print("Testing SINGLE-ITEM instance")
        print("Expected: LP relaxation ~1003.60, Integer optimal (single ZIO) ~1231.76")
        # Note: The 1105.76 was the LP optimal with convex combination of multiple ZIO columns.
        # The true integer optimal (single ZIO column respecting capacity) is ~1231.76.
    else:
        # Multi-item test instance (3 items)
        instance = {
            "period": 10,
            "manual_capacity": [0, 0, 150, 150, 150, 150, 150, 150, 200, 150],
            "items": {
                "0": {
                    "h": [0.4] * 10,
                    "c_var": [2.0] * 10,
                    "setup": [80] * 10,
                    "demand": [0, 0, 68, 49, 66, 38, 17, 17, 41, 43],
                    "shelf_seq": [24] * 10,
                },
                "1": {
                    "h": [0.5] * 10,
                    "c_var": [1.8] * 10,
                    "setup": [90] * 10,
                    "demand": [0, 0, 30, 25, 40, 20, 10, 15, 20, 25],
                    "shelf_seq": [20] * 10,
                },
                "2": {
                    "h": [0.3] * 10,
                    "c_var": [2.2] * 10,
                    "setup": [70] * 10,
                    "demand": [0, 0, 20, 15, 30, 10, 5, 10, 15, 20],
                    "shelf_seq": [18] * 10,
                },
            },
        }
        instance_path = out_dir / "multi_item_test.json"
        print("Testing MULTI-ITEM instance (3 items, shared capacity)")

    # ALTERNATIVE TEST INSTANCES (copy/paste to use):
    #
    #
    # --- SINGLE-ITEM ---
    # Run with: python solver_bnp_dp_full_plans.py single
    #
    # instance = {
    #     "period": 10,
    #     "manual_capacity": [0, 0, 80, 80, 80, 80, 80, 80, 100, 80],
    #     "items": {
    #         "0": {
    #             "h": [0.4, 0.408, 0.416, 0.424, 0.430, 0.435, 0.438, 0.440, 0.440, 0.438],
    #             "c_var": [1.76, 1.71, 2.50, 1.78, 1.67, 2.11, 1.37, 1.24, 2.68, 1.81],
    #             "setup": [80, 81.66, 83.25, 84.70, 85.95, 86.93, 87.61, 87.96, 87.96, 87.61],
    #             "demand": [0, 0, 68, 49, 66, 38, 17, 17, 41, 43],
    #             "shelf_seq": [24, 18, 22, 6, 16, 9, 21, 23, 9, 7],
    #         }
    #     },
    # }
    # Expected: LP ~1003.60, Integer (single ZIO) ~1231.76
    #
    # --- MULTI-ITEM (3 items, shared capacity) ---
    # Run with: python solver_bnp_dp_full_plans.py multi
    #
    # instance = {
    #     "period": 10,
    #     "manual_capacity": [0, 0, 150, 150, 150, 150, 150, 150, 200, 150],
    #     "items": {
    #         "0": {"h": [0.4]*10, "c_var": [2.0]*10, "setup": [80]*10,
    #               "demand": [0, 0, 68, 49, 66, 38, 17, 17, 41, 43], "shelf_seq": [24]*10},
    #         "1": {"h": [0.5]*10, "c_var": [1.8]*10, "setup": [90]*10,
    #               "demand": [0, 0, 30, 25, 40, 20, 10, 15, 20, 25], "shelf_seq": [20]*10},
    #         "2": {"h": [0.3]*10, "c_var": [2.2]*10, "setup": [70]*10,
    #               "demand": [0, 0, 20, 15, 30, 10, 5, 10, 15, 20], "shelf_seq": [18]*10},
    #     },
    # }
    # =========================================================================

    instance_path.write_text(json.dumps(instance, indent=2))
    print(f"Instance: {len(instance['items'])} items, {instance['period']} periods")
    print()

    # Run with detailed column output
    data = json.loads(instance_path.read_text())
    T = int(data["period"])
    items_raw = data["items"]
    items = {}
    for k, v in items_raw.items():
        i = int(k)
        shelf_seq = v.get("shelf_seq", [T] * T)
        if isinstance(shelf_seq, (int, float)):
            shelf_seq = [int(shelf_seq)] * T
        else:
            shelf_seq = [int(s) for s in shelf_seq]
        if len(shelf_seq) < T:
            shelf_seq = shelf_seq + [shelf_seq[-1]] * (T - len(shelf_seq))
        items[i] = {
            "demand": _as_len_T_vector(v["demand"], T),
            "setup": _as_len_T_vector(v["setup"], T),
            "h": _as_len_T_vector(v["h"], T),
            "c_var": _as_len_T_vector(v["c_var"], T),
            "shelf_seq": shelf_seq[:T],
        }

    prod_cap = data.get("production_capacity") or data.get("manual_capacity")
    capacity = _as_len_T_vector(prod_cap, T) if prod_cap else [math.inf] * T

    logger = BnPLogger(out_dir, enabled=True)

    best_ub, best_lb, columns, lam_vals, time_limit_reached = solve_branch_and_price(
        items=items,
        capacity=capacity,
        T=T,
        time_limit=300,
        logger=logger,
        verbose=True,
    )

    logger.write_csvs()

    # Build summary
    if best_ub < math.inf and best_lb > -math.inf:
        gap = (best_ub - best_lb) / max(abs(best_lb), 1e-10) if best_lb != 0 else 0
    else:
        gap = 1.0

    if best_ub < math.inf:
        status = GRB.OPTIMAL if gap < 1e-6 else GRB.SUBOPTIMAL
    elif time_limit_reached:
        status = GRB.TIME_LIMIT
    else:
        status = GRB.INFEASIBLE

    orders = []
    if best_ub < math.inf and lam_vals:
        y_agg = compute_y_aggregates(items, columns, lam_vals, T)
        for i in sorted(items.keys()):
            for t in range(T):
                y_val = y_agg.get((i, t), 0)
                if y_val > 0.5:
                    x_val = 0.0
                    if i in columns and i in lam_vals:
                        for k, col in enumerate(columns[i]):
                            x_val += col.x.get(t, 0) * lam_vals[i].get(k, 0)
                    if x_val > EPS:
                        orders.append(f"item={i}, t={t}, x={x_val:.2f}")

    summary = {
        "status": status,
        "objective": best_ub,
        "best_bound": best_lb,
        "gap": gap,
        "runtime_sec": 0,
    }

    # Print active columns details
    if lam_vals:
        print_active_columns(columns, lam_vals, items, T)

    _print_results(summary, orders, instance)
