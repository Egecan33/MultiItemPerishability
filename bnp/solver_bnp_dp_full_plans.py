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
Through branching, the Y pattern becomes fixed, and the ZIO
column structure emerges naturally (all columns share same Y pattern
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

# #region agent log - LEFO violation checker
import json as _json

_DEBUG_LOG_PATH = (
    "/Users/egecanaktan/github_repositories/MultiItemPerishability/debug_lefo.log"
)


def _debug_log(location: str, message: str, data: dict, hypothesis_id: str):
    """Write debug log entry to NDJSON file."""
    import time

    entry = {
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "data": data,
        "hypothesisId": hypothesis_id,
        "sessionId": "debug-session",
    }
    with open(_DEBUG_LOG_PATH, "a") as f:
        f.write(_json.dumps(entry) + "\n")


def check_theorem_violation(
    col, shelf_seq: List[int], demand: List[float]
) -> List[Tuple]:
    """
    Check the THEOREM: V_T >= E for all production periods T in a subplan ending at E.

    In ZIO Full-Plans, each column can have multiple "blocks" (subplans).
    For each block: all productions in that block must have expiry >= last demand served.

    Actually, in ZIO each production period serves a CONTIGUOUS set of demands.
    The theorem becomes: for production at t serving demands up to e, V_t >= e.

    Returns list of violations: [(t, v_t, max_demand_served), ...]
    """
    T = len(demand)
    violations = []

    # Get all arcs from the column: (production_period, demand_period)
    arcs = [(t, u) for (t, u), val in col.z.items() if val > 0]

    if not arcs:
        return []

    # Group arcs by production period: t -> list of demands served
    demands_by_prod = {}
    for t, u in arcs:
        if t not in demands_by_prod:
            demands_by_prod[t] = []
        demands_by_prod[t].append(u)

    # Check: for each production t, V_t >= max demand served by t
    for t, demands_served in demands_by_prod.items():
        max_demand = max(demands_served)
        v_t = t + shelf_seq[t] if t < len(shelf_seq) else t + T

        if v_t < max_demand:
            violations.append((t, v_t, max_demand))

    return violations


def check_column_lefo_violation(
    col, shelf_seq: List[int], demand: List[float]
) -> List[Tuple]:
    """
    Check if a ZIO column has internal LEFO violations.

    Returns list of violations: [(t1, u, t2, up, v1, v2), ...]
    where production at t1 (expiry v1) serves demand u,
    and production at t2 (expiry v2) serves demand up,
    violating LEFO constraint: v1 < v2 and t2 <= u <= up - 1
    """
    T = len(demand)
    violations = []

    # Get all arcs from the column: (production_period, demand_period)
    arcs = [(t, u) for (t, u), val in col.z.items() if val > 0]

    if len(arcs) < 2:
        return []

    # Compute expiry for each production period used
    prod_periods = list(set(t for t, u in arcs))
    expiry = {}
    for t in prod_periods:
        if t < len(shelf_seq):
            expiry[t] = t + shelf_seq[t]
        else:
            expiry[t] = t + T  # fallback

    # Check all pairs of arcs for LEFO violations
    for t1, u in arcs:
        v1 = expiry.get(t1, T)
        for t2, up in arcs:
            if t1 == t2:
                continue
            v2 = expiry.get(t2, T)

            # LEFO violation: v1 < v2 and t2 <= u <= up - 1
            if v1 < v2 and t2 <= u <= up - 1:
                violations.append((t1, u, t2, up, v1, v2))

    return violations


# #endregion


# CONFIGURATION FLAGS

USE_DFS_UNTIL_INCUMBENT = True  # Use DFS until incumbent found, then best-first

# CUT SETTINGS - Valid inequalities to tighten LP relaxation (exact, no loss of optimality)
ENABLE_COVER_CUTS = False

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
        """Unique signature for deduplication - includes Y, X, and Z."""
        y_sig = tuple(sorted(self.y.keys()))
        x_sig = tuple(sorted(self.x.items()))
        z_sig = tuple(sorted(self.z.keys()))
        return (self.item_id, y_sig, x_sig, z_sig)


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
                # If shelf_seq[t] < 0, truly impossible (negative shelf life)
                if shelf_seq[t] < 0:
                    continue
                # Check if t can reach u (t + shelf_seq[t] >= u)
                # Note: shelf_seq[t]=0 means production at t can serve demand at t only
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
            cost += setup[t]

    col = ZIOColumn(item_id=item_id, y=y, x=x, z=z, cost=cost)

    # #region agent log - Check initial column for LEFO violations
    violations = check_column_lefo_violation(col, shelf_seq, demand)
    if violations:
        _debug_log(
            "generate_initial_column:370",
            f"LEFO VIOLATION in INITIAL column for item {item_id}",
            {
                "item_id": item_id,
                "y_periods": list(y.keys()),
                "z_arcs": [(t, u) for (t, u) in z.keys()],
                "violations": [
                    (t1, u, t2, up, v1, v2) for t1, u, t2, up, v1, v2 in violations
                ],
                "shelf_seq": shelf_seq,
            },
            "H4",
        )
    # #endregion

    return col


def enumerate_zio_columns(
    item_id: int,
    demand: List[float],
    setup: List[float],
    h: List[float],
    c_var: List[float],
    mu: float,  # Dual of convexity constraint
    rho: Dict[int, float],  # Dual of capacity constraints
    shelf_life: int,
    max_columns: int = 50,
    forced_y: Set[int] = None,
    forbidden_y: Set[int] = None,
    forced_z: Set[Tuple[int, int]] = None,
    forbidden_z: Set[Tuple[int, int]] = None,
) -> List[Tuple[ZIOColumn, float]]:
    """
    Enumerate ZIO columns with negative reduced cost using DFS.

    This generates MULTIPLE columns (not just the optimal) to enable
    finding optimal convex combinations.

    Reduced cost = c_k - μ - Σ_t ρ_t * x_k[t]
    """
    T = len(demand)
    forced_y = forced_y or set()
    forbidden_y = forbidden_y or set()
    forced_z = forced_z or set()
    forbidden_z = forbidden_z or set()

    demand_periods = [u for u in range(T) if demand[u] > EPS]

    if not demand_periods:
        # No demand - check if empty column has negative RC
        dummy_rc = sum(setup[t] for t in forced_y if t not in forbidden_y)
        rc = dummy_rc - mu
        if rc < -EPS:
            y = {t: 1 for t in forced_y if t not in forbidden_y}
            cost = sum(setup[t] for t in y)
            return [(ZIOColumn(item_id=item_id, y=y, x={}, z={}, cost=cost), rc)]
        return []

    n = len(demand_periods)
    INF = float("inf")

    # Build map: demand_period -> required production period (from forced_z)
    required_prod = {}
    for t, u in forced_z:
        if u in demand_periods:
            required_prod[u] = t

    def block_cost(t: int, i: int, j: int) -> Tuple[float, float]:
        """Compute (reduced_cost, actual_cost) for producing at t to cover demands[i..j]."""
        if t in forbidden_y:
            return INF, INF

        demands_covered = demand_periods[i : j + 1]

        # Check forbidden arcs
        for u in demands_covered:
            if (t, u) in forbidden_z:
                return INF, INF

        # Check forced_z constraints
        for u in demands_covered:
            if u in required_prod and required_prod[u] != t:
                return INF, INF

        total_prod = sum(demand[u] for u in demands_covered)
        rho_t = rho.get(t, 0.0)

        s_cost = setup[t]
        v_cost = sum(c_var[t] * demand[u] for u in demands_covered)
        # Holding cost: sum of h[r] for each period r from t to u-1, times demand[u]
        h_cost = sum(
            sum(h[r] if isinstance(h, list) else h for r in range(t, u)) * demand[u]
            for u in demands_covered
        )

        actual = s_cost + v_cost + h_cost
        reduced = s_cost + v_cost + h_cost - rho_t * total_prod

        return reduced, actual

    # DFS to enumerate all valid ZIO paths
    results = []

    def dfs(demand_idx: int, current_rc: float, current_actual: float, path: List):
        """DFS to enumerate all ZIO columns."""
        if demand_idx >= n:
            # Complete path - build column
            production_periods = set(p[0] for p in path)

            # Add dummy setup costs for forced_y
            dummy_cost = 0.0
            for t in forced_y:
                if t not in production_periods and t not in forbidden_y:
                    dummy_cost += setup[t]

            final_rc = current_rc + dummy_cost - mu
            final_actual = current_actual + dummy_cost

            if final_rc < -EPS:
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

                col = ZIOColumn(item_id=item_id, y=y, x=x, z=z, cost=final_actual)
                results.append((col, final_rc))

            return

        # Early termination if we have enough columns
        if len(results) >= max_columns:
            return

        u_start = demand_periods[demand_idx]

        # Try all production periods t that can reach u_start
        for t in range(u_start + 1):
            if t in forbidden_y:
                continue

            max_reachable = t + shelf_life - 1

            # Try all block sizes
            for j in range(demand_idx, n):
                u_end = demand_periods[j]

                if u_end > max_reachable:
                    break

                rc_block, actual_block = block_cost(t, demand_idx, j)

                if rc_block < INF:
                    path.append((t, demand_idx, j))
                    dfs(
                        j + 1,
                        current_rc + rc_block,
                        current_actual + actual_block,
                        path,
                    )
                    path.pop()

                    if len(results) >= max_columns:
                        return

    dfs(0, 0.0, 0.0, [])

    # Sort by reduced cost and return
    results.sort(key=lambda x: x[1])
    return results[:max_columns]


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
    shelf_seq: List[int],  # Per-period shelf life
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

    Following Algorithm 3 from the paper with K-best extension:
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

    if not demand_periods:
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
            # If shelf_seq[t] < 0, truly impossible (negative shelf life)
            # Note: shelf_seq[t]=0 means production at t can serve demand at t only
            if shelf_seq[t] < 0:
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

                s_cost = setup[t]
                v_cost = sum(c_var[t] * demand[u] for u in demands_in_block)
                # Holding cost: sum of h[r] for each period r from t to u-1, times demand[u]
                h_cost = sum(
                    sum(h[r] if isinstance(h, list) else h for r in range(t, u))
                    * demand[u]
                    for u in demands_in_block
                )

                actual = s_cost + v_cost + h_cost

                # sig_y discount: if block at period t and t is forced, get discount
                sig_y_discount = sig_y.get(t, 0.0) if t in forced_y else 0.0

                # tau_z discount: for each forced arc (t,u) covered by block
                tau_z_discount = 0.0
                for u in demands_in_block:
                    if (t, u) in forced_z:
                        tau_z_discount += tau_z.get((t, u), 0.0)

                reduced = actual - rho_t * total_prod - sig_y_discount - tau_z_discount

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

        # Add forced_y dummy setup costs (with sig_y discount)
        dummy_setup_cost = 0.0
        dummy_sig_y_discount = 0.0
        for t in forced_y:
            if t not in production_periods and t not in forbidden_y:
                dummy_setup_cost += setup[t]
                dummy_sig_y_discount += sig_y.get(t, 0.0)

        # Add tau_z discount for forced arcs not already covered
        arcs_covered = set()
        for prod_t, start_i, end_j in path:
            for u in demand_periods[start_i : end_j + 1]:
                arcs_covered.add((prod_t, u))
        dummy_tau_z_discount = 0.0
        for t, u in forced_z:
            if (t, u) not in arcs_covered:
                dummy_tau_z_discount += tau_z.get((t, u), 0.0)

        final_rc = (
            rc_path
            + dummy_setup_cost
            - mu
            - dummy_sig_y_discount
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

        # #region agent log - Check for THEOREM violation (V_T >= E for all T)
        theorem_violations = check_theorem_violation(col, shelf_seq, demand)
        if theorem_violations:
            _debug_log(
                "price_zio_column_dp:theorem",
                f"THEOREM VIOLATION in column for item {item_id} - REJECTED",
                {
                    "item_id": item_id,
                    "y_periods": list(y.keys()),
                    "z_arcs": [(t, u) for (t, u) in z.keys()],
                    "theorem_violations": [
                        {"t": t, "expiry": v_t, "max_demand": max_d}
                        for t, v_t, max_d in theorem_violations
                    ],
                    "shelf_seq": shelf_seq,
                },
                "H5",
            )
            # FIX: Skip columns with theorem violations
            continue
        # #endregion

        # #region agent log - Check for LEFO violations in generated column
        lefo_violations = check_column_lefo_violation(col, shelf_seq, demand)
        if lefo_violations:
            _debug_log(
                "price_zio_column_dp:lefo",
                f"LEFO VIOLATION in column for item {item_id} - REJECTED",
                {
                    "item_id": item_id,
                    "y_periods": list(y.keys()),
                    "z_arcs": [(t, u) for (t, u) in z.keys()],
                    "violations": [
                        (t1, u, t2, up, v1, v2)
                        for t1, u, t2, up, v1, v2 in lefo_violations
                    ],
                    "shelf_seq": shelf_seq,
                    "reduced_cost": final_rc,
                },
                "H1",
            )
            # FIX: Skip columns with LEFO violations - they are structurally invalid
            continue
        else:
            _debug_log(
                "price_zio_column_dp:ok",
                f"Column VALID for item {item_id}",
                {
                    "item_id": item_id,
                    "y_periods": list(y.keys()),
                    "num_arcs": len(z),
                },
                "H1",
            )
        # #endregion

        results.append((col, final_rc))

    results.sort(key=lambda x: x[1])
    return results[:max_columns]


# =============================================================================
# RESTRICTED MASTER PROBLEM
# =============================================================================


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
    Dict[Tuple[int, int, int, int, int], float],  # tau_lefo duals
]:
    """
    Solve the RMP with flow variables and LEFO constraints.

    RMP:
        min  Σ_i Σ_k c_k^i λ_k^i
        s.t. Σ_k λ_k^i = 1                    ∀i  (convexity) → dual μ_i
             Σ_i Σ_k x_k^i[t] λ_k^i ≤ C_t    ∀t  (capacity)  → dual ρ_t
             Σ_k y_k[t] λ_k = 1               (forced Y)     → dual τ_y[i,t]
             Σ_k z_k[t,u] λ_k = 1             (forced Z)     → dual τ_z[i,t,u]
             F[t,u] ≤ Z_agg[t,u]              (flow link)
             Σ_t F[t,u] = 1                   (demand coverage)
             Z_agg[t1,u] + Z_agg[t2,u'] ≤ 1   (LEFO no-crossing) → dual τ_lefo
             λ_k^i ≥ 0, F[t,u] ∈ [0,1]

    Columns violating branching constraints are excluded.
    τ_y, τ_z, and τ_lefo duals give "discounts" for pricing.

    Returns: (obj, mu, rho, lam_vals, sig_y, tau_z, tau_lefo)
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

        # Check forced Z: column must have Z[t,u] > 0 for all forced arcs
        # (Z=1 branch means arc is used, any positive value qualifies)
        for t, u in forced_z.get(i, set()):
            if col.z.get((t, u), 0) <= 0:
                return False

        # Check forbidden Z: column must have Z[t,u]=0 for all forbidden arcs
        for t, u in forbidden_z.get(i, set()):
            if col.z.get((t, u), 0) == 1:
                return False

        return True

    m = gp.Model("RMP")
    m.Params.OutputFlag = 0

    BIG_M = 1e6  # Penalty for constraint violations

    # Variables: λ[i][k] for each item i, valid column k
    lam = {}
    valid_col_indices = {}  # item -> list of valid column indices

    for i, cols in columns.items():
        valid_indices = [k for k, col in enumerate(cols) if column_is_valid(col)]
        valid_col_indices[i] = valid_indices
        lam[i] = [m.addVar(lb=0, name=f"lam_{i}_{k}") for k in valid_indices]

    # Convexity constraints
    convex_cons = {}
    art_vars = {}
    for i in items:
        if i in lam and lam[i]:
            convex_cons[i] = m.addConstr(gp.quicksum(lam[i]) == 1, f"convex_{i}")
        else:
            art_vars[i] = m.addVar(lb=0, name=f"art_{i}")
            convex_cons[i] = m.addConstr(art_vars[i] == 1, f"convex_{i}")

    # Capacity constraints
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
            cap_slack[t] = m.addVar(lb=0, name=f"cap_slack_{t}")
            cap_cons[t] = m.addConstr(x_sum <= capacity[t] + cap_slack[t], f"cap_{t}")

    # Y-aggregation constraints for forced Y
    y_fix_cons = {}
    y_fix_slack = {}
    for i, periods in forced_y.items():
        for t in periods:
            y_sum = gp.LinExpr()
            valid_indices = valid_col_indices.get(i, [])
            for idx, orig_k in enumerate(valid_indices):
                col = columns[i][orig_k]
                y_sum += col.y.get(t, 0) * lam[i][idx]
            y_fix_slack[(i, t)] = m.addVar(lb=0, name=f"y_fix_slack_{i}_{t}")
            y_fix_cons[(i, t)] = m.addConstr(
                y_sum + y_fix_slack[(i, t)] == 1, f"y_fix_{i}_{t}"
            )

    # Z-aggregation constraints for forced Z
    z_fix_cons = {}
    z_fix_slack = {}
    for i, arcs in forced_z.items():
        for t, u in arcs:
            z_sum = gp.LinExpr()
            valid_indices = valid_col_indices.get(i, [])
            for idx, orig_k in enumerate(valid_indices):
                col = columns[i][orig_k]
                z_sum += col.z.get((t, u), 0) * lam[i][idx]
            z_fix_slack[(i, t, u)] = m.addVar(lb=0, name=f"z_fix_slack_{i}_{t}_{u}")
            z_fix_cons[(i, t, u)] = m.addConstr(
                z_sum + z_fix_slack[(i, t, u)] == 1, f"z_fix_{i}_{t}_{u}"
            )

    # =========================================================================
    # FLOW VARIABLES AND LEFO CONSTRAINTS (per item)
    # =========================================================================
    flow_vars = {}  # (i, t, u) -> flow variable
    flow_link_cons = {}  # (i, t, u) -> constraint F <= Z_agg
    demand_cov_cons = {}  # (i, u) -> constraint sum_t F = 1
    lefo_cons = {}  # (i, t1, t2, u, up) -> no-crossing constraint

    for i, item_data in items.items():
        demand = item_data.get("demand", [0] * T)
        shelf_seq = item_data.get("shelf_seq", [T] * T)
        demand_periods = [u for u in range(T) if demand[u] > EPS]

        if not demand_periods:
            continue

        # Collect all arcs from valid columns
        all_arcs = set()
        valid_indices = valid_col_indices.get(i, [])
        for idx, orig_k in enumerate(valid_indices):
            col = columns[i][orig_k]
            for arc in col.z.keys():
                all_arcs.add(arc)

        if not all_arcs:
            continue

        # Create flow variables F[t,u] in [0,1]
        for t, u in all_arcs:
            flow_vars[(i, t, u)] = m.addVar(lb=0, ub=1, name=f"flow_{i}_{t}_{u}")

        # Flow link: F[t,u] <= Z_agg[t,u] (if Z=0, flow must be 0)
        for t, u in all_arcs:
            z_agg_expr = gp.LinExpr()
            for idx, orig_k in enumerate(valid_indices):
                col = columns[i][orig_k]
                z_agg_expr += col.z.get((t, u), 0) * lam[i][idx]
            flow_link_cons[(i, t, u)] = m.addConstr(
                flow_vars[(i, t, u)] <= z_agg_expr, f"flow_link_{i}_{t}_{u}"
            )

        # Demand coverage: sum_t F[t,u] = 1 for each demand period
        for u in demand_periods:
            arcs_to_u = [(t, uu) for (t, uu) in all_arcs if uu == u]
            if arcs_to_u:
                demand_cov_cons[(i, u)] = m.addConstr(
                    gp.quicksum(flow_vars[(i, t, u)] for t, uu in arcs_to_u) == 1,
                    f"demand_cov_{i}_{u}",
                )

        # Production capacity for flows: sum_u F[t,u] * demand[u] <= X_agg[t]
        # This ensures flows don't exceed production at each period
        prod_periods_in_arcs = set(t for t, u in all_arcs)
        for t in prod_periods_in_arcs:
            arcs_from_t = [(tt, u) for (tt, u) in all_arcs if tt == t]
            if arcs_from_t:
                # X_agg[t] = sum_k x_k[t] * lambda_k
                x_agg_expr = gp.LinExpr()
                for idx, orig_k in enumerate(valid_indices):
                    col = columns[i][orig_k]
                    x_agg_expr += col.x.get(t, 0) * lam[i][idx]
                # Flow capacity: sum_u F[t,u] * demand[u] <= X_agg[t]
                m.addConstr(
                    gp.quicksum(
                        flow_vars[(i, t, u)] * demand[u] for tt, u in arcs_from_t
                    )
                    <= x_agg_expr,
                    f"flow_cap_{i}_{t}",
                )

        # Note: LEFO no-crossing constraints are enforced through Z branching
        # and verified in post-processing LP, not as hard constraints in RMP.
        # This allows the LP relaxation to explore more column combinations.

    # =========================================================================
    # COVER CUTS (LOT-SIZING VALID INEQUALITIES) - Tighten LP relaxation
    # =========================================================================
    # These are valid for all integer solutions, so they don't break exactness.
    # Flow-cover cuts: If flow goes through arc (t,u), setup at t is needed.
    #     Constraint: Y_agg[i,t] >= F[i,t,u]
    # This directly links flow decisions to setup decisions, tightening the LP.

    if ENABLE_COVER_CUTS:
        for i, item_data in items.items():
            valid_indices = valid_col_indices.get(i, [])

            if not valid_indices:
                continue

            # Flow-cover cuts: Y[t] >= F[t,u] for all arcs
            # This is exact: if any flow uses arc (t,u), then Y[t] must be 1
            for (ii, t, u), f_var in flow_vars.items():
                if ii != i:
                    continue
                # Y_agg[i,t] >= F[i,t,u]
                y_agg_expr = gp.LinExpr()
                for idx, orig_k in enumerate(valid_indices):
                    col = columns[i][orig_k]
                    y_agg_expr += col.y.get(t, 0) * lam[i][idx]
                m.addConstr(y_agg_expr >= f_var, f"flow_cover_{i}_{t}_{u}")

    # =========================================================================
    # OBJECTIVE
    # =========================================================================
    obj_expr = gp.LinExpr()
    for i, cols in columns.items():
        valid_indices = valid_col_indices.get(i, [])
        for idx, orig_k in enumerate(valid_indices):
            col = cols[orig_k]
            obj_expr += col.cost * lam[i][idx]
    # Add penalties
    for art_var in art_vars.values():
        obj_expr += BIG_M * art_var
    for slack_var in cap_slack.values():
        obj_expr += BIG_M * slack_var
    for slack_var in y_fix_slack.values():
        obj_expr += BIG_M * slack_var
    for slack_var in z_fix_slack.values():
        obj_expr += BIG_M * slack_var
    m.setObjective(obj_expr, GRB.MINIMIZE)

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return None, {}, {}, {}, {}, {}, {}

    # Extract duals
    mu = {i: convex_cons[i].Pi for i in convex_cons}
    rho = {t: cap_cons[t].Pi for t in cap_cons}
    sig_y = {key: con.Pi for key, con in y_fix_cons.items()}
    tau_z = {key: con.Pi for key, con in z_fix_cons.items()}
    tau_lefo = {}  # No LEFO constraints in RMP - enforced via branching

    # Extract lambda values
    lam_vals = {}
    for i, cols in columns.items():
        lam_vals[i] = {}
        valid_indices = valid_col_indices.get(i, [])
        for idx, orig_k in enumerate(valid_indices):
            lam_vals[i][orig_k] = lam[i][idx].X

    return m.ObjVal, mu, rho, lam_vals, sig_y, tau_z, tau_lefo


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


# =============================================================================
# COLUMN GENERATION LOOP
# =============================================================================


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
        obj, mu, rho, lam_vals, sig_y, tau_z, tau_lefo = result

        if obj is None:
            return None, {}, cg_iter, total_cols_added

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
            tau_z_item = {(t, u): tau_z.get((i, t, u), 0.0) for t, u in forced_arcs}

            # Add LEFO duals to tau_z_item
            # tau_lefo[(i, t1, t2, u, up)] is dual of Z[t1,u] + Z[t2,up] <= 1
            # Discount for using these arcs (dual is <= 0 for <= constraint)
            for key, dual in tau_lefo.items():
                if key[0] != i:
                    continue
                _, t1, t2, u, up = key
                # Add discount for early arc (t1, u)
                if (t1, u) not in tau_z_item:
                    tau_z_item[(t1, u)] = 0.0
                tau_z_item[(t1, u)] -= dual
                # Add discount for later arc (t2, up)
                if (t2, up) not in tau_z_item:
                    tau_z_item[(t2, up)] = 0.0
                tau_z_item[(t2, up)] -= dual

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
                max_columns=1,
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

                        # #region agent log - Track columns added to pool
                        item_data = items[i]
                        shelf_seq = item_data.get("shelf_seq", [T] * T)
                        demand_vec = item_data.get("demand", [0] * T)
                        violations = check_column_lefo_violation(
                            col, shelf_seq, demand_vec
                        )
                        if violations:
                            _debug_log(
                                "column_generation:1290",
                                f"Adding column WITH LEFO VIOLATION to pool",
                                {
                                    "item_id": i,
                                    "cg_iter": cg_iter,
                                    "node_id": node_id,
                                    "reduced_cost": rc,
                                    "violations": violations,
                                },
                                "H2",
                            )
                        # #endregion

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

    # Max iterations reached
    result = solve_rmp_with_duals(
        items, columns, capacity, T, forced_y, forbidden_y, forced_z, forbidden_z
    )
    obj = result[0]
    lam_vals = result[3]
    return obj, lam_vals, max_cg_iters, total_cols_added


# =============================================================================
# BRANCHING
# =============================================================================


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


# =============================================================================
# BRANCH-AND-PRICE MAIN LOOP
# =============================================================================


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

    # Compute reasonable upper bound (to detect Big-M artificial solutions)
    # Sum of all initial column costs gives a rough upper bound
    reasonable_ub = sum(
        max((col.cost for col in cols), default=0) * 2 for cols in columns.values()
    )
    reasonable_ub = max(reasonable_ub, 1e6)  # At least 1M to be safe

    # Create root node
    root = BranchNode(node_id=0, parent_id=None, depth=0)
    root.forbidden_y = {i: {t for t in range(T) if capacity[t] <= EPS} for i in items}

    # Priority queue: (priority, node_id, lb_estimate, node)
    # lb_estimate is parent's LP bound (valid lower bound for unexplored subtree)
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

        # Check if LP uses Big-M penalties (artificial variables) -> fathom as infeasible
        if lp_bound > reasonable_ub:
            if verbose:
                print(f"  Fathomed: uses Big-M ({lp_bound:.2f} > {reasonable_ub:.2f})")
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
                        status="INFEASIBLE_BIGM",
                        cg_iters=cg_iters,
                        columns_added=cols_added,
                    )
                )
            continue

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

        # Integer solution: Y must be binary
        # Z becomes integer through column generation (tau_z mechanism ensures all columns
        # have Z=1 for forced arcs, just like sig_y ensures Y=1 for forced setups)
        if y_is_int:
            # Check if solution uses Big-M penalties (artificial variables)
            if lp_bound > reasonable_ub:
                if verbose:
                    print(
                        f"  Integer Y but uses Big-M ({lp_bound:.2f} > {reasonable_ub:.2f}), fathoming"
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
                            status="INFEASIBLE_BIGM",
                            cg_iters=cg_iters,
                            columns_added=cols_added,
                        )
                    )
                continue

            if verbose:
                print(f"  Integer Y solution found: {lp_bound:.2f}")

            if lp_bound < best_ub:
                best_ub = lp_bound
                best_lam = lam_vals
                if verbose:
                    print(f"  *** New incumbent: {best_ub:.2f} ***")

                # #region agent log - Check active columns in incumbent for LEFO violations
                for i_item in items:
                    if i_item in columns and i_item in lam_vals:
                        for k, col in enumerate(columns[i_item]):
                            lam_k = lam_vals[i_item].get(k, 0)
                            if lam_k > EPS:
                                item_data = items[i_item]
                                shelf_seq = item_data.get("shelf_seq", [T] * T)
                                demand_vec = item_data.get("demand", [0] * T)
                                violations = check_column_lefo_violation(
                                    col, shelf_seq, demand_vec
                                )
                                if violations:
                                    _debug_log(
                                        "solve_bnp:incumbent",
                                        f"INCUMBENT uses column with LEFO VIOLATION",
                                        {
                                            "item_id": i_item,
                                            "column_idx": k,
                                            "lambda": lam_k,
                                            "objective": lp_bound,
                                            "violations": violations,
                                            "y_periods": list(col.y.keys()),
                                            "z_arcs": list(col.z.keys()),
                                        },
                                        "H3",
                                    )
                # #endregion

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

        # Decide: branch on Y first, then Z
        branch_var_y = None
        branch_var_z = None

        # Branch on Y first to fix the setup pattern
        branch_var_y = find_most_fractional_y(
            y_agg_raw, items, node.forced_y, node.forbidden_y
        )

        # If no fractional Y, branch on Z
        if branch_var_y is None:
            branch_var_z = find_most_fractional_z(
                z_agg_raw, items, node.forced_z, node.forbidden_z
            )

        if branch_var_y is None and branch_var_z is None:
            # No fractional variable found - numerical issue
            if verbose:
                print(f"  No fractional Y or Z found (numerical issue)")
            continue

        # Copy constraints from parent
        forced_y_copy, forbidden_y_copy, forced_z_copy, forbidden_z_copy = (
            node.copy_constraints()
        )

        if branch_var_y is not None:
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

            # Add to queue with DFS-until-incumbent strategy
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

        elif branch_var_z is not None:
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

            # Add to queue with DFS-until-incumbent strategy
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
        print(f"  Gap: {gap:.2f}%")
        if time_limit_reached:
            print(f"  Termination: TIME_LIMIT")

    return best_ub, global_lb, columns, best_lam, time_limit_reached


# =============================================================================
# INTERFACE FUNCTION
# =============================================================================


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
        status = GRB.TIME_LIMIT
    else:
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

        # Format orders like solver_bnp_dp: "Item {id} — orders (t → qty)"
        for item_id in sorted(items.keys()):
            orders.append(f"Item {item_id} — orders (t → qty)")
            production = x_agg_per_item.get(item_id, {})
            for t in sorted(production.keys()):
                qty = production[t]
                if qty > EPS:
                    orders.append(f" {t:2d} → {qty:8.3f}")
            orders.append("")

    summary = {
        "status": status,
        "objective": best_ub if best_ub < math.inf else None,
        "best_bound": best_lb if best_lb > -math.inf else None,
        "gap": gap,
        "runtime_sec": runtime,
    }

    return summary, orders


# =============================================================================
# TEST
# =============================================================================


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
        cost_agg = 0
        for k, col, lam_k in active_cols:
            cost_agg += col.cost * lam_k
            for t in range(T):
                x_agg[t] = x_agg.get(t, 0) + col.x.get(t, 0) * lam_k
                y_agg[t] = y_agg.get(t, 0) + col.y.get(t, 0) * lam_k

        print(f"    Aggregated cost: {cost_agg:.2f}")
        x_periods = [(t, x_agg[t]) for t in range(T) if x_agg[t] > 1e-6]
        print(f"    X_agg: {', '.join([f't{t}:{x:.2f}' for t, x in x_periods])}")
        y_periods_agg = [(t, y_agg[t]) for t in range(T) if y_agg[t] > 1e-6]
        print(f"    Y_agg: {', '.join([f't{t}:{y:.4f}' for t, y in y_periods_agg])}")

        # Compute Z_agg (column aggregates)
        z_agg = {}
        for k, col, lam_k in active_cols:
            for (t, u), z_val in col.z.items():
                if z_val > 0:
                    z_agg[(t, u)] = z_agg.get((t, u), 0) + z_val * lam_k

        if z_agg:
            print(f"    Z_agg (column):")
            for (t, u), z_val in sorted(z_agg.items()):
                if z_val > EPS:
                    # Display as binary: 1 if >0, 0 if =0
                    z_binary = 1 if z_val > EPS else 0
                    print(f"      ({t} to {u}): {z_binary}")

        # Compute LEFO-compatible Z via LP
        item_data = items.get(i, {})
        demand = item_data.get("demand", [])
        shelf_seq = item_data.get("shelf_seq", [T] * T)
        x_vals = {t: x_agg[t] for t in range(T) if x_agg[t] > EPS}

        if x_vals:
            is_feas, z_lefo = compute_lefo_z_values(i, x_vals, demand, shelf_seq, T)
            if is_feas and z_lefo:
                print(f"    LEFO Z (flow LP):")
                for (t, u), z_val in sorted(z_lefo.items()):
                    # Display Z as binary: 1 if >0, 0 if =0
                    z_binary = 1 if z_val > EPS else 0
                    # Show flow quantity as integer
                    flow_qty = z_val * demand[u] if u < len(demand) else 0
                    print(
                        f"      ({t} to {u}): {z_binary} (flow={int(round(flow_qty))})"
                    )
                print(f"    LEFO Status: OK")
            elif not is_feas:
                print(f"    LEFO Status: INFEASIBLE")


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

    if test_type.endswith(".json"):
        # Load from JSON file
        instance_path = Path(test_type)
        with open(instance_path) as f:
            instance = json.load(f)
        # Normalize: items/period/capacity may be in "data" structure or top-level
        if "data" in instance:
            data_part = instance["data"]
            if "items" in data_part:
                instance["items"] = data_part["items"]
            if "period" in data_part:
                instance["period"] = data_part["period"]
            if "manual_capacity" in data_part:
                instance["manual_capacity"] = data_part["manual_capacity"]
            if "production_capacity" in data_part:
                instance["production_capacity"] = data_part["production_capacity"]
        # Also check top-level period and manual_capacity (may be set directly)
        if "period" not in instance and "period" in instance.get("data", {}):
            instance["period"] = instance["data"]["period"]
        print(f"Testing JSON file: {test_type}")
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
                    "shelf_seq": [24, 18, 22, 6, 16, 9, 21, 23, 9, 7],
                }
            },
        }
        instance_path = out_dir / "single_item_test.json"
        print("Testing SINGLE-ITEM instance")
        print("Expected: LP relaxation ~1003.60, Integer optimal (single ZIO) ~1231.76")
        # Note: The 1105.76 was the LP optimal with convex combination of multiple ZIO columns.
        # The true integer optimal (single ZIO column respecting capacity) is ~1231.76.
    elif test_type == "multi":
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

    if not test_type.endswith(".json"):
        instance_path.write_text(json.dumps(instance, indent=2))
    print(f"Instance: {len(instance['items'])} items, {instance['period']} periods")
    print()

    # Run with detailed column output
    data = instance
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
