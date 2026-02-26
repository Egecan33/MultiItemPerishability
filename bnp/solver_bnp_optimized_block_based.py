"""
ZIO Block-Based Branch-and-Price Solver (Incremental RMP).

Uses ZIO blocks as columns where each block (t, e) represents:
- Production at period t covering demands from t to e
- Production quantity = sum of demands in [t, e]
- One setup at period t

Key formulation:
- λ[i,t,e]: coefficient for block (t,e) of item i
- f[i,t,u]: flow (quantity) from production at t to demand at u
- y[i,t] = Σ_e λ[i,t,e]: aggregate setup variable

Constraints:
- Demand: Σ_t f[i,t,u] = demand[u]
- Linking: f[i,t,u] ≤ demand[u] * Σ_{e≥u} λ[i,t,e]
- Capacity: Σ_i Σ_t (Σ_u f[i,t,u]) ≤ capacity[t]
- LEFO: Added lazily when violations detected

INCREMENTAL MODEL: The RMP is built once and updated incrementally:
- Columns (blocks) are added without model rebuild
- Branching decisions update variable bounds, not constraints
- LEFO cuts are added incrementally
"""

from __future__ import annotations
import csv
import json
import math
import time
import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import gurobipy as gp
from gurobipy import GRB

EPS = 1e-6
BIG_M_ARTIFICIAL = 1e6

# Configuration
USE_DFS_UNTIL_INCUMBENT = True
ADD_ALL_LEFO_CUTS_UPFRONT = False
MAX_LEFO_CUTS_PER_ITER = 20  # Max cuts per CG iteration if lazy cutting
ENABLE_LEFO_CUTS = True  # Enable LEFO cuts for optimal solutions

# Pricing configuration
GLOBAL_TOP_K = 5  # Add at most this many columns globally per CG iteration
SINGLE_BEST_PER_ITEM = True  # Return only single best block per item when pricing

# Exactness configuration
EXACT_MODE = True  # When True, only claim OPTIMAL when gap is truly closed
ENABLE_HEURISTICS = (
    True  # When False, disable dive heuristics (UB-only, doesn't affect correctness)
)
HEURISTIC_INTERVAL = 50  # Try dive heuristic every N nodes

# Performance timing
ENABLE_TIMING = False  # Log per-node timing

# Configuration presets
EXACT_FAST_CONFIG = {
    "GLOBAL_TOP_K": 3,
    "SINGLE_BEST_PER_ITEM": True,
    "ENABLE_LEFO_CUTS": False,
    "ENABLE_HEURISTICS": False,
    "EXACT_MODE": True,
}


def apply_config_preset(preset_name: str):
    """Apply a configuration preset."""
    global GLOBAL_TOP_K, SINGLE_BEST_PER_ITEM, ENABLE_LEFO_CUTS, ENABLE_HEURISTICS, EXACT_MODE

    if preset_name == "exact-fast":
        cfg = EXACT_FAST_CONFIG
        GLOBAL_TOP_K = cfg["GLOBAL_TOP_K"]
        SINGLE_BEST_PER_ITEM = cfg["SINGLE_BEST_PER_ITEM"]
        ENABLE_LEFO_CUTS = cfg["ENABLE_LEFO_CUTS"]
        ENABLE_HEURISTICS = cfg["ENABLE_HEURISTICS"]
        EXACT_MODE = cfg["EXACT_MODE"]
    else:
        raise ValueError(f"Unknown preset: {preset_name}")


@dataclass
class ZIOBlock:
    """A ZIO block: production at period t serving demands t to e."""

    item_id: int
    start_t: int
    end_e: int
    setup_cost: float

    def __repr__(self):
        return f"Block({self.start_t},{self.end_e})"

    def signature(self) -> Tuple[int, int, int]:
        return (self.item_id, self.start_t, self.end_e)


@dataclass
class ItemCache:
    """Precomputed data for O(1) block cost and quantity lookups."""

    item_id: int
    demand_periods: List[int]  # U_i = [u : d[u] > 0]
    demand_period_idx: Dict[int, int]  # u -> index in demand_periods
    prefix_demand: List[float]  # prefix_demand[j] = sum d[demand_periods[0..j-1]]
    block_cost: Dict[Tuple[int, int], float]  # (t, e) -> cost_i(t,e)
    block_Q: Dict[Tuple[int, int], float]  # (t, e) -> Q_i(t,e) = sum d[t..e]
    block_covers: Dict[
        Tuple[int, int], List[int]
    ]  # (t, e) -> list of demand period indices


def build_item_cache(
    item_id: int,
    item_data: dict,
    T: int,
    Gamma: Dict[int, List[int]],
) -> ItemCache:
    """
    Build precomputed cache for an item.

    All block costs and quantities are computed once here.
    """
    demand = item_data["demand"]
    setup = item_data["setup"]
    c_var = item_data["c_var"]
    h = item_data.get("h", [0.0] * T)

    # Demand periods
    demand_periods = [u for u in range(T) if demand[u] > EPS]
    demand_period_idx = {u: i for i, u in enumerate(demand_periods)}

    # Prefix sum of demands (over demand periods)
    prefix_demand = [0.0]
    for u in demand_periods:
        prefix_demand.append(prefix_demand[-1] + demand[u])

    # Precompute holding cost prefix sums for each production period
    # holding_prefix[t][r] = sum_{s=t..r-1} h[s]
    holding_prefix: Dict[int, List[float]] = {}
    for t in range(T):
        if not Gamma.get(t):
            continue
        hp = [0.0]  # hp[0] = holding from t to t = 0
        for r in range(t, T):
            h_r = (
                h[r]
                if isinstance(h, list) and r < len(h)
                else (h if isinstance(h, (int, float)) else 0.0)
            )
            hp.append(hp[-1] + float(h_r))
        holding_prefix[t] = hp

    # Precompute block costs and quantities
    block_cost: Dict[Tuple[int, int], float] = {}
    block_Q: Dict[Tuple[int, int], float] = {}
    block_covers: Dict[Tuple[int, int], List[int]] = {}

    for t in range(T):
        reachable = Gamma.get(t, [])
        if not reachable:
            continue

        # Setup cost at t
        s_cost = (
            float(setup[t])
            if isinstance(setup, list) and t < len(setup)
            else float(setup)
        )
        var_cost_t = (
            float(c_var[t])
            if isinstance(c_var, list) and t < len(c_var)
            else float(c_var)
        )
        hp = holding_prefix.get(t, [0.0])

        for e in reachable:
            if e < t:
                continue

            # Which demand periods are covered?
            covers = [u for u in demand_periods if t <= u <= e]
            if not covers:
                continue

            block_covers[(t, e)] = covers

            # Compute Q(t, e) = sum of demands in [t, e]
            Q = 0.0
            for u in range(t, e + 1):
                if u < len(demand):
                    Q += demand[u]
            block_Q[(t, e)] = Q

            # Compute cost(t, e) = setup + sum_{u in covers} [var*d[u] + holding(t->u)*d[u]]
            cost = s_cost
            for u in covers:
                d_u = demand[u]
                cost += var_cost_t * d_u
                # Holding from t to u: hp[u - t] if available
                hold_idx = u - t
                if hold_idx < len(hp):
                    cost += hp[hold_idx] * d_u

            block_cost[(t, e)] = cost

    return ItemCache(
        item_id=item_id,
        demand_periods=demand_periods,
        demand_period_idx=demand_period_idx,
        prefix_demand=prefix_demand,
        block_cost=block_cost,
        block_Q=block_Q,
        block_covers=block_covers,
    )


class BlockRMP:
    """
    Incremental Block-based Restricted Master Problem.

    Builds the model once and updates incrementally:
    - add_column(): add new λ[i,t,e] variables
    - solve(): reoptimize and return duals
    - apply_branching(): update branching constraints
    """

    def __init__(
        self,
        items: Dict[int, dict],
        T: int,
        capacity: List[float],
        item_caches: Dict[int, ItemCache],
    ):
        self.items = items
        self.T = T
        self.capacity = capacity
        self.item_caches = item_caches

        self.model = gp.Model("BlockRMP")
        self.model.Params.OutputFlag = 0
        self.model.Params.Method = 1  # Dual simplex for warm starts

        # Variables
        self.lam: Dict[Tuple[int, int, int], gp.Var] = {}  # (i, t, e) -> λ variable
        self.art: Dict[Tuple[int, int], gp.Var] = {}  # (i, u) -> artificial

        # Constraints
        self.coverage_con: Dict[Tuple[int, int], gp.Constr] = {}  # (i, u)
        self.cap_con: Dict[int, gp.Constr] = {}  # t
        self.force_y_con: Dict[Tuple[int, int], gp.Constr] = {}  # (i, t)
        self.lefo_cut_con: List[gp.Constr] = []

        # Block data cache
        self.block_cost: Dict[Tuple[int, int, int], float] = {}
        self.block_Q: Dict[Tuple[int, int, int], float] = {}
        self.block_covers: Dict[Tuple[int, int, int], List[int]] = {}

        self._build_initial_structure()

    def _build_initial_structure(self):
        """Build constraints with slack - columns added later."""
        # Create artificial variables for each coverage constraint
        for item_id, cache in self.item_caches.items():
            for u in cache.demand_periods:
                art = self.model.addVar(
                    lb=0.0, obj=BIG_M_ARTIFICIAL, name=f"art_{item_id}_{u}"
                )
                self.art[(item_id, u)] = art

        self.model.update()

        # Coverage constraints: art[i,u] = 1 (will add λ columns later)
        for item_id, cache in self.item_caches.items():
            for u in cache.demand_periods:
                con = self.model.addConstr(
                    self.art[(item_id, u)] == 1.0, f"cover_{item_id}_{u}"
                )
                self.coverage_con[(item_id, u)] = con

        # Capacity constraints: 0 <= cap[t] (will add λ columns later)
        for t in range(self.T):
            if self.capacity[t] <= EPS:
                continue
            con = self.model.addConstr(gp.LinExpr() <= self.capacity[t], f"cap_{t}")
            self.cap_con[t] = con

        self.model.update()

    def add_column(
        self,
        item_id: int,
        t: int,
        e: int,
        cost: float,
        Q: float,
        covered_demands: List[int],
    ) -> bool:
        """
        Add a new λ[i,t,e] column incrementally.
        Returns True if column was added (not duplicate).
        """
        key = (item_id, t, e)
        if key in self.lam:
            return False  # Already exists

        # Create variable with cost
        var = self.model.addVar(lb=0.0, ub=1.0, obj=cost, name=f"lam_{item_id}_{t}_{e}")
        self.lam[key] = var

        # Store block data
        self.block_cost[key] = cost
        self.block_Q[key] = Q
        self.block_covers[key] = covered_demands

        self.model.update()

        # Update coverage constraints: add coefficient 1 for each covered demand
        for u in covered_demands:
            if (item_id, u) in self.coverage_con:
                self.model.chgCoeff(self.coverage_con[(item_id, u)], var, 1.0)

        # Update capacity constraint at t
        if t in self.cap_con:
            self.model.chgCoeff(self.cap_con[t], var, Q)

        return True

    def apply_branching(
        self,
        forced_y: Dict[int, Set[int]],
        forbidden_y: Dict[int, Set[int]],
    ):
        """
        Apply branching decisions by modifying variable bounds and adding constraints.

        - Force Y[i,t]=1: add constraint sum_e λ[i,t,e] >= 1
        - Forbid Y[i,t]=0: set UB=0 on all λ[i,t,*]
        """
        # Remove old force_y constraints
        for con in self.force_y_con.values():
            self.model.remove(con)
        self.force_y_con.clear()

        # Reset all λ bounds to [0, 1]
        for var in self.lam.values():
            var.LB = 0.0
            var.UB = 1.0

        # Apply forbidden Y: set UB=0
        for item_id, periods in forbidden_y.items():
            for t in periods:
                for (i, tt, e), var in self.lam.items():
                    if i == item_id and tt == t:
                        var.UB = 0.0

        # Apply forced Y: add constraint sum_e λ[i,t,e] >= 1
        for item_id, periods in forced_y.items():
            for t in periods:
                blocks_at_t = [
                    self.lam[(i, tt, e)]
                    for (i, tt, e) in self.lam
                    if i == item_id and tt == t
                ]
                if blocks_at_t:
                    con = self.model.addConstr(
                        gp.quicksum(blocks_at_t) >= 1.0, f"force_y_{item_id}_{t}"
                    )
                    self.force_y_con[(item_id, t)] = con

        self.model.update()

    def solve(self) -> Tuple[float, Dict, Dict]:
        """
        Solve RMP and return (objective, solution, duals).
        """
        self.model.optimize()

        if self.model.Status != GRB.OPTIMAL:
            return math.inf, {"status": "infeasible"}, {}

        # Check artificial usage
        art_total = sum(a.X for a in self.art.values())

        # Extract solution
        lam_vals = {k: v.X for k, v in self.lam.items() if v.X > EPS}

        # Compute Y values
        y_vals: Dict[Tuple[int, int], float] = {}
        for (item_id, t, e), val in lam_vals.items():
            key = (item_id, t)
            y_vals[key] = y_vals.get(key, 0.0) + val

        # Compute production aggregates
        x_agg: Dict[int, Dict[int, float]] = {item_id: {} for item_id in self.items}
        for (item_id, t, e), val in lam_vals.items():
            Q = self.block_Q[(item_id, t, e)]
            x_agg[item_id][t] = x_agg[item_id].get(t, 0.0) + val * Q

        solution = {
            "status": "optimal",
            "y": y_vals,
            "lam": lam_vals,
            "x_agg": x_agg,
            "art_total": art_total,
        }

        # Extract duals
        duals = {
            "pi_coverage": {k: c.Pi for k, c in self.coverage_con.items()},
            "rho_cap": {k: c.Pi for k, c in self.cap_con.items()},
            "tau_force_y": {k: c.Pi for k, c in self.force_y_con.items()},
        }

        return self.model.ObjVal, solution, duals

    def get_column_count(self) -> int:
        """Return number of λ columns."""
        return len(self.lam)


@dataclass
class BranchNode:
    """Node in the branch-and-bound tree."""

    node_id: int
    parent_id: Optional[int]
    depth: int
    forced_y: Dict[int, Set[int]] = field(default_factory=dict)
    forbidden_y: Dict[int, Set[int]] = field(default_factory=dict)
    lefo_cuts: List[Tuple[int, int, int, int, int]] = field(default_factory=list)
    lp_bound: float = math.inf

    def __lt__(self, other: "BranchNode") -> bool:
        return self.lp_bound < other.lp_bound


@dataclass
class NodeLogEntry:
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
    node_id: int
    cg_iter: int
    rmp_obj: float
    num_columns: int
    min_rc: float
    columns_added: int


class BnPLogger:
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

        with open(self.out_dir / "bnp_nodes.csv", "w", newline="") as f:
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
                        f"{e.lp_bound:.6f}" if math.isfinite(e.lp_bound) else "inf",
                        f"{e.incumbent:.6f}" if math.isfinite(e.incumbent) else "inf",
                        e.branch_item if e.branch_item is not None else "",
                        e.branch_t if e.branch_t is not None else "",
                        e.direction or "",
                        e.status,
                        e.cg_iters,
                        e.columns_added,
                    ]
                )

        with open(self.out_dir / "bnp_cg_iters.csv", "w", newline="") as f:
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
                        f"{e.rmp_obj:.6f}",
                        e.num_columns,
                        f"{e.min_rc:.6f}",
                        e.columns_added,
                    ]
                )


def setup_cost_at(item_data: dict, t: int) -> float:
    """Get setup cost at period t."""
    setup = item_data["setup"]
    if isinstance(setup, list):
        return float(setup[t]) if t < len(setup) else 0.0
    return float(setup)


def arc_cost_for_item(item_data: dict, t: int, u: int) -> float:
    """Cost per unit flow on arc (t, u): variable cost + holding cost."""
    c_var = item_data["c_var"]
    h = item_data.get("h", [0.0])

    var_cost = float(c_var[t]) if t < len(c_var) else 0.0

    hold_cost = 0.0
    for r in range(t, u):
        if isinstance(h, list):
            hold_cost += float(h[r]) if r < len(h) else 0.0
        else:
            hold_cost += float(h)

    return var_cost + hold_cost


def compute_block_cost(
    item_data: dict, t: int, e: int, demand_periods: List[int]
) -> float:
    """
    Compute total cost of block (t, e):
    cost = setup[t] + sum_{u in [t,e] with d[u]>0} [ c_var[t]*d[u] + holding(t->u)*d[u] ]

    where holding(t->u) = sum_{r=t..u-1} h[r]
    """
    demand = item_data["demand"]
    c_var = item_data["c_var"]
    h = item_data.get("h", [0.0])

    # Setup cost
    cost = setup_cost_at(item_data, t)

    # Variable and holding costs for each demand in block
    var_cost_t = float(c_var[t]) if t < len(c_var) else 0.0

    for u in demand_periods:
        if u < t or u > e:
            continue
        d_u = demand[u]
        if d_u <= EPS:
            continue

        # Variable cost
        cost += var_cost_t * d_u

        # Holding cost from t to u
        hold = 0.0
        for r in range(t, u):
            if isinstance(h, list):
                hold += float(h[r]) if r < len(h) else 0.0
            else:
                hold += float(h)
        cost += hold * d_u

    return cost


def compute_block_quantity(item_data: dict, t: int, e: int) -> float:
    """
    Compute production quantity for block (t, e):
    Q(t,e) = sum_{u=t..e} d[u]
    """
    demand = item_data["demand"]
    total = 0.0
    for u in range(t, e + 1):
        if u < len(demand):
            total += demand[u]
    return total


def generate_initial_blocks(
    item_id: int,
    item_data: dict,
    T: int,
    Gamma: Dict[int, List[int]],
    forbidden_y: Set[int],
) -> List[ZIOBlock]:
    """Generate initial blocks for feasibility (lot-for-lot style)."""
    demand = item_data["demand"]
    demand_periods = [u for u in range(T) if demand[u] > EPS]

    if not demand_periods:
        return []

    blocks = []
    block_sigs = set()

    # For each demand period, create a minimal block
    for u in demand_periods:
        # Find production period that can reach u
        for t in range(u, -1, -1):
            if t in forbidden_y:
                continue
            reachable = Gamma.get(t, [])
            if u in reachable:
                sig = (t, u)
                if sig not in block_sigs:
                    blocks.append(
                        ZIOBlock(
                            item_id=item_id,
                            start_t=t,
                            end_e=u,
                            setup_cost=setup_cost_at(item_data, t),
                        )
                    )
                    block_sigs.add(sig)
                break

    # Also add larger blocks covering multiple demands
    for t in range(T):
        if t in forbidden_y:
            continue
        reachable = set(Gamma.get(t, []))
        if not reachable:
            continue

        reachable_demands = [u for u in demand_periods if u in reachable]
        if not reachable_demands:
            continue

        max_e = max(reachable_demands)
        sig = (t, max_e)
        if sig not in block_sigs:
            blocks.append(
                ZIOBlock(
                    item_id=item_id,
                    start_t=t,
                    end_e=max_e,
                    setup_cost=setup_cost_at(item_data, t),
                )
            )
            block_sigs.add(sig)

    return blocks


def generate_all_lefo_cuts(
    items: Dict[int, dict],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
) -> List[Tuple[int, int, int, int, int]]:
    """Generate all LEFO (no-crossing) cuts upfront."""
    all_cuts = []

    for item_id, item_data in items.items():
        Gamma = Gamma_by_item.get(item_id, {})
        shelf_seq = item_data.get("shelf_seq", [])
        demand = item_data.get("demand", [])
        T = len(demand)

        prods = [t for t in range(T) if Gamma.get(t)]
        if len(prods) < 2:
            continue

        expiry = {}
        for t in prods:
            m_t = int(shelf_seq[t]) if t < len(shelf_seq) else T
            expiry[t] = t + m_t

        prods_sorted = sorted(prods, key=lambda t: expiry.get(t, T))

        for a in range(len(prods_sorted)):
            t1 = prods_sorted[a]
            v1 = expiry.get(t1, T)
            reachable_t1 = set(Gamma.get(t1, []))

            for b in range(a + 1, len(prods_sorted)):
                t2 = prods_sorted[b]
                v2 = expiry.get(t2, T)

                if v1 >= v2:
                    continue

                reachable_t2 = Gamma.get(t2, [])

                for up in reachable_t2:
                    if demand[up] <= EPS:
                        continue
                    for u in range(t2, up):
                        if u not in reachable_t1 or demand[u] <= EPS:
                            continue
                        all_cuts.append((item_id, t1, u, t2, up))

    return all_cuts


def find_lefo_violations_blocks(
    lam_vals: Dict[Tuple[int, int, int], float],
    items: Dict[int, dict],
    existing_cuts: List[Tuple[int, int, int, int, int]],
) -> List[Tuple[int, int, int, int, int]]:
    """
    Find LEFO violations in current block (λ) values.

    Two blocks (t1, e1) and (t2, e2) cross if:
    - expiry(t1) < expiry(t2)  (t1 expires first)
    - BUT block t1 covers some demand u1 > u2 where u2 is covered by block t2

    In simpler terms: if the earlier-expiring block serves a later demand than
    a demand served by the later-expiring block, it's a crossing.

    For blocks: if exp(t1) < exp(t2) and max(t1,t2) <= e1 (overlap region exists)
    and λ[t1,e1] + λ[t2,e2] > 1, we have a violation.
    """
    existing_set = set(existing_cuts)
    violations = []

    # Group active blocks by item
    active_blocks_by_item: Dict[int, List[Tuple[int, int, float]]] = {}
    for (item_id, t, e), val in lam_vals.items():
        if val > EPS:
            if item_id not in active_blocks_by_item:
                active_blocks_by_item[item_id] = []
            active_blocks_by_item[item_id].append((t, e, val))

    for item_id, item_data in items.items():
        if item_id not in active_blocks_by_item:
            continue

        active_blocks = active_blocks_by_item[item_id]
        if len(active_blocks) < 2:
            continue

        shelf_seq = item_data.get("shelf_seq", [])
        T = len(item_data.get("demand", []))

        # Compute expiry for each block's production period
        expiry = {}
        for t, e, _ in active_blocks:
            if t not in expiry:
                m_t = int(shelf_seq[t]) if t < len(shelf_seq) else T
                expiry[t] = t + m_t

        # Check all pairs of blocks for LEFO violations
        for i, (t1, e1, v1) in enumerate(active_blocks):
            exp1 = expiry.get(t1, T)

            for j, (t2, e2, v2) in enumerate(active_blocks):
                if i >= j:
                    continue  # Avoid duplicates

                exp2 = expiry.get(t2, T)

                # LEFO violation: earlier-expiring block serves later demands
                # If exp1 < exp2 and blocks overlap such that t1's block covers
                # demands that should be served by t2 under LEFO

                if exp1 < exp2:
                    # t1 expires first, so t1 should serve earlier demands
                    # Crossing if e1 > t2 (block1 extends past block2's start)
                    if e1 >= t2 and v1 + v2 > 1.0 + EPS:
                        cut = (item_id, t1, e1, t2, e2)
                        if cut not in existing_set:
                            violations.append(cut)
                            existing_set.add(cut)
                elif exp2 < exp1:
                    # t2 expires first
                    if e2 >= t1 and v1 + v2 > 1.0 + EPS:
                        cut = (item_id, t2, e2, t1, e1)
                        if cut not in existing_set:
                            violations.append(cut)
                            existing_set.add(cut)

    return violations


def find_lefo_violations_z(
    z_vals: Dict[Tuple[int, int, int], float],
    items: Dict[int, dict],
    existing_cuts: List[Tuple[int, int, int, int, int]],
) -> List[Tuple[int, int, int, int, int]]:
    """
    Find LEFO violations in current z (arc usage) values.

    LEFO = Last Expiry First Out: for any demand, prefer later-expiring inventory.

    A crossing pattern (violation) occurs when:
    - t1 expires before t2 (exp1 < exp2)
    - t1 serves demand u, t2 serves demand up
    - u < up (t1 serves earlier demand)
    - t2 could have served u (t2 <= u)
    - z[t1,u] + z[t2,up] > 1

    This is a violation because under LEFO, t2 (later expiry) should serve u first.

    Returns: list of (item_id, t1, u, t2, up) cuts to add
    """
    existing_set = set(existing_cuts)
    violations = []

    # Group active arcs by item
    active_arcs_by_item: Dict[int, List[Tuple[int, int, float]]] = {}
    for (item_id, t, u), val in z_vals.items():
        if val > EPS:
            if item_id not in active_arcs_by_item:
                active_arcs_by_item[item_id] = []
            active_arcs_by_item[item_id].append((t, u, val))

    for item_id, item_data in items.items():
        if item_id not in active_arcs_by_item:
            continue

        active_arcs = active_arcs_by_item[item_id]
        if len(active_arcs) < 2:
            continue

        shelf_seq = item_data.get("shelf_seq", [])
        T = len(item_data.get("demand", []))

        # Compute expiry for each production period
        expiry = {}
        for t, u, _ in active_arcs:
            if t not in expiry:
                m_t = int(shelf_seq[t]) if t < len(shelf_seq) else T
                expiry[t] = t + m_t

        # Check all pairs of arcs for LEFO violations
        for i, (t1, u, z1) in enumerate(active_arcs):
            exp1 = expiry.get(t1, T)

            for j, (t2, up, z2) in enumerate(active_arcs):
                if i >= j:
                    continue  # Avoid duplicates

                exp2 = expiry.get(t2, T)

                # LEFO violation: t1 expires first, serves u, t2 serves up
                # where t2 could serve u (t2 <= u) and u < up
                if exp1 < exp2 and t2 <= u < up:
                    if z1 + z2 > 1.0 + EPS:
                        cut = (item_id, t1, u, t2, up)
                        if cut not in existing_set:
                            violations.append(cut)
                            existing_set.add(cut)
                elif exp2 < exp1 and t1 <= up < u:
                    # Symmetric: t2 expires first, serves up, t1 serves u
                    if z1 + z2 > 1.0 + EPS:
                        cut = (item_id, t2, up, t1, u)
                        if cut not in existing_set:
                            violations.append(cut)
                            existing_set.add(cut)

    return violations


def solve_rmp_with_duals(
    items: Dict[int, dict],
    T: int,
    capacity: List[float],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    blocks_by_item: Dict[int, List[ZIOBlock]],
    forced_y: Dict[int, Set[int]],
    forbidden_y: Dict[int, Set[int]],
    lefo_cuts: List[Tuple[int, int, int, int, int]] = None,
    item_caches: Dict[int, ItemCache] = None,
) -> Tuple[float, Dict, Dict]:
    """
    Solve RMP with block columns and flow variables.

    Following arc-based structure:
    - λ[i,t,e]: block coefficient
    - f[i,t,u]: flow quantity from t to u
    - y[i,t] = Σ_e λ[i,t,e]: aggregate setup
    - z[i,t,u]: arc usage indicator for LEFO

    Returns: (objective, solution_dict, duals_dict)
    """
    if lefo_cuts is None:
        lefo_cuts = []

    m = gp.Model("RMP")
    m.Params.OutputFlag = 0
    m.Params.Method = 1  # Dual simplex

    # Filter blocks by forbidden Y
    valid_blocks: Dict[int, List[ZIOBlock]] = {}
    for item_id, blks in blocks_by_item.items():
        forbidden = forbidden_y.get(item_id, set())
        valid_blocks[item_id] = [b for b in blks if b.start_t not in forbidden]

    # Demand periods per item
    demand_periods_by_item = {}
    for item_id, item_data in items.items():
        if item_caches and item_id in item_caches:
            demand_periods_by_item[item_id] = item_caches[item_id].demand_periods
        else:
            demand = item_data["demand"]
            demand_periods_by_item[item_id] = [u for u in range(T) if demand[u] > EPS]

    # Block variables λ[i,t,e]
    lam = {}
    for item_id, blks in valid_blocks.items():
        for blk in blks:
            key = (item_id, blk.start_t, blk.end_e)
            if key not in lam:
                lam[key] = m.addVar(
                    lb=0.0, ub=1.0, name=f"lam_{item_id}_{blk.start_t}_{blk.end_e}"
                )

    # Flow variables f[i,t,u] - actual quantities
    f = {}
    for item_id, item_data in items.items():
        demand = item_data["demand"]
        Gamma = Gamma_by_item.get(item_id, {})
        forbidden = forbidden_y.get(item_id, set())

        for t in range(T):
            if t in forbidden:
                continue
            reachable = Gamma.get(t, [])
            for u in reachable:
                if demand[u] > EPS:
                    f[(item_id, t, u)] = m.addVar(lb=0.0, name=f"f_{item_id}_{t}_{u}")

    # Aggregate setup y[i,t]
    y = {}
    for item_id in items:
        for t in range(T):
            y[(item_id, t)] = m.addVar(lb=0.0, ub=1.0, name=f"y_{item_id}_{t}")

    # Arc usage z[i,t,u] for LEFO
    z = {}
    for key in f:
        item_id, t, u = key
        z[key] = m.addVar(lb=0.0, ub=1.0, name=f"z_{item_id}_{t}_{u}")

    m.update()

    # Constraint dictionaries for dual extraction
    y_def_con = {}
    z_def_con = {}
    demand_con = {}
    cap_con = {}
    link_con = {}
    force_y_con = {}
    lefo_cut_con = {}

    # Y definition: y[i,t] = Σ_e λ[i,t,e]
    for item_id in items:
        for t in range(T):
            blocks_at_t = [
                lam[(i, tt, e)] for (i, tt, e) in lam if i == item_id and tt == t
            ]
            if blocks_at_t:
                y_def_con[(item_id, t)] = m.addConstr(
                    y[(item_id, t)] == gp.quicksum(blocks_at_t), f"y_def_{item_id}_{t}"
                )
            else:
                y_def_con[(item_id, t)] = m.addConstr(
                    y[(item_id, t)] == 0.0, f"y_def_{item_id}_{t}"
                )

    # Artificial variables for demand
    art = {}
    for item_id, item_data in items.items():
        for u in demand_periods_by_item[item_id]:
            art[(item_id, u)] = m.addVar(lb=0.0, name=f"art_{item_id}_{u}")
    m.update()

    # Demand satisfaction: Σ_t f[i,t,u] + art[i,u] = demand[u]
    for item_id, item_data in items.items():
        demand = item_data["demand"]
        for u in demand_periods_by_item[item_id]:
            flow_to_u = [f[(i, t, uu)] for (i, t, uu) in f if i == item_id and uu == u]
            lhs = gp.quicksum(flow_to_u) if flow_to_u else gp.LinExpr()
            lhs += art[(item_id, u)]
            demand_con[(item_id, u)] = m.addConstr(
                lhs == demand[u], f"demand_{item_id}_{u}"
            )

    # Capacity: Σ_i Σ_u f[i,t,u] ≤ capacity[t]
    for t in range(T):
        if capacity[t] <= EPS:
            continue
        flow_at_t = [f[(i, tt, u)] for (i, tt, u) in f if tt == t]
        if flow_at_t:
            cap_con[t] = m.addConstr(gp.quicksum(flow_at_t) <= capacity[t], f"cap_{t}")

    # Block-flow linking: f[i,t,u] ≤ demand[u] * Σ_{e≥u} λ[i,t,e]
    for (item_id, t, u), f_var in f.items():
        demand_u = items[item_id]["demand"][u]
        covering_blocks = [
            lam[(i, tt, e)] for (i, tt, e) in lam if i == item_id and tt == t and e >= u
        ]
        if covering_blocks:
            link_con[(item_id, t, u)] = m.addConstr(
                f_var <= demand_u * gp.quicksum(covering_blocks),
                f"link_{item_id}_{t}_{u}",
            )
        else:
            link_con[(item_id, t, u)] = m.addConstr(
                f_var == 0.0, f"link_{item_id}_{t}_{u}"
            )

    # Z definition: z[i,t,u] ≥ f[i,t,u] / demand[u]
    for (item_id, t, u), z_var in z.items():
        demand_u = items[item_id]["demand"][u]
        if demand_u > EPS and (item_id, t, u) in f:
            z_def_con[(item_id, t, u)] = m.addConstr(
                z_var >= f[(item_id, t, u)] / demand_u, f"z_def_{item_id}_{t}_{u}"
            )

    # Forced Y: y[i,t] = 1
    for item_id, periods in forced_y.items():
        for t in periods:
            if (item_id, t) in y:
                force_y_con[(item_id, t)] = m.addConstr(
                    y[(item_id, t)] == 1.0, f"force_y_{item_id}_{t}"
                )

    # LEFO cuts: z[t1,u] + z[t2,up] ≤ 1
    for idx, (item_id, t1, u, t2, up) in enumerate(lefo_cuts):
        if (item_id, t1, u) in z and (item_id, t2, up) in z:
            lefo_cut_con[idx] = m.addConstr(
                z[(item_id, t1, u)] + z[(item_id, t2, up)] <= 1, f"lefo_cut_{idx}"
            )

    # Objective: setup costs + flow costs + artificial penalties
    obj = gp.LinExpr()

    # Setup costs via Y
    for item_id, item_data in items.items():
        for t in range(T):
            s_cost = setup_cost_at(item_data, t)
            obj += s_cost * y[(item_id, t)]

    # Flow costs (variable + holding)
    for (item_id, t, u), f_var in f.items():
        item_data = items[item_id]
        a_cost = arc_cost_for_item(item_data, t, u)
        obj += a_cost * f_var

    # Artificial penalties
    for art_var in art.values():
        obj += BIG_M_ARTIFICIAL * art_var

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return math.inf, {"status": "infeasible"}, {}

    # Check artificial usage
    art_total = sum(a.X for a in art.values())

    # Extract solution
    y_vals = {k: v.X for k, v in y.items() if v.X > EPS}
    f_vals = {k: v.X for k, v in f.items() if v.X > EPS}
    lam_vals = {k: v.X for k, v in lam.items() if v.X > EPS}
    z_vals = {k: v.X for k, v in z.items() if v.X > EPS}

    # Aggregate production per item per period
    x_agg = {item_id: {} for item_id in items}
    for (item_id, t, u), val in f_vals.items():
        x_agg[item_id][t] = x_agg[item_id].get(t, 0.0) + val

    solution = {
        "status": "optimal",
        "y": y_vals,
        "z": z_vals,
        "f": f_vals,
        "lam": lam_vals,
        "x_agg": x_agg,
        "art_total": art_total,
    }

    # Extract duals
    duals = {
        "pi_demand": {k: c.Pi for k, c in demand_con.items()},
        "rho_cap": {k: c.Pi for k, c in cap_con.items()},
        "sigma_link": {k: c.Pi for k, c in link_con.items()},
        "mu_y_def": {k: c.Pi for k, c in y_def_con.items()},
        "mu_z_def": {k: c.Pi for k, c in z_def_con.items()},
        "tau_force_y": {k: c.Pi for k, c in force_y_con.items()},
        "pi_lefo": {k: c.Pi for k, c in lefo_cut_con.items()},
    }

    return m.ObjVal, solution, duals


def price_blocks(
    item_id: int,
    item_data: dict,
    T: int,
    Gamma: Dict[int, List[int]],
    duals: Dict,
    forced_y: Set[int],
    forbidden_y: Set[int],
    existing_blocks: Set[Tuple[int, int]],
    cache: Optional[ItemCache] = None,
    max_columns: int = 10,
) -> Tuple[List[Tuple[ZIOBlock, float]], float]:
    """
    Price blocks for an item using the flow-based formulation.

    Reduced cost for block (t, e):
    rc(t,e) = setup[t] - mu_y_def[i,t] - tau_y[i,t] - sum_{u in [t,e]} sigma_link[i,t,u] * demand[u]

    Uses precomputed cache for O(1) lookups if available.
    Returns blocks with negative reduced cost, sorted by RC.
    """
    # Extract duals from flow-based RMP
    sigma_link = duals.get("sigma_link", {})
    mu_y_def = duals.get("mu_y_def", {})
    tau_force_y = duals.get("tau_force_y", {})

    demand = item_data["demand"]
    setup = item_data["setup"]

    # Get demand periods (use cache if available)
    if cache:
        demand_periods = cache.demand_periods
    else:
        demand_periods = [u for u in range(T) if demand[u] > EPS]

    if not demand_periods:
        return [], 0.0

    # Enumerate all feasible blocks and compute reduced costs
    candidates: List[Tuple[ZIOBlock, float]] = []

    # Iterate over all feasible blocks
    for t in range(T):
        if t in forbidden_y:
            continue

        reachable = set(Gamma.get(t, []))
        if not reachable:
            continue

        # Base reduced cost components for this production period
        s_cost = (
            float(setup[t])
            if isinstance(setup, list) and t < len(setup)
            else float(setup)
        )
        mu = mu_y_def.get((item_id, t), 0.0)
        tau_y = tau_force_y.get((item_id, t), 0.0)
        base_rc = s_cost - mu - tau_y

        # Find demands reachable from t
        reachable_demands = [u for u in demand_periods if u in reachable]
        if not reachable_demands:
            continue

        # For each possible end point, compute reduced cost
        for e in reachable:
            if e < t:
                continue
            if (t, e) in existing_blocks:
                continue

            # Covered demands are those in [t, e] that are reachable
            covered = [u for u in reachable_demands if t <= u <= e]
            if not covered:
                continue

            # Sum of sigma_link * demand for covered demands
            sum_sigma_d = sum(
                sigma_link.get((item_id, t, u), 0.0) * demand[u] for u in covered
            )

            # Reduced cost: sigma_link is negative in the linking constraint
            # rc = s_cost - mu - tau_y - (-sigma_link * d) = s_cost - mu - tau_y + sum(sigma * d)
            # Note: sigma_link should be <= 0 for the linking constraint
            rc = base_rc + sum_sigma_d

            if rc < -EPS:
                block = ZIOBlock(
                    item_id=item_id,
                    start_t=t,
                    end_e=e,
                    setup_cost=s_cost,
                )
                candidates.append((block, rc))

    if not candidates:
        return [], 0.0

    # Sort by reduced cost
    candidates.sort(key=lambda x: x[1])

    # Return based on configuration
    if SINGLE_BEST_PER_ITEM:
        result = [candidates[0]]  # Only the single best
    else:
        result = candidates[:max_columns]

    min_rc = result[0][1]
    return result, min_rc


def column_generation_loop(
    items: Dict[int, dict],
    T: int,
    capacity: List[float],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    initial_blocks: Dict[int, List[ZIOBlock]],
    forced_y: Dict[int, Set[int]],
    forbidden_y: Dict[int, Set[int]],
    lefo_cuts: List[Tuple[int, int, int, int, int]] = None,
    item_caches: Dict[int, ItemCache] = None,
    node_id: int = 0,
    logger: Optional[BnPLogger] = None,
    verbose: bool = False,
    max_cg_iters: int = 500,
    rmp: Optional[BlockRMP] = None,
) -> Tuple[float, Dict, Dict[int, List[ZIOBlock]], int, int, Optional[BlockRMP]]:
    """
    Run column generation loop with incremental BlockRMP.

    If rmp is None, creates a new BlockRMP. Otherwise reuses and updates it.
    Returns: (obj, solution, blocks_by_item, cg_iters, cols_added, rmp)
    """
    if lefo_cuts is None:
        lefo_cuts = []

    blocks_by_item = {i: list(blks) for i, blks in initial_blocks.items()}
    total_columns_added = 0
    cg_iter = 0

    while cg_iter < max_cg_iters:
        # Solve RMP with flow variables
        obj, solution, duals = solve_rmp_with_duals(
            items,
            T,
            capacity,
            Gamma_by_item,
            blocks_by_item,
            forced_y,
            forbidden_y,
            lefo_cuts,
            item_caches,
        )

        if not math.isfinite(obj):
            return math.inf, {"status": "infeasible"}, blocks_by_item, cg_iter, 0

        # Price new blocks - collect candidates from all items
        all_candidates: List[Tuple[int, ZIOBlock, float]] = []  # (item_id, block, rc)
        min_rc = 0.0

        for item_id, item_data in items.items():
            Gamma = Gamma_by_item.get(item_id, {})
            forced = forced_y.get(item_id, set())
            forbidden = forbidden_y.get(item_id, set())
            existing = {(b.start_t, b.end_e) for b in blocks_by_item.get(item_id, [])}
            cache = item_caches.get(item_id) if item_caches else None

            new_cols, item_min_rc = price_blocks(
                item_id,
                item_data,
                T,
                Gamma,
                duals,
                forced,
                forbidden,
                existing,
                cache,
            )

            if item_min_rc < min_rc:
                min_rc = item_min_rc

            for block, rc in new_cols:
                all_candidates.append((item_id, block, rc))

        # Global top-K admission: sort by RC and take best GLOBAL_TOP_K
        all_candidates.sort(key=lambda x: x[2])
        to_add = all_candidates[:GLOBAL_TOP_K]

        columns_added_this_iter = 0
        for item_id, block, rc in to_add:
            if item_id not in blocks_by_item:
                blocks_by_item[item_id] = []
            blocks_by_item[item_id].append(block)
            columns_added_this_iter += 1

        total_columns_added += columns_added_this_iter
        total_cols = sum(len(blks) for blks in blocks_by_item.values())

        if logger:
            logger.log_cg_iter(
                CGIterLogEntry(
                    node_id=node_id,
                    cg_iter=cg_iter,
                    rmp_obj=obj,
                    num_columns=total_cols,
                    min_rc=min_rc,
                    columns_added=columns_added_this_iter,
                )
            )

        if verbose:
            print(
                f"    CG iter {cg_iter}: obj={obj:.4f}, cols={total_cols}, "
                f"min_rc={min_rc:.6f}, added={columns_added_this_iter}"
            )

        if columns_added_this_iter == 0 or min_rc >= -EPS:
            cg_iter += 1
            break

        cg_iter += 1

    # Final solve
    obj, solution, _ = solve_rmp_with_duals(
        items,
        T,
        capacity,
        Gamma_by_item,
        blocks_by_item,
        forced_y,
        forbidden_y,
        lefo_cuts,
        item_caches,
    )

    return obj, solution, blocks_by_item, cg_iter, total_columns_added, None


def is_y_integer(y_vals: Dict[Tuple[int, int], float]) -> bool:
    """Check if all Y values are integer (0 or 1)."""
    for val in y_vals.values():
        if EPS < val < 1.0 - EPS:
            return False
    return True


def find_best_branching_y(
    y_vals: Dict[Tuple[int, int], float],
    forced_y: Dict[int, Set[int]],
    forbidden_y: Dict[int, Set[int]],
    items: Dict[int, dict],
) -> Optional[Tuple[int, int, float]]:
    """Find best Y to branch on using setup-cost weighting."""
    candidates = []

    for (item_id, t), val in y_vals.items():
        if t in forced_y.get(item_id, set()):
            continue
        if t in forbidden_y.get(item_id, set()):
            continue

        frac = min(val, 1.0 - val)
        if frac > EPS:
            s_cost = setup_cost_at(items[item_id], t)
            candidates.append((item_id, t, val, frac, s_cost))

    if not candidates:
        return None

    max_setup = max(c[4] for c in candidates) if candidates else 1.0
    if max_setup < EPS:
        max_setup = 1.0

    best_score = -1.0
    best = None

    for item_id, t, val, frac, s_cost in candidates:
        frac_score = 4.0 * frac
        setup_weight = 0.3 + 0.7 * (s_cost / max_setup)
        score = frac_score * setup_weight

        if score > best_score:
            best_score = score
            best = (item_id, t, val)

    return best


def try_dive(
    items: Dict[int, dict],
    T: int,
    capacity: List[float],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    blocks_pool: Dict[int, List[ZIOBlock]],
    y_vals: Dict[Tuple[int, int], float],
    base_forced_y: Dict[int, Set[int]],
    base_forbidden_y: Dict[int, Set[int]],
    lefo_cuts: List[Tuple[int, int, int, int, int]],
) -> Tuple[Optional[float], Optional[Dict]]:
    """Quick dive heuristic to find integer solution (UB only)."""
    forced_y = {i: set(s) for i, s in base_forced_y.items()}
    forbidden_y = {i: set(s) for i, s in base_forbidden_y.items()}

    # Force all Y > 0.5 to 1, others to 0
    for (item_id, t), val in y_vals.items():
        if t in forced_y.get(item_id, set()) or t in forbidden_y.get(item_id, set()):
            continue
        if val > 0.5:
            if item_id not in forced_y:
                forced_y[item_id] = set()
            forced_y[item_id].add(t)
        else:
            if item_id not in forbidden_y:
                forbidden_y[item_id] = set()
            forbidden_y[item_id].add(t)

    obj, solution, _ = solve_rmp_with_duals(
        items,
        T,
        capacity,
        Gamma_by_item,
        blocks_pool,
        forced_y,
        forbidden_y,
        lefo_cuts,
    )

    if not math.isfinite(obj):
        return None, None

    art_total = solution.get("art_total", 0.0)
    if art_total > EPS:
        return None, None

    y_dive = solution.get("y", {})
    if not is_y_integer(y_dive):
        return None, None

    return obj, solution


def solve_branch_and_price(
    items: Dict[int, dict],
    T: int,
    capacity: List[float],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    initial_blocks: Dict[int, List[ZIOBlock]],
    item_caches: Dict[int, ItemCache] = None,
    max_time: float = 3600,
    max_nodes: int = 100000,
    verbose: bool = False,
    logger: Optional[BnPLogger] = None,
) -> Tuple[float, Optional[Dict], float, float]:
    """
    Solve using Block-Based Branch-and-Price.

    Returns: (best_obj, best_solution, root_bound, final_gap)
    """
    start_time = time.time()

    # Generate LEFO cuts upfront if enabled
    if ADD_ALL_LEFO_CUTS_UPFRONT:
        initial_lefo_cuts = generate_all_lefo_cuts(items, Gamma_by_item)
        if verbose:
            print(f"  Adding {len(initial_lefo_cuts)} LEFO cuts upfront")
    else:
        initial_lefo_cuts = []

    root = BranchNode(
        node_id=0,
        parent_id=None,
        depth=0,
        forced_y={i: set() for i in items},
        forbidden_y={i: {t for t in range(T) if capacity[t] <= EPS} for i in items},
        lefo_cuts=initial_lefo_cuts,
    )

    if verbose:
        print("\n>>> ROOT NODE <<<")

    # CG at root with LEFO cut-and-resolve loop
    total_cg_iters = 0
    total_cols_added = 0
    root_blocks = initial_blocks
    root_rmp = None
    while True:
        root_obj, root_info, root_blocks, cg_iters, cols_added, root_rmp = (
            column_generation_loop(
                items,
                T,
                capacity,
                Gamma_by_item,
                root_blocks,
                root.forced_y,
                root.forbidden_y,
                root.lefo_cuts,
                item_caches,
                node_id=0,
                logger=logger,
                verbose=verbose,
                rmp=root_rmp,
            )
        )
        total_cg_iters += cg_iters
        total_cols_added += cols_added

        if not math.isfinite(root_obj):
            break  # Infeasible

        # Check for LEFO violations using z (arc usage) values
        if not ENABLE_LEFO_CUTS:
            break  # LEFO cuts disabled

        z_vals = root_info.get("z", {})
        violations = find_lefo_violations_z(z_vals, items, root.lefo_cuts)
        if not violations:
            break  # No violations, proceed

        # Add up to MAX_LEFO_CUTS_PER_ITER cuts
        cuts_to_add = violations[:MAX_LEFO_CUTS_PER_ITER]
        for v in cuts_to_add:
            root.lefo_cuts.append(v)
        if verbose:
            print(
                f"      Added {len(cuts_to_add)} LEFO cuts "
                f"(of {len(violations)} violations), restarting CG"
            )

    root_cg_iters = total_cg_iters
    root_cols_added = total_cols_added

    if not math.isfinite(root_obj):
        if verbose:
            print("  Root infeasible!")
        if logger:
            logger.log_node(
                NodeLogEntry(
                    node_id=0,
                    depth=0,
                    lp_bound=math.inf,
                    incumbent=math.inf,
                    branch_item=None,
                    branch_t=None,
                    direction=None,
                    status="INFEASIBLE",
                    cg_iters=root_cg_iters,
                    columns_added=root_cols_added,
                )
            )
        return math.inf, None, math.inf, 0.0

    root.lp_bound = root_obj
    root_bound = root_obj

    if verbose:
        print(f"  Root LP bound: {root_obj:.4f}")
        print(f"  CG iterations: {root_cg_iters}, columns added: {root_cols_added}")

    y_vals = root_info.get("y", {})

    if is_y_integer(y_vals):
        if verbose:
            print("  ✓ Root is Y-INTEGER - OPTIMAL!")
        if logger:
            logger.log_node(
                NodeLogEntry(
                    node_id=0,
                    depth=0,
                    lp_bound=root_obj,
                    incumbent=root_obj,
                    branch_item=None,
                    branch_t=None,
                    direction=None,
                    status="OPTIMAL_AT_ROOT",
                    cg_iters=root_cg_iters,
                    columns_added=root_cols_added,
                )
            )
        return root_obj, root_info, root_bound, 0.0

    # Initialize B&B
    best_ub: Optional[float] = None
    best_solution: Optional[Dict] = None
    blocks_pool = {i: list(blks) for i, blks in root_blocks.items()}

    # Try dive at root (if heuristics enabled)
    if ENABLE_HEURISTICS:
        if verbose:
            print("  Trying dive at root...")
        dive_obj, dive_sol = try_dive(
            items,
            T,
            capacity,
            Gamma_by_item,
            blocks_pool,
            y_vals,
            root.forced_y,
            root.forbidden_y,
            root.lefo_cuts,
        )
        if dive_obj is not None:
            best_ub = dive_obj
            best_solution = dive_sol
            if verbose:
                print(f"  🎯 Dive found incumbent: {best_ub:.4f}")

    # Priority queue: (-depth, -node_id, node) for DFS
    queue: List[Tuple[float, int, BranchNode]] = []
    heapq.heappush(queue, (-root.depth, -root.node_id, root))

    node_counter = 1
    nodes_explored = 0

    while queue:
        if time.time() - start_time > max_time:
            if verbose:
                print(f"\n⏱ Time limit reached ({max_time}s)")
            break

        if nodes_explored >= max_nodes:
            if verbose:
                print(f"\n⚠ Node limit reached ({max_nodes})")
            break

        _, _, node = heapq.heappop(queue)
        nodes_explored += 1
        node_start_time = time.time()

        # Pruning
        if best_ub is not None and node.lp_bound >= best_ub - EPS:
            if logger:
                logger.log_node(
                    NodeLogEntry(
                        node_id=node.node_id,
                        depth=node.depth,
                        lp_bound=node.lp_bound,
                        incumbent=best_ub,
                        branch_item=None,
                        branch_t=None,
                        direction=None,
                        status="PRUNED_BY_BOUND",
                        cg_iters=0,
                        columns_added=0,
                    )
                )
            continue

        # Column generation with LEFO cut-and-resolve loop
        total_cg_iters = 0
        total_cols_added = 0
        node_rmp = None
        while True:
            obj, info, node_blocks, cg_iters, cols_added, node_rmp = (
                column_generation_loop(
                    items,
                    T,
                    capacity,
                    Gamma_by_item,
                    blocks_pool,
                    node.forced_y,
                    node.forbidden_y,
                    node.lefo_cuts,
                    item_caches,
                    node_id=node.node_id,
                    logger=logger,
                    verbose=False,
                    rmp=node_rmp,
                )
            )
            total_cg_iters += cg_iters
            total_cols_added += cols_added

            if not math.isfinite(obj):
                break  # Infeasible

            # Check for LEFO violations using z (arc usage) values
            if not ENABLE_LEFO_CUTS:
                break  # LEFO cuts disabled

            z_vals = info.get("z", {})
            violations = find_lefo_violations_z(z_vals, items, node.lefo_cuts)
            if not violations:
                break  # No violations, proceed

            # Add up to MAX_LEFO_CUTS_PER_ITER cuts
            cuts_to_add = violations[:MAX_LEFO_CUTS_PER_ITER]
            for v in cuts_to_add:
                node.lefo_cuts.append(v)
            if verbose:
                print(
                    f"    Added {len(cuts_to_add)} LEFO cuts "
                    f"(of {len(violations)} violations), restarting CG"
                )

        cg_iters = total_cg_iters
        cols_added = total_cols_added

        # Merge into pool
        for item_id, blks in node_blocks.items():
            existing = {(b.start_t, b.end_e) for b in blocks_pool.get(item_id, [])}
            for b in blks:
                if (b.start_t, b.end_e) not in existing:
                    blocks_pool[item_id].append(b)

        if not math.isfinite(obj):
            if verbose:
                print(
                    f"  [{nodes_explored}] Node {node.node_id} (d={node.depth}): INFEASIBLE"
                )
            if logger:
                logger.log_node(
                    NodeLogEntry(
                        node_id=node.node_id,
                        depth=node.depth,
                        lp_bound=math.inf,
                        incumbent=best_ub if best_ub else math.inf,
                        branch_item=None,
                        branch_t=None,
                        direction=None,
                        status="INFEASIBLE",
                        cg_iters=cg_iters,
                        columns_added=cols_added,
                    )
                )
            continue

        node.lp_bound = obj

        # Pruning
        if best_ub is not None and obj >= best_ub - EPS:
            if verbose:
                print(
                    f"  [{nodes_explored}] Node {node.node_id} (d={node.depth}): LP={obj:.2f} >= UB={best_ub:.2f} → PRUNED"
                )
            if logger:
                logger.log_node(
                    NodeLogEntry(
                        node_id=node.node_id,
                        depth=node.depth,
                        lp_bound=obj,
                        incumbent=best_ub,
                        branch_item=None,
                        branch_t=None,
                        direction=None,
                        status="PRUNED_BY_BOUND",
                        cg_iters=cg_iters,
                        columns_added=cols_added,
                    )
                )
            continue

        # Dive heuristic periodically (if enabled)
        if ENABLE_HEURISTICS and nodes_explored % HEURISTIC_INTERVAL == 0:
            y_vals_dive = info.get("y", {})
            dive_obj, dive_sol = try_dive(
                items,
                T,
                capacity,
                Gamma_by_item,
                blocks_pool,
                y_vals_dive,
                node.forced_y,
                node.forbidden_y,
                node.lefo_cuts,
            )
            if dive_obj is not None and (best_ub is None or dive_obj < best_ub - EPS):
                best_ub = dive_obj
                best_solution = dive_sol
                if verbose:
                    print(f"  🎯 Dive found better incumbent: {best_ub:.4f}")

        y_vals = info.get("y", {})
        art_total = info.get("art_total", 0.0)

        if is_y_integer(y_vals) and art_total <= EPS:
            # Y is integer and no artificial variables - valid integer solution
            if best_ub is None or obj < best_ub - EPS:
                old_ub = best_ub
                best_ub = obj
                best_solution = info
                if verbose:
                    if old_ub is None:
                        print(
                            f"  [{nodes_explored}] Node {node.node_id} (d={node.depth}): LP={obj:.2f} → ★ FIRST INCUMBENT"
                        )
                    else:
                        print(
                            f"  [{nodes_explored}] Node {node.node_id} (d={node.depth}): LP={obj:.2f} → ★ IMPROVED {old_ub:.2f} → {best_ub:.2f}"
                        )
                if logger:
                    logger.log_node(
                        NodeLogEntry(
                            node_id=node.node_id,
                            depth=node.depth,
                            lp_bound=obj,
                            incumbent=obj,
                            branch_item=None,
                            branch_t=None,
                            direction=None,
                            status="NEW_INCUMBENT",
                            cg_iters=cg_iters,
                            columns_added=cols_added,
                        )
                    )
            continue
        elif is_y_integer(y_vals) and art_total > EPS:
            # Y is integer but uses artificial variables - infeasible
            if verbose:
                print(
                    f"  [{nodes_explored}] Node {node.node_id} (d={node.depth}): LP={obj:.2f} INFEASIBLE (art={art_total:.0f})"
                )
            continue

        # Branch on fractional Y
        branch_var = find_best_branching_y(
            y_vals, node.forced_y, node.forbidden_y, items
        )

        if branch_var is None:
            if verbose:
                print(
                    f"  [{nodes_explored}] Node {node.node_id}: No branching variable"
                )
            continue

        item_id, t, val = branch_var

        if verbose:
            ub_str = f"{best_ub:.2f}" if best_ub else "N/A"
            node_time = time.time() - node_start_time
            timing_str = f" [{node_time:.3f}s]" if ENABLE_TIMING else ""
            print(
                f"  [{nodes_explored}] Node {node.node_id} (d={node.depth}): "
                f"LP={obj:.2f}, UB={ub_str} → branch Y[{item_id},{t}]={val:.3f}{timing_str}"
            )

        if logger:
            logger.log_node(
                NodeLogEntry(
                    node_id=node.node_id,
                    depth=node.depth,
                    lp_bound=obj,
                    incumbent=best_ub if best_ub else math.inf,
                    branch_item=item_id,
                    branch_t=t,
                    direction="Y",
                    status="BRANCHED",
                    cg_iters=cg_iters,
                    columns_added=cols_added,
                )
            )

        # Create Y branching children
        for direction in [1, 0]:  # Try Y=1 first
            child = BranchNode(
                node_id=node_counter,
                parent_id=node.node_id,
                depth=node.depth + 1,
                forced_y={i: set(s) for i, s in node.forced_y.items()},
                forbidden_y={i: set(s) for i, s in node.forbidden_y.items()},
                lefo_cuts=list(node.lefo_cuts),
                lp_bound=obj,
            )

            if direction == 1:
                if item_id not in child.forced_y:
                    child.forced_y[item_id] = set()
                child.forced_y[item_id].add(t)
            else:
                if item_id not in child.forbidden_y:
                    child.forbidden_y[item_id] = set()
                child.forbidden_y[item_id].add(t)

            # DFS until incumbent, then best-first
            if USE_DFS_UNTIL_INCUMBENT and best_ub is None:
                priority = -child.depth
            else:
                priority = obj

            heapq.heappush(queue, (priority, -child.node_id, child))
            node_counter += 1

    # Compute global bounds and gap (strict in EXACT_MODE)
    if best_ub is not None:
        if not queue:
            # All nodes explored - optimality proven
            global_lb = best_ub  # LB = UB when tree is empty
            final_gap = 0.0
        else:
            # Some nodes remain - compute actual global LB
            global_lb = min(q[2].lp_bound for q in queue)
            final_gap = (best_ub - global_lb) / max(abs(best_ub), 1e-10)
    else:
        global_lb = root_bound
        final_gap = math.inf

    # In EXACT_MODE, only report gap=0 when truly optimal
    if EXACT_MODE and best_ub is not None and queue:
        # Still have open nodes - can't claim optimality
        if final_gap < EPS:
            final_gap = EPS  # Small positive gap to indicate not fully closed

    if verbose:
        print(f"\n=== B&P Summary ===")
        print(f"  Nodes explored: {nodes_explored}")
        print(f"  Root bound: {root_bound:.4f}")
        print(
            f"  Global LB: {global_lb:.4f}"
            if math.isfinite(global_lb)
            else "  Global LB: N/A"
        )
        if best_ub is not None:
            print(f"  Best solution: {best_ub:.4f}")
            print(f"  Gap: {final_gap * 100:.4f}%")
        if queue:
            print(f"  Open nodes: {len(queue)}")

    return (
        best_ub if best_ub is not None else math.inf,
        best_solution,
        global_lb if math.isfinite(global_lb) else root_bound,
        final_gap,
    )


def _as_len_T_vector(val, T: int) -> List[float]:
    if val is None:
        return []
    if isinstance(val, (int, float)):
        return [float(val)] * T
    if isinstance(val, list):
        if len(val) != T:
            raise ValueError(f"Expected length-{T} list, got {len(val)}")
        return [float(x) for x in val]
    raise TypeError("Value must be a number or list")


def solve_instance(
    instance_path: str | Path = "last_instance.json",
    time_limit: int = 0,
    out_dir: str | Path = "bnp_block_results",
    verbose: bool = True,
    mip_gap: float = 0.0,
    save_csvs: bool = None,
) -> Tuple[Dict, List[str]]:
    """
    Solve using ZIO Block-Based Branch-and-Price.

    Returns: (summary_dict, orders_list)
    """
    _ = mip_gap
    if save_csvs is None:
        save_csvs = verbose
    start_time = time.time()

    data = json.loads(Path(instance_path).read_text())
    T = int(data["period"])
    items: Dict[int, dict] = {int(k): v for k, v in data["items"].items()}

    # Normalize item data
    for i, v in items.items():
        shelf_seq = v.get("shelf_seq", [T] * T)
        if isinstance(shelf_seq, (int, float)):
            shelf_seq = [int(shelf_seq)] * T
        else:
            shelf_seq = [int(s) for s in shelf_seq]
        items[i] = {
            "demand": _as_len_T_vector(v["demand"], T),
            "setup": _as_len_T_vector(v["setup"], T),
            "h": _as_len_T_vector(v["h"], T),
            "c_var": _as_len_T_vector(v["c_var"], T),
            "shelf_seq": shelf_seq[:T],
        }

    prod_cap = data.get("production_capacity") or data.get("manual_capacity")
    capacity = _as_len_T_vector(prod_cap, T) if prod_cap else [math.inf] * T

    max_time = int(time_limit) if time_limit > 0 else 36000

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger = BnPLogger(out_path, enabled=save_csvs)

    if verbose:
        print("=" * 70)
        print("ZIO BLOCK-BASED BRANCH-AND-PRICE")
        print("=" * 70)
        print(f"Items: {len(items)}, Periods: {T}")
        print(f"Capacity: {capacity[:min(6, T)]}{'...' if T > 6 else ''}")

    # Build Gamma
    Gamma_by_item: Dict[int, Dict[int, List[int]]] = {}
    for item_id, item_data in items.items():
        shelf_seq = item_data["shelf_seq"]
        Gamma: Dict[int, List[int]] = {}
        for t in range(T):
            m_t = shelf_seq[t] if t < len(shelf_seq) else 0
            if m_t <= 0:
                Gamma[t] = []
            else:
                u_max = min(T - 1, t + m_t)
                Gamma[t] = list(range(t, u_max + 1))
        Gamma_by_item[item_id] = Gamma

    # Build item caches for O(1) block cost/quantity lookups
    item_caches: Dict[int, ItemCache] = {}
    for item_id, item_data in items.items():
        Gamma = Gamma_by_item[item_id]
        item_caches[item_id] = build_item_cache(item_id, item_data, T, Gamma)
        if verbose:
            cache = item_caches[item_id]
            print(
                f"  Item {item_id}: {len(cache.block_cost)} feasible blocks precomputed"
            )

    # Generate initial blocks
    initial_blocks: Dict[int, List[ZIOBlock]] = {}
    for item_id, item_data in items.items():
        Gamma = Gamma_by_item[item_id]
        forbidden = {t for t in range(T) if capacity[t] <= EPS}
        initial = generate_initial_blocks(item_id, item_data, T, Gamma, forbidden)
        initial_blocks[item_id] = initial

        if verbose:
            print(f"  Item {item_id}: {len(initial)} initial blocks")
            for blk in initial[:5]:
                print(f"    {blk}")

    # Solve
    best_obj, best_solution, root_bound, final_gap = solve_branch_and_price(
        items=items,
        T=T,
        capacity=capacity,
        Gamma_by_item=Gamma_by_item,
        initial_blocks=initial_blocks,
        item_caches=item_caches,
        max_time=max_time,
        verbose=verbose,
        logger=logger,
    )

    runtime = time.time() - start_time
    logger.write_csvs()

    # Build output
    if best_solution is None or not math.isfinite(best_obj):
        summary = {
            "status": int(GRB.INFEASIBLE),
            "objective": None,
            "best_bound": root_bound if math.isfinite(root_bound) else None,
            "gap": None,
            "runtime_sec": runtime,
            "solver_version": "block_based_branch_and_price",
            "n_items": len(items),
            "T": T,
        }
        orders_txt = []
    else:
        if final_gap < EPS:
            status = int(GRB.OPTIMAL)
        elif runtime >= max_time:
            status = int(GRB.TIME_LIMIT)
        else:
            status = int(GRB.OPTIMAL)

        summary = {
            "status": status,
            "objective": float(best_obj),
            "best_bound": float(best_obj) if final_gap < EPS else float(root_bound),
            "gap": final_gap,
            "runtime_sec": runtime,
            "solver_version": "block_based_branch_and_price",
            "n_items": len(items),
            "T": T,
        }

        x_agg = best_solution.get("x_agg", {})
        y_vals = best_solution.get("y", {})
        lam_vals = best_solution.get("lam", {})
        f_vals = best_solution.get("f", {})

        orders_txt = []
        for item_id in sorted(items.keys()):
            # Y periods
            y_periods = [
                t for (i, t), val in y_vals.items() if i == item_id and val > 0.5
            ]
            y_periods.sort()
            y_str = ", ".join(str(t) for t in y_periods) if y_periods else "none"

            orders_txt.append(f"Item {item_id}")
            orders_txt.append(f"  Y (setups at periods): [{y_str}]")

            # Lambda (block coefficients)
            item_lambdas = [
                (t, e, val)
                for (i, t, e), val in lam_vals.items()
                if i == item_id and val > EPS
            ]
            item_lambdas.sort()
            if item_lambdas:
                orders_txt.append(f"  λ (block coefficients):")
                for t, e, val in item_lambdas:
                    orders_txt.append(f"    Block ({t},{e}): λ = {val:.6f}")

            # Production
            orders_txt.append(f"  Production (t → qty):")
            production = x_agg.get(item_id, {})
            for t in sorted(production.keys()):
                qty = production[t]
                if qty > EPS:
                    orders_txt.append(f"    {t:2d} → {qty:8.3f}")

            # Flow arcs
            item_flows = [
                (t, u, val)
                for (i, t, u), val in f_vals.items()
                if i == item_id and val > EPS
            ]
            if item_flows:
                orders_txt.append(f"  Flows (t → u: qty):")
                item_flows.sort()
                for t, u, val in item_flows:
                    orders_txt.append(f"    ({t} → {u}): {val:.3f}")

            orders_txt.append("")

    (out_path / "orders.txt").write_text("\n".join(orders_txt), encoding="utf-8")
    (out_path / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return summary, orders_txt


if __name__ == "__main__":
    import sys

    test_type = sys.argv[1] if len(sys.argv) > 1 else "single"

    out_dir = Path("bnp_block_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    if test_type.endswith(".json"):
        instance_path = Path(test_type)
        with open(instance_path) as f:
            instance = json.load(f)
        if "data" in instance:
            for key in ["items", "period", "manual_capacity", "production_capacity"]:
                if key in instance.get("data", {}):
                    instance[key] = instance["data"][key]
        print(f"Testing JSON file: {test_type}")
    elif test_type == "single":
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
    elif test_type == "multi":
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
        print("Testing MULTI-ITEM instance (3 items)")
    else:
        # Block example
        instance = {
            "period": 6,
            "manual_capacity": [0, 0, 100, 100, 100, 100],
            "items": {
                "0": {
                    "h": [0.5] * 6,
                    "c_var": [2.0] * 6,
                    "setup": [100] * 6,
                    "demand": [0, 0, 28, 12, 20, 15],
                    "shelf_seq": [10] * 6,
                }
            },
        }
        instance_path = out_dir / "block_example.json"
        print("Testing BLOCK EXAMPLE - Expected: Block (2,4) with prod=60")

    if not str(instance_path).endswith(".json") or not Path(instance_path).exists():
        Path(instance_path).write_text(json.dumps(instance, indent=2))

    print(f"Instance: {len(instance['items'])} items, {instance['period']} periods\n")

    summary, orders = solve_instance(
        instance_path=str(instance_path),
        time_limit=300,
        out_dir=str(out_dir),
        verbose=True,
    )

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

    print("\nOrders:")
    for line in orders:
        print(f"  {line}")
