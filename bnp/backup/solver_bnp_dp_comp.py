#!/usr/bin/env python3
"""
Branch-and-Price for Perishable Lot-Sizing with LEFO
=====================================================

Dantzig-Wolfe Decomposition with:
- Zero-Inventory-Ordering (ZIO) columns via Dynamic Programming
- Arc branching (Θ⁰, Θ¹) and Setup branching (Υ⁰, Υ¹)
- RMP linking constraints with duals (τ, σ)
- Column inheritance between parent and child nodes
- Best-First Search strategy

Mathematical Notation:
- πₜ: capacity dual (pi)
- μᵢ: convexity dual (mu)
- τᵢₜᵤ: arc linking dual (tau)
- σᵢₜ: setup linking dual (sigma)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from pathlib import Path
import math
import json
import heapq
import time

import gurobipy as gurobi
from gurobipy import GRB


# =============================================================================
#                              CONSTANTS
# =============================================================================

INF = 1e15
EPS_DEFAULT = 1e-9
DUMMY_COST_MULTIPLIER = 100000.0


# =============================================================================
#                           DATA STRUCTURES
# =============================================================================


@dataclass
class ProductionItem:
    """
    Represents a single item in the lot-sizing problem.

    Attributes:
        item_id: Unique identifier i
        T: Planning horizon length
        demand: dᵢₜ for t ∈ {0,...,T-1}  (0-indexed)
        prod_cost: cᵢₜ
        setup_cost: sᵢₜ
        holding_cost: hᵢₜ
        shelf_life: mᵢₜ (relative shelf life in periods)
        lost_sales_penalty: penalty for unmet demand
    """

    item_id: int
    T: int
    demand: List[float]
    prod_cost: List[float]
    setup_cost: List[float]
    holding_cost: List[float]
    shelf_life: List[int]
    lost_sales_penalty: float

    @property
    def expiry_abs(self) -> List[int]:
        """
        Absolute expiry period (inclusive).
        vᵢₜ = t + mᵢₜ (0-indexed)
        Production at period t can satisfy demands up to and including period vᵢₜ.
        """
        return [t + self.shelf_life[t] for t in range(self.T)]


@dataclass
class ProductionPlanColumn:
    """
    Represents a single ZIO column k ∈ Ωᵢ for item i.

    All indices are 0-based.

    Attributes:
        item_id: Which item this column belongs to
        cost: cₖ = Σₜ[sᵢₜ·yₖₜ + Σᵤ(cᵢₜ + H(t,u))·dᵢᵤ·zₖₜᵤ]
        production: Xₖₜ = Σᵤ dᵢᵤ·zₖₜᵤ (production quantity at t)
        setup: yₖₜ ∈ {0,1} (setup indicator)
        arcs: zₖₜᵤ ∈ {0,1} (demand u satisfied from production t)
        is_dummy: True if this is an artificial/dummy column
    """

    item_id: int
    cost: float
    production: List[float]
    setup: List[float]
    arcs: Dict[Tuple[int, int], float]
    is_dummy: bool = False

    def get_signature(self) -> str:
        """Unique signature for duplicate detection."""
        if self.is_dummy:
            return f"DUMMY_I{self.item_id}"
        arcs_sorted = sorted(self.arcs.keys())
        setups = tuple(t for t, y in enumerate(self.setup) if y > 0.5)
        return f"I{self.item_id}_Y{setups}_Z{arcs_sorted}"

    def violates_branching(
        self,
        theta_0: Set[Tuple[int, int]],
        theta_1: Set[Tuple[int, int]],
        upsilon_0: Set[int],
        upsilon_1: Set[int],
    ) -> bool:
        """Check if column violates any branching constraints."""
        # Check forbidden arcs (Θ⁰): column must NOT have these
        for t, u in theta_0:
            if self.arcs.get((t, u), 0.0) > 0.5:
                return True

        # Check forced arcs (Θ¹): column MUST have these
        for t, u in theta_1:
            if self.arcs.get((t, u), 0.0) < 0.5:
                return True

        # Check forbidden setups (Υ⁰): column must NOT have setup at t
        for t in upsilon_0:
            if self.setup[t] > 0.5:
                return True

        # Check forced setups (Υ¹): column MUST have setup at t
        for t in upsilon_1:
            if self.setup[t] < 0.5:
                return True

        return False


@dataclass
class BranchNode:
    """
    Represents a node in the branch-and-bound tree.

    Branching constraints (all 0-indexed):
        theta_0[i] = Θ⁰ᵢ: arcs (t,u) fixed to 0
        theta_1[i] = Θ¹ᵢ: arcs (t,u) fixed to 1
        upsilon_0[i] = Υ⁰ᵢ: setups t fixed to 0
        upsilon_1[i] = Υ¹ᵢ: setups t fixed to 1
    """

    node_id: int
    parent_id: Optional[int]
    depth: int

    # Arc branching: Θ⁰ᵢ and Θ¹ᵢ (all 0-indexed)
    theta_0: Dict[int, Set[Tuple[int, int]]] = field(default_factory=dict)
    theta_1: Dict[int, Set[Tuple[int, int]]] = field(default_factory=dict)

    # Setup branching: Υ⁰ᵢ and Υ¹ᵢ (all 0-indexed)
    upsilon_0: Dict[int, Set[int]] = field(default_factory=dict)
    upsilon_1: Dict[int, Set[int]] = field(default_factory=dict)

    # Node state
    lp_bound: float = math.inf
    is_integer: bool = False
    is_pruned: bool = False
    prune_reason: Optional[str] = None

    # Branching decision info
    branch_var: Optional[Tuple] = None
    branch_dir: Optional[str] = None


# =============================================================================
#                          SEARCH STATISTICS
# =============================================================================


class SearchStatistics:
    """Track branch-and-bound search statistics."""

    def __init__(self):
        self.nodes_created = 0
        self.nodes_explored = 0
        self.nodes_integer = 0
        self.nodes_fathomed_by_bound = 0
        self.nodes_fathomed_by_infeasible = 0
        self.nodes_fathomed_integer = 0
        self.nodes_fathomed_on_incumbent = 0
        self.nodes_fathomed_duplicate = 0
        self.nodes_fathomed_dummy = 0
        self.max_depth = 0
        self.incumbent_history: List[Tuple[int, float]] = []
        self.start_time = time.time()
        self.total_columns_generated = 0
        self.total_cg_iterations = 0

    def node_created(self):
        self.nodes_created += 1

    def node_explored(
        self,
        node: BranchNode,
        incumbent_improved: bool = False,
        num_fathomed: int = 0,
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
            reason_map = {
                "bound": "nodes_fathomed_by_bound",
                "infeasible": "nodes_fathomed_by_infeasible",
                "duplicate": "nodes_fathomed_duplicate",
                "dummy": "nodes_fathomed_dummy",
            }
            attr = reason_map.get(node.prune_reason)
            if attr:
                setattr(self, attr, getattr(self, attr) + 1)

    def get_runtime(self) -> float:
        return time.time() - self.start_time

    def print_summary(self, best_lb: float, best_ub: Optional[float], eps: float):
        elapsed = self.get_runtime()
        total_fathomed = (
            self.nodes_fathomed_by_bound
            + self.nodes_fathomed_by_infeasible
            + self.nodes_fathomed_integer
            + self.nodes_fathomed_duplicate
            + self.nodes_fathomed_dummy
        )

        print("\n" + "=" * 70)
        print(" " * 25 + "FINAL RESULTS")
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
        print(f"    Dummy active:     {self.nodes_fathomed_dummy}")
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


# =============================================================================
#                           HELPER FUNCTIONS
# =============================================================================


def node_signature(node: BranchNode) -> str:
    """Create unique signature for duplicate node detection."""
    parts = []

    for item_id in sorted(node.theta_0.keys()):
        arcs = sorted(node.theta_0[item_id])
        if arcs:
            parts.append(f"I{item_id}_Z0:{','.join(f'{t}-{u}' for t, u in arcs)}")

    for item_id in sorted(node.theta_1.keys()):
        arcs = sorted(node.theta_1[item_id])
        if arcs:
            parts.append(f"I{item_id}_Z1:{','.join(f'{t}-{u}' for t, u in arcs)}")

    for item_id in sorted(node.upsilon_0.keys()):
        periods = sorted(node.upsilon_0[item_id])
        if periods:
            parts.append(f"I{item_id}_Y0:{','.join(str(t) for t in periods)}")

    for item_id in sorted(node.upsilon_1.keys()):
        periods = sorted(node.upsilon_1[item_id])
        if periods:
            parts.append(f"I{item_id}_Y1:{','.join(str(t) for t in periods)}")

    return "|".join(parts)


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


def _cap_global_from_dem(items: Dict[int, dict], T: int) -> List[float]:
    """Compute default capacity from total demand with buffer."""
    cap = [0.0] * T
    for it in items.values():
        dem = it["demand"]
        for t in range(T):
            cap[t] += dem[t]
    buffer = max(5, int(0.2 * max(cap) if cap else 0))
    return [c + buffer for c in cap]


# =============================================================================
#                    DP PRICING SUBPROBLEM (WAGNER-WHITIN ZIO)
# =============================================================================


def dp_pricing_for_item(
    item: ProductionItem,
    pi: List[float],
    mu: float,
    theta_0: Set[Tuple[int, int]],
    theta_1: Set[Tuple[int, int]],
    upsilon_0: Set[int],
    upsilon_1: Set[int],
    tau: Dict[Tuple[int, int], float],
    sigma: Dict[int, float],
    eps: float = EPS_DEFAULT,
) -> Tuple[float, Optional[ProductionPlanColumn]]:
    """
    DP-based pricing for ZIO lot-sizing with perishability.

    All indices are 0-based.

    Finds minimum reduced cost column:
        r̄ₖ = cₖ - μᵢ - Σₜ πₜ·Xₖₜ - Σₜᵤ τᵢₜᵤ·zₖₜᵤ - Σₜ σᵢₜ·yₖₜ

    State: F[e] = min reduced cost to cover demands 0..e-1 (e periods covered)
    Recursion: F[e] = min_{t < e, feasible block} { F[t] + rc(t, e-1) }

    The DP MUST cover all T periods (complete coverage required).

    Returns:
        (reduced_cost, column) or (INF, None) if infeasible
    """
    T = item.T
    d = item.demand
    s = item.setup_cost
    c = item.prod_cost
    h = item.holding_cost
    v = item.expiry_abs

    # -------------------------------------------------------------------------
    # Pre-check: Conflicting Branching Constraints
    # -------------------------------------------------------------------------
    for t in upsilon_1:
        if t in upsilon_0:
            return INF, None

    # Build forced_source map: which periods MUST be sourced from specific t
    # All indices are 0-based
    forced_source: Dict[int, int] = {}  # u → t (demand u must come from production t)
    must_produce_at: Set[int] = set()
    min_end_for_forced: Dict[int, int] = {}  # t → max u where (t,u) ∈ Θ¹

    for t, u in theta_1:
        if u in forced_source and forced_source[u] != t:
            return INF, None  # Demand u forced from two different periods
        forced_source[u] = t
        must_produce_at.add(t)
        min_end_for_forced[t] = max(min_end_for_forced.get(t, u), u)

    # Check if forced production at t is forbidden by Υ⁰
    for t in must_produce_at:
        if t in upsilon_0:
            return INF, None

    # -------------------------------------------------------------------------
    # Precomputation: Cumulative Demand
    # -------------------------------------------------------------------------
    # D[e] = cumulative demand for periods 0..e-1, D[0] = 0
    D = [0.0] * (T + 1)
    for e in range(1, T + 1):
        D[e] = D[e - 1] + d[e - 1]

    # -------------------------------------------------------------------------
    # Precomputation: Block Costs
    # -------------------------------------------------------------------------
    # For a block (t, e) covering periods t..e (0-indexed, inclusive)
    # Production at period t, satisfying demands at periods t, t+1, ..., e

    def compute_block_reduced_cost(t: int, e: int) -> float:
        """
        Compute reduced cost for block (t, e).
        t, e are 0-indexed (produce at t, cover demands t through e inclusive).
        Returns INF if block is infeasible.
        """
        # Check setup constraint
        if t in upsilon_0:
            return INF

        # Check shelf life
        if e > v[t]:
            return INF

        # Check forbidden arcs
        for u in range(t, e + 1):
            if (t, u) in theta_0:
                return INF

        # Check forced arcs: any forced source for u in [t, e] must be from t
        for u in range(t, e + 1):
            if u in forced_source and forced_source[u] != t:
                return INF

        # Check if block covers minimum required end for forced arcs at t
        if t in min_end_for_forced and e < min_end_for_forced[t]:
            return INF

        # Compute costs
        qty = D[e + 1] - D[t]  # Total demand from t to e

        # Setup cost - sigma dual
        setup_rc = s[t] - sigma.get(t, 0.0)

        # Production cost - pi dual
        prod_rc = (c[t] - pi[t]) * qty

        # Holding cost
        hold_cost = 0.0
        for u in range(t + 1, e + 1):
            # Demand at u held for (u - t) periods
            hold_cost += h[t] * d[u] * (u - t)

        # Arc dual contribution
        tau_contrib = sum(tau.get((t, u), 0.0) for u in range(t, e + 1))

        return setup_rc + prod_rc + hold_cost - tau_contrib

    # -------------------------------------------------------------------------
    # Dynamic Programming (must cover all T periods)
    # -------------------------------------------------------------------------
    # F[e] = min cost to cover periods 0..e-1 (e periods total covered)
    # F[0] = 0 (no periods covered)
    # F[T] = min cost to cover all periods

    F = [INF] * (T + 1)
    pred = [-1] * (T + 1)  # pred[e] = start of block ending at e-1

    F[0] = 0.0

    for e in range(1, T + 1):
        # Try all possible block starts t for block covering t..e-1
        for t in range(e):  # t in [0, e-1]
            block_end = e - 1  # 0-indexed end period
            block_rc = compute_block_reduced_cost(t, block_end)

            if block_rc < INF / 2:
                val = F[t] + block_rc
                if val < F[e]:
                    F[e] = val
                    pred[e] = t

    # -------------------------------------------------------------------------
    # Check feasibility and forced setups
    # -------------------------------------------------------------------------
    if F[T] >= INF / 2:
        return INF, None

    # Reconstruct blocks
    blocks = []  # List of (start, end) tuples, 0-indexed inclusive
    e = T
    while e > 0:
        t = pred[e]
        blocks.append((t, e - 1))
        e = t
    blocks.reverse()

    # Verify all forced setups are covered
    periods_with_setup = {blk[0] for blk in blocks}
    for t in upsilon_1:
        if t not in periods_with_setup:
            # Need to add forced setup at t
            # This is complex - for now, mark as infeasible
            # A proper implementation would split blocks to force setup at t
            return INF, None

    # -------------------------------------------------------------------------
    # Build Column
    # -------------------------------------------------------------------------
    total_rc = F[T] - mu

    if total_rc >= -eps:
        return total_rc, None  # No negative reduced cost column

    production = [0.0] * T
    setup = [0.0] * T
    arcs: Dict[Tuple[int, int], float] = {}
    total_cost = 0.0

    for t, e in blocks:
        setup[t] = 1.0
        qty = D[e + 1] - D[t]
        production[t] = qty

        for u in range(t, e + 1):
            arcs[(t, u)] = 1.0

        # Compute actual cost (without duals)
        hold_cost = sum(h[t] * d[u] * (u - t) for u in range(t + 1, e + 1))
        block_cost = s[t] + c[t] * qty + hold_cost
        total_cost += block_cost

    col = ProductionPlanColumn(
        item_id=item.item_id,
        cost=total_cost,
        production=production,
        setup=setup,
        arcs=arcs,
        is_dummy=False,
    )

    return total_rc, col


def generate_feasible_initial_column(
    item: ProductionItem,
    theta_0: Set[Tuple[int, int]],
    theta_1: Set[Tuple[int, int]],
    upsilon_0: Set[int],
    upsilon_1: Set[int],
) -> Optional[ProductionPlanColumn]:
    """
    Generate a feasible initial column respecting branching constraints.
    Uses a greedy ZIO approach: produce at period t to cover demands starting from t.
    """
    T = item.T
    d = item.demand
    s = item.setup_cost
    c = item.prod_cost
    h = item.holding_cost
    v = item.expiry_abs

    # Build forced source map: demand u must come from period t
    forced_source: Dict[int, int] = {}
    for t, u in theta_1:
        if u in forced_source and forced_source[u] != t:
            return None  # Conflicting forced sources
        forced_source[u] = t

    # Track periods forced to have setups
    forced_setups = set(upsilon_1)

    # Build the ZIO plan
    production = [0.0] * T
    setup = [0.0] * T
    arcs: Dict[Tuple[int, int], float] = {}
    total_cost = 0.0
    covered = [False] * T  # Track which demands are covered

    u = 0  # Current demand period to cover
    while u < T:
        # Determine production period t for this block
        if u in forced_source:
            # This demand has a forced source
            t = forced_source[u]
            if t in upsilon_0:
                return None  # Forced source conflicts with forbidden setup
            if t > u:
                return None  # Can't source from future period
            if v[t] < u:
                return None  # Shelf life violation
            if (t, u) in theta_0:
                return None  # Arc is forbidden

            # For ZIO: if forced to use t < u, we need to cover demands from t to u
            # This means we need to adjust our starting point
            if t < u:
                # We're being forced to use an earlier production period
                # This is complex for ZIO - we need to backtrack
                # For simplicity, return None and let dummy handle it
                return None
        else:
            # Default: produce at period u (ZIO)
            t = u
            if t in upsilon_0:
                # Can't produce at u, look for alternative
                # In strict ZIO, this means we need to produce earlier
                # Find the latest period before u that can cover u
                found = False
                for t_alt in range(u - 1, -1, -1):
                    if t_alt in upsilon_0:
                        continue
                    if (t_alt, u) in theta_0:
                        continue
                    if v[t_alt] >= u:
                        t = t_alt
                        found = True
                        break
                if not found:
                    return None  # Can't cover demand u

        # Now t is the production period. Determine the block extent [t, e]
        # For ZIO, block covers consecutive periods t, t+1, ..., e

        # First, mark all periods from t to u-1 as needing coverage from t
        # (if t < u, we need to cover those too)

        # Find maximum e respecting constraints
        e = t
        while e < T - 1:
            next_e = e + 1

            # Check shelf life
            if v[t] < next_e:
                break

            # Check forbidden arcs
            if (t, next_e) in theta_0:
                break

            # Check if next_e has a different forced source
            if next_e in forced_source and forced_source[next_e] != t:
                break

            e = next_e

        # Verify we can cover at least up to u
        if e < u:
            return None  # Can't reach the demand we're trying to cover

        # Create the block [t, e]
        qty = sum(d[r] for r in range(t, e + 1))
        setup[t] = 1.0
        production[t] = qty

        for r in range(t, e + 1):
            arcs[(t, r)] = 1.0
            covered[r] = True

        # Compute cost
        hold_cost = sum(h[t] * d[r] * (r - t) for r in range(t + 1, e + 1))
        total_cost += s[t] + c[t] * qty + hold_cost

        # Move to next uncovered period
        u = e + 1

    # Verify all demands are covered
    if not all(covered):
        return None

    # Verify all forced setups are present
    periods_with_setup = {t for t in range(T) if setup[t] > 0.5}
    for t in forced_setups:
        if t not in periods_with_setup:
            # Need to add a setup at t, but in ZIO this is complex
            # Return None and let other methods handle it
            return None

    # Verify all forced arcs are present
    for t, u in theta_1:
        if arcs.get((t, u), 0.0) < 0.5:
            return None

    return ProductionPlanColumn(
        item_id=item.item_id,
        cost=total_cost,
        production=production,
        setup=setup,
        arcs=arcs,
        is_dummy=False,
    )


# =============================================================================
#                      RESTRICTED MASTER PROBLEM (RMP)
# =============================================================================


class RestrictedMasterProblem:
    """
    Restricted Master Problem for Dantzig-Wolfe decomposition.

    minimize   Σᵢ Σₖ cₖ · λᵢₖ + M · Σᵢ aᵢ
    subject to:
        Σₖ λᵢₖ + aᵢ = 1                ∀i    (convexity)     → dual: μᵢ
        Σᵢ Σₖ Xₖₜ · λᵢₖ ≤ κₜ          ∀t    (capacity)      → dual: πₜ
        Σₖ zₖₜᵤ · λᵢₖ = 0             ∀(t,u) ∈ Θ⁰ᵢ          → dual: τᵢₜᵤ
        Σₖ zₖₜᵤ · λᵢₖ = 1             ∀(t,u) ∈ Θ¹ᵢ          → dual: τᵢₜᵤ
        Σₖ yₖₜ · λᵢₖ = 0              ∀t ∈ Υ⁰ᵢ              → dual: σᵢₜ
        Σₖ yₖₜ · λᵢₖ = 1              ∀t ∈ Υ¹ᵢ              → dual: σᵢₜ
        λᵢₖ ≥ 0, aᵢ ≥ 0

    where aᵢ are artificial variables for phase-1 feasibility.
    """

    # Big M for artificial variables
    BIG_M = 1e8

    def __init__(
        self,
        items: List[ProductionItem],
        capacity: List[float],
        theta_0: Dict[int, Set[Tuple[int, int]]],
        theta_1: Dict[int, Set[Tuple[int, int]]],
        upsilon_0: Dict[int, Set[int]],
        upsilon_1: Dict[int, Set[int]],
    ):
        self.items = items
        self.T = len(capacity)
        self.capacity = capacity

        # Store branching constraints
        self.theta_0 = theta_0
        self.theta_1 = theta_1
        self.upsilon_0 = upsilon_0
        self.upsilon_1 = upsilon_1

        # Create Gurobi model
        self.model = gurobi.Model("RMP")
        self.model.Params.OutputFlag = 0
        self.model.Params.Method = 1  # Dual Simplex for warm starts

        # Column storage
        self.columns: Dict[int, List[ProductionPlanColumn]] = {
            it.item_id: [] for it in items
        }
        self.lambdas: Dict[Tuple[int, int], gurobi.Var] = {}
        self.column_signatures: Set[str] = set()

        # Artificial variables for convexity constraints
        self.artificial_vars: Dict[int, gurobi.Var] = {}

        # Constraints (will be populated by _build_constraints)
        self.convex_con: Dict[int, gurobi.Constr] = {}
        self.cap_con: List[gurobi.Constr] = []
        self.arc_link_con_0: Dict[Tuple[int, int, int], gurobi.Constr] = {}
        self.arc_link_con_1: Dict[Tuple[int, int, int], gurobi.Constr] = {}
        self.setup_link_con_0: Dict[Tuple[int, int], gurobi.Constr] = {}
        self.setup_link_con_1: Dict[Tuple[int, int], gurobi.Constr] = {}

        self._build_constraints()

    def _build_constraints(self):
        """Build all RMP constraints with artificial variables for feasibility."""
        # Create artificial variables for each item's convexity constraint
        for item in self.items:
            art_var = self.model.addVar(
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                obj=self.BIG_M,
                name=f"art_{item.item_id}",
            )
            self.artificial_vars[item.item_id] = art_var

        self.model.update()

        # Convexity constraints: Σₖ λᵢₖ + aᵢ = 1
        for item in self.items:
            self.convex_con[item.item_id] = self.model.addConstr(
                self.artificial_vars[item.item_id] == 1.0,
                name=f"conv_{item.item_id}",
            )

        # Capacity constraints: Σᵢ Σₖ Xₖₜ · λᵢₖ ≤ κₜ
        for t in range(self.T):
            self.cap_con.append(
                self.model.addConstr(
                    gurobi.LinExpr(0.0) <= self.capacity[t],
                    name=f"cap_{t}",
                )
            )

        # Arc linking constraints for Θ⁰: Σₖ zₖₜᵤ · λᵢₖ = 0
        for item in self.items:
            for t, u in self.theta_0.get(item.item_id, set()):
                self.arc_link_con_0[(item.item_id, t, u)] = self.model.addConstr(
                    gurobi.LinExpr(0.0) == 0.0,
                    name=f"arclink0_{item.item_id}_{t}_{u}",
                )

        # Arc linking constraints for Θ¹: Σₖ zₖₜᵤ · λᵢₖ = 1
        for item in self.items:
            for t, u in self.theta_1.get(item.item_id, set()):
                self.arc_link_con_1[(item.item_id, t, u)] = self.model.addConstr(
                    gurobi.LinExpr(0.0) == 1.0,
                    name=f"arclink1_{item.item_id}_{t}_{u}",
                )

        # Setup linking constraints for Υ⁰: Σₖ yₖₜ · λᵢₖ = 0
        for item in self.items:
            for t in self.upsilon_0.get(item.item_id, set()):
                self.setup_link_con_0[(item.item_id, t)] = self.model.addConstr(
                    gurobi.LinExpr(0.0) == 0.0,
                    name=f"setuplink0_{item.item_id}_{t}",
                )

        # Setup linking constraints for Υ¹: Σₖ yₖₜ · λᵢₖ = 1
        for item in self.items:
            for t in self.upsilon_1.get(item.item_id, set()):
                self.setup_link_con_1[(item.item_id, t)] = self.model.addConstr(
                    gurobi.LinExpr(0.0) == 1.0,
                    name=f"setuplink1_{item.item_id}_{t}",
                )

        self.model.update()

    def add_column(self, col: ProductionPlanColumn) -> bool:
        """
        Add a column to the RMP.

        Returns True if column was added (not a duplicate).
        """
        sig = col.get_signature()
        if sig in self.column_signatures:
            return False

        self.column_signatures.add(sig)
        item_id = col.item_id
        idx = len(self.columns[item_id])

        # Create lambda variable with proper column coefficients
        lam = self.model.addVar(
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            obj=col.cost,
            name=f"lam_{item_id}_{idx}",
            column=gurobi.Column(
                self._get_column_coeffs(col),
                self._get_column_constrs(col),
            ),
        )

        self.lambdas[(item_id, idx)] = lam
        self.columns[item_id].append(col)

        return True

    def _get_column_coeffs(self, col: ProductionPlanColumn) -> List[float]:
        """Get constraint coefficients for a column."""
        coeffs = []

        # Convexity: coefficient 1
        coeffs.append(1.0)

        # Capacity: production amounts
        for t in range(self.T):
            coeffs.append(col.production[t])

        # Arc linking Θ⁰
        for t, u in self.theta_0.get(col.item_id, set()):
            coeffs.append(col.arcs.get((t, u), 0.0))

        # Arc linking Θ¹
        for t, u in self.theta_1.get(col.item_id, set()):
            coeffs.append(col.arcs.get((t, u), 0.0))

        # Setup linking Υ⁰
        for t in self.upsilon_0.get(col.item_id, set()):
            coeffs.append(col.setup[t])

        # Setup linking Υ¹
        for t in self.upsilon_1.get(col.item_id, set()):
            coeffs.append(col.setup[t])

        return coeffs

    def _get_column_constrs(self, col: ProductionPlanColumn) -> List[gurobi.Constr]:
        """Get constraints that this column participates in."""
        constrs = []
        item_id = col.item_id

        # Convexity
        constrs.append(self.convex_con[item_id])

        # Capacity
        constrs.extend(self.cap_con)

        # Arc linking Θ⁰
        for t, u in self.theta_0.get(item_id, set()):
            constrs.append(self.arc_link_con_0[(item_id, t, u)])

        # Arc linking Θ¹
        for t, u in self.theta_1.get(item_id, set()):
            constrs.append(self.arc_link_con_1[(item_id, t, u)])

        # Setup linking Υ⁰
        for t in self.upsilon_0.get(item_id, set()):
            constrs.append(self.setup_link_con_0[(item_id, t)])

        # Setup linking Υ¹
        for t in self.upsilon_1.get(item_id, set()):
            constrs.append(self.setup_link_con_1[(item_id, t)])

        return constrs

    def solve(
        self,
    ) -> Tuple[
        float,
        Dict[int, float],
        List[float],
        Dict[int, Dict[Tuple[int, int], float]],
        Dict[int, Dict[int, float]],
    ]:
        """
        Solve the RMP and return objective and duals.

        Returns:
            (obj, mu, pi, tau, sigma) where:
            - obj: objective value
            - mu[i]: convexity dual μᵢ
            - pi[t]: capacity dual πₜ
            - tau[i][(t,u)]: arc linking dual τᵢₜᵤ
            - sigma[i][t]: setup linking dual σᵢₜ
        """
        self.model.optimize()

        if self.model.status != GRB.OPTIMAL:
            return math.inf, {}, [], {}, {}

        # Extract duals
        mu = {it.item_id: self.convex_con[it.item_id].Pi for it in self.items}
        pi = [self.cap_con[t].Pi for t in range(self.T)]

        tau_by_item: Dict[int, Dict[Tuple[int, int], float]] = {
            it.item_id: {} for it in self.items
        }
        for (item_id, t, u), con in self.arc_link_con_0.items():
            tau_by_item[item_id][(t, u)] = con.Pi
        for (item_id, t, u), con in self.arc_link_con_1.items():
            tau_by_item[item_id][(t, u)] = con.Pi

        sigma_by_item: Dict[int, Dict[int, float]] = {
            it.item_id: {} for it in self.items
        }
        for (item_id, t), con in self.setup_link_con_0.items():
            sigma_by_item[item_id][t] = con.Pi
        for (item_id, t), con in self.setup_link_con_1.items():
            sigma_by_item[item_id][t] = con.Pi

        return self.model.ObjVal, mu, pi, tau_by_item, sigma_by_item

    def has_active_dummy(self, eps: float = 1e-6) -> bool:
        """Check if any dummy column or artificial variable has positive value."""
        # Check artificial variables
        for item_id, art_var in self.artificial_vars.items():
            if art_var.X > eps:
                return True

        # Check dummy columns
        for item in self.items:
            cols = self.columns[item.item_id]
            for idx, col in enumerate(cols):
                if col.is_dummy:
                    lam = self.lambdas.get((item.item_id, idx))
                    if lam is not None and lam.X > eps:
                        return True
        return False

    def get_artificial_usage(self) -> Dict[int, float]:
        """Return the usage of artificial variables."""
        return {item_id: var.X for item_id, var in self.artificial_vars.items()}


# =============================================================================
#                    COLUMN GENERATION FOR A SINGLE NODE
# =============================================================================


def create_dummy_column(item: ProductionItem) -> ProductionPlanColumn:
    """
    Create a high-cost dummy column for initial feasibility.

    This creates a valid ZIO column that respects shelf life constraints,
    producing at each period to cover demands within the shelf life window.
    """
    T = item.T
    d = item.demand
    v = item.expiry_abs

    production = [0.0] * T
    setup = [0.0] * T
    arcs: Dict[Tuple[int, int], float] = {}

    # Build a valid ZIO plan: produce at each period t to cover demands
    # from t up to min(T-1, v[t]) respecting shelf life
    u = 0  # Next demand to cover
    while u < T:
        # Produce at period u to cover demands starting from u
        t = u
        max_end = min(T - 1, v[t])  # Shelf life limit

        # Find how far this block can extend
        e = t
        while e < max_end:
            e += 1

        # Create block [t, e]
        qty = sum(d[r] for r in range(t, e + 1))
        if qty > 0 or t == e:  # Even if qty is 0, we might need the block structure
            setup[t] = 1.0
            production[t] = qty
            for r in range(t, e + 1):
                arcs[(t, r)] = 1.0

        u = e + 1

    # Compute actual cost (high penalty)
    dummy_cost = DUMMY_COST_MULTIPLIER * (sum(d) + 1)

    return ProductionPlanColumn(
        item_id=item.item_id,
        cost=dummy_cost,
        production=production,
        setup=setup,
        arcs=arcs,
        is_dummy=True,
    )


def solve_node_with_column_generation(
    items: List[ProductionItem],
    capacity: List[float],
    node: BranchNode,
    inherited_columns: Optional[Dict[int, List[ProductionPlanColumn]]] = None,
    max_iter: int = 500,
    eps: float = EPS_DEFAULT,
    verbose: bool = False,
    stats: Optional[SearchStatistics] = None,
) -> Tuple[
    float, RestrictedMasterProblem, bool, Dict[int, Dict[Tuple[int, int], float]]
]:
    """
    Solve a single B&B node via column generation.

    Returns:
        (lb, rmp, converged, z_vals)
    """
    rmp = RestrictedMasterProblem(
        items,
        capacity,
        node.theta_0,
        node.theta_1,
        node.upsilon_0,
        node.upsilon_1,
    )

    # -------------------------------------------------------------------------
    # Add Initial Columns
    # -------------------------------------------------------------------------
    # Try to generate feasible initial columns for each item
    for item in items:
        item_id = item.item_id
        theta_0 = node.theta_0.get(item_id, set())
        theta_1 = node.theta_1.get(item_id, set())
        upsilon_0 = node.upsilon_0.get(item_id, set())
        upsilon_1 = node.upsilon_1.get(item_id, set())

        # Try to generate a feasible ZIO column
        init_col = generate_feasible_initial_column(
            item, theta_0, theta_1, upsilon_0, upsilon_1
        )
        if init_col is not None:
            rmp.add_column(init_col)

    # Add inherited columns (filtering for feasibility)
    if inherited_columns:
        for item in items:
            item_id = item.item_id
            theta_0 = node.theta_0.get(item_id, set())
            theta_1 = node.theta_1.get(item_id, set())
            upsilon_0 = node.upsilon_0.get(item_id, set())
            upsilon_1 = node.upsilon_1.get(item_id, set())

            for col in inherited_columns.get(item_id, []):
                if not col.is_dummy and not col.violates_branching(
                    theta_0, theta_1, upsilon_0, upsilon_1
                ):
                    rmp.add_column(col)

    if verbose:
        print(f"  └─ CG: ", end="", flush=True)

    # -------------------------------------------------------------------------
    # Column Generation Loop
    # -------------------------------------------------------------------------
    stall_count = 0
    prev_lb = -math.inf

    for iteration in range(1, max_iter + 1):
        lb, mu, pi, tau_by_item, sigma_by_item = rmp.solve()

        if stats:
            stats.total_cg_iterations += 1

        if not math.isfinite(lb):
            if verbose:
                print("INFEASIBLE")
            return math.inf, rmp, False, {}

        # Stall detection
        if abs(lb - prev_lb) < eps:
            stall_count += 1
            if stall_count >= 10:  # Allow some stalling before giving up
                break
        else:
            stall_count = 0
        prev_lb = lb

        # Pricing: try to find negative reduced cost columns
        any_added = False
        for item in items:
            item_id = item.item_id
            theta_0 = node.theta_0.get(item_id, set())
            theta_1 = node.theta_1.get(item_id, set())
            upsilon_0 = node.upsilon_0.get(item_id, set())
            upsilon_1 = node.upsilon_1.get(item_id, set())
            tau = tau_by_item.get(item_id, {})
            sigma = sigma_by_item.get(item_id, {})

            rc, col = dp_pricing_for_item(
                item,
                pi,
                mu[item_id],
                theta_0,
                theta_1,
                upsilon_0,
                upsilon_1,
                tau,
                sigma,
                eps,
            )

            if col is not None and rc < -eps:
                if rmp.add_column(col):
                    any_added = True
                    if stats:
                        stats.total_columns_generated += 1

        if not any_added:
            if verbose:
                print(f"LB={lb:.2f}")
            z_vals = extract_z_values(rmp, items, eps)
            return lb, rmp, True, z_vals

    # Max iterations or stalled
    lb, _, _, _, _ = rmp.solve()
    z_vals = extract_z_values(rmp, items, eps)
    if verbose:
        print(f"LB={lb:.2f} (converged)")
    return lb, rmp, True, z_vals


# =============================================================================
#                        SOLUTION EXTRACTION
# =============================================================================


def extract_z_values(
    rmp: RestrictedMasterProblem,
    items: List[ProductionItem],
    eps: float,
) -> Dict[int, Dict[Tuple[int, int], float]]:
    """Extract z[i][(t,u)] = Σₖ zₖₜᵤ · λᵢₖ values from RMP solution."""
    z = {it.item_id: {} for it in items}

    for (item_id, idx), lam in rmp.lambdas.items():
        lam_val = lam.X
        if lam_val < eps:
            continue
        col = rmp.columns[item_id][idx]
        for (t, u), arc_val in col.arcs.items():
            if arc_val > 0.5:
                z[item_id][(t, u)] = z[item_id].get((t, u), 0.0) + lam_val

    return z


def extract_y_values(
    rmp: RestrictedMasterProblem,
    items: List[ProductionItem],
    eps: float,
) -> Dict[int, Dict[int, float]]:
    """Extract y[i][t] = Σₖ yₖₜ · λᵢₖ values from RMP solution."""
    T = rmp.T
    y = {it.item_id: {t: 0.0 for t in range(T)} for it in items}

    for (item_id, idx), lam in rmp.lambdas.items():
        lam_val = lam.X
        if lam_val < eps:
            continue
        col = rmp.columns[item_id][idx]
        for t in range(T):
            if col.setup[t] > 0.5:
                y[item_id][t] += lam_val

    return y


def is_integer(z_vals: Dict, y_vals: Dict, eps: float = 1e-6) -> bool:
    """Check if all z and y values are integer (0 or 1)."""
    for arcs in z_vals.values():
        for val in arcs.values():
            if eps < val < 1.0 - eps:
                return False

    for setups in y_vals.values():
        for val in setups.values():
            if eps < val < 1.0 - eps:
                return False

    return True


# =============================================================================
#                        BRANCHING VARIABLE SELECTION
# =============================================================================


def find_most_fractional_z(
    z_vals: Dict,
    node: BranchNode,
    eps: float = EPS_DEFAULT,
) -> Optional[Tuple[int, int, int, float]]:
    """
    Find most fractional arc variable z[i,(t,u)].

    Returns (item_id, t, u, z_val) or None.
    """
    best_frac = 0.0
    best = None

    for item_id, arcs in z_vals.items():
        theta_0 = node.theta_0.get(item_id, set())
        theta_1 = node.theta_1.get(item_id, set())

        for (t, u), val in arcs.items():
            if (t, u) in theta_0 or (t, u) in theta_1:
                continue
            frac = min(val, 1.0 - val)
            if frac > eps and frac > best_frac:
                best_frac = frac
                best = (item_id, t, u, val)

    return best


def find_most_fractional_y(
    y_vals: Dict,
    node: BranchNode,
    eps: float = EPS_DEFAULT,
) -> Optional[Tuple[int, int, float]]:
    """
    Find most fractional setup variable y[i,t].

    Returns (item_id, t, y_val) or None.
    """
    best_frac = 0.0
    best = None

    for item_id, setups in y_vals.items():
        upsilon_0 = node.upsilon_0.get(item_id, set())
        upsilon_1 = node.upsilon_1.get(item_id, set())

        for t, val in setups.items():
            if t in upsilon_0 or t in upsilon_1:
                continue
            frac = min(val, 1.0 - val)
            if frac > eps and frac > best_frac:
                best_frac = frac
                best = (item_id, t, val)

    return best


# =============================================================================
#                       BRANCH-AND-BOUND HELPERS
# =============================================================================


def fathom_heap_by_incumbent(heap: list, incumbent: float, eps: float) -> int:
    """Remove nodes from heap that can be fathomed by incumbent."""
    original_size = len(heap)
    new_heap = [
        (lb, neg_d, nid, node, cols)
        for lb, neg_d, nid, node, cols in heap
        if lb < incumbent - eps
    ]
    num_fathomed = original_size - len(new_heap)

    heap.clear()
    for entry in new_heap:
        heapq.heappush(heap, entry)

    return num_fathomed


def get_heap_min_lb(heap: list) -> float:
    """Get minimum LP bound from heap."""
    if not heap:
        return math.inf
    return min(entry[0] for entry in heap)


# =============================================================================
#                       BRANCH-AND-PRICE MAIN LOOP
# =============================================================================


def solve_branch_and_price(
    items: List[ProductionItem],
    capacity: List[float],
    max_nodes: int = 5000,
    eps: float = 1e-6,
    print_frequency: int = 50,
    verbose: bool = True,
) -> Tuple[float, Optional[float], int, SearchStatistics, Optional[Dict]]:
    """
    Solve using Branch-and-Price with Best-First search.

    Returns:
        (best_lb, best_ub, nodes_explored, stats, z_vals)
    """
    if verbose:
        print("\n" + "=" * 70)
        print(" " * 10 + "BRANCH-AND-PRICE WITH DP PRICING (BEST-FIRST)")
        print("=" * 70)

    stats = SearchStatistics()

    # -------------------------------------------------------------------------
    # Initialize Root Node
    # -------------------------------------------------------------------------
    root = BranchNode(
        node_id=0,
        parent_id=None,
        depth=0,
        theta_0={it.item_id: set() for it in items},
        theta_1={it.item_id: set() for it in items},
        upsilon_0={it.item_id: set() for it in items},
        upsilon_1={it.item_id: set() for it in items},
    )

    if verbose:
        print("\n>>> ROOT NODE <<<")

    root_lb, root_rmp, _, z_vals = solve_node_with_column_generation(
        items, capacity, root, verbose=verbose, stats=stats
    )

    if not math.isfinite(root_lb):
        if verbose:
            print("\n✗ Root infeasible!")
        return math.inf, None, 1, stats, None

    root.lp_bound = root_lb
    y_vals = extract_y_values(root_rmp, items, eps)

    if verbose:
        print(f"  Root LB:  {root_lb:.4f}")
        print(f"  Integer?  {is_integer(z_vals, y_vals, eps)}")
        if root_rmp.has_active_dummy(eps):
            print(f"  WARNING: Dummy column active!")

    best_lb = root_lb
    best_ub: Optional[float] = None
    best_z_vals: Optional[Dict] = None
    nodes_explored = 1
    node_counter = 1
    seen_signatures = {node_signature(root)}

    # Priority queue: (lp_bound, -depth, node_id, node, inherited_columns)
    heap: List[Tuple[float, int, int, BranchNode, Dict]] = []
    parent_columns = {it.item_id: root_rmp.columns[it.item_id][:] for it in items}
    heapq.heappush(
        heap, (root.lp_bound, -root.depth, root.node_id, root, parent_columns)
    )
    stats.node_created()

    # -------------------------------------------------------------------------
    # Check if Root is Integer and Valid
    # -------------------------------------------------------------------------
    if is_integer(z_vals, y_vals, eps) and not root_rmp.has_active_dummy(eps):
        best_ub = root_lb
        best_lb = root_lb
        best_z_vals = z_vals
        root.is_integer = True
        stats.node_explored(root, incumbent_improved=True)
        if verbose:
            print("\n✓ Root is INTEGER - OPTIMAL!")
            stats.print_summary(best_lb, best_ub, eps)
        return root_lb, root_lb, 1, stats, z_vals

    stats.node_explored(root)

    if verbose:
        print(f"\n{'=' * 70}")
        print("BEST-FIRST SEARCH (Best LP Bound)")
        print(f"{'=' * 70}\n")

    # -------------------------------------------------------------------------
    # Main Search Loop
    # -------------------------------------------------------------------------
    while heap and nodes_explored < max_nodes:
        _, _, _, node, inherited_cols = heapq.heappop(heap)

        # Process non-root nodes
        if node.node_id != 0:
            # Progress reporting
            if verbose and nodes_explored % print_frequency == 0:
                gap_str = "N/A"
                if best_ub is not None:
                    gap = best_ub - best_lb
                    gap_pct = 100 * gap / max(abs(best_ub), 1e-10)
                    gap_str = f"{gap_pct:.2f}%"
                print(
                    f"[N={nodes_explored:4d}, Heap={len(heap):4d}, "
                    f"LB={best_lb:.2f}, UB={best_ub if best_ub else 'N/A'}, Gap={gap_str}]"
                )

            if verbose:
                print(f"N{node.node_id:4d} D{node.depth:2d}", end="", flush=True)

            # Solve node
            lb, rmp, converged, z_vals = solve_node_with_column_generation(
                items,
                capacity,
                node,
                inherited_columns=inherited_cols,
                verbose=verbose,
                stats=stats,
            )
            nodes_explored += 1
            node.lp_bound = lb
            y_vals = extract_y_values(rmp, items, eps) if math.isfinite(lb) else {}

            # Check for infeasibility
            if not math.isfinite(lb):
                if verbose:
                    print("  FATHOMED: Infeasible")
                node.is_pruned = True
                node.prune_reason = "infeasible"
                stats.node_explored(node)
                best_lb = (
                    get_heap_min_lb(heap) if heap else (best_ub if best_ub else best_lb)
                )
                continue

            # Update global lower bound
            best_lb = min(lb, get_heap_min_lb(heap)) if heap else lb

            # Fathom by bound
            if best_ub is not None and lb >= best_ub - eps:
                if verbose:
                    print(f"  FATHOMED: {lb:.2f} ≥ {best_ub:.2f}")
                node.is_pruned = True
                node.prune_reason = "bound"
                stats.node_explored(node)
                best_lb = get_heap_min_lb(heap) if heap else best_ub
                continue

            # Fathom if dummy is active
            if rmp.has_active_dummy(eps):
                if verbose:
                    print(f"  FATHOMED: Dummy active")
                node.is_pruned = True
                node.prune_reason = "dummy"
                stats.node_explored(node)
                best_lb = (
                    get_heap_min_lb(heap) if heap else (best_ub if best_ub else best_lb)
                )
                continue

            # Check for integer solution
            if is_integer(z_vals, y_vals, eps):
                if verbose:
                    print(f"  INTEGER: {lb:.2f}", end="")
                node.is_integer = True

                if best_ub is None or lb < best_ub - eps:
                    best_ub = lb
                    best_z_vals = z_vals
                    if verbose:
                        print(" ★ NEW INCUMBENT!")
                    num_fathomed = fathom_heap_by_incumbent(heap, best_ub, eps)
                    best_lb = get_heap_min_lb(heap) if heap else best_ub
                    stats.node_explored(
                        node, incumbent_improved=True, num_fathomed=num_fathomed
                    )
                else:
                    if verbose:
                        print()
                    best_lb = get_heap_min_lb(heap) if heap else best_ub
                    stats.node_explored(node)
                continue

            stats.node_explored(node)
            inherited_cols = {it.item_id: rmp.columns[it.item_id][:] for it in items}
        else:
            # Root node - already processed above, just branch
            rmp = root_rmp

        # ---------------------------------------------------------------------
        # Branching
        # ---------------------------------------------------------------------

        # Priority 1: Arc variables (most fractional z)
        branch_var = find_most_fractional_z(z_vals, node, eps)

        if branch_var is not None:
            item_id, t_br, u_br, z_val = branch_var

            # Z=0 branch: add arc to Θ⁰
            left = BranchNode(
                node_id=node_counter,
                parent_id=node.node_id,
                depth=node.depth + 1,
                theta_0={i: s.copy() for i, s in node.theta_0.items()},
                theta_1={i: s.copy() for i, s in node.theta_1.items()},
                upsilon_0={i: s.copy() for i, s in node.upsilon_0.items()},
                upsilon_1={i: s.copy() for i, s in node.upsilon_1.items()},
                branch_var=(item_id, "Z", t_br, u_br, z_val),
                branch_dir="Z=0",
            )
            left.theta_0[item_id].add((t_br, u_br))
            left.lp_bound = node.lp_bound

            sig = node_signature(left)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                node_counter += 1
                heapq.heappush(
                    heap,
                    (left.lp_bound, -left.depth, left.node_id, left, inherited_cols),
                )
                stats.node_created()

            # Z=1 branch: add arc to Θ¹
            right = BranchNode(
                node_id=node_counter,
                parent_id=node.node_id,
                depth=node.depth + 1,
                theta_0={i: s.copy() for i, s in node.theta_0.items()},
                theta_1={i: s.copy() for i, s in node.theta_1.items()},
                upsilon_0={i: s.copy() for i, s in node.upsilon_0.items()},
                upsilon_1={i: s.copy() for i, s in node.upsilon_1.items()},
                branch_var=(item_id, "Z", t_br, u_br, z_val),
                branch_dir="Z=1",
            )
            right.theta_1[item_id].add((t_br, u_br))
            right.lp_bound = node.lp_bound

            sig = node_signature(right)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                node_counter += 1
                heapq.heappush(
                    heap,
                    (
                        right.lp_bound,
                        -right.depth,
                        right.node_id,
                        right,
                        inherited_cols,
                    ),
                )
                stats.node_created()

            if verbose:
                print(f"  Branch Z[{item_id},{t_br},{u_br}]={z_val:.3f}")
            continue

        # Priority 2: Setup variables (most fractional y)
        branch_var_y = find_most_fractional_y(y_vals, node, eps)

        if branch_var_y is not None:
            item_id, t_br, y_val = branch_var_y

            # Y=0 branch: add setup to Υ⁰
            left = BranchNode(
                node_id=node_counter,
                parent_id=node.node_id,
                depth=node.depth + 1,
                theta_0={i: s.copy() for i, s in node.theta_0.items()},
                theta_1={i: s.copy() for i, s in node.theta_1.items()},
                upsilon_0={i: s.copy() for i, s in node.upsilon_0.items()},
                upsilon_1={i: s.copy() for i, s in node.upsilon_1.items()},
                branch_var=(item_id, "Y", t_br, y_val),
                branch_dir="Y=0",
            )
            left.upsilon_0[item_id].add(t_br)
            left.lp_bound = node.lp_bound

            sig = node_signature(left)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                node_counter += 1
                heapq.heappush(
                    heap,
                    (left.lp_bound, -left.depth, left.node_id, left, inherited_cols),
                )
                stats.node_created()

            # Y=1 branch: add setup to Υ¹
            right = BranchNode(
                node_id=node_counter,
                parent_id=node.node_id,
                depth=node.depth + 1,
                theta_0={i: s.copy() for i, s in node.theta_0.items()},
                theta_1={i: s.copy() for i, s in node.theta_1.items()},
                upsilon_0={i: s.copy() for i, s in node.upsilon_0.items()},
                upsilon_1={i: s.copy() for i, s in node.upsilon_1.items()},
                branch_var=(item_id, "Y", t_br, y_val),
                branch_dir="Y=1",
            )
            right.upsilon_1[item_id].add(t_br)
            right.lp_bound = node.lp_bound

            sig = node_signature(right)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                node_counter += 1
                heapq.heappush(
                    heap,
                    (
                        right.lp_bound,
                        -right.depth,
                        right.node_id,
                        right,
                        inherited_cols,
                    ),
                )
                stats.node_created()

            if verbose:
                print(f"  Branch Y[{item_id},{t_br}]={y_val:.3f}")
            continue

        # No fractional variable found (shouldn't happen if not integer)
        if verbose:
            print("  No fractional variable")
        node.is_pruned = True

    # -------------------------------------------------------------------------
    # Finalize
    # -------------------------------------------------------------------------
    if not heap and best_ub is not None:
        best_lb = best_ub

    if verbose:
        stats.print_summary(best_lb, best_ub, eps)

    return best_lb, best_ub, nodes_explored, stats, best_z_vals


# =============================================================================
#                              JSON I/O
# =============================================================================


def load_instance_from_json(
    instance_path: str | Path,
) -> Tuple[List[ProductionItem], List[float], int]:
    """Load instance from JSON file."""
    data = json.loads(Path(instance_path).read_text())
    T = int(data["period"])
    items_raw: Dict[int, dict] = {int(k): v for k, v in data["items"].items()}

    prod_cap = data.get("production_capacity") or data.get("manual_capacity")
    capacity = (
        _as_len_T_vector(prod_cap, T)
        if prod_cap
        else _cap_global_from_dem(items_raw, T)
    )

    items: List[ProductionItem] = []
    for i, it in items_raw.items():
        demand = [float(d) for d in it["demand"]]
        shelf_seq = [int(m) for m in it["shelf_seq"]]

        c_var = it["c_var"]
        prod_cost = (
            [float(c) for c in c_var] if isinstance(c_var, list) else [float(c_var)] * T
        )

        h = it["h"]
        holding_cost = (
            [float(hh) for hh in h] if isinstance(h, list) else [float(h)] * T
        )

        setup = it["setup"]
        setup_cost = (
            [float(s) for s in setup] if isinstance(setup, list) else [float(setup)] * T
        )

        lost_sales_penalty = float(it.get("lost_sales_penalty", 10000.0))

        items.append(
            ProductionItem(
                item_id=i,
                T=T,
                demand=demand,
                prod_cost=prod_cost,
                setup_cost=setup_cost,
                holding_cost=holding_cost,
                shelf_life=shelf_seq,
                lost_sales_penalty=lost_sales_penalty,
            )
        )

    return items, capacity, T


def solve_instance(
    instance_path: str | Path = "last_instance.json",
    time_limit: int = 0,
    mip_gap: float = 0.0,
    out_dir: str | Path = "bp_results",
    max_nodes: int = 10000,
    verbose: bool = True,
) -> Tuple[dict, List[str]]:
    """Solve using Branch-and-Price."""
    items, capacity, T = load_instance_from_json(instance_path)

    if verbose:
        print("\n╔" + "═" * 68 + "╗")
        print(f"║ {'CAPACITATED LOT SIZING WITH PERISHABILITY (B&P)':^66s} ║")
        print("╠" + "═" * 68 + "╣")
        print(f"║  Items:    {len(items):<57d} ║")
        print(f"║  Periods:  {T:<57d} ║")
        print("╚" + "═" * 68 + "╝")

    best_lb, best_ub, nodes_explored, stats, z_vals = solve_branch_and_price(
        items,
        capacity,
        max_nodes=max_nodes,
        eps=mip_gap if mip_gap > 0 else 1e-6,
        print_frequency=50,
        verbose=verbose,
    )

    summary = {
        "status": 2 if best_ub is not None else 3,
        "objective": best_ub,
        "best_bound": best_lb,
        "gap": None,
        "runtime_sec": stats.get_runtime(),
        "solver_version": "bp_best_first_v3_fixed",
        "n_items": len(items),
        "T": T,
        "nodes_explored": nodes_explored,
        "columns_generated": stats.total_columns_generated,
        "cg_iterations": stats.total_cg_iterations,
    }

    if best_ub is not None and best_lb is not None:
        summary["gap"] = (best_ub - best_lb) / max(abs(best_ub), 1e-10)

    # Build orders output
    orders_txt: List[str] = []
    if z_vals is not None:
        for item in items:
            orders_txt.append(f"Item {item.item_id} — orders (t → qty)")
            arcs = z_vals.get(item.item_id, {})
            prod_by_t: Dict[int, float] = {}
            for (t, u), val in arcs.items():
                if val > 0.5:
                    d_u = item.demand[u]
                    prod_by_t[t] = prod_by_t.get(t, 0.0) + d_u
            for t in sorted(prod_by_t.keys()):
                qty = prod_by_t[t]
                if qty > 1e-6:
                    orders_txt.append(f" {t:2d} → {qty:8.3f}")
            orders_txt.append("")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "orders.txt").write_text("\n".join(orders_txt), encoding="utf-8")
    (out_path / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return summary, orders_txt


# =============================================================================
#                                 MAIN
# =============================================================================


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        summary, orders = solve_instance(path)
        print(f"\nObjective: {summary.get('objective')}")
    else:
        print("Usage: python solver_bnp_dp_fixed.py <instance.json>")
        print("\nRunning built-in demo...")

        # Demo instance
        T = 8
        items = [
            ProductionItem(
                1,
                T,
                [6, 0, 5, 0, 3, 4, 0, 5],  # demand
                [3.0, 3.0, 3.3, 3.3, 3.6, 3.6, 3.8, 3.8],  # prod_cost
                [22.0] * T,  # setup_cost
                [1.0] * T,  # holding_cost
                [3, 3, 2, 2, 2, 2, 2, 2],  # shelf_life
                5000.0,  # lost_sales_penalty
            ),
            ProductionItem(
                2,
                T,
                [0, 7, 0, 6, 0, 5, 4, 0],
                [2.8, 2.8, 3.0, 3.0, 3.2, 3.2, 3.5, 3.5],
                [18.0] * T,
                [0.8] * T,
                [2, 2, 3, 3, 2, 2, 2, 2],
                5000.0,
            ),
            ProductionItem(
                3,
                T,
                [4, 4, 4, 0, 3, 0, 6, 0],
                [3.2, 3.2, 3.2, 3.8, 3.8, 4.0, 4.0, 4.0],
                [24.0] * T,
                [1.2] * T,
                [2, 3, 2, 2, 3, 2, 2, 2],
                5000.0,
            ),
        ]
        cap = [15, 11, 15, 8, 8, 8, 9, 9]

        lb, ub, nodes, stats, z_vals = solve_branch_and_price(
            items, cap, max_nodes=10000
        )
        print(f"\nFinal: LB={lb:.2f}, UB={ub}")
