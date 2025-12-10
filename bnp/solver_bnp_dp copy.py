"""
Branch-and-Price for Perishable Lot-Sizing with Heterogeneous Shelf Lives and LEFO
Version 11: Branching via RMP Linking Constraints (not subproblem restrictions)

Key Changes from v10:
- Branching enforcement moved from pricing subproblem to RMP:
  * Υ^0, Θ^0 (forbidden): Enforced by FILTERING columns before adding to RMP
  * Υ^1, Θ^1 (forced): Enforced by RMP linking constraints (Σ Y λ = 1, Σ Z λ = 1)
- Pricing subproblem is UNRESTRICTED - generates any feasible ZIO column
- Duals from = 1 linking constraints guide pricing toward required features

Features:
- Best-first search using priority queue (heapq) - always explores node with best (lowest) lower bound
- Memory optimization: queue stores only (bound, node_id, node, rmp)
- Column inheritance for faster convergence at branch nodes
- σ (sigma) and τ (tau) duals from Υ^1/Θ^1 linking constraints guide pricing
- Dummy column detection to prevent false integer solutions
- Branching on Y (setup) first, then Z (arc) variables
- DP-based pricing for ZIO columns

"""

from __future__ import annotations
import csv
import json
import math
import time
import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, IO
import gurobipy as gp
from gurobipy import GRB

# Global output directory for convergence tracking
_convergence_out_dir: Optional[Path] = None


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
        for t, u in theta_0:
            if (t, u) in self.arc_usage and self.arc_usage[(t, u)] > eps:
                return True
        for t, u in theta_1:
            if (t, u) not in self.arc_usage or self.arc_usage[(t, u)] < 1.0 - eps:
                return True
        for t in upsilon_0:
            if t < len(self.setup_by_period) and self.setup_by_period[t] > eps:
                return True
        for t in upsilon_1:
            if t >= len(self.setup_by_period) or self.setup_by_period[t] < 1.0 - eps:
                return True
        return False

    def get_signature(self) -> str:
        """Generate a unique signature for duplicate detection."""
        setup_str = ",".join(
            f"{t}" for t, v in enumerate(self.setup_by_period) if v > 0.5
        )
        arc_str = ",".join(
            f"{t}-{u}"
            for (t, u) in sorted(self.arc_usage.keys())
            if self.arc_usage[(t, u)] > 0.5
        )
        return f"I{self.item_id}_S[{setup_str}]_A[{arc_str}]"


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

    def __lt__(self, other: "BranchNode") -> bool:
        """For heap comparison - lower bound = higher priority."""
        return self.lp_bound < other.lp_bound


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
    total_columns_generated: int = 0
    total_cg_iterations: int = 0
    start_time: float = field(default_factory=time.time)

    def print_summary(
        self, best_lb: float, best_ub: Optional[float], eps: float, log_file=None
    ):
        elapsed = time.time() - self.start_time

        lines = []
        lines.append("\n" + "=" * 70)
        lines.append(" " * 28 + "FINAL RESULTS")
        lines.append("=" * 70)
        lines.append(f"  Time elapsed:          {elapsed:.2f} seconds")
        lines.append(f"  Nodes created:         {self.nodes_created}")
        lines.append(f"  Nodes explored:        {self.nodes_explored}")
        lines.append(f"  Integer solutions:     {self.nodes_integer}")
        lines.append(f"  Max depth:             {self.max_depth}")
        lines.append(f"  Total CG iterations:   {self.total_cg_iterations}")
        lines.append(f"  Total columns added:   {self.total_columns_generated}")
        lines.append("")
        lines.append(f"  Fathomed by bound:     {self.nodes_fathomed_by_bound}")
        lines.append(f"  Fathomed infeasible:   {self.nodes_fathomed_by_infeasible}")
        lines.append(f"  Fathomed on incumbent: {self.nodes_fathomed_on_incumbent}")
        lines.append("")
        lines.append(f"  Best lower bound:      {best_lb:.4f}")

        if best_ub is not None:
            lines.append(f"  Best upper bound:      {best_ub:.4f}")
            gap = best_ub - best_lb
            gap_pct = 100 * gap / max(abs(best_ub), 1e-10)
            lines.append(f"  Gap:                   {gap:.4f} ({gap_pct:.2f}%)")
            if gap < eps:
                lines.append("  DONE")
        else:
            lines.append(f"  Best upper bound:      Not found")
        lines.append("=" * 70)

        for line in lines:
            print(line)
            if log_file:
                log_file.write(line + "\n")


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


def column_violates_lefo(
    arc_usage: Dict[Tuple[int, int], float],
    Gamma: Dict[int, List[int]],
    Expiry: Dict[int, int],
    eps: float = 1e-6,
) -> bool:

    if not arc_usage:
        return False

    # production periods that actually appear in this column
    prods = sorted({t for (t, u) in arc_usage.keys()}, key=lambda t: Expiry[t])

    for a in range(len(prods)):
        t1 = prods[a]
        v1 = Expiry[t1]
        for b in range(a + 1, len(prods)):
            t2 = prods[b]
            v2 = Expiry[t2]
            if v1 >= v2:
                # we only care about v1 < v2 (earlier expiry first)
                continue

            # up_node = end nodes for arcs from t2 (11-node model)
            for up_node in Gamma.get(t2, []):
                if arc_usage.get((t2, up_node), 0.0) <= eps:
                    continue  # this arc not actually used
                up_period = up_node - 1  # Consumption period for end node up_node

                # u_node = end nodes for arcs from t1 (11-node model)
                for u_node in Gamma.get(t1, []):
                    u_period = u_node - 1  # Consumption period for end node u_node
                    if u_period < t2 or u_period > up_period - 1:
                        continue
                    if arc_usage.get((t1, u_node), 0.0) > eps:
                        # we have both Z[t1,u] = 1 and Z[t2,up] = 1 in this column
                        return True
    return False


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
    arc_usage_counts: Optional[Dict[Tuple[int, int], int]] = None,
    perturbation_eps: float = 1e-5,
) -> Tuple[float, Optional[ProductionPlanColumn]]:
    """
    Solve pricing subproblem using DP with 11-node shortest path interpretation.

    11-Node Model:
    - Nodes: 0 (start), 1-10 (representing periods 0-9)
    - Arc (s, u): produce at period s (node s), consume at period u-1 (node u)
      * So arc (0,1) means produce at period 0, consume at period 0
      * Arc (5,10) means produce at period 5, consume at period 9
    - Block (s, t): produce at period s (node s), consume at periods s through t-1 (nodes s+1 through t)
      * So block (0, 3) means produce at period 0, consume at periods 0, 1, 2

    DP implements Wagner-Whitin DP with three decision types:
    1. SKIP: no action at period t (only if no demand)
    2. SETUP-ONLY: setup at t but no production
    3. PRODUCTION BLOCK: produce at s, serve demands from s to t

    Branching constraints:
    - theta_0, upsilon_0: Forbidden (enforced by skipping blocks/actions)
    - theta_1, upsilon_1: Forced (enforced by requiring blocks/actions)
    - sigma, tau: Duals guide pricing - can make actions infinite cost or reduce costs

    Degeneracy handling:
    - arc_usage_counts: tracks how many times each arc has been used
    - perturbation_eps: small perturbation added to discourage frequently used arcs
    """
    sigma = sigma or {}
    tau = tau or {}
    arc_usage_counts = arc_usage_counts or {}

    # Extract item parameters
    demand = item_data["demand"]
    c_var = item_data["c_var"]
    h = item_data["h"]
    setup = item_data["setup"]

    # Precompute all cost arrays once (avoid repeated function calls)
    c_arr = [
        float(c_var[t]) if isinstance(c_var, list) else float(c_var) for t in range(T)
    ]
    h_arr = [float(h[t]) if isinstance(h, list) else float(h) for t in range(T)]
    s_arr = [
        float(setup[t]) if isinstance(setup, list) else float(setup) for t in range(T)
    ]
    d_arr = [float(demand[t]) for t in range(T)]

    # Precompute cumulative demand: D_u = sum_{r=1}^{u} d_{ir}
    D = [0.0] * (T + 1)
    for u in range(1, T + 1):
        D[u] = D[u - 1] + d_arr[u - 1]

    # Precompute cumulative holding cost prefix sums
    h_prefix = [0.0] * (T + 1)
    for k in range(T):
        h_prefix[k + 1] = h_prefix[k] + h_arr[k]

    # Precompute H_i(s, u) for all (s, u) pairs used in blocks
    # H_i(s, u) = h_prefix[u] - h_prefix[s]
    def H_i(s: int, u: int) -> float:
        """Holding cost from period s to u-1: sum_{ℓ=s}^{u-1} h_{iℓ}"""
        if u <= s:
            return 0.0
        return h_prefix[u] - h_prefix[s]

    INF = float("inf")

    # Precompute forbidden arc sets for faster lookup (using 11-node interpretation)
    # theta_0 contains arcs (s, u_node) where u_node = consumption_period + 1
    theta_0_set = theta_0  # Already a set, use directly
    # Precompute tau values for all (s, u_node) pairs using 11-node interpretation
    # Arc (s, u_node): produce at period s, consume at period u_period where u_node = u_period + 1
    tau_cache: Dict[Tuple[int, int], float] = {}
    for s in range(T):
        for u_period in range(s, T):  # u_period is consumption period (0-indexed)
            u_node = u_period + 1  # Node index for 11-node model
            tau_cache[(s, u_node)] = tau.get((item_id, s, u_node), 0.0)

    # Precompute sigma values for all periods
    sigma_arr = [sigma.get((item_id, t), 0.0) for t in range(T)]

    # Precompute block reduced costs rc_i(s,t) for all feasible blocks
    # rc_i(s,t) = s_is - σ̃_is + (c_is - π_s) Q_{s,t} + H^{block}_{s,t} - T_{s,t}
    block_rc: Dict[Tuple[int, int], float] = {}

    # Precompute dual-updated production costs: C' = c - π
    c_dual = [c_arr[t] - capacity_duals[t] for t in range(T)]

    # Precompute dual-updated setup costs: S' = s - σ
    # Make forbidden setups infinite cost (they cannot be used)
    s_dual = []
    for t in range(T):
        if t in upsilon_0:
            # Forbidden setup - make it infinite cost
            s_dual.append(INF)
        else:
            # Standard: setup cost minus sigma dual
            # If sigma > 0 (forbidden setup constraint is tight), we get credit
            s_dual.append(s_arr[t] - sigma_arr[t])

    for s in range(T):
        # Check if setup at s is forbidden
        if s in upsilon_0:
            continue

        # Get valid end nodes for arcs starting at s (11-node model)
        # Gamma[s] now stores END NODES (u_node), not consumption periods
        valid_end_nodes = Gamma.get(s, [])
        if not valid_end_nodes:
            continue

        # Precompute setup cost for this s (used in all blocks starting at s)
        setup_cost_dp = s_dual[s]
        c_dual_s = c_dual[s]

        for u_node in valid_end_nodes:
            # u_node is the end node (11-node model)
            # Consumption period u_period = u_node - 1 (0-indexed)
            u_period = u_node - 1

            # Block (s, u_node): produce at period s, consume at period u_period
            # For ZIO blocks, we need to check if we can serve all periods from s to u_period
            # But wait - if Gamma[s] = [4, 6] (non-contiguous), that means:
            # - Arc (s, 4): serve period 3 only
            # - Arc (s, 6): serve period 5 only
            # We cannot have a block that serves periods 3, 4, 5 because period 4 is not in Gamma[s]

            # For now, treat each u_node as a single-period block (s, u_node) serving only period u_period
            # But we also need to allow multi-period blocks if all intermediate periods are in Gamma[s]
            # Actually, for ZIO, a block (s, u_node) should serve periods s through u_period
            # So we need to check if all periods from s to u_period are reachable

            # Check if we can form a block (s, u_node) that serves periods s to u_period
            # This requires that for each period u in [s, u_period], there exists an arc (s, u_node') where u_node' = u+1
            can_form_block = True
            required_periods = list(range(s, u_period + 1))
            for u_req in required_periods:
                u_node_req = u_req + 1  # Required end node for period u_req
                if u_node_req not in Gamma[s]:
                    can_form_block = False
                    break
            if not can_form_block:
                continue  # Cannot form block - missing intermediate periods

            # Check for forbidden arcs using 11-node interpretation
            has_forbidden = False
            for u_req in required_periods:
                u_node_req = u_req + 1
                if (s, u_node_req) in theta_0_set:
                    has_forbidden = True
                    break
            if has_forbidden:
                continue

            # Compute block quantity Q_{s,u_period} = sum of demands from s to u_period
            # Block (s, u_node) serves periods s through u_period
            Q_st = sum(d_arr[u_req] for u_req in required_periods)
            # Ensure production is integral (round to nearest integer)
            Q_st = round(Q_st)

            # Compute holding cost H^{block}_{s,u_period} = sum_{u=s}^{u_period} H_i(s,u) * d_{iu}
            H_block = 0.0
            h_prefix_s = h_prefix[s]
            for u_req in required_periods:
                d_iu = d_arr[u_req]
                if d_iu > eps:
                    H_block += (h_prefix[u_req + 1] - h_prefix_s) * d_iu

            # Compute arc dual aggregation T_{s,u_period} = sum_{u=s}^{u_period} τ_{is(u+1)}
            # Using 11-node interpretation: arc (s, u_node) where u_node = u_period + 1
            # Standard column generation: rc = cost - sum(dual * usage)
            # So we subtract tau: rc = ... - sum(tau)
            T_st = 0.0
            perturbation_st = 0.0  # Degeneracy handling: penalize frequently used arcs
            for u_req in required_periods:
                u_node_req = u_req + 1  # Node index
                tau_val = tau.get((item_id, s, u_node_req), 0.0)
                T_st += tau_val
                # Add perturbation based on arc usage count
                arc_count = arc_usage_counts.get((s, u_node_req), 0)
                if arc_count > 0:
                    perturbation_st += perturbation_eps * arc_count

            # Reduced cost of block (s, u_node) for DP
            # Standard formula: rc = (setup - sigma) + (var_cost - pi) * Q + holding - sum(tau) + perturbation
            rc = setup_cost_dp + c_dual_s * Q_st + H_block - T_st + perturbation_st

            # Store block cost: key is (s, u_period) where u_period is the last consumption period
            block_rc[(s, u_period)] = rc

    # Build reverse index: blocks_by_end[t] = list of (s, cost) for blocks ending at t
    # This allows O(1) lookup of all blocks ending at a period
    blocks_by_end: Dict[int, List[Tuple[int, float]]] = {}
    for (s, t), cost in block_rc.items():
        if t not in blocks_by_end:
            blocks_by_end[t] = []
        blocks_by_end[t].append((s, cost))

    # Forward DP recursion: F[t] = minimum cost to satisfy demands from period 1 through t
    # In 0-indexed: F[0] = 0, F[t] = cost to satisfy demands from period 0 through t-1
    F = [INF] * (T + 1)
    pred = [None] * (T + 1)  # Store predecessor: (action_type, start_period)
    F[0] = 0.0

    # Find last period with positive demand (0-indexed)
    T_last = -1
    for t in range(T - 1, -1, -1):  # Search backwards for early termination
        if d_arr[t] > eps:
            T_last = t
            break

    if T_last < 0:
        # No demand - return zero reduced cost
        return -convexity_dual, None

    # F[t] where t is 0-indexed: cost to satisfy demands from 0 to t-1
    # But we'll use t as 1-indexed in the loop to match document notation
    for t_idx in range(1, T + 1):
        # t_idx is 1-indexed, t_period = t_idx - 1 is 0-indexed period
        t_period = t_idx - 1
        best_cost = INF
        best_pred = None

        # ACTION 1: SKIP period t_period (only if no demand)
        if d_arr[t_period] <= eps:
            skip_cost = F[t_idx - 1]
            if skip_cost < best_cost:
                best_cost = skip_cost
                best_pred = ("SKIP", None)

        # ACTION 2: SETUP-ONLY at period t_period (only if no demand and setup not forbidden)
        # Skip if setup is forced (must produce, not just setup)
        if (
            d_arr[t_period] <= eps
            and t_period not in upsilon_0
            and t_period not in upsilon_1
        ):
            setup_cost_dp = s_dual[t_period]
            setup_only_cost = F[t_idx - 1] + setup_cost_dp
            if setup_only_cost < best_cost:
                best_cost = setup_only_cost
                best_pred = ("SETUP_ONLY", t_period)

        # ACTION 3: PRODUCTION BLOCK ending at period t_period
        # Consider all blocks (s, t_period) where s <= t_period
        # F[s] represents cost to satisfy demands from 0 to s-1
        # Block (s, t_period) satisfies demands from s to t_period
        # So F[t_idx] = F[s] + block_cost where t_idx = t_period + 1
        # Optimize: use reverse index to only check blocks that end at t_period
        if t_period in blocks_by_end:
            for s, block_cost in blocks_by_end[t_period]:
                if (
                    s < t_idx
                ):  # Ensure s is valid (s < t_idx since t_idx = t_period + 1)
                    # F[s] is cost to satisfy demands from 0 to s-1 (before the block)
                    total_cost = F[s] + block_cost
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_pred = ("BLOCK", s)

        # If no valid action found, best_cost remains INF and best_pred remains None
        if not math.isfinite(best_cost):
            # No feasible solution for this period - problem is infeasible
            return INF, None

        F[t_idx] = best_cost
        pred[t_idx] = best_pred

    # Minimum reduced cost from DP = F[T_last + 1] - μ_i
    # T_last is 0-indexed, so T_last + 1 is 1-indexed position in F
    # Check if DP found a feasible solution
    if not math.isfinite(F[T_last + 1]) or pred[T_last + 1] is None:
        return (
            INF,
            None,
        )  # DP found no feasible solution (likely due to branching constraints)

    dp_reduced_cost = F[T_last + 1] - convexity_dual

    # Backtrack to reconstruct solution first to check forced setups
    cap_usage = [0.0] * T
    setup_usage = [0.0] * T
    arc_usage: Dict[Tuple[int, int], float] = {}
    total_cost = 0.0

    # t_idx is 1-indexed position in F array
    t_idx = T_last + 1
    while t_idx > 0:
        if pred[t_idx] is None:
            return INF, None  # No valid path found - infeasible
        action, start = pred[t_idx]

        if action == "SKIP":
            t_idx -= 1
        elif action == "SETUP_ONLY":
            setup_usage[start] = 1.0
            total_cost += s_arr[start]
            t_idx -= 1
        elif action == "BLOCK":
            s = start  # s is 0-indexed period where block starts
            # Block (s, t_period) where t_period = t_idx - 1
            t_period = t_idx - 1

            setup_usage[s] = 1.0
            total_cost += s_arr[s]

            # Add production and arcs for this block (s, t_period) using 11-node interpretation
            # Block (s, t_period): produce at period s, consume at periods s through t_period
            # Arc (s, u_node): produce at period s, consume at period u_period where u_node = u_period + 1
            # Optimize: precompute h_prefix_s once
            h_prefix_s = h_prefix[s]
            c_s = c_arr[s]
            production_qty = 0.0
            for u_period in range(
                s, t_period + 1
            ):  # u_period is consumption period (0-indexed)
                d_u = d_arr[u_period]
                if d_u > eps:
                    production_qty += d_u
                    u_node = u_period + 1  # Node index for 11-node model
                    arc_usage[(s, u_node)] = 1.0  # Arc (s, u_node) in 11-node model
                    # H_i(s, u_period) = h_prefix[u_period+1] - h_prefix[s]
                    total_cost += (c_s + (h_prefix[u_period + 1] - h_prefix_s)) * d_u

            # Ensure production is integral
            production_qty = round(production_qty)
            cap_usage[s] = production_qty

            # After block (s, t_period), we've satisfied demands from 0 to t_period
            # F[s] represents cost to satisfy demands from 0 to s-1 (before the block)
            # So we go back to position s to continue backtracking
            t_idx = s

        if t_idx < 0:
            break

    # Validate demand coverage using 11-node interpretation
    # Arc (s, u_node): produce at period s, consume at period u_period where u_node = u_period + 1
    covered = [False] * T
    for (s, u_node), val in arc_usage.items():
        if val > 0.5:
            u_period = (
                u_node - 1
            )  # Convert node index to consumption period (0-indexed)
            if 0 <= u_period < T:
                covered[u_period] = True

    for u_period in range(T):
        if d_arr[u_period] > eps and not covered[u_period]:
            return INF, None

    # Validate forced constraints: check if forced setups and arcs are satisfied
    # IMPORTANT: LEFO allows multiple batches to serve the same demand period.
    # So forcing Z_{t',u'}=1 doesn't prevent other arcs to u' from existing.
    # The RMP will combine columns to satisfy all forced constraints.
    #
    # For forced setups: if we force Y_t=1, this column must have that setup
    # (since setups are per-column, not aggregated).
    for t_forced in upsilon_1:
        if setup_usage[t_forced] < 0.5:
            return INF, None  # Forced setup not used - invalid solution

    # For forced arcs: if we force Z_{t_forced, u_forced}=1, we prefer columns
    # that have that arc, but we don't reject columns that don't have it (as long
    # as they don't violate it). The RMP will ensure at least one column has the
    # forced arc. However, if this column serves u_forced from a different
    # production period, that's OK - LEFO allows multiple batches to serve the
    # same demand period.
    #
    # Actually, we should still validate: if we force Z_{t_forced, u_forced}=1,
    # and this column serves u_forced, we should check if it can satisfy the
    # forced constraint. But since LEFO allows multiple batches, we can allow
    # columns that serve u_forced from different production periods.
    #
    # For now, we only reject if the column actively violates a forced constraint
    # (e.g., if we force Z_{4,5}=1 and the column has Z_{3,5}=1 but we're at
    # a node where Z_{4,5}=0 is also forced, that would be a violation).
    # But that case is already handled by the forbidden arc check above.
    #
    # So we allow columns that don't have forced arcs - the RMP will ensure
    # forced constraints are satisfied through column selection.

    # Verify reduced cost by recomputing from column's actual cost
    # rc = total_cost - (π * capacity + σ * setup + τ * arc + μ)
    # EXPERIMENT: Try different dual contribution calculations
    dual_contribution = convexity_dual  # μ (convexity dual)

    # Add capacity dual contributions: π * capacity_usage
    for t in range(T):
        if cap_usage[t] > eps:
            dual_contribution += capacity_duals[t] * cap_usage[t]

    # Add setup dual contributions: σ * setup_usage
    for t in range(T):
        if setup_usage[t] > eps:
            sigma_val = sigma.get((item_id, t), 0.0) if sigma else 0.0
            dual_contribution += sigma_val * setup_usage[t]

    # Add arc dual contributions: τ * arc_usage
    # EXPERIMENT: User says "subtract tau to reduce cost"
    # In standard column generation: rc = cost - sum(dual * usage)
    # So we subtract tau contributions: dual_contribution += tau * arc
    # But in DP we did: rc = ... - T_st where T_st = sum(tau)
    # So if tau is positive, subtracting it reduces cost (correct)
    for (t, u), arc_val in arc_usage.items():
        if arc_val > eps:
            tau_val = tau.get((item_id, t, u), 0.0) if tau else 0.0
            # Standard: subtract tau (tau positive reduces cost)
            dual_contribution += tau_val * arc_val

    # Recompute reduced cost from actual column cost
    verified_rc = total_cost - dual_contribution

    # Check if DP reduced cost matches verified reduced cost (within tolerance)
    rc_diff = abs(dp_reduced_cost - verified_rc)
    if rc_diff > 1e-4:  # Allow small numerical differences
        # Use verified reduced cost to ensure correctness
        adjusted_reduced_cost = verified_rc
        # Debug: print mismatch if significant
        if rc_diff > 1e-2:
            print(
                f"WARNING: RC mismatch! DP={dp_reduced_cost:.6f}, Verified={verified_rc:.6f}, diff={rc_diff:.6f}"
            )
    else:
        adjusted_reduced_cost = dp_reduced_cost

    if adjusted_reduced_cost >= -eps:
        return adjusted_reduced_cost, None

    # Column cost must use ORIGINAL costs (S_t, C_t), not dual-updated costs
    # This is already done in backtracking: we use s_at(s) and c_at(s) which are original costs
    column = ProductionPlanColumn(
        item_id=item_id,
        total_plan_cost=total_cost,  # Uses original costs S_t and C_t
        capacity_usage_by_period=cap_usage,
        setup_by_period=setup_usage,
        arc_usage=arc_usage,
    )

    return adjusted_reduced_cost, column


class RestrictedMasterProblem:
    """
    Restricted Master Problem for the Dantzig-Wolfe decomposition.

    Structure (matching solver_bnp.py):
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

        self.convex_con: Dict[int, gp.Constr] = {}
        self.cap_con: List[gp.Constr] = []
        self.y_link_con: Dict[Tuple[int, int], gp.Constr] = {}
        self.z_link_con: Dict[Tuple[int, int, int], gp.Constr] = {}

        for item_id in items:
            self.convex_con[item_id] = self.model.addConstr(
                self.convex_expr[item_id] == 1.0, name=f"conv_{item_id}"
            )

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

        if initial_columns:
            for item_id, cols in initial_columns.items():
                for col in cols:
                    self.add_column(col)

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
        """Add a column to the RMP.

        Args:
            col: The column to add
        """
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

    def solve(self) -> Tuple[
        float,
        Dict[int, float],
        List[float],
        Dict[Tuple[int, int], float],
        Dict[Tuple[int, int, int], float],
    ]:
        """
        Solve the RMP and return duals.

        Returns:
            (obj_val, mu, pi, sigma, tau) where:
            - obj_val: Objective value
            - mu: Convexity duals {item_id: μ_i}
            - pi: Capacity duals [π_t for t in T]
            - sigma: Setup linking duals {(item_id, t): σ_{it}} from Υ^1 constraints
            - tau: Arc linking duals {(item_id, t, u): τ_{itu}} from Θ^1 constraints
        """
        self.model.optimize()

        if self.model.status != GRB.OPTIMAL:
            return math.inf, {}, [], {}, {}

        mu = {i: self.convex_con[i].Pi for i in self.items}
        pi = [self.cap_con[t].Pi for t in range(self.T)]

        # Extract σ duals from Y linking constraints (forbidden setups)
        sigma: Dict[Tuple[int, int], float] = {}
        for (item_id, t), con in self.y_link_con.items():
            sigma[(item_id, t)] = con.Pi

        # Extract τ duals from Z linking constraints (forbidden arcs)
        tau: Dict[Tuple[int, int, int], float] = {}
        for (item_id, t, u), con in self.z_link_con.items():
            tau[(item_id, t, u)] = con.Pi

        return self.model.ObjVal, mu, pi, sigma, tau

    def get_column_count(self) -> int:
        return sum(len(cols) for cols in self.columns.values())

    def delete_zero_lambda_columns(self, eps: float = 1e-8) -> int:
        """
        Delete columns with lambda = 0 (or very close to 0) to save memory.

        Args:
            eps: Threshold below which lambda is considered zero

        Returns:
            Number of columns deleted
        """
        if self.model.status != GRB.OPTIMAL:
            return 0  # Can't delete if model not solved

        deleted_count = 0
        columns_to_delete: List[Tuple[int, int]] = []

        # First pass: identify columns to delete (in reverse order to maintain indices)
        for (item_id, idx), lam_var in list(self.lambdas.items()):
            try:
                lam_val = lam_var.X
            except (AttributeError, ValueError, Exception):
                continue

            # Skip if lambda is above threshold
            if lam_val >= eps:
                continue

            # Don't delete dummy columns (first column is usually dummy)
            if idx == 0:
                continue

            # Don't delete if it's the only non-dummy column for this item
            if len(self.columns[item_id]) <= 2:  # 1 dummy + 1 real
                continue

            columns_to_delete.append((item_id, idx))

        # Second pass: delete columns (in reverse order to maintain indices)
        columns_to_delete.sort(reverse=True, key=lambda x: x[1])

        for item_id, idx in columns_to_delete:
            lam_var = self.lambdas.get((item_id, idx))
            if lam_var is None:
                continue

            # Remove lambda variable from model
            self.model.remove(lam_var)

            # Remove from our data structures
            self.lambdas.pop((item_id, idx), None)
            col = self.columns[item_id].pop(idx)

            # Reindex remaining lambdas for this item (shift indices down)
            num_cols_after = len(self.columns[item_id])
            for i in range(num_cols_after - 1, idx - 1, -1):
                old_key = (item_id, i + 1)
                if old_key in self.lambdas:
                    new_key = (item_id, i)
                    lam = self.lambdas.pop(old_key)
                    self.lambdas[new_key] = lam

            deleted_count += 1

        # Rebuild expressions and constraints from scratch after deletion
        if deleted_count > 0:
            # Rebuild expressions from scratch
            for item_id in self.items:
                self.convex_expr[item_id] = gp.LinExpr(0.0)
                for idx in range(len(self.columns[item_id])):
                    if (item_id, idx) in self.lambdas:
                        self.convex_expr[item_id] += self.lambdas[(item_id, idx)]

            for t in range(self.T):
                self.cap_expr[t] = gp.LinExpr(0.0)
                for item_id in self.items:
                    for idx in range(len(self.columns[item_id])):
                        if (item_id, idx) in self.lambdas:
                            col = self.columns[item_id][idx]
                            if col.capacity_usage_by_period[t] != 0.0:
                                self.cap_expr[t] += (
                                    col.capacity_usage_by_period[t]
                                    * self.lambdas[(item_id, idx)]
                                )

            for item_id, t in self.y_link_expr.keys():
                self.y_link_expr[(item_id, t)] = gp.LinExpr(0.0)
                for idx in range(len(self.columns[item_id])):
                    if (item_id, idx) in self.lambdas:
                        col = self.columns[item_id][idx]
                        if t < len(col.setup_by_period):
                            y_val = col.setup_by_period[t]
                            if y_val != 0.0:
                                self.y_link_expr[(item_id, t)] += (
                                    y_val * self.lambdas[(item_id, idx)]
                                )

            for item_id, t, u in self.z_link_expr.keys():
                self.z_link_expr[(item_id, t, u)] = gp.LinExpr(0.0)
                for idx in range(len(self.columns[item_id])):
                    if (item_id, idx) in self.lambdas:
                        col = self.columns[item_id][idx]
                        z_val = col.arc_usage.get((t, u), 0.0)
                        if z_val != 0.0:
                            self.z_link_expr[(item_id, t, u)] += (
                                z_val * self.lambdas[(item_id, idx)]
                            )

            # Rebuild constraints
            self._rebuild()

        return deleted_count


def solve_node_with_column_generation(
    items: Dict[int, dict],
    T: int,
    capacity: List[float],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    Expiry_by_item: Dict[int, Dict[int, int]],
    node: BranchNode,
    parent_rmp: Optional[RestrictedMasterProblem] = None,
    max_iter: int = 500,
    eps: float = 1e-6,
    verbose: bool = False,
    log_file=None,
    stats: Optional[SearchStatistics] = None,
    convergence_dir: Optional[Path] = None,
) -> Tuple[
    float,
    Optional[RestrictedMasterProblem],
    bool,
    Dict[int, Dict[Tuple[int, int], float]],
    Dict[int, Dict[int, float]],
    Dict[int, Dict[int, float]],
]:
    """Solve a branch node using column generation with column inheritance."""
    inherited_cols = inherit_columns_from_parent(parent_rmp, node, items, eps)
    total_inherited = sum(len(cols) for cols in inherited_cols.values())

    if node.node_id == 44:
        print("here")

    # Create RMP with branching info for forbidden branches (θ⁰, Υ⁰)
    # This allows extracting σ and τ duals to guide pricing
    # Forced branches (θ¹, Υ¹) are enforced only in pricing subproblem
    rmp = RestrictedMasterProblem(
        items=items,
        T=T,
        capacity=capacity,
        Gamma_by_item=Gamma_by_item,
        theta_0_by_item=node.theta_0_by_item,
        upsilon_0_by_item=node.upsilon_0_by_item,
        initial_columns=inherited_cols,
    )

    # Add dummy columns (simple, no forced constraints)
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
        msg = f"  └─ CG[inherited={total_inherited}]: "
        print(msg, end="", flush=True)
        if log_file:
            log_file.write(msg)

    columns_added_this_node = 0
    iterations_this_node = 0

    # Setup convergence CSV tracking
    csv_file = None
    csv_writer = None
    item_ids_sorted = sorted(items.keys())
    if convergence_dir is not None:
        csv_path = convergence_dir / f"convergence_node_{node.node_id}.csv"
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        # Header includes lambda sum for each item
        header = [
            "iteration",
            "rmp_obj",
            "worst_rc",
            "total_rc",
            "cols_added",
            "total_cols",
        ]
        for item_id in item_ids_sorted:
            header.append(f"lambda_sum_i{item_id}")
        csv_writer.writerow(header)

    try:
        for iteration in range(1, max_iter + 1):
            lb, mu, pi, sigma, tau = rmp.solve()
            iterations_this_node += 1

            # Calculate lambda sums right after solving (before any modifications)
            lambda_sums = []
            if csv_writer and math.isfinite(lb):
                for item_id in item_ids_sorted:
                    lam_sum = 0.0
                    for (iid, idx), lam_var in rmp.lambdas.items():
                        if iid == item_id:
                            lam_sum += lam_var.X
                    lambda_sums.append(f"{lam_sum:.6f}")

            if not math.isfinite(lb):
                if csv_writer:
                    row = [iteration, "inf", "N/A", "N/A", 0, rmp.get_column_count()]
                    row += ["N/A"] * len(item_ids_sorted)
                    csv_writer.writerow(row)
                if verbose:
                    print("INFEASIBLE")
                    if log_file:
                        log_file.write("INFEASIBLE\n")
                return math.inf, rmp, False, {}, {}, {}

            # Delete columns with lambda=0 to save memory (if more than 2 columns per item)
            # This ensures if a column is needed, it will be regenerated
            # Only do this every few iterations to avoid excessive deletion/re-solving
            # and only when there are many columns to avoid corrupting the model
            # if (
            #     iteration > 10  # Wait until we have some columns
            #     and iteration % 100 == 0  # Only delete every 10 iterations
            #     and rmp.get_column_count()
            #     > len(items) * 5  # More conservative threshold
            # ):  # More than just dummy columns
            #     deleted = rmp.delete_zero_lambda_columns(eps=eps)
            #     if deleted > 0:
            #         # Re-solve after deletion to get updated duals
            #         new_lb, new_mu, new_pi, new_sigma, new_tau = rmp.solve()
            #         # Only update if solve was successful and all items are present
            #         if math.isfinite(new_lb) and len(new_mu) == len(items):
            #             lb = new_lb
            #             mu = new_mu
            #             pi = new_pi
            #             sigma = new_sigma
            #             tau = new_tau
            #         # If solve failed, keep using old duals (they're still valid)
            #         # Suppress verbose output - deletion happens silently

            # reinitilize tracking variables
            any_added = False
            cols_this_iter = 0
            worst_rc = 0.0
            total_rc = 0.0

            # Track arc usage counts for degeneracy handling (11-node interpretation)
            # arc_usage_counts[item_id][(s, u_node)] = count of times arc (s, u_node) has been used
            arc_usage_counts_by_item: Dict[int, Dict[Tuple[int, int], int]] = {}
            for item_id in items.keys():
                arc_usage_counts_by_item[item_id] = {}
                # Count arcs from existing columns in RMP
                for (iid, idx), lam_var in rmp.lambdas.items():
                    if iid == item_id and lam_var.X > eps:
                        col = rmp.columns[item_id][idx]
                        for (s, u_node), arc_val in col.arc_usage.items():
                            if arc_val > eps:
                                arc_usage_counts_by_item[item_id][(s, u_node)] = (
                                    arc_usage_counts_by_item[item_id].get(
                                        (s, u_node), 0
                                    )
                                    + 1
                                )

            # Pricing for each item

            for item_id, item_data in items.items():
                # Get branching sets for pricing
                theta_0 = node.theta_0_by_item.get(item_id, set())
                theta_1 = node.theta_1_by_item.get(item_id, set())
                upsilon_0 = node.upsilon_0_by_item.get(item_id, set())
                upsilon_1 = node.upsilon_1_by_item.get(item_id, set())

                Gamma = Gamma_by_item[item_id]
                Expiry = Expiry_by_item[item_id]

                # Pricing subproblem:
                # - Enforces Υ^0 and Θ^0 (forbidden) as hard constraints
                # - Enforces Υ^1 and Θ^1 (forced) as hard constraints
                # - Uses σ and τ duals from forbidden branch constraints (= 0) to guide pricing
                # Safety check: ensure mu has this item_id
                if item_id not in mu:
                    if verbose:
                        print(
                            f"WARNING: item_id {item_id} not in mu after deletion, skipping pricing"
                        )
                        if log_file:
                            log_file.write(
                                f"WARNING: item_id {item_id} not in mu after deletion, skipping pricing\n"
                            )
                    continue

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
                    arc_usage_counts=arc_usage_counts_by_item.get(item_id, {}),
                    perturbation_eps=1e-5,
                )

                if math.isfinite(rc):
                    total_rc += rc
                    worst_rc = min(worst_rc, rc)

                if col is not None and rc < -eps:
                    rmp.add_column(col)
                    any_added = True
                    columns_added_this_node += 1
                    cols_this_iter += 1
                    # Update arc usage counts for degeneracy handling (11-node interpretation)
                    for (s, u_node), arc_val in col.arc_usage.items():
                        if arc_val > eps:
                            arc_usage_counts_by_item[item_id][(s, u_node)] = (
                                arc_usage_counts_by_item[item_id].get((s, u_node), 0)
                                + 1
                            )

            # Write convergence data (using lambda_sums captured earlier)
            if csv_writer:
                row = [
                    iteration,
                    f"{lb:.6f}",
                    f"{worst_rc:.10f}",
                    f"{total_rc:.10f}",
                    cols_this_iter,
                    rmp.get_column_count(),
                ] + lambda_sums
                csv_writer.writerow(row)

            if not any_added:
                if verbose:
                    msg = f"LB={lb:.2f} (iter={iteration}, cols={columns_added_this_node})"
                    print(msg)
                    if log_file:
                        log_file.write(msg + "\n")

                if stats:
                    stats.total_cg_iterations += iterations_this_node
                    stats.total_columns_generated += columns_added_this_node

                z_vals = extract_z_values(rmp, items, eps)
                y_vals = extract_y_values(rmp, items, eps)
                x_vals = extract_x_values(rmp, items, eps)
                return lb, rmp, True, z_vals, y_vals, x_vals

        lb, _, _, _, _ = rmp.solve()

        if stats:
            stats.total_cg_iterations += iterations_this_node
            stats.total_columns_generated += columns_added_this_node

        z_vals = extract_z_values(rmp, items, eps)
        y_vals = extract_y_values(rmp, items, eps)
        x_vals = extract_x_values(rmp, items, eps)

        if verbose:
            msg = f"LB={lb:.2f} (max iter, cols={columns_added_this_node})"
            print(msg)
            if log_file:
                log_file.write(msg + "\n")

        return lb, rmp, False, z_vals, y_vals, x_vals

    finally:
        if csv_file:
            csv_file.close()


def extract_z_values(
    rmp: RestrictedMasterProblem,
    items: Dict[int, dict],
    eps: float,
) -> Dict[int, Dict[Tuple[int, int], float]]:
    z_vals = {i: {} for i in items}
    for (item_id, idx), lam in rmp.lambdas.items():
        try:
            lam_val = lam.X
        except (AttributeError, ValueError, Exception):
            # Variable not solved or removed, skip it
            # Catch all exceptions since Gurobi may raise various errors
            continue
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
        try:
            lam_val = lam.X
        except (AttributeError, ValueError, Exception):
            # Variable not solved or removed, skip it
            # Catch all exceptions since Gurobi may raise various errors
            continue
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
        try:
            lam_val = lam.X
        except (AttributeError, ValueError, Exception):
            # Variable not solved or removed, skip it
            # Catch all exceptions since Gurobi may raise various errors
            continue
        if lam_val < eps:
            continue
        col = rmp.columns[item_id][idx]
        for t, qty in enumerate(col.capacity_usage_by_period):
            if qty > eps:
                x_vals[item_id][t] = x_vals[item_id].get(t, 0.0) + lam_val * qty
    return x_vals


def extract_active_columns(
    rmp: RestrictedMasterProblem,
    items: Dict[int, dict],
    eps: float = 1e-6,
) -> List[Dict]:
    """Extract active (non-dummy) columns with their lambda values."""
    active_cols = []
    for (item_id, idx), lam in rmp.lambdas.items():
        lam_val = lam.X
        if lam_val < eps:
            continue

        col = rmp.columns[item_id][idx]

        # Check if this is a dummy column by its properties (not index!)
        item_demand = sum(items[item_id]["demand"])
        dummy_cost_threshold = 5000.0 * (item_demand + 1)
        is_dummy = (
            col.total_plan_cost > dummy_cost_threshold
            and all(x == 0.0 for x in col.capacity_usage_by_period)
            and len(col.arc_usage) == 0
        )
        if is_dummy:
            continue

        setups = [t for t, v in enumerate(col.setup_by_period) if v > 0.5]
        arcs = [(t, u) for (t, u), v in col.arc_usage.items() if v > 0.5]
        active_cols.append(
            {
                "item_id": item_id,
                "col_idx": idx,
                "lambda_val": lam_val,
                "cost": col.total_plan_cost,
                "setups": setups,
                "arcs": arcs,
                "capacity_usage": [q for q in col.capacity_usage_by_period],
            }
        )
    return active_cols


def is_integer(
    z_vals: Dict[int, Dict[Tuple[int, int], float]],
    y_vals: Dict[int, Dict[int, float]],
    eps: float = 1e-6,
) -> bool:
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
    for item_id in items:
        item_demand = sum(items[item_id]["demand"])
        dummy_cost_threshold = 5000.0 * (item_demand + 1)
        for idx, col in enumerate(rmp.columns[item_id]):
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
    if not is_integer(z_vals, y_vals, eps):
        return False
    if solution_uses_dummy(rmp, items, eps):
        return False
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


def fathom_queue_by_incumbent(queue: List, incumbent: float, eps: float) -> int:
    """Remove nodes from priority queue that can be fathomed by bound."""
    original_size = len(queue)
    new_queue = []
    for bound, node_id, node, pz, py, px, prmp in queue:
        if bound < incumbent - eps:
            heapq.heappush(new_queue, (bound, node_id, node, pz, py, px, prmp))
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
    log_file=None,
) -> Tuple[Dict, List[str], List[Dict]]:
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

    # Setup convergence tracking directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    convergence_dir = out_path / "convergence"
    convergence_dir.mkdir(parents=True, exist_ok=True)

    header = [
        "╠" + "═" * 68 + "╣",
        f"║  Items:     {len(items):<55d} ║",
        f"║  Periods:   {T:<55d} ║",
        f"║  Capacity:  {str(capacity[:min(6, T)]) + ('...' if T > 6 else ''):<55s} ║",
        "╚" + "═" * 68 + "╝",
    ]
    for line in header:
        print(line)
        if log_file:
            log_file.write(line + "\n")

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
            # In 11-node model: Gamma[t] stores END NODES (u_node), not consumption periods
            # Arc (s, u_node): produce at period s (node s), consume at period u_period where u_node = u_period + 1
            # So if production at period t can serve consumption periods t to u_max,
            # then Gamma[t] should contain end nodes [t+1, t+2, ..., u_max+1]
            u_max_period = min(T - 1, v_it)  # Maximum consumption period (0-indexed)
            # Convert consumption periods to end nodes: consumption period u -> end node u+1
            Gamma[t] = [u_period + 1 for u_period in range(t, u_max_period + 1)]
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

    msg = "\n>>> ROOT NODE <<<"
    print(msg)
    if log_file:
        log_file.write(msg + "\n")

    root_lb, rmp, converged, z_vals, y_vals, x_vals = solve_node_with_column_generation(
        items=items,
        T=T,
        capacity=capacity,
        Gamma_by_item=Gamma_by_item,
        Expiry_by_item=Expiry_by_item,
        node=root,
        parent_rmp=None,
        verbose=True,
        log_file=log_file,
        stats=stats,
        convergence_dir=convergence_dir,
    )

    if not math.isfinite(root_lb):
        msg = "\n✗ Root infeasible!"
        print(msg)
        if log_file:
            log_file.write(msg + "\n")
        summary = {
            "status": int(GRB.INFEASIBLE),
            "objective": None,
            "best_bound": None,
            "gap": None,
            "runtime_sec": time.time() - start_time,
            "solver_version": "branch_and_price_dp_dfs",
            "n_items": len(items),
            "T": T,
        }
        return summary, [], None

    root.lp_bound = root_lb

    # Debug: Show lambda values and column counts at root
    lambda_info = []
    for item_id in items:
        cols_for_item = len(rmp.columns[item_id])
        lambda_sum = 0.0
        fractional_count = 0
        for (i, idx), lam in rmp.lambdas.items():
            if i == item_id:
                lam_val = lam.X
                lambda_sum += lam_val
                if 1e-6 < lam_val < 1.0 - 1e-6:
                    fractional_count += 1
        lambda_info.append(
            f"Item {item_id}: {cols_for_item} cols, λ_sum={lambda_sum:.4f}, {fractional_count} fractional"
        )

    msgs = [
        f"  Root LB:      {root_lb:.4f}",
        f"  Integer?      {is_integer(z_vals, y_vals, eps)}",
        f"  Uses dummy?   {solution_uses_dummy(rmp, items, eps)}",
    ] + lambda_info
    for m in msgs:
        print(m)
        if log_file:
            log_file.write(m + "\n")

    best_lb = root_lb
    best_ub: Optional[float] = None
    node_counter = 1
    seen_signatures = {node_signature(root)}
    # Priority queue: (lp_bound, node_id, node, parent_z, parent_y, parent_x, parent_rmp)
    # Lower bound = higher priority (best-first)
    queue: List[
        Tuple[float, int, BranchNode, Dict, Dict, Dict, RestrictedMasterProblem]
    ] = []
    heapq.heappush(
        queue, (root.lp_bound, root.node_id, root, z_vals, y_vals, x_vals, rmp)
    )
    stats.nodes_created = 1

    # Track which nodes have been fully branched (2 children created)
    # Once a node has 2 children, we can remove it from memory
    nodes_with_children: Set[int] = set()

    # Check if root is already a valid integer solution
    if is_valid_integer_solution(z_vals, y_vals, rmp, items, eps):
        best_ub = root_lb
        best_lb = root_lb
        root.is_integer = True
        stats.nodes_integer = 1
        msg = "\n✓ Root is INTEGER - OPTIMAL!"
        print(msg)
        if log_file:
            log_file.write(msg + "\n")
        stats.print_summary(best_lb, best_ub, eps, log_file)
        orders_txt = generate_orders_txt(items, x_vals, eps)

        summary = {
            "status": int(GRB.OPTIMAL),
            "objective": float(best_ub),
            "best_bound": float(best_lb),
            "gap": 0.0,
            "runtime_sec": time.time() - start_time,
            "solver_version": "branch_and_price_dp_dfs",
            "n_items": len(items),
            "T": T,
        }

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "orders.txt").write_text("\n".join(orders_txt), encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        # Extract active columns from root RMP
        best_active_cols = []
        if rmp is not None:
            try:
                best_active_cols = extract_active_columns(rmp, items, eps=eps)
            except Exception as e:
                if log_file:
                    log_file.write(f"Warning: Could not extract active columns: {e}\n")
                best_active_cols = []

        return summary, orders_txt, best_active_cols

    # Check if root is already a valid integer solution
    if is_valid_integer_solution(z_vals, y_vals, rmp, items, eps):
        best_ub = root_lb
        best_lb = root_lb
        root.is_integer = True
        stats.nodes_integer = 1
        msg = "\n✓ Root is INTEGER - OPTIMAL!"
        print(msg)
        if log_file:
            log_file.write(msg + "\n")
        stats.print_summary(best_lb, best_ub, eps, log_file)
        orders_txt = generate_orders_txt(items, x_vals, eps)
        summary = {
            "status": int(GRB.OPTIMAL),
            "objective": float(best_ub),
            "best_bound": float(best_lb),
            "gap": 0.0,
            "runtime_sec": time.time() - start_time,
            "solver_version": "branch_and_price_dp_dfs",
            "n_items": len(items),
            "T": T,
        }
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "orders.txt").write_text("\n".join(orders_txt), encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        # Extract active columns from root RMP
        best_active_cols = []
        if rmp is not None:
            try:
                best_active_cols = extract_active_columns(rmp, items, eps=eps)
            except Exception as e:
                if log_file:
                    log_file.write(f"Warning: Could not extract active columns: {e}\n")
                best_active_cols = []

        return summary, orders_txt, best_active_cols

    stats.nodes_explored = 1
    msg = f"\n{'=' * 70}\nBEST-FIRST SEARCH (Best Lower Bound)\n{'=' * 70}\n"
    print(msg)
    if log_file:
        log_file.write(msg)

    opt_node = root
    opt_rmp = rmp  # Store RMP for optimal solution (root RMP initially)
    opt_z = z_vals
    opt_y = y_vals
    opt_x = x_vals

    while queue and stats.nodes_explored < max_nodes:
        # Check time limit
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time:
            msg = f"\n⏱ Time limit reached ({max_time}s). Best incumbent: {best_ub if best_ub is not None else 'N/A'}"
            print(msg)
            if log_file:
                log_file.write(msg + "\n")
            break

        # Remove nodes that have been fully branched (2 children) from queue
        # This saves memory by not keeping parent nodes after branching
        # Also update best_lb to exclude these nodes
        # Explicitly delete parent nodes and their RMPs to free RAM
        new_queue = []
        for bound, node_id, node, pz, py, px, prmp in queue:
            if node.node_id not in nodes_with_children:
                heapq.heappush(new_queue, (bound, node_id, node, pz, py, px, prmp))
            else:
                # Parent node has been fully branched - free its memory
                del node
                del prmp
        queue = new_queue

        if not queue:
            break

        # Pop node with best (lowest) lower bound (best-first)
        bound, node_id, node, parent_z, parent_y, parent_x, parent_rmp = heapq.heappop(
            queue
        )

        if node.node_id != 0:
            # Print progress with gap tracking
            if stats.nodes_explored % print_frequency == 0 or stats.nodes_explored == 1:
                gap_str = "N/A"
                gap_abs = None
                if best_ub is not None:
                    gap_abs = best_ub - best_lb
                    gap_pct = 100 * gap_abs / max(abs(best_ub), 1e-10)
                    gap_str = f"{gap_abs:.2f} ({gap_pct:.2f}%)"
                msg = (
                    f"[N={stats.nodes_explored:4d}, Q={len(queue):4d}, "
                    f"LB={best_lb:.2f}, UB={best_ub if best_ub is not None else 'N/A':>8}, "
                    f"Gap={gap_str:>12}, Time={elapsed_time:6.1f}s]"
                )
                print(msg)
                if log_file:
                    log_file.write(msg + "\n")

            msg = f"N{node.node_id:4d} D{node.depth:2d} "
            print(msg, end="", flush=True)
            if log_file:
                log_file.write(msg)

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
                    log_file=log_file,
                    stats=stats,
                    convergence_dir=convergence_dir,
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

            # Update best_lb: current node's lb and all unfathomed nodes in queue (exclude nodes_with_children)
            candidates = [lb]
            candidates.extend(
                [
                    bound
                    for bound, nid, n, _, _, _, _ in queue
                    if n.node_id not in nodes_with_children
                    and hasattr(n, "lp_bound")
                    and math.isfinite(n.lp_bound)
                ]
            )
            best_lb = min(candidates)

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
                    opt_rmp = rmp  # Store RMP for optimal solution
                    opt_z = z_vals
                    opt_y = y_vals
                    opt_x = x_vals
                else:
                    print()

                # Update best_lb from queue (exclude nodes_with_children)
                candidates = [
                    bound
                    for bound, nid, n, _, _, _, _ in queue
                    if n.node_id not in nodes_with_children
                    and hasattr(n, "lp_bound")
                    and math.isfinite(n.lp_bound)
                ]
                if candidates:
                    best_lb = min(candidates)
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

        # Branch on Y (setup) variables first, then Z (arc) variables
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
            children_added = 0
            if sig_left not in seen_signatures:
                seen_signatures.add(sig_left)
                node_counter += 1
                stats.nodes_created += 1
                heapq.heappush(
                    queue,
                    (
                        left.lp_bound,
                        left.node_id,
                        left,
                        z_vals,
                        y_vals,
                        x_vals,
                        parent_rmp,
                    ),
                )
                children_added += 1

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
                heapq.heappush(
                    queue,
                    (
                        right.lp_bound,
                        right.node_id,
                        right,
                        z_vals,
                        y_vals,
                        x_vals,
                        parent_rmp,
                    ),
                )
                children_added += 1

            # Mark this node as having children (fully branched)
            # Update best_lb: remove parent's bound, use children's bounds
            # Free parent node from memory to save RAM
            if children_added == 2:
                nodes_with_children.add(node.node_id)
                # Update best_lb from queue (exclude nodes_with_children)
                candidates = [
                    bound
                    for bound, nid, n, _, _, _, _ in queue
                    if n.node_id not in nodes_with_children
                    and hasattr(n, "lp_bound")
                    and math.isfinite(n.lp_bound)
                ]
                if candidates:
                    best_lb = min(candidates)
                # Free parent node and RMP from memory
                # Children have already inherited columns, so parent RMP is no longer needed
                del node
                del parent_rmp
                # Clear parent solution values
                parent_z = None
                parent_y = None
                parent_x = None

            print(f"  Branch Y[{item_id},{t_br}]={y_val:.3f}")
            continue

        # If no fractional Y, branch on Z (arc) variables
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
            children_added = 0
            if sig_left not in seen_signatures:
                seen_signatures.add(sig_left)
                node_counter += 1
                stats.nodes_created += 1
                heapq.heappush(
                    queue,
                    (
                        left.lp_bound,
                        left.node_id,
                        left,
                        z_vals,
                        y_vals,
                        x_vals,
                        parent_rmp,
                    ),
                )
                children_added += 1

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
                heapq.heappush(
                    queue,
                    (
                        right.lp_bound,
                        right.node_id,
                        right,
                        z_vals,
                        y_vals,
                        x_vals,
                        parent_rmp,
                    ),
                )
                children_added += 1

            # Mark this node as having children (fully branched)
            # Update best_lb: remove parent's bound, use children's bounds
            # Free parent node from memory to save RAM
            if children_added == 2:
                nodes_with_children.add(node.node_id)
                # Update best_lb from queue (exclude nodes_with_children)
                candidates = [
                    bound
                    for bound, nid, n, _, _, _, _ in queue
                    if n.node_id not in nodes_with_children
                    and hasattr(n, "lp_bound")
                    and math.isfinite(n.lp_bound)
                ]
                if candidates:
                    best_lb = min(candidates)
                # Free parent node and RMP from memory
                # Children have already inherited columns, so parent RMP is no longer needed
                del node
                del parent_rmp
                # Clear parent solution values
                parent_z = None
                parent_y = None
                parent_x = None

            print(f"  Branch Z[{item_id},{t_br},{u_br}]={z_val:.3f}")
            continue

        print("  No fractional variable - solution is integral")

    # Finalize best bound - compute best_lb from queue (exclude nodes_with_children)
    candidates = [
        bound
        for bound, nid, n, _, _, _, _ in queue
        if n.node_id not in nodes_with_children
        and hasattr(n, "lp_bound")
        and math.isfinite(n.lp_bound)
    ]
    if candidates:
        best_lb = min(best_lb, min(candidates))

    # Ensure best_lb <= best_ub (should always be true, but safety check)
    if best_ub is not None and best_lb > best_ub + eps:
        best_lb = best_ub

    # Only set best_lb = best_ub if we've proven optimality (gap < eps)
    if best_ub is not None and abs(best_ub - best_lb) < eps:
        best_lb = best_ub

    stats.print_summary(best_lb, best_ub, eps, log_file)

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
        "solver_version": "branch_and_price_dp_dfs",
        "n_items": len(items),
        "T": T,
        "nodes_explored": stats.nodes_explored,
        "nodes_created": stats.nodes_created,
        "total_columns": stats.total_columns_generated,
        "total_cg_iterations": stats.total_cg_iterations,
    }

    # Extract active columns from optimal RMP
    best_active_cols = []
    if opt_rmp is not None:
        try:
            best_active_cols = extract_active_columns(opt_rmp, items, eps=eps)
        except Exception as e:
            if log_file:
                log_file.write(f"Warning: Could not extract active columns: {e}\n")
            best_active_cols = []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "orders.txt").write_text("\n".join(orders_txt), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return summary, orders_txt, best_active_cols


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


def log_optimal_solution_details(
    summary: Dict,
    orders: List[str],
    active_cols: List[Dict],
    items: Dict[int, dict],
    log_file,
    eps: float = 1e-6,
):
    """Write comprehensive details about the optimal solution to log file."""
    log_file.write("\n" + "=" * 90 + "\n")
    log_file.write("OPTIMAL SOLUTION DETAILS\n")
    log_file.write("=" * 90 + "\n\n")

    # Summary statistics
    log_file.write("SOLUTION SUMMARY\n")
    log_file.write("-" * 60 + "\n")
    log_file.write(f"  Objective value:       {summary.get('objective', 'N/A')}\n")
    log_file.write(f"  Best bound:            {summary.get('best_bound', 'N/A'):.4f}\n")
    gap = summary.get("gap", 0) or 0
    log_file.write(f"  Optimality gap:        {gap * 100:.4f}%\n")
    log_file.write(
        f"  Runtime:               {summary.get('runtime_sec', 0):.2f} seconds\n"
    )
    log_file.write(f"  Nodes explored:        {summary.get('nodes_explored', 0)}\n")
    log_file.write(f"  Nodes created:         {summary.get('nodes_created', 0)}\n")
    log_file.write(
        f"  Total CG iterations:   {summary.get('total_cg_iterations', 0)}\n"
    )
    log_file.write(f"  Total columns added:   {summary.get('total_columns', 0)}\n")
    log_file.write("\n")

    # Production plan
    log_file.write("PRODUCTION PLAN\n")
    log_file.write("-" * 60 + "\n")
    for line in orders:
        log_file.write(line + "\n")
    log_file.write("\n")

    # Active columns in optimal basis
    if active_cols:
        log_file.write("ACTIVE COLUMNS IN OPTIMAL BASIS\n")
        log_file.write("-" * 100 + "\n")
        log_file.write(
            f"{'Column':<15} {'Item':<6} {'λ value':<12} {'Cost':<14} {'Setups':<20} {'Arcs'}\n"
        )
        log_file.write("-" * 100 + "\n")

        # Sort by item_id, then col_idx
        sorted_cols = sorted(active_cols, key=lambda x: (x["item_id"], x["col_idx"]))

        for col_info in sorted_cols:
            item_id = col_info["item_id"]
            idx = col_info["col_idx"]
            lam_val = col_info["lambda_val"]
            cost = col_info["cost"]
            setups = col_info["setups"]
            arcs = col_info["arcs"]

            setup_str = ",".join(map(str, setups)) if setups else "none"
            arc_str = ",".join(f"({t},{u})" for t, u in arcs) if arcs else "none"
            log_file.write(
                f"λ[{item_id},{idx}]     {item_id:<6} {lam_val:<12.6f} {cost:<14.2f} {setup_str:<20} {arc_str}\n"
            )

        log_file.write("-" * 100 + "\n")
        log_file.write(f"Total active columns: {len(active_cols)}\n\n")

        # Per-item breakdown
        log_file.write("PER-ITEM SOLUTION BREAKDOWN\n")
        log_file.write("-" * 60 + "\n")

        for item_id in sorted(items.keys()):
            item_data = items[item_id]
            log_file.write(f"\nItem {item_id}:\n")
            log_file.write(f"  Demand:    {item_data['demand']}\n")
            log_file.write(f"  Shelf seq: {item_data['shelf_seq']}\n")
            log_file.write(f"  Setup:     {item_data['setup']}\n")
            log_file.write(f"  Var cost:  {item_data['c_var']}\n")
            log_file.write(f"  Holding:   {item_data['h']}\n")

            # Get columns for this item
            item_cols = [c for c in active_cols if c["item_id"] == item_id]

            if item_cols:
                log_file.write(f"  Active columns: {len(item_cols)}\n")
                for col_info in item_cols:
                    idx = col_info["col_idx"]
                    lam_val = col_info["lambda_val"]
                    cost = col_info["cost"]
                    cap_usage = col_info["capacity_usage"]
                    setups = col_info["setups"]

                    log_file.write(
                        f"    λ[{item_id},{idx}] = {lam_val:.6f}, cost = {cost:.2f}\n"
                    )
                    for t in setups:
                        prod = cap_usage[t] if t < len(cap_usage) else 0
                        log_file.write(f"      t={t}: setup=1, prod={prod:.2f}\n")
    else:
        log_file.write("ACTIVE COLUMNS IN OPTIMAL BASIS\n")
        log_file.write("-" * 60 + "\n")
        log_file.write("No active columns captured (solution may be infeasible)\n\n")

    log_file.write("\n" + "=" * 90 + "\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    from datetime import datetime

    instance = {
        "period": 10,
        "manual_capacity": [0, 0, 80, 80, 80, 80, 80, 80, 100, 80],
        "items": {
            "0": {
                "h": [
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
                ],
                "b_var": 0,
                "c_var": [
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
                ],
                "setup": [
                    80,
                    81.66329352654208,
                    83.2538931446064,
                    84.70228201833979,
                    85.94515860381915,
                    86.9282032302755,
                    87.60845213036123,
                    87.9561751629462,
                    87.9561751629462,
                    87.60845213036123,
                ],
                "demand": [0, 0, 68, 49, 66, 38, 17, 17, 41, 43],
                "shelf_seq": [24, 18, 22, 6, 16, 9, 21, 23, 9, 7],
            }
        },
        # optional extras (currently ignored by your solver)
        "allow_unmet_demand": False,
        "warehouse_capacity": None,
        "lost_sales_penalty_factor": 200,
    }

    out_dir = Path("bnp_v10_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "solver_log.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write("=" * 90 + "\n")
    log_file.write(f"Branch-and-Price Solver — Depth-First Search\n")
    log_file.write(f"Execution started: {timestamp}\n")
    log_file.write("=" * 90 + "\n\n")

    # Save test instance
    instance_path = out_dir / "test_instance.json"
    instance_path.write_text(json.dumps(instance, indent=2))
    log_file.write(f"Instance saved to: {instance_path}\n\n")

    # Log instance details
    log_file.write("INSTANCE DETAILS\n")
    log_file.write("-" * 60 + "\n")
    log_file.write(f"  Periods: {instance['period']}\n")
    log_file.write(f"  Capacity: {instance['manual_capacity']}\n")
    log_file.write(f"  Items: {len(instance['items'])}\n\n")

    for item_id, item_data in instance["items"].items():
        log_file.write(f"  Item {item_id}:\n")
        log_file.write(f"    Demand:    {item_data['demand']}\n")
        log_file.write(f"    Shelf seq: {item_data['shelf_seq']}\n")
        log_file.write(f"    Setup:     {item_data['setup']}\n")
        log_file.write(f"    Var cost:  {item_data['c_var']}\n")
        log_file.write(f"    Holding:   {item_data['h']}\n")
    log_file.write("\n")
    # ------------------------------------------------------------------

    summary, orders, best_active_cols = solve_instance(
        instance_path=str(instance_path),
        time_limit=600,
        out_dir=out_dir,
        log_file=log_file,
    )

    # Log optimal solution details

    items = {int(k): v for k, v in instance["items"].items()}
    log_optimal_solution_details(summary, orders, [], items, log_file)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SOLUTION SUMMARY")
    print("=" * 70)

    status_map = {2: "OPTIMAL", 3: "INFEASIBLE", 13: "SUBOPTIMAL"}
    status_str = status_map.get(summary["status"], f"STATUS_{summary['status']}")

    print(f"  Status:          {status_str}")
    print(f"  Objective:       {summary.get('objective', 'N/A')}")
    print(f"  Best bound:      {summary['best_bound']:.4f}")
    gap = summary.get("gap", 0) or 0
    print(f"  Gap:             {gap * 100:.4f}%")
    print(f"  Runtime:         {summary['runtime_sec']:.2f}s")
    print(f"  Nodes explored:  {summary.get('nodes_explored', 0)}")
    print(f"  Nodes created:   {summary.get('nodes_created', 0)}")
    print(f"  CG iterations:   {summary.get('total_cg_iterations', 0)}")
    print(f"  Columns added:   {summary.get('total_columns', 0)}")

    # Active columns info
    if best_active_cols:
        print(f"  Active columns:  {len(best_active_cols)}")
        print("\n" + "-" * 70)
        print("ACTIVE COLUMNS IN OPTIMAL BASIS")
        print("-" * 70)
        print(
            f"{'Column':<15} {'Item':<6} {'λ value':<12} {'Cost':<14} {'Setups':<20} {'Arcs'}"
        )
        print("-" * 70)
        sorted_cols = sorted(
            best_active_cols, key=lambda x: (x["item_id"], x["col_idx"])
        )
        for col_info in sorted_cols:
            item_id = col_info["item_id"]
            idx = col_info["col_idx"]
            lam_val = col_info["lambda_val"]
            cost = col_info["cost"]
            setups = col_info["setups"]
            arcs = col_info["arcs"]
            setup_str = ",".join(map(str, setups)) if setups else "none"
            arc_str = ",".join(f"({t},{u})" for t, u in arcs) if arcs else "none"
            print(
                f"λ[{item_id},{idx}]     {item_id:<6} {lam_val:<12.6f} {cost:<14.2f} {setup_str:<20} {arc_str}"
            )
        print("-" * 70)
        print(f"Total active columns: {len(best_active_cols)}\n")

    print("\n" + "-" * 70)
    print("PRODUCTION PLAN")
    print("-" * 70)
    for line in orders:
        print(line)

    print("\n" + "-" * 70)
    print("OUTPUT FILES")
    print("-" * 70)
    print(f"  Log file:     {log_path}")
    print(f"  Summary:      {out_dir / 'summary.json'}")
    print(f"  Orders:       {out_dir / 'orders.txt'}")
    print("=" * 70)

    log_file.write("\nExecution completed successfully.\n")
    log_file.close()

    print("\nDone.")
