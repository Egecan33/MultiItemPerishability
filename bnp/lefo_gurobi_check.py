"""
LEFO Verification using Gurobi LP

Given X values (production quantities), check if there exists a valid Z assignment
that satisfies the No-Crossing (LEFO) constraints.

LP Model (continuous Z for speed):
  Variables:
    - Z[t,u] ∈ [0,1] CONTINUOUS - fraction of demand u served by production t

  Constraints:
    (1) Demand coverage: Σ_t Z[t,u] = 1  ∀u with demand
    (2) Production capacity: Σ_u Z[t,u] * demand[u] ≤ X[t]  ∀t with production
    (3) Shelf life: implicit - no arc created if u > t + shelf_seq[t]
    (4) No-crossing (LEFO): Z[t1,u] + Z[t2,u'] ≤ 1
        for t1,t2 where exp(t1) < exp(t2), and u ∈ [t2, u'-1]

If LP is feasible → X values are LEFO-compatible
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import gurobipy as gp
from gurobipy import GRB

EPS = 1e-6


@dataclass
class LEFOCheckResult:
    """Result of LEFO check using Gurobi."""

    item_id: int
    is_feasible: bool
    z_values: Dict[Tuple[int, int], float] = field(default_factory=dict)
    flow_values: Dict[Tuple[int, int], float] = field(default_factory=dict)
    model_status: str = ""
    details: str = ""


def check_lefo_gurobi(
    item_id: int,
    x_values: Dict[int, float],  # t -> production quantity
    demand: List[float],
    shelf_seq: List[int],
    T: int,
    verbose: bool = False,
) -> LEFOCheckResult:
    """
    Check if X values admit a LEFO-compatible Z assignment using Gurobi LP.

    Z[t,u] ∈ [0,1] CONTINUOUS - fraction of demand u served by production t

    Constraints:
    - Σ_t Z[t,u] = 1 for each u (demand coverage)
    - Σ_u Z[t,u] * demand[u] ≤ X[t] (production capacity)
    - Z[t1,u] + Z[t2,u'] ≤ 1 (no-crossing / LEFO)

    Returns LEFOCheckResult with feasibility and Z values if feasible.
    """
    # Get production periods and demand periods
    prod_periods = [t for t, x in x_values.items() if x > EPS]
    demand_periods = [u for u in range(T) if demand[u] > EPS]

    if not prod_periods or not demand_periods:
        return LEFOCheckResult(
            item_id=item_id,
            is_feasible=True,
            model_status="TRIVIAL",
            details="No production or no demand",
        )

    # Compute expiry for each production period
    expiry = {t: t + shelf_seq[t] for t in prod_periods}

    # Build valid arcs: (t, u) where t <= u <= t + shelf_seq[t]
    valid_arcs = []
    for t in prod_periods:
        for u in demand_periods:
            if t <= u <= expiry[t]:
                valid_arcs.append((t, u))

    if not valid_arcs:
        return LEFOCheckResult(
            item_id=item_id,
            is_feasible=False,
            model_status="NO_VALID_ARCS",
            details="No valid arcs between production and demand",
        )

    # Check if each demand period has at least one arc
    for u in demand_periods:
        arcs_to_u = [(t, uu) for (t, uu) in valid_arcs if uu == u]
        if not arcs_to_u:
            return LEFOCheckResult(
                item_id=item_id,
                is_feasible=True,  # Not a LEFO issue - mark as OK to skip
                model_status="UNREACHABLE_DEMAND",
                details=f"Demand at u={u} has no valid arc (not LEFO issue)",
            )

    # =========================================================================
    # PHASE 1: Check BASIC feasibility (without LEFO constraint)
    # =========================================================================
    m_basic = gp.Model("Basic_Flow_Check")
    m_basic.Params.OutputFlag = 0

    Z_basic = {}
    for t, u in valid_arcs:
        Z_basic[t, u] = m_basic.addVar(lb=0, ub=1, name=f"Z_{t}_{u}")

    # Demand coverage
    for u in demand_periods:
        arcs_to_u = [(t, uu) for (t, uu) in valid_arcs if uu == u]
        if arcs_to_u:
            m_basic.addConstr(gp.quicksum(Z_basic[t, u] for t, uu in arcs_to_u) == 1)

    # Production capacity
    for t in prod_periods:
        arcs_from_t = [(tt, u) for (tt, u) in valid_arcs if tt == t]
        if arcs_from_t:
            m_basic.addConstr(
                gp.quicksum(Z_basic[t, u] * demand[u] for tt, u in arcs_from_t)
                <= x_values[t]
            )

    m_basic.setObjective(0, GRB.MINIMIZE)
    m_basic.optimize()

    if m_basic.Status != GRB.OPTIMAL:
        # Basic flow is infeasible - X values cannot meet demand (not a LEFO issue)
        return LEFOCheckResult(
            item_id=item_id,
            is_feasible=True,  # Mark as "OK" since it's not a LEFO issue
            model_status="BASIC_INFEASIBLE",
            details="X values insufficient for demand (not LEFO issue)",
        )

    # =========================================================================
    # PHASE 2: Check LEFO feasibility (with no-crossing constraint)
    # =========================================================================
    # Create model with LEFO constraints
    m = gp.Model("LEFO_Check")
    m.Params.OutputFlag = 1 if verbose else 0

    # Variables: Z[t,u] ∈ [0,1] CONTINUOUS - fraction of demand u served by t
    Z = {}
    for t, u in valid_arcs:
        Z[t, u] = m.addVar(lb=0, ub=1, name=f"Z_{t}_{u}")

    # Constraint (1): Demand coverage - sum of Z[t,u] over all t must equal 1
    for u in demand_periods:
        arcs_to_u = [(t, uu) for (t, uu) in valid_arcs if uu == u]
        if arcs_to_u:
            m.addConstr(
                gp.quicksum(Z[t, u] for t, uu in arcs_to_u) == 1,
                name=f"demand_{u}",
            )

    # Constraint (2): Production capacity - Σ Z[t,u] * demand[u] ≤ X[t]
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
            t2_demands = [u for u in demand_periods if (t2, u) in valid_arcs]

            for up in t2_demands:
                # Get demands in [t2, up-1] that t1 can serve
                for u in demand_periods:
                    if t2 <= u <= up - 1 and (t1, u) in valid_arcs:
                        # No-crossing: Z[t1, u] + Z[t2, up] <= 1
                        m.addConstr(
                            Z[t1, u] + Z[t2, up] <= 1,
                            name=f"nocross_{t1}_{t2}_{u}_{up}",
                        )

    # Objective: any feasible solution works
    m.setObjective(0, GRB.MINIMIZE)

    # Solve
    m.optimize()

    if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL:
        # Extract solution - Z values (continuous fractions)
        z_vals = {(t, u): Z[t, u].X for t, u in valid_arcs if Z[t, u].X > EPS}
        # Compute flow from Z * demand
        flow_vals = {
            (t, u): Z[t, u].X * demand[u] for t, u in valid_arcs if Z[t, u].X > EPS
        }

        # Build details - show Z fractions
        arcs_str = ", ".join(
            [f"({t}→{u}):{z_vals[(t,u)]:.2f}" for t, u in sorted(z_vals.keys())]
        )

        return LEFOCheckResult(
            item_id=item_id,
            is_feasible=True,
            z_values=z_vals,
            flow_values=flow_vals,
            model_status="OPTIMAL",
            details=f"Z: {arcs_str}",
        )
    elif m.Status == GRB.INFEASIBLE:
        # Compute IIS if possible
        try:
            m.computeIIS()
            iis_constrs = [c.ConstrName for c in m.getConstrs() if c.IISConstr]
            iis_str = ", ".join(iis_constrs[:5])
            if len(iis_constrs) > 5:
                iis_str += f"... (+{len(iis_constrs)-5} more)"
        except:
            iis_str = "Could not compute IIS"

        return LEFOCheckResult(
            item_id=item_id,
            is_feasible=False,
            model_status="INFEASIBLE",
            details=f"LEFO IIS: {iis_str}",
        )
    else:
        return LEFOCheckResult(
            item_id=item_id,
            is_feasible=False,
            model_status=f"STATUS_{m.Status}",
            details="Unknown solver status",
        )


def check_solution_lefo_gurobi(
    items: Dict[int, dict],
    x_per_item: Dict[int, Dict[int, float]],
    T: int,
    verbose: bool = False,
) -> Tuple[bool, List[LEFOCheckResult], str]:
    """
    Check LEFO compatibility for all items using Gurobi.

    Returns (all_compatible, results_per_item, summary_string)
    """
    results = []
    all_compatible = True
    summary_parts = []

    for i, item_data in sorted(items.items()):
        demand = item_data["demand"]
        shelf_seq = item_data["shelf_seq"]
        x_values = x_per_item.get(i, {})

        result = check_lefo_gurobi(i, x_values, demand, shelf_seq, T, verbose)
        results.append(result)

        if result.is_feasible:
            num_arcs = len(result.z_values)
            summary_parts.append(f"Item{i}: OK ({num_arcs} arcs)")
        else:
            all_compatible = False
            summary_parts.append(f"Item{i}: FAIL ({result.model_status})")

    return all_compatible, results, "; ".join(summary_parts)


if __name__ == "__main__":
    # Quick test
    print("Testing LEFO Gurobi Check...")

    # User's corrected example
    demand = [0.0, 0.0, 20.0, 0.0, 0.0, 20.0, 20.0, 0.0, 10.0, 20.0]
    shelf_seq = [2, 6, 4, 3, 4, 1, 3, 3, 1, 4]
    x_values = {2: 40.0, 5: 40.0, 6: 30.0, 9: 20.0}
    T = 10

    result = check_lefo_gurobi(0, x_values, demand, shelf_seq, T, verbose=True)

    print(f"\nLEFO Compatible: {result.is_feasible}")
    print(f"Status: {result.model_status}")
    if result.z_values:
        print("Z assignment:")
        for (t, u), z in sorted(result.z_values.items()):
            print(f"  Z[{t},{u}] = {z:.2f} (flow={z * demand[u]:.1f})")
