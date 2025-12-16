"""
Column Generation for ZIO columns with dummy setups.

Instead of enumerating all columns, we use pricing:
1. Start with initial feasible columns
2. Solve RMP LP to get dual values
3. Use DP pricing to find columns with negative reduced cost
4. Add columns and repeat until convergence

RMP:
    min  Σ c_k λ_k
    s.t. Σ λ_k = 1                           (convexity) → dual: μ
         Σ x_k[t] λ_k ≤ capacity[t]          (capacity)  → dual: ρ_t
         λ_k ≥ 0

Pricing (reduced cost):
    rc_k = c_k - μ - Σ_t ρ_t * x_k[t]

Find ZIO column with minimum reduced cost using DP.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import gurobipy as gp
from gurobipy import GRB


@dataclass
class ZIOColumn:
    """A ZIO column representing a complete production plan."""

    y: Dict[int, int]  # Y[t] = 1 if setup at period t
    x: Dict[int, float]  # X[t] = production quantity at period t
    z: Dict[Tuple[int, int], int]  # Z[t,u] = 1 if production at t serves demand at u
    cost: float  # Total cost (setup + variable + holding)

    def __repr__(self):
        y_periods = sorted([t for t, v in self.y.items() if v == 1])
        x_vals = {t: self.x.get(t, 0) for t in y_periods if self.x.get(t, 0) > 0}
        dummy = [t for t in y_periods if self.x.get(t, 0) == 0]
        return (
            f"ZIOColumn(cost={self.cost:.2f}, Y={y_periods}, X={x_vals}, dummy={dummy})"
        )


def generate_initial_column(
    demand: List[float],
    setup: List[float],
    h: List[float],
    c_var: List[float],
) -> ZIOColumn:
    """Generate a lot-for-lot initial column (produce exactly for each demand period)."""
    T = len(demand)
    y = {}
    x = {}
    z = {}
    cost = 0.0

    for t in range(T):
        if demand[t] > 1e-9:
            y[t] = 1
            x[t] = demand[t]
            z[(t, t)] = 1
            cost += setup[t] + c_var[t] * demand[t]

    return ZIOColumn(y=y, x=x, z=z, cost=cost)


def generate_initial_column_with_constraints(
    demand: List[float],
    setup: List[float],
    h: List[float],
    c_var: List[float],
    forbidden_y: Set[int],
) -> ZIOColumn:
    """
    Generate initial ZIO column respecting forbidden production periods.

    For each demand, find the latest allowed production period.
    """
    T = len(demand)
    demand_periods = [t for t in range(T) if demand[t] > 1e-9]

    y = {}
    x = {}
    z = {}
    cost = 0.0

    i = 0
    while i < len(demand_periods):
        u = demand_periods[i]

        # Find the latest production period <= u that is allowed
        prod_t = None
        for t in range(u, -1, -1):
            if t not in forbidden_y:
                prod_t = t
                break

        if prod_t is None:
            raise ValueError(
                f"Cannot produce for demand at period {u} - all earlier periods forbidden"
            )

        # Produce at prod_t to cover demands starting at i
        # Cover as many consecutive demands as possible from this production
        y[prod_t] = 1
        total_prod = 0.0
        covered_demands = []

        for j in range(i, len(demand_periods)):
            uj = demand_periods[j]
            # Check if this demand can be served from prod_t
            if uj >= prod_t:  # Basic reachability
                total_prod += demand[uj]
                covered_demands.append(uj)
                z[(prod_t, uj)] = 1
            else:
                break

            # Stop if next demand could have its own production period
            if j + 1 < len(demand_periods):
                next_u = demand_periods[j + 1]
                # Check if there's an allowed production period for next demand
                has_allowed = any(
                    t not in forbidden_y for t in range(next_u + 1) if t > prod_t
                )
                if has_allowed:
                    break

        x[prod_t] = total_prod

        # Compute cost
        s_cost = setup[prod_t]
        v_cost = sum(c_var[prod_t] * demand[uj] for uj in covered_demands)
        h_cost = sum(h[prod_t] * (uj - prod_t) * demand[uj] for uj in covered_demands)
        cost += s_cost + v_cost + h_cost

        i += len(covered_demands)

    return ZIOColumn(y=y, x=x, z=z, cost=cost)


def generate_initial_column_with_required_y(
    demand: List[float],
    setup: List[float],
    h: List[float],
    c_var: List[float],
    forbidden_y: Set[int],
    required_y: Set[int],
) -> ZIOColumn:
    """
    Generate initial ZIO column with required Y pattern (professor's approach).

    All periods in required_y must have Y=1 (either production or dummy).
    """
    # First generate a basic column
    col = generate_initial_column_with_constraints(demand, setup, h, c_var, forbidden_y)

    # Add dummy setups for required_y periods not covered
    y = dict(col.y)
    x = dict(col.x)
    cost = col.cost

    for t in required_y:
        if t not in y and t not in forbidden_y:
            y[t] = 1
            x[t] = 0
            cost += setup[t]  # Dummy setup cost

    return ZIOColumn(y=y, x=x, z=col.z, cost=cost)


def price_zio_column_dp(
    demand: List[float],
    setup: List[float],
    h: List[float],
    c_var: List[float],
    mu: float,  # Dual of convexity constraint
    rho: Dict[int, float],  # Dual of capacity constraints
    shelf_life: int = None,
    forced_y: Set[int] = None,
    forbidden_y: Set[int] = None,
) -> Tuple[Optional[ZIOColumn], float]:
    """
    Find ZIO column with minimum reduced cost using DP.

    The reduced cost of a column is:
        rc = c_k - μ - Σ_t ρ_t * x_k[t]

    For a ZIO column, the cost c_k = Σ_blocks (setup + var + holding costs)
    And x_k[t] = production at period t

    So we need to find the ZIO plan that minimizes:
        Σ_blocks [setup[t] + Σ_{u in block} (c_var[t] - ρ_t) * d[u] + h[t] * (u-t) * d[u]] - μ

    This is a shortest path problem that can be solved with DP.
    """
    T = len(demand)
    if shelf_life is None:
        shelf_life = T

    forced_y = forced_y or set()
    forbidden_y = forbidden_y or set()

    # Find demand periods
    demand_periods = [u for u in range(T) if demand[u] > 1e-9]

    if not demand_periods:
        # No demand - trivial column
        rc = -mu
        if rc < -1e-6:
            return ZIOColumn(y={}, x={}, z={}, cost=0), rc
        return None, 0

    n_demands = len(demand_periods)

    # Build reachability: Gamma[t] = demand periods reachable from production at t
    Gamma = {}
    for t in range(T):
        reachable = []
        for u in demand_periods:
            if t <= u <= min(t + shelf_life - 1, T - 1):
                reachable.append(u)
        Gamma[t] = reachable

    def block_reduced_cost(t: int, demands_covered: List[int]) -> Tuple[float, float]:
        """
        Compute reduced cost contribution and actual cost for a production block.

        Block produces at t to serve demands_covered.
        Reduced cost = setup[t] + Σ_u [(c_var[t] - ρ_t) * d[u] + h[t] * (u-t) * d[u]]
        Actual cost = setup[t] + Σ_u [c_var[t] * d[u] + h[t] * (u-t) * d[u]]
        """
        if t in forbidden_y:
            return float("inf"), float("inf")

        # Setup cost
        s_cost = setup[t]

        # Variable and holding costs (with dual adjustment for reduced cost)
        total_prod = sum(demand[u] for u in demands_covered)
        rho_t = rho.get(t, 0.0)

        v_cost = sum(c_var[t] * demand[u] for u in demands_covered)
        h_cost = sum(h[t] * (u - t) * demand[u] for u in demands_covered)

        actual_cost = s_cost + v_cost + h_cost
        reduced_cost = s_cost + (v_cost - rho_t * total_prod) + h_cost

        return reduced_cost, actual_cost

    # DP to find minimum reduced cost ZIO plan
    # F[i] = (min_reduced_cost, actual_cost, path) to cover demands 0..i-1
    INF = float("inf")
    F = [(INF, 0, [])] * (n_demands + 1)
    F[0] = (0, 0, [])

    for i in range(n_demands):
        if F[i][0] >= INF:
            continue

        u_start = demand_periods[i]

        # Try each production period that can reach u_start
        for t in range(u_start + 1):
            if t in forbidden_y:
                continue

            reachable = Gamma.get(t, [])
            if u_start not in reachable:
                continue

            # Try covering consecutive demands starting from i
            for j in range(i, n_demands):
                u_end = demand_periods[j]

                if u_end not in reachable:
                    break

                # This block covers demands[i:j+1]
                demands_covered = demand_periods[i : j + 1]
                rc_block, actual_block = block_reduced_cost(t, demands_covered)

                new_rc = F[i][0] + rc_block
                new_actual = F[i][1] + actual_block

                if new_rc < F[j + 1][0]:
                    new_path = F[i][2] + [(t, demands_covered)]
                    F[j + 1] = (new_rc, new_actual, new_path)

    # Check if we covered all demands
    if F[n_demands][0] >= INF:
        return None, 0

    # Apply forced_y dummy setup costs
    base_rc = F[n_demands][0]
    base_actual = F[n_demands][1]
    path = F[n_demands][2]

    production_periods = set(t for t, _ in path)

    for t in forced_y:
        if t not in production_periods and t not in forbidden_y:
            base_rc += setup[t]  # Dummy setup still costs setup
            base_actual += setup[t]

    # Final reduced cost includes -μ
    final_rc = base_rc - mu

    if final_rc >= -1e-6:
        return None, final_rc

    # Build the column
    y = {}
    x = {}
    z = {}

    for t, demands_covered in path:
        y[t] = 1
        prod = sum(demand[u] for u in demands_covered)
        x[t] = prod
        for u in demands_covered:
            z[(t, u)] = 1

    # Add forced dummy setups
    for t in forced_y:
        if t not in y and t not in forbidden_y:
            y[t] = 1
            x[t] = 0

    return ZIOColumn(y=y, x=x, z=z, cost=base_actual), final_rc


def enumerate_negative_rc_columns(
    demand: List[float],
    setup: List[float],
    h: List[float],
    c_var: List[float],
    mu: float,
    rho: Dict[int, float],
    shelf_life: int = None,
    max_columns: int = 10,
    forced_y: Set[int] = None,
    forbidden_y: Set[int] = None,
    required_y: Set[
        int
    ] = None,  # Periods that MUST have Y=1 (for professor's approach)
) -> List[Tuple[ZIOColumn, float]]:
    """
    Enumerate multiple ZIO columns with negative reduced cost.

    Uses DFS to find diverse columns, not just the single best one.
    This helps column generation converge faster.

    If required_y is specified, ALL columns must have Y=1 at those periods.
    This enables the professor's approach where all columns share a common Y pattern.
    """
    T = len(demand)
    if shelf_life is None:
        shelf_life = T

    forced_y = forced_y or set()
    forbidden_y = forbidden_y or set()
    required_y = required_y or set()  # Periods that MUST have Y=1

    demand_periods = [u for u in range(T) if demand[u] > 1e-9]

    if not demand_periods:
        return []

    n_demands = len(demand_periods)

    # Build reachability
    Gamma = {}
    for t in range(T):
        reachable = []
        for u in demand_periods:
            if t <= u <= min(t + shelf_life - 1, T - 1):
                reachable.append(u)
        Gamma[t] = reachable

    def block_costs(t: int, demands_covered: List[int]) -> Tuple[float, float]:
        """Return (reduced_cost, actual_cost) for block."""
        if t in forbidden_y:
            return float("inf"), float("inf")

        s_cost = setup[t]
        total_prod = sum(demand[u] for u in demands_covered)
        rho_t = rho.get(t, 0.0)

        v_cost = sum(c_var[t] * demand[u] for u in demands_covered)
        h_cost = sum(h[t] * (u - t) * demand[u] for u in demands_covered)

        actual = s_cost + v_cost + h_cost
        reduced = s_cost + (v_cost - rho_t * total_prod) + h_cost

        return reduced, actual

    columns_found = []
    seen_signatures = set()

    def dfs(
        demand_idx: int,
        current_path: List[Tuple[int, List[int]]],
        current_rc: float,
        current_actual: float,
    ):
        """DFS to enumerate columns."""
        if len(columns_found) >= max_columns:
            return

        if demand_idx >= n_demands:
            # Complete column
            base_rc = current_rc
            base_actual = current_actual

            production_periods = set(t for t, _ in current_path)

            # Add forced_y as dummy setups
            for t in forced_y:
                if t not in production_periods and t not in forbidden_y:
                    base_rc += setup[t]
                    base_actual += setup[t]

            # Add required_y as dummy setups (professor's approach)
            for t in required_y:
                if t not in production_periods and t not in forbidden_y:
                    base_rc += setup[t]
                    base_actual += setup[t]

            final_rc = base_rc - mu

            if final_rc < -1e-6:
                # Build column
                y = {}
                x = {}
                z = {}

                for t, demands_covered in current_path:
                    y[t] = 1
                    prod = sum(demand[u] for u in demands_covered)
                    x[t] = prod
                    for u in demands_covered:
                        z[(t, u)] = 1

                # Add forced_y as dummy
                for t in forced_y:
                    if t not in y and t not in forbidden_y:
                        y[t] = 1
                        x[t] = 0

                # Add required_y as dummy (professor's approach)
                for t in required_y:
                    if t not in y and t not in forbidden_y:
                        y[t] = 1
                        x[t] = 0

                sig = tuple(sorted(x.items()))
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    col = ZIOColumn(y=y, x=x, z=z, cost=base_actual)
                    columns_found.append((col, final_rc))
            return

        u_start = demand_periods[demand_idx]

        # Try production periods, prioritizing later ones (more diverse columns)
        for t in range(u_start, -1, -1):
            if len(columns_found) >= max_columns:
                return
            if t in forbidden_y:
                continue

            reachable = Gamma.get(t, [])
            if u_start not in reachable:
                continue

            # Try different block sizes
            for j in range(demand_idx, n_demands):
                if len(columns_found) >= max_columns:
                    return

                u_end = demand_periods[j]
                if u_end not in reachable:
                    break

                demands_covered = demand_periods[demand_idx : j + 1]
                rc_block, actual_block = block_costs(t, demands_covered)

                if rc_block < float("inf"):
                    current_path.append((t, demands_covered))
                    dfs(
                        j + 1,
                        current_path,
                        current_rc + rc_block,
                        current_actual + actual_block,
                    )
                    current_path.pop()

    dfs(0, [], 0.0, 0.0)

    # Sort by reduced cost
    columns_found.sort(key=lambda x: x[1])

    return columns_found


def solve_column_generation(
    demand: List[float],
    setup: List[float],
    h: List[float],
    c_var: List[float],
    capacity: List[float],
    shelf_life: int = None,
    max_iterations: int = 100,
    verbose: bool = True,
    required_y: Set[
        int
    ] = None,  # Professor's approach: all columns must have Y=1 at these periods
) -> Tuple[float, List[ZIOColumn], Dict[int, float]]:
    """
    Solve the capacitated lot-sizing problem using column generation with ZIO columns.

    Returns:
        - Optimal objective value
        - List of active columns with their lambda values
        - Aggregate X values by period
    """
    T = len(demand)
    if shelf_life is None:
        shelf_life = T

    # Periods with zero capacity are forbidden for production
    forbidden_y = {t for t in range(T) if capacity[t] <= 0}
    required_y = required_y or set()

    if verbose:
        print("=" * 70)
        print("COLUMN GENERATION FOR ZIO COLUMNS")
        print("=" * 70)
        print(f"Demand: {demand}")
        print(f"Total demand: {sum(demand)}")
        print(f"Capacity: {capacity}")
        print(f"Forbidden production periods (capacity=0): {sorted(forbidden_y)}")
        if required_y:
            print(f"Required Y periods (professor's approach): {sorted(required_y)}")
        print()

    # Initialize with lot-for-lot column (respecting forbidden periods and required_y)
    columns = [
        generate_initial_column_with_required_y(
            demand, setup, h, c_var, forbidden_y, required_y
        )
    ]

    if verbose:
        print(f"Initial column: {columns[0]}")
        print()

    for iteration in range(max_iterations):
        # Build and solve RMP
        m = gp.Model("RMP")
        m.Params.OutputFlag = 0

        n_cols = len(columns)
        lam = [m.addVar(lb=0, name=f"lam_{k}") for k in range(n_cols)]

        # Convexity constraint
        convex_con = m.addConstr(gp.quicksum(lam) == 1, "convex")

        # Capacity constraints
        cap_cons = {}
        for t in range(T):
            if capacity[t] > 0:
                x_sum = gp.quicksum(
                    columns[k].x.get(t, 0) * lam[k] for k in range(n_cols)
                )
                cap_cons[t] = m.addConstr(x_sum <= capacity[t], f"cap_{t}")

        # Objective
        m.setObjective(
            gp.quicksum(columns[k].cost * lam[k] for k in range(n_cols)), GRB.MINIMIZE
        )

        m.optimize()

        if m.Status != GRB.OPTIMAL:
            if verbose:
                print(f"Iteration {iteration}: RMP not optimal (status={m.Status})")
            break

        obj = m.ObjVal

        # Get duals
        mu = convex_con.Pi
        rho = {t: cap_cons[t].Pi for t in cap_cons}

        # Pricing: find column with negative reduced cost
        new_cols = enumerate_negative_rc_columns(
            demand=demand,
            setup=setup,
            h=h,
            c_var=c_var,
            mu=mu,
            rho=rho,
            shelf_life=shelf_life,
            max_columns=1,  # Add up to 5 columns per iteration
            forbidden_y=forbidden_y,
            required_y=required_y,  # Professor's approach
        )

        min_rc = new_cols[0][1] if new_cols else 0

        if verbose:
            active_cols = sum(1 for k in range(n_cols) if lam[k].X > 1e-6)
            print(
                f"Iter {iteration:3d}: obj={obj:10.2f}, cols={n_cols:4d}, "
                f"active={active_cols:3d}, min_rc={min_rc:10.4f}"
            )

        if min_rc >= -1e-6:
            if verbose:
                print("\nConverged! No negative reduced cost columns found.")
            break

        # Add new columns
        for col, rc in new_cols:
            # Check if column already exists
            is_new = True
            for existing in columns:
                if existing.x == col.x:
                    is_new = False
                    break
            if is_new:
                columns.append(col)
                if verbose and len(new_cols) <= 3:
                    print(f"  Added: {col}, rc={rc:.4f}")

    # Final solve to get solution
    m = gp.Model("RMP_Final")
    m.Params.OutputFlag = 0

    n_cols = len(columns)
    lam = [m.addVar(lb=0, name=f"lam_{k}") for k in range(n_cols)]

    m.addConstr(gp.quicksum(lam) == 1, "convex")

    for t in range(T):
        if capacity[t] > 0:
            x_sum = gp.quicksum(columns[k].x.get(t, 0) * lam[k] for k in range(n_cols))
            m.addConstr(x_sum <= capacity[t], f"cap_{t}")

    m.setObjective(
        gp.quicksum(columns[k].cost * lam[k] for k in range(n_cols)), GRB.MINIMIZE
    )

    m.optimize()

    # Extract solution
    active_columns = []
    for k in range(n_cols):
        if lam[k].X > 1e-6:
            active_columns.append((columns[k], lam[k].X))

    # Compute aggregate X
    X_agg = {}
    for t in range(T):
        X_agg[t] = sum(columns[k].x.get(t, 0) * lam[k].X for k in range(n_cols))

    if verbose:
        print()
        print("=" * 70)
        print("SOLUTION")
        print("=" * 70)
        print(f"Optimal value: {m.ObjVal:.4f}")
        print(f"Target: 1105.0000")
        print()
        print("Active columns:")
        for col, lam_val in active_columns:
            print(f"  λ={lam_val:.6f}: {col}")
        print()
        print("Aggregate X vs Capacity:")
        for t in range(T):
            if capacity[t] > 0:
                x_t = X_agg[t]
                print(f"  X[{t}] = {x_t:6.2f} / {capacity[t]}")

    return m.ObjVal, active_columns, X_agg


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test instance
    demand = [0, 0, 68, 49, 66, 38, 17, 17, 41, 43]
    setup = [80, 81.66, 83.25, 84.70, 85.95, 86.93, 87.61, 87.96, 87.96, 87.61]
    h = [0.4, 0.408, 0.416, 0.424, 0.430, 0.435, 0.438, 0.440, 0.440, 0.438]
    c_var = [1.76, 1.71, 2.50, 1.78, 1.67, 2.11, 1.37, 1.24, 2.68, 1.81]
    capacity = [0, 0, 80, 80, 80, 80, 80, 80, 100, 80]

    print("\n" + "=" * 70)
    print("TEST 1: UNRESTRICTED COLUMN GENERATION (TRUE OPTIMAL)")
    print("=" * 70 + "\n")

    obj1, active_cols1, X_agg1 = solve_column_generation(
        demand=demand,
        setup=setup,
        h=h,
        c_var=c_var,
        capacity=capacity,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("TEST 2: PROFESSOR'S APPROACH (Required Y at periods 2,3,4,6,7)")
    print("=" * 70 + "\n")

    # Professor's columns all have Y=1 at periods 2,3,4,6,7
    required_y = {2, 3, 4, 6, 7}

    obj2, active_cols2, X_agg2 = solve_column_generation(
        demand=demand,
        setup=setup,
        h=h,
        c_var=c_var,
        capacity=capacity,
        verbose=True,
        required_y=required_y,
    )

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Unrestricted optimal:     {obj1:.4f}")
    print(f"Professor's approach:     {obj2:.4f}")
    print(f"Target (from professor):  1105.0000")
