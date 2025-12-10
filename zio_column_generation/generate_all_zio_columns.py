#!/usr/bin/env python3
"""Generate all possible ZIO columns for an item using 11-node shortest path model.

11-Node Shortest Path Interpretation:
- Nodes: 0 (start), 1-10 (representing periods 0-9)
- Arc (s, u): produce at period s (node s), consume at period u-1 (node u)
- Block (s, t): produce at period s, consume at periods s through t-1
- Lambdas are over FULL COLUMNS (complete ZIO plans), not arcs
- Convex combination of full plans to satisfy all demands
- Include all periods 0 to T-1 (including zero demands)
- Allow Y=1 even when X=0 (setup-only periods)
"""

import json
import sys
import numpy as np
from scipy.optimize import linprog
from pathlib import Path
from typing import List, Dict, Tuple, Set
import itertools

# Load instance from solver_bnp_dp.py test data
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
}

data = instance
T = int(data["period"])
items_raw = {int(k): v for k, v in data["items"].items()}

item_id = 0
item_data = items_raw[item_id]
demand = list(item_data["demand"])
setup = item_data["setup"]
c_var = item_data["c_var"]
h = item_data["h"]
shelf_seq = item_data["shelf_seq"]


def s_at(t):
    return setup[t] if isinstance(setup, list) else setup


def c_at(t):
    return c_var[t] if isinstance(c_var, list) else c_var


def h_at(t):
    return h[t] if isinstance(h, list) else h


# Calculate holding cost prefix
h_prefix = [0.0]
for k in range(T):
    h_prefix.append(h_prefix[-1] + h_at(k))


def H_i(s, u):
    if u <= s:
        return 0.0
    return h_prefix[u] - h_prefix[s]


# Calculate expiry and Gamma
Gamma = {}
Expiry = {}
for t in range(T):
    m_it = int(shelf_seq[t])
    v_it = t + m_it
    Expiry[t] = v_it
    if m_it <= 0:
        Gamma[t] = []
    else:
        u_max = min(T - 1, v_it)
        Gamma[t] = list(range(t, u_max + 1))

print(f"=== GENERATING ALL ZIO COLUMNS FOR ITEM {item_id} ===")
print(f"Demand: {demand}")
print(f"Gamma: {Gamma}")
print(f"Expiry: {Expiry}")
print()


def generate_all_zio_columns() -> List[Dict]:
    """Generate all possible ZIO columns using 11-node shortest path interpretation.

    Nodes: 0 (start), 1-10 (periods 0-9)
    Arc (s, u): produce at period s, consume at period u-1
    Block (s, t): produce at period s, consume at periods s through t-1
    """
    columns = []

    num_nodes = T + 1  # 11 nodes for 10 periods (0-10)
    print(f"Modeling as {num_nodes}-node shortest path problem")
    print(f"Nodes: 0 (start), 1-10 (periods 0-9)")

    # All periods 0 to T-1
    all_periods = list(range(T))
    demand_periods = [t for t in range(T) if demand[t] > 0]

    print(f"All periods: {all_periods}")
    print(f"Periods with positive demand: {demand_periods}")

    # Generate all possible ways to partition periods into blocks using node-based interpretation
    # We need to find paths from node 0 to node 10 (or earlier) that cover all periods

    def is_consecutive(periods):
        """Check if periods form a consecutive sequence."""
        if not periods:
            return True
        sorted_periods = sorted(periods)
        return sorted_periods == list(range(sorted_periods[0], sorted_periods[-1] + 1))

    def generate_partitions(periods):
        """Generate all ways to partition periods into consecutive blocks.

        Block (s, t) where s and t are node indices:
        - s is the production node (period s)
        - t is the end node (consumption up to period t-1, which is node t)
        - Production at period s (node s)
        - Consumption at periods s through t-1 (nodes s+1 through t)
        - So block (s, t) covers periods s, s+1, ..., t-1
        - Arc (s, u) exists if u is a node (u <= 10) and u > s
        """
        if not periods:
            return [[]]

        partitions = []
        # Try all possible starting points for the first block
        for start in range(len(periods)):
            first_period = periods[start]
            # Try all possible ending points for the first block
            for end in range(start, len(periods)):
                block_periods = periods[start : end + 1]
                if not is_consecutive(block_periods):
                    continue

                # Block covers periods from first_period to last_period (inclusive)
                first_period_val = block_periods[0]
                last_period_val = block_periods[-1]

                # Production node s = first_period_val (produce at period first_period_val)
                # End node t = last_period_val + 1 (consume up to period last_period_val)
                s = first_period_val  # Production node (period s)
                t = (
                    last_period_val + 1
                )  # End node (consume up to period t-1 = last_period_val)

                # Check if end node is valid (t <= 10)
                if t > num_nodes - 1:  # t > 10
                    continue

                # Check if production period s can serve all periods in block
                # Production at period s can serve periods in Gamma[s]
                if s not in Gamma or not Gamma[s]:
                    continue

                # Check if all periods from s to last_period_val are in Gamma[s]
                can_serve = True
                for u in range(s, last_period_val + 1):
                    if u not in Gamma[s]:
                        can_serve = False
                        break
                if not can_serve:
                    continue

                # Recursively generate partitions for remaining periods
                remaining = periods[:start] + periods[end + 1 :]
                for sub_partition in generate_partitions(remaining):
                    partitions.append([(s, t)] + sub_partition)

        return partitions

    # Generate all block partitions
    print("Generating block partitions...")
    all_partitions = generate_partitions(all_periods)

    print(f"Found {len(all_partitions)} possible block partitions")
    print("Generating ALL partitions (no limit)")

    # For each partition, generate columns with different setup combinations
    # A column can have setups at production periods (required for production)
    # Also allow setup-only periods (Y=1, X=0) at any period that can produce (in Gamma)

    # Get all periods that can produce (have non-empty Gamma)
    producible_periods = [t for t in range(T) if Gamma.get(t)]
    print(f"Producible periods: {producible_periods}")
    print(f"Generating columns from partitions (no limit)...")

    col_count = 0
    for idx, partition in enumerate(all_partitions):
        if (idx + 1) % 1000 == 0:
            print(
                f"  Processing partition {idx+1}/{len(all_partitions)}, columns so far: {col_count}"
            )
        # Get all production periods from blocks
        # Block (s,t) means production at period s (node s)
        production_periods_from_blocks = set()
        for s, t in partition:
            prod_period = s  # Production at period s (node s)
            if 0 <= prod_period < T:
                production_periods_from_blocks.add(prod_period)

        # For setups, we need at least the production periods (for production to happen)
        # But we can also add setup-only periods
        # Try: minimal setups (only production periods) and extended setups (production + setup-only)

        # Minimal: setups only at production periods
        minimal_setups = production_periods_from_blocks.copy()

        # Extended: production periods + setup-only at other producible periods
        setup_only_candidates = set(producible_periods) - production_periods_from_blocks

        # Try all combinations: minimal + subsets of setup-only candidates
        setup_combinations = [minimal_setups]  # At least minimal

        # Add ALL combinations with setup-only periods (NO LIMIT - true enumeration)
        # This generates all possible ZIO columns
        for r in range(1, len(setup_only_candidates) + 1):
            for setup_only_subset in itertools.combinations(setup_only_candidates, r):
                setup_combinations.append(minimal_setups | set(setup_only_subset))

        for setups in setup_combinations:
            col_count += 1

            # Calculate production and arcs
            production = {}
            arcs = []
            total_cost = 0.0

            for s, t in partition:
                # Block (s,t) where s and t are node indices:
                # - Production at period s (node s)
                # - Consumption at periods s through t-1 (nodes s+1 through t)
                # - So block (s, t) covers periods s, s+1, ..., t-1
                prod_period = s  # Production at period s (node s)
                if prod_period >= T:
                    continue

                # Calculate quantity: sum of demands from period s to period t-1
                # t is the end node, so we consume up to period t-1
                qty = sum(
                    demand[u] for u in range(prod_period, t)
                )  # t is node, period is t-1, so range(s, t)

                # Production happens even if qty is 0 (for convex combination)
                production[prod_period] = production.get(prod_period, 0) + qty

                # Arcs: arc (s, u) where s and u are node indices
                # - Production at period s (node s)
                # - Consumption at period u-1 (node u = period u-1)
                # - So for block (s, t), we have arcs (s, u) for u from s+1 to t
                # - Arc (s, u) means produce at period s, consume at period u-1
                for u in range(s + 1, t + 1):  # u is node index, from s+1 to t
                    if u <= num_nodes - 1:  # u <= 10
                        arcs.append((s, u))

                # Cost (only if there's actual production/demand)
                if qty > 0:
                    block_cost = 0.0
                    if prod_period in setups:
                        block_cost += s_at(prod_period)
                    block_cost += c_at(prod_period) * qty
                    # Holding costs for periods s through t-1
                    for u in range(prod_period, t):  # u is period index
                        if demand[u] > 0:
                            block_cost += H_i(prod_period, u) * demand[u]
                    total_cost += block_cost
                elif prod_period in setups:
                    # Setup-only cost (Y=1, X=0)
                    total_cost += s_at(prod_period)

            # Add setup-only periods (Y=1, X=0)
            for t in setups:
                if t not in production:
                    total_cost += s_at(t)

            columns.append(
                {
                    "setups": sorted(setups),
                    "production": production,
                    "arcs": sorted(arcs),
                    "blocks": partition,
                    "cost": total_cost,
                }
            )

    print(f"Generated {len(columns)} columns before deduplication")

    # Remove duplicates
    print("Removing duplicates...")
    seen = set()
    unique_columns = []
    for col in columns:
        # Key: (tuple of setups, tuple of (production period, qty) pairs, tuple of arcs)
        key = (
            tuple(col["setups"]),
            tuple(sorted(col["production"].items())),
            tuple(col["arcs"]),
        )
        if key not in seen:
            seen.add(key)
            unique_columns.append(col)

    print(f"After deduplication: {len(unique_columns)} unique columns")
    return unique_columns


print("Generating all ZIO columns with new interpretation...")
import time

start_time = time.time()
zio_columns = generate_all_zio_columns()
elapsed = time.time() - start_time
print(f"Generated {len(zio_columns)} unique ZIO columns in {elapsed:.2f} seconds\n")

# Create output directory (same directory as script)
output_dir = Path(__file__).parent
output_dir.mkdir(exist_ok=True)

# Write columns to file
output_file = output_dir / "zio_columns_generated.txt"
print(f"Writing columns to {output_file}...")
with open(output_file, "w") as f:
    f.write(f"=== ALL ZIO COLUMNS FOR ITEM {item_id} ===\n")
    f.write(f"Total columns: {len(zio_columns)}\n")
    f.write(f"Demand: {demand}\n")
    f.write(f"Gamma: {Gamma}\n")
    f.write(f"Expiry: {Expiry}\n")
    f.write("=" * 80 + "\n\n")

    for i, col in enumerate(zio_columns):
        f.write(f"Column {i+1}:\n")
        f.write(f"  Cost: {col['cost']:.4f}\n")
        f.write(f"  Setups (Y): {col['setups']}\n")
        f.write(f"  Production (X): {dict(col['production'])}\n")
        f.write(f"  Arcs (Z): {col['arcs']}\n")
        f.write(f"  Blocks: {col['blocks']}\n")
        f.write("\n")

print(f"Written {len(zio_columns)} columns to {output_file}\n")

# Show sample columns
print("=" * 80)
print("SAMPLE COLUMNS (first 30):")
print("=" * 80)
for i, col in enumerate(zio_columns[:30]):
    print(f"\nColumn {i+1}:")
    print(f"  Cost: {col['cost']:.4f}")
    print(f"  Setups (Y): {col['setups']}")
    print(f"  Production (X): {dict(col['production'])}")
    print(f"  Arcs (Z): {col['arcs'][:10]}{'...' if len(col['arcs']) > 10 else ''}")
    print(f"  Blocks: {col['blocks']}")

if len(zio_columns) > 30:
    print(
        f"\n... and {len(zio_columns) - 30} more columns (see {output_file} for full list)"
    )

# Now test with RMP
print(f"\n=== TESTING WITH RMP ===")
import gurobipy as gp
from gurobipy import GRB

# Import RMP class from solver_bnp_dp
sys.path.insert(0, str(Path(__file__).parent.parent / "bnp"))
from solver_bnp_dp import RestrictedMasterProblem, ProductionPlanColumn

# Create single item instance
items = {item_id: item_data}
capacity = data["manual_capacity"]
Gamma_by_item = {item_id: Gamma}

# Create RMP
rmp = RestrictedMasterProblem(
    items=items,
    T=T,
    capacity=capacity,
    Gamma_by_item=Gamma_by_item,
)

# Add demand satisfaction constraints to RMP
# For each period u with demand > 0, we need: Σ_k Σ_{t: (t,u) in arcs} λ^k * demand[u] >= demand[u]
# Or equivalently: Σ_k (sum of arcs to u) * λ^k >= 1 for each u with demand > 0
print("Adding demand satisfaction constraints to RMP...")
demand_expr = {}  # demand_expr[u] = expression for period u
demand_con = {}  # demand_con[u] = constraint for period u

for u in range(T):
    if demand[u] > 0:
        demand_expr[u] = gp.LinExpr(0.0)

# Convert ZIO columns to ProductionPlanColumn format
print(f"Converting {len(zio_columns)} ZIO columns to ProductionPlanColumn format...")
for col_data in zio_columns:
    # Create capacity usage by period
    cap_usage = [0.0] * T
    for prod_period, qty in col_data["production"].items():
        if 0 <= prod_period < T:
            cap_usage[prod_period] = qty

    # Create setup by period
    setup_by_period = [0.0] * T
    for t in col_data["setups"]:
        if 0 <= t < T:
            setup_by_period[t] = 1.0

    # Create arc usage
    arc_usage = {}
    for s, u in col_data["arcs"]:
        arc_usage[(s, u)] = 1.0

    # Create column
    col = ProductionPlanColumn(
        item_id=item_id,
        total_plan_cost=col_data["cost"],
        capacity_usage_by_period=cap_usage,
        setup_by_period=setup_by_period,
        arc_usage=arc_usage,
    )

    rmp.add_column(col)

    # Track which demands this column serves for demand constraints
    # We'll add this after all columns are added

print(f"Added {len(zio_columns)} columns to RMP")

# Note: Demand satisfaction is typically handled in the pricing subproblem in column generation
# For now, we solve RMP without explicit demand constraints to see the solution
# The columns themselves are valid ZIO plans, but the RMP combination might not satisfy all demands
print("Note: Solving RMP without explicit demand constraints.")
print("Demand satisfaction should be verified in the solution analysis below.")

# Solve RMP
print("\nSolving RMP...")
lb, mu, pi, sigma, tau = rmp.solve()

print(f"\nRMP Status: {rmp.model.status}")
if rmp.model.status == GRB.OPTIMAL:
    print(f"RMP Lower Bound: {lb:.2f}")
    print(f"Convexity dual (mu): {mu}")
    print(f"Capacity duals (pi): {pi}")
elif rmp.model.status == GRB.INFEASIBLE:
    print("RMP is INFEASIBLE!")
    print("Computing IIS (Irreducible Inconsistent Subsystem)...")
    rmp.model.computeIIS()
    print("IIS Constraints:")
    for con in rmp.model.getConstrs():
        if con.IISConstr:
            print(f"  {con.ConstrName}: {con}")
    print(
        "\nThis means the generated columns cannot satisfy all demands with the given capacity constraints."
    )
    print("Exiting analysis.")
    sys.exit(1)
else:
    print(f"RMP solve failed with status: {rmp.model.status}")
    sys.exit(1)

# Extract solution
eps = 1e-6
active_columns = []
for (i, idx), lam_var in rmp.lambdas.items():
    try:
        lam_val = lam_var.X
        if lam_val > eps:
            col = rmp.columns[i][idx]
            active_columns.append((i, idx, lam_val, col))
    except:
        pass

print(f"\nActive columns (lambda > {eps}):")
for item_id, col_idx, lam_val, col in sorted(active_columns, key=lambda x: -x[2]):
    print(
        f"  λ[{item_id},{col_idx}] = {lam_val:.4f}: "
        f"setups={[t for t, v in enumerate(col.setup_by_period) if v > 0.5]}, "
        f"production={[(t, col.capacity_usage_by_period[t]) for t in range(T) if col.capacity_usage_by_period[t] > eps]}, "
        f"cost={col.total_plan_cost:.2f}"
    )

# Check if solution satisfies demand
print(f"\n{'='*80}")
print("=== RMP SOLUTION ANALYSIS ===")
print(f"{'='*80}")

total_production = [0.0] * T
total_setups = [0.0] * T
total_arcs = {}
total_demand_served = [0.0] * T

for item_id, col_idx, lam_val, col in active_columns:
    for t in range(T):
        total_production[t] += col.capacity_usage_by_period[t] * lam_val
        total_setups[t] += col.setup_by_period[t] * lam_val
    # Calculate demand served by arcs
    # Arc (s,u) where s and u are node indices:
    # - Production at period s (node s)
    # - Consumption at period u-1 (node u = period u-1)
    # So arc (s,u) serves period u-1
    for (s, u), val in col.arc_usage.items():
        total_arcs[(s, u)] = total_arcs.get((s, u), 0.0) + val * lam_val
        # Arc (s,u) serves period u-1 (node u = period u-1)
        cons_period = u - 1
        if 0 <= cons_period < T and demand[cons_period] > 0:
            # Column serves period cons_period, so lam_val fraction is served
            total_demand_served[cons_period] += lam_val

print("\n--- PRODUCTION PLAN ---")
print(
    f"{'Period':<8} {'Production':<12} {'Demand':<10} {'Lambda Sum':<12} {'Satisfied':<12} {'Capacity':<10} {'Utilization':<12}"
)
print("-" * 80)
total_demand = sum(demand)
total_prod = 0.0
for t in range(T):
    prod = total_production[t]
    dem = demand[t]
    lambda_sum = total_demand_served[t] if t < len(total_demand_served) else 0.0
    satisfied = "Yes" if lambda_sum >= 1.0 - eps else "No"
    cap = capacity[t]
    util = (prod / cap * 100) if cap > 0 else 0.0
    total_prod += prod
    status = "✓" if (dem == 0 or lambda_sum >= 1.0 - eps) else "✗"
    print(
        f"{t:<8} {prod:<12.2f} {dem:<10.0f} {lambda_sum:<12.4f} {satisfied:<12} {cap:<10.0f} {util:<11.1f}% {status}"
    )

print("-" * 80)
print(f"{'TOTAL':<8} {total_prod:<12.2f} {total_demand:<10.0f}")

print("\n--- SETUP PLAN ---")
print(f"{'Period':<8} {'Setup (Y)':<12} {'Has Production':<15}")
print("-" * 40)
for t in range(T):
    setup_val = total_setups[t]
    has_prod = "Yes" if total_production[t] > eps else "No"
    setup_str = f"{setup_val:.4f}" if setup_val > eps else "0.0000"
    print(f"{t:<8} {setup_str:<12} {has_prod:<15}")

print("\n--- ARC USAGE (aggregated from column lambdas) ---")
print(f"{'Arc (s,u)':<15} {'Aggregated Weight':<18} {'Meaning':<50}")
print("-" * 85)
print(
    "Note: Arc weights are aggregated from column lambdas. Each column (full plan) contributes to arcs."
)
for (s, u), val in sorted(total_arcs.items()):
    if val > 0.001:
        # Arc (s, u) where s and u are node indices:
        # - Production at period s (node s)
        # - Consumption at period u-1 (node u = period u-1)
        prod_period = s
        cons_period = u - 1  # node u = period u-1
        meaning = f"Produce at period {prod_period} (node {s}), consume at period {cons_period} (node {u})"
        print(f"({s},{u}):{'':<8} {val:<18.6f} {meaning:<50}")

print("\n--- ACTIVE COLUMNS DETAIL (Lambdas over FULL PLANS) ---")
print(f"Number of active columns: {len(active_columns)}")
print(f"Total lambda sum: {sum(lam for _, _, lam, _ in active_columns):.6f}")
print(
    "Note: Each lambda represents the weight of a FULL ZIO production plan (column) in the convex combination."
)
print("\nDetailed breakdown:")
for item_id, col_idx, lam_val, col in sorted(active_columns, key=lambda x: -x[2]):
    setups = [t for t, v in enumerate(col.setup_by_period) if v > 0.5]
    production = [
        (t, col.capacity_usage_by_period[t])
        for t in range(T)
        if col.capacity_usage_by_period[t] > eps
    ]
    arcs = sorted([(s, u) for (s, u) in col.arc_usage.keys()])
    print(f"\n  Column {col_idx} (λ={lam_val:.6f}):")
    print(f"    Cost: {col.total_plan_cost:.4f}")
    print(f"    Setups: {setups}")
    print(f"    Production: {production}")
    print(f"    Arcs: {arcs[:10]}{'...' if len(arcs) > 10 else ''}")

# Verify feasibility
print(f"\n--- FEASIBILITY CHECK ---")
demand_satisfied = True
capacity_ok = True
for t in range(T):
    if demand[t] > 0:
        lambda_sum = total_demand_served[t] if t < len(total_demand_served) else 0.0
        if lambda_sum < 1.0 - eps:
            demand_satisfied = False
            print(
                f"  ✗ Period {t}: demand {demand[t]} not fully served (lambda sum: {lambda_sum:.4f}, need >= 1.0)"
            )
    if total_production[t] > capacity[t] + eps:
        capacity_ok = False
        print(
            f"  ✗ Period {t}: production {total_production[t]:.2f} exceeds capacity {capacity[t]}"
        )

if demand_satisfied:
    print("  ✓ All demands satisfied")
if capacity_ok:
    print("  ✓ All capacity constraints satisfied")

convexity_ok = abs(sum(lam for _, _, lam, _ in active_columns) - 1.0) < eps
if convexity_ok:
    print("  ✓ Convexity constraint satisfied (sum λ = 1)")
else:
    print(
        f"  ✗ Convexity constraint violated (sum λ = {sum(lam for _, _, lam, _ in active_columns):.6f})"
    )

print(f"\n{'='*80}")
print("=== RMP SOLUTION SUMMARY ===")
print(f"{'='*80}")
print(f"Lower Bound: {lb:.4f}")
print(
    f"Status: {'OPTIMAL' if demand_satisfied and capacity_ok and convexity_ok else 'FEASIBLE' if lb < float('inf') else 'INFEASIBLE'}"
)
print(f"Active columns: {len(active_columns)}")
print(
    f"Total cost: {sum(col.total_plan_cost * lam for _, _, lam, col in active_columns):.4f}"
)
print(f"\nDone.")
