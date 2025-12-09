#!/usr/bin/env python3
"""Generate all possible ZIO columns for an item and try to decompose MIP solution."""

import json
import sys
import numpy as np
from scipy.optimize import linprog
from pathlib import Path
from typing import List, Dict, Tuple, Set
import itertools

# Load instance
instance_path = "bnp_v10_results/test_instance.json"
data = json.loads(Path(instance_path).read_text())
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
print()

# A ZIO column is a set of blocks where:
# 1. Each block (s,t) serves consecutive demands from s to t
# 2. All positive demands are covered exactly once
# 3. Inventory is zero at the end of each block


def generate_all_zio_columns() -> List[Dict]:
    """Generate all possible ZIO columns using recursive enumeration."""
    columns = []

    # Find all periods with positive demand
    demand_periods = [t for t in range(T) if demand[t] > 0]
    if not demand_periods:
        # No demand - return empty column
        return [{"setups": [], "production": {}, "arcs": [], "blocks": [], "cost": 0.0}]

    # Generate all possible ways to partition demand_periods into consecutive blocks
    # Each block (s, t) must satisfy: s <= t, and all periods from s to t are consecutive

    def is_consecutive(periods):
        """Check if periods form a consecutive sequence."""
        if not periods:
            return True
        sorted_periods = sorted(periods)
        return sorted_periods == list(range(sorted_periods[0], sorted_periods[-1] + 1))

    def generate_partitions(periods):
        """Generate all ways to partition periods into consecutive blocks."""
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

                # Check if this block can be produced (s must be in Gamma[s])
                s = block_periods[0]
                t = block_periods[-1]
                if t not in Gamma.get(s, []):
                    continue

                # Recursively generate partitions for remaining periods
                remaining = periods[:start] + periods[end + 1 :]
                for sub_partition in generate_partitions(remaining):
                    partitions.append([(s, t)] + sub_partition)

        return partitions

    # Generate all block partitions
    all_partitions = generate_partitions(demand_periods)

    print(f"Found {len(all_partitions)} possible block partitions")

    # For each partition, generate columns with different setup combinations
    # A column can have setups at any subset of block start periods
    # Also allow setup-only periods (Y=1, X=0)

    for partition in all_partitions:
        if not partition:
            continue

        # Get all block start periods
        block_starts = [s for s, t in partition]

        # Try all subsets of block_starts as setups (including empty set)
        for setup_subset in itertools.chain.from_iterable(
            itertools.combinations(block_starts, r)
            for r in range(len(block_starts) + 1)
        ):
            setups = set(setup_subset)

            # Calculate production
            production = {}
            arcs = []
            total_cost = 0.0

            for s, t in partition:
                # Production at period s
                qty = sum(demand[u] for u in range(s, t + 1))
                if qty > 0:
                    production[s] = production.get(s, 0) + qty
                    # Arcs
                    for u in range(s, t + 1):
                        if demand[u] > 0:
                            arcs.append((s, u))
                    # Cost
                    block_cost = 0.0
                    if s in setups:
                        block_cost += s_at(s)
                    block_cost += c_at(s) * qty
                    for u in range(s, t + 1):
                        if demand[u] > 0:
                            block_cost += H_i(s, u) * demand[u]
                    total_cost += block_cost

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

    # Remove duplicates
    seen = set()
    unique_columns = []
    for col in columns:
        # Key: (tuple of setups, tuple of (production period, qty) pairs)
        key = (tuple(col["setups"]), tuple(sorted(col["production"].items())))
        if key not in seen:
            seen.add(key)
            unique_columns.append(col)

    return unique_columns


print("Generating all ZIO columns...")
zio_columns = generate_all_zio_columns()
print(f"Generated {len(zio_columns)} unique ZIO columns\n")

# Show first 10 columns
print("First 10 columns:")
for i, col in enumerate(zio_columns[:10]):
    print(
        f"  Col {i+1}: setups={col['setups']}, production={col['production']}, cost={col['cost']:.2f}"
    )

# Now get MIP solution
print(f"\n=== GETTING MIP SOLUTION ===")
import gurobipy as gp
from gurobipy import GRB

# Build MIP model (simplified - just to get solution)
m = gp.Model("get_solution")
m.Params.OutputFlag = 0

Triples = []
for t in range(T):
    for u in Gamma.get(t, []):
        Triples.append((t, u))

X = {}
Z = {}
Y = {}
for t, u in Triples:
    X[(t, u)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"X_{t}_{u}")
    Z[(t, u)] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f"Z_{t}_{u}")

for t in range(T):
    Y[t] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f"Y_{t}")

m.update()

# Constraints
for t in range(T):
    if not Gamma.get(t):
        m.addConstr(Y[t] == 0)
    else:
        mu_t = sum(float(demand[u]) for u in Gamma.get(t, []))
        expr = gp.quicksum(X.get((t, u), 0) for u in Gamma.get(t, []))
        m.addConstr(expr <= mu_t * Y[t])

for u in range(T):
    if demand[u] <= 0:
        continue
    origins = [t for t in range(u + 1) if u in Gamma.get(t, [])]
    m.addConstr(gp.quicksum(X[(t, u)] for t in origins if (t, u) in X) == demand[u])

for t, u in Triples:
    m.addConstr(X[(t, u)] <= float(demand[u]) * Z[(t, u)])

# LEFO
prods = [t for t in range(T) if Gamma.get(t)]
prods.sort(key=lambda t: Expiry.get(t, t))
for a in range(len(prods)):
    t1 = prods[a]
    v1 = Expiry[t1]
    for b in range(a + 1, len(prods)):
        t2 = prods[b]
        v2 = Expiry[t2]
        if v1 >= v2:
            continue
        for up in Gamma[t2]:
            for u in [uu for uu in Gamma[t1] if t2 <= uu <= up - 1]:
                if (t1, u) in Z and (t2, up) in Z:
                    m.addConstr(Z[(t1, u)] + Z[(t2, up)] <= 1)

# Capacity
capacity = data.get("manual_capacity") or data.get("production_capacity")
if isinstance(capacity, (int, float)):
    capacity = [float(capacity)] * T
else:
    capacity = [float(c) for c in capacity]

for t in range(T):
    expr = gp.quicksum(X.get((t, u), 0) for u in Gamma.get(t, []))
    m.addConstr(expr <= capacity[t])

# Objective
obj = gp.LinExpr()
for t in range(T):
    obj += s_at(t) * Y[t]
for t, u in Triples:
    h_sum = sum(h_at(k) for k in range(t, u))
    obj += (c_at(t) + h_sum) * X[(t, u)]

m.setObjective(obj, GRB.MINIMIZE)
m.optimize()

if m.Status != GRB.OPTIMAL:
    print(f"MIP solve failed: {m.Status}")
    sys.exit(1)

# Extract MIP solution
mip_X = {}
mip_Y = {}
mip_Z = {}
eps = 1e-6

for t in range(T):
    y_val = Y[t].X
    if y_val > eps:
        mip_Y[t] = y_val

for t, u in Triples:
    x_val = X[(t, u)].X
    z_val = Z[(t, u)].X
    if x_val > eps:
        mip_X[(t, u)] = x_val
    if z_val > eps:
        mip_Z[(t, u)] = z_val

prod_by_period = {}
for (t, u), val in mip_X.items():
    prod_by_period[t] = prod_by_period.get(t, 0) + val

print(f"MIP Objective: {m.ObjVal:.2f}")
print(f"Y (setups): {sorted(mip_Y.keys())}")
print(f"Production by period: {prod_by_period}")
print(f"X arcs: {sorted(mip_X.keys())}")

# Try to decompose
print(f"\n=== DECOMPOSING MIP SOLUTION ===")

# Build constraint matrix
constraints = []
for t in range(T):
    row = []
    for col in zio_columns:
        row.append(col["production"].get(t, 0))
    constraints.append(row)

# Convexity
constraints.append([1] * len(zio_columns))

A_eq = np.array(constraints)
b_eq = np.array([prod_by_period.get(t, 0) for t in range(T)] + [1.0])
c = np.array([col["cost"] for col in zio_columns])

print(
    f"Solving LP with {len(zio_columns)} columns and {len(constraints)} constraints..."
)
result = linprog(c, A_eq=A_eq, b_eq=b_eq, method="highs")

if result.success:
    lambdas = result.x
    print(f"\n✓ Solution found!")
    print(f"Total cost: {result.fun:.2f}")
    print(f"MIP cost: {m.ObjVal:.2f}")
    print(f"Match: {abs(result.fun - m.ObjVal) < 1.0}")

    # Show used columns
    print(f"\nColumns used (lambda > 0.001):")
    for i, lam in enumerate(lambdas):
        if lam > 0.001:
            print(
                f"  λ_{i+1} = {lam:.4f}: setups={zio_columns[i]['setups']}, production={zio_columns[i]['production']}, cost={zio_columns[i]['cost']:.2f}"
            )
else:
    print(f"\n✗ Could not find convex combination!")
    print(f"Status: {result.message}")
