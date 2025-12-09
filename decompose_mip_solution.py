#!/usr/bin/env python3
"""Decompose MIP solution into ZIO columns to verify if it can be written as convex combination."""

import json
import sys
import numpy as np
from scipy.optimize import linprog
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB

# Load instance
instance_path = 'bnp_v10_results/test_instance.json'
data = json.loads(Path(instance_path).read_text())
T = int(data['period'])
items_raw = {int(k): v for k, v in data['items'].items()}

# Build MIP model to get solution
m = gp.Model("extract_solution")
m.Params.OutputFlag = 0

# Calculate Gamma and Expiry
Gamma: dict = {}
Expiry: dict = {}
Triples = []
for i, it in items_raw.items():
    mseq = list(it['shelf_seq'])
    for t in range(T):
        m_it = int(mseq[t])
        v_it = t + m_it
        Expiry[(i, t)] = v_it
        if m_it <= 0:
            Gamma[(i, t)] = []
            continue
        u_max = min(T - 1, v_it)  # Match BNP solver
        us = [u for u in range(t, u_max + 1)]
        Gamma[(i, t)] = us
        for u in us:
            Triples.append((i, t, u))

# Variables
X = {}
Z = {}
Y = {}
for i, t, u in Triples:
    X[(i, t, u)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"X_{i}_{t}_{u}")
    Z[(i, t, u)] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f"Z_{i}_{t}_{u}")

for i in items_raw:
    for t in range(T):
        Y[(i, t)] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f"Y_{i}_{t}")

m.update()

# Constraints
for i, it in items_raw.items():
    demand = list(it['demand'])
    setup = it['setup']
    c_var = it['c_var']
    h = it['h']
    
    # Setup linking
    for t in range(T):
        if not Gamma.get((i, t)):
            m.addConstr(Y[(i, t)] == 0)
        else:
            mu_it = sum(float(demand[u]) for u in Gamma.get((i, t), []))
            expr = gp.quicksum(X.get((i, t, u), 0) for u in Gamma.get((i, t), []))
            m.addConstr(expr <= mu_it * Y[(i, t)])
    
    # Demand satisfaction
    for u in range(T):
        if demand[u] <= 0:
            continue
        origins = [t for t in range(u + 1) if u in Gamma.get((i, t), [])]
        m.addConstr(gp.quicksum(X[(i, t, u)] for t in origins if (i, t, u) in X) == demand[u])
    
    # Arc activation
    for t, u_list in [(t, Gamma.get((i, t), [])) for t in range(T)]:
        for u in u_list:
            if (i, t, u) in X:
                m.addConstr(X[(i, t, u)] <= float(demand[u]) * Z[(i, t, u)])

# LEFO constraints
for i in items_raw:
    prods = [t for t in range(T) if Gamma.get((i, t))]
    prods.sort(key=lambda t: Expiry[(i, t)])
    for a in range(len(prods)):
        t1 = prods[a]
        v1 = Expiry[(i, t1)]
        for b in range(a + 1, len(prods)):
            t2 = prods[b]
            v2 = Expiry[(i, t2)]
            if v1 >= v2:
                continue
            for up in Gamma[(i, t2)]:
                for u in [uu for uu in Gamma[(i, t1)] if t2 <= uu <= up - 1]:
                    if (i, t1, u) in Z and (i, t2, up) in Z:
                        m.addConstr(Z[(i, t1, u)] + Z[(i, t2, up)] <= 1)

# Capacity
capacity = data.get('manual_capacity') or data.get('production_capacity')
if isinstance(capacity, (int, float)):
    capacity = [float(capacity)] * T
else:
    capacity = [float(c) for c in capacity]

for t in range(T):
    expr = gp.quicksum(X.get((i, t, u), 0) for i in items_raw for u in Gamma.get((i, t), []) if (i, t, u) in X)
    m.addConstr(expr <= capacity[t])

# Objective
obj = gp.LinExpr()
for i, it in items_raw.items():
    demand = list(it['demand'])
    setup = it['setup']
    c_var = it['c_var']
    h = it['h']
    
    def s_at(t):
        return setup[t] if isinstance(setup, list) else setup
    
    def c_at(t):
        return c_var[t] if isinstance(c_var, list) else c_var
    
    def h_at(t):
        return h[t] if isinstance(h, list) else h
    
    # Setup costs
    for t in range(T):
        obj += s_at(t) * Y[(i, t)]
    
    # Production and holding costs
    for t, u_list in [(t, Gamma.get((i, t), [])) for t in range(T)]:
        for u in u_list:
            if (i, t, u) in X:
                h_sum = sum(h_at(k) for k in range(t, u))
                obj += (c_at(t) + h_sum) * X[(i, t, u)]

m.setObjective(obj, GRB.MINIMIZE)
m.optimize()

if m.Status != GRB.OPTIMAL:
    print(f"MIP solve failed: {m.Status}")
    sys.exit(1)

print(f"MIP Objective: {m.ObjVal:.2f}\n")

# Extract solution for item 0
item_id = 0
item_data = items_raw[item_id]
demand = list(item_data['demand'])
setup = item_data['setup']
c_var = item_data['c_var']
h = item_data['h']

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

# Extract MIP solution
mip_X = {}
mip_Y = {}
mip_Z = {}
eps = 1e-6

for t in range(T):
    y_val = Y[(item_id, t)].X
    if y_val > eps:
        mip_Y[t] = y_val

for t, u_list in [(t, Gamma.get((item_id, t), [])) for t in range(T)]:
    for u in u_list:
        if (item_id, t, u) in X:
            x_val = X[(item_id, t, u)].X
            z_val = Z[(item_id, t, u)].X
            if x_val > eps:
                mip_X[(t, u)] = x_val
            if z_val > eps:
                mip_Z[(t, u)] = z_val

print(f"=== MIP SOLUTION FOR ITEM {item_id} ===")
print(f"Y (setups): {sorted(mip_Y.keys())}")
print(f"X arcs: {sorted(mip_X.keys())}")
print(f"Z arcs: {sorted(mip_Z.keys())}")

# Calculate production by period
prod_by_period = {}
for (t, u), val in mip_X.items():
    prod_by_period[t] = prod_by_period.get(t, 0) + val

print(f"Production by period: {prod_by_period}")

# Check for setup-only periods
setup_only = []
for t in mip_Y.keys():
    if t not in prod_by_period or prod_by_period[t] < eps:
        setup_only.append(t)
        print(f"  *** SETUP-ONLY at period {t} (Y=1, X=0) ***")

print(f"\n=== GENERATING ZIO COLUMNS ===")

# Generate ZIO columns
# A ZIO column is a set of blocks where each block (s,t) serves consecutive demands
# from s to t with zero inventory at the end

zio_columns = []

# Strategy: Generate columns by trying different block combinations
# We'll generate columns that could potentially form the MIP solution

# Helper to check if a block (s, t) is valid (serves consecutive demands)
def is_valid_block(s, t, served_demands):
    """Check if block (s,t) serves consecutive demands from s to t."""
    if t < s:
        return False
    # Check that all periods from s to t have positive demand or are served
    for u in range(s, t + 1):
        if u not in served_demands and demand[u] > 0:
            return False
    return True

# Generate columns systematically
# For each setup period, try different block endings

def generate_zio_column(setup_periods, block_assignments):
    """Generate a ZIO column given setup periods and block assignments.
    
    block_assignments: list of (s, t) tuples where block (s,t) serves demands from s to t
    """
    # Verify all demands are covered
    covered_demands = set()
    for s, t in block_assignments:
        for u in range(s, t + 1):
            if u < T:
                covered_demands.add(u)
    
    # Check if all positive demands are covered
    for u in range(T):
        if demand[u] > 0 and u not in covered_demands:
            return None
    
    # Calculate production quantities
    production = {}
    arcs = []
    total_cost = 0.0
    
    for s, t in block_assignments:
        # Production at period s
        qty = sum(demand[u] for u in range(s, min(t + 1, T)) if demand[u] > 0)
        if qty > 0:
            production[s] = production.get(s, 0) + qty
            # Arcs
            for u in range(s, min(t + 1, T)):
                if demand[u] > 0:
                    arcs.append((s, u))
            # Cost
            block_cost = s_at(s) + c_at(s) * qty
            for u in range(s, min(t + 1, T)):
                if demand[u] > 0:
                    block_cost += H_i(s, u) * demand[u]
            total_cost += block_cost
    
    # Add setup-only periods
    for t in setup_periods:
        if t not in production:
            total_cost += s_at(t)
    
    return {
        'setups': sorted(set(setup_periods)),
        'production': production,
        'arcs': sorted(arcs),
        'blocks': block_assignments,
        'cost': total_cost
    }

# Try to generate columns that match the MIP structure
# MIP has production at periods: 2, 3, 4, 6, 7

# Column 1: Block (2,2) - serve demand[2]
col1 = generate_zio_column([2], [(2, 2)])
if col1:
    zio_columns.append(col1)

# Column 2: Block (3,4) - serve demand[3] and demand[4]
col2 = generate_zio_column([3], [(3, 4)])
if col2:
    zio_columns.append(col2)

# Column 3: Block (4,5) - serve demand[4] and demand[5]
col3 = generate_zio_column([4], [(4, 5)])
if col3:
    zio_columns.append(col3)

# Column 4: Block (6,8) - serve demand[6], [7], [8]
col4 = generate_zio_column([6], [(6, 8)])
if col4:
    zio_columns.append(col4)

# Column 5: Block (7,9) - serve demand[8] and demand[9]
col5 = generate_zio_column([7], [(7, 9)])
if col5:
    zio_columns.append(col5)

# Also try alternative decompositions
# Column 6: Block (3,3) and (4,5)
col6 = generate_zio_column([3, 4], [(3, 3), (4, 5)])
if col6:
    zio_columns.append(col6)

# Column 7: Block (6,7) and (7,9)
col7 = generate_zio_column([6, 7], [(6, 7), (7, 9)])
if col7:
    zio_columns.append(col7)

# Column 8: Block (2,2), (3,3), (4,5), (6,8), (7,9)
col8 = generate_zio_column([2, 3, 4, 6, 7], [(2, 2), (3, 3), (4, 5), (6, 8), (7, 9)])
if col8:
    zio_columns.append(col8)

# Remove duplicates based on production pattern
seen = set()
unique_columns = []
for col in zio_columns:
    key = tuple(sorted(col['production'].items()))
    if key not in seen:
        seen.add(key)
        unique_columns.append(col)

zio_columns = unique_columns

print(f"Generated {len(zio_columns)} unique ZIO columns:")
for i, col in enumerate(zio_columns):
    print(f"  Col {i+1}: setups={col['setups']}, production={col['production']}, cost={col['cost']:.2f}")

# Now try to decompose MIP solution
print(f"\n=== DECOMPOSING MIP SOLUTION ===")

# Build constraint matrix
# For each period t, sum(λ_i * col_i.prod[t]) = mip_prod[t]
# For each arc (t,u), sum(λ_i * col_i.arc_usage[(t,u)]) = mip_Z[(t,u)] (if needed)
# Convexity: sum(λ_i) = 1

constraints = []
constraint_names = []

# Production constraints by period
for t in range(T):
    row = []
    for col in zio_columns:
        row.append(col['production'].get(t, 0))
    constraints.append(row)
    constraint_names.append(f'prod_{t}')

# Arc constraints (if Z is fractional, we need to match it)
# For now, let's focus on production matching

# Convexity constraint
constraints.append([1] * len(zio_columns))
constraint_names.append('convexity')

A_eq = np.array(constraints)

# Right-hand side
b_eq = []
for t in range(T):
    b_eq.append(prod_by_period.get(t, 0))
b_eq.append(1.0)  # Convexity
b_eq = np.array(b_eq)

# Objective: minimize cost
c = np.array([col['cost'] for col in zio_columns])

print(f"Solving LP with {len(zio_columns)} columns and {len(constraints)} constraints...")
result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')

if result.success:
    lambdas = result.x
    print(f"\n✓ Solution found!")
    print(f"Lambdas: {[f'{l:.4f}' for l in lambdas]}")
    print(f"Total cost: {result.fun:.2f}")
    print(f"MIP cost: {m.ObjVal:.2f}")
    print(f"Match: {abs(result.fun - m.ObjVal) < 1.0}")
    
    # Verify production
    print(f"\nVerifying production:")
    for t in range(T):
        total_prod = sum(lambdas[i] * zio_columns[i]['production'].get(t, 0) for i in range(len(zio_columns)))
        mip_prod = prod_by_period.get(t, 0)
        match = abs(total_prod - mip_prod) < 0.1
        print(f"  Period {t}: lambda_prod={total_prod:.2f}, MIP={mip_prod:.2f}, match={match}")
    
    # Show which columns are used
    print(f"\nColumns used (lambda > 0.01):")
    for i, lam in enumerate(lambdas):
        if lam > 0.01:
            print(f"  λ_{i+1} = {lam:.4f}: {zio_columns[i]}")
else:
    print(f"\n✗ Could not find convex combination!")
    print(f"Status: {result.message}")
    print(f"\nTrying with more columns...")
    
    # Generate more columns by trying different block combinations
    # This is a simplified approach - in practice, we'd use the DP to generate all ZIO columns
    
    print(f"\nNote: May need to generate more ZIO columns using DP pricing subproblem.")

