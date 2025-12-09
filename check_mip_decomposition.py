#!/usr/bin/env python3
"""Check if MIP solution can be decomposed into BNP columns."""

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

item_id = 0
item_data = items_raw[item_id]
demand = list(item_data['demand'])
setup = item_data['setup']
c_var = item_data['c_var']
h = item_data['h']
shelf_seq = item_data['shelf_seq']

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

# Get MIP solution
print("=== GETTING MIP SOLUTION ===")
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
capacity = data.get('manual_capacity') or data.get('production_capacity')
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

# Now generate some candidate ZIO columns manually
# These are columns that the BNP solver might generate

print(f"\n=== GENERATING CANDIDATE ZIO COLUMNS ===")

zio_columns = []

# Helper to create a ZIO column from a list of blocks
def create_zio_column(blocks, setups=None):
    """Create a ZIO column from blocks (s, t) tuples."""
    if setups is None:
        setups = [s for s, t in blocks]
    
    production = {}
    arcs = []
    total_cost = 0.0
    
    for s, t in blocks:
        qty = sum(demand[u] for u in range(s, t + 1))
        if qty > 0:
            production[s] = production.get(s, 0) + qty
            for u in range(s, t + 1):
                if demand[u] > 0:
                    arcs.append((s, u))
            block_cost = 0.0
            if s in setups:
                block_cost += s_at(s)
            block_cost += c_at(s) * qty
            for u in range(s, t + 1):
                if demand[u] > 0:
                    block_cost += H_i(s, u) * demand[u]
            total_cost += block_cost
    
    # Setup-only periods
    for t in setups:
        if t not in production:
            total_cost += s_at(t)
    
    return {
        'setups': sorted(setups),
        'production': production,
        'arcs': sorted(arcs),
        'blocks': blocks,
        'cost': total_cost
    }

# Generate columns that might help decompose the MIP solution
# MIP has production at: 2, 3, 4, 6, 7
# MIP arcs: (2,2), (3,3), (3,5), (4,4), (4,5), (6,6), (6,9), (7,7), (7,8), (7,9)

# Column 1: Block (2,2) - serve demand[2]
zio_columns.append(create_zio_column([(2, 2)], [2]))

# Column 2: Block (3,3) - serve demand[3]
zio_columns.append(create_zio_column([(3, 3)], [3]))

# Column 3: Block (3,4) - serve demand[3] and [4]
zio_columns.append(create_zio_column([(3, 4)], [3]))

# Column 4: Block (3,5) - serve demand[3], [4], [5]
zio_columns.append(create_zio_column([(3, 5)], [3]))

# Column 5: Block (4,4) - serve demand[4]
zio_columns.append(create_zio_column([(4, 4)], [4]))

# Column 6: Block (4,5) - serve demand[4] and [5]
zio_columns.append(create_zio_column([(4, 5)], [4]))

# Column 7: Block (6,6) - serve demand[6]
zio_columns.append(create_zio_column([(6, 6)], [6]))

# Column 8: Block (6,7) - serve demand[6] and [7]
zio_columns.append(create_zio_column([(6, 7)], [6]))

# Column 9: Block (6,8) - serve demand[6], [7], [8]
zio_columns.append(create_zio_column([(6, 8)], [6]))

# Column 10: Block (6,9) - serve demand[6], [7], [8], [9]
zio_columns.append(create_zio_column([(6, 9)], [6]))

# Column 11: Block (7,7) - serve demand[7]
zio_columns.append(create_zio_column([(7, 7)], [7]))

# Column 12: Block (7,8) - serve demand[7] and [8]
zio_columns.append(create_zio_column([(7, 8)], [7]))

# Column 13: Block (7,9) - serve demand[7], [8], [9]
zio_columns.append(create_zio_column([(7, 9)], [7]))

# Column 14: Multiple blocks: (2,2), (3,3), (4,5), (6,6), (7,9)
zio_columns.append(create_zio_column([(2, 2), (3, 3), (4, 5), (6, 6), (7, 9)], [2, 3, 4, 6, 7]))

# Column 15: Multiple blocks: (2,2), (3,5), (4,4), (6,9), (7,8)
zio_columns.append(create_zio_column([(2, 2), (3, 5), (4, 4), (6, 9), (7, 8)], [2, 3, 4, 6, 7]))

print(f"Generated {len(zio_columns)} candidate ZIO columns")

# Try to decompose
print(f"\n=== DECOMPOSING MIP SOLUTION ===")

# Build constraint matrix
constraints = []
for t in range(T):
    row = []
    for col in zio_columns:
        row.append(col['production'].get(t, 0))
    constraints.append(row)

# Convexity
constraints.append([1] * len(zio_columns))

A_eq = np.array(constraints)
b_eq = np.array([prod_by_period.get(t, 0) for t in range(T)] + [1.0])
c = np.array([col['cost'] for col in zio_columns])

print(f"Solving LP with {len(zio_columns)} columns and {len(constraints)} constraints...")
result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')

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
            print(f"  λ_{i+1} = {lam:.4f}: setups={zio_columns[i]['setups']}, production={zio_columns[i]['production']}, cost={zio_columns[i]['cost']:.2f}")
    
    # Verify production
    print(f"\nVerifying production:")
    for t in range(T):
        total_prod = sum(lambdas[i] * zio_columns[i]['production'].get(t, 0) for i in range(len(zio_columns)))
        mip_prod = prod_by_period.get(t, 0)
        match = abs(total_prod - mip_prod) < 0.1
        print(f"  Period {t}: lambda_prod={total_prod:.2f}, MIP={mip_prod:.2f}, match={match}")
else:
    print(f"\n✗ Could not find convex combination with these columns!")
    print(f"Status: {result.message}")
    print(f"\nThis suggests we need different ZIO columns to decompose the MIP solution.")

