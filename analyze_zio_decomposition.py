#!/usr/bin/env python3
"""Analyze what's needed to decompose MIP solution into ZIO columns."""

import json
import numpy as np
from scipy.optimize import linprog
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB

# Load instance
data = json.loads(Path("bnp_v10_results/test_instance.json").read_text())
T = 10
item_data = data["items"]["0"]
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

h_prefix = [0.0]
for k in range(T):
    h_prefix.append(h_prefix[-1] + h_at(k))

def H_i(s, u):
    if u <= s:
        return 0.0
    return h_prefix[u] - h_prefix[s]

# Calculate Gamma
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
    X[(t, u)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS)
    Z[(t, u)] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)

for t in range(T):
    Y[t] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)

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

# Extract MIP solution
mip_X = {}
mip_Y = {}
eps = 1e-6

for t in range(T):
    y_val = Y[t].X
    if y_val > eps:
        mip_Y[t] = y_val

for t, u in Triples:
    x_val = X[(t, u)].X
    if x_val > eps:
        mip_X[(t, u)] = x_val

prod_by_period = {}
for (t, u), val in mip_X.items():
    prod_by_period[t] = prod_by_period.get(t, 0) + val

print(f"MIP Objective: {m.ObjVal:.2f}")
print(f"Production by period: {prod_by_period}")
print(f"\nMIP arcs (X values):")
for (t, u) in sorted(mip_X.keys()):
    print(f"  X[{t},{u}] = {mip_X[(t, u)]:.1f} (demand[{u}] = {demand[u]})")

print(f"\n=== ZIO VIOLATION ANALYSIS ===")
print("MIP solution violates ZIO:")
violations = []
for t in sorted(prod_by_period.keys()):
    arcs_from_t = [(t2, u) for (t2, u) in mip_X.keys() if t2 == t]
    if len(arcs_from_t) > 1:
        u_values = sorted([u for (t2, u) in arcs_from_t])
        # Check for gaps
        for i in range(len(u_values) - 1):
            u1, u2 = u_values[i], u_values[i + 1]
            if u2 - u1 > 1:
                # Gap found - check if intermediate demands are positive
                for u_mid in range(u1 + 1, u2):
                    if demand[u_mid] > 0:
                        violations.append((t, u1, u_mid, u2))
                        print(f"  Period {t}: serves demand[{u1}] and demand[{u2}], but NOT demand[{u_mid}]")

print(f"\n=== WHAT'S NEEDED FOR ZIO DECOMPOSITION ===")
print("To decompose the MIP solution into ZIO columns, we need:")
print("\n1. Multiple ZIO columns that can be combined with fractional lambdas")
print("2. Columns where:")
print("   - Some columns serve demand[4] from period 3 (block (3,4) or (3,5))")
print("   - Some columns serve demand[4] from period 4 (block (4,4) or (4,5))")
print("   - When combined: λ_A * (serve 4 from 3) + λ_B * (serve 4 from 4) = serve 4 from 4")
print("   - This requires λ_A = 0, λ_B = 1, which doesn't help")
print("\n3. The key insight: We need columns that serve demand[5] from period 3")
print("   WITHOUT serving demand[4] from period 3.")
print("   But this violates ZIO! In a ZIO column, if you serve demand[5] from period 3,")
print("   you MUST also serve demand[4] from period 3 (if demand[4] > 0).")
print("\n4. CONCLUSION: The MIP solution CANNOT be decomposed into pure ZIO columns")
print("   because it violates the ZIO property.")
print("\n5. However, we might be able to APPROXIMATE it with ZIO columns:")
print("   - Use columns that serve demand[4] from period 3 (even though MIP doesn't)")
print("   - Use columns that serve demand[4] from period 4")
print("   - Combine them so the net effect is close to the MIP pattern")
print("   - But this will have a different cost structure")

print(f"\n=== GENERATING COLUMNS FOR APPROXIMATION ===")
# Generate columns that might approximate the MIP solution
zio_columns = []

def create_zio_column(blocks, setups=None):
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
    
    for t in setups:
        if t not in production:
            total_cost += s_at(t)
    
    return {
        "setups": sorted(setups),
        "production": production,
        "arcs": sorted(arcs),
        "blocks": blocks,
        "cost": total_cost,
    }

# Try columns that might approximate the MIP
# MIP: period 3 serves [3] and [5], period 4 serves [4] and [5]
# We can't have a ZIO column that serves [5] from 3 without [4]
# But we can have:
# - Column A: (3,5) serves [3],[4],[5] from 3
# - Column B: (3,3) and (4,5) serves [3] from 3, [4],[5] from 4
# - If λ_A = 0, λ_B = 1, we get: [3] from 3, [4],[5] from 4 (close but not exact)

# Column 1: (2,2), (3,5), (4,4), (6,6), (7,9)
zio_columns.append(create_zio_column([(2, 2), (3, 5), (4, 4), (6, 6), (7, 9)], [2, 3, 4, 6, 7]))

# Column 2: (2,2), (3,3), (4,5), (6,6), (7,9)
zio_columns.append(create_zio_column([(2, 2), (3, 3), (4, 5), (6, 6), (7, 9)], [2, 3, 4, 6, 7]))

# Column 3: (2,2), (3,4), (5,5), (6,6), (7,9)
zio_columns.append(create_zio_column([(2, 2), (3, 4), (5, 5), (6, 6), (7, 9)], [2, 3, 5, 6, 7]))

# Column 4: (2,2), (3,3), (4,4), (5,5), (6,6), (7,9)
zio_columns.append(create_zio_column([(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 9)], [2, 3, 4, 5, 6, 7]))

print(f"Generated {len(zio_columns)} approximation columns")
for i, col in enumerate(zio_columns):
    print(f"  Col {i+1}: blocks={col['blocks']}, cost={col['cost']:.2f}")

# Try to decompose
print(f"\n=== TRYING DECOMPOSITION ===")
constraints = []
for t in range(T):
    row = []
    for col in zio_columns:
        row.append(col["production"].get(t, 0))
    constraints.append(row)

constraints.append([1] * len(zio_columns))
A_eq = np.array(constraints)
b_eq = np.array([prod_by_period.get(t, 0) for t in range(T)] + [1.0])
c = np.array([col["cost"] for col in zio_columns])

result = linprog(c, A_eq=A_eq, b_eq=b_eq, method="highs")

if result.success:
    lambdas = result.x
    print(f"✓ Found decomposition!")
    print(f"Cost: {result.fun:.2f} (MIP: {m.ObjVal:.2f})")
    for i, lam in enumerate(lambdas):
        if lam > 0.001:
            print(f"  λ_{i+1} = {lam:.4f}: {zio_columns[i]['blocks']}")
else:
    print(f"✗ Cannot decompose with these columns: {result.message}")

print(f"\n=== SUMMARY ===")
print("The MIP solution violates ZIO and cannot be exactly decomposed into ZIO columns.")
print("However, we can approximate it using ZIO columns with fractional lambdas.")
print("The approximation may have a different cost than the MIP solution.")

