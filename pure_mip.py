#!/usr/bin/env python3
# ---------------------------------------------------------------------------
#  Compact MIP for the perishable lot–sizing instance in last_instance.json
#  – works with / without back-orders, controlled by ALLOW_BACKLOG.
#  – Gurobi ≥ 10 required.
# ---------------------------------------------------------------------------

import json, gurobipy as gp
from gurobipy import GRB

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
ALLOW_BACKLOG = True  # ← flip to False to forbid back-orders
TIME_LIMIT = 3600  # seconds (set 0 for unlimited)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ---------------------------------------------------------------------------
# 1.  Load data --------------------------------------------------------------
# ---------------------------------------------------------------------------
with open("last_instance.json", "r") as fh:
    data = json.load(fh)

T = data["period"]
cap = data["manual_capacity"]
Periods = range(T)

items = {}
for k, it in data["items"].items():
    k = int(k)
    items[k] = dict(
        demand=it["demand"],
        setup=it["setup"],
        c_var=it["c_var"],
        h=it["h"],
        b_var=it["b_var"],
        life=it["shelf"],
        M=sum(it["demand"]),  # safe Big-M
    )

# ---------------------------------------------------------------------------
# 2.  Model ------------------------------------------------------------------
# ---------------------------------------------------------------------------
m = gp.Model("perishable_LS_compact")
m.Params.OutputFlag = 1
if TIME_LIMIT:
    m.Params.TimeLimit = TIME_LIMIT

# ----- index helper ---------------------------------------------------------
Triples = [  # (i,t,u) with 0 ≤ t ≤ u < T,  u-t < life
    (i, t, u)
    for i, it in items.items()
    for t in Periods
    for u in range(t, min(T, t + it["life"]))
]

# ----- variables ------------------------------------------------------------
x = m.addVars(Triples, vtype=GRB.INTEGER, name="x")  # production
y = m.addVars(
    [(i, t) for i in items for t in Periods], vtype=GRB.BINARY, name="y"
)  # setup

if ALLOW_BACKLOG:
    b = m.addVars(
        [(i, u) for i in items for u in Periods],
        vtype=GRB.INTEGER,
        lb=0,
        name="backlog",
    )
else:  # dummy zero-vars (constant 0) to keep code uniform
    b = {(i, u): gp.LinExpr(0) for i in items for u in Periods}

# ----- objective ------------------------------------------------------------
obj = gp.quicksum(
    (items[i]["c_var"] + items[i]["h"] * (u - t)) * x[i, t, u] for (i, t, u) in Triples
)

obj += gp.quicksum(items[i]["setup"] * y[i, t] for (i, t) in y)

if ALLOW_BACKLOG:
    obj += gp.quicksum(items[i]["b_var"] * b[i, u] for (i, u) in b)

m.setObjective(obj, GRB.MINIMIZE)

# ----- capacity -------------------------------------------------------------
for t in Periods:
    m.addConstr(
        gp.quicksum(
            x[i, t, u]
            for i, it in items.items()
            for u in range(t, min(T, t + it["life"]))
        )
        <= cap[t],
        name=f"cap_{t}",
    )

# ----- setup linking --------------------------------------------------------
for (i, t), y_it in y.items():
    m.addConstr(
        gp.quicksum(x[i, t, u] for u in range(t, min(T, t + items[i]["life"])))
        <= items[i]["M"] * y_it,
        name=f"setupLink_{i}_{t}",
    )


# ---------- helper: feasible origins for a given consumption period u -------
def origins(u, life):
    """Return all t that can still be 'fresh' when consumed in period u."""
    return range(max(0, u - life + 1), u + 1)  # ensures t ≤ u and u-t < life


# ---------- demand balance (incl. backlog) ----------------------------------
for i, it in items.items():
    d, L = it["demand"], it["life"]

    # period 0
    m.addConstr(
        b[i, 0] + gp.quicksum(x[i, t, 0] for t in origins(0, L)) == d[0],
        name=f"demand_{i}_0",
    )

    # periods 1 … T-1
    for u in range(1, T):
        m.addConstr(
            b[i, u] + gp.quicksum(x[i, t, u] for t in origins(u, L))
            == d[u] + b[i, u - 1],
            name=f"demand_{i}_{u}",
        )

        if not ALLOW_BACKLOG:  # forbid back-orders → b ≡ 0
            m.chgCoeff(m.getConstrByName(f"demand_{i}_{u}"), b[i, u], 0)

    if not ALLOW_BACKLOG:  # also zero-out period-0 backlog
        m.chgCoeff(m.getConstrByName(f"demand_{i}_0"), b[i, 0], 0)

# ---------------------------------------------------------------------------
# 3.  Solve & report ---------------------------------------------------------
# ---------------------------------------------------------------------------
m.optimize()

print("\n=================  RESULTS  =================\n")
if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
    print(f"Objective value     : {m.ObjVal:,.2f}")
    print(f"Best bound          : {m.ObjBound:,.2f}")
    print(f"MIP gap             : {m.MIPGap*100:6.2f} %")
else:
    print("Solver ended with status:", m.Status)

# ----- pretty print orders --------------------------------------------------
for i, it in items.items():
    print(f"\nItem {i} — orders (period → qty)")
    for t in Periods:
        qty = sum(x[i, t, u].X for u in range(t, min(T, t + it["life"])))
        if qty > 1e-6:
            print(f"  {t:2d} → {qty:4.0f}")
