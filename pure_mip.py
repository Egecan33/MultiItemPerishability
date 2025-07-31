#!/usr/bin/env python3
# ---------------------------------------------------------------
# Perishable lot-sizing – compact MIP (with or without back-orders)
# ---------------------------------------------------------------

import json, gurobipy as gp
from gurobipy import GRB

# ───────────── user switches ───────────────────────────────────
ALLOW_BACKLOG = False  # ← flip to True to allow back-orders
TIME_LIMIT = 3600  # seconds (0 → unlimited)
# ───────────────────────────────────────────────────────────────

# 1. load instance ----------------------------------------------------------
with open("last_instance.json") as fh:
    data = json.load(fh)

T = data["period"]
cap = data["manual_capacity"]
Periods = range(T)

items = {
    int(k): dict(
        demand=it["demand"],
        setup=it["setup"],
        c_var=it["c_var"],
        h=it["h"],
        b_var=it["b_var"],
        life=it["shelf"],
        M=sum(it["demand"]),
    )
    for k, it in data["items"].items()
}

# 2. model ------------------------------------------------------------------
m = gp.Model("perishable_LS_compact")
m.Params.OutputFlag = 1
if TIME_LIMIT:
    m.Params.TimeLimit = TIME_LIMIT

# helper: feasible production triples (i,t,u)
Triples = [
    (i, t, u)
    for i, it in items.items()
    for t in Periods
    for u in range(t, min(T, t + it["life"]))
]

# decision variables --------------------------------------------------------
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
else:
    # dummy 0-expressions so later code can stay uniform
    b = {(i, u): gp.LinExpr(0) for i in items for u in Periods}

# objective -----------------------------------------------------------------
obj = gp.quicksum(
    (items[i]["c_var"] + items[i]["h"] * (u - t)) * x[i, t, u] for (i, t, u) in Triples
)
obj += gp.quicksum(items[i]["setup"] * y[i, t] for (i, t) in y)

if ALLOW_BACKLOG:
    obj += gp.quicksum(items[i]["b_var"] * b[i, u] for (i, u) in b)

m.setObjective(obj, GRB.MINIMIZE)

# capacity ------------------------------------------------------------------
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

# setup linking -------------------------------------------------------------
for (i, t), y_it in y.items():
    m.addConstr(
        gp.quicksum(x[i, t, u] for u in range(t, min(T, t + items[i]["life"])))
        <= items[i]["M"] * y_it,
        name=f"setupLink_{i}_{t}",
    )


# helper: origins that are still fresh when consumed in period u ------------
def origins(u, life):
    return range(max(0, u - life + 1), u + 1)


# demand balance ------------------------------------------------------------
for i, it in items.items():
    d, L = it["demand"], it["life"]

    # period 0
    if ALLOW_BACKLOG:
        m.addConstr(
            b[i, 0] + gp.quicksum(x[i, t, 0] for t in origins(0, L)) == d[0],
            name=f"demand_{i}_0",
        )
    else:
        m.addConstr(
            gp.quicksum(x[i, t, 0] for t in origins(0, L)) == d[0],
            name=f"demand_{i}_0",
        )

    # periods 1…T-1
    for u in range(1, T):
        if ALLOW_BACKLOG:
            m.addConstr(
                b[i, u] + gp.quicksum(x[i, t, u] for t in origins(u, L))
                == d[u] + b[i, u - 1],
                name=f"demand_{i}_{u}",
            )
        else:  # no backlog ⇒ simple flow-balance
            m.addConstr(
                gp.quicksum(x[i, t, u] for t in origins(u, L)) == d[u],
                name=f"demand_{i}_{u}",
            )

# 3. solve & report ---------------------------------------------------------
m.optimize()

# pretty print orders -------------------------------------------------------
for i, it in items.items():
    print(f"\nItem {i} — orders (period → qty)")
    for t in Periods:
        qty = sum(x[i, t, u].X for u in range(t, min(T, t + it["life"])))
        if qty > 1e-6:
            print(f"  {t:2d} → {qty:4.0f}")

print("\n=================  RESULTS  =================\n")
if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
    print(f"Objective value     : {m.ObjVal:,.2f}")
    print(f"Best bound          : {m.ObjBound:,.2f}")
    print(f"MIP gap             : {m.MIPGap*100:6.2f} %")
    print(f"Total time          : {m.Runtime:,.2f} s")
else:
    print("Solver ended with status:", m.Status)
