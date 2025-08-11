#!/usr/bin/env python3
# ---------------------------------------------------------------
# Perishable lot-sizing – compact MIP (supports variable m_it)
# ---------------------------------------------------------------

import json, gurobipy as gp
from gurobipy import GRB
from pathlib import Path

# ───────────── user switches ───────────────────────────────────
ALLOW_BACKLOG = False  # ← flip to True to allow back-orders
TIME_LIMIT = 3600  # seconds (0 → unlimited)
INSTANCE = "last_instance.json"
# ───────────────────────────────────────────────────────────────

# 1) load instance ----------------------------------------------------------
with open(INSTANCE) as fh:
    data = json.load(fh)

T = int(data["period"])
Periods = range(T)

# capacity: prefer manual; else fallback like generator’s default
cap = data.get("manual_capacity")
if not cap:
    # fallback = sum demands per period + safety buffer (~20% of max period demand)
    cap_raw = [0] * T
    for it in data["items"].values():
        dem = it["demand"]
        for t in Periods:
            cap_raw[t] += dem[t]
    max_cap = max(cap_raw) if cap_raw else 0
    buffer = max(5, int(0.2 * max_cap))
    cap = [c + buffer for c in cap_raw]

# item data (note: variable perishability sequence "shelf_seq")
items = {
    int(k): dict(
        demand=list(it["demand"]),
        setup=float(it["setup"]),
        c_var=float(it["c_var"]),
        h=float(it["h"]),
        b_var=float(it["b_var"]),
        m_seq=list(it["shelf_seq"]),  # m_it per period
        M=sum(it["demand"]),
    )
    for k, it in data["items"].items()
}

# 2) model ------------------------------------------------------------------
m = gp.Model("perishable_LS_compact_mit")
m.Params.OutputFlag = 1
if TIME_LIMIT:
    m.Params.TimeLimit = TIME_LIMIT

# Feasible production triples (i,t,u) with per-period perishability m_it:
# u ranges from t to min(T-1, t + m_it[t] - 1)
Triples = []
for i, it in items.items():
    mseq = it["m_seq"]
    for t in Periods:
        life_t = int(mseq[t])
        u_max = min(T - 1, t + life_t - 1)
        for u in range(t, u_max + 1):
            Triples.append((i, t, u))

# decision variables --------------------------------------------------------
# x[i,t,u]: quantity produced at t and consumed at u
x = m.addVars(Triples, vtype=GRB.INTEGER, lb=0, name="x")

# y[i,t]: setup binary at t (opens production at t)
y = m.addVars([(i, t) for i in items for t in Periods], vtype=GRB.BINARY, name="y")

if ALLOW_BACKLOG:
    # b[i,u]: backlog (end of period u, >=0)
    b = m.addVars(
        [(i, u) for i in items for u in Periods],
        vtype=GRB.INTEGER,
        lb=0,
        name="backlog",
    )
else:
    # dummy zeros to keep code uniform
    b = {(i, u): gp.LinExpr(0) for i in items for u in Periods}

# objective -----------------------------------------------------------------
obj = gp.quicksum(
    (items[i]["c_var"] + items[i]["h"] * (u - t)) * x[i, t, u] for (i, t, u) in Triples
) + gp.quicksum(items[i]["setup"] * y[i, t] for (i, t) in y.keys())

if ALLOW_BACKLOG:
    obj += gp.quicksum(items[i]["b_var"] * b[i, u] for (i, u) in b.keys())

m.setObjective(obj, GRB.MINIMIZE)

# capacity per production period t -----------------------------------------
for t in Periods:
    m.addConstr(
        gp.quicksum(x[i, t, u] for (i, tt, u) in Triples if tt == t) <= cap[t],
        name=f"cap_{t}",
    )

# setup linking (tight big-M by total demand M_i) ---------------------------
for i, t in y.keys():
    # all u that are feasible from (i,t)
    mseq = items[i]["m_seq"]
    life_t = int(mseq[t])
    u_max = min(T - 1, t + life_t - 1)
    m.addConstr(
        gp.quicksum(x[i, t, u] for u in range(t, u_max + 1)) <= items[i]["M"] * y[i, t],
        name=f"setupLink_{i}_{t}",
    )


# helper: feasible origins for (i,u) given m_it -----------------------------
def origins_for(i: int, u: int):
    """All production periods t for item i that can still serve demand at u under m_it."""
    mseq = items[i]["m_seq"]
    # t must satisfy t ≤ u and u ≤ t + m_it[t] − 1
    tmin = 0
    tmax = u
    feasible = []
    for t in range(tmin, tmax + 1):
        if u <= t + int(mseq[t]) - 1:
            feasible.append(t)
    return feasible


# demand balance per item/period -------------------------------------------
for i, it in items.items():
    d = it["demand"]
    # period 0
    if ALLOW_BACKLOG:
        m.addConstr(
            gp.quicksum(x[i, t, 0] for t in origins_for(i, 0)) + b[i, 0] == d[0],
            name=f"demand_{i}_0",
        )
    else:
        m.addConstr(
            gp.quicksum(x[i, t, 0] for t in origins_for(i, 0)) == d[0],
            name=f"demand_{i}_0",
        )
    # periods 1..T-1
    for u in range(1, T):
        if ALLOW_BACKLOG:
            m.addConstr(
                gp.quicksum(x[i, t, u] for t in origins_for(i, u)) + b[i, u]
                == d[u] + b[i, u - 1],
                name=f"demand_{i}_{u}",
            )
        else:
            m.addConstr(
                gp.quicksum(x[i, t, u] for t in origins_for(i, u)) == d[u],
                name=f"demand_{i}_{u}",
            )

# 3) solve & report ---------------------------------------------------------
m.optimize()

# pretty print orders -------------------------------------------------------
for i, it in items.items():
    print(f"\nItem {i} — orders (period → qty)")
    for t in Periods:
        life_t = int(items[i]["m_seq"][t])
        u_max = min(T - 1, t + life_t - 1)
        qty = sum(x[i, t, u].X for u in range(t, u_max + 1) if (i, t, u) in x)
        if qty > 1e-6:
            print(f"  {t:2d} → {qty:4.0f}")

print("\n=================  RESULTS  =================\n")
if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
    print(f"Objective value     : {m.ObjVal:,.2f}")
    print(f"Best bound          : {m.ObjBound:,.2f}")
    # guard for pure LP / no MIP attributes
    gap = getattr(m, "MIPGap", None)
    if gap is not None:
        print(f"MIP gap             : {gap*100:6.2f} %")
    print(f"Total time          : {m.Runtime:,.2f} s")
else:
    print("Solver ended with status:", m.Status)
