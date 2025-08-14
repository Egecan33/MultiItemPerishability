#!/usr/bin/env python3
# ---------------------------------------------------------------
# Perishable lot-sizing – compact MIP (supports variable m_it)
# + cap_{i,t}, s_{i,t} (time-varying allowed), and fixed
#   warehouse holding capacity
# ---------------------------------------------------------------

import json, csv, shutil
from pathlib import Path
from datetime import datetime
import gurobipy as gp
from gurobipy import GRB

# ───────────── user switches ───────────────────────────────────
ALLOW_BACKLOG = False  # ← flip to True to allow back-orders
TIME_LIMIT = 3600  # seconds (0 → unlimited)
INSTANCE = "last_instance.json"
# ───────────────────────────────────────────────────────────────

# 1) load instance ----------------------------------------------------------
with open(INSTANCE, "r", encoding="utf-8") as fh:
    data = json.load(fh)

T = int(data["period"])
Periods = range(T)

# global capacity per period (prefer manual; else compute fallback)
cap_global = data.get("manual_capacity")
if not cap_global:
    # fallback = sum of demands per period + safety buffer (~20% of max period demand)
    cap_raw = [0] * T
    for it in data["items"].values():
        dem = it["demand"]
        for t in Periods:
            cap_raw[t] += dem[t]
    max_cap = max(cap_raw) if cap_raw else 0
    buffer = max(5, int(0.2 * max_cap))
    cap_global = [c + buffer for c in cap_raw]

# fixed warehouse holding capacity (optional). If None → no limit.
warehouse_capacity = data.get("warehouse_capacity", None)

# item data (variable perishability "shelf_seq", optional cap_seq, optional time-varying setup)
items = {}
for k, it in data["items"].items():
    i = int(k)
    demand = list(it["demand"])
    m_seq = list(it["shelf_seq"])  # m_{i,t}

    setup_raw = it["setup"]
    if isinstance(setup_raw, list):
        setup_seq = [float(v) for v in setup_raw]
        setup_scalar = None
    else:
        setup_scalar = float(setup_raw)
        setup_seq = None

    cap_seq = it.get("cap_seq", None)
    if cap_seq is not None:
        cap_seq = [float(v) for v in cap_seq]

    items[i] = dict(
        demand=demand,
        setup=setup_scalar,  # scalar (if provided)
        setup_seq=setup_seq,  # list length T (if provided)
        c_var=float(it["c_var"]),
        h=float(it["h"]),
        b_var=float(it["b_var"]),
        m_seq=m_seq,  # list length T
        cap_seq=cap_seq,  # optional list length T
        M=sum(demand),
    )


def setup_at(i: int, t: int) -> float:
    """Return s_{i,t} (time-varying) or scalar s_i if sequence not given."""
    seq = items[i].get("setup_seq")
    if seq is not None:
        return float(seq[t])
    s = items[i].get("setup")
    return float(s) if s is not None else 0.0


# 2) model ------------------------------------------------------------------
m = gp.Model("perishable_LS_compact_mit_cap_it_s_it_whcap")
m.Params.OutputFlag = 1
if TIME_LIMIT:
    m.Params.TimeLimit = TIME_LIMIT

# Adjust Gurobi parameters to start monitoring after hitting a 1% gap
m.Params.ImproveStartGap = 0.0002  # Terminate if gap improvement is less than 0.02%
m.Params.ImproveStartNodes = 0  # Start monitoring after reaching a 1% gap
m.Params.ImproveStartTime = 60  # Reset timer if improvement occurs

m.Params.MIPGap = 0.01  # Ensure the gap is below 1%

# Feasible production triples (i,t,u) with per-period perishability m_it:
# u ranges from t to min(T-1, t + m_it[t] - 1)
Triples = []
Gamma = {}  # Gamma[(i,t)] = feasible consumption periods u for production at (i,t)
for i, it in items.items():
    mseq = it["m_seq"]
    for t in Periods:
        life_t = int(mseq[t])
        u_max = min(T - 1, t + life_t - 1)
        gu = list(range(t, u_max + 1))
        Gamma[(i, t)] = gu
        for u in gu:
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
    # dummy zeros to keep code uniform downstream
    b = {(i, u): gp.LinExpr(0) for i in items for u in Periods}

# objective -----------------------------------------------------------------
obj = gp.quicksum(
    (items[i]["c_var"] + items[i]["h"] * (u - t)) * x[i, t, u] for (i, t, u) in Triples
) + gp.quicksum(setup_at(i, t) * y[i, t] for (i, t) in y.keys())

if ALLOW_BACKLOG:
    obj += gp.quicksum(items[i]["b_var"] * b[i, u] for (i, u) in b.keys())

m.setObjective(obj, GRB.MINIMIZE)

# (A) global capacity per production period t -------------------------------
if cap_global is not None:
    for t in Periods:
        m.addConstr(
            gp.quicksum(x[i, t, u] for (i, tt, u) in Triples if tt == t)
            <= cap_global[t],
            name=f"cap_global_{t}",
        )

# (B) per-item per-time capacity cap_{i,t} (optional; enforced if provided) -
for i, it in items.items():
    cap_seq = it.get("cap_seq")
    if cap_seq is None:
        continue
    for t in Periods:
        m.addConstr(
            gp.quicksum(x[i, t, u] for u in Gamma[(i, t)]) <= cap_seq[t],
            name=f"cap_item_{i}_{t}",
        )

# (C) setup linking (tight big-M by total demand M_i) -----------------------
for i, t in y.keys():
    m.addConstr(
        gp.quicksum(x[i, t, u] for u in Gamma[(i, t)]) <= items[i]["M"] * y[i, t],
        name=f"setupLink_{i}_{t}",
    )


# helper: feasible origins for (i,u) given m_it -----------------------------
def origins_for(i: int, u: int):
    """All production periods t for item i that can still serve demand at u under m_it."""
    mseq = items[i]["m_seq"]
    feasible = []
    for t in range(0, u + 1):
        if u <= t + int(mseq[t]) - 1:
            feasible.append(t)
    return feasible


# (D) warehouse holding capacity (inventory limit, optional) ----------------
# Inventory at end of period u equals all units produced up to u that will be consumed after u.
# inv[u] = sum_{i} sum_{t=0..u} sum_{w in Gamma(i,t), w>u} x[i,t,w] ≤ W
if warehouse_capacity is not None:
    for u in Periods[
        :-1
    ]:  # at u = T-1 inventory is always zero (no future consumption)
        inv_expr = gp.quicksum(
            x[i, t, w]
            for i in items
            for t in range(0, u + 1)
            for w in Gamma[(i, t)]
            if w > u
        )
        m.addConstr(inv_expr <= float(warehouse_capacity), name=f"whcap_end_{u}")

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

# 3) solve & console report -------------------------------------------------
m.optimize()

for i, it in items.items():
    print(f"\nItem {i} — orders (period → qty)")
    for t in Periods:
        qty = sum(x[i, t, u].X for u in Gamma[(i, t)] if (i, t, u) in x)
        if qty > 1e-6:
            print(f"  {t:2d} → {qty:4.0f}")

print("\n=================  RESULTS  =================\n")
if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
    print(f"Objective value     : {m.ObjVal:,.2f}")
    print(f"Best bound          : {m.ObjBound:,.2f}")
    gap = getattr(m, "MIPGap", None)
    if gap is not None:
        print(f"MIP gap             : {gap*100:6.2f} %")
    print(f"Total time          : {m.Runtime:,.2f} s")
else:
    print("Solver ended with status:", m.Status)

# 4) artifacts (JSON/CSV/TXT/LP) -------------------------------------------
run_id = f"{Path(INSTANCE).stem}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
out_dir = Path("mip_results") / run_id
out_dir.mkdir(parents=True, exist_ok=True)


def _status_name(st: int) -> str:
    return {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }.get(st, str(st))


def _val(v):
    try:
        return float(v.X)
    except Exception:
        return 0.0


# (a) summary.json
summary = {
    "instance": Path(INSTANCE).name,
    "run_id": run_id,
    "allow_backlog": ALLOW_BACKLOG,
    "time_limit_sec": TIME_LIMIT,
    "status_code": int(m.Status),
    "status_name": _status_name(m.Status),
    "objective": getattr(m, "ObjVal", None),
    "best_bound": getattr(m, "ObjBound", None),
    "mip_gap": getattr(m, "MIPGap", None),
    "runtime_sec": getattr(m, "Runtime", None),
    "sol_count": getattr(m, "SolCount", None),
    "n_items": len(items),
    "n_periods": T,
    "n_arcs": len(Triples),
    "warehouse_capacity": warehouse_capacity,
    "has_item_cap_seq": any(items[i]["cap_seq"] is not None for i in items),
    "has_setup_seq": any(items[i]["setup_seq"] is not None for i in items),
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

# (b) capacity.csv (uses cap_global)
with open(out_dir / "capacity.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["t", "cap_global_t"])
    for t, c in enumerate(cap_global):
        w.writerow([t, c])

# (c) model.lp (optional)
try:
    m.write(str(out_dir / "model.lp"))
except Exception:
    pass

# (d) metrics.txt (plain text with runtime, objective, gap)
obj_txt = (
    f"{getattr(m, 'ObjVal', float('nan')):,.6f}" if getattr(m, "SolCount", 0) else "N/A"
)
bnd_txt = (
    f"{getattr(m, 'ObjBound', float('nan')):,.6f}"
    if getattr(m, "ObjBound", None) is not None
    else "N/A"
)
gap_attr = getattr(m, "MIPGap", None)
gap_txt = f"{gap_attr*100:.4f} %" if gap_attr is not None else "N/A"
rt_txt = f"{getattr(m, 'Runtime', float('nan')):,.3f} s"

(out_dir / "metrics.txt").write_text(
    "\n".join(
        [
            f"Status     : {_status_name(m.Status)}",
            f"Objective  : {obj_txt}",
            f"Best bound : {bnd_txt}",
            f"MIP gap    : {gap_txt}",
            f"Runtime    : {rt_txt}",
        ]
    ),
    encoding="utf-8",
)

# If no incumbent solution, stop here after summary/metrics/capacity/model
if not getattr(m, "SolCount", 0):
    print(f"Artifacts saved to: {out_dir.resolve()}")
    raise SystemExit

# (e) x_nonzero.csv
with open(out_dir / "x_nonzero.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["i", "t", "u", "x"])
    for i, t, u in Triples:
        val = _val(x[i, t, u])
        if abs(val) > 1e-9:
            w.writerow([i, t, u, f"{val:.6f}"])

# (f) y_setups.csv
with open(out_dir / "y_setups.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["i", "t", "y"])
    for i, t in y.keys():
        w.writerow([i, t, int(round(_val(y[i, t])))])

# (g) backlog_nonzero.csv (if backlog enabled)
if ALLOW_BACKLOG:
    with open(out_dir / "backlog_nonzero.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i", "u", "backlog"])
        for i, u in b.keys():
            val = _val(b[i, u])
            if abs(val) > 1e-9:
                w.writerow([i, u, f"{val:.6f}"])

# (h) human-friendly orders
lines = []
for i in items:
    lines.append(f"Item {i} — orders (period → qty)")
    for t in Periods:
        qty = sum(_val(x[i, t, u]) for u in Gamma[(i, t)] if (i, t, u) in x)
        if qty > 1e-6:
            lines.append(f"  {t:2d} → {qty:4.0f}")
    lines.append("")
(out_dir / "orders.txt").write_text("\n".join(lines), encoding="utf-8")

# (i) copy instance JSON into results folder
shutil.copy(INSTANCE, out_dir / "last_instance.json")

print(f"Artifacts saved to: {out_dir.resolve()}")
