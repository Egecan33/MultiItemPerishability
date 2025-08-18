# mip/solver_mip_lefo.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB


def _cap_global_from_dem(items: Dict[int, dict], T: int) -> List[int]:
    cap_raw = [0] * T
    for it in items.values():
        dem = it["demand"]
        for t in range(T):
            cap_raw[t] += dem[t]
    buf = max(5, int(0.2 * max(cap_raw) if cap_raw else 0))
    return [c + buf for c in cap_raw]


def solve_instance(
    instance_path: str | Path = "last_instance.json",
    time_limit: int = 0,
    mip_gap: float = 0.0,
    out_dir: str | Path = "mip_results_lefo",
):
    data = json.loads(Path(instance_path).read_text())
    T = int(data["period"])
    Periods = list(range(T))
    items_raw: Dict[int, dict] = {int(k): v for k, v in data["items"].items()}

    cap_global = data.get("manual_capacity")
    if not cap_global:
        cap_global = _cap_global_from_dem(items_raw, T)

    W = data.get("warehouse_capacity", None)

    m = gp.Model("perishable_LS_compact_LEFO")
    m.Params.OutputFlag = 1
    if time_limit:
        m.Params.TimeLimit = int(time_limit)
    if mip_gap:
        m.Params.MIPGap = float(mip_gap)

    # Feasible arcs and triples
    Gamma: Dict[Tuple[int, int], List[int]] = {}
    Triples: List[Tuple[int, int, int]] = []
    for i, it in items_raw.items():
        mseq = list(it["shelf_seq"])
        for t in Periods:
            u_max = min(T - 1, t + int(mseq[t]) - 1)
            us = [u for u in range(t, u_max + 1)]
            Gamma[(i, t)] = us
            for u in us:
                Triples.append((i, t, u))

    # Tight μ_it
    mu = {}
    for i, it in items_raw.items():
        d = list(it["demand"])
        mseq = list(it["shelf_seq"])
        for t in Periods:
            u_max = min(T - 1, t + int(mseq[t]) - 1)
            mu[(i, t)] = sum(d[u] for u in range(t, u_max + 1))

    # Vars
    X = m.addVars(Triples, vtype=GRB.CONTINUOUS, lb=0.0, name="X")
    Y = m.addVars(
        [(i, t) for i in items_raw for t in Periods], vtype=GRB.BINARY, name="Y"
    )
    S = {}
    for i, it in items_raw.items():
        for u in Periods:
            for tau in range(0, u):
                S[(i, u, tau)] = m.addVar(
                    vtype=GRB.CONTINUOUS, lb=0.0, name=f"S_{i}_{u}_{tau}"
                )

    # Helpers for time-varying costs
    def c_at(i: int, t: int) -> float:
        c = items_raw[i]["c_var"]
        return float(c[t]) if isinstance(c, list) else float(c)

    # prefix sums for h if list
    h_pref: Dict[int, List[float]] = {}
    for i, it in items_raw.items():
        h = it["h"]
        if isinstance(h, list):
            pref = [0.0] * (T + 1)
            for k in range(T):
                pref[k + 1] = pref[k] + float(h[k])
            h_pref[i] = pref

    def hsum(i: int, t: int, u: int) -> float:
        """sum_{tau=t}^{u-1} h_{i,tau}  (0 if u==t)"""
        h = items_raw[i]["h"]
        if isinstance(h, list):
            pref = h_pref[i]
            return float(pref[u] - pref[t])
        return float(h) * (u - t)

    def s_at(i: int, t: int) -> float:
        s = items_raw[i]["setup"]
        return float(s[t]) if isinstance(s, list) else float(s)

    # Objective
    obj = gp.LinExpr()
    for i, t, u in Triples:
        g_itu = c_at(i, t) + hsum(i, t, u)
        obj += g_itu * X[i, t, u]
    for i, t in Y.keys():
        obj += s_at(i, t) * Y[i, t]
    m.setObjective(obj, GRB.MINIMIZE)

    # (1) capacity
    for t in Periods:
        m.addConstr(
            gp.quicksum(X[i, t, u] for (i, tt, u) in Triples if tt == t)
            <= cap_global[t],
            name=f"cap_global_{t}",
        )

    # (2) setup-link
    for i, t in Y.keys():
        m.addConstr(
            gp.quicksum(X[i, t, u] for u in Gamma[(i, t)]) <= mu[(i, t)] * Y[i, t],
            name=f"setupLink_{i}_{t}",
        )

    # (3) warehouse cap
    if W is not None:
        for u in Periods[:-1]:
            inv_u = gp.quicksum(
                X[i, t, w]
                for i in items_raw
                for t in range(0, u + 1)
                for w in Gamma[(i, t)]
                if w > u
            )
            m.addConstr(inv_u <= float(W), name=f"whcap_{u}")

    # (4) demand balance
    for i, it in items_raw.items():
        d = list(it["demand"])
        for u in Periods:
            origins = [t for t in range(0, u + 1) if u in Gamma[(i, t)]]
            m.addConstr(
                gp.quicksum(X[i, t, u] for t in origins) == d[u], name=f"demand_{i}_{u}"
            )

    # (5) LEFO
    for i, it in items_raw.items():
        d = list(it["demand"])
        for u in Periods:
            for tau in range(0, u):
                newer = [t for t in range(tau + 1, u + 1) if u in Gamma[(i, t)]]
                older = [t for t in range(0, tau + 1) if u in Gamma[(i, t)]]
                if newer:
                    m.addConstr(
                        S[(i, u, tau)] >= d[u] - gp.quicksum(X[i, t, u] for t in newer),
                        name=f"lefo_slack_{i}_{u}_{tau}",
                    )
                else:
                    m.addConstr(
                        S[(i, u, tau)] >= d[u], name=f"lefo_slack_{i}_{u}_{tau}"
                    )
                if older:
                    m.addConstr(
                        gp.quicksum(X[i, t, u] for t in older) <= S[(i, u, tau)],
                        name=f"lefo_older_{i}_{u}_{tau}",
                    )

    m.optimize()

    # report
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    orders_txt = []
    for i in items_raw:
        orders_txt.append(f"Item {i} — orders (t → qty)")
        for t in Periods:
            qty = sum(X[i, t, u].X for u in Gamma[(i, t)] if (i, t, u) in X)
            if qty > 1e-6:
                orders_txt.append(f"  {t:2d} → {qty:8.3f}")
        orders_txt.append("")
    (out_dir / "orders.txt").write_text("\n".join(orders_txt), encoding="utf-8")

    summary = {
        "status": int(m.Status),
        "objective": getattr(m, "ObjVal", None),
        "best_bound": getattr(m, "ObjBound", None),
        "gap": getattr(m, "MIPGap", None),
        "runtime_sec": getattr(m, "Runtime", None),
        "n_items": len(items_raw),
        "T": T,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary, orders_txt
