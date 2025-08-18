from __future__ import annotations
from typing import Dict, List, Tuple, Union
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


def _val_at(seq_or_scalar: Union[float, List[float]], t: int) -> float:
    return (
        float(seq_or_scalar[t])
        if isinstance(seq_or_scalar, list)
        else float(seq_or_scalar)
    )


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

    # Feasible arcs and triples with m_{i,t}
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

    # μ_{i,t}
    mu = {}
    for i, it in items_raw.items():
        d = list(it["demand"])
        mseq = list(it["shelf_seq"])
        for t in Periods:
            u_max = min(T - 1, t + int(mseq[t]) - 1)
            mu[(i, t)] = sum(d[u] for u in range(t, u_max + 1))

    # Variables
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

    # Objective with time-varying c_{i,t}, h_{i,t} allowed
    obj = gp.LinExpr()
    for i, t, u in Triples:
        c = items_raw[i]["c_var"]
        h = items_raw[i]["h"]
        g_itu = _val_at(c, t) + _val_at(h, t) * (u - t)
        obj += g_itu * X[i, t, u]

    def s_at(i: int, t: int) -> float:
        return _val_at(items_raw[i]["setup"], t)

    for i, t in Y.keys():
        obj += s_at(i, t) * Y[i, t]
    m.setObjective(obj, GRB.MINIMIZE)

    # Global capacity
    for t in Periods:
        m.addConstr(
            gp.quicksum(X[i, t, u] for (i, tt, u) in Triples if tt == t)
            <= cap_global[t],
            name=f"cap_global_{t}",
        )
    # Setup-link with tight μ
    for i, t in Y.keys():
        m.addConstr(
            gp.quicksum(X[i, t, u] for u in Gamma[(i, t)]) <= mu[(i, t)] * Y[i, t],
            name=f"setupLink_{i}_{t}",
        )
    # Warehouse capacity
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

    # Demand balance (no backorders)
    for i, it in items_raw.items():
        d = list(it["demand"])
        for u in Periods:
            origins = [t for t in range(0, u + 1) if u in Gamma[(i, t)]]
            m.addConstr(
                gp.quicksum(X[i, t, u] for t in origins) == d[u], name=f"demand_{i}_{u}"
            )

    # LEFO
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

    # Reports
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
