from __future__ import annotations
import time, json
from typing import Dict, List, Tuple
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB


# ---------------- helpers ----------------
def _cap_global_from_dem(items: Dict[int, dict], T: int) -> List[int]:
    cap_raw = [0] * T
    for it in items.values():
        dem = it["demand"]
        for t in range(T):
            cap_raw[t] += dem[t]
    buf = max(5, int(0.2 * max(cap_raw) if cap_raw else 0))
    return [c + buf for c in cap_raw]


def _as_len_T_vector(val, T: int) -> List[float]:
    if val is None:
        return []
    if isinstance(val, (int, float)):
        return [float(val)] * T
    if isinstance(val, list):
        if len(val) != T:
            raise ValueError(f"Expected length-{T} list, got {len(val)}")
        return [float(x) for x in val]
    raise TypeError("Capacity must be a number or a list")


def solve_instance(
    instance_path: str | Path = "last_instance.json",
    time_limit: int = 0,
    mip_gap: float = 0.0,
    out_dir: str | Path = "mip_results_no_crossing",
):
    """Perishable lot-sizing with LEFO (pairwise C5), feature-parity with the other solver."""
    data = json.loads(Path(instance_path).read_text())
    T = int(data["period"])
    Periods = list(range(T))
    items_raw: Dict[int, dict] = {int(k): v for k, v in data["items"].items()}
    # κ_t (global production capacity)
    prod_cap = data.get("production_capacity") or data.get("manual_capacity")
    prod_cap = (
        _as_len_T_vector(prod_cap, T)
        if prod_cap is not None
        else _cap_global_from_dem(items_raw, T)
    )
    # Optional: per-item production caps p_it
    item_cap_raw = data.get("item_capacity")
    per_item_cap: Dict[Tuple[int, int], float] = {}
    if item_cap_raw is not None:
        if isinstance(item_cap_raw, dict):
            for i, cap in item_cap_raw.items():
                vec = _as_len_T_vector(cap, T)
                for t in Periods:
                    per_item_cap[(int(i), t)] = vec[t]
        else:
            vec = _as_len_T_vector(item_cap_raw, T)
            for i in items_raw:
                for t in Periods:
                    per_item_cap[(i, t)] = vec[t]
    # Optional warehouse capacity (inventory between u and u+1)
    W = data.get("warehouse_capacity", None)
    W = float(W) if W is not None else None
    # Lost sales
    allow_lost_sales = bool(
        data.get("allow_unmet_demand", False) or data.get("allow_lost_sales", False)
    )
    loss_penalty_global = data.get("lost_sales_penalty", None)
    loss_penalty_factor = float(data.get("lost_sales_penalty_factor", 200.0))
    m = gp.Model("perishable_LEFO_no_crossing_no_shelf")
    m.Params.OutputFlag = 1
    if time_limit:
        m.Params.TimeLimit = int(time_limit)
    if mip_gap:
        m.Params.MIPGap = float(mip_gap)

    # -------- Feasible arcs (NO SHELF LIVES) --------
    # We IGNORE items[i]['shelf_seq'] entirely and allow production at time t
    # to satisfy ANY future demand period u >= t (full-horizon arcs).
    Gamma: Dict[Tuple[int, int], List[int]] = {}
    Triples: List[Tuple[int, int, int]] = []

    for i, it in items_raw.items():
        for t in Periods:
            us = list(range(t, T))  # no expiry → full horizon
            Gamma[(i, t)] = us
            for u in us:
                Triples.append((i, t, u))
    # μ_it : tight setup-linking upper bound
    mu: Dict[Tuple[int, int], float] = {}
    for i, it in items_raw.items():
        d = list(it["demand"])
        for t in Periods:
            mu[(i, t)] = float(sum(d[u] for u in Gamma.get((i, t), [])))
    # -------- Variables --------
    X = m.addVars(Triples, vtype=GRB.CONTINUOUS, lb=0.0, name="X")
    Y = m.addVars(
        [(i, t) for i in items_raw for t in Periods], vtype=GRB.BINARY, name="Y"
    )
    Z = m.addVars(Triples, vtype=GRB.BINARY, name="Z")
    if allow_lost_sales:
        LS = m.addVars(
            [(i, u) for i in items_raw for u in Periods],
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="LS",
        )

    # -------- Costs --------
    def c_at(i: int, t: int) -> float:
        c = items_raw[i]["c_var"]
        return float(c[t]) if isinstance(c, list) else float(c)

    h_pref: Dict[int, List[float]] = {}
    for i, it in items_raw.items():
        h = it["h"]
        if isinstance(h, list):
            pref = [0.0] * (T + 1)
            for k in range(T):
                pref[k + 1] = pref[k] + float(h[k])
            h_pref[i] = pref

    def hsum(i: int, t: int, u: int) -> float:
        h = items_raw[i]["h"]
        if isinstance(h, list):
            pref = h_pref[i]
            return float(pref[u] - pref[t])
        return float(h) * (u - t)

    def s_at(i: int, t: int) -> float:
        s = items_raw[i]["setup"]
        return float(s[t]) if isinstance(s, list) else float(s)

    # Auto-size LS penalties (if enabled)
    if allow_lost_sales:
        max_unit_var_cost = 0.0
        for i, t, u in Triples:
            max_unit_var_cost = max(max_unit_var_cost, c_at(i, t) + hsum(i, t, u))
        if max_unit_var_cost <= 0.0:
            max_unit_var_cost = 1.0
        max_setup = 0.0
        for i in items_raw:
            s = items_raw[i]["setup"]
            max_setup = max(
                max_setup, max(map(float, s)) if isinstance(s, list) else float(s)
            )
        default_loss_penalty = loss_penalty_global
        if default_loss_penalty is None:
            base = max_unit_var_cost + max_setup
            default_loss_penalty = max(
                10.0 * max_unit_var_cost, loss_penalty_factor * base
            )
            default_loss_penalty = float(min(default_loss_penalty + 1.0, 1e9))
        loss_pen = {}
        for i, it in items_raw.items():
            lp = it.get("lost_sales_penalty", None)
            if lp is not None:
                vec = _as_len_T_vector(lp, T)
                for u in Periods:
                    loss_pen[(i, u)] = float(vec[u])
            else:
                for u in Periods:
                    loss_pen[(i, u)] = float(default_loss_penalty)
    # -------- Objective --------
    obj = gp.LinExpr()
    for i, t, u in Triples:
        obj += (c_at(i, t) + hsum(i, t, u)) * X[i, t, u]
    for i, t in Y.keys():
        obj += s_at(i, t) * Y[i, t]
    if allow_lost_sales:
        for i in items_raw:
            for u in Periods:
                obj += loss_pen[(i, u)] * LS[i, u]
    m.setObjective(obj, GRB.MINIMIZE)
    # -------- Constraints --------
    # (C1) Global production capacity
    for t in Periods:
        m.addConstr(
            gp.quicksum(X[i, t, u] for (i, tt, u) in Triples if tt == t) <= prod_cap[t],
            name=f"prod_cap_{t}",
        )
    # (C1b) Per-item cap
    if per_item_cap:
        for (i, t), pit in per_item_cap.items():
            if Gamma.get((i, t)):
                m.addConstr(
                    gp.quicksum(X[i, t, u] for u in Gamma[(i, t)]) <= float(pit),
                    name=f"item_cap_{i}_{t}",
                )
    # (C1c) Warehouse cap (inventory carried to u+1)
    if W is not None:
        for u in Periods[:-1]:
            inv_u = gp.quicksum(
                X[i, t, w]
                for i in items_raw
                for t in range(0, u + 1)
                for w in Gamma.get((i, t), [])
                if w > u
            )
            m.addConstr(inv_u <= W, name=f"whcap_{u}")
    # (C2) Setup linking
    for i, t in Y.keys():
        if Gamma.get((i, t)):
            m.addConstr(
                gp.quicksum(X[i, t, u] for u in Gamma[(i, t)]) <= mu[(i, t)] * Y[i, t],
                name=f"setupLink_{i}_{t}",
            )
        else:
            m.addConstr(Y[i, t] == 0, name=f"setupLink_zero_{i}_{t}")
    # (C3) Demand satisfaction (soft if LS)
    for i, it in items_raw.items():
        d = list(it["demand"])
        for u in Periods:
            origins = [t for t in range(0, u + 1) if u in Gamma.get((i, t), [])]
            if allow_lost_sales:
                m.addConstr(
                    gp.quicksum(X[i, t, u] for t in origins) + LS[i, u] == d[u],
                    name=f"demand_{i}_{u}",
                )
            else:
                m.addConstr(
                    gp.quicksum(X[i, t, u] for t in origins) == d[u],
                    name=f"demand_{i}_{u}",
                )
    # (C4) Arc activation
    for i, t, u in Triples:
        Ciu = float(items_raw[i]["demand"][u])
        m.addConstr(X[i, t, u] <= Ciu * Z[i, t, u], name=f"arc_on_{i}_{t}_{u}")

    # (C5) No--crossing (LEFO) not needed in this version.

    # -------- Solve & report --------
    m.optimize()
    status = m.Status
    summary = {
        "status": int(status),
        "objective": None,
        "best_bound": None,
        "gap": None,
        "runtime_sec": float(getattr(m, "Runtime", 0.0)),
        "solver_version": "no_shelf_v1_no_shelf",
    }
    try:
        summary["best_bound"] = float(m.ObjBound)
    except:
        pass
    try:
        summary["gap"] = float(m.MIPGap)
    except:
        pass
    orders_txt: List[str] = []
    if m.SolCount and status not in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in items_raw:
            orders_txt.append(f"Item {i} — orders (t → qty)")
            for t in Periods:
                qty = sum(X[i, t, u].X for u in Gamma.get((i, t), []) if (i, t, u) in X)
                if qty > 1e-6:
                    orders_txt.append(f" {t:2d} → {qty:8.3f}")
            if allow_lost_sales:
                for u in Periods:
                    val = LS[i, u].X
                    if val > 1e-6:
                        orders_txt.append(f" u={u:2d} → LOST {val:8.3f}")
            orders_txt.append("")
        (out_dir / "orders.txt").write_text("\n".join(orders_txt), encoding="utf-8")
        summary.update(
            {
                "objective": getattr(m, "ObjVal", None),
                "best_bound": getattr(m, "ObjBound", None),
                "gap": getattr(m, "MIPGap", None),
                "runtime_sec": getattr(m, "Runtime", None),
                "n_items": len(items_raw),
                "T": T,
            }
        )
        try:
            summary["objective"] = float(m.ObjVal)
        except:
            pass
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    else:
        try:
            m.computeIIS()
            iis_path = f"iis_{int(time.time())}.ilp"
            m.write(iis_path)
            summary["iis_file"] = iis_path
        except:
            pass
    return summary, orders_txt
