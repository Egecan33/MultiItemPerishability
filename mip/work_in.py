# mip/solver_mip_lefo.py
from __future__ import annotations
import time
from typing import Dict, List, Tuple
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

    # ----------------------------
    # Feasible arcs and triples (FIX: build from consumption windows)
    # ----------------------------
    # Gamma[(i,t)] = list of feasible consumption periods u for production at t
    Gamma: Dict[Tuple[int, int], List[int]] = {
        (i, t): [] for i in items_raw for t in Periods
    }
    Triples: List[Tuple[int, int, int]] = []

    for i, it in items_raw.items():
        # Interpret shelf_seq[u] as max allowed age at consumption u (>=0).
        mseq = list(it["shelf_seq"])
        for u in Periods:
            L_u = int(mseq[u])
            if L_u <= 0:
                continue  # nothing can be consumed at u
            # contiguous origin window for consumption at u
            t_min = max(0, u - (L_u - 1))
            for t in range(t_min, u + 1):
                Gamma[(i, t)].append(u)
                Triples.append((i, t, u))

    # Tight μ_it for setup-link: sum of demands that (i,t) can serve
    mu: Dict[Tuple[int, int], float] = {}
    for i, it in items_raw.items():
        d = list(it["demand"])
        for t in Periods:
            mu[(i, t)] = float(sum(d[u] for u in Gamma[(i, t)]))  # 0 if no arcs

    # ----------------------------
    # Variables
    # ----------------------------
    X = m.addVars(Triples, vtype=GRB.CONTINUOUS, lb=0.0, name="X")
    Y = m.addVars(
        [(i, t) for i in items_raw for t in Periods], vtype=GRB.BINARY, name="Y"
    )

    # LEFO permission binaries L_{i,u,a} only for ages present at (i,u)
    L_keys: List[Tuple[int, int, int]] = []
    for i in items_raw:
        for u in Periods:
            ages = sorted(
                {u - t for t in range(0, u + 1) if u in Gamma.get((i, t), [])}
            )
            for a in ages:
                L_keys.append((i, u, a))
    L = m.addVars(L_keys, vtype=GRB.BINARY, name="L")

    # ----------------------------
    # Helpers for time-varying costs
    # ----------------------------
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

    # ----------------------------
    # Objective
    # ----------------------------
    obj = gp.LinExpr()
    for i, t, u in Triples:
        g_itu = c_at(i, t) + hsum(i, t, u)
        obj += g_itu * X[i, t, u]
    for i, t in Y.keys():
        obj += s_at(i, t) * Y[i, t]
    m.setObjective(obj, GRB.MINIMIZE)

    # ----------------------------
    # Constraints
    # ----------------------------

    # (1) global capacity (per production period)
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

    # (3) warehouse capacity (inventory between u and u+1)
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

    # (4) demand balance (now contiguous origin windows by construction)
    for i, it in items_raw.items():
        d = list(it["demand"])
        for u in Periods:
            origins = [t for t in range(0, u + 1) if u in Gamma[(i, t)]]
            m.addConstr(
                gp.quicksum(X[i, t, u] for t in origins) == d[u], name=f"demand_{i}_{u}"
            )

    # (5) LEFO (permission-based): Gate + Monotonicity + Sufficiency
    EPS = 1e-6
    for i, it in items_raw.items():
        d = list(it["demand"])
        for u in Periods:
            Ciu = float(d[u])  # tight gating bound; no-backorder case
            ages = sorted(
                {u - t for t in range(0, u + 1) if u in Gamma.get((i, t), [])}
            )
            if not ages:
                continue

            # (5a) Gate: sum over arcs of age a <= Ciu * L_{i,u,a}
            for a in ages:
                T_a = [
                    t
                    for t in range(0, u + 1)
                    if (u in Gamma.get((i, t), []) and (u - t) == a)
                ]
                if T_a:
                    m.addConstr(
                        gp.quicksum(X[i, t, u] for t in T_a) <= Ciu * L[i, u, a],
                        name=f"lefo_gate_{i}_{u}_{a}",
                    )

            # (5b) Monotonicity: L_{i,u,a} >= L_{i,u,a+1}
            for k in range(len(ages) - 1):
                a = ages[k]
                ap1 = ages[k + 1]
                m.addConstr(L[i, u, a] >= L[i, u, ap1], name=f"lefo_mono_{i}_{u}_{a}")

            # (5c) Sufficiency cut
            if Ciu > 0.0:
                for k in range(len(ages) - 1):
                    a = ages[k]
                    ap1 = ages[k + 1]
                    newer_terms = []
                    for j in ages:
                        if j <= a:
                            T_j = [
                                t
                                for t in range(0, u + 1)
                                if (u in Gamma.get((i, t), []) and (u - t) == j)
                            ]
                            if T_j:
                                newer_terms.append(gp.quicksum(X[i, t, u] for t in T_j))
                    if newer_terms:
                        m.addConstr(
                            Ciu * (1 - L[i, u, ap1])
                            >= gp.quicksum(newer_terms) - Ciu + EPS,
                            name=f"lefo_suff_{i}_{u}_{ap1}",
                        )

    # ----------------------------
    # Optimize and report
    # ----------------------------
    m.optimize()

    status = m.Status
    summary = {
        "status": int(status),
        "objective": None,
        "best_bound": None,
        "gap": None,
        "runtime_sec": float(getattr(m, "Runtime", 0.0)),
        "solver_version": "gurobi_12_0_3",
    }

    try:
        summary["best_bound"] = float(m.ObjBound)
    except Exception:
        pass
    try:
        summary["gap"] = float(m.MIPGap)
    except Exception:
        pass

    orders_txt: List[str] = []

    if m.SolCount and status not in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
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
        try:
            summary["objective"] = float(m.ObjVal)
        except Exception:
            pass
    else:
        try:
            m.computeIIS()
            iis_path = f"iis_{int(time.time())}.ilp"
            m.write(iis_path)
            summary["iis_file"] = iis_path
        except Exception:
            pass

    return summary, orders_txt
