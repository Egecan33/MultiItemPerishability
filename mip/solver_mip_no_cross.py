from __future__ import annotations
import json
import time
from typing import Dict, List, Tuple
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
    out_dir: str | Path = "mip_results",
):

    data = json.loads(Path(instance_path).read_text())
    T = int(data["period"])
    Periods = list(range(T))
    items_raw: Dict[int, dict] = {int(k): v for k, v in data["items"].items()}

    prod_cap = data.get("production_capacity") or data.get("manual_capacity")
    prod_cap = (
        _as_len_T_vector(prod_cap, T)
        if prod_cap is not None
        else _cap_global_from_dem(items_raw, T)
    )

    m = gp.Model("perishable_LEFO")
    m.Params.OutputFlag = 1

    # Set time limit and gap if provided
    if time_limit > 0:
        m.Params.TimeLimit = time_limit
    if mip_gap > 0:
        m.Params.MIPGap = mip_gap

    Gamma: Dict[Tuple[int, int], List[int]] = {}
    Expiry: Dict[Tuple[int, int], int] = {}
    Triples: List[Tuple[int, int, int]] = []

    for i, it in items_raw.items():
        mseq = list(it["shelf_seq"])
        for t in Periods:
            m_it = int(mseq[t])
            v_it = t + m_it
            Expiry[(i, t)] = v_it
            if m_it < 0:
                Gamma[(i, t)] = []
                continue
            u_max = min(T - 1, v_it)
            us = [u for u in range(t, u_max + 1)]
            Gamma[(i, t)] = us
            for u in us:
                Triples.append((i, t, u))

    mu: Dict[Tuple[int, int], float] = {}
    for i, it in items_raw.items():
        d = list(it["demand"])
        for t in Periods:
            mu[(i, t)] = float(sum(d[u] for u in Gamma.get((i, t), [])))

    X = m.addVars(Triples, vtype=GRB.CONTINUOUS, lb=0.0, name="X")
    Y = m.addVars(
        [(i, t) for i in items_raw for t in Periods], vtype=GRB.BINARY, name="Y"
    )
    Z = m.addVars(Triples, vtype=GRB.BINARY, name="Z")

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

    obj = gp.LinExpr()
    for i, t, u in Triples:
        obj += (c_at(i, t) + hsum(i, t, u)) * X[i, t, u]
    for i, t in Y.keys():
        obj += s_at(i, t) * Y[i, t]
    m.setObjective(obj, GRB.MINIMIZE)

    # (C1) Global production capacity
    for t in Periods:
        m.addConstr(
            gp.quicksum(X[i, t, u] for (i, tt, u) in Triples if tt == t) <= prod_cap[t],
            name=f"prod_cap_{t}",
        )

    # (C2) Setup linking
    for i, t in Y.keys():
        if Gamma.get((i, t)):
            m.addConstr(
                gp.quicksum(X[i, t, u] for u in Gamma[(i, t)]) <= mu[(i, t)] * Y[i, t],
                name=f"setupLink_{i}_{t}",
            )
        else:
            m.addConstr(Y[i, t] == 0, name=f"setupLink_zero_{i}_{t}")

    # (C3) Demand satisfaction
    for i, it in items_raw.items():
        d = list(it["demand"])
        for u in Periods:
            origins = [t for t in range(0, u + 1) if u in Gamma.get((i, t), [])]
            m.addConstr(
                gp.quicksum(X[i, t, u] for t in origins) == d[u],
                name=f"demand_{i}_{u}",
            )

    # (C4) Arc activation
    for i, t, u in Triples:
        Ciu = float(items_raw[i]["demand"][u])
        m.addConstr(X[i, t, u] <= Ciu * Z[i, t, u], name=f"arc_on_{i}_{t}_{u}")

    # # (C5) No--crossing (LEFO)
    for i in items_raw:
        prods = [t for t in Periods if Gamma.get((i, t))]
        prods.sort(key=lambda t: Expiry[(i, t)])  # ascending by v_{it}
        for a in range(len(prods)):
            t1 = prods[a]
            v1 = Expiry[(i, t1)]
            for b in range(a + 1, len(prods)):
                t2 = prods[b]
                v2 = Expiry[(i, t2)]
                if v1 >= v2:  # we only care about v1 < v2
                    continue
                for up in Gamma[(i, t2)]:  # u' for t_2
                    for u in [
                        uu for uu in Gamma[(i, t1)] if t2 <= uu <= up - 1
                    ]:  # u for t_1 in [t_2, u'-1]
                        m.addConstr(
                            Z[i, t1, u] + Z[i, t2, up] <= 1,
                            name=f"nocross_{i}_{t1}_{t2}_{u}_{up}",
                        )

    # -------- Solve & report --------
    m.optimize()
    status = m.Status

    summary = {
        "status": int(status),
        "objective": None,
        "best_bound": None,
        "gap": None,
        "runtime_sec": float(getattr(m, "Runtime", 0.0)),
        "solver_version": "lefo_mip_v1",
        "n_items": len(items_raw),
        "T": T,
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

    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if m.SolCount and status not in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
        try:
            summary["objective"] = float(m.ObjVal)
        except:
            pass

        for i in items_raw:
            orders_txt.append(f"Item {i} — orders (t → qty)")
            for t in Periods:
                qty = sum(X[i, t, u].X for u in Gamma.get((i, t), []) if (i, t, u) in X)
                if qty > 1e-6:
                    orders_txt.append(f" {t:2d} → {qty:8.3f}")
            orders_txt.append("")

        (out_path / "orders.txt").write_text("\n".join(orders_txt), encoding="utf-8")
        (out_path / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    else:
        # Try to compute IIS for infeasible models
        try:
            m.computeIIS()
            iis_path = out_path / f"iis_{int(time.time())}.ilp"
            m.write(str(iis_path))
            summary["iis_file"] = str(iis_path)
        except:
            pass

    return summary, orders_txt


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test_instance.json"
    summary, orders = solve_instance(path)
    print(f"\nObjective: {summary.get('objective')}")
