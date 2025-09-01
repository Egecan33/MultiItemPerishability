# mip/solver_mip_no_crossing.py
from __future__ import annotations
import time, json
from typing import Dict, List, Tuple
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB


# -------- helpers (same contracts as your existing solver) ----------------
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


# ========================================================================
#                           NO-CROSSING SOLVER
# ========================================================================
def solve_instance(
    instance_path: str | Path = "last_instance.json",
    time_limit: int = 0,
    mip_gap: float = 0.0,
    out_dir: (
        str | Path
    ) = "mip_results_no_crossing",  # different folder to avoid clashes
):
    data = json.loads(Path(instance_path).read_text())
    T = int(data["period"])
    Periods = list(range(T))
    items_raw: Dict[int, dict] = {int(k): v for k, v in data["items"].items()}

    # Production capacity κ_t (accepts "production_capacity" or legacy "manual_capacity")
    prod_cap = data.get("production_capacity")
    if prod_cap is None:
        prod_cap = data.get("manual_capacity")
    prod_cap = (
        _as_len_T_vector(prod_cap, T)
        if prod_cap is not None
        else _cap_global_from_dem(items_raw, T)
    )

    # Optional per-item production cap p_it
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
    if W is not None:
        W = float(W)

    # Lost-sales option (kept for I/O parity with your current solver)
    allow_lost_sales = bool(
        data.get("allow_unmet_demand", False) or data.get("allow_lost_sales", False)
    )
    loss_penalty_global = data.get("lost_sales_penalty", None)
    loss_penalty_factor = float(data.get("lost_sales_penalty_factor", 200.0))

    m = gp.Model("perishable_LEFO_no_crossing")
    m.Params.OutputFlag = 1
    if time_limit:
        m.Params.TimeLimit = int(time_limit)
    if mip_gap:
        m.Params.MIPGap = float(mip_gap)

    # ----------------- Feasible arcs & expiry markers -----------------
    # Gamma[(i,t)] = list of u such that (t,u) feasible; vit = t + m_it
    Gamma: Dict[Tuple[int, int], List[int]] = {}
    Expiry: Dict[Tuple[int, int], int] = {}  # (i,t) -> v_it
    Triples: List[Tuple[int, int, int]] = []  # (i,t,u)
    for i, it in items_raw.items():
        mseq = list(it["shelf_seq"])
        if len(mseq) != T:
            raise ValueError(f"items[{i}]['shelf_seq'] must have length {T}")
        for t in Periods:
            m_it = int(mseq[t])
            v_it = t + m_it
            Expiry[(i, t)] = v_it
            if m_it <= 0:
                Gamma[(i, t)] = []
                continue
            u_max = min(T - 1, v_it - 1)
            us = [u for u in range(t, u_max + 1)]
            Gamma[(i, t)] = us
            for u in us:
                Triples.append((i, t, u))

    # Tight µ_it for setup-linking
    mu: Dict[Tuple[int, int], float] = {}
    for i, it in items_raw.items():
        d = list(it["demand"])
        for t in Periods:
            mu[(i, t)] = float(sum(d[u] for u in Gamma.get((i, t), [])))

    # ---------------------------- Variables ----------------------------
    # Flow on arcs
    X = m.addVars(Triples, vtype=GRB.CONTINUOUS, lb=0.0, name="X")
    # Setups
    Y = m.addVars(
        [(i, t) for i in items_raw for t in Periods], vtype=GRB.BINARY, name="Y"
    )
    # Arc activations (turn on iff arc carries positive flow)
    Z = m.addVars(Triples, vtype=GRB.BINARY, name="Z")
    # Lost sales
    if allow_lost_sales:
        LS = m.addVars(
            [(i, u) for i in items_raw for u in Periods],
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="LS",
        )

    # --------------------- Cost helpers (time-varying) ---------------------
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
        h = items_raw[i]["h"]
        if isinstance(h, list):
            pref = h_pref[i]
            return float(pref[u] - pref[t])
        return float(h) * (u - t)

    def s_at(i: int, t: int) -> float:
        s = items_raw[i]["setup"]
        return float(s[t]) if isinstance(s, list) else float(s)

    # ---------- Lost-sales penalties (safe, auto) ----------
    if allow_lost_sales:
        max_unit_var_cost = 0.0
        for i, t, u in Triples:
            max_unit_var_cost = max(max_unit_var_cost, c_at(i, t) + hsum(i, t, u))
        if max_unit_var_cost <= 0.0:
            max_unit_var_cost = 1.0

        max_setup = 0.0
        for i in items_raw:
            s = items_raw[i]["setup"]
            if isinstance(s, list):
                try:
                    max_setup = max(max_setup, max(float(x) for x in s))
                except ValueError:
                    pass
            else:
                max_setup = max(max_setup, float(s))

        default_loss_penalty = (
            float(loss_penalty_global) if loss_penalty_global is not None else None
        )
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

    # ---------------------------- Objective ----------------------------
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

    # ---------------------------- Constraints ----------------------------

    # (1) Global production capacity κ_t
    for t in Periods:
        m.addConstr(
            gp.quicksum(X[i, t, u] for (i, tt, u) in Triples if tt == t) <= prod_cap[t],
            name=f"prod_cap_{t}",
        )

    # (2) Optional per–item capacity p_it
    if per_item_cap:
        for (i, t), pit in per_item_cap.items():
            if Gamma.get((i, t)):
                m.addConstr(
                    gp.quicksum(X[i, t, u] for u in Gamma[(i, t)]) <= float(pit),
                    name=f"item_cap_{i}_{t}",
                )

    # (3) Setup linking  sum_u X_{i,t,u} ≤ μ_{i,t} Y_{i,t}
    for i, t in Y.keys():
        if Gamma.get((i, t)):
            m.addConstr(
                gp.quicksum(X[i, t, u] for u in Gamma[(i, t)]) <= mu[(i, t)] * Y[i, t],
                name=f"setupLink_{i}_{t}",
            )
        else:
            m.addConstr(Y[i, t] == 0, name=f"setupLink_zero_{i}_{t}")

    # (4) Warehouse capacity (inventory between u and u+1)
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

    # (5) Demand satisfaction (equals demand; soft if lost sales is enabled)
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

    # (6) Arc activation linking  0 ≤ X_{i,t,u} ≤ C_{i,u} Z_{i,t,u}
    # Use C_{i,u} = demand of (i,u) as tight Big-M (valid even with lost sales)
    for i, t, u in Triples:
        Ciu = float(items_raw[i]["demand"][u])
        m.addConstr(X[i, t, u] <= Ciu * Z[i, t, u], name=f"arc_on_{i}_{t}_{u}")

    # (7) No–Crossing: for s<u and v_it < v_i,t'  ⇒  Z_{i,t,s} + Z_{i,t',u} ≤ 1
    # Generate only truly crossable pairs to keep the count modest.
    for i, it in items_raw.items():
        # Pre-collect for each consumption period the feasible origins with their expiry
        arcs_to_u = {u: [] for u in Periods}
        for t in Periods:
            for u in Gamma.get((i, t), []):
                arcs_to_u[u].append((t, Expiry[(i, t)]))  # (origin, expiry)

        for s in Periods:
            if not arcs_to_u[s]:
                continue
            # For speed: sort once by expiry (ascending)
            left = sorted(arcs_to_u[s], key=lambda x: x[1])  # (t, v_t)
            for u in range(s + 1, T):
                if not arcs_to_u[u]:
                    continue
                right = sorted(arcs_to_u[u], key=lambda x: x[1])  # (t', v_t')
                # two-pointer: add only pairs where v_left < v_right
                j = 0
                for tL, vL in left:
                    # advance j until vR > vL (since right sorted asc)
                    while j < len(right) and right[j][1] <= vL:
                        j += 1
                    for k in range(j, len(right)):
                        tR, vR = right[k]
                        # vL < vR and s < u by construction
                        m.addConstr(
                            Z[i, tL, s] + Z[i, tR, u] <= 1,
                            name=f"nocross_{i}_{tL}_{s}__{tR}_{u}",
                        )

    # ---------------------------- Optimize ----------------------------
    m.optimize()

    status = m.Status
    summary = {
        "status": int(status),
        "objective": None,
        "best_bound": None,
        "gap": None,
        "runtime_sec": float(getattr(m, "Runtime", 0.0)),
        "solver_version": "no_crossing_v1",
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

        for i in items_raw:
            orders_txt.append(f"Item {i} — orders (t → qty)")
            for t in Periods:
                qty = sum(X[i, t, u].X for u in Gamma.get((i, t), []) if (i, t, u) in X)
                if qty > 1e-6:
                    orders_txt.append(f"  {t:2d} → {qty:8.3f}")
            if allow_lost_sales:
                for u in Periods:
                    val = LS[i, u].X
                    if val > 1e-6:
                        orders_txt.append(f"  u={u:2d} → LOST {val:8.3f}")
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
        if allow_lost_sales:
            try:
                summary["lost_sales_total"] = float(
                    sum(LS[i, u].X for i in items_raw for u in Periods)
                )
            except Exception:
                pass

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
