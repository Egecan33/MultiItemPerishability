# mip/solver_mip_lefo.py
from __future__ import annotations
import time
from typing import Dict, List, Tuple
import json
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB


def _cap_global_from_dem(items: Dict[int, dict], T: int) -> List[int]:
    """Fallback κ_t: total demand that *could* be produced, with a small buffer."""
    cap_raw = [0] * T
    for it in items.values():
        dem = it["demand"]
        for t in range(T):
            cap_raw[t] += dem[t]
    buf = max(5, int(0.2 * max(cap_raw) if cap_raw else 0))
    return [c + buf for c in cap_raw]


def _as_len_T_vector(val, T: int) -> List[float]:
    """Accept scalar or list; return length-T list of floats."""
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
    out_dir: str | Path = "mip_results_lefo",
):
    data = json.loads(Path(instance_path).read_text())
    T = int(data["period"])
    Periods = list(range(T))
    items_raw: Dict[int, dict] = {int(k): v for k, v in data["items"].items()}

    # κ_t : production capacity per period (NOT inventory capacity)
    prod_cap = data.get("production_capacity")
    if prod_cap is None:
        # backwards-compat alias
        prod_cap = data.get("manual_capacity")
    prod_cap = (
        _as_len_T_vector(prod_cap, T)
        if prod_cap is not None
        else _cap_global_from_dem(items_raw, T)
    )

    # Optional: per-item production capacity p_it
    # Expect dict of {item_id: list|scalar} or a single list|scalar applied to all items
    item_cap_raw = data.get("item_capacity")  # optional
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

    # Optional warehouse capacity W (inventory between u and u+1)
    W = data.get("warehouse_capacity", None)
    if W is not None:
        W = float(W)

    # --- Lost sales / unmet demand options ---
    allow_lost_sales = bool(
        data.get("allow_unmet_demand", False) or data.get("allow_lost_sales", False)
    )
    loss_penalty_global = data.get("lost_sales_penalty", None)  # scalar or None
    # optional scalar to scale auto penalty
    loss_penalty_factor = float(data.get("lost_sales_penalty_factor", 200.0))

    m = gp.Model("perishable_LS_compact_LEFO")
    m.Params.OutputFlag = 1
    if time_limit:
        m.Params.TimeLimit = int(time_limit)
    if mip_gap:
        m.Params.MIPGap = float(mip_gap)

    # ----------------------------
    # Feasible arcs and triples (production-based shelf lives m_it)
    # ----------------------------
    # Interpret items_raw[i]["shelf_seq"][t] as m_{it} (shelf life of a lot produced at t)
    Gamma: Dict[Tuple[int, int], List[int]] = {}  # (i,t) -> list of u
    Triples: List[Tuple[int, int, int]] = []
    for i, it in items_raw.items():
        mseq = list(it["shelf_seq"])
        if len(mseq) != T:
            raise ValueError(f"items[{i}]['shelf_seq'] must have length {T}")
        for t in Periods:
            m_it = int(mseq[t])
            if m_it <= 0:
                Gamma[(i, t)] = []
                continue
            # lot produced at t can be consumed at u in [t, t + m_it - 1], clipped to horizon
            u_max = min(T - 1, t + m_it - 1)
            us = [u for u in range(t, u_max + 1)]
            Gamma[(i, t)] = us
            for u in us:
                Triples.append((i, t, u))

    # Tight µ_it for setup linking: sum of demands that (i,t) can serve
    mu: Dict[Tuple[int, int], float] = {}
    for i, it in items_raw.items():
        d = list(it["demand"])
        for t in Periods:
            mu[(i, t)] = float(sum(d[u] for u in Gamma.get((i, t), [])))  # 0 if no arcs

    # ----------------------------
    # Expiry groups per (i,u) for Last-Expiry-First
    # ----------------------------
    # For each (i,u), collect origins t in Gamma with expiry e_{itu} = t + m_{it}
    # Group by distinct expiry dates; sort groups from latest to earliest
    expiry_groups: Dict[Tuple[int, int], List[List[int]]] = (
        {}
    )  # (i,u) -> list of groups [t,...]
    for i, it in items_raw.items():
        mseq = list(it["shelf_seq"])
        for u in Periods:
            origins = [t for t in range(0, u + 1) if u in Gamma.get((i, t), [])]
            if not origins:
                continue
            groups_by_expiry: Dict[int, List[int]] = {}
            for t in origins:
                e_itu = t + int(mseq[t])
                groups_by_expiry.setdefault(e_itu, []).append(t)
            # sort expiry dates descending (latest first)
            ordered_expiries = sorted(groups_by_expiry.keys(), reverse=True)
            expiry_groups[(i, u)] = [groups_by_expiry[e] for e in ordered_expiries]

    # ----------------------------
    # Variables
    # ----------------------------
    X = m.addVars(Triples, vtype=GRB.CONTINUOUS, lb=0.0, name="X")
    Y = m.addVars(
        [(i, t) for i in items_raw for t in Periods], vtype=GRB.BINARY, name="Y"
    )

    # LEFO permission binaries L_{i,u,g} only for groups present at (i,u)
    L_keys: List[Tuple[int, int, int]] = []
    for (i, u), groups in expiry_groups.items():
        for g in range(len(groups)):
            L_keys.append((i, u, g))
    L = m.addVars(L_keys, vtype=GRB.BINARY, name="L")

    # Lost sales variables
    if allow_lost_sales:
        LS = m.addVars(
            [(i, u) for i in items_raw for u in Periods],
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="LS",
        )

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

    # ---------- Auto-size lost-sales penalties (safe Big-M) ----------
    if allow_lost_sales:
        # Max variable unit cost (produce at t and ship to u)
        max_unit_var_cost = 0.0
        for i, t, u in Triples:
            max_unit_var_cost = max(max_unit_var_cost, c_at(i, t) + hsum(i, t, u))
        if max_unit_var_cost <= 0.0:
            max_unit_var_cost = 1.0

        # Max setup cost (worst-case "setup per unit" effect)
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

        # Base default
        default_loss_penalty = (
            loss_penalty_global if loss_penalty_global is not None else None
        )
        if default_loss_penalty is None:
            base = max_unit_var_cost + max_setup
            # at least 10× var cost, scaled by factor, and capped for numerics
            default_loss_penalty = max(
                10.0 * max_unit_var_cost, loss_penalty_factor * base
            )
            default_loss_penalty = float(min(default_loss_penalty + 1.0, 1e9))

        # allow per-item override via items_raw[i]["lost_sales_penalty"] (scalar|list)
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

    # ----------------------------
    # Objective
    # ----------------------------
    obj = gp.LinExpr()
    for i, t, u in Triples:
        g_itu = c_at(i, t) + hsum(i, t, u)
        obj += g_itu * X[i, t, u]
    for i, t in Y.keys():
        obj += s_at(i, t) * Y[i, t]
    if allow_lost_sales:
        for i in items_raw:
            for u in Periods:
                obj += loss_pen[(i, u)] * LS[i, u]
    m.setObjective(obj, GRB.MINIMIZE)

    # ----------------------------
    # Constraints
    # ----------------------------

    # (1a) Production capacity per period κ_t  (NOT inventory capacity)
    for t in Periods:
        m.addConstr(
            gp.quicksum(X[i, t, u] for (i, tt, u) in Triples if tt == t) <= prod_cap[t],
            name=f"prod_cap_{t}",
        )

    # (1b) Optional per–item production cap p_it
    if per_item_cap:
        for (i, t), pit in per_item_cap.items():
            if Gamma.get((i, t)):
                m.addConstr(
                    gp.quicksum(X[i, t, u] for u in Gamma[(i, t)]) <= float(pit),
                    name=f"item_cap_{i}_{t}",
                )

    # (1c) Setup linking
    for i, t in Y.keys():
        if Gamma.get((i, t)):
            m.addConstr(
                gp.quicksum(X[i, t, u] for u in Gamma[(i, t)]) <= mu[(i, t)] * Y[i, t],
                name=f"setupLink_{i}_{t}",
            )
        else:
            # no arcs -> force Y[i,t] = 0 to tighten
            m.addConstr(Y[i, t] == 0, name=f"setupLink_zero_{i}_{t}")

    # (optional) Warehouse capacity: inventory carried from u to u+1
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

    # (1d) Demand satisfaction (soft if lost sales)
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

    # Last-Expiry-First (group-based): gate + monotonicity + sufficiency
    EPS = 1e-6
    for (i, u), groups in expiry_groups.items():
        Ciu = float(items_raw[i]["demand"][u])
        G = len(groups)
        # (a) Gate by expiry group
        for g in range(G):
            T_g = groups[g]  # list of t in group g
            if T_g:
                m.addConstr(
                    gp.quicksum(X[i, t, u] for t in T_g) <= Ciu * L[i, u, g],
                    name=f"lefo_gate_{i}_{u}_{g}",
                )
        # (b) Monotonicity
        for g in range(G - 1):
            m.addConstr(L[i, u, g] >= L[i, u, g + 1], name=f"lefo_mono_{i}_{u}_{g}")
        # (c) Sufficiency (open next group only after earlier groups cover *served* demand)
        if G >= 2:
            cum_terms = []
            for g in range(G - 1):
                T_g = groups[g]
                if T_g:
                    cum_terms.append(gp.quicksum(X[i, t, u] for t in T_g))
                if cum_terms:
                    cum_sum = gp.quicksum(cum_terms)
                    if allow_lost_sales:
                        # cum + LS >= demand * L[g+1] + EPS
                        m.addConstr(
                            cum_sum + LS[i, u] >= Ciu * L[i, u, g + 1] + EPS,
                            name=f"lefo_suff_soft_{i}_{u}_{g+1}",
                        )
                    else:
                        # Classic (without lost sales)
                        # Ciu*(1 - L[g+1]) >= cum - Ciu + EPS
                        m.addConstr(
                            Ciu * (1 - L[i, u, g + 1]) >= cum_sum - Ciu + EPS,
                            name=f"lefo_suff_{i}_{u}_{g+1}",
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

    # best bound, gap if available
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
                # print lost sales per period (if any)
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
        # Infeasible or no incumbent: try to write an IIS for debugging
        try:
            m.computeIIS()
            iis_path = f"iis_{int(time.time())}.ilp"
            m.write(iis_path)
            summary["iis_file"] = iis_path
        except Exception:
            pass

    return summary, orders_txt
