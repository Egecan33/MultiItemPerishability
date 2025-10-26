# lefo_bp_solver_fast.py
# --------------------------------------------------------------------------------------
# Branch-and-Price solver for perishable lot-sizing with LEFO (no-crossing) blocks.
# Based on the CG solver, extended to full B&P for exactness.
# Starts with CG at root, gets initial UB via MIP over pool.
# Then branches on aggregated Y_it (setup for item i at t).
# At each node, adjusts lambda bounds, runs CG with modified pricing (MIP for branched items).
# Solves MIP over pool at each node to update incumbent.
# Continues until tree explored or time limit.
# I/O same as other solvers: returns (summary, orders).
# CLI unchanged, with extra optional flag: --outsource_unit_cost
#
# Improvements applied:
# A) Add master-side binary Ȳ_{i,t} with linking constraints sum_p y_{i,p,t} λ_{i,p} = Ȳ_{i,t}
#    and columns contribute to those rows. Pricing reduced costs now include -γ_{i,t}·Y_{i,t}.
# B) Keep λ continuous in the MIP phase (incumbent search) — integrality is carried by Ȳ.
# C) Branch directly on Ȳ_{i,t}; optional no-good cuts support (disabled by default).
# D)branch-on-set variant (pair equality vs XOR) when two fractional Ȳ exist.
# --------------------------------------------------------------------------------------

from __future__ import annotations
import time, json, math, random, sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Set
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

EPS = 1e-9
RC_EPS = 1e-6  # Tuned: stricter for adding columns


@dataclass
class ColumnPlan:
    item: int
    plan_id: int
    cost: float
    prod_by_t: List[float]  # production qty at period t
    inv_end_by_u: List[float]  # inventory at end of u (0..T-2)
    flows: List[Tuple[int, int, float]]  # (t,u,qty)
    setups: List[int]  # t with y_values[t] > EPS
    lost_sales_by_u: List[float]  # per-period lost sales
    y_values: List[float]  # Y[t] values from subproblem (0/1 since binary)


@dataclass
class Node:
    id: int
    parent: Optional["Node"]
    branches: Dict[int, Dict[int, int]]  # i -> {t: val (0/1), ...}
    # === NEW: Ryan–Foster style restrictions stored symbolically for this node
    rf_rules: List[Tuple[str, int, int, int]] = None  # list of ("eq"/"xor", i, t1, t2)
    lp_bound: Optional[float] = None


# ---------------- helpers ----------------


def _log(msg: str, flush: bool = True):
    print(msg, file=sys.stdout, flush=flush)


def _as_len_T_vector(val, T: int) -> List[float]:
    if val is None:
        return []
    if isinstance(val, (int, float)):
        return [float(val)] * T
    if isinstance(val, list):
        if len(val) != T:
            raise ValueError(f"Expected length-{T} list, got {len(val)}")
        return [float(x) for x in val]
    raise TypeError("Expected number or length-T list")


def _cap_global_from_dem(items: Dict[int, dict], T: int) -> List[int]:
    cap_raw = [0.0] * T
    for it in items.values():
        for t in range(T):
            cap_raw[t] += float(it["demand"][t])
    buf = max(5.0, 0.2 * (max(cap_raw) if cap_raw else 0.0))
    # keep integer-like capacities if original are ints
    return [int(round(c + buf)) for c in cap_raw]


# ---------------- parsing & costs ----------------


def _parse_instance(instance_path: str | Path):
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

    W = data.get("warehouse_capacity", None)
    W = float(W) if W is not None else None

    allow_lost_sales = bool(
        data.get("allow_unmet_demand", False) or data.get("allow_lost_sales", False)
    )
    loss_penalty_global = data.get("lost_sales_penalty", None)
    loss_penalty_factor = float(data.get("lost_sales_penalty_factor", 200.0))

    return (
        data,
        T,
        Periods,
        items_raw,
        prod_cap,
        per_item_cap,
        W,
        allow_lost_sales,
        loss_penalty_global,
        loss_penalty_factor,
    )


def _precompute_gamma_and_expiry(
    items_raw: Dict[int, dict], T: int, inclusive_shelf: bool = True
):
    """
    Gamma[(i,t)] = list of consumption periods u that can be served by production at t.
    If inclusive_shelf == True: shelf_seq[t] = L -> u in [t .. t+L]  (this + next L)
    Else (legacy):               shelf_seq[t] = L -> u in [t .. t+L-1]
    Expiry[(i,t)] is kept as an exclusive index (first invalid period),
    so Expiry = t + L + 1 (inclusive) vs t + L (legacy).
    """
    Gamma: Dict[Tuple[int, int], List[int]] = {}
    Expiry: Dict[Tuple[int, int], int] = {}
    for i, it in items_raw.items():
        mseq = list(it["shelf_seq"])
        if len(mseq) != T:
            raise ValueError(f"items[{i}]['shelf_seq'] must have length {T}")
        for t in range(T):
            L = int(mseq[t])
            if inclusive_shelf:
                v_it = t + L + 1  # exclusive
                u_max = min(T - 1, t + L)
            else:
                v_it = t + L  # exclusive
                u_max = min(T - 1, v_it - 1)
            Expiry[(i, t)] = v_it
            if L <= 0:
                Gamma[(i, t)] = []
            else:
                Gamma[(i, t)] = [u for u in range(t, u_max + 1)]
    return Gamma, Expiry


def _cost_accessors(items_raw: Dict[int, dict], T: int):
    """
    Returns:
      c_at(i,t), s_at(i,t), hsum(i,t,u), hpref(i,u)
    where hpref(i,u) == sum_{k=0..u-1} h_i[k].
    """
    h_pref: Dict[int, List[float]] = {}
    for i, it in items_raw.items():
        h = it["h"]
        if isinstance(h, list):
            pref = [0.0] * (T + 1)
            for k in range(T):
                pref[k + 1] = pref[k] + float(h[k])
            h_pref[i] = pref

    def c_at(i: int, t: int) -> float:
        c = items_raw[i]["c_var"]
        return float(c[t]) if isinstance(c, list) else float(c)

    def s_at(i: int, t: int) -> float:
        s = items_raw[i]["setup"]
        return float(s[t]) if isinstance(s, list) else float(s)

    def hsum(i: int, t: int, u: int) -> float:
        h = items_raw[i]["h"]
        if isinstance(h, list):
            pref = h_pref[i]
            return float(pref[u] - pref[t])
        return float(h) * (u - t)

    def hpref(i: int, u: int) -> float:
        h = items_raw[i]["h"]
        if isinstance(h, list):
            return float(h_pref[i][u])
        return float(h) * u

    return c_at, s_at, hsum, hpref


def _lost_sales_penalties(
    items_raw: Dict[int, dict],
    T: int,
    Gamma,
    c_at,
    s_at,
    hsum,
    loss_penalty_global,
    loss_penalty_factor: float,
):
    max_unit_var_cost = 0.0
    for i in items_raw:
        for t in range(T):
            for u in Gamma.get((i, t), []):
                max_unit_var_cost = max(max_unit_var_cost, c_at(i, t) + hsum(i, t, u))
    if max_unit_var_cost <= 0.0:
        max_unit_var_cost = 1.0
    max_setup = 0.0
    for i in items_raw:
        s = items_raw[i]["setup"]
        max_setup = max(
            max_setup, max(map(float, s)) if isinstance(s, list) else float(s)
        )
    default_lp = loss_penalty_global
    if default_lp is None:
        base = max_unit_var_cost + max_setup
        default_lp = max(10.0 * max_unit_var_cost, loss_penalty_factor * base)
        default_lp = float(min(default_lp + 1.0, 1e9))

    loss_pen = {}
    for i in items_raw:
        lp = items_raw[i].get("lost_sales_penalty", None)
        if lp is not None:
            vec = _as_len_T_vector(lp, T)
            for u in range(T):
                loss_pen[(i, u)] = float(vec[u])
        else:
            for u in range(T):
                loss_pen[(i, u)] = float(default_lp)
    return loss_pen, float(default_lp)


# ---------------- DUMMY (outsourcing) seeding ----------------


def seed_plan_dummy_outsource(
    i: int, items_raw: Dict[int, dict], T: int, unit_cost: float
) -> ColumnPlan:
    """
    Dış tedarik/dummy: Hiç üretim ve envanter yok; kapasite/depoya yük bindirmez.
    Tüm talebi dışarıdan aldığımızı temsil eder; maliyet = unit_cost * toplam talep.
    """
    d = [float(x) for x in items_raw[i]["demand"]]
    total = sum(d)
    cost = float(unit_cost) * float(total)  # büyük ama sonlu
    prod_by_t = [0.0] * T
    inv_end_by_u = [0.0] * (T - 1 if T >= 2 else 0)
    flows: List[Tuple[int, int, float]] = []  # iç üretim akışı yok
    setups: List[int] = []  # setup yok
    lost_sales_by_u = [0.0] * T  # LS yok (force_no_lost_sales)
    y_values = [0.0] * T
    return ColumnPlan(
        i, -1, cost, prod_by_t, inv_end_by_u, flows, setups, lost_sales_by_u, y_values
    )


# ---------------- pricing (MIP, LEFO) ----------------


def _check_lefo_flows(
    i: int,
    flows: List[Tuple[int, int, float]],
    Expiry: Dict[Tuple[int, int], int],
) -> None:
    """
    flows: (t,u,q) listesi (q>0 olanlar)
    Kural: u1 < u2 ise v(i,t1) <= v(i,t2) olmalı (expiry nondecreasing).
    """
    # u'ya göre sırala, her tüketimde kullanılan üretimin v(i,t)'sine bak
    seq = sorted(
        ((u, Expiry[(i, t)]) for (t, u, q) in flows if q > EPS), key=lambda x: x[0]
    )
    last_v = -(10**12)
    for u, v in seq:
        if v < last_v - 1e-12:
            raise RuntimeError(
                f"LEFO violation at u={u}: expiry {v} < previous {last_v}"
            )
        last_v = v


def price_item_plan_mip(
    i: int,
    items_raw: Dict[int, dict],
    T: int,
    Gamma: Dict[Tuple[int, int], List[int]],
    Expiry: Dict[Tuple[int, int], int],
    c_at,
    s_at,
    hsum,
    per_item_cap: Dict[Tuple[int, int], float],
    allow_lost_sales: bool,
    loss_pen: Dict[Tuple[int, int], float],
    pi_cap: List[float],
    rho_wh: Optional[List[float]],
    fixed_y: Dict[int, int],
    k: int,
    # === NEW: duals for linking rows γ_{i,t}
    gamma_it: Optional[Dict[int, float]] = None,
) -> List[Tuple[ColumnPlan, float]]:
    sub = gp.Model(f"price_mip_i{i}")
    sub.Params.OutputFlag = 0
    sub.Params.PoolSearchMode = 2 if k > 1 else 0
    sub.Params.PoolSolutions = max(10, k * 2)
    sub.Params.PoolGap = 1.0  # Allow suboptimal for diversity

    # per-item mu
    d_i = [float(x) for x in items_raw[i]["demand"]]
    mu_i: Dict[int, float] = {}
    for t in range(T):
        mu_i[t] = sum(d_i[u] for u in Gamma.get((i, t), []))

    # triples for i
    triples = [(t, u) for t in range(T) for u in Gamma.get((i, t), [])]

    X = sub.addVars(triples, lb=0.0, vtype=GRB.CONTINUOUS, name="X")
    Y = sub.addVars(range(T), vtype=GRB.BINARY, name="Y")
    Z = sub.addVars(triples, vtype=GRB.BINARY, name="Z")
    if allow_lost_sales:
        LS = sub.addVars(range(T), lb=0.0, vtype=GRB.CONTINUOUS, name="LS")
    if rho_wh is not None and T >= 2:
        Inv = sub.addVars(range(T - 1), lb=0.0, vtype=GRB.CONTINUOUS, name="Inv")

    # objective: rc = true_cost - pi*prod - rho*inv - gamma*y
    obj = gp.LinExpr()
    for t, u in triples:
        obj += (c_at(i, t) + hsum(i, t, u)) * X[t, u]
    for t in range(T):
        obj += s_at(i, t) * Y[t]
    if allow_lost_sales:
        for u in range(T):
            obj += loss_pen[(i, u)] * LS[u]
    # subtract duals
    for t in range(T):
        prod_t = sum(X[t, u] for _, u in triples if _ == t)
        obj -= pi_cap[t] * prod_t
    if rho_wh is not None and T >= 2:
        for uu in range(T - 1):
            inv_uu = sum(X[t, w] for t, w in triples if t <= uu < w)
            sub.addConstr(Inv[uu] == inv_uu)
            obj -= rho_wh[uu] * Inv[uu]
    # === NEW: subtract γ_{i,t} for Y-bits
    if gamma_it is not None:
        for t in range(T):
            g = float(gamma_it.get(t, 0.0))
            if abs(g) > 0.0:
                obj -= g * Y[t]

    sub.setObjective(obj, GRB.MINIMIZE)

    # constraints
    # C2 setup link
    for t in range(T):
        if Gamma.get((i, t)):
            sub.addConstr(sum(X[t, u] for u in Gamma[(i, t)]) <= mu_i[t] * Y[t])
        else:
            sub.addConstr(Y[t] == 0)
    # C3 demand
    for u in range(T):
        inc = sum(X[t, u] for t, _ in triples if _ == u)
        if allow_lost_sales:
            sub.addConstr(inc + LS[u] == d_i[u])
        else:
            sub.addConstr(inc == d_i[u])
    # C4 arc on
    for t, u in triples:
        sub.addConstr(X[t, u] <= d_i[u] * Z[t, u])
    # C5 no-crossing LEFO
    prods = [t for t in range(T) if Gamma.get((i, t))]
    prods.sort(key=lambda tt: Expiry[(i, tt)])
    for a in range(len(prods)):
        t1 = prods[a]
        v1 = Expiry[(i, t1)]
        for b in range(a + 1, len(prods)):
            t2 = prods[b]
            v2 = Expiry[(i, t2)]
            if v1 >= v2:
                continue
            for up in Gamma[(i, t2)]:
                for uu in [uuu for uuu in Gamma[(i, t1)] if t2 <= uuu <= up - 1]:
                    sub.addConstr(Z[t1, uu] + Z[t2, up] <= 1)
    # per-item cap
    for t in range(T):
        pit = per_item_cap.get((i, t), math.inf)
        sub.addConstr(sum(X[t, u] for _, u in triples if _ == t) <= pit)
    # fixed Y
    for tt, val in fixed_y.items():
        sub.addConstr(Y[tt] == val)

    sub.optimize()
    if sub.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return []

    out = []
    seen = set()
    nsol = min(k * 2, sub.SolCount)  # extra for uniqueness
    for sol_idx in range(nsol):
        try:
            sub.setParam(GRB.Param.SolutionNumber, sol_idx)
            flows = []
            prod_by_t = [0.0] * T
            lost_by_u = [0.0] * T
            y_values = [0.0] * T
            for t in range(T):
                y_values[t] = Y[t].getAttr("Xn")
                qty = sum(X[t, u].getAttr("Xn") for _, u in triples if _ == t)
                if qty > EPS:
                    prod_by_t[t] = qty
            for t, u in triples:
                q = X[t, u].getAttr("Xn")
                if q > EPS:
                    flows.append((t, u, q))
            if allow_lost_sales:
                for u in range(T):
                    lost_by_u[u] = LS[u].getAttr("Xn")
            inv_end = [0.0] * max(0, T - 1)
            if rho_wh is not None and T >= 2:
                for u in range(T - 1):
                    inv_end[u] = Inv[u].getAttr("Xn")
            else:
                prod_at = [0.0] * T
                cons_at = [0.0] * T
                for t, u, q in flows:
                    prod_at[t] += q
                    cons_at[u] += q
                inv = 0.0
                for u in range(T):
                    inv += prod_at[u] - cons_at[u]
                    if u < T - 1:
                        inv_end[u] = max(0.0, inv)
            cost = 0.0
            for t, u, q in flows:
                cost += (c_at(i, t) + hsum(i, t, u)) * q
            for t in range(T):
                cost += s_at(i, t) * y_values[t]
            if allow_lost_sales:
                for u in range(T):
                    cost += loss_pen[(i, u)] * lost_by_u[u]
            rc = sub.getAttr("PoolObjVal")
            key = tuple(sorted((t, u, round(q, 9)) for t, u, q in flows))
            if key in seen:
                continue
            seen.add(key)
            setups = [t for t in range(T) if y_values[t] > EPS]
            pl = ColumnPlan(
                i,
                -1,
                cost,
                prod_by_t,
                inv_end,
                flows,
                sorted(setups),
                lost_by_u,
                y_values,
            )
            out.append((pl, rc))
            if len(out) >= k:
                break
        except (gp.GurobiError, AttributeError):
            continue
    return out


def price_item_plan_k(
    i: int,
    items_raw: Dict[int, dict],
    T: int,
    Gamma,
    Expiry,
    c_at,
    s_at,
    hsum,
    per_item_cap: Dict[Tuple[int, int], float],
    allow_lost_sales: bool,
    loss_pen: Dict[Tuple[int, int], float],
    pi_cap: List[float],
    rho_wh: Optional[List[float]],
    k: int,
    seed: int = 0,
    fixed_y: Optional[Dict[int, int]] = None,
    # === NEW:
    gamma_it: Optional[Dict[int, float]] = None,
) -> List[Tuple[ColumnPlan, float]]:
    return price_item_plan_mip(
        i,
        items_raw,
        T,
        Gamma,
        Expiry,
        c_at,
        s_at,
        hsum,
        per_item_cap,
        allow_lost_sales,
        loss_pen,
        pi_cap,
        rho_wh,
        fixed_y or {},
        k,
        gamma_it=gamma_it,
    )


# ---------------- (legacy) random & greedy seeding (kept for reference, unused) ----------------


def seed_plan_naive_latest(
    i, items_raw, T, Gamma, c_at, s_at, hsum, allow_lost_sales, loss_pen
):
    d = [float(x) for x in items_raw[i]["demand"]]
    flows = []
    prod_by_t = [0.0] * T
    lost = [0.0] * T
    for u in range(T):
        if d[u] <= EPS:
            continue
        cand_t = None
        for t in range(u, -1, -1):
            if u in Gamma.get((i, t), []):
                cand_t = t
                break
        if cand_t is None:
            for t in range(0, u + 1):
                if u in Gamma.get((i, t), []):
                    cand_t = t
                    break
        if cand_t is None:
            if allow_lost_sales:
                lost[u] = d[u]
            continue
        flows.append((cand_t, u, d[u]))
        prod_by_t[cand_t] += d[u]
    cost = 0.0
    for t, u, q in flows:
        cost += (c_at(i, t) + hsum(i, t, u)) * q
    for t in range(T):
        if prod_by_t[t] > EPS:
            cost += s_at(i, t)
    if allow_lost_sales:
        for u in range(T):
            if lost[u] > EPS:
                cost += loss_pen[(i, u)] * lost[u]
    inv_end = [0.0] * (T - 1 if T >= 2 else 0)
    prod_at = [0.0] * T
    cons_at = [0.0] * T
    for t, u, q in flows:
        prod_at[t] += q
        cons_at[u] += q
    inv = 0.0
    for u in range(T):
        inv += prod_at[u]
        inv -= cons_at[u]
        if u <= T - 2:
            inv_end[u] = max(inv, 0.0)
    setups = [t for t in range(T) if prod_by_t[t] > EPS]
    y_values = [1.0 if t in setups else 0.0 for t in range(T)]
    return ColumnPlan(
        i, -1, float(cost), prod_by_t, inv_end, flows, setups, lost, y_values
    )


def seed_plan_naive_earliest(
    i, items_raw, T, Gamma, c_at, s_at, hsum, allow_lost_sales, loss_pen
):
    d = [float(x) for x in items_raw[i]["demand"]]
    flows = []
    prod_by_t = [0.0] * T
    lost = [0.0] * T
    for u in range(T):
        if d[u] <= EPS:
            continue
        cand_t = None
        for t in range(0, u + 1):
            if u in Gamma.get((i, t), []):
                cand_t = t
                break
        if cand_t is None:
            for t in range(u, -1, -1):
                if u in Gamma.get((i, t), []):
                    cand_t = t
                    break
        if cand_t is None:
            if allow_lost_sales:
                lost[u] = d[u]
            continue
        flows.append((cand_t, u, d[u]))
        prod_by_t[cand_t] += d[u]
    cost = 0.0
    for t, u, q in flows:
        cost += (c_at(i, t) + hsum(i, t, u)) * q
    for t in range(T):
        if prod_by_t[t] > EPS:
            cost += s_at(i, t)
    if allow_lost_sales:
        for u in range(T):
            if lost[u] > EPS:
                cost += loss_pen[(i, u)] * lost[u]
    inv_end = [0.0] * (T - 1 if T >= 2 else 0)
    prod_at = [0.0] * T
    cons_at = [0.0] * T
    for t, u, q in flows:
        prod_at[t] += q
        cons_at[u] += q
    inv = 0.0
    for u in range(T):
        inv += prod_at[u]
        inv -= cons_at[u]
        if u <= T - 2:
            inv_end[u] = max(inv, 0.0)
    setups = [t for t in range(T) if prod_by_t[t] > EPS]
    y_values = [1.0 if t in setups else 0.0 for t in range(T)]
    return ColumnPlan(
        i, -1, float(cost), prod_by_t, inv_end, flows, setups, lost, y_values
    )


def seed_plan_random_blocks(
    i, items_raw, T, Gamma, Expiry, per_item_cap, c_at, s_at, hsum
):
    d = [float(x) for x in items_raw[i]["demand"]]
    flows: List[Tuple[int, int, float]] = []
    prod_by_t = [0.0] * T
    setups: List[int] = []

    def next_pos(u: int) -> int:
        while u < T and d[u] <= EPS:
            u += 1
        return u

    s = next_pos(0)
    v_last = -(10**9)
    while s < T:
        starts = []
        for t in range(0, s + 1):
            if s in Gamma.get((i, t), []) and Expiry[(i, t)] >= v_last:
                starts.append(t)
        if not starts:
            s = next_pos(s + 1)
            continue
        t = random.choice(starts)
        vmax = Expiry[(i, t)] - 1
        e_max = min(T - 1, vmax)
        pit = per_item_cap.get((i, t), math.inf)
        run = 0.0
        e = s
        target_span = s + random.randint(0, max(0, min(6, e_max - s)))
        while e <= e_max:
            if d[e] > EPS:
                run += d[e]
            if run - pit > EPS:
                break
            if e >= target_span and run > 0.0 and random.random() < 0.5:
                break
            e += 1
        e = min(e, e_max)
        if e < s and d[s] <= EPS:
            s = next_pos(s + 1)
            continue
        if e < s:
            e = s
        q = 0.0
        for u in range(s, e + 1):
            if d[u] > EPS:
                flows.append((t, u, d[u]))
                q += d[u]
        if q > EPS:
            prod_by_t[t] += q
            setups.append(t)
            v_last = Expiry[(i, t)]
        s = next_pos(e + 1)

    cost = 0.0
    for t, u, q in flows:
        cost += (c_at(i, t) + hsum(i, t, u)) * q
    for t in range(T):
        if prod_by_t[t] > EPS:
            cost += s_at(i, t)
    inv_end = [0.0] * (T - 1 if T >= 2 else 0)
    prod_at = [0.0] * T
    cons_at = [0.0] * T
    for t, u, q in flows:
        prod_at[t] += q
        cons_at[u] += q
    inv = 0.0
    for u in range(T):
        inv += prod_at[u]
        inv -= cons_at[u]
        if u <= T - 2:
            inv_end[u] = max(inv, 0.0)
    setups = sorted(list(set(setups)))
    y_values = [1.0 if t in setups else 0.0 for t in range(T)]
    return ColumnPlan(
        i, -1, float(cost), prod_by_t, inv_end, flows, setups, [0.0] * T, y_values
    )


def seed_plan_chunked(
    i, items_raw, T, Gamma, Expiry, per_item_cap, c_at, s_at, hsum, chunk_len: int = 3
):
    d = [float(x) for x in items_raw[i]["demand"]]
    flows: List[Tuple[int, int, float]] = []
    prod_by_t = [0.0] * T

    u = 0
    while u < T:
        while u < T and d[u] <= EPS:
            u += 1
        if u >= T:
            break
        cand_t = None
        for t in range(u, -1, -1):
            if u in Gamma.get((i, t), []):
                cand_t = t
                break
        if cand_t is None:
            for t in range(0, u + 1):
                if u in Gamma.get((i, t), []):
                    cand_t = t
                    break
        if cand_t is None:
            u += 1
            continue
        vmax = Expiry[(i, cand_t)] - 1
        e_max = min(T - 1, vmax)
        e_target = min(u + chunk_len - 1, e_max)
        pit = per_item_cap.get((i, cand_t), math.inf)
        run = 0.0
        e = u
        while e <= e_target and e <= e_max:
            if d[e] > EPS:
                run += d[e]
            if run - pit > EPS:
                break
            e += 1
        e = min(e - 1, e_max)
        if e < u and d[u] <= EPS:
            u += 1
            continue
        if e < u:
            e = u
        q = 0.0
        for uu in range(u, e + 1):
            if d[uu] > EPS:
                flows.append((cand_t, uu, d[uu]))
                q += d[uu]
        if q > EPS:
            prod_by_t[cand_t] += q
        u = e + 1

    cost = 0.0
    for t, u2, q in flows:
        cost += (c_at(i, t) + hsum(i, t, u2)) * q
    for t in range(T):
        if prod_by_t[t] > EPS:
            cost += s_at(i, t)
    inv_end = [0.0] * (T - 1 if T >= 2 else 0)
    prod_at = [0.0] * T
    cons_at = [0.0] * T
    for t, u2, q in flows:
        prod_at[t] += q
        cons_at[u2] += q
    inv = 0.0
    for u2 in range(T):
        inv += prod_at[u2]
        inv -= cons_at[u2]
        if u2 <= T - 2:
            inv_end[u2] = max(inv, 0.0)
    setups = [t for t in range(T) if prod_by_t[t] > EPS]
    y_values = [1.0 if t in setups else 0.0 for t in range(T)]
    return ColumnPlan(
        i,
        -1,
        float(cost),
        prod_by_t,
        inv_end,
        flows,
        sorted(setups),
        [0.0] * T,
        y_values,
    )


# ---------------- master (RMP) ----------------


def build_rmp(T, prod_cap, use_wh, W, items_raw):
    rmp = gp.Model("RMP_perishable_LEFO_fast")
    rmp.Params.OutputFlag = 0
    cap_con: Dict[int, gp.Constr] = {}
    for t in range(T):
        cap_con[t] = rmp.addConstr(gp.LinExpr() <= float(prod_cap[t]), name=f"cap_{t}")
    inv_con: Dict[int, gp.Constr] = {}
    if use_wh and T >= 2 and W is not None:
        for u in range(T - 1):
            inv_con[u] = rmp.addConstr(gp.LinExpr() <= float(W), name=f"wh_{u}")
    one_con: Dict[int, gp.Constr] = {}
    for i in items_raw:
        one_con[i] = rmp.addConstr(gp.LinExpr() == 1.0, name=f"one_{i}")

    # === NEW: Ybar binaries and linking rows: sum_p y_{i,p,t} λ_{i,p} - Ybar_{i,t} == 0
    Ybin: Dict[Tuple[int, int], gp.Var] = {}
    ylink: Dict[Tuple[int, int], gp.Constr] = {}
    for i in items_raw:
        for t in range(T):
            cons = rmp.addConstr(gp.LinExpr() == 0.0, name=f"ylink_{i}_{t}")
            col = gp.Column()
            col.addTerms(-1.0, cons)
            yv = rmp.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{t}", column=col)
            Ybin[(i, t)] = yv
            ylink[(i, t)] = cons

    rmp.update()
    return rmp, cap_con, inv_con, one_con, Ybin, ylink


# ---------------- no-good cuts (optional) ----------------


def add_no_good_cut_for_pattern(
    rmp: gp.Model,
    Ybin: Dict[Tuple[int, int], gp.Var],
    i: int,
    y_star: List[int],
    T: int,
    cut_id: int,
):
    """
    Forbid exactly the pattern y_star for item i:
        sum_t [y*_t==1](1 - Y_{i,t}) + [y*_t==0] Y_{i,t} >= 1
    (i.e., at least one bit must differ)
    """
    expr = gp.LinExpr()
    for t in range(T):
        if int(y_star[t]) == 1:
            expr += 1 - Ybin[(i, t)]
        else:
            expr += Ybin[(i, t)]
    rmp.addConstr(expr >= 1, name=f"nogood_i{i}_{cut_id}")


# ---------------- fix-and-price diving ----------------


def _dive_fix_and_price(
    rmp: gp.Model,
    Ybin: Dict[Tuple[int, int], gp.Var],
    items_raw: Dict[int, dict],
    pool: Dict[int, List[ColumnPlan]],
    lam_vars: Dict[Tuple[int, int], gp.Var],
    cap_con: Dict[int, gp.Constr],
    inv_con: Dict[int, gp.Constr],
    one_con: Dict[int, gp.Constr],
    T: int,
    use_wh: bool,
    # === NEW: ylink to read γ-duals during diving
    ylink: Dict[Tuple[int, int], gp.Constr],
    price_fn_per_item: Callable[
        [int, List[float], Optional[List[float]], Dict[int, float]], bool
    ],
    max_fix: Optional[int] = None,
    reprice_iters: int = 40,
    stab_alpha: float = 0.6,
    verbose: bool = True,
) -> Dict[Tuple[int, int], float]:
    fixed: Set[int] = set()
    saved_bounds: Dict[Tuple[int, int], Tuple[float, float]] = {}

    def _duals():
        pi = [cap_con[t].Pi for t in range(T)]
        rho = [inv_con[u].Pi for u in range(T - 1)] if use_wh else []
        sigma = {i: one_con[i].Pi for i in items_raw}
        gamma: Dict[Tuple[int, int], float] = {}
        for (ii, tt), cons in ylink.items():
            gamma[(ii, tt)] = cons.Pi
        return pi, rho, sigma, gamma

    def _save_bound(i: int, pid: int):
        v = lam_vars.get((i, pid))
        if v is None:
            return
        if (i, pid) not in saved_bounds:
            saved_bounds[(i, pid)] = (v.LB, v.UB)

    def _set_bound(i: int, pid: int, lb: float, ub: float):
        v = lam_vars.get((i, pid))
        if v is None:
            return
        _save_bound(i, pid)
        v.LB = lb
        v.UB = ub

    def _revert_all_bounds():
        for (i, pid), (lb, ub) in saved_bounds.items():
            v = lam_vars.get((i, pid))
            if v is not None:
                v.LB = lb
                v.UB = ub

    iter_fixes = 0
    # Ensure Ybar variables are continuous during diving so γ duals are available
    for yvar in Ybin.values():
        yvar.VType = GRB.CONTINUOUS
    rmp.update()
    while True:
        rmp.optimize()
        # pick most fractional item (largest support >1 positive λ)
        fractional_i = None
        best_support = 0
        for i in items_raw:
            if i in fixed:
                continue
            supp = [
                (pid, lam_vars[(i, pid)].X)
                for pid in range(len(pool[i]))
                if (i, pid) in lam_vars
            ]
            pos = [x for x in supp if x[1] > 1e-8]
            if len(pos) > 1 and sum(v for _, v in pos) > 0.999:
                if len(pos) > best_support:
                    best_support = len(pos)
                    fractional_i = i
        if fractional_i is None:
            break

        # fix chosen item to its largest-λ plan
        support = [
            (pid, lam_vars[(fractional_i, pid)].X)
            for pid in range(len(pool[fractional_i]))
            if (fractional_i, pid) in lam_vars
        ]
        if not support:
            break
        pid_star, _ = max(support, key=lambda p: p[1])
        for pid in range(len(pool[fractional_i])):
            if (fractional_i, pid) not in lam_vars:
                continue
            if pid == pid_star:
                _set_bound(fractional_i, pid, 1.0, 1.0)
            else:
                _set_bound(fractional_i, pid, 0.0, 0.0)
        fixed.add(fractional_i)
        iter_fixes += 1
        if verbose:
            _log(f"[DIVE] fix item {fractional_i} -> plan {pid_star}")
        if max_fix and iter_fixes >= max_fix:
            break

        # re-pricing to repair pool for the remaining items
        last_pi, last_rho = [0.0] * T, [0.0] * (T - 1 if use_wh else 0)
        last_gamma: Dict[Tuple[int, int], float] = {}
        for _ in range(reprice_iters):
            rmp.optimize()
            pi_raw, rho_raw, _, gamma_raw = _duals()
            # stab on pi/rho only (gamma usually small/noisy)
            pi = [
                stab_alpha * p + (1 - stab_alpha) * lp for p, lp in zip(pi_raw, last_pi)
            ]
            rho = [
                stab_alpha * r + (1 - stab_alpha) * lr
                for r, lr in zip(rho_raw, last_rho)
            ]
            last_pi, last_rho = pi, rho
            gamma_i_map: Dict[int, float] = {}
            any_add = False
            for i in items_raw:
                # prepare per-item gamma vector γ_{i,t}
                gamma_i_map.clear()
                for t in range(T):
                    gamma_i_map[t] = float(gamma_raw.get((i, t), 0.0))
                added = price_fn_per_item(i, pi, rho, dict(gamma_i_map))
                any_add = any_add or added
            if not any_add:
                break

    # snapshot warm start
    rmp.optimize()
    warm_start: Dict[Tuple[int, int], float] = {}
    for (i, pid), v in lam_vars.items():
        try:
            warm_start[(i, pid)] = float(v.X)
        except Exception:
            warm_start[(i, pid)] = 0.0

    _revert_all_bounds()
    # Restore Ybar variable types (binary) before returning
    for yvar in Ybin.values():
        yvar.VType = GRB.BINARY
    rmp.update()
    return warm_start


# ---------------- main ----------------


def solve_instance(
    instance_path: str | Path = "last_instance.json",
    out_dir: str | Path = "bp_fast_results",
    # Seed knobs (unused with dummy-only, but kept for API)
    seed_random_per_item: int = 0,
    seed_chunk_len: int = 3,
    seed_extra_chunked: bool = False,
    # Pricing knobs
    stabilize: bool = False,
    stab_alpha: float = 0.8,  # Tuned: less smoothing for more columns
    max_iter: int = 50000,  # Tuned: increased for more iterations
    max_add_per_item_per_iter: int = 5,  # Tuned: more adds per iter
    pricing_k: int = 5,  # Tuned: more diverse plans
    drop_age: int = 10,
    # final
    finalize_as_mip: bool = True,
    # limits
    time_limit: int = 0,
    mip_gap: float = 0.0,
    # semantics/feasibility
    inclusive_shelf: bool = False,  # L covers this + next L (inclusive shelf-life)
    force_no_lost_sales: bool = False,  # Changed default to False to follow instance
    # diving
    enable_diving: bool = True,
    diving_reprice_iters: int = 40,
    verbose: bool = True,
    # === NEW switches ===
    lambda_binary: bool = False,  # B) keep λ continuous by default
    rf_branching: bool = True,  # D) enable Ryan–Foster branching when possible
    enable_nogood_cuts: bool = False,  # C) optional no-good cuts (off by default)
):
    t0 = time.time()
    (
        data,
        T,
        Periods,
        items_raw,
        prod_cap,
        per_item_cap,
        W,
        allow_lost_sales_in,
        loss_penalty_global,
        loss_penalty_factor,
    ) = _parse_instance(instance_path)

    # semantics / feasibility overrides - now default follows instance
    allow_lost_sales = allow_lost_sales_in and not force_no_lost_sales

    Gamma, Expiry = _precompute_gamma_and_expiry(
        items_raw, T, inclusive_shelf=inclusive_shelf
    )
    c_at, s_at, hsum, _hpref = _cost_accessors(items_raw, T)
    loss_pen, default_lp = _lost_sales_penalties(
        items_raw, T, Gamma, c_at, s_at, hsum, loss_penalty_global, loss_penalty_factor
    )
    use_wh = (W is not None) and (T >= 2)

    # ---- DUMMY outsourcing per-unit cost ----
    outsource_unit_cost = float(data.get("outsource_unit_cost", default_lp))
    # if CLI provided a specific cost, override (best-effort)
    try:
        from __main__ import args as _cli_args  # type: ignore

        if getattr(_cli_args, "outsource_unit_cost", None) is not None:
            outsource_unit_cost = float(_cli_args.outsource_unit_cost)
    except Exception:
        pass

    # RMP with feasibility slacks (kept but punished heavily, yet numerically safe)
    rmp, cap_con, inv_con, one_con, Ybin, ylink = build_rmp(
        T, prod_cap, use_wh, W, items_raw
    )
    M_cap = 1e7  # large but not extreme
    M_wh = 1e7
    cap_slack: Dict[int, gp.Var] = {}
    for t in range(T):
        col = gp.Column()
        col.addTerms(-1.0, cap_con[t])
        cap_slack[t] = rmp.addVar(lb=0.0, obj=M_cap, name=f"s_cap_{t}", column=col)
    inv_slack: Dict[int, gp.Var] = {}
    if use_wh:
        for u in range(T - 1):
            col = gp.Column()
            col.addTerms(-1.0, inv_con[u])
            inv_slack[u] = rmp.addVar(lb=0.0, obj=M_wh, name=f"s_wh_{u}", column=col)
    rmp.update()

    pool: Dict[int, List[ColumnPlan]] = {i: [] for i in items_raw}
    lam_vars: Dict[Tuple[int, int], gp.Var] = {}
    ages: Dict[Tuple[int, int], int] = {}

    def add_column(i: int, pl: ColumnPlan):
        pid = len(pool[i])
        pl.plan_id = pid
        pool[i].append(pl)
        col = gp.Column()
        for t in range(T):
            q = pl.prod_by_t[t]
            if abs(q) > EPS:
                col.addTerms(q, cap_con[t])
        if use_wh:
            for u in range(T - 1):
                inv = pl.inv_end_by_u[u]
                if abs(inv) > EPS:
                    col.addTerms(inv, inv_con[u])
        # per-item selection
        col.addTerms(1.0, one_con[i])
        # === NEW: contribute Y-bits to linking rows
        for t in range(T):
            yt = float(pl.y_values[t])
            if abs(yt) > EPS:
                col.addTerms(yt, ylink[(i, t)])

        v = rmp.addVar(lb=0.0, obj=float(pl.cost), name=f"lam_{i}_{pid}", column=col)
        lam_vars[(i, pid)] = v
        ages[(i, pid)] = 0
        rmp.update()
        if verbose:
            setups_log = sorted({t for t, y in enumerate(pl.y_values) if y > EPS})
            _log(f"[ADD] i={i} plan={pid} cost={pl.cost:.3f} setups={setups_log}")

    # ------- Seed: DUMMY + Legacy greedy/random for better starting pool -------
    if verbose:
        _log(
            "[SEED] Adding per-item DUMMY outsourcing + greedy/random columns (feasible root)."
        )
    for i in items_raw:
        pl_dummy = seed_plan_dummy_outsource(
            i, items_raw, T, unit_cost=outsource_unit_cost
        )
        add_column(i, pl_dummy)
        # Add legacy seeds for diversity
        add_column(
            i,
            seed_plan_naive_latest(
                i, items_raw, T, Gamma, c_at, s_at, hsum, allow_lost_sales, loss_pen
            ),
        )
        add_column(
            i,
            seed_plan_naive_earliest(
                i, items_raw, T, Gamma, c_at, s_at, hsum, allow_lost_sales, loss_pen
            ),
        )
        add_column(
            i,
            seed_plan_random_blocks(
                i, items_raw, T, Gamma, Expiry, per_item_cap, c_at, s_at, hsum
            ),
        )
        add_column(
            i,
            seed_plan_chunked(
                i,
                items_raw,
                T,
                Gamma,
                Expiry,
                per_item_cap,
                c_at,
                s_at,
                hsum,
                chunk_len=seed_chunk_len,
            ),
        )

    # ------- CG loop (root node) -------
    # Ensure Ybar variables are continuous during CG so duals (γ) are available.
    for yvar in Ybin.values():
        yvar.VType = GRB.CONTINUOUS
    rmp.update()

    iter_no = 0
    last_pi = [0.0] * T
    last_rho = [0.0] * (T - 1 if use_wh else 0)

    # helper pricing wrapper (now with gamma)
    def _price_and_add_for(
        i: int,
        pi_vec: List[float],
        rho_vec: Optional[List[float]],
        fixed_y: Dict[int, int] = {},
        gamma_it: Optional[Dict[int, float]] = None,
    ):
        nonlocal worst_rc
        cand_list = price_item_plan_k(
            i,
            items_raw,
            T,
            Gamma,
            Expiry,
            c_at,
            s_at,
            hsum,
            per_item_cap,
            allow_lost_sales,
            loss_pen,
            pi_vec,
            (rho_vec if use_wh else None),
            k=(
                max_add_per_item_per_iter
                if pricing_k <= 0
                else min(pricing_k, max_add_per_item_per_iter)
            ),
            fixed_y=fixed_y,
            gamma_it=gamma_it,
        )
        cand_list.sort(key=lambda tup: tup[1])
        added_for_i = 0
        added_local = False
        for plan, rc_wo_sigma_gamma in cand_list:
            # pricing returns rc already accounting for pi/rho/gamma. Subtract sigma (one_con dual).
            rc_total = rc_wo_sigma_gamma - sigma[i]
            worst_rc = min(worst_rc, rc_total)
            if rc_total < -RC_EPS:
                if verbose:
                    _log(f"[PRICE] i={i} rc={rc_total:.6e} -> ADD")
                add_column(i, plan)
                added_local = True
                added_for_i += 1
                if added_for_i >= max_add_per_item_per_iter:
                    break
        return added_local

    while True:
        iter_no += 1
        if time_limit and time.time() - t0 > time_limit:
            _log("[STOP] Time limit hit during root CG.")
            break

        rmp.Params.OutputFlag = 0
        rmp.optimize()
        try:
            obj_now = float(rmp.ObjVal)
            obj_str = f"{obj_now:.6f}"
        except Exception:
            obj_str = "NA"

        # update ages
        for key, v in list(lam_vars.items()):
            try:
                val = float(v.X)
            except Exception:
                val = 0.0
            ages[key] = 0 if val > 1e-10 else ages.get(key, 0) + 1

        # optional drop
        to_drop = []
        for (ii, pid), age in ages.items():
            if age > drop_age and (ii, pid) in lam_vars and len(pool[ii]) > 5:
                try:
                    if float(lam_vars[(ii, pid)].X) <= 1e-10:
                        to_drop.append((ii, pid))
                except Exception:
                    to_drop.append((ii, pid))
        if to_drop:
            for ii, pid in to_drop:
                v = lam_vars.pop((ii, pid), None)
                if v is None:
                    continue
                rmp.remove(v)
                ages.pop((ii, pid), None)
            rmp.update()
            if verbose:
                _log(f"[DROP] {len(to_drop)} cold columns")

        # duals
        try:
            pi_raw = [cap_con[t].Pi for t in range(T)]
            rho_raw = [inv_con[u].Pi for u in range(T - 1)] if use_wh else []
            sigma = {i: one_con[i].Pi for i in items_raw}
            gamma = {(i, t): ylink[(i, t)].Pi for i in items_raw for t in range(T)}
        except Exception:
            _log("[STOP] Duals not available.")
            break

        # stabilization (on pi/rho only)
        if stabilize and iter_no > 1:
            pi = [
                stab_alpha * p + (1 - stab_alpha) * lp for p, lp in zip(pi_raw, last_pi)
            ]
            rho = [
                stab_alpha * r + (1 - stab_alpha) * lr
                for r, lr in zip(rho_raw, last_rho)
            ]
        else:
            pi, rho = pi_raw, rho_raw
        last_pi, last_rho = pi, rho

        # pricing: per item MIP (multi-add)
        any_added = False
        worst_rc = 0.0

        for i in items_raw:
            gamma_it = {t: float(gamma.get((i, t), 0.0)) for t in range(T)}
            added = _price_and_add_for(i, pi, rho, {}, gamma_it)
            any_added = any_added or added

        if verbose:
            total_cols = sum(len(v) for v in pool.values())
            _log(
                f"[ROOT ITER] {iter_no} obj={obj_str} worst_rc={worst_rc:.6f} cols={total_cols}"
            )

        if not any_added:
            _log("[ROOT STOP] No negative reduced-cost columns.")
            break

        if iter_no >= max_iter:
            _log("[ROOT STOP] Max iterations reached.")
            break

    # ------- Tail-off phase: more pricing without stabilization -------
    if verbose:
        _log("[ROOT TAIL] Starting tail-off pricing without stabilization...")
    stabilize = False  # Disable stab for tail-off
    tail_iters = 10  # Tuned: 10 extra iters
    for _ in range(tail_iters):
        if time_limit and time.time() - t0 > time_limit:
            break
        rmp.optimize()
        try:
            pi_raw = [cap_con[t].Pi for t in range(T)]
            rho_raw = [inv_con[u].Pi for u in range(T - 1)] if use_wh else []
            sigma = {i: one_con[i].Pi for i in items_raw}
            gamma = {(i, t): ylink[(i, t)].Pi for i in items_raw for t in range(T)}
        except Exception:
            break
        pi, rho = pi_raw, rho_raw
        any_added = False
        for i in items_raw:
            gamma_it = {t: float(gamma.get((i, t), 0.0)) for t in range(T)}
            added = _price_and_add_for(i, pi, rho, {}, gamma_it)
            any_added = any_added or added
        if not any_added:
            _log("[ROOT TAIL] No more adds in tail-off.")
            break
    if verbose:
        _log("[ROOT TAIL] Tail-off complete.")

    # ------- Optional: Fix-and-Price Diving to get a strong incumbent --------
    warm_start = None
    if enable_diving:
        _log("[ROOT DIVE] starting fix-and-price diving...")

        def _pf(
            i: int,
            pi_vec: List[float],
            rho_vec: Optional[List[float]],
            gamma_it_local: Dict[int, float],
        ):
            return _price_and_add_for(i, pi_vec, rho_vec, {}, gamma_it_local)

        warm_start = _dive_fix_and_price(
            rmp,
            Ybin,
            items_raw,
            pool,
            lam_vars,
            cap_con,
            inv_con,
            one_con,
            T,
            use_wh,
            ylink,
            _pf,
            max_fix=None,
            reprice_iters=diving_reprice_iters,
            stab_alpha=stab_alpha,
            verbose=verbose,
        )
        _log("[ROOT DIVE] finished; warm start constructed.")

    # ------- Solve root MIP over pool for initial UB -------
    # Keep Ybar continuous here so root_lp and fractional Y can be read from LP
    rmp.optimize()
    root_lp = float(rmp.ObjVal) if rmp.Status == GRB.OPTIMAL else float("inf")
    best_lb = root_lp
    best_ub = float("inf")
    best_orders = []
    best_cap_slack_sum = 0.0
    best_wh_slack_sum = 0.0

    # Freeze slacks if zero
    cap_slack_sum_LP = sum(float(v.X) for v in cap_slack.values())
    wh_slack_sum_LP = sum(float(v.X) for v in inv_slack.values()) if use_wh else 0.0
    if cap_slack_sum_LP <= 1e-9:
        for v in cap_slack.values():
            v.UB = 0.0
    if use_wh and wh_slack_sum_LP <= 1e-9:
        for v in inv_slack.values():
            v.UB = 0.0

    if mip_gap:
        rmp.Params.MIPGap = float(mip_gap)
    remaining_time = max(1, int(time_limit - (time.time() - t0))) if time_limit else 0
    if remaining_time:
        rmp.Params.TimeLimit = remaining_time
    rmp.Params.OutputFlag = 1 if verbose else 0
    rmp.Params.NumericFocus = 1
    rmp.Params.MIPFocus = 3
    rmp.Params.Cuts = 2
    rmp.Params.Heuristics = 0.1
    rmp.Params.Presolve = 2

    # === B) Keep λ continuous by default (no flip to binary).
    # Optional: allow binary λ if user requests.
    var_types = {}
    if lambda_binary:
        for key, var in lam_vars.items():
            var_types[key] = var.VType
            var.VType = GRB.BINARY

    # warm start
    if warm_start is None:
        warm_start = {}
        for key, var in lam_vars.items():
            warm_start[key] = float(var.X) if hasattr(var, "X") else 0.0
    for i in items_raw:
        candidates = [
            (pid, warm_start.get((i, pid), 0.0)) for pid in range(len(pool.get(i, [])))
        ]
        if not candidates:
            continue
        pid_star = max(candidates, key=lambda z: z[1])[0]
        for pid in range(len(pool.get(i, []))):
            v = lam_vars.get((i, pid))
            if v:
                v.Start = 1.0 if pid == pid_star else 0.0

    # Before final MIP solve at root, set Ybar variables to binary so we get integer UB
    for yvar in Ybin.values():
        yvar.VType = GRB.BINARY
    rmp.update()
    rmp.optimize()

    if rmp.SolCount > 0 and rmp.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        best_ub = float(rmp.ObjVal)
        orders_txt: List[str] = []
        for i in pool:
            orders_txt.append(f"Item {i} — orders (t → qty)")
            prod_t = [0.0] * T
            ls_t = [0.0] * T
            for pid, pl in enumerate(pool[i]):
                val = float(lam_vars[(i, pid)].X) if (i, pid) in lam_vars else 0.0
                if val > EPS:
                    for t in range(T):
                        prod_t[t] += val * pl.prod_by_t[t]
                    for u in range(T):
                        ls_t[u] += val * pl.lost_sales_by_u[u]
            for t in range(T):
                if prod_t[t] > 1e-6:
                    orders_txt.append(f" {t:2d} → {prod_t[t]:8.3f}")
            orders_txt.append("")
        best_orders = orders_txt
        best_cap_slack_sum = sum(float(v.X) for v in cap_slack.values())
        best_wh_slack_sum = (
            sum(float(v.X) for v in inv_slack.values()) if use_wh else 0.0
        )

    # revert VType if we flipped
    if lambda_binary:
        for key, vtype in var_types.items():
            lam_vars[key].VType = vtype

    if abs(best_ub - root_lp) < 1e-5:
        _log("[ROOT] Optimal at root.")
        summary = {
            "status": int(GRB.OPTIMAL),
            "objective": best_ub,
            "best_bound": root_lp,
            "gap": 0.0,
            "runtime_sec": float(time.time() - t0),
            "solver_version": "lefo_bp_fast_ybar_v2",
            "n_items": len(items_raw),
            "T": T,
            "columns_total": sum(len(v) for v in pool.values()),
            "cap_slack_sum": best_cap_slack_sum,
            "wh_slack_sum": best_wh_slack_sum,
        }
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "orders.txt").write_text(
            "\n".join(best_orders), encoding="utf-8"
        )
        (Path(out_dir) / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary, best_orders

    # ------- Branch-and-Price tree -------
    if verbose:
        _log("[BP] Starting branch-and-price tree...")
    stack: List[Node] = []
    node_id = 1
    root_node = Node(0, None, {}, rf_rules=[])
    root_node.lp_bound = root_lp

    # helper to collect fractional Ybar candidates
    def _collect_fractional_Y():
        frac_map: Dict[int, List[Tuple[int, float]]] = {}
        for ii in items_raw:
            for tt in range(T):
                yv = float(Ybin[(ii, tt)].X)
                if EPS < yv < 1 - EPS:
                    frac_map.setdefault(ii, []).append((tt, yv))
        return frac_map

    # initial branching at root
    rmp.optimize()
    frac_map = _collect_fractional_Y()
    if frac_map:
        # D) try RF branching if possible
        if rf_branching and any(len(v) >= 2 for v in frac_map.values()):
            # pick the item with two most fractional Y's (closest to 0.5)
            ii = max(
                frac_map.items(),
                key=lambda kv: sum(
                    0.5 - abs(x[1] - 0.5)
                    for x in sorted(kv[1], key=lambda x: abs(x[1] - 0.5))[:2]
                ),
            )[0]
            twos = sorted(frac_map[ii], key=lambda x: abs(x[1] - 0.5))[:2]
            t1, _ = twos[0]
            t2, _ = twos[1]
            # child A: equality Y_{i,t1} == Y_{i,t2}
            childA = Node(node_id, root_node, {}, rf_rules=[("eq", ii, t1, t2)])
            node_id += 1
            stack.append(childA)
            # child B: XOR Y_{i,t1} + Y_{i,t2} == 1
            childB = Node(node_id, root_node, {}, rf_rules=[("xor", ii, t1, t2)])
            node_id += 1
            stack.append(childB)
        else:
            # fallback: single Y branching on the most fractional
            ii, lst = max(
                frac_map.items(),
                key=lambda kv: max(0.5 - abs(y - 0.5) for _, y in kv[1]),
            )
            tt, _ = sorted(lst, key=lambda x: abs(x[1] - 0.5))[0]
            branches0 = {ii: {tt: 0}}
            child0 = Node(node_id, root_node, branches0, rf_rules=[])
            node_id += 1
            stack.append(child0)
            branches1 = {ii: {tt: 1}}
            child1 = Node(node_id, root_node, branches1, rf_rules=[])
            node_id += 1
            stack.append(child1)
    else:
        _log("[BP] Root LP integer on Ybar, but not pruned earlier?")

    processed_nodes = 1  # root
    nogood_id_counter = 0

    while stack:
        current_time = time.time() - t0
        if time_limit and current_time > time_limit:
            _log("[BP STOP] Time limit hit.")
            break

        node = stack.pop()
        processed_nodes += 1
        if verbose:
            depth = sum(len(fixes) for fixes in node.branches.values()) + (
                len(node.rf_rules) if node.rf_rules else 0
            )
            _log(
                f"[BP] Node {node.id}, depth {depth}, stack {len(stack)}, processed {processed_nodes}"
            )

        # set bounds/constraints for node: kill columns inconsistent with fixed_y;
        # also add temporary equalities Ybar==val and RF cuts
        saved_bounds: Dict[Tuple[int, int], Tuple[float, float]] = {}
        temp_node_constrs: List[gp.Constr] = []

        # Apply fixed Y equalities to master and column pruning
        for ii, fixes in node.branches.items():
            # master-side Ybar equalities
            for tt, val in fixes.items():
                c = rmp.addConstr(
                    Ybin[(ii, tt)] == int(val), name=f"fixY_node{node.id}_{ii}_{tt}"
                )
                temp_node_constrs.append(c)
            # column pruning
            for pid in range(len(pool.get(ii, []))):
                key = (ii, pid)
                v = lam_vars.get(key)
                if not v:
                    continue
                saved_bounds[key] = (v.LB, v.UB)
                violate = False
                pl = pool[ii][pid]
                for tt, val in fixes.items():
                    y_plan = pl.y_values[tt]
                    if (val == 1 and y_plan < 1 - EPS) or (val == 0 and y_plan > EPS):
                        violate = True
                        break
                if violate:
                    v.UB = 0.0

        # Apply RF rules (eq/xor) at this node
        if node.rf_rules:
            for rtype, ii, t1, t2 in node.rf_rules:
                if rtype == "eq":
                    c = rmp.addConstr(
                        Ybin[(ii, t1)] - Ybin[(ii, t2)] == 0,
                        name=f"rf_eq_{node.id}_{ii}_{t1}_{t2}",
                    )
                else:  # xor
                    c = rmp.addConstr(
                        Ybin[(ii, t1)] + Ybin[(ii, t2)] == 1,
                        name=f"rf_xor_{node.id}_{ii}_{t1}_{t2}",
                    )
                temp_node_constrs.append(c)

        # Ensure Ybar variables are continuous during node CG so γ duals are available
        for yvar in Ybin.values():
            yvar.VType = GRB.CONTINUOUS
        rmp.update()

        # CG at node
        iter_no = 0
        last_pi = [0.0] * T
        last_rho = [0.0] * (T - 1 if use_wh else 0)
        max_iter_node = (
            max_iter // 10 if len(stack) > 0 else max_iter
        )  # smaller for non-root
        while True:
            iter_no += 1
            if time_limit and time.time() - t0 > time_limit:
                break
            if iter_no > max_iter_node:
                _log(f"[BP NODE {node.id}] Max iter hit.")
                break

            rmp.optimize()
            try:
                obj_now = float(rmp.ObjVal)
            except Exception:
                break

            # drop cold as before
            to_drop = []
            for (ii, pid), age in ages.items():
                if age > drop_age and (ii, pid) in lam_vars and len(pool[ii]) > 5:
                    try:
                        if float(lam_vars[(ii, pid)].X) <= 1e-10:
                            to_drop.append((ii, pid))
                    except Exception:
                        to_drop.append((ii, pid))
            if to_drop:
                for ii, pid in to_drop:
                    v = lam_vars.pop((ii, pid), None)
                    if v is None:
                        continue
                    rmp.remove(v)
                    ages.pop((ii, pid), None)
                rmp.update()

            # duals
            try:
                pi_raw = [cap_con[t].Pi for t in range(T)]
                rho_raw = [inv_con[u].Pi for u in range(T - 1)] if use_wh else []
                sigma = {ii: one_con[ii].Pi for ii in items_raw}
                gamma = {
                    (ii, tt): ylink[(ii, tt)].Pi for ii in items_raw for tt in range(T)
                }
            except Exception:
                break

            # stab
            if stabilize and iter_no > 1:
                pi = [
                    stab_alpha * p + (1 - stab_alpha) * lp
                    for p, lp in zip(pi_raw, last_pi)
                ]
                rho = [
                    stab_alpha * r + (1 - stab_alpha) * lr
                    for r, lr in zip(rho_raw, last_rho)
                ]
            else:
                pi, rho = pi_raw, rho_raw
            last_pi, last_rho = pi, rho

            any_added = False
            worst_rc = 0.0
            for ii in items_raw:
                fixed_y_ii = node.branches.get(ii, {})
                gamma_it = {t: float(gamma.get((ii, t), 0.0)) for t in range(T)}
                added = _price_and_add_for(ii, pi, rho, fixed_y_ii, gamma_it)
                any_added = any_added or added

            if not any_added:
                break

        # node LP bound
        rmp.optimize()
        if rmp.Status == GRB.INFEASIBLE:
            node_lp = float("inf")
        else:
            node_lp = float(rmp.ObjVal) if rmp.Status == GRB.OPTIMAL else float("inf")
        node.lp_bound = node_lp

        if node_lp >= best_ub - EPS:
            _log(f"[BP NODE {node.id}] Pruned by bound {node_lp:.2f} >= {best_ub:.2f}")
            # revert
            for key, (lb, ub) in saved_bounds.items():
                lam_vars[key].LB = lb
                lam_vars[key].UB = ub
            # remove temporary node constraints
            for c in temp_node_constrs:
                rmp.remove(c)
            rmp.update()
            continue

        # Before solving MIP: collect fractional Ybar from LP to decide branching
        rmp.optimize()  # ensure LP solution available
        frac_map = {}
        for ii in items_raw:
            yfr = []
            for tt in range(T):
                yv = float(Ybin[(ii, tt)].X)
                if EPS < yv < 1 - EPS:
                    if ii in node.branches and tt in node.branches[ii]:
                        continue
                    yfr.append((tt, yv))
            if yfr:
                frac_map[ii] = yfr

        # Before solving node MIP: set Ybar variables to binary
        for yvar in Ybin.values():
            yvar.VType = GRB.BINARY
        rmp.update()

        # solve MIP at node for possible better UB
        var_types = {}
        if lambda_binary:
            for key, var in lam_vars.items():
                var_types[key] = var.VType
                var.VType = GRB.BINARY
        remaining_time_node = (
            max(1, int(time_limit - (time.time() - t0))) if time_limit else 0
        )
        if remaining_time_node:
            rmp.Params.TimeLimit = remaining_time_node
        rmp.Params.OutputFlag = 0
        rmp.optimize()

        if rmp.SolCount > 0:
            mip_val = float(rmp.ObjVal)
            if mip_val < best_ub - EPS:
                best_ub = mip_val
                orders_txt = []
                for ii in pool:
                    orders_txt.append(f"Item {ii} — orders (t → qty)")
                    prod_t = [0.0] * T
                    ls_t = [0.0] * T
                    for pid, pl in enumerate(pool[ii]):
                        val = (
                            float(lam_vars[(ii, pid)].X)
                            if (ii, pid) in lam_vars
                            else 0.0
                        )
                        if val > EPS:
                            for t in range(T):
                                prod_t[t] += val * pl.prod_by_t[t]
                            for u in range(T):
                                ls_t[u] += val * pl.lost_sales_by_u[u]
                    for t in range(T):
                        if prod_t[t] > 1e-6:
                            orders_txt.append(f" {t:2d} → {prod_t[t]:8.3f}")
                    orders_txt.append("")
                best_orders = orders_txt
                best_cap_slack_sum = sum(float(v.X) for v in cap_slack.values())
                best_wh_slack_sum = (
                    sum(float(v.X) for v in inv_slack.values()) if use_wh else 0.0
                )
                _log(f"[BP NODE {node.id}] New UB {best_ub:.2f}")

            if abs(mip_val - node_lp) < 1e-5:
                _log(f"[BP NODE {node.id}] Pruned by optimality at node")
                # revert types and bounds
                if lambda_binary:
                    for key, vtype in var_types.items():
                        lam_vars[key].VType = vtype
                for key, (lb, ub) in saved_bounds.items():
                    lam_vars[key].LB = lb
                    lam_vars[key].UB = ub
                # remove temporary node constraints
                for c in temp_node_constrs:
                    rmp.remove(c)
                rmp.update()
                # Optional: add nogood on Y patterns (disabled by default)
                if enable_nogood_cuts:
                    for ii in items_raw:
                        patt = [int(round(float(Ybin[(ii, t)].X))) for t in range(T)]
                        add_no_good_cut_for_pattern(
                            rmp, Ybin, ii, patt, T, nogood_id_counter
                        )
                        nogood_id_counter += 1
                continue

        # revert λ types if flipped
        if lambda_binary:
            for key, vtype in var_types.items():
                lam_vars[key].VType = vtype

        # Branch if not pruned - frac_map was collected before MIP
        if not frac_map:
            _log(f"[BP NODE {node.id}] No fractional Ybar, but not pruned?")
            # revert bounds and remove temp constraints
            for key, (lb, ub) in saved_bounds.items():
                lam_vars[key].LB = lb
                lam_vars[key].UB = ub
            for c in temp_node_constrs:
                rmp.remove(c)
            rmp.update()
            continue

        # D) RF branching if possible: pick an item with >=2 fractional Y's
        if rf_branching and any(len(v) >= 2 for v in frac_map.values()):
            ii = max(
                frac_map.items(),
                key=lambda kv: sum(
                    0.5 - abs(x[1] - 0.5)
                    for x in sorted(kv[1], key=lambda x: abs(x[1] - 0.5))[:2]
                ),
            )[0]
            twos = sorted(frac_map[ii], key=lambda x: abs(x[1] - 0.5))[:2]
            t1, _ = twos[0]
            t2, _ = twos[1]
            # create RF children inheriting node's constraints
            childA = Node(
                node_id,
                node,
                {k: v.copy() for k, v in node.branches.items()},
                rf_rules=(node.rf_rules[:] if node.rf_rules else [])
                + [("eq", ii, t1, t2)],
            )
            node_id += 1
            stack.append(childA)
            childB = Node(
                node_id,
                node,
                {k: v.copy() for k, v in node.branches.items()},
                rf_rules=(node.rf_rules[:] if node.rf_rules else [])
                + [("xor", ii, t1, t2)],
            )
            node_id += 1
            stack.append(childB)
        else:
            # fallback: branch on single most fractional Y_{i,t}
            ii, lst = max(
                frac_map.items(),
                key=lambda kv: max(0.5 - abs(y - 0.5) for _, y in kv[1]),
            )
            tt, _ = sorted(lst, key=lambda x: abs(x[1] - 0.5))[0]
            # 0 branch
            branches0 = {key: val.copy() for key, val in node.branches.items()}
            if ii not in branches0:
                branches0[ii] = {}
            branches0[ii][tt] = 0
            child0 = Node(
                node_id,
                node,
                branches0,
                rf_rules=(node.rf_rules[:] if node.rf_rules else []),
            )
            node_id += 1
            stack.append(child0)
            # 1 branch
            branches1 = {key: val.copy() for key, val in node.branches.items()}
            if ii not in branches1:
                branches1[ii] = {}
            branches1[ii][tt] = 1
            child1 = Node(
                node_id,
                node,
                branches1,
                rf_rules=(node.rf_rules[:] if node.rf_rules else []),
            )
            node_id += 1
            stack.append(child1)

        # revert bounds and remove temp constraints
        for key, (lb, ub) in saved_bounds.items():
            lam_vars[key].LB = lb
            lam_vars[key].UB = ub
        for c in temp_node_constrs:
            rmp.remove(c)
        rmp.update()

    # ------- Prepare output -------
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if best_orders:
        (Path(out_dir) / "orders.txt").write_text(
            "\n".join(best_orders), encoding="utf-8"
        )
    gap = (
        abs(best_ub - root_lp) / best_ub
        if best_ub < float("inf") and best_ub > 0
        else None
    )
    status = (
        GRB.OPTIMAL
        if not stack
        else (
            GRB.TIME_LIMIT
            if time_limit and time.time() - t0 > time_limit
            else GRB.INTERRUPTED
        )
    )
    if not stack and best_ub < float("inf"):
        root_lp = best_ub
        gap = 0.0
    summary = {
        "status": int(status),
        "objective": best_ub if best_ub < float("inf") else None,
        "best_bound": root_lp,
        "gap": gap,
        "runtime_sec": float(time.time() - t0),
        "solver_version": "lefo_bp_fast_ybar_v2",
        "n_items": len(items_raw),
        "T": T,
        "columns_total": sum(len(v) for v in pool.values()),
        "cap_slack_sum": best_cap_slack_sum,
        "wh_slack_sum": best_wh_slack_sum,
        "nodes_processed": processed_nodes,
    }
    (Path(out_dir) / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if best_ub >= float("inf"):
        summary["note"] = "No incumbent found."
    return summary, best_orders


# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="last_instance.json")
    p.add_argument("--out", default="bp_fast_results")

    # pricing / CG döngüsü
    p.add_argument("--stab_off", action="store_true")
    p.add_argument("--stab_alpha", type=float, default=0.8)
    p.add_argument("--max_iter", type=int, default=50000)
    p.add_argument("--max_add_per_item_per_iter", type=int, default=5)
    p.add_argument("--pricing_k", type=int, default=5)
    p.add_argument("--drop_age", type=int, default=10)

    # finalize / limitler
    p.add_argument("--finalize_off", action="store_true")
    p.add_argument("--time_limit", type=int, default=0)
    p.add_argument("--mip_gap", type=float, default=0.0)

    # semantics (daha anlaşılır bayraklar)
    p.add_argument(
        "--exclusive_shelf",
        action="store_true",
        help="Shelf ömrünü dışlayıcı (u ≤ t+L-1) yapar. Varsayılan: dahil edici (u ≤ t+L).",
    )
    p.add_argument(
        "--allow_lost_sales",
        action="store_true",
        help="Lost sales’a izin ver. Varsayılan: instance'a göre.",
    )

    # diving
    p.add_argument("--diving_off", action="store_true")
    p.add_argument("--diving_reprice_iters", type=int, default=40)

    # dummy outsourcing cost (opsiyonel)
    p.add_argument("--outsource_unit_cost", type=float, default=None)

    # === NEW flags ===
    p.add_argument(
        "--lambda_binary",
        action="store_true",
        help="If set, flip λ to binary in pool MIP; default keeps λ continuous.",
    )
    p.add_argument(
        "--rf_branching_off",
        action="store_true",
        help="Disable Ryan–Foster branching (use single Y branching only).",
    )
    p.add_argument(
        "--nogood_on",
        action="store_true",
        help="Enable no-good cuts on discovered Y patterns (use with care; off by default).",
    )

    args = p.parse_args()

    summary, orders = solve_instance(
        instance_path=args.instance,
        out_dir=args.out,
        stabilize=(not args.stab_off),
        stab_alpha=args.stab_alpha,
        max_iter=args.max_iter,
        max_add_per_item_per_iter=args.max_add_per_item_per_iter,
        pricing_k=args.pricing_k,
        drop_age=args.drop_age,
        finalize_as_mip=(not args.finalize_off),
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        inclusive_shelf=(not args.exclusive_shelf),  # <-- DÜZGÜN
        force_no_lost_sales=(not args.allow_lost_sales),  # <-- Adjusted for consistency
        enable_diving=(not args.diving_off),
        diving_reprice_iters=args.diving_reprice_iters,
        verbose=True,
        lambda_binary=args.lambda_binary,  # B)
        rf_branching=(not args.rf_branching_off),  # D)
        enable_nogood_cuts=args.nogood_on,  # C) (optional)
    )
    print(json.dumps(summary, indent=2))
