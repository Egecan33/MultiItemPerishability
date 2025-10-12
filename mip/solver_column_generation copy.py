# lefo_cg_solver_fast.py
# --------------------------------------------------------------------------------------
# Column Generation solver for perishable lot-sizing with LEFO (no-crossing) blocks.
# This version starts from a FEASIBLE root with per-item DUMMY (outsourcing) columns
# that consume ZERO capacity and ZERO warehouse but have a LARGE per-unit cost.
# Then it repeatedly solves the pricing subproblem (DP with LEFO) using RMP duals,
# adds negative reduced-cost columns, and finally flips λ to binary to get an exact MIP.
#
# I/O is unchanged: solve_instance(instance_path, out_dir, ...) returns (summary, orders).
# CLI also unchanged, with an extra optional flag: --outsource_unit_cost
# --------------------------------------------------------------------------------------

from __future__ import annotations
import time, json, math, random, sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Set
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

EPS = 1e-9
RC_EPS = 1e-7


@dataclass
class ColumnPlan:
    item: int
    plan_id: int
    cost: float
    prod_by_t: List[float]  # production qty at period t
    inv_end_by_u: List[float]  # inventory at end of u (0..T-2)
    flows: List[Tuple[int, int, float]]  # (t,u,qty)
    setups: List[int]  # t with prod>0
    lost_sales_by_u: List[float]  # per-period lost sales


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
    return ColumnPlan(
        i, -1, cost, prod_by_t, inv_end_by_u, flows, setups, lost_sales_by_u
    )


# ---------------- pricing (DP, LEFO) ----------------


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


def price_item_plan(
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
    rho_wh: Optional[List[float]] = None,
    vlast_floor: int = -(10**9),
    rc_jitter: Optional[Callable[[int, int, int], float]] = None,
):
    """
    DP pricing with LEFO. Uses fast block-cost formula:
      base(t,s,e) = setup(i,t) + (c(i,t) - hpref(i,t)) * Q(s,e) + (Hdem[e+1] - Hdem[s])

    Reduced cost of a plan λ is:
      rc = true_cost(λ) - sum_t pi_cap[t] * prod_t(λ) - sum_u rho_wh[u] * inv_end_u(λ)
           - sigma_i   (sigma handled by master when deciding to add)
    """
    d = [float(x) for x in items_raw[i]["demand"]]
    pref_d = [0.0] * (T + 1)
    for u in range(T):
        pref_d[u + 1] = pref_d[u] + d[u]

    # infer hprefix via hsum(i,0,u)
    def hpref_local(u: int) -> float:
        return hsum(i, 0, u)

    Hdem = [0.0] * (T + 1)
    for u in range(T):
        Hdem[u + 1] = Hdem[u] + d[u] * hpref_local(u)

    # enumerate all candidate blocks (t,s,e)
    blocks = []
    for t in range(T):
        if not Gamma.get((i, t)):
            continue
        v_exp = Expiry[(i, t)]  # exclusive
        e_max = min(T - 1, v_exp - 1)
        if e_max < t:
            continue
        pit = per_item_cap.get((i, t), math.inf)
        Ct = c_at(i, t)
        hpt = hpref_local(t)
        setup_t = s_at(i, t)

        # sliding window on [s..e] to satisfy per-item cap
        run = 0.0
        e = t - 1
        for s in range(t, e_max + 1):
            if e < s - 1:
                e = s - 1
                run = 0.0
            while e + 1 <= e_max and run + d[e + 1] <= pit + EPS:
                e += 1
                run += d[e]
            if run <= EPS:
                continue

            Q = pref_d[e + 1] - pref_d[s]
            base = setup_t + (Ct - hpt) * Q + (Hdem[e + 1] - Hdem[s])

            # ---- reduced-cost adjustments (subtract duals) ----
            rc = base
            # capacity at t
            rc -= pi_cap[t] * Q

            if rho_wh is not None and T >= 2:
                # full Q carried until s-1
                if s > t:
                    for uu in range(t, min(s - 1, T - 2) + 1):
                        rc -= rho_wh[uu] * Q
                # then deplete over s..e
                cons = 0.0
                for uu in range(s, e + 1):
                    cons += d[uu]
                    inv_end = max(Q - cons, 0.0)
                    if uu < T - 1 and inv_end > 0.0:
                        rc -= rho_wh[uu] * inv_end

            if rc_jitter is not None:
                rc += rc_jitter(t, s, e)

            blocks.append((t, s, e, Q, rc))

            # move s -> s+1
            if d[s] > EPS:
                run -= d[s]

    # DP with LEFO: expiry nondecreasing across chosen blocks
    exp_values = sorted(set(Expiry[(i, t)] for t in range(T)))
    exp_to_idx = {v: k for k, v in enumerate(exp_values)}

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(s: int, v_last_idx: int):
        if s >= T:
            return 0.0, None
        # skip zero-demand periods
        u = s
        while u < T and d[u] <= EPS:
            u += 1
        if u >= T:
            return 0.0, None

        best = float("inf")
        choice = None
        v_last = (
            exp_values[v_last_idx] if 0 <= v_last_idx < len(exp_values) else vlast_floor
        )

        # lost sale option (disable if allow_lost_sales=False)
        if allow_lost_sales and d[u] > EPS:
            c_ls = loss_pen[(i, u)] * d[u]
            nxt, _ = dp(u + 1, v_last_idx)
            val = c_ls + nxt
            if val < best - 1e-12:
                best, choice = val, ("LS", u)

        # try blocks starting at u
        for t, ss, e, Q, rc in blocks:
            if ss != u:
                continue
            v_new = Expiry[(i, t)]
            if v_new < v_last:
                continue
            nxt, _ = dp(e + 1, exp_to_idx[v_new])
            val = rc + nxt
            if val < best - 1e-12:
                best, choice = val, ("BLK", t, ss, e, Q)
        return best, choice

    best_rc, _ = dp(0, -1)

    # reconstruct plan
    flows: List[Tuple[int, int, float]] = []
    prod_by_t = [0.0] * T
    lost_by_u = [0.0] * T
    s = 0
    v_idx = -1
    while s < T:
        while s < T and d[s] <= EPS:
            s += 1
        if s >= T:
            break
        _, ch = dp(s, v_idx)
        if ch is None:
            # If lost sales are forbidden, we must keep covering; safety fallback
            s += 1
            continue
        if ch[0] == "LS":
            _, u = ch
            lost_by_u[u] += d[u]
            s = u + 1
        else:
            _, t, ss, e, Q = ch
            for u in range(ss, e + 1):
                q_u = d[u]
                if q_u > EPS:
                    flows.append((t, u, q_u))
            prod_by_t[t] += Q
            v_new = Expiry[(i, t)]
            v_idx = exp_to_idx[v_new]
            s = e + 1

    # inventories
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

    # true cost (not RC): base costs + LS penalties
    cost = 0.0
    used_t = set()
    for t, u, q in flows:
        cost += (c_at(i, t) + hsum(i, t, u)) * q
        used_t.add(t)
    for t in used_t:
        cost += s_at(i, t)
    if allow_lost_sales:
        for u in range(T):
            if lost_by_u[u] > EPS:
                cost += loss_pen[(i, u)] * lost_by_u[u]

    # LEFO check (debug)
    if flows:
        # LEFO güvence kontrolü (debug)
        try:
            _check_lefo_flows(i, flows, Expiry)
        except RuntimeError as e:
            _log(f"[WARN] LEFO check failed in pricing for item {i}: {e}")
            # İstersen burada farklı bir strateji uygulayabilirsin; biz sadece uyarı basıyoruz.

    return ColumnPlan(
        i, -1, float(cost), prod_by_t, inv_end, flows, setups, lost_by_u
    ), float(best_rc)


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
) -> List[Tuple[ColumnPlan, float]]:
    """
    Get up to k DISTINCT plans by rerunning DP with tiny jitters.
    """
    rnd = random.Random(seed)
    seen: Set[Tuple[Tuple[int, int, float], ...]] = set()
    out: List[Tuple[ColumnPlan, float]] = []
    for r in range(max(1, k)):
        # tiny jitter based on (t,s,e) and r, symmetric around 0
        def _jit(t: int, s: int, e: int) -> float:
            # scale tied to magnitude of costs; keep extremely small
            return (rnd.random() - 0.5) * 1e-9 * (1 + r)

        pl, rc = price_item_plan(
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
            rc_jitter=_jit,
        )
        # unique by flows
        key = tuple((t, u, float(f"{q:.9f}")) for (t, u, q) in sorted(pl.flows))
        if key in seen:
            continue
        seen.add(key)
        out.append((pl, rc))
    return out


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
    used_t = set()
    cost = 0.0
    for t, u, q in flows:
        cost += (c_at(i, t) + hsum(i, t, u)) * q
        used_t.add(t)
    for t in used_t:
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
    setups = sorted(list(used_t))
    return ColumnPlan(i, -1, float(cost), prod_by_t, inv_end, flows, setups, lost)


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
    used_t = set()
    cost = 0.0
    for t, u, q in flows:
        cost += (c_at(i, t) + hsum(i, t, u)) * q
        used_t.add(t)
    for t in used_t:
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
    setups = sorted(list(used_t))
    return ColumnPlan(i, -1, float(cost), prod_by_t, inv_end, flows, setups, lost)


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

    used = set()
    cost = 0.0
    for t, u, q in flows:
        cost += (c_at(i, t) + hsum(i, t, u)) * q
        used.add(t)
    for t in used:
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
    return ColumnPlan(
        i, -1, float(cost), prod_by_t, inv_end, flows, sorted(set(setups)), [0.0] * T
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

    used = set()
    cost = 0.0
    for t, u2, q in flows:
        cost += (c_at(i, t) + hsum(i, t, u2)) * q
        used.add(t)
    for t in used:
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
    return ColumnPlan(
        i, -1, float(cost), prod_by_t, inv_end, flows, sorted(list(used)), [0.0] * T
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
    return rmp, cap_con, inv_con, one_con


# ---------------- fix-and-price diving ----------------


def _dive_fix_and_price(
    rmp: gp.Model,
    items_raw: Dict[int, dict],
    pool: Dict[int, List[ColumnPlan]],
    lam_vars: Dict[Tuple[int, int], gp.Var],
    cap_con: Dict[int, gp.Constr],
    inv_con: Dict[int, gp.Constr],
    one_con: Dict[int, gp.Constr],
    T: int,
    use_wh: bool,
    price_fn_per_item: Callable[[int, List[float], Optional[List[float]]], bool],
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
        return pi, rho, sigma

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
        for _ in range(reprice_iters):
            rmp.optimize()
            pi_raw, rho_raw, _ = _duals()
            pi = [
                stab_alpha * p + (1 - stab_alpha) * lp for p, lp in zip(pi_raw, last_pi)
            ]
            rho = [
                stab_alpha * r + (1 - stab_alpha) * lr
                for r, lr in zip(rho_raw, last_rho)
            ]
            last_pi, last_rho = pi, rho
            any_add = False
            for i in items_raw:
                if i in fixed:
                    continue  # don't price fixed item
                added = price_fn_per_item(i, pi, rho)
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
    return warm_start


# ---------------- main ----------------


def solve_instance(
    instance_path: str | Path = "last_instance.json",
    out_dir: str | Path = "cg_fast_results",
    # Seed knobs (unused with dummy-only, but kept for API)
    seed_random_per_item: int = 0,
    seed_chunk_len: int = 3,
    seed_extra_chunked: bool = False,
    # Pricing knobs
    stabilize: bool = True,
    stab_alpha: float = 0.6,
    max_iter: int = 40000,
    max_add_per_item_per_iter: int = 3,
    pricing_k: int = 3,
    drop_age: int = 10,
    # final
    finalize_as_mip: bool = True,
    # limits
    time_limit: int = 0,
    mip_gap: float = 0.0,
    # semantics/feasibility
    inclusive_shelf: bool = False,  # L covers this + next L (inclusive shelf-life)
    force_no_lost_sales: bool = True,  # forbid lost sales regardless of instance
    # diving
    enable_diving: bool = True,
    diving_reprice_iters: int = 40,
    verbose: bool = True,
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

    # semantics / feasibility overrides
    allow_lost_sales = False if force_no_lost_sales else allow_lost_sales_in

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
    rmp, cap_con, inv_con, one_con = build_rmp(T, prod_cap, use_wh, W, items_raw)
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
        col.addTerms(1.0, one_con[i])
        v = rmp.addVar(lb=0.0, obj=float(pl.cost), name=f"lam_{i}_{pid}", column=col)
        lam_vars[(i, pid)] = v
        ages[(i, pid)] = 0
        rmp.update()
        if verbose:
            setups_log = sorted({t for t, q in enumerate(pl.prod_by_t) if q > EPS})
            _log(f"[ADD] i={i} plan={pid} cost={pl.cost:.3f} setups={setups_log}")

    # ------- Seed: ONLY DUMMY outsourcing columns -------
    if verbose:
        _log("[SEED] Adding per-item DUMMY outsourcing columns (feasible root).")
    for i in items_raw:
        pl_dummy = seed_plan_dummy_outsource(
            i, items_raw, T, unit_cost=outsource_unit_cost
        )
        add_column(i, pl_dummy)

    # ------- CG loop -------
    iter_no = 0
    last_pi = [0.0] * T
    last_rho = [0.0] * (T - 1 if use_wh else 0)

    while True:
        iter_no += 1
        if time_limit and time.time() - t0 > time_limit:
            _log("[STOP] Time limit hit.")
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
        except Exception:
            _log("[STOP] Duals not available.")
            break

        # stabilization
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

        # pricing: per item DP (multi-add)
        any_added = False
        worst_rc = 0.0

        def _price_and_add_for(
            i: int, pi_vec: List[float], rho_vec: Optional[List[float]]
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
                seed=iter_no * 7919 + i * 104729,
            )
            cand_list.sort(key=lambda tup: tup[1])
            added_for_i = 0
            added_local = False
            for plan, rc_wo_sigma in cand_list:
                rc_total = rc_wo_sigma - sigma[i]
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

        for i in items_raw:
            added = _price_and_add_for(i, pi, rho)
            any_added = any_added or added

        if verbose:
            total_cols = sum(len(v) for v in pool.values())
            _log(
                f"[ITER] {iter_no} obj={obj_str} worst_rc={worst_rc:.6f} cols={total_cols}"
            )

        if not any_added:
            _log("[STOP] No negative reduced-cost columns.")
            break

        if iter_no >= max_iter:
            _log("[STOP] Max iterations reached.")
            break

    # ------- Optional: Fix-and-Price Diving to get a strong incumbent --------
    warm_start = None
    if enable_diving:
        _log("[DIVE] starting fix-and-price diving...")

        def _pf(i: int, pi_vec: List[float], rho_vec: Optional[List[float]]):
            # price and add up to 'pricing_k' plans for item i at current duals
            return _price_and_add_for(i, pi_vec, rho_vec)

        warm_start = _dive_fix_and_price(
            rmp,
            items_raw,
            pool,
            lam_vars,
            cap_con,
            inv_con,
            one_con,
            T,
            use_wh,
            _pf,
            max_fix=None,
            reprice_iters=diving_reprice_iters,
            stab_alpha=stab_alpha,
            verbose=verbose,
        )
        _log("[DIVE] finished; warm start constructed.")

    # ------- Finalize as MILP (exact) by flipping λ to binary on the SAME RMP -------
    if finalize_as_mip:
        # If LP slacks are exactly zero, freeze them off to strengthen MILP & numerics
        rmp.optimize()
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
        if time_limit:
            rmp.Params.TimeLimit = max(1, int(time_limit - (time.time() - t0)))
        rmp.Params.OutputFlag = 1
        rmp.Params.NumericFocus = 1
        rmp.Params.MIPFocus = 3  # emphasize bound to prove optimality
        rmp.Params.Cuts = 2
        rmp.Params.Heuristics = 0.1
        rmp.Params.Presolve = 2

        # flip lam_ vars to binary; keep slacks continuous
        for (i, pid), var in lam_vars.items():
            var.VType = GRB.BINARY

        # Warm start from dive (or from current LP rounding if no dive)
        if warm_start is None:
            warm_start = {}
            rmp.optimize()
            for (i, pid), var in lam_vars.items():
                try:
                    warm_start[(i, pid)] = float(var.X)
                except Exception:
                    warm_start[(i, pid)] = 0.0
        # Convert to a one-hot start per item
        for i in items_raw:
            candidates = [
                (pid, warm_start.get((i, pid), 0.0)) for pid in range(len(pool[i]))
            ]
            if not candidates:
                continue
            pid_star, _ = max(candidates, key=lambda z: z[1])
            for pid in range(len(pool[i])):
                v = lam_vars.get((i, pid))
                if v is None:
                    continue
                v.Start = 1.0 if pid == pid_star else 0.0

        rmp.ModelSense = GRB.MINIMIZE
        rmp.optimize()

        # If no incumbent
        if rmp.SolCount == 0:
            try:
                rmp.computeIIS()
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                rmp.write(str(Path(out_dir) / "infeasible.ilp"))
            except gp.GurobiError:
                pass
            summary = {
                "status": int(rmp.Status),
                "objective": None,
                "best_bound": float(getattr(rmp, "ObjBound", float("nan"))),
                "gap": None,
                "runtime_sec": float(time.time() - t0),
                "solver_version": "lefo_cg_fast_exact_v3",
                "n_items": len(items_raw),
                "T": T,
                "columns_total": sum(len(v) for v in pool.values()),
                "cap_slack_sum": 0.0,
                "wh_slack_sum": 0.0,
                "note": "MILP over current column pool had no incumbent; IIS written if possible.",
            }
            return summary, []

        # write outputs
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        orders_txt: List[str] = []
        for i, plist in pool.items():
            orders_txt.append(f"Item {i} — orders (t → qty)")
            prod_t = [0.0] * T
            ls_t = [0.0] * T
            for pl in plist:
                try:
                    val = float(lam_vars[(i, pl.plan_id)].X)
                except Exception:
                    val = 0.0
                if val > EPS:
                    for t in range(T):
                        prod_t[t] += val * pl.prod_by_t[t]
                    for u in range(T):
                        ls_t[u] += val * pl.lost_sales_by_u[u]
            for t in range(T):
                if prod_t[t] > 1e-6:
                    orders_txt.append(f" {t:2d} → {prod_t[t]:8.3f}")
            orders_txt.append("")
        (Path(out_dir) / "orders.txt").write_text(
            "\n".join(orders_txt), encoding="utf-8"
        )

        def _safe(mdl, attr, default=None):
            try:
                return float(getattr(mdl, attr))
            except Exception:
                return default

        cap_slack_sum = sum(
            (float(v.X) if hasattr(v, "X") else 0.0) for v in cap_slack.values()
        )
        wh_slack_sum = (
            sum((float(v.X) if hasattr(v, "X") else 0.0) for v in inv_slack.values())
            if use_wh
            else 0.0
        )

        summary = {
            "status": int(rmp.Status),
            "objective": _safe(rmp, "ObjVal", None),
            "best_bound": _safe(rmp, "ObjBound", None),
            "gap": _safe(rmp, "MIPGap", 0.0),
            "runtime_sec": float(time.time() - t0),
            "solver_version": "lefo_cg_fast_exact_v3",
            "n_items": len(items_raw),
            "T": T,
            "columns_total": sum(len(v) for v in pool.values()),
            "cap_slack_sum": float(cap_slack_sum),
            "wh_slack_sum": float(wh_slack_sum),
        }
        (Path(out_dir) / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        if cap_slack_sum > 1e-6 or wh_slack_sum > 1e-6:
            _log(
                "[WARN] Feasibility slacks positive in final solution — consider more CG iterations/seeds."
            )

        return summary, orders_txt

    # ------- LP output (not exact) -------
    rmp.optimize()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    orders_txt: List[str] = []
    for i, plist in pool.items():
        orders_txt.append(f"Item {i} — orders (t → qty)")
        prod_t = [0.0] * T
        ls_t = [0.0] * T
        for pl in plist:
            try:
                val = float(lam_vars[(i, pl.plan_id)].X)
            except Exception:
                val = 0.0
            if val > EPS:
                for t in range(T):
                    prod_t[t] += val * pl.prod_by_t[t]
                for u in range(T):
                    ls_t[u] += val * pl.lost_sales_by_u[u]
        for t in range(T):
            if prod_t[t] > 1e-6:
                orders_txt.append(f" {t:2d} → {prod_t[t]:8.3f}")
        orders_txt.append("")
    (Path(out_dir) / "orders.txt").write_text("\n".join(orders_txt), encoding="utf-8")

    summary = {
        "status": int(rmp.Status),
        "objective": (
            float(getattr(rmp, "ObjVal", 0.0)) if hasattr(rmp, "ObjVal") else None
        ),
        "best_bound": (
            float(getattr(rmp, "ObjBound", 0.0)) if hasattr(rmp, "ObjBound") else None
        ),
        "gap": 0.0,
        "runtime_sec": float(time.time() - t0),
        "solver_version": "lefo_cg_fast_lp_v3",
        "n_items": len(items_raw),
        "T": T,
        "columns_total": sum(len(v) for v in pool.values()),
        "cap_slack_sum": float(
            sum((float(v.X) if hasattr(v, "X") else 0.0) for v in cap_slack.values())
        ),
        "wh_slack_sum": (
            float(
                sum(
                    (float(v.X) if hasattr(v, "X") else 0.0) for v in inv_slack.values()
                )
            )
            if use_wh
            else 0.0
        ),
    }
    (Path(out_dir) / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary, orders_txt


# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="last_instance.json")
    p.add_argument("--out", default="cg_fast_results")

    # pricing / CG döngüsü
    p.add_argument("--stab_off", action="store_true")
    p.add_argument("--stab_alpha", type=float, default=0.6)
    p.add_argument("--max_iter", type=int, default=40000)
    p.add_argument("--max_add_per_item_per_iter", type=int, default=3)
    p.add_argument("--pricing_k", type=int, default=3)
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
        help="Lost sales’a izin ver. Varsayılan: izin yok.",
    )

    # diving
    p.add_argument("--diving_off", action="store_true")
    p.add_argument("--diving_reprice_iters", type=int, default=40)

    # dummy outsourcing cost (opsiyonel)
    p.add_argument("--outsource_unit_cost", type=float, default=None)

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
        force_no_lost_sales=(not args.allow_lost_sales),  # <-- DÜZGÜN
        enable_diving=(not args.diving_off),
        diving_reprice_iters=args.diving_reprice_iters,
        verbose=True,
    )
    print(json.dumps(summary, indent=2))
