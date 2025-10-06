from __future__ import annotations
import time, json, math, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
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


def _precompute_gamma_and_expiry(items_raw: Dict[int, dict], T: int):
    Gamma: Dict[Tuple[int, int], List[int]] = {}
    Expiry: Dict[Tuple[int, int], int] = {}
    for i, it in items_raw.items():
        mseq = list(it["shelf_seq"])
        if len(mseq) != T:
            raise ValueError(f"items[{i}]['shelf_seq'] must have length {T}")
        for t in range(T):
            m_it = int(mseq[t])
            v_it = t + m_it
            Expiry[(i, t)] = v_it
            if m_it <= 0:
                Gamma[(i, t)] = []
            else:
                u_max = min(T - 1, v_it - 1)
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


# ---------------- pricing (DP, LEFO) ----------------


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
):
    """
    DP pricing with LEFO. Uses a faster block-cost formula:
      base(t,s,e) = setup(i,t) + (c(i,t) - hpref(i,t)) * Q(s,e) + (Hdem[e+1] - Hdem[s])
    where hpref(i,u) = sum_{k< u} h[k], and Hdem[u] = sum_{k< u} d[k] * hpref(i,k).
    """
    d = [float(x) for x in items_raw[i]["demand"]]
    pref_d = [0.0] * (T + 1)
    for u in range(T):
        pref_d[u + 1] = pref_d[u] + d[u]

    # precompute hprefix per period and Hdem prefix
    # reuse hsum closure to infer hprefix(.,.) via tiny lambda to avoid refactor
    # (we already returned hpref in _cost_accessors but keep compatibility)
    def hpref_local(u: int) -> float:
        # hsum(i,0,u) == pref_h[u] - pref_h[0] == pref_h[u]
        return hsum(i, 0, u)

    Hdem = [0.0] * (T + 1)
    for u in range(T):
        Hdem[u + 1] = Hdem[u] + d[u] * hpref_local(u)

    # enumerate all candidate blocks (t,s,e) fast
    blocks = []
    for t in range(T):
        if not Gamma.get((i, t)):
            continue
        v_exp = Expiry[(i, t)]
        e_max = min(T - 1, v_exp - 1)
        if e_max < t:
            continue
        pit = per_item_cap.get((i, t), math.inf)
        Ct = c_at(i, t)
        hpt = hpref_local(t)
        setup_t = s_at(i, t)

        # sliding window on [s..e] to enforce per-item cap quickly
        run = 0.0
        e = t - 1
        for s in range(t, e_max + 1):
            # advance e as far as capacity allows
            if e < s - 1:
                e = s - 1
                run = 0.0
            while e + 1 <= e_max and run + d[e + 1] <= pit + EPS:
                e += 1
                run += d[e]
            if run <= EPS:
                # even d[s..e] has no positive demand; still allow but skip as useless
                continue

            # base cost via prefix sums:
            # Q = pref_d[e+1] - pref_d[s]
            Q = pref_d[e + 1] - pref_d[s]
            base = setup_t + (Ct - hpt) * Q + (Hdem[e + 1] - Hdem[s])

            # reduced-cost adjustment for cap and warehouse duals
            rc = base + pi_cap[t] * Q

            if rho_wh is not None and T >= 2:
                # Add inventory contributions from t to s-1 if s > t
                if s > t:
                    for uu in range(t, s):
                        if uu < T - 1 and Q > 0.0:
                            rc += rho_wh[uu] * Q
                cons = 0.0
                for u in range(s, e + 1):
                    cons += d[u]
                    inv_end = max(Q - cons, 0.0)
                    if u < T - 1 and inv_end > 0.0:
                        rc += rho_wh[u] * inv_end

            blocks.append((t, s, e, Q, rc))

            # before moving s -> s+1, subtract d[s] from run (if positive)
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

        # lost sale option
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
            if allow_lost_sales and d[s] > EPS:
                lost_by_u[s] += d[s]
                s += 1
                continue
            # fallback: skip (shouldn't happen)
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

    return ColumnPlan(
        i, -1, float(cost), prod_by_t, inv_end, flows, setups, lost_by_u
    ), float(best_rc)


# ---------------- random & greedy seeding ----------------


def seed_plan_naive_latest(
    i, items_raw, T, Gamma, c_at, s_at, hsum, allow_lost_sales, loss_pen
):
    """Produce at latest feasible t for each u (minimal holding)."""
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
            if allow_lost_sales:
                lost[u] = d[u]
                continue
            cand_t = u  # fallback
        flows.append((cand_t, u, d[u]))
        prod_by_t[cand_t] += d[u]
    # cost
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
    # inv
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
    """Random LEFO-respecting block plan."""
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

    # costs
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


# ---------------- main ----------------


def solve_instance(
    instance_path: str | Path = "last_instance.json",
    out_dir: str | Path = "cg_fast_results",
    # CG knobs
    seed_random_per_item: int = 2,  # naive + K random seeds
    stabilize: bool = True,
    stab_alpha: float = 0.6,
    max_iter: int = 20000,
    max_add_per_iter: int = 1,  # reserved; per-item add is <= 1 by DP
    drop_age: int = 8,
    # final
    finalize_as_mip: bool = True,
    # limits
    time_limit: int = 0,
    mip_gap: float = 0.0,
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
        allow_lost_sales,
        loss_penalty_global,
        loss_penalty_factor,
    ) = _parse_instance(instance_path)

    Gamma, Expiry = _precompute_gamma_and_expiry(items_raw, T)
    c_at, s_at, hsum, _hpref = _cost_accessors(items_raw, T)
    loss_pen, default_lp = _lost_sales_penalties(
        items_raw, T, Gamma, c_at, s_at, hsum, loss_penalty_global, loss_penalty_factor
    )
    use_wh = (W is not None) and (T >= 2)

    # RMP with feasibility slacks (for capacity/warehouse)
    rmp, cap_con, inv_con, one_con = build_rmp(T, prod_cap, use_wh, W, items_raw)
    M_cap = max(1e6, 1000.0 * (max(loss_pen.values()) if loss_pen else 1e6))
    M_wh = M_cap
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
            # unique & sorted setups just for logging clarity
            setups_log = sorted({t for t, q in enumerate(pl.prod_by_t) if q > EPS})
            print(
                f"[ADD] i={i} plan={pid} cost={pl.cost:.3f} setups={setups_log}",
                flush=True,
            )

    # ------- Seed -------
    random.seed(42)
    for i in items_raw:
        # 1) naive "latest feasible" seed
        pl0 = seed_plan_naive_latest(
            i, items_raw, T, Gamma, c_at, s_at, hsum, allow_lost_sales, loss_pen
        )
        add_column(i, pl0)
        # 2) K random block seeds
        for _ in range(seed_random_per_item):
            plr = seed_plan_random_blocks(
                i, items_raw, T, Gamma, Expiry, per_item_cap, c_at, s_at, hsum
            )
            add_column(i, plr)

    # ------- CG loop -------
    iter_no = 0
    last_pi = [0.0] * T
    last_rho = [0.0] * (T - 1 if use_wh else 0)

    while True:
        iter_no += 1
        if time_limit and time.time() - t0 > time_limit:
            print("[STOP] Time limit hit.", flush=True)
            break

        rmp.Params.OutputFlag = 0
        rmp.optimize()
        try:
            obj_now = float(rmp.ObjVal)
            obj_str = f"{obj_now:.6f}"
        except Exception:
            obj_str = "NA"

        # update ages for drop policy
        for key, v in list(lam_vars.items()):
            try:
                val = float(v.X)
            except Exception:
                val = 0.0
            ages[key] = 0 if val > 1e-10 else ages.get(key, 0) + 1

        # optional drop (safe)
        to_drop = []
        for (ii, pid), age in ages.items():
            if age > drop_age and (ii, pid) in lam_vars and len(pool[ii]) > 3:
                try:
                    if float(lam_vars[(ii, pid)].X) <= 1e-10:
                        to_drop.append((ii, pid))
                except Exception:
                    to_drop.append((ii, pid))
        if to_drop:
            # drop oldest first
            for ii, pid in to_drop:
                v = lam_vars.pop((ii, pid), None)
                if v is None:
                    continue
                rmp.remove(v)
                ages.pop((ii, pid), None)
            rmp.update()
            if verbose:
                print(f"[DROP] {len(to_drop)} cold columns", flush=True)

        # duals
        try:
            pi_raw = [cap_con[t].Pi for t in range(T)]
            rho_raw = [inv_con[u].Pi for u in range(T - 1)] if use_wh else []
            sigma = {i: one_con[i].Pi for i in items_raw}
        except Exception:
            print("[STOP] Duals not available.", flush=True)
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

        # pricing: per item DP
        any_added = False
        worst_rc = 0.0
        for i in items_raw:
            plan, rc_wo_sigma = price_item_plan(
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
                pi,
                (rho if use_wh else None),
            )
            rc_total = rc_wo_sigma - sigma[i]
            worst_rc = min(worst_rc, rc_total)
            if rc_total < -RC_EPS:
                add_column(i, plan)
                any_added = True

        if verbose:
            total_cols = sum(len(v) for v in pool.values())
            print(
                f"[ITER] {iter_no} obj={obj_str} worst_rc={worst_rc:.6f} cols={total_cols}",
                flush=True,
            )

        if not any_added:
            print("[STOP] No negative reduced-cost columns.", flush=True)
            break

        if iter_no >= max_iter:
            print("[STOP] Max iterations reached.", flush=True)
            break

    # ------- Finalize as MILP (exact) by flipping λ to binary on the SAME RMP -------
    if finalize_as_mip:
        if mip_gap:
            rmp.Params.MIPGap = float(mip_gap)
        if time_limit:
            # whatever time remains
            rmp.Params.TimeLimit = max(1, int(time_limit - (time.time() - t0)))
        rmp.Params.OutputFlag = 1
        rmp.Params.NumericFocus = 1
        # flip only lam_ vars to binary; keep slacks as continuous (feasible MILP)
        for (i, pid), var in lam_vars.items():
            var.VType = GRB.BINARY
        rmp.ModelSense = GRB.MINIMIZE
        rmp.optimize()

        # If no incumbent (should be rare with slacks), exit gracefully
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
                "solver_version": "lefo_cg_fast_exact_v1",
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
            # collect chosen (binary) combination — could be fractional if gap tolerance allowed
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
            # LOST lines (if any)
            if allow_lost_sales:
                for u in range(T):
                    if ls_t[u] > 1e-6:
                        orders_txt.append(f" u={u:2d} → LOST {ls_t[u]:8.3f}")
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
            "solver_version": "lefo_cg_fast_exact_v1",
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
            print(
                "[WARN] Feasibility slacks are positive in final solution — consider more CG iterations or larger seed_random_per_item.",
                flush=True,
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
        if allow_lost_sales:
            for u in range(T):
                if ls_t[u] > 1e-6:
                    orders_txt.append(f" u=={u:2d} → LOST {ls_t[u]:8.3f}")
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
        "solver_version": "lefo_cg_fast_lp_v1",
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


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="last_instance.json")
    p.add_argument("--out", default="cg_fast_results")
    p.add_argument("--seed_random_per_item", type=int, default=2)
    p.add_argument("--stab_off", action="store_true")
    p.add_argument("--stab_alpha", type=float, default=0.6)
    p.add_argument("--max_iter", type=int, default=20000)
    p.add_argument(
        "--max_add_per_iter", type=int, default=1
    )  # reserved; we add 1 per item/iter by DP
    p.add_argument("--drop_age", type=int, default=8)
    p.add_argument("--finalize_off", action="store_true")
    p.add_argument("--time_limit", type=int, default=0)
    p.add_argument("--mip_gap", type=float, default=0.0)
    args = p.parse_args()

    summary, orders = solve_instance(
        instance_path=args.instance,
        out_dir=args.out,
        seed_random_per_item=args.seed_random_per_item,
        stabilize=(not args.stab_off),
        stab_alpha=args.stab_alpha,
        max_iter=args.max_iter,
        max_add_per_iter=args.max_add_per_iter,
        drop_age=args.drop_age,
        finalize_as_mip=(not args.finalize_off),
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        verbose=True,
    )
    print(json.dumps(summary, indent=2))
