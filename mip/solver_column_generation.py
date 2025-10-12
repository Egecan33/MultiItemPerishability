from __future__ import annotations
import time, json, math, random, sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Set
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB, quicksum

EPS = 1e-9
RC_EPS = 1e-6
SIG_DIGITS = 7


@dataclass
class ColumnPlan:
    item: int
    plan_id: int
    cost: float
    prod_by_t: List[float]
    inv_end_by_u: List[float]
    flows: List[Tuple[int, int, float]]
    setups: List[int]
    lost_sales_by_u: List[float]
    is_dummy: bool = False


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
    return [int(round(c + buf)) for c in cap_raw]


def _plan_signature(pl: ColumnPlan, nd: int = SIG_DIGITS) -> tuple:
    flows_sig = tuple(sorted((t, u, round(q, nd)) for (t, u, q) in pl.flows if q > EPS))
    prod_sig = tuple(round(q, nd) for q in pl.prod_by_t)
    ls_sig = tuple(round(q, nd) for q in pl.lost_sales_by_u)
    inv_sig = tuple(round(x, nd) for x in pl.inv_end_by_u)
    setups_sig = tuple(sorted(pl.setups))
    return (flows_sig, prod_sig, ls_sig, inv_sig, setups_sig)


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
    items_raw: Dict[int, dict], T: int, inclusive_shelf: bool = False
):
    Gamma: Dict[Tuple[int, int], List[int]] = {}
    Expiry: Dict[Tuple[int, int], int] = {}
    for i, it in items_raw.items():
        mseq = list(it["shelf_seq"])
        if len(mseq) != T:
            raise ValueError(f"items[{i}]['shelf_seq'] must have length {T}")
        for t in range(T):
            L = int(mseq[t])
            if inclusive_shelf:
                v_it = t + L + 1
                u_max = min(T - 1, t + L)
            else:
                v_it = t + L
                u_max = min(T - 1, v_it - 1)
            Expiry[(i, t)] = v_it
            if L <= 0:
                Gamma[(i, t)] = []
            else:
                Gamma[(i, t)] = [u for u in range(t, u_max + 1)]
    return Gamma, Expiry


def _cost_accessors(items_raw: Dict[int, dict], T: int):
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


def seed_plan_dummy_outsource(
    i: int,
    items_raw: Dict[int, dict],
    T: int,
    unit_cost: Optional[float],
    loss_pen: Optional[Dict[Tuple[int, int], float]] = None,
) -> ColumnPlan:
    d = [float(x) for x in items_raw[i]["demand"]]
    if unit_cost is not None:
        cost = float(unit_cost) * float(sum(d))
    else:
        if loss_pen is None:
            raise ValueError(
                "seed_plan_dummy_outsource needs loss_pen when unit_cost is None"
            )
        cost = sum(float(loss_pen[(i, u)]) * d[u] for u in range(T))
    prod_by_t = [0.0] * T
    inv_end_by_u = [0.0] * (T - 1 if T >= 2 else 0)
    flows: List[Tuple[int, int, float]] = []
    setups: List[int] = []
    lost_sales_by_u = d[:]
    return ColumnPlan(
        i,
        -1,
        float(cost),
        prod_by_t,
        inv_end_by_u,
        flows,
        setups,
        lost_sales_by_u,
        is_dummy=True,
    )


def _check_lefo_flows(
    i: int,
    flows: List[Tuple[int, int, float]],
    Expiry: Dict[Tuple[int, int], int],
) -> None:
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
    rc_jitter: Optional[Callable[[int, int, int], float]] = None,
    outsource_unit_cost: Optional[float] = None,
    pool_size: int = 1,  # for multiple solutions
) -> List[Tuple[ColumnPlan, float]]:
    """Returns list of (plan, rc_wo_sigma) up to pool_size."""
    d = [float(items_raw[i]["demand"][u]) for u in range(T)]
    m = gp.Model(f"price_i{i}")
    m.Params.OutputFlag = 0
    m.Params.PoolSearchMode = 2 if pool_size > 1 else 0
    m.Params.PoolSolutions = pool_size
    triples_i = [(t, u) for t in range(T) for u in Gamma.get((i, t), [])]

    X = m.addVars(triples_i, lb=0.0, vtype=GRB.CONTINUOUS, name="X")
    Y = m.addVars(range(T), vtype=GRB.BINARY, name="Y")
    Z = m.addVars(triples_i, vtype=GRB.BINARY, name="Z")

    use_ls_or_os = allow_lost_sales or outsource_unit_cost is not None
    if use_ls_or_os:
        LS = m.addVars(range(T), lb=0.0, vtype=GRB.CONTINUOUS, name="LS")

    # Setup linking and arc activation
    mu = {t: sum(d[u] for u in Gamma.get((i, t), [])) for t in range(T)}
    for t in range(T):
        if Gamma.get((i, t)):
            sum_x = gp.quicksum(X[t, u] for u in Gamma[(i, t)])
            m.addConstr(sum_x <= mu[t] * Y[t])
            pit = per_item_cap.get((i, t), math.inf)
            if pit < math.inf:
                m.addConstr(sum_x <= pit)

    for t, u in triples_i:
        m.addConstr(X[t, u] <= d[u] * Z[t, u])

    # Demand
    for u in range(T):
        lhs = gp.quicksum(X[t, u] for t, uu in triples_i if uu == u)
        if use_ls_or_os:
            m.addConstr(lhs + LS[u] == d[u])
        else:
            m.addConstr(lhs == d[u])

    # LEFO no-crossing
    prods = [t for t in range(T) if Gamma.get((i, t))]
    prods.sort(key=lambda t: Expiry[(i, t)])
    for a in range(len(prods)):
        t1 = prods[a]
        v1 = Expiry[(i, t1)]
        for b in range(a + 1, len(prods)):
            t2 = prods[b]
            v2 = Expiry[(i, t2)]
            if v1 >= v2:
                continue
            for up in Gamma[(i, t2)]:
                for uu in Gamma[(i, t1)]:
                    if t2 <= uu <= up - 1:
                        m.addConstr(Z[t1, uu] + Z[t2, up] <= 1)

    # Inventory for rho
    inv_end_by_u = None
    if rho_wh is not None and T >= 2:
        Inv = m.addVars(range(T - 1), lb=0.0, name="Inv")
        for uu in range(T - 1):
            m.addConstr(
                Inv[uu] == gp.quicksum(X[t, w] for t, w in triples_i if t <= uu < w)
            )

    # Objective: true_cost - dual terms + jitter if any
    obj = gp.LinExpr()
    for t, u in triples_i:
        obj += (c_at(i, t) + hsum(i, t, u) - pi_cap[t]) * X[t, u]
    for t in range(T):
        obj += s_at(i, t) * Y[t]
    if use_ls_or_os:
        for u in range(T):
            penalty = (
                outsource_unit_cost
                if outsource_unit_cost is not None
                else loss_pen[(i, u)]
            )
            obj += penalty * LS[u]
    if rho_wh is not None and T >= 2:
        for uu in range(T - 1):
            obj -= rho_wh[uu] * Inv[uu]
    # Jitter (if provided, perturb setups)
    if rc_jitter is not None:
        for t in range(T):
            obj += rc_jitter(t, t, t) * Y[t]  # Arbitrary call to jitter
    m.setObjective(obj, GRB.MINIMIZE)

    m.optimize()
    if m.Status != GRB.OPTIMAL:
        return []

    out = []
    seen = set()
    for sn in range(min(pool_size, m.SolCount)):
        m.setParam(GRB.Param.SolutionNumber, sn)
        rc = float(m.PoolObjVal)
        flows = [(t, u, X[t, u].Xn) for t, u in triples_i if X[t, u].Xn > EPS]
        sig = _plan_signature(
            ColumnPlan(i, -1, 0.0, [0.0] * T, [0.0] * (T - 1), flows, [], [0.0] * T)
        )
        if sig in seen:
            continue
        seen.add(sig)

        prod_by_t = [0.0] * T
        for t, u, q in flows:
            prod_by_t[t] += q
        setups = [t for t, q in enumerate(prod_by_t) if q > EPS]

        lost_sales_by_u = [0.0] * T
        if use_ls_or_os:
            for u in range(T):
                lost_sales_by_u[u] = LS[u].Xn

        inv_end_by_u = [0.0] * (T - 1 if T >= 2 else 0)
        if rho_wh is not None and T >= 2:
            for uu in range(T - 1):
                inv_end_by_u[uu] = Inv[uu].Xn
        else:
            inv = 0.0
            prod_at = [0.0] * T
            cons_at = [0.0] * T
            for t, u, q in flows:
                prod_at[t] += q
                cons_at[u] += q
            for uu in range(T):
                inv += prod_at[uu] - cons_at[uu]
                if uu < T - 1:
                    inv_end_by_u[uu] = max(0.0, inv)

        # True cost
        cost = 0.0
        used_t = set(t for t, _, _ in flows)
        for t in used_t:
            cost += s_at(i, t)
        for t, u, q in flows:
            cost += (c_at(i, t) + hsum(i, t, u)) * q
        for u in range(T):
            if lost_sales_by_u[u] > EPS:
                penalty = (
                    outsource_unit_cost
                    if outsource_unit_cost is not None
                    else loss_pen[(i, u)]
                )
                cost += penalty * lost_sales_by_u[u]

        # LEFO check
        if flows:
            try:
                _check_lefo_flows(i, flows, Expiry)
            except RuntimeError as e:
                continue  # Skip invalid

        pl = ColumnPlan(
            i, -1, cost, prod_by_t, inv_end_by_u, flows, setups, lost_sales_by_u
        )
        out.append((pl, rc))

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
    outsource_unit_cost: Optional[float] = None,
) -> List[Tuple[ColumnPlan, float]]:
    rnd = random.Random(seed)
    seen: Set[tuple] = set()
    out: List[Tuple[ColumnPlan, float]] = []
    # Run once with pool for efficiency
    cand = price_item_plan(
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
        None,
        outsource_unit_cost,
        pool_size=k * 2,  # Oversample
    )
    for pl, rc in sorted(cand, key=lambda z: z[1]):
        sig = _plan_signature(pl)
        if sig in seen:
            continue
        seen.add(sig)
        out.append((pl, rc))
        if len(out) >= k:
            break
    # If not enough, rerun with jitter
    for r in range(k - len(out) + 1):

        def _jit(t: int, s: int, e: int) -> float:
            return (rnd.random() - 0.5) * 1e-9 * (1 + r)

        cand = price_item_plan(
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
            _jit,
            outsource_unit_cost,
            pool_size=1,
        )
        if not cand:
            continue
        pl, rc = cand[0]
        sig = _plan_signature(pl)
        if sig in seen:
            continue
        seen.add(sig)
        out.append((pl, rc))
    return out


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
                    continue
                added = price_fn_per_item(i, pi, rho)
                any_add = any_add or added
            if not any_add:
                break

    rmp.optimize()
    warm_start: Dict[Tuple[int, int], float] = {}
    for (i, pid), v in lam_vars.items():
        try:
            warm_start[(i, pid)] = float(v.X)
        except Exception:
            warm_start[(i, pid)] = 0.0

    _revert_all_bounds()
    return warm_start


def solve_instance(
    instance_path: str | Path = "last_instance.json",
    out_dir: str | Path = "cg_fast_results",
    stabilize: bool = True,
    stab_alpha: float = 0.6,
    max_iter: int = 40000,
    max_add_per_item_per_iter: int = 1,
    pricing_k: int = 3,
    drop_age: int = 10,
    global_add_limit: int = 1,
    finalize_as_mip: bool = True,
    time_limit: int = 0,
    mip_gap: float = 0.0,
    inclusive_shelf: bool = False,
    force_no_lost_sales: bool = True,
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

    allow_lost_sales = False if force_no_lost_sales else allow_lost_sales_in

    Gamma, Expiry = _precompute_gamma_and_expiry(
        items_raw, T, inclusive_shelf=inclusive_shelf
    )
    c_at, s_at, hsum, _hpref = _cost_accessors(items_raw, T)
    loss_pen, default_lp = _lost_sales_penalties(
        items_raw, T, Gamma, c_at, s_at, hsum, loss_penalty_global, loss_penalty_factor
    )
    use_wh = (W is not None) and (T >= 2)

    outsource_unit_cost = float(data.get("outsource_unit_cost", default_lp))
    try:
        from __main__ import args as _cli_args

        if getattr(_cli_args, "outsource_unit_cost", None) is not None:
            outsource_unit_cost = float(_cli_args.outsource_unit_cost)
    except Exception:
        pass

    rmp, cap_con, inv_con, one_con = build_rmp(T, prod_cap, use_wh, W, items_raw)
    M_cap = 1e7
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
    known_sigs: Dict[int, Set[tuple]] = {i: set() for i in items_raw}

    def add_column(i: int, pl: ColumnPlan):
        sig = _plan_signature(pl)
        if sig in known_sigs[i]:
            if verbose:
                _log(f"[SKIP] i={i} duplicate plan (signature match)")
            return
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
        known_sigs[i].add(sig)
        rmp.update()
        if verbose:
            setups_log = sorted({t for t, q in enumerate(pl.prod_by_t) if q > EPS})
            _log(f"[ADD] i={i} plan={pid} cost={pl.cost:.3f} setups={setups_log}")

    if verbose:
        _log("[SEED] Adding per-item DUMMY outsourcing columns (feasible root).")
    for i in items_raw:
        pl_dummy = seed_plan_dummy_outsource(
            i,
            items_raw,
            T,
            unit_cost=(
                outsource_unit_cost if outsource_unit_cost is not None else None
            ),
            loss_pen=loss_pen,
        )
        add_column(i, pl_dummy)

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

        for key, v in list(lam_vars.items()):
            try:
                val = float(v.X)
            except Exception:
                val = 0.0
            ages[key] = 0 if val > 1e-10 else ages.get(key, 0) + 1

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

        try:
            pi_raw = [cap_con[t].Pi for t in range(T)]
            rho_raw = [inv_con[u].Pi for u in range(T - 1)] if use_wh else []
            sigma = {i: one_con[i].Pi for i in items_raw}
        except Exception:
            _log("[STOP] Duals not available.")
            break

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

        basket: List[Tuple[float, int, ColumnPlan]] = []
        worst_rc = 0.0

        for i in items_raw:
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
                pi,
                (rho if use_wh else None),
                k=max_add_per_item_per_iter,
                seed=iter_no * 7919 + i * 104729,
                outsource_unit_cost=outsource_unit_cost,
            )
            added_for_i = 0
            for plan, rc_wo_sigma in sorted(cand_list, key=lambda z: z[1]):
                rc_total = rc_wo_sigma - sigma[i]
                worst_rc = min(worst_rc, rc_total)
                if rc_total < -max(RC_EPS, 1e-6 * (abs(plan.cost) + 1.0)):
                    sig = _plan_signature(plan)
                    if sig in known_sigs[i]:
                        continue
                    basket.append((rc_total, i, plan))
                    added_for_i += 1
                    if added_for_i >= max_add_per_item_per_iter:
                        break

        basket.sort(key=lambda x: x[0])
        if global_add_limit and global_add_limit > 0:
            basket = basket[:global_add_limit]

        any_added = False
        for rc_total, i, plan in basket:
            if verbose:
                _log(f"[PRICE] i={i} rc={rc_total:.6e} -> ADD")
            add_column(i, plan)
            any_added = True

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

    warm_start = None
    if enable_diving:
        _log("[DIVE] starting fix-and-price diving...")

        def _pf(i: int, pi_vec: List[float], rho_vec: Optional[List[float]]):
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
                rho_vec,
                k=1,
                seed=iter_no * 7919 + i * 104729,
                outsource_unit_cost=outsource_unit_cost,
            )
            if not cand_list:
                return False
            plan, _ = cand_list[0]
            sig = _plan_signature(plan)
            if sig in known_sigs[i]:
                return False
            add_column(i, plan)
            return True

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

    if finalize_as_mip:
        rmp.optimize()
        cap_slack_sum_LP = sum(float(v.X) for v in cap_slack.values())
        wh_slack_sum_LP = sum(float(v.X) for v in inv_slack.values()) if use_wh else 0.0
        if cap_slack_sum_LP <= 1e-9:
            for v in cap_slack.values():
                v.UB = 0.0
        if use_wh and wh_slack_sum_LP <= 1e-9:
            for v in inv_slack.values():
                v.UB = 0.0

        if force_no_lost_sales:
            for (ii, pid), var in lam_vars.items():
                pl = pool[ii][pid]
                if pl.is_dummy or sum(pl.lost_sales_by_u) > EPS:
                    var.UB = 0.0

        if mip_gap:
            rmp.Params.MIPGap = float(mip_gap)
        if time_limit:
            rmp.Params.TimeLimit = max(1, int(time_limit - (time.time() - t0)))
        rmp.Params.OutputFlag = 1
        rmp.Params.NumericFocus = 1
        rmp.Params.MIPFocus = 3
        rmp.Params.Cuts = 2
        rmp.Params.Heuristics = 0.1
        rmp.Params.Presolve = 2

        for (i, pid), var in lam_vars.items():
            var.VType = GRB.BINARY

        if warm_start is None:
            warm_start = {}
            rmp.optimize()
            for (i, pid), var in lam_vars.items():
                try:
                    warm_start[(i, pid)] = float(var.X)
                except Exception:
                    warm_start[(i, pid)] = 0.0
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
                "solver_version": "lefo_cg_fast_exact_v4_fixed",
                "n_items": len(items_raw),
                "T": T,
                "columns_total": sum(len(v) for v in pool.values()),
                "cap_slack_sum": 0.0,
                "wh_slack_sum": 0.0,
                "note": "MILP over current column pool had no incumbent; IIS written if possible.",
            }
            return summary, []

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

        cap_slack_sum = sum((_safe(v, "X", 0.0)) for v in cap_slack.values())
        wh_slack_sum = (
            sum((_safe(v, "X", 0.0)) for v in inv_slack.values()) if use_wh else 0.0
        )

        summary = {
            "status": int(rmp.Status),
            "objective": _safe(rmp, "ObjVal", None),
            "best_bound": _safe(rmp, "ObjBound", None),
            "gap": _safe(rmp, "MIPGap", 0.0),
            "runtime_sec": float(time.time() - t0),
            "solver_version": "lefo_cg_fast_exact_v4_fixed",
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
                "[WARN] Feasibility slacks positive in final solution — consider more CG iterations."
            )

        return summary, orders_txt

    rmp.optimize()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    orders_txt: List[str] = []
    for i, plist in pool.items():
        orders_txt.append(f"Item {i} — orders (t → qty)")
        prod_t = [0.0] * T
        ls_t = [0.0] * T
        for pl in plist:
            val = _safe(lam_vars[(i, pl.plan_id)], "X", 0.0)
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
        "objective": _safe(rmp, "ObjVal", None),
        "best_bound": _safe(rmp, "ObjBound", None),
        "gap": 0.0,
        "runtime_sec": float(time.time() - t0),
        "solver_version": "lefo_cg_fast_lp_v4_fixed",
        "n_items": len(items_raw),
        "T": T,
        "columns_total": sum(len(v) for v in pool.values()),
        "cap_slack_sum": sum(_safe(v, "X", 0.0) for v in cap_slack.values()),
        "wh_slack_sum": (
            sum(_safe(v, "X", 0.0) for v in inv_slack.values()) if use_wh else 0.0
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

    p.add_argument("--stab_off", action="store_true")
    p.add_argument("--stab_alpha", type=float, default=0.6)
    p.add_argument("--max_iter", type=int, default=40000)
    p.add_argument("--max_add_per_item_per_iter", type=int, default=1)
    p.add_argument("--pricing_k", type=int, default=3)
    p.add_argument("--drop_age", type=int, default=10)
    p.add_argument("--global_add_limit", type=int, default=1)

    p.add_argument("--finalize_off", action="store_true")
    p.add_argument("--time_limit", type=int, default=0)
    p.add_argument("--mip_gap", type=float, default=0.0)

    p.add_argument("--inclusive_shelf", action="store_true")

    p.add_argument("--allow_lost_sales", action="store_true")

    p.add_argument("--diving_off", action="store_true")
    p.add_argument("--diving_reprice_iters", type=int, default=40)

    p.add_argument("--outsource_unit_cost", type=float, default=None)

    args = p.parse_args()

    summary, orders = solve_instance(
        instance_path=args.instance,
        out_dir=args.out,
        stabilize=not args.stab_off,
        stab_alpha=args.stab_alpha,
        max_iter=args.max_iter,
        max_add_per_item_per_iter=args.max_add_per_item_per_iter,
        pricing_k=args.pricing_k,
        drop_age=args.drop_age,
        global_add_limit=args.global_add_limit,
        finalize_as_mip=not args.finalize_off,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        inclusive_shelf=args.inclusive_shelf,
        force_no_lost_sales=not args.allow_lost_sales,
        enable_diving=not args.diving_off,
        diving_reprice_iters=args.diving_reprice_iters,
        verbose=True,
    )
    print(json.dumps(summary, indent=2))
