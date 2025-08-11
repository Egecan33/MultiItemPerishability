# main.py
# Single-file branch-and-price for multi-item perishable lot-sizing (no backorders)
# - Robust specs loader (list/dict items, old/new keys, auto demand, flexible shelf)
# - FEFO DP pricing, shortest-path fallback, MIP pricing last
# - Branch on y_{i,t} (order/no-order)
# - Dual stabilization + CG stagnation early-stop (+ don't starve pool)
# - Greedy primal repair + DFS dive for early incumbents
# - Global time & node limits so it doesn't run forever

from __future__ import annotations

import contextlib, io, itertools, json, math, os, random, sys, time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import gurobipy as grb

# ============================== Config ========================================
SPECS_PATH = Path(__file__).with_name("specs.json")
OUTDIR = "results"
SEED = int(os.environ.get("SEED", "0"))

# Global run guards
GLOBAL_TIME_LIMIT_SEC = float(os.environ.get("TIME_LIMIT_SEC", "120"))  # 2 min
GLOBAL_NODE_LIMIT = int(os.environ.get("NODE_LIMIT", "200"))

# Column Generation
CG_MAX_ITERS = 100
CG_ESTOP_MIN_ITERS = 3
CG_ESTOP_PATIENCE = 3
CG_ESTOP_MIN_REL_IMPROVE = 5e-5
MIN_PATTERNS_PER_ITEM = 10  # don't integerize too early

# Gap to stop B&P early if bound~incumbent
BAP_GAP_TOL = 1e-3  # 0.1%

# Reduced-cost micro threshold
RED_COST_EPS = 1e-3

# Stabilization
STAB_ALPHA = 0.9

# Logging
VERBOSE = True
LOG_EVERY_ITER = 1
DEBUG_BRANCH = True

# Threads
N_MASTER_THREADS = max(1, (os.cpu_count() or 4) // 2)

# MIP pricing time limit
MIP_PRICE_TIMELIMIT_SEC = 5.0


# =========================== Helpers / Logging ================================
def log(msg: str) -> None:
    if VERBOSE:
        print(msg)


def _stats(v: List[float]) -> str:
    if not v:
        return "min=NA max=NA mean=NA l1=NA"
    mn, mx, mean = min(v), max(v), sum(v) / len(v)
    l1 = sum(abs(x) for x in v)
    return f"min={mn:.3g} max={mx:.3g} mean={mean:.3g} l1={l1:.3g}"


# =============================== Data =========================================
@dataclass
class Item:
    id: int
    demand: List[int]
    setup: float
    b_var: float
    c_var: float
    h: float
    shelf_seq: List[int]
    agency: Optional[Dict[str, str]] = None  # {"name": ..., "sku": ...}


@dataclass
class Lot:
    period: int
    capacity_pad: int
    items: Dict[int, Item] = field(default_factory=dict)
    manual_capacity: Optional[List[int]] = None

    @property
    def capacity(self) -> List[int]:
        if self.manual_capacity is not None:
            return self.manual_capacity
        cap = [
            sum(it.demand[t] for it in self.items.values()) for t in range(self.period)
        ]
        if self.capacity_pad:
            cap = [c + self.capacity_pad for c in cap]
        return cap

    @property
    def kmax(self) -> Dict[int, int]:
        return {
            i: sum(it.demand) + max(5, int(0.2 * sum(it.demand)))
            for i, it in self.items.items()
        }

    def to_dicts(self):
        demand = {i: it.demand for i, it in self.items.items()}
        c_var = {i: it.c_var for i, it in self.items.items()}
        h = {i: it.h for i, it in self.items.items()}
        setup = {i: it.setup for i, it in self.items.items()}
        b_var = {i: it.b_var for i, it in self.items.items()}
        cap = self.capacity
        mseq = {i: it.shelf_seq for i, it in self.items.items()}
        kmax = self.kmax
        return demand, c_var, h, setup, b_var, cap, mseq, kmax

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        data = asdict(self)
        data["items"] = {str(k): v for k, v in data["items"].items()}
        Path(path).write_text(json.dumps(data, indent=indent))

    @classmethod
    def from_json(cls, path: str | Path) -> "Lot":
        data = json.loads(Path(path).read_text())
        items = {int(k): Item(**v) for k, v in data["items"].items()}
        return cls(
            period=data["period"],
            capacity_pad=data["capacity_pad"],
            items=items,
            manual_capacity=data.get("manual_capacity"),
        )


# ============ Spec loader / default generator (with agency fields) ============
def _pick(d: dict, *keys, default=None, required=False):
    """Return the first present key in d from keys."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    if required:
        raise KeyError(f"Missing one of keys {keys} in {list(d.keys())}")
    return default


def _as_seq(x, T: int, name: str):
    """Ensure x is a list of length T. If scalar, repeat; if a length-2 tuple/list, treat as range; if list length T, return."""
    # scalar
    if isinstance(x, (int, float)):
        return [int(x)] * T
    # 2-length range (lo, hi)
    if (
        isinstance(x, (list, tuple))
        and len(x) == 2
        and all(isinstance(v, (int, float)) for v in x)
    ):
        lo, hi = int(min(x)), int(max(x))
        rnd = random.Random((SEED << 8) ^ (lo << 2) ^ hi)
        return [rnd.randint(lo, hi) for _ in range(T)]
    # full sequence
    if isinstance(x, (list, tuple)):
        if len(x) != T:
            raise ValueError(f"{name} length {len(x)} != period {T}")
        return list(x)
    raise ValueError(f"Cannot coerce {name} into length-{T} sequence")


def load_specs(path: Path) -> Lot:
    """
    Robust loader:
    - Accepts items as dict or list.
    - Accepts keys: demand|D|demands|d  (optional – will auto-generate)
                    setup|S|setup_cost|fixed
                    c_var|c|unit_cost|price|p
                    h|holding|h_var|hold
                    b_var|b|backorder_cost
                    shelf_life_sequence|shelf_seq|shelf|mseq|shelfLife|life
                      (scalar, length-2 range [lo,hi], or full list length T)
    - If demand missing: generates deterministic uniform ints in [demand_min,demand_max].
      You can set these at the top-level of specs.json.
    """
    data = json.loads(path.read_text())

    items_blob = data["items"]  # can be dict or list

    # --- determine T (period) ---
    if "period" in data:
        period = int(data["period"])
    else:
        # try infer from first present sequence among demand / shelf (if full length) / manual_capacity
        if isinstance(items_blob, dict):
            first = next(iter(items_blob.values()))
        else:
            first = items_blob[0]
        dem_raw = _pick(first, "demand", "D", "demands", "d", default=None)
        if isinstance(dem_raw, (list, tuple)) and len(dem_raw) > 2:
            period = len(dem_raw)
        else:
            shelf_raw = _pick(
                first,
                "shelf_life_sequence",
                "shelf_seq",
                "shelf",
                "mseq",
                "shelfLife",
                "life",
                default=None,
            )
            if isinstance(shelf_raw, (list, tuple)) and len(shelf_raw) > 2:
                period = len(shelf_raw)
            elif isinstance(data.get("manual_capacity"), (list, tuple)):
                period = len(data["manual_capacity"])
            else:
                period = 60  # last resort
    capacity_pad = int(data.get("capacity_pad", 0))

    # manual_capacity can be scalar or list
    mancap_raw = data.get("manual_capacity", None)
    if isinstance(mancap_raw, (int, float)):
        manual_capacity = [int(mancap_raw)] * period
    elif isinstance(mancap_raw, (list, tuple)):
        if len(mancap_raw) != period:
            raise ValueError(
                f"manual_capacity length {len(mancap_raw)} != period {period}"
            )
        manual_capacity = list(map(int, mancap_raw))
    else:
        manual_capacity = None

    # demand generation defaults
    dmin = int(data.get("demand_min", 1))
    dmax = int(data.get("demand_max", 200))
    # optional top-level demand: list or dict {id: list}
    global_dem = data.get("demand", None)

    lot = Lot(period=period, capacity_pad=capacity_pad, manual_capacity=manual_capacity)

    def add_one(idx: int, rec: dict):
        i = int(rec.get("id", idx))

        # ----- demand -----
        dem = _pick(rec, "demand", "D", "demands", "d", default=None)
        if dem is None and isinstance(global_dem, dict):
            # top-level per-item map (keys may be str)
            key = i if i in global_dem else str(i)
            dem = global_dem.get(key)
        if dem is None and isinstance(global_dem, (list, tuple)):
            dem = global_dem
        if dem is None:
            # generate deterministic demand using SEED and item id
            rnd = random.Random((SEED << 16) ^ i)
            dem = [rnd.randint(dmin, dmax) for _ in range(period)]
        demand = list(_as_seq(dem, period, "demand"))

        # ----- costs -----
        setup = float(_pick(rec, "setup", "S", "setup_cost", "fixed", required=True))
        c_var = float(
            _pick(rec, "c_var", "c", "unit_cost", "price", "p", required=True)
        )
        h = float(_pick(rec, "h", "holding", "h_var", "hold", required=True))
        b_var = float(_pick(rec, "b_var", "b", "backorder_cost", default=0.0))

        # ----- shelf life sequence -----
        shelf_raw = _pick(
            rec,
            "shelf_life_sequence",
            "shelf_seq",
            "shelf",
            "mseq",
            "shelfLife",
            "life",
            required=True,
        )
        # Coerce into length-T sequence:
        shelf_seq = [
            max(1, int(v)) for v in _as_seq(shelf_raw, period, "shelf_life_sequence")
        ]

        lot.items[i] = Item(
            id=i,
            demand=[int(x) for x in demand],
            setup=setup,
            b_var=b_var,
            c_var=c_var,
            h=h,
            shelf_seq=shelf_seq,
            agency=rec.get("agency"),
        )

    if isinstance(items_blob, dict):
        for k, rec in items_blob.items():
            add_one(int(k), rec)
    else:
        for idx, rec in enumerate(items_blob):
            add_one(idx, rec)

    # If no manual capacity, derive from demand + pad
    if lot.manual_capacity is None:
        lot.manual_capacity = lot.capacity  # property computes from items + pad
    return lot


def default_specs() -> Lot:
    period = 60
    manual_caps = [870] * period
    base_specs = [
        (0, 127.5, 5.0, 2.0, 0.4, (1, 50), "Agency-0", "SKU-0"),
        (1, 29.0, 5.0, 3.0, 0.6, (1, 60), "Agency-1", "SKU-1"),
        (2, 50.0, 5.0, 1.0, 0.3, (3, 70), "Agency-2", "SKU-2"),
        (3, 75.0, 5.0, 4.0, 0.5, (3, 5), "Agency-3", "SKU-3"),
        (4, 100.0, 5.0, 2.5, 0.4, (3, 5), "Agency-4", "SKU-4"),
    ]
    lot = Lot(period=period, capacity_pad=10, manual_capacity=manual_caps)
    rnd = random.Random(SEED)
    for i, setup, b, c, h, shelf_range, aname, sku in base_specs:
        demand = [rnd.randint(1, 200) for _ in range(period)]
        shelf_seq = _as_seq(shelf_range, period, "shelf_life_sequence")
        lot.items[i] = Item(
            id=i,
            demand=demand,
            setup=setup,
            b_var=b,
            c_var=c,
            h=h,
            shelf_seq=[max(1, int(v)) for v in shelf_seq],
            agency={"name": aname, "sku": sku},
        )
    return lot


# =========================== Gurobi environment ===============================
def make_env(quiet: bool = True) -> grb.Env:
    env = grb.Env(empty=True)
    if quiet:
        env.setParam("LogToConsole", 0)
        env.setParam("OutputFlag", 0)
    env.start()
    return env


GRB_ENV = make_env(quiet=True)

# ============================ FEFO DP pricing =================================
Number = float


@dataclass
class DPResult:
    total_cost: Number
    x_ti: List[List[Number]]
    x_t: List[Number]
    y_t: List[int]


def _prefix(a: List[Number]) -> List[Number]:
    ps = [0.0]
    s = 0.0
    for v in a:
        s += v
        ps.append(s)
    return ps


def _c_ti(p: List[Number], h: List[Number]) -> List[List[Number]]:
    T = len(p)
    H = _prefix(h)
    c = [[math.inf] * T for _ in range(T)]
    for t in range(T):
        for i in range(t, T):
            c[t][i] = p[t] + (H[i] - H[t])
    return c


def dp_fefo_solve(
    D, p_eff, h_seq, S_seq, v_last, forbid_t: Optional[set] = None
) -> DPResult:
    T = len(D)
    inf = math.inf
    S_eff = [(inf if forbid_t and t in forbid_t else S_seq[t]) for t in range(T)]
    c = _c_ti(p_eff, h_seq)
    f_t = {}
    g_t = {}
    f_min = [[inf] * T for _ in range(T)]
    g_min = [[inf] * T for _ in range(T)]
    tstar = {}
    split_g = {}

    def gmin(a, b):
        return 0.0 if a > b else g_min[a][b]

    for τ in range(T):
        for t in range(τ + 1):
            if v_last[t] < τ or math.isinf(S_eff[t]):
                val = inf
            else:
                val = S_eff[t] + c[t][τ] * D[τ]
            f_t[(t, τ, τ)] = val
            g_t[(t, τ, τ)] = val
        best = inf
        best_t = -1
        for t in range(τ + 1):
            if f_t[(t, τ, τ)] < best:
                best = f_t[(t, τ, τ)]
                best_t = t
        f_min[τ][τ] = best
        g_min[τ][τ] = best
        tstar[(τ, τ)] = best_t
        split_g[(τ, τ)] = τ

    for length in range(1, T):
        for τ1 in range(0, T - length):
            τ2 = τ1 + length
            for t in range(0, τ1 + 1):
                if v_last[t] < τ2 or math.isinf(S_eff[t]):
                    f_cost = inf
                    g_cost = inf
                else:
                    prev = g_t.get((t, τ1, τ2 - 1), inf)
                    f_cost = prev + c[t][τ2] * D[τ2]
                    g_cost = inf
                    for τ in range(τ1, τ2 + 1):
                        f_part = f_t.get((t, τ1, τ), inf)
                        val = f_part + gmin(τ + 1, τ2)
                        if val < g_cost:
                            g_cost = val
                f_t[(t, τ1, τ2)] = f_cost
                g_t[(t, τ1, τ2)] = g_cost
            best_f = inf
            best_tf = -1
            for t in range(0, τ1 + 1):
                val = f_t[(t, τ1, τ2)]
                if val < best_f:
                    best_f = val
                    best_tf = t
            f_min[τ1][τ2] = best_f
            tstar[(τ1, τ2)] = best_tf
            best_g = inf
            best_split = -1
            for τ in range(τ1, τ2 + 1):
                val = f_min[τ1][τ] + gmin(τ + 1, τ2)
                if val < best_g:
                    best_g = val
                    best_split = τ
            g_min[τ1][τ2] = best_g
            split_g[(τ1, τ2)] = best_split

    total_cost = g_min[0][T - 1]
    x_ti = [[0.0] * T for _ in range(T)]
    x_t = [0.0] * T
    y_t = [0] * T

    def rec(a, b):
        if a > b:
            return
        τ_split = split_g[(a, b)]
        t_best = tstar[(a, τ_split)]
        if t_best == -1 or math.isinf(f_min[a][τ_split]):
            raise RuntimeError(f"Infeasible subplan [{a},{τ_split}]")
        for i in range(a, τ_split + 1):
            x_ti[t_best][i] = D[i]
            x_t[t_best] += D[i]
        y_t[t_best] = 1
        rec(τ_split + 1, b)

    if T > 0:
        rec(0, T - 1)
    return DPResult(total_cost=total_cost, x_ti=x_ti, x_t=x_t, y_t=y_t)


# ======================= Shortest-path fallback pricing =======================
def build_fefo_flow_from_q(
    demand: List[int], q: List[int], m_seq: List[int]
) -> Dict[Tuple[int, int], float]:
    T = len(demand)
    x_flow = {}
    cohorts = []
    for s in range(T):
        qty = q[s]
        if qty > 0:
            life = max(1, int(m_seq[s]))
            exp = min(T - 1, s + life - 1)
            cohorts.append((exp, float(qty), s))
    for t in range(T):
        need = float(demand[t])
        cohorts.sort(key=lambda c: (c[0], c[2]))
        new = []
        for exp, rem, s in cohorts:
            if need <= 1e-12:
                new.append((exp, rem, s))
                continue
            if t > exp or rem <= 0:
                continue
            use = min(rem, need)
            if use > 0:
                x_flow[(s, t)] = x_flow.get((s, t), 0.0) + use
                rem -= use
                need -= use
            if rem > 1e-12:
                new.append((exp, rem, s))
        cohorts = [(exp - 1, qty, s) for (exp, qty, s) in new if exp > t]
    return x_flow


def price_shortest_path(
    item_id, demand, c_var, h, setup, mu, pi, m_seq, k_max, order_fix
):
    T = len(demand)
    dist = [float("inf")] * (T + 1)
    next_arc = [None] * (T + 1)
    dist[T] = -pi
    forbidden_starts = {
        tt for (ii, tt), (lb, ub) in order_fix.items() if ii == item_id and ub == 0
    }
    for t in range(T - 1, -1, -1):
        best = float("inf")
        best_arc = None
        if t not in forbidden_starts:
            q_acc = 0
            hold_acc = 0
            u_max = T - 1
            life = max(1, int(m_seq[t]))
            u_max = min(u_max, t + life - 1)
            for u in range(t, u_max + 1):
                q_acc += demand[u]
                if q_acc > k_max:
                    break
                hold_acc += demand[u] * (u - t)
                arc_cost = (
                    (c_var - mu[t]) * q_acc + h * hold_acc + (setup if q_acc > 0 else 0)
                )
                cand = arc_cost + dist[u + 1]
                if cand < best:
                    best = cand
                    best_arc = (u, q_acc)
        dist[t] = best
        next_arc[t] = best_arc
    if math.isinf(dist[0]) or dist[0] >= -RED_COST_EPS:
        return float("inf"), float("inf"), None
    q_plan = [0] * T
    true_cost = 0.0
    t = 0
    while t < T:
        arc = next_arc[t]
        if arc is None:
            break
        u, qty = arc
        q_plan[t] = qty
        hold_cost = sum(demand[tau] * (tau - t) for tau in range(t, u + 1))
        true_cost += (setup if qty > 0 else 0) + c_var * qty + h * hold_cost
        t = u + 1
    if sum(q_plan) == 0:
        return float("inf"), float("inf"), None
    return dist[0], true_cost, q_plan


# =============================== Master (RMP) =================================
class MasterModel:
    def __init__(self, items, T, capacity, env: grb.Env, verbose=False):
        self.items = items
        self.T = T
        self.capacity = capacity
        self.model = grb.Model("RMP", env=env)
        self.model.Params.OutputFlag = 1 if verbose else 0
        self.model.Params.Method = 1
        self.model.Params.Threads = N_MASTER_THREADS
        self.lambda_vars = {i: [] for i in items}
        self.patterns = {i: [] for i in items}
        self.sel_constr = {
            i: self.model.addConstr(grb.LinExpr() == 1.0, name=f"sel_{i}")
            for i in items
        }
        self.cap_constr = [
            self.model.addConstr(grb.LinExpr() <= capacity[t], name=f"cap_{t}")
            for t in range(T)
        ]
        self.order_constr = {}

    def _ensure_order_constraint(self, ii, tt, lb, ub):
        key = (ii, tt)
        if key in self.order_constr:
            return
        expr = grb.LinExpr()
        if lb == 1 and ub == 1:
            constr = self.model.addConstr(expr >= 1.0, name=f"branch_y1_{ii}_{tt}")
        elif lb == 0 and ub == 0:
            constr = self.model.addConstr(expr <= 0.0, name=f"branch_y0_{ii}_{tt}")
        else:
            return
        self.order_constr[key] = constr

    def add_pattern(self, i, cost, q, y):
        if sum(q) == 0:
            return
        for pat in self.patterns[i]:
            if pat["q"] == q:
                return
        col = grb.Column()
        col.addTerms(1.0, self.sel_constr[i])
        for t, qty in enumerate(q):
            if qty != 0:
                col.addTerms(qty, self.cap_constr[t])
        for (ii, tt), constr in self.order_constr.items():
            if ii == i and y[tt] == 1:
                col.addTerms(1.0, constr)
        var = self.model.addVar(
            obj=cost, column=col, name=f"lambda_{i}_{len(self.lambda_vars[i])}"
        )
        self.lambda_vars[i].append(var)
        self.patterns[i].append({"cost": cost, "q": q, "y": y})
        log(f"[ADD] item {i} pattern#{len(self.lambda_vars[i])-1} cost={cost:.2f}")

    def optimize(self):
        self.model.optimize()
        st = self.model.Status
        if st == grb.GRB.INFEASIBLE:
            return None
        if st != grb.GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi status {st} in RMP")
        obj = self.model.ObjVal
        pi = {i: self.sel_constr[i].Pi for i in self.items}
        mu = [c.Pi for c in self.cap_constr]
        lambdas = {i: [v.X for v in self.lambda_vars[i]] for i in self.items}
        return obj, pi, mu, lambdas

    def clone_with_branching(self, order_fix):
        new = MasterModel(self.items, self.T, self.capacity, env=GRB_ENV, verbose=False)
        for i in self.items:
            for pat in self.patterns[i]:
                q = pat["q"]
                y = pat["y"]
                cost = pat["cost"]
                col = grb.Column()
                col.addTerms(1.0, new.sel_constr[i])
                for t, qty in enumerate(q):
                    if qty != 0:
                        col.addTerms(qty, new.cap_constr[t])
                for (ii, tt), constr in new.order_constr.items():
                    if ii == i and y[tt] == 1:
                        col.addTerms(1.0, constr)
                var = new.model.addVar(
                    obj=cost, column=col, name=f"lambda_{i}_{len(new.lambda_vars[i])}"
                )
                new.lambda_vars[i].append(var)
                new.patterns[i].append({"cost": cost, "q": q, "y": y})
        for (ii, tt), (lb, ub) in order_fix.items():
            if lb == ub:
                new._ensure_order_constraint(ii, tt, lb, ub)
                constr = new.order_constr[(ii, tt)]
                for idx, pat in enumerate(new.patterns[ii]):
                    if pat["y"][tt] == 1:
                        var = new.lambda_vars[ii][idx]
                        new.model.chgCoeff(constr, var, 1.0)
        new.model.update()
        return new


# ============================= Convergence guard ==============================
class ConvergenceGuard:
    def __init__(self, min_iters, patience, min_rel):
        self.min_iters = min_iters
        self.patience = patience
        self.min_rel = min_rel
        self.prev = None
        self.stale = 0

    def step(self, obj):
        if self.prev is not None:
            denom = max(1.0, abs(self.prev))
            rel = (self.prev - obj) / denom
            self.stale = self.stale + 1 if rel < self.min_rel else 0
        self.prev = obj
        return (
            (self.prev is not None)
            and (self.stale >= self.patience)
            and (self.prev != float("inf"))
        )


# ============================ Branch-and-Price ================================
class BranchPrice:
    def __init__(self, demand, c_var, h, setup, b_var, capacity, shelf_seq, k_max):
        self.items = list(demand.keys())
        self.dem = demand
        self.c_var = c_var
        self.h = h
        self.setup = setup
        self.b_var = b_var
        self.cap = capacity
        self.T = len(capacity)
        self.mseq = shelf_seq
        self.k_max = k_max
        self.order_fix = {}
        self.prev_mu = [0.0] * self.T
        self.alpha = STAB_ALPHA
        self.master = MasterModel(
            self.items, self.T, self.cap, env=GRB_ENV, verbose=False
        )
        # seed with "order exactly demand" pattern (always feasible wrt FEFO + capacity-derived caps)
        for i in self.items:
            q_pattern = self.dem[i][:]
            total_cost = sum(self.c_var[i] * q for q in q_pattern) + self.setup[
                i
            ] * sum(1 for q in q_pattern if q > 0)
            y_pattern = [1 if q > 0 else 0 for q in q_pattern]
            self.master.add_pattern(i, total_cost, q_pattern, y_pattern)
        self.master.model.update()
        self.tree = {}
        self._node_id_counter = itertools.count()
        self.parent_stack = []
        self._global_nodes = 0
        self._start_time = time.perf_counter()
        self._last_greedy = None  # (obj, sol) side channel
        log(
            f"[INFO] Init Branch-and-Price: items={len(self.items)}, T={self.T}, capacity_stats[{_stats(self.cap)}]"
        )

    # ---------- small utilities ----------
    def _min_patterns_per_item(self) -> int:
        return min(len(self.master.patterns[i]) for i in self.items)

    def compute_y_values(self, lambda_vals):
        y_usage = {}
        for i in self.items:
            for p_idx, lam in enumerate(lambda_vals.get(i, [])):
                if lam <= 1e-9:
                    continue
                y_vec = self.master.patterns[i][p_idx]["y"]
                for t, yb in enumerate(y_vec):
                    if yb == 1:
                        y_usage[(i, t)] = y_usage.get((i, t), 0.0) + lam
        return y_usage

    # ---------- pricing ----------
    def price_item(self, i, mu_hat, pi_i):
        demand = self.dem[i]
        T = len(demand)
        forbid = {
            t
            for (ii, t), (lb, ub) in self.order_fix.items()
            if ii == i and lb == ub == 0
        }
        # DP pricing (FEFO)
        p_eff = [self.c_var[i] - mu_hat[t] for t in range(T)]
        h_seq = [self.h[i]] * T
        S_seq = [self.setup[i]] * T
        v_last = [min(T - 1, t + max(1, int(self.mseq[i][t])) - 1) for t in range(T)]
        dp = dp_fefo_solve(
            demand, p_eff, h_seq, S_seq, v_last, forbid_t=forbid if forbid else None
        )
        rc = dp.total_cost - pi_i
        if rc < -RED_COST_EPS:
            q_plan = [int(round(q)) for q in dp.x_t]
            total_var = self.c_var[i] * sum(q_plan)
            total_hold = 0.0
            for t in range(T):
                for u in range(T):
                    if dp.x_ti[t][u] > 1e-12:
                        total_hold += self.h[i] * (u - t) * dp.x_ti[t][u]
            total_setup = self.setup[i] * sum(1 for t in range(T) if q_plan[t] > 0)
            true_cost = total_setup + total_var + total_hold
            return i, rc, true_cost, q_plan, "DP"

        # Shortest-path fallback
        rc_sp, true_sp, q_sp = price_shortest_path(
            i,
            demand,
            self.c_var[i],
            self.h[i],
            self.setup[i],
            mu_hat,
            pi_i,
            self.mseq[i],
            self.k_max[i],
            self.order_fix,
        )
        if q_sp is not None and rc_sp < -RED_COST_EPS:
            return i, rc_sp, true_sp, q_sp, "SP"

        # MIP pricing as last resort
        rc_mip, true_mip, q_mip = self.price_mip(i, mu_hat, pi_i, None, None)
        if q_mip is not None and rc_mip < -RED_COST_EPS:
            return i, rc_mip, true_mip, q_mip, "MIP"
        return i, float("inf"), float("inf"), None, "NONE"

    def price_mip(self, item_id, mu, pi_i, warm_q, warm_flow):
        D = self.dem[item_id]
        T = len(D)
        m_seq = [max(1, int(v)) for v in self.mseq[item_id]]
        kmax = self.k_max[item_id]
        model = grb.Model(f"price_item_{item_id}", env=GRB_ENV)
        model.Params.LogToConsole = 0
        model.Params.OutputFlag = 0
        model.Params.Presolve = 2
        model.Params.Cuts = 2
        model.Params.Heuristics = 0.5
        model.Params.Method = 2
        model.Params.Threads = 1
        model.Params.TimeLimit = MIP_PRICE_TIMELIMIT_SEC
        model.Params.MIPFocus = 1
        model.Params.Cutoff = -RED_COST_EPS
        model.Params.IntFeasTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        q = model.addVars(T, vtype=grb.GRB.INTEGER, lb=0, ub=kmax, name="q")
        y = model.addVars(T, vtype=grb.GRB.BINARY, name="y")
        x = {}
        for s in range(T):
            v_last = min(T - 1, s + m_seq[s] - 1)
            for t in range(s, v_last + 1):
                x[(s, t)] = model.addVar(
                    vtype=grb.GRB.CONTINUOUS, lb=0.0, name=f"x_{s}_{t}"
                )
        for t in range(T):
            model.addConstr(
                grb.quicksum(x[(s, t)] for s in range(0, t + 1) if (s, t) in x) == D[t],
                name=f"demand_{t}",
            )
        for s in range(T):
            model.addConstr(
                grb.quicksum(x[(s, t)] for t in range(s, T) if (s, t) in x) <= q[s],
                name=f"flow_{s}",
            )
        for s in range(T):
            model.addGenConstrIndicator(y[s], True, q[s] >= 1, name=f"setup_on_{s}")
            model.addGenConstrIndicator(y[s], False, q[s] == 0, name=f"setup_off_{s}")
        for (ii, tt), (lb, ub) in self.order_fix.items():
            if ii == item_id:
                y[tt].LB = lb
                y[tt].UB = ub
        setup_term = grb.quicksum(self.setup[item_id] * y[s] for s in range(T))
        var_term = grb.quicksum(self.c_var[item_id] * q[s] for s in range(T))
        hold_term = grb.quicksum(self.h[item_id] * (t - s) * x[(s, t)] for (s, t) in x)
        dual_adj = grb.quicksum(-mu[s] * q[s] for s in range(T)) - pi_i
        model.setObjective(
            setup_term + var_term + hold_term + dual_adj, grb.GRB.MINIMIZE
        )
        if warm_q:
            for t in range(T):
                q[t].Start = int(max(0, warm_q[t]))
                y[t].Start = 1 if warm_q[t] and warm_q[t] > 0 else 0
        if warm_flow:
            for (s, t), val in warm_flow.items():
                if (s, t) in x:
                    x[(s, t)].Start = float(val)
        model.optimize()
        if (
            model.Status in (grb.GRB.OPTIMAL, grb.GRB.CUTOFF)
        ) and model.ObjVal < -RED_COST_EPS:
            q_sol = [int(round(q[t].X)) for t in range(T)]
            total_setup = self.setup[item_id] * sum(1 for t in range(T) if q_sol[t] > 0)
            total_var = self.c_var[item_id] * sum(q_sol)
            total_hold = sum(self.h[item_id] * (t - s) * x[(s, t)].X for (s, t) in x)
            true_cost = total_setup + total_var + total_hold
            return model.ObjVal, true_cost, q_sol
        return float("inf"), float("inf"), None

    # ---------- greedy primal from LP (capacity repair + FEFO flow) ----------
    def try_greedy_incumbent(self, lambda_vals):
        T = self.T
        usage = [0.0] * T
        q_int = {i: [0] * T for i in self.items}

        # 1) blend q from λ and round
        for i in self.items:
            lam = lambda_vals.get(i, [])
            if not lam:
                continue
            q_avg = [0.0] * T
            for p_idx, w in enumerate(lam):
                if w <= 1e-9:
                    continue
                pat = self.master.patterns[i][p_idx]
                for t, v in enumerate(pat["q"]):
                    q_avg[t] += w * v
            q_round = [int(round(v)) for v in q_avg]
            q_int[i] = q_round
            for t, v in enumerate(q_round):
                usage[t] += v

        # 2) capacity repair by right-shifting within shelf-life windows
        for t in range(T):
            extra = usage[t] - self.cap[t]
            if extra <= 1e-9:
                continue
            extra = int(math.ceil(extra))
            # Greedy: move units to earliest later period with slack & not beyond expiry
            while extra > 0:
                moved = False
                # sort items by small setup (so we prefer moving items that won't increase setups later)
                for i in sorted(self.items, key=lambda ii: self.setup[ii]):
                    if q_int[i][t] <= 0:
                        continue
                    life = max(1, int(self.mseq[i][t]))
                    t_exp = min(T - 1, t + life - 1)
                    # find earliest t' > t with slack and t' <= t_exp
                    dest = None
                    for tp in range(t + 1, t_exp + 1):
                        if usage[tp] + 1 <= self.cap[tp] + 1e-9:
                            dest = tp
                            break
                    if dest is None:
                        continue
                    # move 1 unit
                    q_int[i][t] -= 1
                    q_int[i][dest] += 1
                    usage[t] -= 1
                    usage[dest] += 1
                    extra -= 1
                    moved = True
                    if extra <= 0:
                        break
                if not moved:
                    # cannot repair this period; give up
                    break

        # 3) compute FEFO flows & cost; verify capacity
        if any(usage[t] > self.cap[t] + 1e-9 for t in range(T)):
            return float("inf"), None

        total_cost = 0.0
        for i in self.items:
            D = self.dem[i]
            mseq = self.mseq[i]
            flow = build_fefo_flow_from_q(D, q_int[i], mseq)
            served = [0.0] * T
            for (s, t), x in flow.items():
                served[t] += x
            if any(abs(served[t] - D[t]) > 1e-6 for t in range(T)):
                return float("inf"), None  # unmet demand
            setups = sum(1 for t in range(T) if q_int[i][t] > 0)
            var = self.c_var[i] * sum(q_int[i])
            hold = sum(self.h[i] * (t - s) * x for (s, t), x in flow.items())
            total_cost += self.setup[i] * setups + var + hold

        # 4) package as solution: add constructed patterns and pick λ=1 on them
        lam_fake = {i: [0.0] * len(self.master.patterns[i]) for i in self.items}
        for i in self.items:
            y = [1 if v > 0 else 0 for v in q_int[i]]
            # cost we computed includes all items; still attach 0 obj here, it won't be re-optimized
            self.master.add_pattern(i, 0.0, q_int[i], y)
            idx = len(self.master.patterns[i]) - 1
            lam_fake[i] += [0.0]  # ensure same length; we appended 1 var
            lam_fake[i][idx] = 1.0

        sol = {
            "lambda": lam_fake,
            "patterns": deepcopy(self.master.patterns),
            "order_fix": dict(self.order_fix),
        }
        return total_cost, sol

    # ---------- column generation ----------
    def column_generation(self):
        iteration = 0
        thin_hits = 0  # Counter for thin hits
        cg_guard = ConvergenceGuard(
            CG_ESTOP_MIN_ITERS, CG_ESTOP_PATIENCE, CG_ESTOP_MIN_REL_IMPROVE
        )
        while True:
            iteration += 1
            result = self.master.optimize()
            if result is None:
                log("[WARN] RMP infeasible; trying to restore via pricing.")
                added_any = False
                for i in self.items:
                    _, _, _, q_plan, _ = self.price_item(i, [0.0] * self.T, 0.0)
                    if q_plan:
                        cost = sum(
                            self.setup[i] * (1 if q > 0 else 0) for q in q_plan
                        ) + sum(self.c_var[i] * q for q in q_plan)
                        y_plan = [1 if q > 0 else 0 for q in q_plan]
                        self.master.add_pattern(i, cost, q_plan, y_plan)
                        added_any = True
                self.master.model.update()
                result = self.master.optimize()
                if result is None:
                    log("[FAIL] Could not restore feasibility.")
                    return float("inf"), ({}, [], {})
            obj, pi, mu, lambda_vals = result
            if VERBOSE and (iteration % LOG_EVERY_ITER == 0 or iteration == 1):
                log(f"[CG] iter {iteration:2d}  obj={obj:.2f}")
            mu_hat = [
                (self.alpha * m + (1 - self.alpha) * pm)
                for m, pm in zip(mu, self.prev_mu)
            ]
            self.prev_mu = mu_hat[:]
            added = False
            for i in self.items:
                _, rc, true_cost, q_plan, method = self.price_item(i, mu_hat, pi[i])
                if q_plan is not None and rc < -RED_COST_EPS:
                    y_plan = [1 if q > 0 else 0 for q in q_plan]
                    self.master.add_pattern(i, true_cost, q_plan, y_plan)
                    added = True
            self.master.model.update()

            # don't starve the pool: if stagnating but pool too small, keep going a bit
            if iteration >= CG_ESTOP_MIN_ITERS and cg_guard.step(obj):
                if (
                    self._min_patterns_per_item() < MIN_PATTERNS_PER_ITEM
                    and thin_hits < 3
                ):
                    thin_hits += 1
                    log(
                        f"[EARLY-STOP] stalled; pool thin (hit {thin_hits}/3) → 1 more sweep"
                    )
                    continue
                log("[EARLY-STOP] stopping CG")
                break
            if (not added) or iteration >= CG_MAX_ITERS:
                if (
                    self._min_patterns_per_item() < MIN_PATTERNS_PER_ITEM
                    and iteration < CG_MAX_ITERS
                ):
                    # force at least one more sweep to try add patterns
                    continue
                break

        # opportunistic greedy incumbent from current pool
        try:
            prim_val, prim_sol = self.try_greedy_incumbent(lambda_vals)
            if prim_sol:
                self._last_greedy = (prim_val, prim_sol)
                log(f"[HEUR] Greedy incumbent candidate {prim_val:.2f}")
        except Exception as e:
            log(f"[HEUR] Greedy incumbent failed: {e}")

        return obj, (pi, mu, lambda_vals)

    # ---------- branching ----------
    def choose_branch_variable(self, lambda_vals, mu):
        y_usage = self.compute_y_values(lambda_vals)
        best_score = -1.0
        best_key = None
        best_val = None
        frac = []
        for (i, t), val in y_usage.items():
            if 1e-6 < val < 1 - 1e-6:
                frac.append(((i, t), val))
                closeness = 0.5 - abs(val - 0.5)
                score = closeness * (1.0 + max(0.0, mu[t]))
                if score > best_score:
                    best_score = score
                    best_key = (i, t)
                    best_val = val
        if DEBUG_BRANCH and frac:
            top = sorted(frac, key=lambda kv: abs(0.5 - kv[1]))[:5]
            log(
                "[INFO] Fractional y (top): "
                + ", ".join(f"y[{i},{t}]={v:.3f}" for ((i, t), v) in top)
            )
        return best_key, best_val

    def solve_restricted_IP(self, time_limit=5.0):
        ip = grb.Model("RMP_IP", env=GRB_ENV)
        ip.Params.LogToConsole = 0
        ip.Params.OutputFlag = 0
        ip.Params.Threads = N_MASTER_THREADS
        lam_ip = {i: [] for i in self.items}
        for i in self.items:
            for p_idx, pat in enumerate(self.master.patterns[i]):
                v = ip.addVar(
                    vtype=grb.GRB.BINARY, obj=pat["cost"], name=f"lam_{i}_{p_idx}"
                )
                try:
                    v.Start = 1 if self.master.lambda_vars[i][p_idx].X > 0.5 else 0
                except Exception:
                    pass
                lam_ip[i].append(v)
        for i in self.items:
            ip.addConstr(grb.quicksum(lam_ip[i]) == 1, name=f"sel_{i}")
        for t in range(self.T):
            ip.addConstr(
                grb.quicksum(
                    pat["q"][t] * lam_ip[i][p_idx]
                    for i in self.items
                    for p_idx, pat in enumerate(self.master.patterns[i])
                    if pat["q"][t] > 0
                )
                <= self.cap[t],
                name=f"cap_{t}",
            )
        ip.ModelSense = grb.GRB.MINIMIZE
        ip.Params.TimeLimit = time_limit
        start = time.perf_counter()
        ip.optimize()
        elapsed = time.perf_counter() - start
        if ip.Status in (grb.GRB.OPTIMAL, grb.GRB.TIME_LIMIT):
            lam = {i: [v.X for v in lam_ip[i]] for i in self.items}
            log(
                f"[INFO] Restricted IP solved in {elapsed:.2f}s, status={ip.Status}, obj={ip.ObjVal:.2f}"
            )
            return ip.ObjVal, lam
        log(f"[WARN] Restricted IP failed status {ip.Status}")
        return float("inf"), None

    # ---------- DFS dive ----------
    def dive_for_incumbent(self, max_depth=15):
        saved_master = self.master
        saved_fix = dict(self.order_fix)
        saved_prev_mu = self.prev_mu[:]
        best = float("inf")
        best_sol = None
        depth = 0
        try:
            while depth < max_depth and not self._time_up():
                # short CG
                obj, (pi, mu, lambda_vals) = self.column_generation()
                # quick IP
                ip_obj, lam_int = self.solve_restricted_IP(time_limit=2.0)
                if lam_int is not None and ip_obj < best - 1e-6:
                    best = ip_obj
                    best_sol = {
                        "lambda": lam_int,
                        "patterns": deepcopy(self.master.patterns),
                        "order_fix": dict(self.order_fix),
                    }
                    break
                # branch direction: round toward side of y closest to 1
                (key, yhat) = self.choose_branch_variable(lambda_vals, mu)
                if not key:
                    break
                i_b, t_b = key
                fix = (1, 1) if (yhat or 0.0) >= 0.5 else (0, 0)
                self.order_fix[(i_b, t_b)] = fix
                self.master = self.master.clone_with_branching(order_fix=self.order_fix)
                depth += 1
        finally:
            # restore
            self.master = saved_master
            self.order_fix = saved_fix
            self.prev_mu = saved_prev_mu
        return best, best_sol

    # ---------- B&P main ----------
    def _node_log(self, parent_id, branch_fix, bound, incumbent, status):
        node_id = next(self._node_id_counter)
        self.tree[node_id] = {
            "id": node_id,
            "parent": parent_id,
            "fix": (branch_fix if branch_fix is not None else None),
            "obj": bound,
            "incumbent": (incumbent if incumbent < float("inf") else float("inf")),
            "status": status,
        }
        self.parent_stack.append(node_id)
        return node_id

    @staticmethod
    def _print_gap(bound, incumbent, prefix="[GAP] "):
        if incumbent < float("inf"):
            gap = 100.0 * (incumbent - bound) / max(1e-12, incumbent)
            log(f"{prefix}bound={bound:.2f}  best={incumbent:.2f}  gap={gap:.2f}%")
        else:
            log(f"{prefix}bound={bound:.2f}  best=∞  gap=∞")

    def _time_up(self) -> bool:
        return (time.perf_counter() - self._start_time) >= GLOBAL_TIME_LIMIT_SEC

    def branch_and_price(self, best_inc=float("inf"), best_sol=None):
        if self._time_up() or self._global_nodes >= GLOBAL_NODE_LIMIT:
            return best_inc, best_sol

        lp_obj, (pi, mu, lambda_vals) = self.column_generation()
        parent_id = self.parent_stack[-1] if self.parent_stack else None
        status = "branching" if lp_obj < best_inc - 1e-9 else "pruned"
        self._node_log(
            parent_id,
            (list(self.order_fix.items())[-1] if self.order_fix else None),
            lp_obj,
            best_inc,
            status,
        )
        BranchPrice._print_gap(lp_obj, best_inc)
        if lp_obj >= best_inc - 1e-9:
            return best_inc, best_sol

        # accept greedy candidate if good
        if self._last_greedy:
            gval, gsol = self._last_greedy
            if gsol and gval < best_inc:
                best_inc, best_sol = gval, gsol
                log(f"[SOL] Incumbent from greedy {best_inc:.2f}")
                BranchPrice._print_gap(lp_obj, best_inc, prefix="    ")

        if best_inc < float("inf"):
            gap_pct = 100.0 * (best_inc - lp_obj) / max(1e-12, best_inc)
            if gap_pct < BAP_GAP_TOL * 100:
                log(f"[EARLY-STOP] Search gap {gap_pct:.3f}% < {BAP_GAP_TOL*100:.2f}%")
                return best_inc, best_sol

        res = self.master.optimize()
        if res is None:
            return best_inc, best_sol
        _, _, mu_final, lambda_vals = res

        # quick dive for an incumbent
        inc_obj, inc_sol = self.dive_for_incumbent(max_depth=12)
        if inc_sol is not None and inc_obj < best_inc:
            best_inc, best_sol = inc_obj, inc_sol
            log(f"[SOL] Incumbent from dive {best_inc:.2f}")
            BranchPrice._print_gap(lp_obj, best_inc, prefix="    ")
            if best_inc < float("inf"):
                gap_pct = 100.0 * (best_inc - lp_obj) / max(1e-12, best_inc)
                if gap_pct < BAP_GAP_TOL * 100:
                    log(
                        f"[EARLY-STOP] Search gap {gap_pct:.3f}% < {BAP_GAP_TOL*100:.2f}%"
                    )
                    return best_inc, best_sol

        # choose branch var
        branch_key, yhat = self.choose_branch_variable(lambda_vals, mu_final)
        if branch_key is None:
            ip_obj, lam_int = self.solve_restricted_IP(time_limit=5.0)
            if lam_int is not None and ip_obj <= lp_obj + 1e-6:
                best_inc = ip_obj
                best_sol = {
                    "lambda": lam_int,
                    "patterns": deepcopy(self.master.patterns),
                    "order_fix": dict(self.order_fix),
                }
                log(f"[SOL] New incumbent {ip_obj:.2f} (restricted IP).")
                BranchPrice._print_gap(lp_obj, best_inc, prefix="    ")
            else:
                best_inc = lp_obj
                best_sol = {
                    "lambda": lambda_vals,
                    "patterns": deepcopy(self.master.patterns),
                    "order_fix": dict(self.order_fix),
                }
                log(f"[SOL] Incumbent {lp_obj:.2f} (LP solution, y integral).")
            if self.parent_stack:
                self.parent_stack.pop()
            return best_inc, best_sol

        i_b, t_b = branch_key
        log(
            f"[BRANCH] y[{i_b},{t_b}] fractional → branch (μ[{t_b}]={mu_final[t_b]:.3f}, y≈{(yhat or 0.0):.3f})"
        )

        # Explore rounded side first (DFS)
        order = (
            [((1, 1), "y=1"), ((0, 0), "y=0")]
            if (yhat or 0.0) >= 0.5
            else [((0, 0), "y=0"), ((1, 1), "y=1")]
        )

        for fix, tag in order:
            self._global_nodes += 1
            if self._time_up() or self._global_nodes >= GLOBAL_NODE_LIMIT:
                break
            self.order_fix[(i_b, t_b)] = fix
            child = BranchPrice.__new__(BranchPrice)
            for attr in (
                "items",
                "dem",
                "c_var",
                "h",
                "setup",
                "b_var",
                "cap",
                "T",
                "mseq",
                "k_max",
                "prev_mu",
                "alpha",
                "tree",
                "_node_id_counter",
                "parent_stack",
                "_global_nodes",
                "_start_time",
                "_last_greedy",
            ):
                setattr(child, attr, deepcopy(getattr(self, attr)))
            child.order_fix = dict(self.order_fix)
            child.master = self.master.clone_with_branching(order_fix=child.order_fix)
            log(f"  -> exploring {tag}")
            best_inc, best_sol = child.branch_and_price(best_inc, best_sol)
            self.order_fix.pop((i_b, t_b), None)

        if self.parent_stack:
            self.parent_stack.pop()
        return best_inc, best_sol


# ========================== Reporting / Audits ================================
def detailed_exec_rep(bp: BranchPrice, sol: Dict, lot: Optional[Lot] = None):
    log("\n=== Detailed execution report (per item) ===")
    lam = sol.get("lambda", {})
    patterns = sol.get("patterns", bp.master.patterns)
    for i in bp.items:
        print(f"\nItem {i}")
        lam_i = lam.get(i, [])
        if not lam_i:
            print("[ERROR] No pattern for this item.")
            continue
        sel = max(range(len(lam_i)), key=lambda k: lam_i[k])
        pat = patterns[i][sel]
        q = pat["q"]
        D = bp.dem[i]
        m_seq = [max(1, int(v)) for v in bp.mseq[i]]
        cohorts = []
        backlog = 0
        print(" t | demand | order | inventory_after | backlog")
        print("--------------------------------------------")
        for t in range(bp.T):
            if q[t] > 0:
                cohorts.append((m_seq[t], float(q[t])))
            need = D[t] + backlog
            backlog = 0
            cohorts.sort(key=lambda x: x[0])
            new = []
            for life, qty in cohorts:
                if need <= 0:
                    if life > 1:
                        new.append((life - 1, qty))
                    continue
                if life <= 0 or qty <= 0:
                    continue
                take = min(qty, need)
                qty -= take
                need -= take
                if qty > 1e-9 and life > 1:
                    new.append((life - 1, qty))
            cohorts = new
            if need > 1e-9:
                backlog = need
            inv = sum(qty for _, qty in cohorts)
            print(f"{t:2d} | {D[t]:6.1f} | {q[t]:5.1f} | {inv:14.1f} | {backlog:7.1f}")


def capacity_audit(bp: BranchPrice, sol: Dict):
    print("\n=== Capacity utilization audit ===")
    lam = sol.get("lambda", {})
    patterns = sol.get("patterns", bp.master.patterns)
    usage = [0.0] * bp.T
    for i in bp.items:
        lam_i = lam.get(i, [])
        if not lam_i:
            continue
        sel = max(range(len(lam_i)), key=lambda k: lam_i[k])
        q = patterns[i][sel]["q"]
        for t in range(bp.T):
            usage[t] += q[t]
    print("Period | Used  | Capacity | Slack")
    print("-------------------------------")
    for t in range(bp.T):
        used = usage[t]
        cap = bp.cap[t]
        slack = cap - used
        print(f"{t:6d} | {used:5.1f} | {cap:8.1f} | {slack:5.1f}")


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, s: str):
        for stream in self.streams:
            stream.write(s)
            try:
                stream.flush()
            except Exception:
                pass
        return len(s)

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def save_results(
    lot: Lot,
    bp: BranchPrice,
    solution: Dict,
    objective: float,
    elapsed_sec: float,
    log_text: str,
    instance_path: Optional[Path] = None,
    outdir: str = OUTDIR,
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if instance_path and instance_path.exists():
        try:
            (outdir / f"instance_{ts}.json").write_text(instance_path.read_text())
        except Exception:
            pass
    demand, c_var, h, setup, b_var, cap, mseq, kmax = lot.to_dicts()
    patterns_data = {
        i: [
            {"cost": pat["cost"], "y": pat["y"], "q": pat["q"]}
            for pat in (solution.get("patterns", bp.master.patterns))[i]
        ]
        for i in bp.items
    }
    tree_list = list(bp.tree.values())
    order_fix = (
        {
            f"{i}_{t}": list(bounds)
            for (i, t), bounds in solution.get("order_fix", {}).items()
        }
        if solution.get("order_fix")
        else {}
    )
    run = {
        "timestamp": ts,
        "objective": objective,
        "elapsed_seconds": elapsed_sec,
        "instance_path": str(instance_path) if instance_path else None,
        "capacity": cap,
        "items": {
            str(i): {
                "setup": setup[i],
                "b_var": b_var[i],
                "c_var": c_var[i],
                "h": h[i],
                "demand": demand[i],
                "shelf_life_sequence": mseq[i],
                "kmax": kmax[i],
                "agency": (lot.items[i].agency if i in lot.items else None),
            }
            for i in bp.items
        },
        "patterns": patterns_data,
        "lambda_solution": solution.get("lambda", {}),
        "order_fix": order_fix,
        "tree": tree_list,
    }
    json_path = outdir / f"run_{ts}.json"
    txt_path = outdir / f"run_{ts}.txt"
    json_path.write_text(json.dumps(run, indent=2))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Perishable Lot-Sizing Branch-and-Price Run Report ===\n")
        f.write(
            f"Timestamp: {ts}\nObjective: {objective:.2f}\nElapsed time (s): {elapsed_sec:.2f}\n"
        )
        if instance_path:
            f.write(f"Instance file: {instance_path}\n")
        f.write(
            "\n-- Capacity (per period) --\n" + ", ".join(str(c) for c in cap) + "\n"
        )
        f.write("\n-- Selected patterns (λ>0) --\n")
        lam = solution.get("lambda", {})
        for i in bp.items:
            lam_i = lam.get(i, [])
            if not lam_i:
                f.write(f"Item {i}: No pattern\n")
                continue
            sel = max(range(len(lam_i)), key=lambda k: lam_i[k])
            f.write(
                f"Item {i}: pattern {sel}, cost={patterns_data[i][sel]['cost']:.2f}, y={patterns_data[i][sel]['y']}, q={patterns_data[i][sel]['q']}\n"
            )
        f.write("\n-- Tree nodes --\n")
        for node in tree_list:
            f.write(json.dumps(node) + "\n")
        f.write("\n=== Console Output Log ===\n")
        f.write(log_text)
    return txt_path


# ================================== main ======================================
def main():
    global SEED
    random.seed(SEED)
    np.random.seed(SEED)
    if SPECS_PATH.exists():
        print(f"[INFO] Loading specs from {SPECS_PATH.resolve()}")
        lot = load_specs(SPECS_PATH)
    else:
        print("[INFO] specs.json not found; using default random instance.")
        lot = default_specs()

    output_buffer = io.StringIO()
    tee = Tee(sys.stdout, output_buffer)
    start = time.perf_counter()
    with contextlib.redirect_stdout(tee):
        demand, c_var, h, setup, b_var, cap, mseq, kmax = lot.to_dicts()
        bp = BranchPrice(demand, c_var, h, setup, b_var, cap, mseq, kmax)
        log("[INFO] Solving LP relaxation with column generation + branching...")
        best_obj, best_sol = bp.branch_and_price()
        if best_sol is None:
            print("[ERROR] No feasible solution found (capacity may be insufficient).")
            for t in range(lot.period):
                total_d = sum(item.demand[t] for item in lot.items.values())
                print(f"Period {t}: Demand={total_d}, Capacity={lot.capacity[t]}")
        else:
            elapsed = time.perf_counter() - start
            print(
                f"\nFinal Objective: {best_obj:.2f}   (Time elapsed: {elapsed:.2f} s)\n"
            )
            root_nodes = [n for n in bp.tree.values() if n["parent"] is None]
            root_bound = root_nodes[0]["obj"] if root_nodes else best_obj
            BranchPrice._print_gap(root_bound, best_obj, prefix="[FINAL] ")
            print(f"[INFO] Backorders allowed: False")
            # clean summary with agency
            lam = best_sol.get("lambda", {})
            patterns = best_sol.get("patterns", bp.master.patterns)
            print("\n-- Selected patterns by item --")
            print("Item | Agency.name / SKU     | λ*  | Setups | TotalQ | First 10 q[]")
            print(
                "-----+------------------------+-----+--------+--------+---------------------------"
            )
            for i in bp.items:
                lam_i = lam.get(i, [])
                if not lam_i:
                    print(f"{i:4d} | (none)               | 0   |   0    |   0    | []")
                    continue
                sel = max(range(len(lam_i)), key=lambda k: lam_i[k])
                pat = patterns[i][sel]
                q = pat["q"]
                ag = lot.items[i].agency or {}
                tag = f"{ag.get('name','-')}/{ag.get('sku','-')}"
                print(
                    f"{i:4d} | {tag:22s} | {lam_i[sel]:>3.2f} | {sum(1 for v in q if v>0):6d} | {sum(q):6d} | {str(q[:10])}"
                )
            detailed_exec_rep(bp, best_sol, lot)
            capacity_audit(bp, best_sol)
            print(
                f"\nFinal Objective: {best_obj:.2f}   (Time elapsed: {elapsed:.2f} s)"
            )

    log_text = output_buffer.getvalue()
    elapsed_total = time.perf_counter() - start
    if best_sol is None:
        best_obj = float("inf")
        best_sol = {}
    result_path = save_results(
        lot,
        bp,
        best_sol,
        best_obj,
        elapsed_total,
        log_text,
        instance_path=SPECS_PATH if SPECS_PATH.exists() else None,
        outdir=OUTDIR,
    )
    print(
        f"[RESULT] Output saved to {result_path.parent} (main report: {result_path.name})"
    )


if __name__ == "__main__":
    main()
