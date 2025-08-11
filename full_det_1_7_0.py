"""
Branch-and-price for a multi-item perishable lot-sizing problem
with setup costs, holding costs, NO backorders, dual stabilisation,
and order/no-order branching on (item, period).

This revision (DP-enabled):
  • Backorders OFF.
  • Pricing order (single-item):
        (1) FEFO Dynamic Program (Önal et al., 2015) — primary
        (2) Shortest-Path (SP) with variable shelf-life — fallback
        (3) MIP pricing (perishability-aware) — last resort
  • DP enforces FEFO and variable shelf-life; supports reduced-cost adjustments.
  • Early-stop guard in Column Generation if bound improves insignificantly
    over a substantial number of iterations — returns current best and gap.
  • Micro-RC filter RED_COST_EPS=1e-3 to avoid churn.
  • RMP cloning is silent (no [ADD] spam).
  • Extra debug:
        - capacity audit for the selected incumbent
        - stronger pattern validity guard-rails.
"""

from __future__ import annotations

# stdlib
import contextlib
import io
import itertools
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# third-party
import numpy as np
import gurobipy as grb

try:
    import matplotlib.pyplot as plt  # optional
    import networkx as nx  # optional
except Exception:
    plt = None
    nx = None

# =============================================================================
# Config
# =============================================================================
ENABLE_LIVE_PLOTS = False
TREE_VISUALIZATION = False

RANDOMIZE = True
USE_MANUAL_CAPACITY = True
ALLOW_BACKORDER = False  # keep OFF
SEED = 0
INSTANCE_PATH = Path(__file__).with_name("last_instance.json")

VERBOSE = True
LOG_EVERY_ITER = 1
CG_MAX_ITERS = 100  # allow many, early-stop will end sooner if stagnant

# Column gen early-stop ("substantial iterations" with no "significant" improve)

CG_ESTOP_MIN_ITERS = 12
CG_ESTOP_PATIENCE = 12
CG_ESTOP_MIN_REL_IMPROVE = 5e-5

# kill near-zero columns
RED_COST_EPS = 1e-3

# Debug toggles
DEBUG_DUALS = True
DEBUG_PRICING = True
DEBUG_BRANCH = True
DEBUG_SUMMARY_WIDTH = 8

# Threads
N_MASTER_THREADS = max(1, (os.cpu_count() or 4) // 2)
PRICING_N_THREADS = max(1, (os.cpu_count() or 4) // 2)

# MIP pricing time limit
MIP_PRICE_TIMELIMIT_SEC = 5.0

# Dual stabilization
STAB_ALPHA = 0.9  # was 0.6

# =============================================================================
# Quiet Gurobi env
# =============================================================================


def make_env(quiet: bool = True) -> grb.Env:
    env = grb.Env(empty=True)
    if quiet:
        env.setParam("LogToConsole", 0)
        env.setParam("OutputFlag", 0)
    env.start()
    return env


GRB_ENV = make_env(quiet=True)

# =============================================================================
# Logging helpers
# =============================================================================


def log(msg: str) -> None:
    if VERBOSE:
        print(msg)


def _stats(v: List[float]) -> str:
    if not v:
        return "min=NA max=NA mean=NA l1=NA"
    mn = min(v)
    mx = max(v)
    mean = sum(v) / len(v)
    l1 = sum(abs(x) for x in v)
    return f"min={mn:.3g} max={mx:.3g} mean={mean:.3g} l1={l1:.3g}"


def _head(v: List[float], k: int) -> str:
    return ", ".join(f"{x:.3g}" for x in v[:k])


# =============================================================================
# FEFO Dynamic Program (Önal et al., 2015) for single item (no backlog)
# =============================================================================

Number = float


@dataclass
class DPResult:
    total_cost: Number  # reduced-cost objective if p already includes duals
    x_ti: List[List[Number]]  # allocation matrix [t][i]
    x_t: List[Number]  # order quantity at t
    y_t: List[int]  # setup indicator at t


def _prefix_sums(a: List[Number]) -> List[Number]:
    ps = [0.0]
    s = 0.0
    for v in a:
        s += v
        ps.append(s)
    return ps


def _compute_c_ti(p: List[Number], h: List[Number]) -> List[List[Number]]:
    """c[t][i] = p_t + sum_{j=t}^{i-1} h_j  for i >= t; else inf."""
    T = len(p)
    H = _prefix_sums(h)  # H[k] = sum_{j<k} h_j
    c = [[math.inf] * T for _ in range(T)]
    for t in range(T):
        for i in range(t, T):
            c[t][i] = p[t] + (H[i] - H[t])
    return c


def dp_fefo_solve(
    D: List[Number],
    p_eff: List[Number],  # variable cost per unit adjusted by duals (c - mu)
    h_seq: List[Number],  # holding cost per period (constant ⇒ repeat)
    S_seq: List[Number],  # setup cost per period
    v_last: List[int],  # last period each order can serve (inclusive)
    forbid_t: Optional[set] = None,  # forbid placing an order at these periods (y_t=0)
) -> DPResult:
    """
    Single-item FEFO DP as in Önal et al. (2015) for an uncapacitated item.
    - No backorders: every demand D[i] must be served by some order t<=i with i<=v_last[t].
    - We compute f(t; τ1, τ2), g(t; τ1, τ2), f_min(τ1, τ2), g_min(τ1, τ2) over a demand
      interval [τ1, τ2] with the FEFO structure. p_eff already includes dual adjustments.
    - For forbidding orders at certain t (due to branching y_t=0), we set S_t=+inf.
    Returns allocation matrix x_ti, aggregate x_t, and y_t along with total cost.
    """
    T = len(D)
    assert len(p_eff) == T and len(h_seq) == T and len(S_seq) == T and len(v_last) == T

    # Apply forbidden starts by inflating S to +inf
    if forbid_t:
        S_eff = [(math.inf if t in forbid_t else S_seq[t]) for t in range(T)]
    else:
        S_eff = S_seq[:]

    c = _compute_c_ti(p_eff, h_seq)

    inf = math.inf
    # sparse dicts for f_t and g_t
    f_t: Dict[Tuple[int, int, int], Number] = {}
    g_t: Dict[Tuple[int, int, int], Number] = {}

    # dense for minima and backpointers
    f_min = [[inf] * T for _ in range(T)]
    g_min = [[inf] * T for _ in range(T)]
    tstar: Dict[Tuple[int, int], int] = {}
    split_g: Dict[Tuple[int, int], int] = {}

    def gmin(a: int, b: int) -> Number:
        return 0.0 if a > b else g_min[a][b]

    # base: spans of length 0
    for tau in range(T):
        for t in range(tau + 1):
            if v_last[t] < tau or math.isinf(S_eff[t]):
                fval = inf
            else:
                fval = S_eff[t] + c[t][tau] * D[tau]
            f_t[(t, tau, tau)] = fval
            g_t[(t, tau, tau)] = fval
        # f_min and g_min at (tau, tau)
        best = inf
        best_t = -1
        for t in range(tau + 1):
            val = f_t[(t, tau, tau)]
            if val < best:
                best = val
                best_t = t
        f_min[tau][tau] = best
        g_min[tau][tau] = best
        tstar[(tau, tau)] = best_t
        split_g[(tau, tau)] = tau

    # main DP by span length k
    for k in range(1, T):
        for tau1 in range(0, T - k):
            tau2 = tau1 + k
            # update f_t & g_t for all t <= tau1
            for t in range(0, tau1 + 1):
                if v_last[t] < tau2 or math.isinf(S_eff[t]):
                    fval = inf
                    gval = inf
                else:
                    prev = g_t.get((t, tau1, tau2 - 1), inf)
                    fval = prev + c[t][tau2] * D[tau2]
                    gval = inf
                    for tau in range(tau1, tau2 + 1):
                        f_part = f_t.get((t, tau1, tau), inf)
                        cand = f_part + gmin(tau + 1, tau2)
                        if cand < gval:
                            gval = cand
                f_t[(t, tau1, tau2)] = fval
                g_t[(t, tau1, tau2)] = gval

            # f_min(tau1, tau2)
            bestf = inf
            best_t = -1
            for t in range(0, tau1 + 1):
                val = f_t[(t, tau1, tau2)]
                if val < bestf:
                    bestf = val
                    best_t = t
            f_min[tau1][tau2] = bestf
            tstar[(tau1, tau2)] = best_t

            # g_min(tau1, tau2) = min_{tau} f_min(tau1, tau) + g_min(tau+1, tau2)
            bestg = inf
            best_tau = -1
            for tau in range(tau1, tau2 + 1):
                left = f_min[tau1][tau]
                right = gmin(tau + 1, tau2)
                cand = left + right
                if cand < bestg:
                    bestg = cand
                    best_tau = tau
            g_min[tau1][tau2] = bestg
            split_g[(tau1, tau2)] = best_tau

    total = g_min[0][T - 1]

    # reconstruct FEFO subplans and allocation x_{t,i}
    x_ti = [[0.0] * T for _ in range(T)]
    x_t = [0.0] * T
    y_t = [0] * T

    def rec(a: int, b: int):
        if a > b:
            return
        tau_star = split_g[(a, b)]
        t_best = tstar[(a, tau_star)]
        if t_best == -1 or math.isinf(f_min[a][tau_star]):
            raise RuntimeError(
                f"Infeasible FEFO subplan reconstruction on [{a},{tau_star}]."
            )
        for i in range(a, tau_star + 1):
            x_ti[t_best][i] = D[i]
            x_t[t_best] += D[i]
        y_t[t_best] = 1
        rec(tau_star + 1, b)

    if T > 0:
        rec(0, T - 1)

    return DPResult(total_cost=total, x_ti=x_ti, x_t=x_t, y_t=y_t)


def build_fefo_flow_from_q(
    demand: List[int], q: List[int], m_seq: List[int]
) -> Dict[Tuple[int, int], float]:
    """
    Build a FEFO-consistent flow x[(s,t)] from a contiguous 'q' plan.
    Each positive q[s] starts a cohort with initial lifetime m_seq[s].
    We allocate to periods t >= s, respecting expiry t <= s + m_seq[s] - 1,
    serving demand in FEFO order (earliest expiry first).
    """
    T = len(demand)
    x: Dict[Tuple[int, int], float] = {}

    # Cohorts list: (expiry_time, remaining_qty, start_period)
    cohorts: List[Tuple[int, float, int]] = []
    for s in range(T):
        qty = q[s]
        if qty > 0:
            exp = min(T - 1, s + m_seq[s] - 1)
            cohorts.append((exp, float(qty), s))

    # Serve demand period by period in FEFO
    for t in range(T):
        need = float(demand[t])
        # sort by expiry, then by start period (stable FEFO)
        cohorts.sort(key=lambda e: (e[0], e[2]))
        new_coh = []
        for exp, rem, s in cohorts:
            if need <= 1e-12:
                new_coh.append((exp, rem, s))
                continue
            if t > exp or rem <= 0:
                # expired or empty; drop it
                continue
            use = min(rem, need)
            if use > 0:
                x[(s, t)] = x.get((s, t), 0.0) + use
                rem -= use
                need -= use
            if rem > 1e-12:
                new_coh.append((exp, rem, s))
        cohorts = new_coh

        # If need > 0 here, the (q,demand) pair is infeasible; leave as is.
        # (MIP will still accept the warm start; it's just a hint.)

    return x


# =============================================================================
# Shortest path pricing (no backorders) — retained as fallback
# =============================================================================


def price_shortest_path(
    item_id: int,
    demand: List[int],
    c_var: float,
    h: float,
    setup: float,
    mu: List[float],
    pi: float,
    m_seq: Optional[List[int]],
    k_max: int,
    order_fix: Dict[Tuple[int, int], Tuple[int, int]],
) -> Tuple[float, float, Optional[List[int]]]:
    """
    Layered DAG shortest path: nodes 0..T; arc (t -> u+1) = one order at t covering demand[t..u].
    Feasible if u <= t + m_seq[t] - 1 and cumulative qty ≤ k_max.
    We DO NOT allow "skip at t". Forbid y[t]=0 by removing arcs starting at t.
    Returns (reduced_cost, true_cost, q) or (inf, inf, None) if no improvement.
    """
    T = len(demand)
    dist = [float("inf")] * (T + 1)
    nxt_arc: List[Optional[Tuple[int, int]]] = [None] * T
    dist[T] = -pi

    forbidden_starts = {
        tt for (ii, tt), (lb, ub) in order_fix.items() if ii == item_id and ub == 0
    }

    for t in range(T - 1, -1, -1):
        best = float("inf")
        best_arc: Optional[Tuple[int, int]] = None

        if t not in forbidden_starts:
            q_acc = 0
            hold_acc = 0
            u_cap = T - 1
            if m_seq is not None:
                u_cap = min(u_cap, t + m_seq[t] - 1)

            for u in range(t, u_cap + 1):
                q_acc += demand[u]
                if q_acc > k_max:
                    break
                hold_acc += demand[u] * (u - t)
                red_arc = (
                    (c_var - mu[t]) * q_acc + h * hold_acc + (setup if q_acc else 0)
                )
                cand = red_arc + dist[u + 1]
                if cand < best:
                    best = cand
                    best_arc = (u, q_acc)

        dist[t] = best
        nxt_arc[t] = best_arc

    if math.isinf(dist[0]) or dist[0] >= -RED_COST_EPS:
        return float("inf"), float("inf"), None

    # reconstruct plan (no 'None' arcs allowed when dist[0] < inf)
    q = [0] * T
    true_cost = 0.0
    t = 0
    while t < T:
        arc = nxt_arc[t]
        assert arc is not None, "SP reconstruction encountered a missing arc (bug)."
        u, qty = arc
        q[t] = qty
        hold = sum(demand[tau] * (tau - t) for tau in range(t, u + 1))
        true_cost += setup + c_var * qty + h * hold
        t = u + 1

    if sum(q) == 0:
        return float("inf"), float("inf"), None

    return dist[0], true_cost, q


# =============================================================================
# Early-stop guard for column generation
# =============================================================================


class ConvergenceGuard:
    def __init__(self, min_iters: int, patience: int, min_rel_improve: float):
        self.min_iters = min_iters
        self.patience = patience
        self.min_rel = min_rel_improve
        self.prev = None
        self.stale = 0

    def step(self, value: float) -> bool:
        """Return True if should stop early."""
        stop = False
        if self.prev is not None:
            denom = max(1.0, abs(self.prev))
            rel = (self.prev - value) / denom  # minimize: improvement if positive
            if rel < self.min_rel:
                self.stale += 1
            else:
                self.stale = 0
        self.prev = value
        if self.prev is not None and self.stale >= self.patience:
            stop = True
        return stop


# =============================================================================
# Master problem (Restricted Master Problem / RMP)
# =============================================================================


class MasterModel:
    """
    RMP:
      - λ_{i,p} select pattern p for item i
      - selection: sum_p λ_{i,p} = 1
      - capacity:  sum_i sum_p q_{i,p,t} λ_{i,p} ≤ cap_t
      - branching fix y_{i,t} via additional rows
    """

    def __init__(
        self,
        items: List[int],
        T: int,
        capacity: List[int],
        *,
        env: grb.Env,
        verbose: bool = False,
    ):
        self.items, self.T, self.capacity = items, T, capacity
        self.model = grb.Model("RMP", env=env)
        self.model.Params.OutputFlag = 1 if verbose else 0
        self.model.Params.Method = 1  # dual simplex
        self.model.Params.Threads = N_MASTER_THREADS
        self.lambda_vars: Dict[int, List[grb.Var]] = {i: [] for i in items}
        self.patterns: Dict[int, List[Dict]] = {i: [] for i in items}
        le = grb.LinExpr
        self.sel_constr = {
            i: self.model.addConstr(le() == 1.0, name=f"sel_{i}") for i in items
        }
        self.cap_constr = [
            self.model.addConstr(le() <= capacity[t], name=f"cap_{t}") for t in range(T)
        ]
        self.order_rows: Dict[Tuple[int, int], grb.Constr] = {}

    def _ensure_order_row(self, ii: int, tt: int, lb: int, ub: int) -> None:
        if lb != ub:
            return
        key = (ii, tt)
        if key in self.order_rows:
            return
        expr = grb.LinExpr()
        if lb == 1:
            constr = self.model.addConstr(expr >= 1.0, name=f"ord1_{ii}_{tt}")
        else:
            constr = self.model.addConstr(expr <= 0.0, name=f"ord0_{ii}_{tt}")
        self.order_rows[key] = constr

    def add_pattern(
        self, i: int, cost: float, q: List[int], *, log_add: bool = True
    ) -> None:
        # Refuse duplicates / all-zero
        if sum(q) == 0:
            if log_add:
                log(f"[WARN] ignoring zero-qty pattern for item {i}")
            return
        if any(p["q"] == q for p in self.patterns[i]):
            return
        y = [1 if qty > 0 else 0 for qty in q]
        col = grb.Column()
        col.addTerms(1.0, self.sel_constr[i])
        for t, qty in enumerate(q):
            if qty:
                col.addTerms(qty, self.cap_constr[t])
        for (ii, tt), row in self.order_rows.items():
            if ii == i and y[tt]:
                col.addTerms(1.0, row)
        v = self.model.addVar(
            obj=cost, column=col, name=f"lam_{i}_{len(self.lambda_vars[i])}"
        )
        self.lambda_vars[i].append(v)
        self.patterns[i].append(dict(cost=cost, q=q, y=y))
        if log_add:
            log(f"[ADD] item {i} col#{len(self.lambda_vars[i])-1} cost={cost:.2f}")

    def optimize(self):
        self.model.optimize()
        if self.model.Status == grb.GRB.INFEASIBLE:
            return None
        if self.model.Status not in (grb.GRB.OPTIMAL,):
            raise RuntimeError(f"Unexpected RMP status: {self.model.Status}")
        pi = {i: self.sel_constr[i].Pi for i in self.items}
        mu = [c.Pi for c in self.cap_constr]
        lam = {i: [v.X for v in self.lambda_vars[i]] for i in self.items}
        return self.model.ObjVal, pi, mu, lam

    def copy(self, order_fix=None) -> "MasterModel":
        clone = MasterModel(
            self.items, self.T, self.capacity, env=GRB_ENV, verbose=False
        )
        # add existing patterns silently
        for i in self.items:
            for pat in self.patterns[i]:
                clone.add_pattern(i, pat["cost"], pat["q"], log_add=False)
        if order_fix:
            for (ii, tt), (lb, ub) in order_fix.items():
                if lb != ub:
                    continue
                clone._ensure_order_row(ii, tt, lb, ub)
            for (ii, tt), row in clone.order_rows.items():
                for idx, pat in enumerate(clone.patterns[ii]):
                    if pat["y"][tt]:
                        v = clone.lambda_vars[ii][idx]
                        clone.model.chgCoeff(row, v, 1.0)
        clone.model.update()
        return clone


# =============================================================================
# Branch-and-Price driver
# =============================================================================


class BranchPrice:
    def __init__(self, demand, c_var, h, setup, b_var, capacity, shelf_seq, k_max):
        self.items = list(demand)
        self.dem, self.c_var, self.h = demand, c_var, h
        self.setup, self.b_var = setup, b_var
        self.cap, self.T = capacity, len(capacity)
        self.mseq = shelf_seq
        self.k_max = k_max

        self.order_fix: Dict[Tuple[int, int], Tuple[int, int]] = {}

        self.prev_mu = [0.0] * self.T
        self.alpha = STAB_ALPHA
        self.master = MasterModel(
            self.items, self.T, self.cap, env=GRB_ENV, verbose=False
        )

        # Seed columns: produce-to-demand
        for i in self.items:
            q = self.dem[i]
            cost = sum(self.c_var[i] * q_t for q_t in q) + self.setup[i] * sum(
                1 for q_t in q if q_t > 0
            )
            self.master.add_pattern(i, cost, q)
        self.master.model.update()

        # Tree bookkeeping
        self.tree: Dict[int, Dict] = {}
        self._idgen = itertools.count()
        self.parent_stack: List[Optional[int]] = []

        self.last_mu: List[float] = [0.0] * self.T

        log(
            f"[INFO] Init BP: items={len(self.items)} T={self.T} cap_stats[{_stats(self.cap)}]"
        )
        if DEBUG_DUALS:
            log(f"[INFO] ALLOW_BACKORDER={ALLOW_BACKORDER}  STAB_ALPHA={self.alpha}")

    # ---- logging helpers
    def log_node(self, parent, fix, obj, incumbent, status):
        node_id = next(self._idgen)
        self.tree[node_id] = dict(
            id=node_id,
            parent=parent,
            fix=fix,
            obj=obj,
            incumbent=incumbent,
            status=status,
        )
        self.parent_stack.append(node_id)
        return node_id

    def compute_y(self, lam) -> Dict[Tuple[int, int], float]:
        y = {(i, t): 0.0 for i in self.items for t in range(self.T)}
        for i, vlist in lam.items():
            for idx, v in enumerate(vlist):
                if v <= 1e-12:
                    continue
                yvec = self.master.patterns[i][idx]["y"]
                for t, bit in enumerate(yvec):
                    if bit:
                        y[(i, t)] += v
        return y

    # ---- DP pricing (FEFO) — primary
    def price_dp_fefo(
        self, item_id: int, mu_hat: List[float], pi_i: float
    ) -> Tuple[
        float, float, Optional[List[int]], Optional[Dict[Tuple[int, int], float]]
    ]:
        D = self.dem[item_id]
        T = len(D)
        m_seq = self.mseq[item_id]
        kmax = self.k_max[item_id]

        # Enforce forbidden starts (y=0) via +inf setup at those t
        forbid = {
            t
            for (ii, t), (lb, ub) in self.order_fix.items()
            if ii == item_id and ub == 0
        }
        # If any period is forced to y=1, defer to MIP (DP doesn't hard-enforce y=1)
        forced = any(
            (ii == item_id and lb == 1 and ub == 1)
            for (ii, _t), (lb, ub) in self.order_fix.items()
        )
        if forced:
            return float("inf"), float("inf"), None, None

        p_eff = [self.c_var[item_id] - mu_hat[t] for t in range(T)]
        h_seq = [self.h[item_id]] * T
        S_seq = [self.setup[item_id]] * T
        v_last = [min(T - 1, t + m_seq[t] - 1) for t in range(T)]

        dp = dp_fefo_solve(D, p_eff, h_seq, S_seq, v_last, forbid_t=forbid)
        # Reduced cost subtracts π_i
        rc = dp.total_cost - pi_i
        if rc >= -RED_COST_EPS:
            return float("inf"), float("inf"), None, None

        # Build pattern quantities q[t] and compute true (undiscounted) cost
        q = [int(round(x)) for x in dp.x_t]
        if sum(q) == 0 or sum(q) > kmax:
            return float("inf"), float("inf"), None, None

        # true cost with original costs
        h = self.h[item_id]
        c = self.c_var[item_id]
        S = self.setup[item_id]
        hold_true = 0.0
        for t in range(T):
            for i in range(T):
                if dp.x_ti[t][i] > 1e-12:
                    hold_true += h * (i - t) * dp.x_ti[t][i]
        setup_true = S * sum(1 for t in range(T) if q[t] > 0)
        var_true = c * sum(q)
        true_cost = setup_true + var_true + hold_true

        # warm_x straight from the DP allocations
        warm_x = {
            (t, i): float(dp.x_ti[t][i])
            for t in range(T)
            for i in range(T)
            if dp.x_ti[t][i] > 1e-12
        }
        return rc, true_cost, q, warm_x

    # ---- MIP pricing (perishability-aware) — no backorders
    def price_mip_perishable(
        self,
        item_id: int,
        mu: List[float],
        pi_i: float,
        *,
        warm_q: Optional[List[int]] = None,
        warm_x: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> Tuple[float, float, Optional[List[int]]]:
        D = self.dem[item_id]
        T = len(D)
        m_seq = self.mseq[item_id]
        kmax = self.k_max[item_id]

        m = grb.Model(f"price_per_{item_id}", env=GRB_ENV)
        m.Params.OutputFlag = 0
        m.Params.Presolve = 2
        m.Params.Method = 2
        m.Params.Cuts = 2
        m.Params.Heuristics = 0.5
        m.Params.Threads = 1
        m.Params.TimeLimit = MIP_PRICE_TIMELIMIT_SEC
        # New: make pricing focus on negative reduced-cost quickly
        m.Params.MIPFocus = 1
        m.Params.Cutoff = -RED_COST_EPS  # we only care about rc < -eps
        m.Params.IntFeasTol = 1e-9
        m.Params.OptimalityTol = 1e-9

        q = m.addVars(T, vtype=grb.GRB.INTEGER, lb=0, ub=kmax, name="q")
        y = m.addVars(T, vtype=grb.GRB.BINARY, name="y")
        x = {}
        for s in range(T):
            v_s = min(T - 1, s + m_seq[s] - 1)
            for t in range(s, v_s + 1):
                x[(s, t)] = m.addVar(
                    vtype=grb.GRB.CONTINUOUS, lb=0.0, name=f"x_{s}_{t}"
                )

        # demand satisfaction (no backlog)
        for t in range(T):
            m.addConstr(
                grb.quicksum(x[(s, t)] for s in range(t + 1) if (s, t) in x) == D[t],
                name=f"demand_{t}",
            )

        # flow capacity per cohort
        for s in range(T):
            out = grb.quicksum(x[(s, t)] for t in range(s, T) if (s, t) in x)
            m.addConstr(out <= q[s], name=f"cap_{s}")

        # setup linkage
        for s in range(T):
            m.addGenConstrIndicator(y[s], True, q[s] >= 1)
            m.addGenConstrIndicator(y[s], False, q[s] == 0)

        setup_cost = self.setup[item_id] * y.sum()
        var_cost = self.c_var[item_id] * q.sum()
        hold_cost = grb.quicksum(self.h[item_id] * (t - s) * x[(s, t)] for (s, t) in x)
        dual_adj = grb.quicksum((-mu[s]) * q[s] for s in range(T)) - pi_i
        m.setObjective(
            setup_cost + var_cost + hold_cost + dual_adj, sense=grb.GRB.MINIMIZE
        )

        # apply any y-fixes
        for t in range(T):
            lb, ub = self.order_fix.get((item_id, t), (0, 1))
            y[t].LB, y[t].UB = lb, ub

        # Warm starts for q,y
        if warm_q is not None:
            for t, qty in enumerate(warm_q):
                if qty is None:
                    continue
                q[t].Start = max(0, int(qty))
                y[t].Start = 1 if qty and qty > 0 else 0

        # Warm starts for x (flow) — huge speedup on many instances
        if warm_x is not None:
            for key, val in warm_x.items():
                if key in x and val is not None and val > 0.0:
                    x[key].Start = float(val)

        m.optimize()
        # With Cutoff set, solver may prove no column with rc<-eps; treat as no-improve
        if (
            m.Status not in (grb.GRB.OPTIMAL, grb.GRB.CUTOFF)
            or m.ObjVal >= -RED_COST_EPS
        ):
            # If m.Status == CUTOFF → no solution < cutoff found (i.e., no improving col)
            return float("inf"), float("inf"), None

        q_plan = [int(round(q[t].X)) for t in range(T)]
        setup_true = self.setup[item_id] * sum(1 for t in range(T) if q_plan[t] > 0)
        var_true = self.c_var[item_id] * sum(q_plan)
        hold_true = sum(self.h[item_id] * (t - s) * x[(s, t)].X for (s, t) in x)
        true_val = setup_true + var_true + hold_true
        return m.ObjVal, true_val, q_plan

    # ---- pricing driver for one item
    def price_item(self, i: int, mu_hat: List[float], pi_i: float):
        must_order = {
            t
            for (ii, t), (lb, ub) in self.order_fix.items()
            if ii == i and lb == ub == 1
        }
        warm_q: Optional[List[int]] = None
        warm_x: Optional[Dict[Tuple[int, int], float]] = None

        # 1) DP FEFO (only if no lb==1)
        if not must_order:
            rc, cost, q, x_hint = self.price_dp_fefo(i, mu_hat, pi_i)
            if q is not None and rc < -RED_COST_EPS:
                return i, rc, cost, q, "DP"
            warm_q = q
            warm_x = x_hint

        # 2) SP fallback (only if no lb==1)
        if not must_order:
            rc, cost, q = price_shortest_path(
                i,
                self.dem[i],
                self.c_var[i],
                self.h[i],
                self.setup[i],
                mu_hat,
                pi_i,
                self.mseq[i],
                self.k_max[i],
                self.order_fix,
            )
            if q is not None and rc < -RED_COST_EPS:
                return i, rc, cost, q, "SP"
            if warm_q is None and q is not None:
                warm_q = q
                # Build a consistent FEFO warm_x from SP's contiguous coverage
                warm_x = build_fefo_flow_from_q(self.dem[i], q, self.mseq[i])

        # 3) MIP pricing (handles y-fixes too)
        rc, cost, q = self.price_mip_perishable(
            i, mu_hat, pi_i, warm_q=warm_q, warm_x=warm_x
        )
        if q is not None and rc < -RED_COST_EPS:
            return i, rc, cost, q, "MIP"

        # 4) No improving column
        return i, float("inf"), float("inf"), None, "NONE"

    def column_generation(self):
        it = 0
        suppressed_micro = 0
        cg_guard = ConvergenceGuard(
            min_iters=CG_ESTOP_MIN_ITERS,
            patience=CG_ESTOP_PATIENCE,
            min_rel_improve=CG_ESTOP_MIN_REL_IMPROVE,
        )

        best_obj = float("inf")

        while True:
            it += 1
            res = self.master.optimize()
            if res is None:
                # ---- Feasibility restoration via pricing (NEW) ----
                log(
                    "[WARN] RMP infeasible at this node; attempting feasibility restoration via pricing..."
                )
                # Use last stabilized duals if available, else zeros
                mu_hat_boot = (
                    self.prev_mu if getattr(self, "prev_mu", None) else [0.0] * self.T
                )
                added_any = False
                for i in self.items:
                    # π_i unknown without feasible LP; use 0.0
                    _i, rc, cost, q, tag = self.price_item(i, mu_hat_boot, 0.0)
                    if q is not None:
                        # We don't care about rc sign here; we just need feasibility-enabling columns
                        self.master.add_pattern(i, cost, q, log_add=False)
                        added_any = True
                self.master.model.update()
                res = self.master.optimize()
                if res is None:
                    log("[FAIL] Could not restore feasibility at this node.")
                    return float("inf"), (None, None, None)
                else:
                    log("[INFO] Feasibility restored; continuing CG.")

            obj, pi, mu, lam = res
            self.last_mu = mu[:]
            best_obj = min(best_obj, obj)

            if VERBOSE and (it % LOG_EVERY_ITER == 0 or it == 1):
                log(f"[CG] iter {it:2d}  obj={obj:.2f}")
                if DEBUG_DUALS:
                    log(
                        f"[INFO] μ stats: {_stats(mu)}; head[{_head(mu, DEBUG_SUMMARY_WIDTH)}]"
                    )

            # stabilized duals
            mu_hat = [
                self.alpha * m + (1 - self.alpha) * p for m, p in zip(mu, self.prev_mu)
            ]
            self.prev_mu = mu_hat

            if DEBUG_DUALS:
                log(
                    f"[INFO] μ̂ stats (stab α={self.alpha}): {_stats(mu_hat)}; head[{_head(mu_hat, DEBUG_SUMMARY_WIDTH)}]"
                )

            added = False
            added_cnt = 0
            per_item_rc = {}
            per_item_tag = {}

            from concurrent.futures import ThreadPoolExecutor, as_completed

            futures = []
            with ThreadPoolExecutor(max_workers=PRICING_N_THREADS) as pool:
                for i in self.items:
                    futures.append(pool.submit(self.price_item, i, mu_hat, pi[i]))
                for f in as_completed(futures):
                    i, rc, cost, q, tag = f.result()
                    per_item_rc[i] = rc
                    per_item_tag[i] = tag
                    if q is not None:
                        if rc < -RED_COST_EPS:
                            self.master.add_pattern(i, cost, q)
                            added = True
                            added_cnt += 1
                        else:
                            suppressed_micro += 1  # micro-negative or ~0

            if DEBUG_PRICING:
                rc_pairs = ", ".join(
                    f"{i}: {per_item_tag.get(i,'?')} rc={per_item_rc.get(i, float('inf')):.3g}"
                    for i in sorted(self.items)
                )
                log(f"[INFO] pricing summary → {rc_pairs}")
                log(
                    f"[INFO] added {added_cnt} columns (total now={sum(len(self.master.patterns[i]) for i in self.items)})"
                )

            self.master.model.update()

            # Early-stop guard check
            if it >= CG_ESTOP_MIN_ITERS and cg_guard.step(obj):
                log(
                    f"[EARLY-STOP] CG stalled: < {CG_ESTOP_MIN_REL_IMPROVE:.3g} rel improvement for {CG_ESTOP_PATIENCE} iters."
                )
                break

            # Natural termination: no added columns, or hard cap on iters
            if not added or it >= CG_MAX_ITERS:
                if suppressed_micro:
                    log(
                        f"[INFO] suppressed {suppressed_micro} micro-columns (|rc| ≤ {RED_COST_EPS})"
                    )
                break

        return obj, (pi, mu, lam)

    # ---- choose branching variable (closest to 0.5; tie-break by μ_t)
    def choose_branch(self, lam, mu):
        y = self.compute_y(lam)
        best_key = None
        best_score = -1.0
        frac = []
        for (i, t), val in y.items():
            if val <= 1e-6 or val >= 1 - 1e-6:
                continue
            closeness = 0.5 - abs(val - 0.5)
            # Count how many positive-λ columns for item i actually use y[i,t]=1
            usage = 0
            for idx, v in enumerate(lam.get(i, [])):
                if v > 1e-9 and self.master.patterns[i][idx]["y"][t]:
                    usage += 1
            score = closeness * (1.0 + max(0.0, mu[t])) * (1.0 + 0.1 * usage)
            frac.append(((i, t), val, score))
            if score > best_score:
                best_score, best_key = score, (i, t)

        if DEBUG_BRANCH:
            log(f"[INFO] {len(frac)} fractional y entries")
            if frac:
                top = sorted(frac, key=lambda x: abs(0.5 - x[1]))[:5]
                pretty = ", ".join(f"y[{i},{t}]={v:.3f}" for ((i, t), v, _) in top)
                log(f"[INFO] top-close-to-0.5: {pretty}")

        return best_key, y

    # ---- Solve a restricted IP to get integer λ on current columns
    def solve_restricted_ip(self, timelimit: float = 10.0):
        ip = grb.Model("RMP_IP", env=GRB_ENV)
        ip.Params.LogToConsole = 0
        ip.Params.OutputFlag = 0
        ip.Params.Threads = N_MASTER_THREADS

        lam_ip = {i: [] for i in self.items}
        for i in self.items:
            for p, pat in enumerate(self.master.patterns[i]):
                v = ip.addVar(
                    vtype=grb.GRB.BINARY, name=f"lam_{i}_{p}", obj=pat["cost"]
                )
                try:
                    v.Start = self.master.lambda_vars[i][p].X
                except Exception:
                    pass
                lam_ip[i].append(v)

        for i in self.items:
            ip.addConstr(grb.quicksum(lam_ip[i]) == 1, name=f"sel_{i}")

        for t in range(self.T):
            ip.addConstr(
                grb.quicksum(
                    pat["q"][t] * lam_ip[i][p]
                    for i in self.items
                    for p, pat in enumerate(self.master.patterns[i])
                    if pat["q"][t] > 0
                )
                <= self.cap[t],
                name=f"cap_{t}",
            )

        ip.ModelSense = grb.GRB.MINIMIZE
        ip.Params.TimeLimit = timelimit
        t0 = time.perf_counter()
        ip.optimize()
        t1 = time.perf_counter()

        if ip.Status in (grb.GRB.OPTIMAL, grb.GRB.TIME_LIMIT):
            lam_best = {i: [v.X for v in lam_ip[i]] for i in self.items}
            log(
                f"[INFO] restricted IP finished in {t1 - t0:.2f}s with obj={ip.ObjVal:.2f} status={ip.Status}"
            )
            return ip.ObjVal, lam_best
        log(f"[WARN] restricted IP status={ip.Status}")
        return float("inf"), None

    # ---- main recursion
    def branch_and_price(self, best=float("inf"), best_sol=None):
        bound, (pi, mu, lam) = self.column_generation()
        parent_id = self.parent_stack[-1] if self.parent_stack else None
        self.log_node(
            parent=parent_id,
            fix=None if not self.order_fix else list(self.order_fix.items())[-1],
            obj=bound,
            incumbent=best,
            status="branching" if bound < best else "pruned",
        )
        self._print_gap(bound, best)
        if bound >= best - RED_COST_EPS:
            return best, best_sol

        # RMP primal (lam) for branching decision
        res = self.master.optimize()
        if res is None:
            log("[FAIL] Final RMP infeasible.")
            return float("inf"), None
        _, _, mu_now, lam = res

        frac_key, y = self.choose_branch(lam, mu_now)
        if frac_key is None:
            # Try restricted IP to get integer λ on current column set
            ip_obj, lam_ip = self.solve_restricted_ip(timelimit=5.0)
            if lam_ip is not None and ip_obj <= bound + 1e-6:
                patterns_snapshot = deepcopy(self.master.patterns)
                log(f"[SOL] incumbent {ip_obj:.2f} (restricted IP)")
                self._print_gap(ip_obj, ip_obj, prefix="    ")
                return ip_obj, dict(
                    lam=lam_ip,
                    patterns=patterns_snapshot,
                    order_fix=dict(self.order_fix),
                )
            # fallback: accept LP λ as incumbent
            patterns_snapshot = deepcopy(self.master.patterns)
            log(f"[SOL] incumbent {bound:.2f} (LP y integral; λ may be fractional)")
            self._print_gap(bound, bound, prefix="    ")
            return bound, dict(
                lam=lam, patterns=patterns_snapshot, order_fix=dict(self.order_fix)
            )

        i_b, t_b = frac_key
        log(
            f"[BRANCH] choose y[{i_b},{t_b}]≈{y[(i_b,t_b)]:.3f}  (μ[{t_b}]={mu_now[t_b]:.3g})"
        )

        # Explore children: y=0 then y=1 (DFS)
        best, best_sol = self.branch_child(i_b, t_b, (0, 0), best, best_sol)
        best, best_sol = self.branch_child(i_b, t_b, (1, 1), best, best_sol)
        self.parent_stack.pop()
        return best, best_sol

    def branch_child(self, i, t, fix, best, best_sol):
        log(f"    |-- child: fix y[{i},{t}]={fix}")

        child = object.__new__(BranchPrice)
        # immutable data
        child.items = self.items
        child.dem = self.dem
        child.c_var = self.c_var
        child.h = self.h
        child.setup = self.setup
        child.b_var = self.b_var
        child.cap = self.cap
        child.T = self.T
        child.mseq = self.mseq
        child.k_max = self.k_max

        # per-node state
        child.prev_mu = list(self.prev_mu)
        child.alpha = self.alpha

        child.order_fix = dict(self.order_fix)
        child.order_fix[(i, t)] = fix

        # fresh RMP with existing columns and active branching rows (silent copy)
        child.master = self.master.copy(order_fix=child.order_fix)

        # share tree and id generator
        child.tree = self.tree
        child._idgen = self._idgen
        child.parent_stack = self.parent_stack.copy()

        return child.branch_and_price(best, best_sol)

    @staticmethod
    def _print_gap(bound: float, incumbent: float, prefix: str = ""):
        if incumbent < float("inf"):
            gap = 100.0 * (incumbent - bound) / max(1e-12, incumbent)
            log(
                f"{prefix}[GAP] bound={bound:.2f}  best={incumbent:.2f}  gap={gap:.2f}%"
            )
        else:
            log(f"{prefix}[GAP] bound={bound:.2f}  best=∞  gap=∞")


# =============================================================================
# Reporting & audits
# =============================================================================


def detailed_exec_rep(bp: BranchPrice, sol: Dict) -> None:
    log("\n=== Detailed execution report (per item) ===")
    lam = sol.get("lam", {})
    patterns = sol.get("patterns", bp.master.patterns)
    for i in bp.items:
        print(f"\nItem {i}")
        lam_i = lam.get(i, [])
        if not lam_i:
            print(f"[ERROR] No λ for item {i} — skipping.")
            continue
        sel_idx = max(range(len(lam_i)), key=lambda k: lam_i[k])
        if lam_i[sel_idx] < 0.5:
            print(f"[WARN] fractional λ for item {i} (best={lam_i[sel_idx]:.3f})")
        if sel_idx >= len(patterns[i]):
            print(f"[ERROR] Column index mismatch for item {i} — skipping.")
            continue
        pat = patterns[i][sel_idx]
        orders = pat["q"]
        yvec = pat.get("y", [1 if q > 0 else 0 for q in orders])

        # sanity: y[t]==1 ⇒ q[t]>0
        for t, bit in enumerate(yvec):
            if bit and orders[t] == 0:
                print(f"[BUG] y[{i},{t}]=1 but q=0 in selected pattern")

        D = bp.dem[i]
        m_seq = bp.mseq[i]
        cohorts: List[Tuple[int, int]] = []
        back = 0
        print(" t | dem | ord | inv+ | back")
        print("-" * 27)
        for t in range(bp.T):
            if orders[t] > 0:
                cohorts.append((m_seq[t], orders[t]))
            need = D[t] + back
            cohorts.sort(key=lambda x: x[0])  # FEFO
            new_coh = []
            for life, qty in cohorts:
                if need == 0:
                    new_coh.append((life, qty))
                    continue
                use = min(qty, need)
                qty -= use
                need -= use
                if qty > 0:
                    new_coh.append((life, qty))
            back = need  # no backorders allowed → positive means infeasible
            aged = []
            for life, qty in new_coh:
                life -= 1
                if life > 0 and qty > 0:
                    aged.append((life, qty))
            cohorts = aged
            inv_plus = sum(qty for _, qty in cohorts)
            print(f"{t:2d} | {D[t]:3d} | {orders[t]:3d} | {inv_plus:4d} | {back:4d}")


def capacity_audit(bp: BranchPrice, sol: Dict) -> None:
    print("\n=== Capacity audit (selected patterns) ===")
    lam = sol.get("lam", {})
    patterns = sol.get("patterns", bp.master.patterns)

    use = [0] * bp.T
    for i in bp.items:
        lam_i = lam.get(i, [])
        if not lam_i:
            continue
        sel_idx = max(range(len(lam_i)), key=lambda k: lam_i[k])
        q = patterns[i][sel_idx]["q"]
        for t in range(bp.T):
            use[t] += q[t]
    print(" t | used | cap | slack")
    print("------------------------")
    for t in range(bp.T):
        slack = bp.cap[t] - use[t]
        print(f"{t:2d} | {use[t]:4d} | {bp.cap[t]:3d} | {slack:5d}")


# =============================================================================
# Data classes & instance I/O
# =============================================================================


@dataclass
class Item:
    id: int
    demand: List[int]
    setup: float
    b_var: float
    c_var: float
    h: float
    shelf_seq: List[int]


@dataclass
class Lot:
    period: int
    capacity_pad: int
    items: Dict[int, Item] = field(default_factory=dict)
    manual_capacity: Optional[List[int]] = field(default=None)

    @property
    def capacity(self) -> List[int]:
        if self.manual_capacity is not None:
            return self.manual_capacity
        cap_raw = [
            sum(it.demand[t] for it in self.items.values()) for t in range(self.period)
        ]
        max_cap = max(cap_raw) if cap_raw else 0
        buffer = max(5, int(0.2 * max_cap))
        return [c + buffer for c in cap_raw]

    @property
    def kmax(self) -> Dict[int, int]:
        max_dem = max((max(it.demand) for it in self.items.values()), default=0)
        buffer = max(5, int(0.2 * max_dem))
        return {i: sum(it.demand) + buffer for i, it in self.items.items()}

    def to_dicts(self):
        demand = {i: it.demand for i, it in self.items.items()}
        setup = {i: it.setup for i, it in self.items.items()}
        b_var = {i: it.b_var for i, it in self.items.items()}
        c_var = {i: it.c_var for i, it in self.items.items()}
        h = {i: it.h for i, it in self.items.items()}
        mseq = {i: it.shelf_seq for i, it in self.items.items()}
        cap = self.capacity
        kmax = self.kmax
        return demand, c_var, h, setup, b_var, cap, mseq, kmax

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serial = asdict(self)
        serial["items"] = {str(k): v for k, v in serial["items"].items()}
        if self.manual_capacity is None:
            serial.pop("manual_capacity", None)
        path.write_text(json.dumps(serial, indent=indent))

    @classmethod
    def from_json(cls, path: str | Path) -> "Lot":
        data = json.loads(Path(path).read_text())
        items = {int(k): Item(**v) for k, v in data["items"].items()}
        return cls(
            period=data["period"],
            capacity_pad=data["capacity_pad"],
            items=items,
            manual_capacity=data.get("manual_capacity", None),
        )


# =============================================================================
# Tee + Save results
# =============================================================================


class Tee(io.TextIOBase):
    """Write to multiple text streams (e.g., terminal + capture buffer)."""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, s):
        for st in self.streams:
            st.write(s)
            try:
                st.flush()
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def save_results(
    lot: "Lot",
    bp: "BranchPrice",
    sol: Dict,
    objective: float,
    elapsed_sec: float,
    stdout_text: str,
    instance_path: Path,
    outdir: str | Path = "results",
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    instance_copy = outdir / f"instance_{ts}.json"
    try:
        instance_copy.write_text(Path(instance_path).read_text())
    except Exception:
        pass

    demand, c_var, h, setup, b_var, cap, mseq, kmax = lot.to_dicts()

    patterns_src = sol.get("patterns", bp.master.patterns)
    patterns = {
        i: [
            dict(
                cost=p["cost"],
                y=p.get("y", [1 if q > 0 else 0 for q in p["q"]]),
                q=p["q"],
            )
            for p in patterns_src[i]
        ]
        for i in bp.items
    }
    lam = sol.get("lam", {})

    tree_list = list(bp.tree.values())
    order_fix = {f"{i}_{t}": list(bounds) for (i, t), bounds in bp.order_fix.items()}

    run_json = {
        "timestamp": ts,
        "objective": objective,
        "elapsed_seconds": elapsed_sec,
        "allow_backorder": ALLOW_BACKORDER,
        "dual_stabilization_alpha": getattr(bp, "alpha", None),
        "instance_path": str(instance_path),
        "capacity": cap,
        "items": {
            str(i): {
                "setup": setup[i],
                "b_var": b_var[i],
                "c_var": c_var[i],
                "h": h[i],
                "demand": demand[i],
                "perishability_seq": mseq[i],
                "kmax": kmax[i],
            }
            for i in bp.items
        },
        "patterns": patterns,
        "lambda_solution": lam,
        "order_fix": order_fix,
        "tree": tree_list,
    }
    (outdir / f"run_{ts}.json").write_text(json.dumps(run_json, indent=2))

    txt_path = outdir / f"run_{ts}.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=== Multi-Item Perishable Lot-Sizing — Run Report ===\n")
        f.write(f"Timestamp           : {ts}\n")
        f.write(f"Objective           : {objective:.2f}\n")
        f.write(f"Elapsed (seconds)   : {elapsed_sec:.2f}\n")
        f.write(f"Instance JSON       : {instance_path}\n")
        f.write(f"Instance copy       : {instance_copy}\n")
        f.write(f"Allow backorder     : {ALLOW_BACKORDER}\n")
        f.write(f"Items               : {len(bp.items)}\n")
        f.write(f"Periods             : {bp.T}\n")
        f.write("\n-- Capacity --\n")
        f.write(", ".join(map(str, cap)) + "\n")
        f.write("\n-- Per-item costs & perishability ranges --\n")
        for i in bp.items:
            mseq_i = mseq[i]
            f.write(
                f"Item {i}: setup={setup[i]}, c={c_var[i]}, h={h[i]}, b={b_var[i]}, "
                f"min(m_it)={min(mseq_i)}, max(m_it)={max(mseq_i)}\n"
            )

        f.write("\n-- Selected patterns (λ>0) --\n")
        for i in bp.items:
            lam_i = lam.get(i, [])
            if not lam_i:
                f.write(f"Item {i}: no λ selected.\n")
                continue
            sel = max(range(len(lam_i)), key=lambda k: lam_i[k])
            pat = patterns_src[i][sel]
            f.write(
                f"Item {i}: pattern {sel}, cost={pat['cost']:.2f}, "
                f"y={pat.get('y')}, q={pat['q']}\n"
            )

        f.write("\n-- Branch-and-Price Tree (nodes) --\n")
        for n in tree_list:
            f.write(json.dumps(n) + "\n")

        f.write("\n\n=== Terminal Output (verbatim) ===\n")
        f.write(stdout_text)

    return txt_path


# =============================================================================
# Instance generation
# =============================================================================


def build_lot(
    period: int,
    lb_dem: int,
    ub_dem: int,
    capacity_pad: int,
    specs: List[tuple],
    manual_capacity: Optional[List[int]],
    default_shelf_rng: Tuple[int, int] = (3, 5),
) -> Lot:
    random.seed(SEED)
    lot = Lot(period=period, capacity_pad=capacity_pad, manual_capacity=manual_capacity)
    for rec in specs:
        # spec can be (idx, setup, b, c, h) or (idx, setup, b, c, h, (m_lo, m_hi))
        if len(rec) == 5:
            idx, stp, b, c, hold = rec
            shelf_rng = default_shelf_rng
        elif len(rec) == 6:
            idx, stp, b, c, hold, shelf_rng = rec
            if not (isinstance(shelf_rng, tuple) and len(shelf_rng) == 2):
                raise ValueError("specs[5] must be (m_lo, m_hi)")
        else:
            raise ValueError(
                "spec must be (idx, stp, b, c, h) or (idx, stp, b, c, h, (m_lo,m_hi))"
            )
        demand = [random.randint(lb_dem, ub_dem) for _ in range(period)]
        shelf_seq = [random.randint(shelf_rng[0], shelf_rng[1]) for _ in range(period)]
        lot.items[idx] = Item(idx, demand, stp, b, c, hold, shelf_seq)
    return lot


# =============================================================================
# Main runner with tee + audits
# =============================================================================


def run_and_save(lot: "Lot") -> None:
    """Runs BP+CG while teeing stdout to both terminal and a buffer, then saves."""
    buf = io.StringIO()
    tee = Tee(sys.stdout, buf)

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(tee):
        demand, c_var, h, setup, b_var, cap, mseq, k_max = lot.to_dicts()
        bp = BranchPrice(demand, c_var, h, setup, b_var, cap, mseq, k_max)
        log(
            "[INFO] Running RMP (dual simplex) & Pricing (DP→SP→MIP), micro-rc filter on"
        )
        best, sol = bp.branch_and_price()
        if sol is None:
            print("[❌ ERROR] No feasible solution — check capacity vs demand.")
            for t in range(lot.period):
                total_d = sum(it.demand[t] for it in lot.items.values())
                print(
                    f"Period {t:2d} → Total demand: {total_d}, Capacity: {lot.capacity[t]}"
                )
        else:
            elapsed = time.perf_counter() - t0
            print(f"\nObjective: {best:.2f}   (elapsed {elapsed:.2f} s)\n")
            root_nodes = [n for n in bp.tree.values() if n["parent"] is None]
            root_bound = root_nodes[0]["obj"] if root_nodes else best
            BranchPrice._print_gap(root_bound, best, prefix="[FINAL] ")
            print(f"[INFO] Backorders allowed: {ALLOW_BACKORDER}")

            lam = sol.get("lam", {})
            patterns_src = sol.get("patterns", bp.master.patterns)
            for i in bp.items:
                lam_i = lam.get(i, [])
                if not lam_i:
                    print(f"[WARN] No valid pattern for item {i}.")
                    continue
                sel = max(range(len(lam_i)), key=lambda k: lam_i[k])
                print(
                    f"Item {i}: pattern {sel}, y={patterns_src[i][sel].get('y')}, q={patterns_src[i][sel]['q']}"
                )

            detailed_exec_rep(bp, sol)
            capacity_audit(bp, sol)
            if TREE_VISUALIZATION and plt and nx:
                visualize_bnp_tree(bp.tree)
                print(f"[INFO] Branch-and-Price tree visualized.")
            if plt:
                plt.ioff()
                plt.show()
            print(f"\nObjective: {best:.2f}   (elapsed {elapsed:.2f} s)\n")

    stdout_text = buf.getvalue()
    elapsed = time.perf_counter() - t0
    if sol is None:
        best = float("inf")
        sol = {}
    out_txt = save_results(
        lot=lot,
        bp=bp,
        sol=sol,
        objective=best,
        elapsed_sec=elapsed,
        stdout_text=stdout_text,
        instance_path=INSTANCE_PATH,
        outdir="results",
    )
    print(f"[RESULTS] saved to: {out_txt.parent}  (main report: {out_txt.name})")


# =============================================================================
# Optional: tree viz (only if users enable)
# =============================================================================


def visualize_bnp_tree(tree: list):  # pragma: no cover
    if not (plt and nx):
        print("[WARN] Visualization requires matplotlib and networkx.")
        return
    G = nx.DiGraph()
    for node in tree:
        node_id = node["id"]
        fix = node.get("fix")
        obj = node.get("obj", float("inf"))
        inc = node.get("incumbent", float("inf"))
        gap = (100 * (inc - obj) / inc) if inc < float("inf") else float("inf")
        label = f"{node_id}\n"
        if fix:
            label += f"{fix[0]}={fix[1]}\n"
        label += f"obj={obj:.1f}\ninc={inc if inc < float('inf') else 'inf'}\n"
        if inc < float("inf"):
            label += f"gap={gap:.1f}%"
        G.add_node(node_id, label=label)
        parent_id = node.get("parent")
        if parent_id is not None:
            G.add_edge(parent_id, node_id)
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")
    except Exception:
        print("[WARN] Falling back to spring layout.")
        pos = nx.spring_layout(G, seed=42)
    node_labels = nx.get_node_attributes(G, "label")
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=node_labels,
        node_color="lightblue",
        edge_color="gray",
        node_size=3000,
        font_size=8,
        font_weight="bold",
    )
    plt.title("Branch-and-Price Tree")
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    if (not RANDOMIZE) and INSTANCE_PATH.exists():
        lot = Lot.from_json(INSTANCE_PATH)
        print(f"[INFO] Loaded instance from {INSTANCE_PATH}")
    else:
        # specs: (item_id, setup, b, c, h, (m_min, m_max))
        specs = [
            (0, 127.5, 5.0, 2.0, 0.4, (1, 50)),
            (1, 29.0, 5.0, 3.0, 0.6, (1, 60)),
            (2, 50.0, 5.0, 1.0, 0.3, (3, 70)),
            (3, 75.0, 5.0, 4.0, 0.5, (3, 5)),
            (4, 100.0, 5.0, 2.5, 0.4, (3, 5)),
        ]
        period = 80
        manual_caps = [1300] * period if USE_MANUAL_CAPACITY else None
        lot = build_lot(
            period=period,
            lb_dem=1,
            ub_dem=200,
            capacity_pad=10,
            specs=[s[:5] for s in specs],
            manual_capacity=manual_caps,
            default_shelf_rng=(3, 5),
        )
        # Overwrite each item's perishability with its per-item range
        for (i, *_rest, rng) in specs:
            lo, hi = rng
            lot.items[i].shelf_seq = [random.randint(lo, hi) for _ in range(period)]
        lot.to_json(INSTANCE_PATH)
        print(f"[INFO] Generated new instance → {INSTANCE_PATH}")

    run_and_save(lot)
