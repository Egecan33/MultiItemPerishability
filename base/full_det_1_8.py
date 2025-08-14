"""
Branch-and-price for multi-item perishable lot-sizing with setup/holding costs,
NO backorders, dual stabilization, and branching on (item, period) order decisions.

This revision (tolerance-enabled):
  • Backorders OFF.
  • Pricing per item (single-item subproblem):
        (1) FEFO Dynamic Program (Önal et al., 2015) — primary
        (2) Shortest-Path with variable shelf-life — fallback
        (3) MIP pricing (perishability-aware) — last resort (especially if branching constraints needed)
  • DP enforces FEFO and variable shelf-life; accounts for reduced costs.
  • Branch-and-Price with order/no-order branching on y_{i,t} (whether item i orders in period t).
  • Early stopping: if global optimality gap falls below 0.1%, terminate search.
  • Column Generation early-stop if objective improves < 5e-5 over 12 iterations.
  • Micro reduced-cost filter (|rc|<1e-3) to avoid minor improvements causing churn.
  • Removed live plotting/visualization for simplicity; detailed textual output and result saving.
"""

from __future__ import annotations

# stdlib imports
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

# third-party imports
import numpy as np
import gurobipy as grb

# =============================================================================
# Config
# =============================================================================
RANDOMIZE = True
USE_MANUAL_CAPACITY = True
ALLOW_BACKORDER = False  # keep OFF (no backorders)
SEED = 0
INSTANCE_PATH = Path(__file__).with_name("last_instance.json")

VERBOSE = True
LOG_EVERY_ITER = 1
CG_MAX_ITERS = 100

# Column generation early-stop (stagnation criteria)
CG_ESTOP_MIN_ITERS = 12
CG_ESTOP_PATIENCE = 12
CG_ESTOP_MIN_REL_IMPROVE = 5e-5

# Stopping criteria for branch-and-price search (optimality gap threshold)
BAP_GAP_TOL = 1e-3  # 0.1% relative gap

# Reduced cost threshold (treat very small improvements as zero)
RED_COST_EPS = 1e-3

# Debug toggles
DEBUG_DUALS = False
DEBUG_PRICING = False
DEBUG_BRANCH = True
DEBUG_SUMMARY_WIDTH = 8

# Threads
N_MASTER_THREADS = max(1, (os.cpu_count() or 4) // 2)
PRICING_N_THREADS = max(
    1, (os.cpu_count() or 4) // 2
)  # (not used if sequential pricing)

# MIP pricing time limit
MIP_PRICE_TIMELIMIT_SEC = 5.0

# Dual stabilization factor for smoothing dual prices
STAB_ALPHA = 0.9


# =============================================================================
# Quiet Gurobi environment setup
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
    total_cost: (
        Number  # total (reduced) cost for pattern (includes dual-adjusted costs)
    )
    x_ti: List[
        List[Number]
    ]  # allocation matrix x[t][i]: quantity from order at t used in period i
    x_t: List[Number]  # order quantity at t (sum of x_ti over i)
    y_t: List[int]  # setup indicator at t (1 if order placed at t, 0 otherwise)


def _prefix_sums(a: List[Number]) -> List[Number]:
    ps = [0.0]
    s = 0.0
    for v in a:
        s += v
        ps.append(s)
    return ps


def _compute_c_ti(p: List[Number], h: List[Number]) -> List[List[Number]]:
    """Precompute cost to have an order at t satisfy demand up to i (inclusive): c[t][i] = p[t] + sum_{j=t}^{i-1} h_j for i >= t, else inf."""
    T = len(p)
    H = _prefix_sums(h)  # cumulative holding cost coefficients
    c = [[math.inf] * T for _ in range(T)]
    for t in range(T):
        for i in range(t, T):
            c[t][i] = p[t] + (H[i] - H[t])
    return c


def dp_fefo_solve(
    D: List[Number],
    p_eff: List[Number],  # effective unit cost in each period (c_var minus dual)
    h_seq: List[Number],  # holding cost per period
    S_seq: List[Number],  # setup cost per period
    v_last: List[int],  # last period an order in t can serve (inclusive)
    forbid_t: Optional[
        set
    ] = None,  # set of periods where no order is allowed (for branching y=0)
) -> DPResult:
    """
    FEFO Dynamic Program for single-item lot-sizing (no backlog).
    - D[i] is demand in period i.
    - p_eff[t] = (unit production cost c_var minus dual mu_t) for period t.
    - h_seq[t] = holding cost per unit from period t to t+1 (we assume constant per period or given sequence).
    - S_seq[t] = setup cost if an order is placed at period t.
    - v_last[t] = last period index that an order at t can serve (due to perishability/shelf-life).
    - forbid_t: optional set of periods forbidden to have an order (enforced by setting S_t = +inf for those t).
    Returns DPResult with minimum cost (taking into account dual adjustments) and corresponding order plan.
    """
    T = len(D)
    assert len(p_eff) == T and len(h_seq) == T and len(S_seq) == T and len(v_last) == T

    # Adjust setup costs for forbidden periods by setting them to infinity (to avoid ordering at those periods).
    S_eff = [(math.inf if forbid_t and t in forbid_t else S_seq[t]) for t in range(T)]

    # Precompute cost contributions c[t][i] for an order at t covering demand up to i.
    c = _compute_c_ti(p_eff, h_seq)

    inf = math.inf
    # DP state values f and g as described in Onal et al. (2015):
    # f_t[(t, τ1, τ2)] = cost of fulfilling demand interval [τ1, τ2] with a single order at t (t <= τ1).
    # g_t[(t, τ1, τ2)] = cost of fulfilling [τ1, τ2] with an order at t for some sub-interval and possibly another order in [τ1, τ2].
    f_t: Dict[Tuple[int, int, int], Number] = {}
    g_t: Dict[Tuple[int, int, int], Number] = {}
    # f_min[τ1][τ2] = min_t f_t[(t, τ1, τ2)] (minimum cost to cover [τ1, τ2] with one order starting at some t <= τ1)
    # g_min[τ1][τ2] = minimum cost to cover [τ1, τ2] with one or more orders (optimal cost for that subproblem).
    f_min = [[inf] * T for _ in range(T)]
    g_min = [[inf] * T for _ in range(T)]
    # tstar[τ1, τ2] = argmin_t f_t[(t, τ1, τ2)] (best order start for covering [τ1, τ2] in one order)
    # split_g[τ1, τ2] = optimal split point τ* in [τ1, τ2] for multiple orders (for g_min).
    tstar: Dict[Tuple[int, int], int] = {}
    split_g: Dict[Tuple[int, int], int] = {}

    def gmin(a: int, b: int) -> Number:
        # cost to cover interval [a, b] optimally (g_min), with boundary check
        return 0.0 if a > b else g_min[a][b]

    # Base case: intervals of length 0 (single period τ)
    for τ in range(T):
        # For each possible start t <= τ:
        for t in range(τ + 1):
            # If an order at t can cover τ (i.e., τ <= v_last[t]) and order at t is allowed:
            if v_last[t] < τ or math.isinf(S_eff[t]):
                cost_val = inf
            else:
                # cost of one order at t covering demand at τ (only that period)
                cost_val = S_eff[t] + c[t][τ] * D[τ]
            f_t[(t, τ, τ)] = cost_val
            g_t[(t, τ, τ)] = cost_val
        # Compute f_min and g_min for interval [τ, τ]
        best_cost = inf
        best_t = -1
        for t in range(τ + 1):
            if f_t[(t, τ, τ)] < best_cost:
                best_cost = f_t[(t, τ, τ)]
                best_t = t
        f_min[τ][τ] = best_cost
        g_min[τ][τ] = best_cost
        tstar[(τ, τ)] = best_t
        split_g[(τ, τ)] = τ

    # DP recursion for intervals of length > 0
    for length in range(1, T):
        for τ1 in range(0, T - length):
            τ2 = τ1 + length
            # Compute f_t and g_t for covering [τ1, τ2] with orders
            for t in range(0, τ1 + 1):
                # If an order at t can serve up to τ2 (i.e., τ2 <= v_last[t]) and ordering at t is allowed:
                if v_last[t] < τ2 or math.isinf(S_eff[t]):
                    f_cost = inf
                    g_cost = inf
                else:
                    # Case 1: use one order at t to cover [τ1, τ2] entirely
                    prev = g_t.get((t, τ1, τ2 - 1), inf)
                    f_cost = (
                        prev + c[t][τ2] * D[τ2]
                    )  # extend coverage of order at t to τ2
                    # Case 2: multiple orders: try splitting at some τ in [τ1, τ2]
                    g_cost = inf
                    for τ in range(τ1, τ2 + 1):
                        f_part = f_t.get((t, τ1, τ), inf)
                        # cost = cost of covering [τ1, τ] with one order at t plus optimal cost for [τ+1, τ2]
                        val = f_part + gmin(τ + 1, τ2)
                        if val < g_cost:
                            g_cost = val
                f_t[(t, τ1, τ2)] = f_cost
                g_t[(t, τ1, τ2)] = g_cost

            # Compute f_min for [τ1, τ2] (best single-order solution)
            best_f = inf
            best_t_f = -1
            for t in range(0, τ1 + 1):
                val = f_t[(t, τ1, τ2)]
                if val < best_f:
                    best_f = val
                    best_t_f = t
            f_min[τ1][τ2] = best_f
            tstar[(τ1, τ2)] = best_t_f

            # Compute g_min for [τ1, τ2] (best multi-order solution)
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
    # Reconstruct the order plan (y and x allocation) from DP results
    x_ti = [[0.0] * T for _ in range(T)]
    x_t = [0.0] * T
    y_t = [0] * T

    def reconstruct_interval(a: int, b: int):
        if a > b:
            return
        τ_split = split_g[(a, b)]
        t_best = tstar[(a, τ_split)]
        if t_best == -1 or math.isinf(f_min[a][τ_split]):
            raise RuntimeError(
                f"Infeasible subplan reconstruction for interval [{a},{τ_split}]."
            )
        # Assign all demand from interval [a, τ_split] to order at t_best
        for i in range(a, τ_split + 1):
            x_ti[t_best][i] = D[i]
            x_t[t_best] += D[i]
        y_t[t_best] = 1
        # Recursively reconstruct remaining interval [τ_split+1, b]
        reconstruct_interval(τ_split + 1, b)

    if T > 0:
        reconstruct_interval(0, T - 1)
    return DPResult(total_cost=total_cost, x_ti=x_ti, x_t=x_t, y_t=y_t)


def build_fefo_flow_from_q(
    demand: List[int], q: List[int], m_seq: List[int]
) -> Dict[Tuple[int, int], float]:
    """
    Given a production plan q (orders in each period) for one item, build a FEFO allocation flow x[(s,t)].
    Ensures first-expiry-first-out: products from earlier orders (with shorter remaining shelf life) are used first to meet demand.
    """
    T = len(demand)
    x_flow: Dict[Tuple[int, int], float] = {}
    # list of cohorts as (expiry_time, remaining_qty, start_period)
    cohorts: List[Tuple[int, float, int]] = []
    for s in range(T):
        qty = q[s]
        if qty > 0:
            exp = min(T - 1, s + m_seq[s] - 1)  # compute expiry period for order at s
            cohorts.append((exp, float(qty), s))
    # Serve demand period by period
    for t in range(T):
        need = float(demand[t])
        # sort cohorts by expiry (then by start for stability)
        cohorts.sort(key=lambda c: (c[0], c[2]))
        new_cohorts: List[Tuple[int, float, int]] = []
        for exp, rem, s in cohorts:
            if need <= 1e-12:
                new_cohorts.append((exp, rem, s))
                continue
            if t > exp or rem <= 0:
                # cohort expired or empty -> skip (expired units are wasted)
                continue
            # use as much as possible from this cohort to meet current demand
            use_qty = min(rem, need)
            if use_qty > 0:
                x_flow[(s, t)] = x_flow.get((s, t), 0.0) + use_qty
                rem -= use_qty
                need -= use_qty
            if rem > 1e-12:
                new_cohorts.append((exp, rem, s))
        cohorts = [
            (exp - 1, qty, s) for (exp, qty, s) in new_cohorts if exp > t
        ]  # decrease remaining shelf life for leftover batches
        # If need > 0 here, demand is not met (which should not happen if q covers all D without backlog).
    return x_flow


# =============================================================================
# Shortest Path pricing (alternate subproblem solver, no backorders)
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
    Solve single-item pricing via shortest-path in a layered DAG:
    Nodes represent time periods, arcs represent placing an order at time t covering demand from t to u.
    We impose that each pattern covers contiguous segments of demand without gaps (no backlogs).
    We do not allow "skip" at t (if no order at t, that's handled by arcs from earlier nodes).
    Branching constraints (order_fix) for y=0 at t mean we disallow arcs that start at t.
    Returns (reduced_cost, true_cost, q_plan) if found improving column, or (inf, inf, None) if none.
    """
    T = len(demand)
    # distance (cost) to reach node j from node T (which acts as sink)
    dist = [float("inf")] * (T + 1)
    next_arc: List[Optional[Tuple[int, int]]] = [None] * (T + 1)
    dist[T] = -pi  # reaching end with pi offset for selection constraint
    forbidden_starts = {
        tt for (ii, tt), (lb, ub) in order_fix.items() if ii == item_id and ub == 0
    }
    # Process periods backwards (dynamic programming on DAG)
    for t in range(T - 1, -1, -1):
        best_cost = float("inf")
        best_arc: Optional[Tuple[int, int]] = None
        if t not in forbidden_starts:
            q_acc = 0
            hold_acc = 0
            # limit on how far this order can go due to shelf life
            u_max = T - 1
            if m_seq is not None:
                u_max = min(u_max, t + m_seq[t] - 1)
            for u in range(t, u_max + 1):
                q_acc += demand[u]
                if q_acc > k_max:
                    break  # capacity (k_max) of single order exceeded
                hold_acc += demand[u] * (u - t)
                # reduced cost for an arc (order at t serving demand through u)
                arc_cost = (
                    (c_var - mu[t]) * q_acc + h * hold_acc + (setup if q_acc > 0 else 0)
                )
                candidate = arc_cost + dist[u + 1]
                if candidate < best_cost:
                    best_cost = candidate
                    best_arc = (u, q_acc)
        dist[t] = best_cost
        next_arc[t] = best_arc

    if math.isinf(dist[0]) or dist[0] >= -RED_COST_EPS:
        return float("inf"), float("inf"), None
    # Reconstruct q plan from next_arc
    q_plan = [0] * T
    true_cost = 0.0
    t = 0
    while t < T:
        arc = next_arc[t]
        if arc is None:
            # Should not happen if we found dist[0] < inf
            break
        u, qty = arc
        q_plan[t] = qty
        # True (actual) cost calculation (without dual adjustments)
        hold_cost = sum(demand[tau] * (tau - t) for tau in range(t, u + 1))
        true_cost += (setup if qty > 0 else 0) + c_var * qty + h * hold_cost
        t = u + 1
    if sum(q_plan) == 0:
        return float("inf"), float("inf"), None
    return dist[0], true_cost, q_plan


# =============================================================================
# Column Generation early-stopping guard
# =============================================================================
class ConvergenceGuard:
    def __init__(self, min_iters: int, patience: int, min_rel_improve: float):
        self.min_iters = min_iters
        self.patience = patience
        self.min_rel = min_rel_improve
        self.prev_obj = None
        self.stale_count = 0

    def step(self, current_obj: float) -> bool:
        """Update with current objective value. Return True if stagnation criteria met (should stop)."""
        if self.prev_obj is not None:
            # relative improvement (since minimizing)
            denom = max(1.0, abs(self.prev_obj))
            rel_improve = (self.prev_obj - current_obj) / denom
            if rel_improve < self.min_rel:
                self.stale_count += 1
            else:
                self.stale_count = 0
        self.prev_obj = current_obj
        if (
            self.prev_obj is not None
            and self.stale_count >= self.patience
            and self.prev_obj != float("inf")
        ):
            return True
        return False


# =============================================================================
# Master Problem (Restricted Master) Representation
# =============================================================================
class MasterModel:
    """
    Restricted Master Problem (RMP) for multi-item formulation.
    Decision variables:
      λ_{i,p} >= 0 for each pattern p of item i (pattern covers all demands of item i).
    Constraints:
      - Selection: sum_p λ_{i,p} = 1  for each item i (each item exactly one pattern).
      - Capacity:  sum_i sum_p q_{i,p,t} * λ_{i,p} <= cap_t  for each period t (capacity per period).
      - Branching (order fix): for each fixed y_{i,t}=0 or 1, enforce that constraint via additional row.
            * If y_{i,t}=1 (must order), then sum_p λ_{i,p} * I(p has order at t) >= 1 (i must pick a pattern with an order at t).
            * If y_{i,t}=0 (forbidden), then sum_p λ_{i,p} * I(p has order at t) = 0 (no pattern chosen can have an order at t).
    """

    def __init__(
        self,
        items: List[int],
        T: int,
        capacity: List[int],
        env: grb.Env,
        verbose: bool = False,
    ):
        self.items = items
        self.T = T
        self.capacity = capacity
        self.model = grb.Model("RMP", env=env)
        self.model.Params.OutputFlag = 1 if verbose else 0
        self.model.Params.Method = 1  # dual simplex for reoptimization
        self.model.Params.Threads = N_MASTER_THREADS
        # variables and structures to store patterns
        self.lambda_vars: Dict[int, List[grb.Var]] = {i: [] for i in items}
        self.patterns: Dict[int, List[Dict]] = {i: [] for i in items}
        # selection constraints for each item
        self.sel_constr = {
            i: self.model.addConstr(grb.LinExpr() == 1.0, name=f"sel_{i}")
            for i in items
        }
        # capacity constraints for each period
        self.cap_constr = [
            self.model.addConstr(grb.LinExpr() <= capacity[t], name=f"cap_{t}")
            for t in range(T)
        ]
        # branching order constraints (added on the fly)
        self.order_constr: Dict[Tuple[int, int], grb.Constr] = {}

    def _ensure_order_constraint(self, ii: int, tt: int, lb: int, ub: int) -> None:
        """
        Ensure a branching constraint for (item ii, period tt) is present in the model:
        If lb==ub==1 (force order), constraint: sum_{patterns p with y_{p,tt}=1} λ_{i,p} >= 1.
        If lb==ub==0 (forbid order), constraint: sum_{p with y_{p,tt}=1} λ_{i,p} = 0.
        """
        key = (ii, tt)
        if key in self.order_constr:
            return  # already added
        expr = grb.LinExpr()
        if lb == 1 and ub == 1:
            # must have an order at t: sum_p I(y_p_t=1) * λ_{i,p} >= 1
            constr = self.model.addConstr(expr >= 1.0, name=f"branch_y1_{ii}_{tt}")
        elif lb == 0 and ub == 0:
            # no order at t allowed: sum_p I(y_p_t=1) * λ_{i,p} <= 0
            constr = self.model.addConstr(expr <= 0.0, name=f"branch_y0_{ii}_{tt}")
        else:
            return
        self.order_constr[key] = constr

    def add_pattern(self, i: int, cost: float, q: List[int], y: List[int]) -> None:
        """
        Add a new pattern (order plan) for item i to the RMP.
        cost: objective coefficient (actual cost of pattern)
        q: list of order quantities in each period (length T)
        y: list of binary indicators for ordering in each period (length T)
        """
        if sum(q) == 0:
            log(f"[WARN] Ignoring zero-quantity pattern for item {i}")
            return
        # avoid adding duplicate patterns
        for pat in self.patterns[i]:
            if pat["q"] == q:
                return
        # Build column for the new pattern
        col = grb.Column()
        col.addTerms(1.0, self.sel_constr[i])  # contribute to selection constraint
        for t, qty in enumerate(q):
            if qty != 0:
                col.addTerms(qty, self.cap_constr[t])  # capacity usage
        # account for branching constraints: if pattern has order at (ii,tt) that is constrained
        for (ii, tt), constr in self.order_constr.items():
            if ii == i and y[tt] == 1:
                col.addTerms(1.0, constr)
        # Add new variable λ with given cost (objective coefficient)
        var = self.model.addVar(
            obj=cost, column=col, name=f"lambda_{i}_{len(self.lambda_vars[i])}"
        )
        self.lambda_vars[i].append(var)
        self.patterns[i].append({"cost": cost, "q": q, "y": y})
        log(f"[ADD] item {i} pattern#{len(self.lambda_vars[i])-1} cost={cost:.2f}")

    def optimize(
        self,
    ) -> Optional[Tuple[float, Dict[int, float], List[float], Dict[int, List[float]]]]:
        """
        Solve the current RMP LP relaxation to optimality.
        Returns tuple (obj_value, pi, mu, lambda_values) if solved, or None if infeasible.
         - pi[i] is dual for item i selection constraint.
         - mu[t] is dual for capacity constraint at period t.
         - lambda_values[i] is list of solution values for λ_{i,p} variables.
        """
        self.model.optimize()
        status = self.model.Status
        if status == grb.GRB.INFEASIBLE:
            return None
        if status != grb.GRB.OPTIMAL:
            raise RuntimeError(
                f"Unexpected Gurobi status {status} in RMP optimization."
            )
        obj_val = self.model.ObjVal
        pi = {i: self.sel_constr[i].Pi for i in self.items}
        mu = [c.Pi for c in self.cap_constr]
        lambda_vals = {i: [var.X for var in self.lambda_vars[i]] for i in self.items}
        return obj_val, pi, mu, lambda_vals

    def clone_with_branching(
        self, order_fix: Dict[Tuple[int, int], Tuple[int, int]]
    ) -> MasterModel:
        """
        Create a fresh MasterModel copy with the same columns (patterns) but including branching constraints.
        This is used when branching to a child node: constraints for fixed or forbidden orders are added.
        """
        new_master = MasterModel(
            self.items, self.T, self.capacity, env=GRB_ENV, verbose=False
        )
        # add existing patterns (silently, without logging [ADD] messages)
        for i in self.items:
            for pat in self.patterns[i]:
                # reuse pattern info; add without duplicate checking or logging
                q = pat["q"]
                y = pat["y"]
                cost = pat["cost"]
                col = grb.Column()
                col.addTerms(1.0, new_master.sel_constr[i])
                for t, qty in enumerate(q):
                    if qty != 0:
                        col.addTerms(qty, new_master.cap_constr[t])
                for (ii, tt), constr in new_master.order_constr.items():
                    if ii == i and y[tt] == 1:
                        col.addTerms(1.0, constr)
                var = new_master.model.addVar(
                    obj=cost,
                    column=col,
                    name=f"lambda_{i}_{len(new_master.lambda_vars[i])}",
                )
                new_master.lambda_vars[i].append(var)
                new_master.patterns[i].append({"cost": cost, "q": q, "y": y})
        # apply branching order constraints for this node
        for (ii, tt), (lb, ub) in order_fix.items():
            if lb == ub:  # actual fixed condition (0 or 1)
                new_master._ensure_order_constraint(ii, tt, lb, ub)
                # update coefficients of existing patterns for this item i in new constraint
                constr = new_master.order_constr[(ii, tt)]
                for idx, pat in enumerate(new_master.patterns[ii]):
                    if pat["y"][tt] == 1:
                        var = new_master.lambda_vars[ii][idx]
                        new_master.model.chgCoeff(constr, var, 1.0)
        new_master.model.update()
        return new_master


# =============================================================================
# Branch-and-Price Solver
# =============================================================================
class BranchPrice:
    def __init__(
        self,
        demand: Dict[int, List[int]],
        c_var: Dict[int, float],
        h: Dict[int, float],
        setup: Dict[int, float],
        b_var: Dict[int, float],
        capacity: List[int],
        shelf_seq: Dict[int, List[int]],
        k_max: Dict[int, int],
    ):
        self.items = list(demand.keys())
        self.dem = demand
        self.c_var = c_var
        self.h = h
        self.setup = setup
        self.b_var = b_var  # backorder costs (not used since ALLOW_BACKORDER=False)
        self.cap = capacity
        self.T = len(capacity)
        self.mseq = shelf_seq
        self.k_max = k_max

        self.order_fix: Dict[Tuple[int, int], Tuple[int, int]] = (
            {}
        )  # branch fixes for y (lb, ub)
        # Keep previous duals for stabilization
        self.prev_mu = [0.0] * self.T
        self.alpha = STAB_ALPHA

        # Initialize master model with initial patterns
        self.master = MasterModel(
            self.items, self.T, self.cap, env=GRB_ENV, verbose=False
        )
        # Seed initial patterns: produce exactly to meet demand (one order at each period where demand occurs)
        for i in self.items:
            # trivial pattern: order exactly each period's demand at that period (i.e., no holding, immediate consumption)
            q_pattern = self.dem[i][:]
            # cost calculation for that pattern
            total_cost = sum(self.c_var[i] * q for q in q_pattern) + self.setup[
                i
            ] * sum(1 for q in q_pattern if q > 0)
            y_pattern = [1 if qty > 0 else 0 for qty in q_pattern]
            self.master.add_pattern(i, total_cost, q_pattern, y_pattern)
        self.master.model.update()

        # Branch-and-bound tree tracking
        self.tree: Dict[int, Dict] = {}
        self._node_id_counter = itertools.count()
        self.parent_stack: List[Optional[int]] = []
        log(
            f"[INFO] Init Branch-and-Price: items={len(self.items)}, T={self.T}, capacity_stats[{_stats(self.cap)}]"
        )

    def log_node(
        self,
        parent_id: Optional[int],
        branch_fix: Optional[Tuple[int, int, Tuple[int, int]]],
        bound: float,
        incumbent: float,
        status: str,
    ) -> int:
        """
        Record a node in the search tree.
        parent_id: id of parent node or None if root.
        branch_fix: branching decision leading to this node (item, period, (lb,ub)) or None for root.
        bound: LP bound at this node.
        incumbent: current best incumbent objective when this node was processed.
        status: "branching" if node will be explored, "pruned" if not.
        Returns the assigned node id.
        """
        node_id = next(self._node_id_counter)
        self.tree[node_id] = {
            "id": node_id,
            "parent": parent_id,
            "fix": (branch_fix if branch_fix is not None else None),
            "obj": bound,
            "incumbent": (incumbent if incumbent < float("inf") else float("inf")),
            "status": status,
        }
        # Push this node on the stack (to be the parent of potential children)
        self.parent_stack.append(node_id)
        return node_id

    def compute_y_values(
        self, lambda_vals: Dict[int, List[float]]
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute aggregated y values (order indicator usage) from the current solution λ.
        Returns y_usage[(i,t)] = sum_p λ_{i,p} * I(pattern p for item i orders in period t).
        """
        y_usage = {}
        for i in self.items:
            for p_idx, lam_val in enumerate(lambda_vals.get(i, [])):
                if lam_val <= 1e-9:
                    continue
                y_vec = self.master.patterns[i][p_idx]["y"]
                for t, y_bit in enumerate(y_vec):
                    if y_bit == 1:
                        y_usage[(i, t)] = y_usage.get((i, t), 0.0) + lam_val
        return y_usage

    def price_item(
        self, i: int, mu_hat: List[float], pi_i: float
    ) -> Tuple[int, float, float, Optional[List[int]], str]:
        """
        Solve pricing problem for item i to find a new pattern with negative reduced cost.
        Returns (item_id, reduced_cost, true_cost, q_plan, method_tag).
        method_tag indicates which pricing method found the pattern ("DP", "SP", "MIP", or "NONE" if none found).
        """
        demand = self.dem[i]
        T = len(demand)
        must_order_periods = {
            t
            for (ii, t), (lb, ub) in self.order_fix.items()
            if ii == i and lb == ub == 1
        }
        forbid_periods = {
            t
            for (ii, t), (lb, ub) in self.order_fix.items()
            if ii == i and lb == ub == 0
        }

        # Primary: Dynamic Programming (if no mandatory order constraint)
        warm_q: Optional[List[int]] = None
        warm_flow: Optional[Dict[Tuple[int, int], float]] = None
        if not must_order_periods:
            # Prepare adjusted costs for DP (unit cost minus dual for each period)
            p_eff = [self.c_var[i] - mu_hat[t] for t in range(T)]
            h_seq = [self.h[i]] * T
            S_seq = [self.setup[i]] * T
            # Determine shelf-life horizon: v_last[t] is last period an order at t can cover
            v_last = [min(T - 1, t + self.mseq[i][t] - 1) for t in range(T)]
            dp_result = dp_fefo_solve(
                demand,
                p_eff,
                h_seq,
                S_seq,
                v_last,
                forbid_t=forbid_periods if forbid_periods else None,
            )
            reduced_cost = dp_result.total_cost - pi_i
            if reduced_cost < -RED_COST_EPS:
                # Construct q plan from DP result
                q_plan = [int(round(qty)) for qty in dp_result.x_t]
                # Validate capacity of single order (q_plan should not exceed k_max in any period ideally)
                if sum(q_plan) > 0:
                    # Compute true cost (with original costs, not dual-adjusted)
                    total_var = self.c_var[i] * sum(q_plan)
                    total_hold = 0.0
                    for t in range(T):
                        for u in range(T):
                            if dp_result.x_ti[t][u] > 1e-12:
                                total_hold += self.h[i] * (u - t) * dp_result.x_ti[t][u]
                    total_setup = self.setup[i] * sum(
                        1 for t in range(T) if q_plan[t] > 0
                    )
                    true_cost = total_setup + total_var + total_hold
                    return i, reduced_cost, true_cost, q_plan, "DP"
            # If DP did not find a negative reduced cost, store its q_plan as warm start (even if suboptimal, it can help MIP)
            warm_q = [int(round(qty)) for qty in dp_result.x_t] if dp_result else None
            warm_flow = (
                {
                    (t, u): dp_result.x_ti[t][u]
                    for t in range(T)
                    for u in range(T)
                    if dp_result.x_ti[t][u] > 1e-12
                }
                if dp_result
                else None
            )

        # Secondary: Shortest Path fallback (if not must_order)
        if not must_order_periods:
            rc_sp, true_cost_sp, q_sp = price_shortest_path(
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
                return i, rc_sp, true_cost_sp, q_sp, "SP"
            # If SP yields a pattern (even if not negative), use it as warm start if DP didn't provide one
            if warm_q is None and q_sp is not None:
                warm_q = q_sp
                warm_flow = build_fefo_flow_from_q(demand, q_sp, self.mseq[i])

        # Last resort: MIP pricing (handles must_order or complex cases)
        rc_mip, true_cost_mip, q_mip = self.price_mip(
            item_id=i, mu=mu_hat, pi_i=pi_i, warm_q=warm_q, warm_flow=warm_flow
        )
        if q_mip is not None and rc_mip < -RED_COST_EPS:
            return i, rc_mip, true_cost_mip, q_mip, "MIP"
        # No improving column found
        return i, float("inf"), float("inf"), None, "NONE"

    def price_mip(
        self,
        item_id: int,
        mu: List[float],
        pi_i: float,
        warm_q: Optional[List[int]],
        warm_flow: Optional[Dict[Tuple[int, int], float]],
    ) -> Tuple[float, float, Optional[List[int]]]:
        """
        Solve single-item pricing via a MILP (using Gurobi) explicitly modeling periods and flows.
        This can handle additional constraints like forced orders at certain periods (y fixes).
        Returns (reduced_cost, true_cost, q_plan) or (inf, inf, None) if no column with negative reduced cost found.
        """
        D = self.dem[item_id]
        T = len(D)
        m_seq = self.mseq[item_id]
        kmax = self.k_max[item_id]
        # Build model
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
        model.Params.Cutoff = (
            -RED_COST_EPS
        )  # stop when no better than -eps (no improving columns)
        model.Params.IntFeasTol = 1e-9
        model.Params.OptimalityTol = 1e-9

        # Variables
        q = model.addVars(T, vtype=grb.GRB.INTEGER, lb=0, ub=kmax, name="q")
        y = model.addVars(T, vtype=grb.GRB.BINARY, name="y")
        x = (
            {}
        )  # flow variables: x[s,t] = quantity from order at s used to satisfy demand at t
        for s in range(T):
            v_last = min(T - 1, s + m_seq[s] - 1)
            for t in range(s, v_last + 1):
                x[(s, t)] = model.addVar(
                    vtype=grb.GRB.CONTINUOUS, lb=0.0, name=f"x_{s}_{t}"
                )

        # Constraints:
        # (1) Demand satisfaction: sum_{s<=t} x[s,t] = D[t] for each period t (all demand must be met exactly, no backorder)
        for t in range(T):
            model.addConstr(
                grb.quicksum(x[(s, t)] for s in range(0, t + 1) if (s, t) in x) == D[t],
                name=f"demand_{t}",
            )
        # (2) Flow capacity: for each order s, sum_{t >= s} x[s,t] <= q[s] (an order at s can supply at most q[s] units total)
        for s in range(T):
            model.addConstr(
                grb.quicksum(x[(s, t)] for t in range(s, T) if (s, t) in x) <= q[s],
                name=f"flow_{s}",
            )
        # (3) Setup linking: y[s] = 1 if and only if q[s] > 0 (order placed implies positive quantity)
        for s in range(T):
            model.addGenConstrIndicator(y[s], True, q[s] >= 1, name=f"setup_on_{s}")
            model.addGenConstrIndicator(y[s], False, q[s] == 0, name=f"setup_off_{s}")
        # (4) Branching (fixed y) constraints:
        for (ii, tt), (lb, ub) in self.order_fix.items():
            if ii == item_id:
                # If y fixed to 0: enforce y[tt] = 0; if fixed to 1: enforce y[tt] = 1
                y[tt].LB = lb
                y[tt].UB = ub

        # Objective: minimize reduced cost = (setup_cost + variable_cost + holding_cost) - mu* q - pi_i
        setup_cost_term = grb.quicksum(self.setup[item_id] * y[s] for s in range(T))
        var_cost_term = grb.quicksum(self.c_var[item_id] * q[s] for s in range(T))
        hold_cost_term = grb.quicksum(
            self.h[item_id] * (t - s) * x[(s, t)] for (s, t) in x
        )
        dual_adjustment = grb.quicksum(-mu[s] * q[s] for s in range(T)) - pi_i
        model.setObjective(
            setup_cost_term + var_cost_term + hold_cost_term + dual_adjustment,
            grb.GRB.MINIMIZE,
        )

        # Warm start (if available) to speed up MIP:
        if warm_q:
            for t in range(T):
                if warm_q[t] is not None:
                    q[t].Start = int(max(0, warm_q[t]))
                    y[t].Start = 1 if warm_q[t] and warm_q[t] > 0 else 0
        if warm_flow:
            for (s, t), val in warm_flow.items():
                if (s, t) in x:
                    x[(s, t)].Start = float(val)

        model.optimize()
        if (
            model.Status in (grb.GRB.OPTIMAL, grb.GRB.CUTOFF)
            and model.ObjVal < -RED_COST_EPS
        ):
            # Construct pattern from solution
            q_solution = [int(round(q[t].X)) for t in range(T)]
            # Compute true cost (without dual adjustments) of this pattern
            total_setup = self.setup[item_id] * sum(
                1 for t in range(T) if q_solution[t] > 0
            )
            total_var = self.c_var[item_id] * sum(q_solution)
            total_hold = sum(self.h[item_id] * (t - s) * x[(s, t)].X for (s, t) in x)
            true_cost = total_setup + total_var + total_hold
            # reduced cost from model objective already includes -pi in objective, so:
            reduced_cost = model.ObjVal
            return reduced_cost, true_cost, q_solution
        return float("inf"), float("inf"), None

    def column_generation(
        self,
    ) -> Tuple[float, Tuple[Dict[int, float], List[float], Dict[int, List[float]]]]:
        """
        Perform column generation (price new patterns) at the current node until no improving column or stopping criteria.
        Returns tuple (LP_obj_value, (pi, mu, lambda_vals)) for the final LP solution.
        """
        iteration = 0
        suppressed_count = 0
        cg_guard = ConvergenceGuard(
            CG_ESTOP_MIN_ITERS, CG_ESTOP_PATIENCE, CG_ESTOP_MIN_REL_IMPROVE
        )
        best_obj = float("inf")
        while True:
            iteration += 1
            result = self.master.optimize()
            if result is None:
                # RMP is infeasible (no patterns satisfy branching constraints). Try to recover by generating any feasible pattern.
                log(
                    "[WARN] RMP infeasible under current constraints. Attempting to restore feasibility via pricing."
                )
                added_any = False
                for i in self.items:
                    # Force a pattern regardless of reduced cost, just to get feasibility
                    _, _, _, q_plan, _ = self.price_item(
                        i, mu_hat=[0.0] * self.T, pi_i=0.0
                    )
                    if q_plan:
                        cost = (
                            sum(self.setup[i] * (1 if q > 0 else 0) for q in q_plan)
                            + sum(self.c_var[i] * q for q in q_plan)
                            + sum(
                                self.h[i] * (t - s) * q
                                for s, q in enumerate(q_plan)
                                for t in range(s, self.T)
                            )
                        )
                        y_plan = [1 if q > 0 else 0 for q in q_plan]
                        self.master.add_pattern(i, cost, q_plan, y_plan)
                        added_any = True
                self.master.model.update()
                result = self.master.optimize()
                if result is None:
                    # Could not recover feasibility
                    log("[FAIL] Unable to restore feasibility at this node. Pruning.")
                    return float("inf"), ({}, [], {})
                else:
                    log("[INFO] Feasibility restored by adding emergency columns.")
            obj_val, pi, mu, lambda_vals = result
            best_obj = min(best_obj, obj_val)
            if VERBOSE and (iteration % LOG_EVERY_ITER == 0 or iteration == 1):
                log(f"[CG] iter {iteration:2d}  obj={obj_val:.2f}")
                if DEBUG_DUALS:
                    log(
                        f"[INFO] Dual μ stats: {_stats(mu)}; head[{_head(mu, DEBUG_SUMMARY_WIDTH)}]"
                    )
            # Dual stabilization: smoothed duals for pricing
            mu_hat = [
                (self.alpha * mu_t + (1 - self.alpha) * prev_mu_t)
                for mu_t, prev_mu_t in zip(mu, self.prev_mu)
            ]
            self.prev_mu = mu_hat[:]
            if DEBUG_DUALS:
                log(
                    f"[INFO] Stabilized dual μ̂ stats: {_stats(mu_hat)}; head[{_head(mu_hat, DEBUG_SUMMARY_WIDTH)}]"
                )

            # Pricing for each item
            added = False
            added_count = 0
            pricing_info = []
            for i in self.items:
                i_id, rc, true_cost, q_plan, method = self.price_item(i, mu_hat, pi[i])
                pricing_info.append((i_id, method, rc))
                if q_plan is not None and rc < -RED_COST_EPS:
                    # Add new column/pattern to master
                    y_plan = [1 if qty > 0 else 0 for qty in q_plan]
                    self.master.add_pattern(i_id, true_cost, q_plan, y_plan)
                    added = True
                    added_count += 1
            if DEBUG_PRICING:
                info_str = ", ".join(
                    f"{i}: {tag} rc={rc:.3g}" for (i, tag, rc) in pricing_info
                )
                log(f"[INFO] Pricing results → {info_str}")
                log(
                    f"[INFO] Added {added_count} columns (total patterns now {sum(len(self.master.patterns[i]) for i in self.items)})"
                )

            self.master.model.update()
            # Check CG stagnation stop
            if iteration >= CG_ESTOP_MIN_ITERS and cg_guard.step(obj_val):
                log(
                    f"[EARLY-STOP] Column generation stagnation: < {CG_ESTOP_MIN_REL_IMPROVE:.0e} rel. improvement for {CG_ESTOP_PATIENCE} iterations."
                )
                break
            # Termination: no columns added or reached iteration limit
            if not added or iteration >= CG_MAX_ITERS:
                if suppressed_count:
                    log(
                        f"[INFO] Suppressed {suppressed_count} micro-negative columns (|rc| ≤ {RED_COST_EPS})."
                    )
                break
        return obj_val, (pi, mu, lambda_vals)

    def choose_branch_variable(
        self, lambda_vals: Dict[int, List[float]], mu: List[float]
    ) -> Optional[Tuple[int, int]]:
        """
        Identify a branching candidate (item i, period t) for which y_{i,t} is fractional (0 < sum_p λ_{i,p} * I(order at t) < 1).
        Use a selection criterion to pick the most fractional (closest to 0.5) possibly weighted by dual mu.
        Returns (i, t) if found, or None if all y are effectively integer.
        """
        y_usage = self.compute_y_values(lambda_vals)
        best_score = -1.0
        best_key: Optional[Tuple[int, int]] = None
        fractional_list = []
        for (i, t), val in y_usage.items():
            if val > 1e-6 and val < 1 - 1e-6:
                # fractional usage
                fractional_list.append(((i, t), val))
                # score closeness to 0.5, adjust by dual to prioritize tight constraints maybe
                closeness = 0.5 - abs(val - 0.5)
                score = closeness * (1.0 + max(0.0, mu[t]))
                if score > best_score:
                    best_score = score
                    best_key = (i, t)
        if DEBUG_BRANCH and fractional_list:
            top_candidates = sorted(fractional_list, key=lambda kv: abs(0.5 - kv[1]))[
                :5
            ]
            top_str = ", ".join(
                f"y[{i},{t}]={val:.3f}" for ((i, t), val) in top_candidates
            )
            log(f"[INFO] Fractional y values (top): {top_str}")
        return best_key

    def solve_restricted_IP(
        self, time_limit: float = 10.0
    ) -> Tuple[float, Optional[Dict[int, List[float]]]]:
        """
        Solve a restricted IP (all λ variables integer) on the current master columns, to attempt finding an integer solution.
        Use as heuristic to get an incumbent if the LP solution is fractional.
        Returns (obj_value, lambda_solution) if found or time limit reached, or (inf, None) if no solution.
        """
        ip_model = grb.Model("RMP_IP", env=GRB_ENV)
        ip_model.Params.LogToConsole = 0
        ip_model.Params.OutputFlag = 0
        ip_model.Params.Threads = N_MASTER_THREADS
        lambda_ip = {i: [] for i in self.items}
        # add binary λ variables corresponding to current master patterns
        for i in self.items:
            for p_idx, pat in enumerate(self.master.patterns[i]):
                v = ip_model.addVar(
                    vtype=grb.GRB.BINARY, obj=pat["cost"], name=f"lam_{i}_{p_idx}"
                )
                # provide current LP solution as a hint
                try:
                    v.Start = 1 if self.master.lambda_vars[i][p_idx].X > 0.5 else 0
                except Exception:
                    pass
                lambda_ip[i].append(v)
        # Constraints: selection and capacity (copied from LP)
        for i in self.items:
            ip_model.addConstr(grb.quicksum(lambda_ip[i]) == 1, name=f"sel_{i}")
        for t in range(self.T):
            ip_model.addConstr(
                grb.quicksum(
                    pat["q"][t] * lambda_ip[i][p_idx]
                    for i in self.items
                    for p_idx, pat in enumerate(self.master.patterns[i])
                    if pat["q"][t] > 0
                )
                <= self.cap[t],
                name=f"cap_{t}",
            )
        ip_model.ModelSense = grb.GRB.MINIMIZE
        ip_model.Params.TimeLimit = time_limit
        start_time = time.perf_counter()
        ip_model.optimize()
        elapsed = time.perf_counter() - start_time
        if ip_model.Status in (grb.GRB.OPTIMAL, grb.GRB.TIME_LIMIT):
            lambda_solution = {i: [var.X for var in lambda_ip[i]] for i in self.items}
            log(
                f"[INFO] Restricted IP solved in {elapsed:.2f}s, status={ip_model.Status}, obj={ip_model.ObjVal:.2f}"
            )
            return ip_model.ObjVal, lambda_solution
        log(f"[WARN] Restricted IP failed with status {ip_model.Status}")
        return float("inf"), None

    def branch_and_price(
        self, best_incumbent: float = float("inf"), best_solution: Optional[Dict] = None
    ) -> Tuple[float, Optional[Dict]]:
        """
        Recursively perform branch-and-price search.
        Returns the best incumbent objective value and the corresponding solution (patterns & λ values).
        """
        # Solve LP relaxation at this node (with column generation)
        lp_obj, (pi, mu, lambda_vals) = self.column_generation()
        parent_id = self.parent_stack[-1] if self.parent_stack else None
        status = "branching" if lp_obj < best_incumbent - 1e-9 else "pruned"
        node_id = self.log_node(
            parent_id,
            (list(self.order_fix.items())[-1] if self.order_fix else None),
            lp_obj,
            best_incumbent,
            status,
        )
        BranchPrice._print_gap(lp_obj, best_incumbent)
        # Prune node if bound is not better than current best
        if lp_obj >= best_incumbent - 1e-9:
            # Pop this node from stack and return
            return best_incumbent, best_solution
        # Check global optimality gap criterion
        if best_incumbent < float("inf"):
            gap_percent = 100.0 * (best_incumbent - lp_obj) / max(1e-12, best_incumbent)
            if gap_percent < BAP_GAP_TOL * 100:
                log(
                    f"[EARLY-STOP] Terminating search early: gap {gap_percent:.3f}% below {BAP_GAP_TOL*100:.2f}% threshold."
                )
                # Do not explore further branches; return current best incumbent
                return best_incumbent, best_solution

        # Get fractional pattern usage solution
        res = self.master.optimize()
        if res is None:
            log("[FAIL] RMP became infeasible at node (should not happen after CG).")
            return best_incumbent, best_solution
        _, _, mu_final, lambda_vals = res

        # Choose branching variable (item, period) with fractional y
        branch_var = self.choose_branch_variable(lambda_vals, mu_final)
        if branch_var is None:
            # All y are effectively integer (no fractional ordering decisions)
            # Try to obtain an integer λ solution from current columns
            ip_obj, lambda_int = self.solve_restricted_IP(time_limit=5.0)
            if lambda_int is not None and ip_obj <= lp_obj + 1e-6:
                # Found a valid integer solution not worse than LP bound
                best_incumbent = ip_obj
                patterns_snapshot = deepcopy(self.master.patterns)
                best_solution = {
                    "lambda": lambda_int,
                    "patterns": patterns_snapshot,
                    "order_fix": dict(self.order_fix),
                }
                log(f"[SOL] New incumbent {ip_obj:.2f} found (restricted IP solution).")
                BranchPrice._print_gap(lp_obj, best_incumbent, prefix="    ")
            else:
                # Use the LP solution (which has integer y by assumption) as incumbent (patterns combination may be fractional)
                best_incumbent = lp_obj
                patterns_snapshot = deepcopy(self.master.patterns)
                best_solution = {
                    "lambda": lambda_vals,
                    "patterns": patterns_snapshot,
                    "order_fix": dict(self.order_fix),
                }
                log(f"[SOL] Incumbent {lp_obj:.2f} (LP solution, y integral).")
                BranchPrice._print_gap(lp_obj, best_incumbent, prefix="    ")
            # End branching at this node
            self.parent_stack.pop()
            return best_incumbent, best_solution

        # Branch on chosen fractional y
        i_b, t_b = branch_var
        log(
            f"[BRANCH] Branch on y[{i_b},{t_b}] ≈ {self.compute_y_values(lambda_vals).get((i_b, t_b), 0):.3f} (μ[{t_b}]={mu_final[t_b]:.3f})"
        )
        # Explore branch y[i_b, t_b] = 0 (no order for item i_b in period t_b)
        self.order_fix[(i_b, t_b)] = (0, 0)
        child0 = BranchPrice.__new__(BranchPrice)
        # Copy static data
        child0.items = self.items
        child0.dem = self.dem
        child0.c_var = self.c_var
        child0.h = self.h
        child0.setup = self.setup
        child0.b_var = self.b_var
        child0.cap = self.cap
        child0.T = self.T
        child0.mseq = self.mseq
        child0.k_max = self.k_max
        # Copy dynamic state
        child0.order_fix = dict(self.order_fix)
        child0.prev_mu = self.prev_mu[:]  # carry over stabilized duals
        child0.alpha = self.alpha
        # Create a new MasterModel for child node including branch constraint
        child0.master = self.master.clone_with_branching(order_fix=child0.order_fix)
        # Inherit tree tracking info
        child0.tree = self.tree
        child0._node_id_counter = self._node_id_counter
        child0.parent_stack = list(self.parent_stack)
        best_incumbent, best_solution = child0.branch_and_price(
            best_incumbent, best_solution
        )
        # Backtrack branching decision for next sibling
        self.order_fix.pop((i_b, t_b), None)

        # Branch on y[i_b, t_b] = 1 (must have an order at item i_b, period t_b)
        self.order_fix[(i_b, t_b)] = (1, 1)
        child1 = BranchPrice.__new__(BranchPrice)
        # Copy static data
        child1.items = self.items
        child1.dem = self.dem
        child1.c_var = self.c_var
        child1.h = self.h
        child1.setup = self.setup
        child1.b_var = self.b_var
        child1.cap = self.cap
        child1.T = self.T
        child1.mseq = self.mseq
        child1.k_max = self.k_max
        # Copy dynamic state
        child1.order_fix = dict(self.order_fix)
        child1.prev_mu = self.prev_mu[:]
        child1.alpha = self.alpha
        child1.master = self.master.clone_with_branching(order_fix=child1.order_fix)
        child1.tree = self.tree
        child1._node_id_counter = self._node_id_counter
        child1.parent_stack = list(self.parent_stack)
        best_incumbent, best_solution = child1.branch_and_price(
            best_incumbent, best_solution
        )
        # Remove branch fix and cleanup parent stack as we return
        self.order_fix.pop((i_b, t_b), None)
        self.parent_stack.pop()
        return best_incumbent, best_solution

    @staticmethod
    def _print_gap(bound: float, incumbent: float, prefix: str = "[GAP] ") -> None:
        """
        Print current gap between LP bound and best incumbent in percentage.
        """
        if incumbent < float("inf"):
            gap_pct = 100.0 * (incumbent - bound) / max(1e-12, incumbent)
            log(f"{prefix}bound={bound:.2f}  best={incumbent:.2f}  gap={gap_pct:.2f}%")
        else:
            log(f"{prefix}bound={bound:.2f}  best=∞  gap=∞")


# =============================================================================
# Solution reporting and audits
# =============================================================================
def detailed_exec_rep(bp: BranchPrice, sol: Dict) -> None:
    """
    Print a detailed execution report for each item, verifying inventory and backlogs.
    """
    log("\n=== Detailed execution report (per item) ===")
    lam = sol.get("lambda", {})
    patterns = sol.get("patterns", bp.master.patterns)
    for i in bp.items:
        print(f"\nItem {i}")
        lam_i = lam.get(i, [])
        if not lam_i:
            print("[ERROR] No pattern selected for this item.")
            continue
        # Choose the pattern with highest λ value (should be 1 for basic solutions)
        sel_idx = max(range(len(lam_i)), key=lambda k: lam_i[k])
        if lam_i[sel_idx] < 0.5:
            print(
                f"[WARN] Item {i} selected pattern is fractional (λ={lam_i[sel_idx]:.3f})"
            )
        if sel_idx >= len(patterns[i]):
            print("[ERROR] Pattern index out of range.")
            continue
        pat = patterns[i][sel_idx]
        q = pat["q"]
        y = pat["y"]
        D = bp.dem[i]
        m_seq = bp.mseq[i]
        # Simulate inventory flows for this item
        cohorts: List[Tuple[int, float]] = []  # (remaining shelf life, qty)
        backlog = 0  # (should remain 0 since no backorders allowed)
        print(" t | demand | order | inventory_after | backlog")
        print("--------------------------------------------")
        for t in range(bp.T):
            # If an order is placed at t, add it to inventory (with its shelf life)
            if q[t] > 0:
                shelf_life = m_seq[i][t] if isinstance(m_seq[i], list) else m_seq[i]
                remaining_life = shelf_life
                cohorts.append((remaining_life, float(q[t])))
            # Apply FEFO: consume from cohorts in order of remaining life
            need = D[t] + backlog
            backlog = 0
            # Sort by remaining life (FEFO)
            cohorts.sort(key=lambda x: x[0])
            new_cohorts = []
            for life, qty in cohorts:
                if need <= 0:
                    # carry remaining cohort
                    new_cohorts.append((life - 1, qty)) if life > 1 else None
                    continue
                if life <= 0 or qty <= 0:
                    # expired or empty
                    continue
                take = min(qty, need)
                qty -= take
                need -= take
                # any remaining from this cohort continues with reduced shelf life
                if qty > 1e-9 and life > 1:
                    new_cohorts.append((life - 1, qty))
            cohorts = new_cohorts
            if need > 1e-9:
                # not enough inventory, becomes backlog (which should not happen if solution is feasible without backorders)
                backlog = need
            inv_amount = sum(qty for _, qty in cohorts)
            print(
                f"{t:2d} | {D[t]:6.1f} | {q[t]:5.1f} | {inv_amount:14.1f} | {backlog:7.1f}"
            )


def capacity_audit(bp: BranchPrice, sol: Dict) -> None:
    """
    Print capacity utilization by period for the final selected patterns.
    """
    print("\n=== Capacity utilization audit ===")
    lam = sol.get("lambda", {})
    patterns = sol.get("patterns", bp.master.patterns)
    usage = [0.0] * bp.T
    for i in bp.items:
        lam_i = lam.get(i, [])
        if not lam_i:
            continue
        sel_idx = max(range(len(lam_i)), key=lambda k: lam_i[k])
        pattern = patterns[i][sel_idx]
        q = pattern["q"]
        for t in range(bp.T):
            usage[t] += q[t]
    print("Period | Used  | Capacity | Slack")
    print("-------------------------------")
    for t in range(bp.T):
        used = usage[t]
        cap = bp.cap[t]
        slack = cap - used
        print(f"{t:6d} | {used:5.1f} | {cap:8.1f} | {slack:5.1f}")


# =============================================================================
# Data classes and instance generation
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
    manual_capacity: Optional[List[int]] = None

    @property
    def capacity(self) -> List[int]:
        if self.manual_capacity is not None:
            return self.manual_capacity
        # capacity = total demand per period + padding
        cap = [
            sum(item.demand[t] for item in self.items.values())
            for t in range(self.period)
        ]
        if self.capacity_pad:
            cap = [c + self.capacity_pad for c in cap]
        return cap

    @property
    def kmax(self) -> Dict[int, int]:
        # Maximum quantity that could be ordered for each item (e.g., sum of demands + some buffer)
        return {
            i: sum(item.demand) + max(5, int(0.2 * sum(item.demand)))
            for i, item in self.items.items()
        }

    def to_dicts(
        self,
    ) -> Tuple[
        Dict[int, List[int]],
        Dict[int, float],
        Dict[int, float],
        Dict[int, float],
        Dict[int, float],
        List[int],
        Dict[int, List[int]],
        Dict[int, int],
    ]:
        demand = {i: item.demand for i, item in self.items.items()}
        c_var = {i: item.c_var for i, item in self.items.items()}
        h = {i: item.h for i, item in self.items.items()}
        setup = {i: item.setup for i, item in self.items.items()}
        b_var = {i: item.b_var for i, item in self.items.items()}
        cap = self.capacity
        mseq = {i: item.shelf_seq for i, item in self.items.items()}
        kmax = self.kmax
        return demand, c_var, h, setup, b_var, cap, mseq, kmax

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        data = asdict(self)
        # convert keys to str for JSON
        data["items"] = {str(k): v for k, v in data["items"].items()}
        path = Path(path)
        path.write_text(json.dumps(data, indent=indent))

    @classmethod
    def from_json(cls, path: str | Path) -> Lot:
        data = json.loads(Path(path).read_text())
        # convert item keys back to int
        items = {int(k): Item(**v) for k, v in data["items"].items()}
        return cls(
            period=data["period"],
            capacity_pad=data["capacity_pad"],
            items=items,
            manual_capacity=data.get("manual_capacity"),
        )


# =============================================================================
# Logging capture and results saving
# =============================================================================
class Tee(io.TextIOBase):
    """Utility to write to multiple text streams (e.g., stdout and a buffer)."""

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
    instance_path: Path,
    outdir: str = "results",
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save instance copy
    instance_copy_path = outdir / f"instance_{timestamp}.json"
    try:
        instance_copy_path.write_text(Path(instance_path).read_text())
    except Exception:
        pass
    # Prepare output JSON data
    demand, c_var, h, setup, b_var, cap, mseq, kmax = lot.to_dicts()
    patterns_data = {
        i: [
            {"cost": pat["cost"], "y": pat["y"], "q": pat["q"]}
            for pat in (solution.get("patterns", bp.master.patterns))[i]
        ]
        for i in bp.items
    }
    lambda_solution = solution.get("lambda", {})
    tree_list = list(bp.tree.values())
    order_fix = (
        {
            f"{i}_{t}": list(bounds)
            for (i, t), bounds in solution.get("order_fix", {}).items()
        }
        if solution.get("order_fix")
        else {}
    )
    run_data = {
        "timestamp": timestamp,
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
                "shelf_life_sequence": mseq[i],
                "kmax": kmax[i],
            }
            for i in bp.items
        },
        "patterns": patterns_data,
        "lambda_solution": lambda_solution,
        "order_fix": order_fix,
        "tree": tree_list,
    }
    json_path = outdir / f"run_{timestamp}.json"
    txt_path = outdir / f"run_{timestamp}.txt"
    json_path.write_text(json.dumps(run_data, indent=2))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Perishable Lot-Sizing Branch-and-Price Run Report ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Objective: {objective:.2f}\n")
        f.write(f"Elapsed time (s): {elapsed_sec:.2f}\n")
        f.write(f"Instance file: {instance_path}\n")
        f.write(f"Instance copy: {instance_copy_path}\n")
        f.write(f"Backorders allowed: {ALLOW_BACKORDER}\n")
        f.write(f"Items: {len(bp.items)}   Periods: {bp.T}\n")
        f.write("\n-- Capacity (per period) --\n")
        f.write(", ".join(str(c) for c in cap) + "\n")
        f.write("\n-- Item parameters (costs and shelf-life) --\n")
        for i in bp.items:
            f.write(
                f"Item {i}: setup={setup[i]}, c_var={c_var[i]}, h={h[i]}, b_var={b_var[i]}, "
                f"min_shelf={min(mseq[i])}, max_shelf={max(mseq[i])}\n"
            )
        f.write("\n-- Selected patterns (λ > 0) --\n")
        for i in bp.items:
            lam_i = lambda_solution.get(i, [])
            if not lam_i:
                f.write(f"Item {i}: No pattern selected.\n")
                continue
            sel_idx = max(range(len(lam_i)), key=lambda k: lam_i[k])
            pat = patterns_data[i][sel_idx]
            f.write(
                f"Item {i}: pattern {sel_idx}, cost={pat['cost']:.2f}, y={pat['y']}, q={pat['q']}\n"
            )
        f.write("\n-- Branch-and-Price Tree (nodes) --\n")
        for node in tree_list:
            f.write(json.dumps(node) + "\n")
        f.write("\n=== Console Output Log ===\n")
        f.write(log_text)
    return txt_path


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    if (not RANDOMIZE) and INSTANCE_PATH.exists():
        lot = Lot.from_json(INSTANCE_PATH)
        print(f"[INFO] Loaded instance from {INSTANCE_PATH}")
    else:
        # Define item specs: (id, setup, backorder_cost, variable_cost, holding_cost, (min_shelf, max_shelf))
        specs = [
            (0, 127.5, 5.0, 2.0, 0.4, (1, 50)),
            (1, 29.0, 5.0, 3.0, 0.6, (1, 60)),
            (2, 50.0, 5.0, 1.0, 0.3, (3, 70)),
            (3, 75.0, 5.0, 4.0, 0.5, (3, 5)),
            (4, 100.0, 5.0, 2.5, 0.4, (3, 5)),
        ]
        period = 80
        manual_caps = [870] * period if USE_MANUAL_CAPACITY else None
        # Build random instance
        lot = Lot(period=period, capacity_pad=10, manual_capacity=manual_caps)
        for spec in [s[:5] for s in specs]:
            i, stp, b, c, h_cost = spec
            demand = [random.randint(1, 200) for _ in range(period)]
            # temporary shelf sequence (using default range, will override next)
            shelf_seq = [random.randint(3, 5) for _ in range(period)]
            lot.items[i] = Item(i, demand, stp, b, c, h_cost, shelf_seq)
        # Override shelf life sequences with specified ranges from specs
        for i, _stp, _b, _c, _h, shelf_rng in specs:
            lo, hi = shelf_rng
            lot.items[i].shelf_seq = [random.randint(lo, hi) for _ in range(period)]
        lot.to_json(INSTANCE_PATH)
        print(f"[INFO] Generated new instance at {INSTANCE_PATH}")

    # Run branch-and-price and capture output
    output_buffer = io.StringIO()
    tee = Tee(sys.stdout, output_buffer)
    start_time = time.perf_counter()
    with contextlib.redirect_stdout(tee):
        demand, c_var, h, setup, b_var, cap, mseq, kmax = lot.to_dicts()
        bp = BranchPrice(demand, c_var, h, setup, b_var, cap, mseq, kmax)
        log(
            "[INFO] Solving LP relaxation with column generation and branching on fractional decisions."
        )
        best_obj, best_sol = bp.branch_and_price()
        if best_sol is None:
            print(
                "[ERROR] No feasible solution found (capacity might be insufficient)."
            )
            # If no solution, output demands vs capacity for debugging
            for t in range(lot.period):
                total_d = sum(item.demand[t] for item in lot.items.values())
                print(f"Period {t}: Demand={total_d}, Capacity={lot.capacity[t]}")
        else:
            elapsed = time.perf_counter() - start_time
            print(
                f"\nFinal Objective: {best_obj:.2f}   (Time elapsed: {elapsed:.2f} s)\n"
            )
            # Display final gap using root node bound (if exists in tree) vs best
            root_nodes = [node for node in bp.tree.values() if node["parent"] is None]
            root_bound = root_nodes[0]["obj"] if root_nodes else best_obj
            BranchPrice._print_gap(root_bound, best_obj, prefix="[FINAL] ")
            print(f"[INFO] Backorders allowed: {ALLOW_BACKORDER}")
            # Print chosen patterns for each item
            lam_solution = best_sol.get("lambda", {})
            patterns = best_sol.get("patterns", bp.master.patterns)
            for i in bp.items:
                lam_i = lam_solution.get(i, [])
                if not lam_i:
                    print(f"[WARN] Item {i} has no selected pattern.")
                    continue
                sel_idx = max(range(len(lam_i)), key=lambda k: lam_i[k])
                print(
                    f"Item {i}: pattern {sel_idx}, y={patterns[i][sel_idx].get('y')}, q={patterns[i][sel_idx]['q']}"
                )
            # Detailed per-item report and capacity audit
            detailed_exec_rep(bp, best_sol)
            capacity_audit(bp, best_sol)
            print(
                f"\nFinal Objective: {best_obj:.2f}   (Time elapsed: {elapsed:.2f} s)"
            )

    log_text = output_buffer.getvalue()
    elapsed_total = time.perf_counter() - start_time
    if best_sol is None:
        best_obj = float("inf")
        best_sol = {}
    result_path = save_results(
        lot, bp, best_sol, best_obj, elapsed_total, log_text, INSTANCE_PATH
    )
    print(
        f"[RESULT] Output saved to {result_path.parent} (main report: {result_path.name})"
    )
