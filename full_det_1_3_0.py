"""
Advanced branch-and-price for a 3-item, 30-period perishable lot-sizing problem
with setup costs, holding costs, optional back-orders, dual stabilisation,
and order/no-order branching on (item,period).

Python ≥ 3.8 , gurobipy installed, academic licence assumed.
"""

from __future__ import annotations
import itertools, random, sys
from copy import deepcopy
import numpy as np
import gurobipy as grb

import random
from dataclasses import asdict, dataclass, field
from typing import List, Dict

# main.py  (top of file)
import time, json
from pathlib import Path
import numpy as np

from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

import pygraphviz


ENABLE_LIVE_PLOTS = False
TREE_VISUALIZATION = False

if TREE_VISUALIZATION or ENABLE_LIVE_PLOTS:
    fig, ax = plt.subplots()
    plt.ion()
    plt.show(block=False)


# ────────────────────────────────────────────────────────────────────────────
# adaptive k_max  +  DP progress pulses
# ────────────────────────────────────────────────────────────────────────────
DP_STATE_STEP = 250_000  # print every 100k DP states
HARD_KMAX_CAP = 128  # safety ceiling; raise if you really need it


def live_plot_inventory(t, inv, back, item_id, line_dict):
    ax.clear()
    ax.set_title(f"Inventory / Backorders for Item {item_id}")
    ax.set_xlabel("Period")
    ax.set_ylabel("Units")
    ax.grid(True)
    ax.plot(range(t + 1), line_dict["inv"][: t + 1], label="Inventory")
    ax.plot(range(t + 1), line_dict["back"][: t + 1], label="Backorders")
    ax.legend()
    plt.pause(0.1)


# ──────────────────────────────────────────────────────────────────────
#  Network-based pricing  (replaces the huge DP)
# ──────────────────────────────────────────────────────────────────────
def _arc_cost(demand, c_var, h, setup, mu_t, t: int, u: int) -> tuple[int, float]:
    """
    One order at period *t* covers demand up to and incl. *u* (t ≤ u).
    Returns (q, reduced_cost).  Assumes u-t < shelf_life and q ≤ k_max.
    """
    q = sum(demand[t : u + 1])  # order size
    # holding cost = Σ_{τ=t}^{u} demand[τ] · (τ-t)
    hold = sum(demand[τ] * (τ - t) for τ in range(t, u + 1))
    red = (c_var - mu_t) * q + h * hold + (setup if q else 0)
    return q, red


def price_shortest_path(
    item_id: int,
    demand: list[int],
    c_var: float,
    h: float,
    setup: float,
    mu: list[float],
    pi: float,
    shelf: int,
    k_max: int,
    order_fix: dict[tuple[int, int], tuple[int, int]],
) -> tuple[float, float, list[int]]:
    """
    Layered DAG shortest path:
        node 0 … T      (node T is the sink)
        arc (t → u+1)   = one order at t covering demand[t…u]
    Returns  (red_cost, true_cost, q_plan).
    *No* back-orders; leaves k_max & branching (order_fix) checks in place.
    """
    T = len(demand)
    dist = [float("inf")] * (T + 1)
    succ = [None] * T  # store (u, q) chosen for period t
    dist[T] = -pi  # reduced cost contribution of λ-variable

    # reverse topological order
    for t in range(T - 1, -1, -1):
        lb, ub = order_fix.get((item_id, t), (0, 1))
        # if order is forbidden at t                     (LB=UB=0 ⇒ δ_t = 0)
        if ub == 0:
            dist[t] = dist[t + 1]  # must rely on earlier inventory
            continue

        best = float("inf")
        best_arc = None
        q_acc = 0
        hold_acc = 0  # accumulate h·Σ d(τ)(τ-t)

        for u in range(t, min(T, t + shelf)):
            q_acc += demand[u]
            if q_acc > k_max:  # order-size ceiling
                break
            hold_acc += demand[u] * (u - t)
            q, red_arc = q_acc, (c_var - mu[t]) * q_acc + h * hold_acc + setup
            cand = red_arc + dist[u + 1]
            if cand < best:
                best = cand
                best_arc = (u, q)

        dist[t] = best
        succ[t] = best_arc

    if dist[0] >= -1e-6:  # no negative reduced cost
        return float("inf"), None, None

    # -------- reconstruct order plan -----------
    q_plan = [0] * T
    true_cost = 0.0
    t = 0
    while t < T:
        u, q = succ[t]
        q_plan[t] = q
        # true cost uses *original* coefficients
        hold = sum(demand[τ] * (τ - t) for τ in range(t, u + 1))
        true_cost += setup + c_var * q + h * hold
        t = u + 1

    red_cost = dist[0]
    return red_cost, true_cost, q_plan


def dp_pricing_general_pulsed(
    item_id,
    demand,
    c_var,
    h,
    setup,
    b_var,
    mu,
    pi,
    shelf_life,
    k_max,
    order_fix,
    allow_backorder=False,
    dbg=False,
):
    """Original DP with a heartbeat print and *no* back-order option."""
    T, L = len(demand), shelf_life

    cum_demand = [0] * (T + 1)  # cum_demand[t] = Σ_{τ=t}^{T} demand[τ]
    running = 0
    for τ in range(T - 1, -1, -1):  # walk backwards
        running += demand[τ]
        cum_demand[τ] = running

    rng = range(-k_max, k_max + 1) if allow_backorder else range(k_max + 1)

    zero = tuple(0 for _ in range(L - 1))
    dp = [dict() for _ in range(T + 2)]
    dp[T + 1][zero] = (-pi, None)

    state_cnt = 0
    for t in range(T, 0, -1):
        d, mu_t = demand[t - 1], mu[t - 1]
        lb, ub = order_fix.get((item_id, t - 1), (0, 1))
        for state in itertools.product(rng, repeat=L - 1):
            state_cnt += 1
            if dbg and state_cnt % DP_STATE_STEP == 0:
                print(
                    f"      [DP] item={item_id} t={t:2d} "
                    f"states={state_cnt//1_000:,}k"
                )
            # --------------- body of the original DP (copy verbatim) -----
            avail = sum(max(x, 0) for x in state)  # ignore back-ordered ages

            # ── PRUNE: too much stock to ever consume ───────────────────────────
            if avail > cum_demand[t - 1]:
                continue

            # ─── PRUNE: if even with the biggest order we’d still throw stock away, skip ──
            if avail > d + k_max:
                continue

            need = d - avail
            q_min = max(0, need, 1 if lb == 1 else 0)
            q_max = k_max if ub else 0  # ub==0 ⇒ no order allowed
            best_val, best_dec = float("inf"), None
            for q in range(q_min, q_max + 1):
                inv = list(state)
                rem = d
                for age in range(L - 1, 0, -1):
                    idx = age - 1
                    use = min(inv[idx], rem)
                    inv[idx] -= use
                    rem -= use
                use_q = min(q, rem)
                rem -= use_q
                age1 = q - use_q - rem  # rem≥0 ⇒ age1≥0
                nxt = tuple([age1] + inv[:-1])
                if age1 > k_max or nxt not in dp[t + 1]:
                    continue
                fixed = setup if q else 0
                true = c_var * q + h * sum(nxt) + b_var * rem + fixed
                red = (
                    (c_var - mu_t) * q
                    + h * sum(nxt)
                    + b_var * rem
                    + fixed
                    + dp[t + 1][nxt][0]
                )
                if red < best_val:
                    best_val, best_dec = red, (q, nxt, true)
            if best_dec:
                dp[t][state] = (best_val, best_dec)
    if zero not in dp[1]:
        return float("inf"), None, None, None
    red_cost, _ = dp[1][zero]
    q_plan, cost, st = [], 0.0, zero

    inv_history = []
    back_history = []
    for t in range(1, T + 1):
        q, st_next, true = dp[t][st][1]
        q_plan.append(q)
        cost += true
        st = st_next

        # Inventory/backorder simulation (replicating logic from detailed_exec_rep)
        inv = list(st)
        demand_t = demand[t - 1]
        inv_new = [0] * L
        inv_new[1:] = inv[:-1]
        inv_new[0] += q
        on_hand = sum(inv_new)
        sell = min(demand_t, on_hand)
        rem = demand_t - sell
        back = rem
        inv_history.append(sum(inv_new))
        back_history.append(back)

        if ENABLE_LIVE_PLOTS:
            live_plot_inventory(
                t - 1,
                inv_history,
                back_history,
                item_id,
                {"inv": inv_history, "back": back_history},
            )

    return red_cost, cost, q_plan, None


# -----------------------------------------------------------------------
#  Dynamic-programming pricing routine
# -----------------------------------------------------------------------
def dp_pricing_general(
    item_id: int,
    demand: list[int],
    c_var: float,
    h: float,
    setup: float,
    b_var: float,
    mu: list[float],
    pi: float,
    shelf_life: int,
    k_max: int,
    order_fix: dict[tuple[int, int], tuple[int, int]],
    allow_backorder: bool = False,
):
    T = len(demand)
    L = shelf_life
    rng = range(-k_max, k_max + 1) if allow_backorder else range(k_max + 1)
    zero_state = tuple(0 for _ in range(L - 1))
    dp = [dict() for _ in range(T + 2)]
    dp[T + 1][zero_state] = (-pi, None)

    for t in range(T, 0, -1):
        d = demand[t - 1]
        mu_t = mu[t - 1]
        lb, ub = order_fix.get((item_id, t - 1), (0, 1))
        for state in itertools.product(rng, repeat=L - 1):
            avail = sum(max(x, 0) for x in state)
            need = d - avail
            q_min = max(0, need, 1 if lb == 1 else 0)
            q_max = 0 if ub == 0 else k_max
            best_val, best_dec = float("inf"), None
            for q in range(q_min, q_max + 1):
                inv = list(state)
                rem = d
                for age in range(L - 1, 0, -1):
                    idx = age - 1
                    use = min(max(inv[idx], 0), rem)
                    inv[idx] -= use
                    rem -= use
                use_q = min(q, rem)
                rem -= use_q
                leftover_q = q - use_q
                inv_neg = -rem if rem > 0 else 0
                age1 = leftover_q - inv_neg
                next_state = [age1] + inv[:-1]
                if max(map(abs, next_state)) > k_max:
                    continue
                next_state = tuple(next_state)
                if next_state not in dp[t + 1]:
                    continue
                fixed = setup if q > 0 else 0
                true = (
                    c_var * q
                    + h * sum(max(x, 0) for x in next_state)
                    + b_var * inv_neg
                    + fixed
                )
                red = (
                    (c_var - mu_t) * q
                    + h * sum(max(x, 0) for x in next_state)
                    + b_var * inv_neg
                    + fixed
                    + dp[t + 1][next_state][0]
                )
                if red < best_val:
                    best_val, best_dec = red, (q, next_state, true)
            if best_dec:
                dp[t][state] = (best_val, best_dec)

    if zero_state not in dp[1]:
        return float("inf"), None, None, None
    red_cost, _ = dp[1][zero_state]
    q_plan, leftover_plan = [], []
    cost = 0.0
    state = zero_state
    for t in range(1, T + 1):
        q, state_next, true = dp[t][state][1]
        q_plan.append(q)
        leftover_plan.append(state_next)
        cost += true
        state = state_next
    return red_cost, cost, q_plan, leftover_plan


# -----------------------------------------------------------------------
#   Gurobi restricted master
# -----------------------------------------------------------------------
try:
    import gurobipy as grb
except ImportError:
    sys.exit("Please install gurobipy and ensure a valid licence.")


class MasterModel:
    def __init__(self, items, T, capacity):
        self.items, self.T, self.capacity = items, T, capacity
        self.model = grb.Model("RMP")
        self.model.Params.OutputFlag = 0
        self.lambda_vars = {i: [] for i in items}
        self.patterns = {i: [] for i in items}
        le = grb.LinExpr
        self.sel_constr = {
            i: self.model.addConstr(le() == 1.0, name=f"sel_{i}") for i in items
        }
        self.cap_constr = [
            self.model.addConstr(le() <= capacity[t], name=f"cap_{t}") for t in range(T)
        ]
        self.order_rows = {}  # (i,t) -> constraint

    def add_pattern(
        self,
        i: int,
        cost: float,
        q: list[int],
        order_fix: dict[tuple[int, int], tuple[int, int]],
    ):

        if any(p["q"] == q for p in self.patterns[i]):
            return  # Skip duplicate pattern

        delta = [1 if qty > 0 else 0 for qty in q]
        col = grb.Column()
        col.addTerms(1.0, self.sel_constr[i])
        for t, val in enumerate(q):
            if val:
                col.addTerms(val, self.cap_constr[t])
        for (ii, tt), (lb, ub) in order_fix.items():
            if (ii, tt) not in self.order_rows:
                sense = (
                    ">" if lb == 1 and ub == 1 else "<" if lb == 0 and ub == 0 else ">"
                )
                rhs = lb if lb == ub else 1
                expr = grb.LinExpr()
                if sense == ">":
                    constraint = expr >= rhs
                elif sense == "<":
                    constraint = expr <= rhs
                else:
                    raise ValueError("Invalid sense for constraint")
                self.order_rows[(ii, tt)] = self.model.addConstr(
                    constraint, name=f"ord_{ii}_{tt}"
                )
            constr = self.order_rows[(ii, tt)]
            if delta[tt]:
                col.addTerms(delta[tt], constr)
        v = self.model.addVar(
            obj=cost, column=col, name=f"λ_{i}_{len(self.lambda_vars[i])}"
        )
        self.lambda_vars[i].append(v)
        self.patterns[i].append(dict(cost=cost, q=q, delta=delta))
        print(f"[ADD] item {i} col#{len(self.lambda_vars[i])-1} cost={cost:.2f}")

    def optimize(self):
        self.model.optimize()
        if self.model.Status == grb.GRB.INFEASIBLE:
            return None
        if self.model.Status != grb.GRB.OPTIMAL:
            raise RuntimeError("Unexpected status")
        pi = {i: self.sel_constr[i].Pi for i in self.items}
        mu = [c.Pi for c in self.cap_constr]
        lam = {i: [v.X for v in self.lambda_vars[i]] for i in self.items}
        return self.model.ObjVal, pi, mu, lam

    def copy(self, order_fix=None):
        clone = MasterModel(self.items, self.T, self.capacity)
        for i in self.items:
            for pat in self.patterns[i]:
                clone.add_pattern(i, pat["cost"], pat["q"], {})
        if order_fix:
            for (ii, tt), (lb, ub) in order_fix.items():
                sense = (
                    ">" if lb == 1 and ub == 1 else "<" if lb == 0 and ub == 0 else ">"
                )
                rhs = lb if lb == ub else 1
                expr = grb.LinExpr()
                if sense == ">":
                    constr = clone.model.addConstr(expr >= rhs, name=f"ord_{ii}_{tt}")
                elif sense == "<":
                    constr = clone.model.addConstr(expr <= rhs, name=f"ord_{ii}_{tt}")
                else:
                    raise ValueError("Invalid sense for constraint")
                clone.order_rows[(ii, tt)] = constr
                for idx, pat in enumerate(clone.patterns[ii]):
                    delta_tt = pat["delta"][tt]
                    if delta_tt:
                        v = clone.lambda_vars[ii][idx]
                        clone.model.chgCoeff(constr, v, delta_tt)
        clone.model.update()
        return clone


# -----------------------------------------------------------------------
#  Branch-and-price driver
# -----------------------------------------------------------------------
class BranchPrice:
    def __init__(self, demand, c_var, h, setup, b_var, capacity, shelf, k_max):
        self.items = list(demand)
        self.dem, self.c_var, self.h = demand, c_var, h
        self.setup, self.b_var = setup, b_var
        self.cap, self.T = capacity, len(capacity)
        self.shelf, self.k_max = shelf, k_max
        self.order_fix = {}  # (i,t)->(LB,UB)
        self.prev_mu = [0.0] * self.T  # dual stabilisation
        self.alpha = 0.6
        self.master = MasterModel(self.items, self.T, self.cap)
        for i in self.items:
            q = self.dem[i]
            cost = sum(c_var[i] * q_t for q_t in q) + setup[i] * sum(
                1 for q_t in q if q_t > 0
            )
            self.master.add_pattern(i, cost, q, self.order_fix)
        self.master.model.update()
        self.tree = {}
        self.node_counter = 0
        self.parent_stack = []

    def log_node(self, parent, fix, obj, incumbent, status):
        self.tree[self.node_counter] = {
            "parent": parent,
            "fix": fix,
            "obj": obj,
            "incumbent": incumbent,
            "status": status,
        }
        self.parent_stack.append(self.node_counter)
        self.node_counter += 1
        return self.node_counter - 1

    def compute_y(self, lam):
        y = {(i, t): 0.0 for i in self.items for t in range(self.T)}
        for i, vlist in lam.items():
            for idx, v in enumerate(vlist):
                q = self.master.patterns[i][idx]["q"]
                for t, qty in enumerate(q):
                    if qty:
                        y[(i, t)] += v
        return y

    def column_generation(self):
        while True:
            res = self.master.optimize()
            if res is None:
                print(f"[FAIL] Master problem infeasible. No patterns for some item.")
                return float("inf"), (None, None, None)  # instead of just float("inf")
            obj, pi, mu, lam = res
            print(
                f"[CG] iter {len(self.master.lambda_vars[self.items[0]])}  obj={obj:.2f}"
            )

            mu_hat = [
                self.alpha * m + (1 - self.alpha) * p for m, p in zip(mu, self.prev_mu)
            ]
            self.prev_mu = mu_hat
            added = False
            for i in self.items:
                red, cost, q = self.price_with_growth(i, mu_hat, pi[i])
                if red is not None and q is not None and red < -1e-6:
                    before_len = len(self.master.lambda_vars[i])
                    self.master.add_pattern(i, cost, q, self.order_fix)
                    if len(self.master.lambda_vars[i]) > before_len:
                        print(f"    ↳ new col for item {i}  rc={red:.2f}")
                        added = True
                    else:
                        print(
                            f"    ↳ duplicate col for item {i}  rc={red:.2f} (skipped)"
                        )
            if not added:
                break
        return obj, (pi, mu, lam)  # ← obj is the current LP bound

    def branch_and_price(self, best=float("inf"), best_sol=None):
        bound, (pi, mu, lam) = self.column_generation()
        self._print_gap(bound, best)  # <── NEW
        node_id = self.log_node(
            parent=self.parent_stack[-1] if self.parent_stack else None,
            fix=None if not self.order_fix else list(self.order_fix.items())[-1],
            obj=bound,
            incumbent=best,
            status="branching" if bound < best else "pruned",
        )
        if bound >= best - 1e-6:
            return best, best_sol
        res = self.master.optimize()
        if res is None:
            print(f"[FAIL] Final master problem infeasible.")
            return float("inf"), None
        _, _, _, lam = res
        for i in self.items:
            if not self.master.patterns[i]:
                print(f"[WARN] No patterns generated for item {i}")
        y = self.compute_y(lam)
        frac = None
        for (i, t), val in y.items():
            if 1e-6 < val < 1 - 1e-6:
                frac = (i, t)
                print(f"[BRANCH] depth?  frac y[{i},{t}]={val:.3f}")
                break
        if frac is None:
            print(f"[SOL] incumbent {bound:.2f}")
            self._print_gap(bound, bound, prefix="    ")
            return bound, lam
        i_b, t_b = frac
        best, best_sol = self.branch_child(i_b, t_b, (0, 0), best, best_sol)
        best, best_sol = self.branch_child(i_b, t_b, (1, 1), best, best_sol)
        self.parent_stack.pop()
        return best, best_sol

    def branch_child(self, i, t, fix, best, best_sol):
        print(f"    |-- create child  fix y[{i},{t}]={fix}")
        child = deepcopy(self)
        child.order_fix[(i, t)] = fix
        child.master = child.master.copy(order_fix=child.order_fix)
        child.parent_stack = self.parent_stack.copy()
        child.tree = self.tree  # shared reference
        child.node_counter = self.node_counter  # preserve counter
        return child.branch_and_price(best, best_sol)

    def fallback_heuristic_pattern(
        self, i: int, mu: list[float], pi_i: float
    ) -> tuple[float, float, list[int]]:
        """Generate a fallback greedy pattern if DP explodes."""
        demand_i = self.dem[i]
        shelf = self.shelf[i]
        T = len(demand_i)
        q = [0] * T
        inventory = [0] * shelf
        total_cost = 0.0

        for t in range(T):
            d = demand_i[t]
            available = sum(inventory)
            shortage = max(0, d - available)
            order = shortage  # naive greedy: order just enough
            q[t] = order

            # Update inventory
            inventory = [order] + inventory[:-1]
            consumed = d
            for age in reversed(range(shelf)):
                use = min(inventory[age], consumed)
                inventory[age] -= use
                consumed -= use

            holding_cost = self.h[i] * sum(inventory)
            setup_cost = self.setup[i] if order > 0 else 0
            var_cost = self.c_var[i] * order
            back_cost = self.b_var[i] * consumed  # if unmet

            total_cost += holding_cost + setup_cost + var_cost + back_cost

        # Reduced cost: use duals (mu and pi_i)
        red = total_cost - sum(mu[t] * q[t] for t in range(T)) - pi_i
        return red, total_cost, q

    def price_with_growth(self, i, mu_hat, pi_i, k_start=8):
        """Try small k; double until bound no longer active or hard cap reached."""
        # fallback heuristic pattern before full DP if mu is positive
        heur = self.fallback_heuristic_pattern(i, mu_hat, pi_i)
        if heur is not None:
            rc, cost_heur, q_heur = heur
            if all(p["q"] != q_heur for p in self.master.patterns[i]):
                self.master.add_pattern(i, cost_heur, q_heur, self.order_fix)
                if self.verbose:
                    print(
                        f"[HEUR] item {i} -> fallback added with rc={rc:.2f} cost={cost_heur:.2f}"
                    )

        k = max(4, min(k_start, self.k_max[i]))  # conservative starting point
        while True:

            red, cost, q = price_shortest_path(
                i,
                self.dem[i],
                self.c_var[i],
                self.h[i],
                self.setup[i],
                mu_hat,
                pi_i,
                self.shelf[i],
                self.k_max[i],
                self.order_fix,
            )

            # for using dp_pricing_general_pulsed
            # red, cost, q, _ = dp_pricing_general_pulsed(
            #     i,
            #     self.dem[i],
            #     self.c_var[i],
            #     self.h[i],
            #     self.setup[i],
            #     self.b_var[i],
            #     mu_hat,
            #     pi_i,
            #     self.shelf[i],
            #     k_max=k,
            #     order_fix=self.order_fix,
            #     allow_backorder=ALLOW_BACKORDER,
            #     dbg=True,
            # )
            if q is None:
                if k >= HARD_KMAX_CAP:
                    print(
                        f"[FAIL] Item {i} could not find a feasible pattern even at k={k}"
                    )
                    print(f"[HEUR] Fallback pattern used for item {i}")
                    return self.fallback_heuristic_pattern(i, mu_hat, pi_i)
                k *= 2
                continue
            if any(q_t == k for q_t in q) and k < HARD_KMAX_CAP:
                k *= 2  # cap was tight, enlarge
                continue
            return red, cost, q

        # ── pretty printer -------------------------------------------------

    @staticmethod
    def _print_gap(bound: float, incumbent: float, prefix: str = "") -> None:
        if incumbent < float("inf"):
            gap = 100.0 * (incumbent - bound) / incumbent
            print(
                f"{prefix}[GAP] bound={bound:.2f}  best={incumbent:.2f}  "
                f"gap={gap:.2f}%"
            )
        else:  # no incumbent yet
            print(f"{prefix}[GAP] bound={bound:.2f}  best=∞  gap=∞")

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "master":
                result.master = self.master.copy()
            else:
                setattr(result, k, deepcopy(v, memo))
        return result


def detailed_exec_rep(bp, sol):
    print("\n=== Detailed execution report ===")
    for i in bp.items:
        print(f"\nItem {i}")
        try:
            sel_idx = next(idx for idx, v in enumerate(sol[i]) if v > 0.9)
        except StopIteration:
            print(f"[ERROR] No selected pattern for item {i} — skipping.")
            continue

        if sel_idx >= len(bp.master.patterns[i]):
            print(f"[ERROR] Pattern index {sel_idx} out of bounds for item {i}")
            continue

        orders = bp.master.patterns[i][sel_idx]["q"]
        demand_i = bp.dem[i]
        shelf = bp.shelf[i]
        inv = [0] * shelf
        backorder = 0

        print(" t | dem | ord | inv+ | back")
        print("-" * 27)
        for t in range(bp.T):
            on_hand_today = sum(inv)
            inv_new = [0] * shelf
            inv_new[1:] = inv[:-1]
            inv_new[0] += orders[t]
            sell = min(demand_i[t] + backorder, sum(inv_new))
            remaining = demand_i[t] + backorder - sell
            for age in range(shelf - 1, -1, -1):
                use = min(inv_new[age], sell)
                inv_new[age] -= use
                sell -= use
            backorder = remaining
            print(
                f"{t:2d} | {demand_i[t]:3d} | {orders[t]:3d} | {sum(inv_new):4d} | {backorder:4d}"
            )
            inv = inv_new


@dataclass
class Item:
    id: int
    demand: List[int]
    setup: float
    b_var: float
    c_var: float
    h: float
    shelf: int


@dataclass
class Lot:
    period: int
    capacity_pad: int
    items: Dict[int, Item] = field(default_factory=dict)
    manual_capacity: List[int] = field(default=None)

    # -------- convenience properties -------------
    @property
    def capacity(self) -> List[int]:
        if self.manual_capacity is not None:
            return self.manual_capacity

        cap_raw = [
            sum(it.demand[t] for it in self.items.values()) for t in range(self.period)
        ]
        max_cap = max(cap_raw)
        buffer = max(5, int(0.2 * max_cap))  # add at least 5 units or 20%
        return [c + buffer for c in cap_raw]

    @property
    def kmax(self) -> Dict[int, int]:
        max_dem = max(max(it.demand) for it in self.items.values())
        buffer = max(5, int(0.2 * max_dem))
        return {i: max(it.demand) + buffer for i, it in self.items.items()}

    # -------- factory-like helper ----------------
    def to_dicts(self):
        """Convert into the dict inputs expected by BranchPrice (legacy)."""
        demand = {i: it.demand for i, it in self.items.items()}
        setup = {i: it.setup for i, it in self.items.items()}
        b_var = {i: it.b_var for i, it in self.items.items()}
        c_var = {i: it.c_var for i, it in self.items.items()}
        h = {i: it.h for i, it in self.items.items()}
        shelf = {i: it.shelf for i, it in self.items.items()}
        cap = self.capacity
        kmax = self.kmax
        return demand, c_var, h, setup, b_var, cap, shelf, kmax

        # ---- convenience helpers --------------------------------------------

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serial = asdict(self)
        serial["items"] = {str(k): v for k, v in serial["items"].items()}
        if self.manual_capacity is None:
            serial.pop("manual_capacity", None)  # remove if not used
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


# -----------------------------------------------------------------------
#  Build instance + run
# -----------------------------------------------------------------------
def build_lot(
    period: int = 10,
    lb_dem: int = 1,
    ub_dem: int = 7,
    capacity_pad: int = 14,
    specs: List[tuple] = None,
    manual_capacity: List[int] = None,
) -> Lot:
    """Return a Lot object populated with x random items."""
    random.seed(0)
    if specs is None:
        specs = []

    lot = Lot(period=period, capacity_pad=capacity_pad, manual_capacity=manual_capacity)

    for idx, stp, b, c, hold, sh in specs:
        demand = [random.randint(lb_dem, ub_dem) for _ in range(period)]
        lot.items[idx] = Item(idx, demand, stp, b, c, hold, sh)

    return lot


def visualize_bnp_tree(tree: list):
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
        label += f"obj={obj:.1f}\n"
        label += f"inc={inc if inc < float('inf') else 'inf'}\n"
        if inc < float("inf"):
            label += f"gap={gap:.1f}%"

        G.add_node(node_id, label=label)

        parent_id = node.get("parent")
        if parent_id is not None:
            G.add_edge(parent_id, node_id)

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")
    except:
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


def wrapper_main():
    # 3️⃣  Solve with Branch-and-Price
    demand, c_var, h, setup, b_var, cap, shelf, k_max = lot.to_dicts()
    bp = BranchPrice(demand, c_var, h, setup, b_var, cap, shelf, k_max)

    t0 = time.perf_counter()
    best, sol = bp.branch_and_price()
    if sol is None:
        print("[❌ ERROR] No feasible solution — check your capacity vs demand.")
        for t in range(lot.period):
            total_d = sum(it.demand[t] for it in lot.items.values())
            print(
                f"Period {t:2d} → Total demand: {total_d}, Capacity: {lot.capacity[t]}"
            )
        sys.exit(1)
    elapsed = time.perf_counter() - t0

    print(f"\nObjective: {best:.2f}   (elapsed {elapsed:.2f} s)\n")
    # final MIP gap with respect to LP bound at root
    root_bound = bp.tree[0]["obj"] if bp.tree else best
    BranchPrice._print_gap(root_bound, best, prefix="[FINAL] ")
    print(f"[INFO] Backorders allowed: {ALLOW_BACKORDER}")

    for i in bp.items:
        try:
            sel = next(idx for idx, v in enumerate(sol[i]) if v > 0.9)
            patterns_i = bp.master.patterns[i]
            if sel >= len(patterns_i):
                print(
                    f"[WARN] Pattern index {sel} out of range for item {i}. Skipping."
                )
                continue
            print(f"Item {i}: pattern {sel}, orders={patterns_i[sel]['q']}")
        except (StopIteration, IndexError):
            print(f"[ERROR] No valid pattern found for item {i}, likely infeasible.")

    detailed_exec_rep(bp, sol)

    for i in bp.items:
        sel = next(idx for idx, v in enumerate(sol[i]) if v > 0.9)
        orders = bp.master.patterns[i][sel]["q"]
        demand_i = bp.dem[i]
        shelf = bp.shelf[i]
        inv = [0] * shelf
        back = 0
        inv_hist, back_hist = [], []
        for t in range(bp.T):
            inv_new = [0] * shelf
            inv_new[1:] = inv[:-1]
            inv_new[0] += orders[t]
            sell = min(demand_i[t] + back, sum(inv_new))
            rem = demand_i[t] + back - sell
            back = rem
            inv = inv_new
            inv_hist.append(sum(inv))
            back_hist.append(back)

        if ENABLE_LIVE_PLOTS:
            live_plot_inventory(
                bp.T - 1, inv_hist, back_hist, i, {"inv": inv_hist, "back": back_hist}
            )

        if TREE_VISUALIZATION:
            visualize_bnp_tree(bp.tree)
            print(f"[INFO] Branch-and-Price tree visualized.")

        plt.ioff()
        plt.show()


# --------- core data objects ---------------------------------------------

if __name__ == "__main__":

    # ---- basic switches -------------------------------------------------
    RANDOMIZE = True  # → False to re-use the cached instance
    USE_MANUAL_CAPACITY = True  # ← switch here
    ALLOW_BACKORDER = False  # allow backorders in pricing

    INSTANCE_PATH = Path(__file__).with_name("last_instance.json")
    SEED = 0  # keeps random runs reproducible

    # --------------------------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)

    # 1️⃣  Load existing instance (only when RANDOMIZE is off and file exists)
    if (not RANDOMIZE) and INSTANCE_PATH.exists():
        lot = Lot.from_json(INSTANCE_PATH)
        print(f"[INFO] Loaded instance from {INSTANCE_PATH}")

    # 2️⃣  Otherwise build a fresh instance and overwrite the cache
    else:
        specs = [
            # id  setup  b_var  c_var  h    shelf
            (0, 7.5, 5.0, 2.0, 0.4, 3),
            (1, 29.0, 5.0, 3.0, 0.6, 4),
            # (2, 6.0, 5.0, 1.8, 0.3, 5),
            # (3, 8.0, 5.0, 2.5, 0.5, 4),
            # (4, 10.0, 5.0, 3.5, 0.7, 5),
            # (5, 12.0, 5.0, 4.0, 0.8, 2),
            # (6, 11.0, 5.0, 3.8, 0.75, 3),
            # (7, 13.0, 5.0, 4.2, 0.85, 5),
            # add more items here if you like
        ]
        period = 10  # number of periods in the lot
        manual_caps = [35] * period if USE_MANUAL_CAPACITY else None
        lot = build_lot(
            period=period,
            lb_dem=1,
            ub_dem=20,
            capacity_pad=10,
            specs=specs,
            manual_capacity=manual_caps,
        )
        lot.to_json(INSTANCE_PATH)
        print(f"[INFO] Generated new instance → {INSTANCE_PATH}")

    # 3️⃣  Solve with Branch-and-Price
    wrapper_main()
