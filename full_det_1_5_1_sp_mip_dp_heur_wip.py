"""
Advanced branch-and-price for a 3-item, 30-period perishable lot-sizing problem
with setup costs, holding costs, optional back-orders, dual stabilisation,
and order/no-order branching on (item,period).

Python ≥ 3.8, gurobipy installed, academic licence assumed.
"""

from __future__ import annotations
import itertools, random, sys
from copy import deepcopy
import numpy as np
import gurobipy as grb
import time, json
from pathlib import Path
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz
import concurrent.futures
from typing import List, Dict
from dataclasses import asdict, dataclass, field

# ────────────────────────────────────────────────────────────────────────────

# Configuration switches
ENABLE_LIVE_PLOTS = False
TREE_VISUALIZATION = False
RANDOMIZE = False  # False to re-use cached instance
USE_MANUAL_CAPACITY = True
ALLOW_BACKORDER = False
SEED = 0  # For reproducibility
INSTANCE_PATH = Path(__file__).with_name("last_instance.json")
DP_STATE_STEP = 250_000  # Print progress every 250k DP states
DP_TIMEOUT = 20.0  # Timeout for DP in seconds
MAX_DP_STATES = 2_000_000  # stop after two million states per item

if TREE_VISUALIZATION or ENABLE_LIVE_PLOTS:
    fig, ax = plt.subplots()
    plt.ion()
    plt.show(block=False)


# ────────────────────────────────────────────────────────────────────────────
# Plotting and Visualization
# ────────────────────────────────────────────────────────────────────────────
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
        label += f"obj={obj:.1f}\ninc={inc if inc < float('inf') else 'inf'}\n"
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


# ────────────────────────────────────────────────────────────────────────────
# Pricing Subproblems
# ────────────────────────────────────────────────────────────────────────────
def _arc_cost(demand, c_var, h, setup, mu_t, t: int, u: int) -> tuple[int, float]:
    """
    One order at period t covers demand up to and incl. u (t ≤ u).
    Returns (q, reduced_cost). Assumes u-t < shelf_life and q ≤ k_max.
    """
    q = sum(demand[t : u + 1])
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
    Layered DAG shortest path: nodes 0…T (T is sink), arc (t → u+1) = one order at t
    covering demand[t…u]. Returns (red_cost, true_cost, q_plan).
    No back-orders; respects k_max & order_fix.
    """
    T = len(demand)
    dist = [float("inf")] * (T + 1)
    succ = [None] * T
    dist[T] = -pi
    for t in range(T - 1, -1, -1):
        lb, ub = order_fix.get((item_id, t), (0, 1))
        if ub == 0:
            dist[t] = dist[t + 1]
            continue
        best = float("inf")
        best_arc = None
        q_acc = 0
        hold_acc = 0
        for u in range(t, min(T, t + shelf)):
            q_acc += demand[u]
            if q_acc > k_max:
                break
            hold_acc += demand[u] * (u - t)
            q, red_arc = q_acc, (c_var - mu[t]) * q_acc + h * hold_acc + setup
            cand = red_arc + dist[u + 1]
            if cand < best:
                best = cand
                best_arc = (u, q)
        dist[t] = best
        succ[t] = best_arc
    if dist[0] >= -1e-6:
        return float("inf"), None, None
    q_plan = [0] * T
    true_cost = 0.0
    t = 0
    while t < T:
        if succ[t] is None:
            t += 1
            continue
        u, q = succ[t]
        q_plan[t] = q
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
    """
    DP pricing with progress pulses. Tracks inventory ages, handles perishability.
    Returns (red_cost, cost, q_plan, None).
    """
    T, L = len(demand), shelf_life
    cum_demand = [0] * (T + 1)
    running = 0
    for τ in range(T - 1, -1, -1):
        running += demand[τ]
        cum_demand[τ] = running
    rng = range(-k_max, k_max + 1) if allow_backorder else range(k_max + 1)
    zero = tuple(0 for _ in range(L - 1))
    dp = [dict() for _ in range(T + 2)]
    dp[T + 1][zero] = (-pi, None)
    for t in range(T, 0, -1):
        state_cnt = 0
        d, mu_t = demand[t - 1], mu[t - 1]
        lb, ub = order_fix.get((item_id, t - 1), (0, 1))
        for state in itertools.product(rng, repeat=L - 1):
            state_cnt += 1
            if state_cnt >= MAX_DP_STATES:
                # abort – let the caller know we failed
                return float("inf"), None, None, None
            if dbg and state_cnt % DP_STATE_STEP == 0:
                print(
                    f"      [DP] item={item_id} t={t:2d} states={state_cnt//1_000:,}k"
                )
            avail = sum(max(x, 0) for x in state)
            if avail > cum_demand[t - 1] or avail > d + k_max:
                continue
            need = d - avail
            q_min = max(0, need, 1 if lb == 1 else 0)
            q_max = k_max if ub else 0
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
                age1 = q - use_q - rem
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
    inv_history, back_history = [], []
    for t in range(1, T + 1):
        q, st_next, true = dp[t][st][1]
        q_plan.append(q)
        cost += true
        st = st_next
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
    """
    General DP pricing. Tracks inventory ages, supports backorders.
    Returns (red_cost, cost, q_plan, leftover_plan).
    """
    T, L = len(demand), shelf_life
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


# ────────────────────────────────────────────────────────────────────────────
# Restricted Master Problem
# ────────────────────────────────────────────────────────────────────────────
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
        self.order_rows = {}

    def add_pattern(
        self,
        i: int,
        cost: float,
        q: list[int],
        order_fix: dict[tuple[int, int], tuple[int, int]],
    ):
        if any(p["q"] == q for p in self.patterns[i]):
            return
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
                constraint = expr >= rhs if sense == ">" else expr <= rhs
                constr = self.model.addConstr(constraint, name=f"ord_{ii}_{tt}")
                self.order_rows[(ii, tt)] = constr
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
                constraint = expr >= rhs if sense == ">" else expr <= rhs
                constr = clone.model.addConstr(constraint, name=f"ord_{ii}_{tt}")
                clone.order_rows[(ii, tt)] = constr
                for idx, pat in enumerate(clone.patterns[ii]):
                    delta_tt = pat["delta"][tt]
                    if delta_tt:
                        v = clone.lambda_vars[ii][idx]
                        clone.model.chgCoeff(constr, v, delta_tt)
        clone.model.update()
        return clone


# ────────────────────────────────────────────────────────────────────────────
# Branch-and-Price Driver
# ────────────────────────────────────────────────────────────────────────────
class BranchPrice:
    def __init__(self, demand, c_var, h, setup, b_var, capacity, shelf, k_max):
        self.items = list(demand)
        self.dem, self.c_var, self.h = demand, c_var, h
        self.setup, self.b_var = setup, b_var
        self.cap, self.T = capacity, len(capacity)
        self.shelf, self.k_max = shelf, k_max
        self.order_fix = {}
        self.prev_mu = [0.0] * self.T
        self.alpha = 0.6
        self.verbose = True
        self.master = MasterModel(self.items, self.T, self.cap)
        BIG_M = 1e6
        for i in self.items:
            q = self.dem[i]
            cost = sum(c_var[i] * q_t for q_t in q) + setup[i] * sum(
                1 for q_t in q if q_t > 0
            )
            self.master.add_pattern(i, cost, q, self.order_fix)
            q_dummy = [0] * self.T
            self.master.add_pattern(i, BIG_M, q_dummy, self.order_fix)
        self.master.model.update()
        self.tree = {}
        self.node_counter = 0
        self.parent_stack = []

    def log_node(self, parent, fix, obj, incumbent, status):
        self.tree[self.node_counter] = {
            "id": self.node_counter,
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

    # ----------------------------------------------------------------------

    def _get_mip_template(self, item_id: int):
        """
        Build the single-item MILP template (flow-balance version).
        The template is cached – only coefficients / bounds change later.
        """
        if hasattr(self, "_mip_cache") and item_id in self._mip_cache:
            return self._mip_cache[item_id]

        # ── instance data ───────────────────────────────────────────────
        T = len(self.dem[item_id])
        k_max = self.k_max[item_id]
        demand = self.dem[item_id]

        # ── model skeleton ──────────────────────────────────────────────
        m = grb.Model(f"price_item_{item_id}")
        m.Params.OutputFlag = 0
        m.Params.Presolve = 2
        m.Params.Method = 2  # barrier at root
        m.Params.Cuts = 2
        m.Params.Heuristics = 0.5
        m.Params.Threads = 1

        # ── variables ───────────────────────────────────────────────────
        q = m.addVars(T, vtype=grb.GRB.INTEGER, lb=0, ub=k_max, name="q")  # order
        y = m.addVars(T, vtype=grb.GRB.BINARY, name="y")  # setup flag
        s = m.addVars(T, vtype=grb.GRB.CONTINUOUS, lb=0, name="s")  # stock

        # ── balance:  s[t] = s[t-1] + q[t] − demand[t] ─────────────────
        bal_ctrs = []
        for t in range(T):
            prev = s[t - 1] if t else 0
            ctr = m.addConstr(prev + q[t] - s[t] == demand[t], name=f"bal_{t}")
            bal_ctrs.append(ctr)

            # exact setup/quantity linkage (tight, no big-M)
            m.addGenConstrIndicator(y[t], True, q[t] >= 1)
            m.addGenConstrIndicator(y[t], False, q[t] == 0)

        # ── cost parts independent of duals ─────────────────────────────
        setup_cost = self.setup[item_id] * y.sum()
        hold_cost = self.h[item_id] * s.sum()
        var_cost = self.c_var[item_id] * q.sum()

        m.setObjective(setup_cost + hold_cost + var_cost, sense=grb.GRB.MINIMIZE)
        m.update()

        # ── cache template ──────────────────────────────────────────────
        tpl = {
            "model": m,
            "q": q,
            "y": y,
            "s": s,
            "bal": bal_ctrs,
            "inv": s,
        }  # 'inv' kept for backward compatibility
        if not hasattr(self, "_mip_cache"):
            self._mip_cache = {}
        self._mip_cache[item_id] = tpl
        return tpl

    # ----------------------------------------------------------------------
    # here is the *pricing* routine that uses that template
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------

    def price_mip(self, item_id: int, mu, pi_i, order_fix):
        tpl = self._get_mip_template(item_id)
        m = tpl["model"]
        q, y = tpl["q"], tpl["y"]
        bal_ctrs = tpl["bal"]

        # ── make sure base_obj is present ─────────────────────────
        if "base_obj" not in tpl:
            tpl["base_obj"] = m.getObjective()
        base_obj = tpl["base_obj"]

        T = len(self.dem[item_id])
        dem = self.dem[item_id]

        # 1) update balance RHS
        for t in range(T):
            bal_ctrs[t].RHS = dem[t]

        # 2) branching fixes
        for t in range(T):
            lb, ub = order_fix.get((item_id, t), (0, 1))
            y[t].LB, y[t].UB = lb, ub

        # 3) fresh reduced-cost objective
        rc_expr = grb.quicksum((-mu[t]) * q[t] for t in range(T)) - pi_i
        m.setObjective(base_obj + rc_expr, sense=grb.GRB.MINIMIZE)

        m.Params.TimeLimit = 30
        m.optimize()

        if m.Status != grb.GRB.OPTIMAL or m.ObjVal >= -1e-6:
            return float("inf"), None, None

        q_plan = [int(round(q[t].X)) for t in range(T)]
        true_val = m.ObjVal + pi_i + sum(mu[t] * q_plan[t] for t in range(T))

        return m.ObjVal, true_val, q_plan

    def column_generation(self):
        while True:
            res = self.master.optimize()
            if res is None:
                print(f"[FAIL] Master problem infeasible. No patterns for some item.")
                return float("inf"), (None, None, None)
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
        return obj, (pi, mu, lam)

    def branch_and_price(self, best=float("inf"), best_sol=None):
        bound, (pi, mu, lam) = self.column_generation()
        self._print_gap(bound, best)
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
        child.tree = self.tree
        child.node_counter = self.node_counter
        return child.branch_and_price(best, best_sol)

    def fallback_heuristic_pattern(
        self, i: int, mu: list[float], pi_i: float
    ) -> tuple[float, float, list[int]]:
        """Generate a fallback greedy pattern if DP times out or fails."""
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
            order = shortage
            q[t] = order
            inventory = [order] + inventory[:-1]
            consumed = d
            for age in reversed(range(shelf)):
                use = min(inventory[age], consumed)
                inventory[age] -= use
                consumed -= use
            holding_cost = self.h[i] * sum(inventory)
            setup_cost = self.setup[i] if order > 0 else 0
            var_cost = self.c_var[i] * order
            back_cost = self.b_var[i] * consumed
            total_cost += holding_cost + setup_cost + var_cost + back_cost
        red = total_cost - sum(mu[t] * q[t] for t in range(T)) - pi_i
        return red, total_cost, q

        # ──────────────────────────────────────────────────────────────

    #  replace the whole method with this version
    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────
    #  price_with_growth  –  SP  →  DP  →  Heuristic
    #      now with extra “what’s-happening” prints
    # ──────────────────────────────────────────────────────────────
    def price_with_growth(self, i, mu_hat, pi_i, k_start: int = 8):
        """
        (1) shortest-path   →   (2) DP w/ heartbeat, adaptive k, timeout
                               →   (3) greedy fallback.
        Each stage prints a one-line status so you can follow the flow.
        """

    def price_with_growth(self, i, mu_hat, pi_i, k_start=8):
        # ── 0️⃣  is there any y-fix of type LB=UB=1 for this item? ─────────
        must_order_periods = {
            t
            for (ii, t), (lb, ub) in self.order_fix.items()
            if ii == i and lb == ub == 1
        }

        # ---------- 1)  SHORTEST-PATH --------------------------------------
        if not must_order_periods:  # safe to try the network model
            rc, cost, q = price_shortest_path(
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
            if q is not None and rc < -1e-6:
                print(f"  ✔  [SP] rc={rc:.2f}")
                return rc, cost, q

            print("  ⇢ no improving column")
            print(f"[SP]  item {i} …", end="", flush=True)
        else:
            print(f"[SP]  item {i} skipped (mandatory order present)")

        # ---------- 2)  MIP One item ---------------------------
        rc, cost, q = self.price_mip(i, mu_hat, pi_i, self.order_fix)

        if q is not None and rc < -1e-6:
            print(f"[MIP] item {i}  ✔  rc={rc:.2f}")
            return rc, cost, q
        print("[MIP] item {i}  ⇢ no improving column")

        # # ---------- 3)  DYNAMIC PROGRAMME ----------------------------
        # SHELF_MAX = max(self.shelf.values())
        # DEMAND_MAX = max(max(d) for d in self.dem.values())
        # HARD_KCAP = DEMAND_MAX * SHELF_MAX
        # k = max(4, min(self.k_max[i], k_start))

        # def run_dp(k_val):
        #     return dp_pricing_general_pulsed(
        #         i,
        #         self.dem[i],
        #         self.c_var[i],
        #         self.h[i],
        #         self.setup[i],
        #         self.b_var[i],
        #         mu_hat,
        #         pi_i,
        #         self.shelf[i],
        #         k_max=k_val,
        #         order_fix=self.order_fix,
        #         allow_backorder=ALLOW_BACKORDER,
        #         dbg=True,  # keeps the “[DP] item… states=…” pulses
        #     )

        # while k <= HARD_KCAP:
        #     print(f"[DP] item {i}  k={k} …", end="", flush=True)
        #     try:
        #         with concurrent.futures.ThreadPoolExecutor() as ex:
        #             red, cost, q, _ = ex.submit(run_dp, k).result(timeout=DP_TIMEOUT)
        #     except concurrent.futures.TimeoutError:
        #         print("  ⏳ timeout!")
        #         red, q = float("inf"), None
        #     else:
        #         if q is not None and red < -1e-6:
        #             print(f"  ✔  rc={red:.2f}")
        #             if any(q_t == k for q_t in q) and k < HARD_KCAP:
        #                 # k was binding → enlarge once and try again
        #                 k *= 2
        #                 continue
        #             return red, cost, q
        #         print("  ⇢ no column")

        #     k *= 2  # grow k and retry

        # print(f"[FAIL] DP gave up for item {i}  (k>{HARD_KCAP})")

        # ---------- 4)  GREEDY HEURISTIC -----------------------------
        print(f"[HEUR] item {i}  generating greedy pattern")
        red, cost, q = self.fallback_heuristic_pattern(i, mu_hat, pi_i)
        return red, cost, q

    @staticmethod
    def _print_gap(bound: float, incumbent: float, prefix: str = "") -> None:
        if incumbent < float("inf"):
            gap = 100.0 * (incumbent - bound) / incumbent
            print(
                f"{prefix}[GAP] bound={bound:.2f}  best={incumbent:.2f}  gap={gap:.2f}%"
            )
        else:
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


# ────────────────────────────────────────────────────────────────────────────
# Execution Report
# ────────────────────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────────────────────────
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

    @property
    def capacity(self) -> List[int]:
        if self.manual_capacity is not None:
            return self.manual_capacity
        cap_raw = [
            sum(it.demand[t] for it in self.items.values()) for t in range(self.period)
        ]
        max_cap = max(cap_raw)
        buffer = max(5, int(0.2 * max_cap))
        return [c + buffer for c in cap_raw]

    @property
    def kmax(self) -> Dict[int, int]:
        max_dem = max(max(it.demand) for it in self.items.values())
        buffer = max(5, int(0.2 * max_dem))
        return {i: sum(it.demand) + buffer for i, it in self.items.items()}

    def to_dicts(self):
        demand = {i: it.demand for i, it in self.items.items()}
        setup = {i: it.setup for i, it in self.items.items()}
        b_var = {i: it.b_var for i, it in self.items.items()}
        c_var = {i: it.c_var for i, it in self.items.items()}
        h = {i: it.h for i, it in self.items.items()}
        shelf = {i: it.shelf for i, it in self.items.items()}
        cap = self.capacity
        kmax = self.kmax
        return demand, c_var, h, setup, b_var, cap, shelf, kmax

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


# ────────────────────────────────────────────────────────────────────────────
# Instance Generation and Main
# ────────────────────────────────────────────────────────────────────────────
def build_lot(
    period: int = 10,
    lb_dem: int = 1,
    ub_dem: int = 7,
    capacity_pad: int = 14,
    specs: List[tuple] = None,
    manual_capacity: List[int] = None,
) -> Lot:
    random.seed(0)
    if specs is None:
        specs = []
    lot = Lot(period=period, capacity_pad=capacity_pad, manual_capacity=manual_capacity)
    for idx, stp, b, c, hold, sh in specs:
        demand = [random.randint(lb_dem, ub_dem) for _ in range(period)]
        lot.items[idx] = Item(idx, demand, stp, b, c, hold, sh)
    return lot


def wrapper_main():
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
    print(f"\nObjective: {best:.2f}   (elapsed {elapsed:.2f} s)\n")


if __name__ == "__main__":

    random.seed(SEED)
    np.random.seed(SEED)
    if (not RANDOMIZE) and INSTANCE_PATH.exists():
        lot = Lot.from_json(INSTANCE_PATH)
        print(f"[INFO] Loaded instance from {INSTANCE_PATH}")
    else:
        specs = [
            (0, 127.5, 5.0, 2.0, 0.4, 10),
            (1, 29.0, 5.0, 3.0, 0.6, 4),
            (2, 50.0, 5.0, 1.0, 0.3, 6),
            (3, 100.0, 5.0, 4.0, 0.8, 8),
            (4, 80.0, 5.0, 2.5, 0.5, 12),
            (5, 60.0, 5.0, 3.5, 0.7, 14),
            (6, 90.0, 5.0, 1.5, 0.2, 16),
            (7, 110.0, 5.0, 2.0, 0.4, 18),
            (8, 70.0, 5.0, 3.0, 0.6, 20),
            (9, 40.0, 5.0, 1.0, 0.3, 22),
            (10, 30.0, 5.0, 4.0, 0.8, 24),
            (11, 20.0, 5.0, 2.5, 0.5, 26),
            (12, 10.0, 5.0, 3.5, 0.7, 28),
            (13, 15.0, 5.0, 1.5, 0.2, 30),
            (14, 25.0, 5.0, 2.0, 0.4, 32),
            (15, 35.0, 5.0, 3.0, 0.6, 34),
            (16, 45.0, 5.0, 1.0, 0.3, 36),
        ]
        period = 200
        manual_caps = [2000] * period if USE_MANUAL_CAPACITY else None
        lot = build_lot(
            period=period,
            lb_dem=1,
            ub_dem=100,
            capacity_pad=10,
            specs=specs,
            manual_capacity=manual_caps,
        )
        lot.to_json(INSTANCE_PATH)
        print(f"[INFO] Generated new instance → {INSTANCE_PATH}")
    wrapper_main()
