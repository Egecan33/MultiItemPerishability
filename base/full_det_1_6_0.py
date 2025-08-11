"""
Advanced branch-and-price for a multi-item, perishable lot-sizing problem
with setup costs, holding costs, optional back-orders, dual stabilisation,
and order/no-order branching on (item,period).

This version includes a FEFO-based dynamic program (DP) for the pricing
subproblem following Önal et al. (2015), Algorithm 2 and recurrences (9)–(12),
adapted to fixed shelf life L via v_t = min(T-1, t+L-1), and integrated with
dual adjustments for column generation.

Python ≥ 3.8, gurobipy installed, academic licence assumed.
"""

from __future__ import annotations
import itertools, random, sys, math
from copy import deepcopy
import numpy as np
import gurobipy as grb
import time, json
from pathlib import Path
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz  # optional for tree viz
import concurrent.futures
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict, dataclass, field

# ────────────────────────────────────────────────────────────────────────────
# Configuration switches
# ────────────────────────────────────────────────────────────────────────────
ENABLE_LIVE_PLOTS = False
TREE_VISUALIZATION = False
RANDOMIZE = True  # False to re-use cached instance
USE_MANUAL_CAPACITY = True
ALLOW_BACKORDER = False
SEED = 0  # For reproducibility
INSTANCE_PATH = Path(__file__).with_name("last_instance.json")
DP_STATE_STEP = 250_000  # legacy DP pulse (not used by FEFO DP)
DP_TIMEOUT = 20.0  # timeout for legacy DP
MAX_DP_STATES = 2_000_000

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


# ────────────────────────────────────────────────────────────────────────────
# Simple shortest-path pricing (no backorders)
# ────────────────────────────────────────────────────────────────────────────
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
) -> tuple[float, float, list[int] | None]:
    """
    Layered DAG shortest path: nodes 0…T (T is sink), arc (t → u+1) = one order at t
    covering demand[t…u]. Returns (reduced_cost, true_cost, q_plan) if improving.
    Respects k_max, shelf life, and only forbids periods with ub==0. Does NOT enforce
    lb==1; caller should skip this routine when any lb==1 fixes exist.
    """
    T = len(demand)
    dist = [float("inf")] * (T + 1)
    next_arc: list[Optional[Tuple[int, int]]] = [None] * T
    dist[T] = -pi  # dual of selection row

    forbidden = {
        tt for (ii, tt), (lb, ub) in order_fix.items() if ii == item_id and ub == 0
    }

    for t in range(T - 1, -1, -1):
        if t in forbidden:
            dist[t] = dist[t + 1]  # can't order, just pass-through
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
            red_arc = (c_var - mu[t]) * q_acc + h * hold_acc + (setup if q_acc else 0)
            cand = red_arc + dist[u + 1]
            if cand < best:
                best = cand
                best_arc = (u, q_acc)
        dist[t] = best
        next_arc[t] = best_arc

    if dist[0] >= -1e-6:
        return float("inf"), float("inf"), None

    # reconstruct
    q_plan = [0] * T
    true_cost = 0.0
    t = 0
    while t < T:
        arc = next_arc[t]
        if arc is None:
            t += 1
            continue
        u, q = arc
        q_plan[t] = q
        hold = sum(demand[τ] * (τ - t) for τ in range(t, u + 1))
        true_cost += setup + c_var * q + h * hold
        t = u + 1
    red_cost = dist[0]
    return red_cost, true_cost, q_plan


# ────────────────────────────────────────────────────────────────────────────
# Restricted Master Problem (Column Generation Master)
# ────────────────────────────────────────────────────────────────────────────
try:
    import gurobipy as grb  # noqa: F401
except ImportError:
    sys.exit("Please install gurobipy and ensure a valid licence.")


class MasterModel:
    """
    RMP structure:
      - λ_{i,p} variables select pattern p for item i
      - selection constraints: sum_p λ_{i,p} = 1
      - capacity constraints: sum_i sum_p q_{i,p,t} λ_{i,p} ≤ cap_t
      - branching constraints (order/no-order on y_{i,t}) added as:
          * fix to 1: sum_p δ_{i,p,t} λ_{i,p} ≥ 1
          * fix to 0: sum_p δ_{i,p,t} λ_{i,p} ≤ 0
      where δ_{i,p,t} = 1 if pattern p orders at (i,t), else 0
    """

    def __init__(self, items, T, capacity):
        self.items, self.T, self.capacity = items, T, capacity
        self.model = grb.Model("RMP")
        self.model.Params.OutputFlag = 0
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

    def _ensure_order_row(self, ii: int, tt: int, lb: int, ub: int):
        """
        Only creates a row when it's a true fix (lb==ub). For lb=ub=1, row is ≥ 1.
        For lb=ub=0, row is ≤ 0.
        """
        if lb != ub:
            return  # not a fix, ignore
        key = (ii, tt)
        if key in self.order_rows:
            return
        expr = grb.LinExpr()
        if lb == 1:
            constr = self.model.addConstr(expr >= 1.0, name=f"ord_1_{ii}_{tt}")
        else:
            constr = self.model.addConstr(expr <= 0.0, name=f"ord_0_{ii}_{tt}")
        self.order_rows[key] = constr

    def add_pattern(self, i: int, cost: float, q: list[int]):
        """
        Adds a column (pattern) for item i with true cost 'cost' and quantity vector q.
        Delta is inferred (1 if q_t>0). Existing branching rows (if any) get coefficients.
        """
        # avoid duplicates
        if any(p["q"] == q for p in self.patterns[i]):
            return
        delta = [1 if qty > 0 else 0 for qty in q]
        col = grb.Column()
        # selection row
        col.addTerms(1.0, self.sel_constr[i])
        # capacity rows
        for t, qty in enumerate(q):
            if qty:
                col.addTerms(qty, self.cap_constr[t])
        # existing branching rows (if any)
        for (ii, tt), row in self.order_rows.items():
            if ii == i and delta[tt]:
                col.addTerms(delta[tt], row)

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
            raise RuntimeError("Unexpected status in RMP")
        pi = {i: self.sel_constr[i].Pi for i in self.items}  # selection duals
        mu = [c.Pi for c in self.cap_constr]  # capacity duals
        lam = {i: [v.X for v in self.lambda_vars[i]] for i in self.items}
        return self.model.ObjVal, pi, mu, lam

    def copy(self, order_fix=None):
        """
        Deep copy the RMP into a fresh model, re-adding patterns and (optionally)
        adding branching rows for order_fix where lb==ub.
        """
        clone = MasterModel(self.items, self.T, self.capacity)
        # add all existing patterns
        for i in self.items:
            for pat in self.patterns[i]:
                clone.add_pattern(i, pat["cost"], pat["q"])
        # add branching rows if provided, and hook coefficients
        if order_fix:
            for (ii, tt), (lb, ub) in order_fix.items():
                if lb != ub:
                    continue
                clone._ensure_order_row(ii, tt, lb, ub)
            # now attach coefficients λ to newly created rows
            for (ii, tt), row in clone.order_rows.items():
                for idx, pat in enumerate(clone.patterns[ii]):
                    if pat["delta"][tt]:
                        v = clone.lambda_vars[ii][idx]
                        clone.model.chgCoeff(row, v, pat["delta"][tt])
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

        # branching fixes on y_{i,t} : (lb,ub) ∈ {(0,0),(1,1)} if fixed, else not present
        self.order_fix: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # dual stabilisation
        self.prev_mu = [0.0] * self.T
        self.alpha = 0.6
        self.verbose = True

        # master init with two seed patterns per item: "produce-to-demand" and "dummy zero"
        self.master = MasterModel(self.items, self.T, self.cap)
        BIG_M = 1e6
        for i in self.items:
            q = self.dem[i]  # produce demand exactly each period
            cost = (
                sum(self.c_var[i] * q_t for q_t in q)
                + self.setup[i] * sum(1 for q_t in q if q_t > 0)
                + self.h[i]
                * sum(
                    max(0, sum(q[: t + 1]) - sum(self.dem[i][: t + 1]))
                    for t in range(self.T)
                )
                * 0.0
            )
            self.master.add_pattern(i, cost, q)
            q_dummy = [0] * self.T
            self.master.add_pattern(i, BIG_M, q_dummy)
        self.master.model.update()

        # bookkeeping for tree viz
        self.tree = {}
        self.node_counter = 0
        self.parent_stack = []

    # ── logging helpers ────────────────────────────────────────────────────
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

    # ────────────────────────────────────────────────────────────────────
    # Single-item MIP pricing template (respects y fixes via bounds)
    # ────────────────────────────────────────────────────────────────────
    def _get_mip_template(self, item_id: int):
        if hasattr(self, "_mip_cache") and item_id in self._mip_cache:
            return self._mip_cache[item_id]

        T = len(self.dem[item_id])
        k_max = self.k_max[item_id]
        demand = self.dem[item_id]

        m = grb.Model(f"price_item_{item_id}")
        m.Params.OutputFlag = 0
        m.Params.Presolve = 2
        m.Params.Method = 2
        m.Params.Cuts = 2
        m.Params.Heuristics = 0.5
        m.Params.Threads = 1

        q = m.addVars(T, vtype=grb.GRB.INTEGER, lb=0, ub=k_max, name="q")
        y = m.addVars(T, vtype=grb.GRB.BINARY, name="y")
        s = m.addVars(T, vtype=grb.GRB.CONTINUOUS, lb=0, name="s")

        bal_ctrs = []
        for t in range(T):
            prev = s[t - 1] if t else 0
            ctr = m.addConstr(prev + q[t] - s[t] == demand[t], name=f"bal_{t}")
            bal_ctrs.append(ctr)
            # exact linkage
            m.addGenConstrIndicator(y[t], True, q[t] >= 1)
            m.addGenConstrIndicator(y[t], False, q[t] == 0)

        setup_cost = self.setup[item_id] * y.sum()
        hold_cost = self.h[item_id] * s.sum()
        var_cost = self.c_var[item_id] * q.sum()

        m.setObjective(setup_cost + hold_cost + var_cost, sense=grb.GRB.MINIMIZE)
        m.update()

        tpl = {"model": m, "q": q, "y": y, "s": s, "bal": bal_ctrs}
        if not hasattr(self, "_mip_cache"):
            self._mip_cache = {}
        self._mip_cache[item_id] = tpl
        return tpl

    def price_mip(self, item_id: int, mu, pi_i, order_fix):
        tpl = self._get_mip_template(item_id)
        m = tpl["model"]
        q, y = tpl["q"], tpl["y"]
        bal_ctrs = tpl["bal"]

        # keep base objective
        if "base_obj" not in tpl:
            tpl["base_obj"] = m.getObjective()
        base_obj = tpl["base_obj"]

        T = len(self.dem[item_id])
        dem = self.dem[item_id]

        # update balance RHS
        for t in range(T):
            bal_ctrs[t].RHS = dem[t]

        # branching fixes as bounds
        for t in range(T):
            lb, ub = order_fix.get((item_id, t), (0, 1))
            y[t].LB, y[t].UB = lb, ub

        # reduced cost objective
        rc_expr = grb.quicksum((-mu[t]) * q[t] for t in range(T)) - pi_i
        m.setObjective(base_obj + rc_expr, sense=grb.GRB.MINIMIZE)

        m.Params.TimeLimit = 30
        m.optimize()

        if m.Status != grb.GRB.OPTIMAL or m.ObjVal >= -1e-6:
            return float("inf"), None, None

        q_plan = [int(round(q[t].X)) for t in range(T)]
        true_val = m.ObjVal + pi_i + sum(mu[t] * q_plan[t] for t in range(T))
        return m.ObjVal, true_val, q_plan

    # ────────────────────────────────────────────────────────────────────
    # NEW: FEFO-based DP pricing from Önal et al. (2015) Algorithm 2
    # Uses adjusted per-unit costs ctilde_{t,i} = c_{t,i} - μ_i[t] and setup S
    # Returns an improving column if g(1,T) - π_i < 0
    # NOTE: only used when there are no lb==1 fixes for this item
    # ────────────────────────────────────────────────────────────────────
    def price_fefo_dp(
        self, item_id: int, mu: List[float], pi_i: float
    ) -> tuple[float, float, list[int] | None]:
        D = self.dem[item_id]
        T = len(D)
        if T == 0:
            return float("inf"), float("inf"), None

        L = self.shelf[item_id]
        v = [min(T - 1, t + L - 1) for t in range(T)]
        S = self.setup[item_id]
        cvar = self.c_var[item_id]
        h = self.h[item_id]

        # If there is any mandatory fix y=1, skip DP (handled by MIP)
        must_order = {
            tt
            for (ii, tt), (lb, ub) in self.order_fix.items()
            if ii == item_id and lb == 1 and ub == 1
        }
        if must_order:
            return float("inf"), float("inf"), None
        forbidden = {
            tt
            for (ii, tt), (lb, ub) in self.order_fix.items()
            if ii == item_id and ub == 0
        }

        # Prefix sums: sum D[a..b] and sum i*D[i] quickly
        prefD = [0] * (T + 1)
        prefAge = [0] * (T + 1)
        for i in range(T):
            prefD[i + 1] = prefD[i] + D[i]
            prefAge[i + 1] = prefAge[i] + D[i] * i

        def sub_true_cost(t: int, a: int, b: int) -> float:
            # S + sum_{i=a..b} (cvar + h*(i-t)) D[i]
            sumD = prefD[b + 1] - prefD[a]
            sumAge = prefAge[b + 1] - prefAge[a]
            return S + cvar * sumD + h * (sumAge - t * sumD)

        def sub_red_cost(t: int, a: int, b: int) -> float:
            # (true subplan cost) - μ[t] * sumD
            sumD = prefD[b + 1] - prefD[a]
            return sub_true_cost(t, a, b) - mu[t] * sumD

        # fmin[a][tau] = min_t sub_red_cost(t,a,tau) over feasible t
        fmin = [[float("inf")] * T for _ in range(T)]
        argt = [[-1] * T for _ in range(T)]
        for a in range(T - 1, -1, -1):
            for tau in range(a, T):
                # feasible t must satisfy: t ≤ a ≤ tau ≤ v[t]  ⇒ t ≥ tau - (L - 1)
                t_lo = max(0, tau - (L - 1))
                t_hi = a
                best = float("inf")
                best_t = -1
                for t in range(t_lo, t_hi + 1):
                    if t in forbidden:
                        continue
                    if tau > v[t]:
                        continue
                    val = sub_red_cost(t, a, tau)
                    if val < best:
                        best = val
                        best_t = t
                fmin[a][tau] = best
                argt[a][tau] = best_t

        # DP over intervals: g[a][b] = min_{tau in [a..b]} fmin[a][tau] + g[tau+1][b]
        g = [[float("inf")] * T for _ in range(T)]
        split_tau = [[-1] * T for _ in range(T)]
        for a in range(T):
            g[a][a - 1 if a > 0 else 0] = 0.0  # not used except base

        for length in range(T):  # 0..T-1
            for a in range(0, T - length):
                b = a + length
                best = float("inf")
                best_tau = -1
                for tau in range(a, b + 1):
                    left = fmin[a][tau]
                    if math.isinf(left):
                        continue
                    right = 0.0 if tau == b else g[tau + 1][b]
                    if math.isinf(right):
                        continue
                    cand = left + right
                    if cand < best:
                        best = cand
                        best_tau = tau
                g[a][b] = best
                split_tau[a][b] = best_tau

        if g[0][T - 1] >= pi_i - 1e-6:  # red_cost = g - pi
            return float("inf"), float("inf"), None

        # reconstruct q and true cost from subplans
        q = [0] * T
        true_cost = 0.0

        def reconstruct(a: int, b: int):
            nonlocal true_cost
            if a > b:
                return
            tau = split_tau[a][b]
            if tau < a:
                return
            t = argt[a][tau]
            if t < 0:
                return
            # order at t for sum D[a..tau]
            qty = prefD[tau + 1] - prefD[a]
            q[t] += qty
            true_cost += sub_true_cost(t, a, tau)
            reconstruct(tau + 1, b)

        reconstruct(0, T - 1)

        red_cost = g[0][T - 1] - pi_i
        return red_cost, true_cost, q

    # ────────────────────────────────────────────────────────────────────
    # Column generation loop
    # ────────────────────────────────────────────────────────────────────
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
                if q is not None and red < -1e-6:
                    before_len = len(self.master.lambda_vars[i])
                    self.master.add_pattern(i, cost, q)
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

    # ────────────────────────────────────────────────────────────────────
    # Branch-and-price recursion
    # ────────────────────────────────────────────────────────────────────
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

        # get RMP primal solution
        res = self.master.optimize()
        if res is None:
            print(f"[FAIL] Final master problem infeasible.")
            return float("inf"), None
        _, _, _, lam = res

        y = self.compute_y(lam)

        # choose a fractional y_{i,t} to branch on
        frac = None
        for (i, t), val in y.items():
            if 1e-6 < val < 1 - 1e-6:
                frac = (i, t)
                print(f"[BRANCH] frac y[{i},{t}]={val:.3f}")
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

    # ────────────────────────────────────────────────────────────────────
    # Pricing driver (SP → FEFO-DP → MIP → greedy)
    # ────────────────────────────────────────────────────────────────────
    def price_with_growth(self, i, mu_hat, pi_i, k_start=8):
        # 0) detect mandatory y==1 fix for this item
        must_order_periods = {
            t
            for (ii, t), (lb, ub) in self.order_fix.items()
            if ii == i and lb == ub == 1
        }

        # 1) shortest-path (only if no lb==1 fixes)
        if not must_order_periods:
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
            print(f"[SP]  item {i} ⇢ no improving column (or mandatory order present)")

        # 2) FEFO-based DP (skip if lb==1 exists, because DP cannot force y[t]=1)
        if not must_order_periods:
            red, cost, q = self.price_fefo_dp(i, mu_hat, pi_i)
            if q is not None and red < -1e-6:
                print(f"  ✔  [DP-FEFO] rc={red:.2f}")
                return red, cost, q
            print(f"[DP-FEFO] item {i} ⇢ no improving column")

        # 3) MIP pricing (respects both lb and ub fixes)
        rc, cost, q = self.price_mip(i, mu_hat, pi_i, self.order_fix)
        if q is not None and rc < -1e-6:
            print(f"[MIP] item {i}  ✔  rc={rc:.2f}")
            return rc, cost, q
        print(f"[MIP] item {i}  ⇢ no improving column")

        # 4) Greedy fallback (always returns something, may be non-improving)
        print(f"[HEUR] item {i}  generating greedy pattern")
        red, cost, q = self.fallback_heuristic_pattern(i, mu_hat, pi_i)
        return red, cost, q

    # ────────────────────────────────────────────────────────────────────
    # Greedy fallback (unchanged)
    # ────────────────────────────────────────────────────────────────────
    def fallback_heuristic_pattern(self, i: int, mu: list[float], pi_i: float):
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
                result.master = self.master.copy(order_fix=self.order_fix)
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
# Data Classes and Instance I/O
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
        ]
        period = 30
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
