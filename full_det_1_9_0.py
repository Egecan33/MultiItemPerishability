"""
Branch-and-price for a multi-item perishable lot-sizing problem
with setup costs, holding costs, optional back-orders, dual stabilisation,
and order/no-order branching on (item,period).
This version supports VARIABLE perishability m_{i,t} (per item & period).
Pricing order: Shortest-Path (SP) → FEFO DP → Perishability-aware MIP → Greedy.
Python ≥ 3.8, gurobipy installed, academic licence assumed.
"""

from __future__ import annotations
import random, sys, math
from copy import deepcopy
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict, dataclass, field
from pathlib import Path
import time, json
import numpy as np
import gurobipy as grb
import io, contextlib, os
from datetime import datetime

# Optional viz
import matplotlib.pyplot as plt
import networkx as nx

# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────
ENABLE_LIVE_PLOTS = False
TREE_VISUALIZATION = False
RANDOMIZE = True  # set False to reuse last_instance.json
USE_MANUAL_CAPACITY = True
ALLOW_BACKORDER = False  # DP/SP assume no backorders; MIP can model backorder if needed
SEED = 0
INSTANCE_PATH = Path(__file__).with_name("last_instance.json")
if TREE_VISUALIZATION or ENABLE_LIVE_PLOTS:
    fig, ax = plt.subplots()
    plt.ion()
    plt.show(block=False)


# ────────────────────────────────────────────────────────────────────────────
# Helpers (viz)
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
# Shortest path pricing (no backorders), supports variable m_{t}
# ────────────────────────────────────────────────────────────────────────────
def price_shortest_path(
    item_id: int,
    demand: List[int],
    c_var: float,
    h: float,
    setup: float,
    mu: List[float],
    pi: float,
    m_seq: List[int] | None,
    k_max: int,
    order_fix: Dict[Tuple[int, int], Tuple[int, int]],
) -> Tuple[float, float, Optional[List[int]]]:
    """
    Layered DAG shortest path: nodes 0..T; arc (t -> u+1) = one order at t covering demand[t..u].
    Feasible if u <= t + m_seq[t] - 1 (variable perishability) and cumulative qty ≤ k_max.
    Respects ub==0 (forbidden orders). Does NOT enforce lb==1; caller should skip SP if any lb==1.
    """
    T = len(demand)
    dist = [float("inf")] * (T + 1)
    nxt_arc: List[Optional[Tuple[int, int]]] = [None] * T
    dist[T] = -pi
    forbidden = {
        tt for (ii, tt), (lb, ub) in order_fix.items() if ii == item_id and ub == 0
    }
    for t in range(T - 1, -1, -1):
        if t in forbidden:
            dist[t] = dist[t + 1]
            continue
        best = float("inf")
        best_arc = None
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
            red_arc = (c_var - mu[t]) * q_acc + h * hold_acc + (setup if q_acc else 0)
            cand = red_arc + dist[u + 1]
            if cand < best:
                best = cand
                best_arc = (u, q_acc)
        dist[t] = best
        nxt_arc[t] = best_arc
    if dist[0] >= -1e-6:
        return float("inf"), float("inf"), None
    q = [0] * T
    true_cost = 0.0
    t = 0
    while t < T:
        arc = nxt_arc[t]
        if arc is None:
            t += 1
            continue
        u, qty = arc
        q[t] = qty
        hold = sum(demand[τ] * (τ - t) for τ in range(t, u + 1))
        true_cost += setup + c_var * qty + h * hold
        t = u + 1
    return dist[0], true_cost, q


# ────────────────────────────────────────────────────────────────────────────
# Master problem
# ────────────────────────────────────────────────────────────────────────────
class MasterModel:
    """
    RMP:
      - λ_{i,p} variables select pattern p for item i
      - selection: sum_p λ_{i,p} = 1
      - capacity: sum_i sum_p q_{i,p,t} λ_{i,p} ≤ cap_t
      - branching (fix y_{i,t}): if y=1 ⇒ sum_p δ_{i,p,t} λ_{i,p} ≥ 1
                                 if y=0 ⇒ sum_p δ_{i,p,t} λ_{i,p} ≤ 0
        where δ_{i,p,t} = 1 if pattern p orders at t (q_{i,p,t} > 0)
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

    def add_pattern(self, i: int, cost: float, q: List[int]):
        if any(p["q"] == q for p in self.patterns[i]):
            return
        y = [1 if qty > 0 else 0 for qty in q]  # explicit y vector inside column
        col = grb.Column()
        col.addTerms(1.0, self.sel_constr[i])
        for t, qty in enumerate(q):
            if qty:
                col.addTerms(qty, self.cap_constr[t])
        for (ii, tt), row in self.order_rows.items():
            if ii == i and y[tt]:
                col.addTerms(1.0, row)
        v = self.model.addVar(
            obj=cost, column=col, name=f"λ_{i}_{len(self.lambda_vars[i])}"
        )
        self.lambda_vars[i].append(v)
        self.patterns[i].append(dict(cost=cost, q=q, y=y))
        print(f"[ADD] item {i} col#{len(self.lambda_vars[i])-1} cost={cost:.2f}")

    def optimize(self):
        self.model.optimize()
        if self.model.Status == grb.GRB.INFEASIBLE:
            return None
        if self.model.Status != grb.GRB.OPTIMAL:
            raise RuntimeError("Unexpected RMP status")
        pi = {i: self.sel_constr[i].Pi for i in self.items}
        mu = [c.Pi for c in self.cap_constr]
        lam = {i: [v.X for v in self.lambda_vars[i]] for i in self.items}
        return self.model.ObjVal, pi, mu, lam

    def copy(self, order_fix=None):
        clone = MasterModel(self.items, self.T, self.capacity)
        for i in self.items:
            for pat in self.patterns[i]:
                clone.add_pattern(i, pat["cost"], pat["q"])
        if order_fix:
            for (ii, tt), (lb, ub) in order_fix.items():
                if lb != ub:  # we only add true fixes
                    continue
                clone._ensure_order_row(ii, tt, lb, ub)
            # hook existing columns to new rows
            for (ii, tt), row in clone.order_rows.items():
                for idx, pat in enumerate(clone.patterns[ii]):
                    if pat["y"][tt]:
                        v = clone.lambda_vars[ii][idx]
                        clone.model.chgCoeff(row, v, 1.0)
        clone.model.update()
        return clone


# ────────────────────────────────────────────────────────────────────────────
# Branch-and-Price driver
# ────────────────────────────────────────────────────────────────────────────
class BranchPrice:
    def __init__(self, demand, c_var, h, setup, b_var, capacity, shelf_seq, k_max):
        self.items = list(demand)
        self.dem, self.c_var, self.h = demand, c_var, h
        self.setup, self.b_var = setup, b_var
        self.cap, self.T = capacity, len(capacity)
        self.mseq = shelf_seq  # m_seq[i][t] = perishability of lot ordered at t
        self.k_max = k_max
        self.order_fix: Dict[Tuple[int, int], Tuple[int, int]] = {}  # y-fixes (lb,ub)
        self.prev_mu = [0.0] * self.T
        self.alpha = 0.6
        self.master = MasterModel(self.items, self.T, self.cap)
        BIG_M = 1e6
        # Seed columns: produce-to-demand & dummy zero
        for i in self.items:
            q = self.dem[i]
            cost = sum(self.c_var[i] * q_t for q_t in q) + self.setup[i] * sum(
                1 for q_t in q if q_t > 0
            )
            self.master.add_pattern(i, cost, q)
            self.master.add_pattern(i, BIG_M, [0] * self.T)
        self.master.model.update()
        self.tree = {}
        self.node_counter = 0
        self.parent_stack = []

    # logging
    def log_node(self, parent, fix, obj, incumbent, status):
        self.tree[self.node_counter] = dict(
            id=self.node_counter,
            parent=parent,
            fix=fix,
            obj=obj,
            incumbent=incumbent,
            status=status,
        )
        self.parent_stack.append(self.node_counter)
        self.node_counter += 1
        return self.node_counter - 1

    def compute_y(self, lam):
        y = {(i, t): 0.0 for i in self.items for t in range(self.T)}
        for i, vlist in lam.items():
            for idx, v in enumerate(vlist):
                yvec = self.master.patterns[i][idx]["y"]
                for t, bit in enumerate(yvec):
                    if bit:
                        y[(i, t)] += v
        return y

    # ────────────────────────────────────────────────────────────────────
    # Perishability-aware pricing MIP (time-expanded flows)
    # ────────────────────────────────────────────────────────────────────
    def price_mip_perishable(
        self, item_id: int, mu: List[float], pi_i: float
    ) -> Tuple[float, float, Optional[List[int]]]:
        D = self.dem[item_id]
        T = len(D)
        m_seq = self.mseq[item_id]
        kmax = self.k_max[item_id]
        m = grb.Model(f"price_per_{item_id}")
        m.Params.OutputFlag = 0
        m.Params.Presolve = 2
        m.Params.Method = 2
        m.Params.Cuts = 2
        m.Params.Heuristics = 0.5
        m.Params.Threads = 1
        q = m.addVars(T, vtype=grb.GRB.INTEGER, lb=0, ub=kmax, name="q")
        y = m.addVars(T, vtype=grb.GRB.BINARY, name="y")
        # x_{s,t}: serve demand at t from order at s (feasible only if s ≤ t ≤ s+m_seq[s]-1)
        x = {}
        for s in range(T):
            v_s = min(T - 1, s + m_seq[s] - 1)
            for t in range(s, v_s + 1):
                x[(s, t)] = m.addVar(
                    vtype=grb.GRB.CONTINUOUS, lb=0.0, name=f"x_{s}_{t}"
                )
        # demand satisfaction
        for t in range(T):
            m.addConstr(
                grb.quicksum(x[(s, t)] for s in range(t + 1) if (s, t) in x) == D[t],
                name=f"demand_{t}",
            )
        # flow capacity: shipped out of cohort s ≤ q[s]
        for s in range(T):
            out = grb.quicksum(x[(s, t)] for t in range(s, T) if (s, t) in x)
            m.addConstr(out <= q[s], name=f"cap_{s}")
        # setup linkage (tight)
        for s in range(T):
            m.addGenConstrIndicator(y[s], True, q[s] >= 1)
            m.addGenConstrIndicator(y[s], False, q[s] == 0)
        # objective: true cost − μ[s]*q[s] − π_i
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
        m.Params.TimeLimit = 30
        m.optimize()
        if m.Status != grb.GRB.OPTIMAL or m.ObjVal >= -1e-6:
            return float("inf"), None, None
        q_plan = [int(round(q[t].X)) for t in range(T)]
        # recover true cost (without duals)
        setup_true = self.setup[item_id] * sum(1 for t in range(T) if q_plan[t] > 0)
        var_true = self.c_var[item_id] * sum(q_plan)
        hold_true = sum(self.h[item_id] * (t - s) * x[(s, t)].X for (s, t) in x)
        true_val = setup_true + var_true + hold_true
        return m.ObjVal, true_val, q_plan

    # ────────────────────────────────────────────────────────────────────
    # Pricing driver
    # ────────────────────────────────────────────────────────────────────
    def price_with_growth(self, i: int, mu_hat: List[float], pi_i: float):
        must_order = {
            t
            for (ii, t), (lb, ub) in self.order_fix.items()
            if ii == i and lb == ub == 1
        }
        # 1) SP (only if no lb==1)
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
            if q is not None and rc < -1e-6:
                print(f" ✔ [SP] rc={rc:.2f}")
                return rc, cost, q
            print(f"[SP] item {i} ⇢ no improving column (or mandatory order present)")
        # 2) FEFO DP (only if no lb==1)
        if not must_order:
            rc, cost, q = self.price_fefo_dp(i, mu_hat, pi_i)
            if q is not None and rc < -1e-6:
                print(f" ✔ [DP-FEFO] rc={rc:.2f}")
                return rc, cost, q
            print(f"[DP-FEFO] item {i} ⇢ no improving column")
        # 3) Perishability-aware MIP pricing (handles lb/ub)
        rc, cost, q = self.price_mip_perishable(i, mu_hat, pi_i)
        if q is not None and rc < -1e-6:
            print(f"[MIP] item {i} ✔ rc={rc:.2f}")
            return rc, cost, q
        print(f"[MIP] item {i} ⇢ no improving column")
        # 4) Greedy fallback
        print(f"[HEUR] item {i} generating greedy pattern")
        return self.fallback_heuristic_pattern(i, mu_hat, pi_i)

    # --- helper: Li Chao segment tree (min) with segment insertion ---
    class _LiChao:
        __slots__ = ("L", "R", "m", "b", "idx", "left", "right")

        def __init__(self, L, R):
            self.L, self.R = L, R
            self.m = 0.0
            self.b = float("inf")
            self.idx = -1
            self.left = None
            self.right = None

        def _f(self, m, b, x):
            return m * x + b

        def _mid(self):
            return (self.L + self.R) // 2

        def add_line(self, m, b, idx):
            l, r = self.L, self.R
            mid = self._mid()
            cur_m, cur_b, cur_idx = self.m, self.b, self.idx
            # place better-at-mid line at node
            if self._f(m, b, mid) < self._f(cur_m, cur_b, mid):
                self.m, self.b, self.idx, m, b, idx = m, b, idx, cur_m, cur_b, cur_idx
            if l == r:
                return
            # decide where the worse line could win
            if self._f(m, b, l) < self._f(self.m, self.b, l):
                if not self.left:
                    self.left = type(self)(l, mid)
                self.left.add_line(m, b, idx)
            elif self._f(m, b, r) < self._f(self.m, self.b, r):
                if not self.right:
                    self.right = type(self)(mid + 1, r)
                self.right.add_line(m, b, idx)

        def add_segment(self, m, b, idx, L, R):
            if R < self.L or self.R < L:
                return
            if L <= self.L and self.R <= R:
                self.add_line(m, b, idx)
                return
            mid = self._mid()
            if not self.left:
                self.left = type(self)(self.L, mid)
            if not self.right:
                self.right = type(self)(mid + 1, self.R)
            self.left.add_segment(m, b, idx, L, R)
            self.right.add_segment(m, b, idx, L, R)

        def query(self, x):
            res = (self._f(self.m, self.b, x), self.idx)
            if self.L == self.R:
                return res
            mid = self._mid()
            child = self.left if x <= mid else self.right
            if child:
                v = child.query(x)
                if v[0] < res[0]:
                    res = v
            return res

    # --- drop-in replacement for price_fefo_dp ---
    def price_fefo_dp(self, item_id: int, mu: List[float], pi_i: float):
        D = self.dem[item_id]
        T = len(D)
        if T == 0:
            return float("inf"), float("inf"), None
        m_seq = self.mseq[item_id]
        S = self.setup[item_id]
        cvar = self.c_var[item_id]
        h = self.h[item_id]
        # any mandatory y==1? (we keep your original guard)
        if any(
            lb == 1 and ub == 1
            for (ii, _), (lb, ub) in self.order_fix.items()
            if ii == item_id
        ):
            return float("inf"), float("inf"), None
        forbidden = {
            tt
            for (ii, tt), (lb, ub) in self.order_fix.items()
            if ii == item_id and ub == 0
        }
        # prefix sums
        U = [0] * (T + 1)
        V = [0] * (T + 1)
        for k in range(T):
            U[k + 1] = U[k] + D[k]
            V[k + 1] = V[k] + D[k] * (k)
        # u_max per origin t
        u_max = [min(T - 1, t + m_seq[t] - 1) for t in range(T)]
        # slope B_t for each t
        B = [cvar - mu[t] - h * t for t in range(T)]
        # fmin and argt via Li Chao envelope
        fmin = [[float("inf")] * T for _ in range(T)]
        argt = [[-1] * T for _ in range(T)]
        Xmin, Xmax = 0, U[T]  # domain for Li Chao in U-space (integers)
        for a in range(T - 1, -1, -1):
            root = self._LiChao(Xmin, Xmax)
            Ua, Va = U[a], V[a]
            # add all feasible order lines (t <= a, not forbidden)
            for t in range(0, a + 1):
                if t in forbidden:
                    continue
                R_tau = u_max[t]
                if R_tau < a:
                    continue
                # A_t(a) + B_t * U_tau, valid for U in [U[a], U[R_tau+1]-epsilon]
                At = S - (cvar - mu[t]) * Ua - h * Va + h * t * Ua
                Lx = Ua
                Rx = U[R_tau + 1]  # inclusive; our Li Chao uses integer grid
                root.add_segment(B[t], At, t, Lx, Rx)
            # evaluate for tau=a..T-1
            for tau in range(a, T):
                val, who = root.query(U[tau + 1])
                if who == -1:
                    break  # no feasible cover
                fmin[a][tau] = val + h * V[tau + 1]
                argt[a][tau] = who
        # DP over [a..b]: g[a][b] = min_{tau∈[a..b]} fmin[a][tau] + g[tau+1][b]
        g = [[float("inf")] * T for _ in range(T)]
        split = [[-1] * T for _ in range(T)]
        for a in range(T - 1, -1, -1):
            if fmin[a][a] < float("inf"):
                g[a][a] = fmin[a][a]
                split[a][a] = a
            for b in range(a + 1, T):
                best, bt = float("inf"), -1
                for tau in range(a, b + 1):
                    left = fmin[a][tau]
                    if math.isinf(left):
                        continue
                    right = 0.0 if tau == b else g[tau + 1][b]
                    if math.isinf(right):
                        continue
                    cand = left + right
                    if cand < best:
                        best, bt = cand, tau
                g[a][b] = best
                split[a][b] = bt
        if g[0][T - 1] >= pi_i - 1e-6:
            return float("inf"), float("inf"), None
        # reconstruct q and true cost
        q = [0] * T
        true_cost = 0.0

        def cost_true_block(t, a, b):
            sumD = U[b + 1] - U[a]
            sumAge = V[b + 1] - V[a]
            return S + cvar * sumD + h * (sumAge - t * sumD)

        def rec(a, b):
            nonlocal true_cost
            if a > b:
                return
            tau = split[a][b]
            if tau < 0:
                return
            t = argt[a][tau]
            if t < 0:
                return
            sumD = U[tau + 1] - U[a]
            q[t] += sumD
            true_cost += cost_true_block(t, a, tau)
            rec(tau + 1, b)

        rec(0, T - 1)
        red_cost = g[0][T - 1] - pi_i
        return red_cost, true_cost, q

    # simple greedy fallback (perishability-agnostic but ok as last resort)
    def fallback_heuristic_pattern(self, i: int, mu: List[float], pi_i: float):
        D = self.dem[i]
        T = len(D)
        q = [max(0, d) for d in D]  # naive
        true = self.setup[i] * sum(1 for t in range(T) if q[t] > 0) + self.c_var[
            i
        ] * sum(
            q
        )  # ignore holding
        red = true - sum(mu[t] * q[t] for t in range(T)) - pi_i
        return red, true, q

    # column generation loop
    def column_generation(self):
        while True:
            res = self.master.optimize()
            if res is None:
                print("[FAIL] Master infeasible")
                return float("inf"), (None, None, None)
            obj, pi, mu, lam = res
            print(
                f"[CG] iter {len(self.master.lambda_vars[self.items[0]])} obj={obj:.2f}"
            )
            mu_hat = [
                self.alpha * m + (1 - self.alpha) * p for m, p in zip(mu, self.prev_mu)
            ]
            self.prev_mu = mu_hat
            added = False
            for i in self.items:
                rc, cost, q = self.price_with_growth(i, mu_hat, pi[i])
                if q is not None and rc < -1e-6:
                    before = len(self.master.lambda_vars[i])
                    self.master.add_pattern(i, cost, q)
                    if len(self.master.lambda_vars[i]) > before:
                        print(f" ↳ new col for item {i} rc={rc:.2f}")
                        added = True
                    else:
                        print(f" ↳ duplicate col for item {i} rc={rc:.2f} (skipped)")
            if not added:
                break
        return obj, (pi, mu, lam)

    # B&P recursion
    def branch_and_price(self, best=float("inf"), best_sol=None, best_patterns=None):
        bound, (pi, mu, lam) = self.column_generation()
        self._print_gap(bound, best)
        self.log_node(
            parent=self.parent_stack[-1] if self.parent_stack else None,
            fix=None if not self.order_fix else list(self.order_fix.items())[-1],
            obj=bound,
            incumbent=best,
            status="branching" if bound < best else "pruned",
        )
        if bound >= best - 1e-6:
            return best, best_sol, best_patterns
        # RMP primal
        res = self.master.optimize()
        if res is None:
            print("[FAIL] Final RMP infeasible.")
            return float("inf"), None, None
        _, _, _, lam = res
        y = self.compute_y(lam)
        frac = None
        for (i, t), val in y.items():
            if 1e-6 < val < 1 - 1e-6:
                frac = (i, t)
                print(f"[BRANCH] frac y[{i},{t}]={val:.3f}")
                break
        if frac is None:
            print(f"[SOL] incumbent {bound:.2f}")
            self._print_gap(bound, bound, prefix=" ")
            if bound < best:
                return bound, lam, deepcopy(self.master.patterns)
            return best, best_sol, best_patterns
        i_b, t_b = frac
        best, best_sol, best_patterns = self.branch_child(
            i_b, t_b, (0, 0), best, best_sol, best_patterns
        )
        best, best_sol, best_patterns = self.branch_child(
            i_b, t_b, (1, 1), best, best_sol, best_patterns
        )
        self.parent_stack.pop()
        return best, best_sol, best_patterns

    def branch_child(self, i, t, fix, best, best_sol, best_patterns):
        print(f" |-- child: fix y[{i},{t}]={fix}")
        # Create a lightweight child object without deep-copying the Gurobi model.
        child = object.__new__(BranchPrice)
        # Shallow-copy immutable / shared data structures (dicts/lists are OK since we
        # won't mutate them structurally; we only mutate order_fix and master).
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
        # Per-node state
        child.prev_mu = list(self.prev_mu)
        child.alpha = self.alpha
        # Apply branching fix
        child.order_fix = dict(self.order_fix)
        child.order_fix[(i, t)] = fix
        # Rebuild the RMP in a fresh Gurobi model, reusing all columns you’ve stored.
        child.master = self.master.copy(order_fix=child.order_fix)
        # Share the search tree bookkeeping
        child.tree = self.tree  # share dict to accumulate nodes
        child.node_counter = self.node_counter
        child.parent_stack = self.parent_stack.copy()
        return child.branch_and_price(best, best_sol, best_patterns)

    @staticmethod
    def _print_gap(bound: float, incumbent: float, prefix: str = ""):
        if incumbent < float("inf"):
            gap = 100.0 * (incumbent - bound) / incumbent
            print(
                f"{prefix}[GAP] bound={bound:.2f} best={incumbent:.2f} gap={gap:.2f}%"
            )
        else:
            print(f"{prefix}[GAP] bound={bound:.2f} best=∞ gap=∞")


# ────────────────────────────────────────────────────────────────────────────
# Reporting (FEFO simulation with variable m_{t})
# ────────────────────────────────────────────────────────────────────────────
def detailed_exec_rep(bp: BranchPrice, sol, patterns):
    print("\n=== Detailed execution report ===")
    for i in bp.items:
        print(f"\nItem {i}")
        try:
            sel_idx = next(idx for idx, v in enumerate(sol[i]) if v > 0.9)
        except StopIteration:
            print(f"[ERROR] No selected pattern for item {i} — skipping.")
            continue
        pat = patterns[i][sel_idx]
        orders = pat["q"]
        D = bp.dem[i]
        m_seq = bp.mseq[i]
        # FEFO cohorts: list of (remaining_life, qty)
        cohorts: List[Tuple[int, int]] = []
        back = 0
        print(" t | dem | ord | inv+ | back")
        print("-" * 27)
        for t in range(bp.T):
            # add new cohort
            if orders[t] > 0:
                cohorts.append((m_seq[t], orders[t]))
            # consume FEFO: serve demand+back from cohorts with smallest remaining life
            need = D[t] + back
            # sort by remaining life ascending (closest to expiry)
            cohorts.sort(key=lambda x: x[0])
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
            back = need
            # age and drop expired
            aged = []
            for life, qty in new_coh:
                life -= 1
                if life > 0 and qty > 0:
                    aged.append((life, qty))
            cohorts = aged
            inv_plus = sum(qty for _, qty in cohorts)
            print(f"{t:2d} | {D[t]:3d} | {orders[t]:3d} | {inv_plus:4d} | {back:4d}")


# ────────────────────────────────────────────────────────────────────────────
# Data classes & instance I/O
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class Item:
    id: int
    demand: List[int]
    setup: float
    b_var: float
    c_var: float
    h: float
    shelf_seq: List[int]  # m_{t} per period (variable perishability)


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


# ────────────────────────────────────────────────────────────────────────────
# Instance generation
# ────────────────────────────────────────────────────────────────────────────
def build_lot(
    period: int,
    lb_dem: int,
    ub_dem: int,
    capacity_pad: int,
    specs: List[tuple],
    manual_capacity: List[int] | None,
    default_shelf_rng: Tuple[int, int] = (3, 5),  # geri uyumluluk için
) -> Lot:
    random.seed(SEED)
    lot = Lot(period=period, capacity_pad=capacity_pad, manual_capacity=manual_capacity)
    for rec in specs:
        # desteklenen formatlar:
        # (idx, stp, b, c, h) → default_shelf_rng kullan
        # (idx, stp, b, c, h, (m_lo, m_hi)) → per-item aralık
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


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
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
    sol,
    patterns,
    objective: float,
    elapsed_sec: float,
    stdout_text: str,
    instance_path: Path,
    outdir: str | Path = "results",
) -> Path:
    """
    Save a full run snapshot:
      - TXT: full terminal output + a compact summary
      - JSON: structured inputs/outputs (instance, patterns, λ, tree, etc.)
      - Copy of last_instance.json for reproducibility
    Returns path to the TXT file.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Snapshot instance (copy)
    instance_copy = outdir / f"instance_{ts}.json"
    try:
        instance_copy.write_text(Path(instance_path).read_text())
    except Exception as e:
        # Not fatal
        pass
    # Build structured JSON
    demand, c_var, h, setup, b_var, cap, mseq, kmax = lot.to_dicts()
    # Patterns + λ solution
    patterns_json = {
        i: [
            dict(
                cost=p["cost"],
                y=p.get("y", [1 if q > 0 else 0 for q in p["q"]]),
                q=p["q"],
            )
            for p in patterns[i]
        ]
        for i in bp.items
    }
    lam = sol  # already a dict {i: [λ values]} from branch_and_price()
    # Tree as a list for easier JSON
    tree_list = list(bp.tree.values())
    # Order fixes
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
                "perishability_seq": mseq[i],  # m_{i,t}
                "kmax": kmax[i],
            }
            for i in bp.items
        },
        "patterns": patterns_json,
        "lambda_solution": lam,
        "order_fix": order_fix,
        "tree": tree_list,
    }
    (outdir / f"run_{ts}.json").write_text(json.dumps(run_json, indent=2))
    # Human-readable TXT: summary + whole terminal output
    txt_path = outdir / f"run_{ts}.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=== Multi-Item Perishable Lot-Sizing — Run Report ===\n")
        f.write(f"Timestamp : {ts}\n")
        f.write(f"Objective : {objective:.2f}\n")
        f.write(f"Elapsed (seconds) : {elapsed_sec:.2f}\n")
        f.write(f"Instance JSON : {instance_path}\n")
        f.write(f"Instance copy : {instance_copy}\n")
        f.write(f"Allow backorder : {ALLOW_BACKORDER}\n")
        f.write(f"Items : {len(bp.items)}\n")
        f.write(f"Periods : {bp.T}\n")
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
            try:
                sel_idx = next(idx for idx, v in enumerate(lam[i]) if v > 0.9)
            except StopIteration:
                f.write(f"Item {i}: no λ selected.\n")
                continue
            pat = patterns_json[i][sel_idx]
            f.write(
                f"Item {i}: pattern {sel_idx}, cost={pat['cost']:.2f}, "
                f"y={pat.get('y')}, q={pat['q']}\n"
            )
        f.write("\n-- Branch-and-Price Tree (nodes) --\n")
        for n in tree_list:
            f.write(json.dumps(n) + "\n")
        f.write("\n\n=== Terminal Output (verbatim) ===\n")
        f.write(stdout_text)
    return txt_path


def run_and_save(lot: "Lot") -> None:
    """
    Runs the whole pipeline while teeing stdout to both the terminal and a buffer.
    At the end, writes results/ files with the full terminal output + inputs.
    """
    buf = io.StringIO()
    tee = Tee(sys.stdout, buf)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(tee):
        # (this is basically wrapper_main, unchanged behavior)
        demand, c_var, h, setup, b_var, cap, mseq, k_max = lot.to_dicts()
        bp = BranchPrice(demand, c_var, h, setup, b_var, cap, mseq, k_max)
        best, sol, patterns = bp.branch_and_price()
        if sol is None:
            print("[❌ ERROR] No feasible solution — check capacity vs demand.")
            for t in range(lot.period):
                total_d = sum(it.demand[t] for it in lot.items.values())
                print(
                    f"Period {t:2d} → Total demand: {total_d}, Capacity: {lot.capacity[t]}"
                )
            # end capture before raising
        else:
            elapsed = time.perf_counter() - t0
            print(f"\nObjective: {best:.2f} (elapsed {elapsed:.2f} s)\n")
            root_bound = bp.tree[0]["obj"] if bp.tree else best
            BranchPrice._print_gap(root_bound, best, prefix="[FINAL] ")
            print(f"[INFO] Backorders allowed: {ALLOW_BACKORDER}")
            for i in bp.items:
                try:
                    sel = next(idx for idx, v in enumerate(sol[i]) if v > 0.9)
                    print(
                        f"Item {i}: pattern {sel}, y={patterns[i][sel].get('y')}, q={patterns[i][sel]['q']}"
                    )
                except (StopIteration, IndexError):
                    print(f"[ERROR] No valid pattern for item {i}.")
            detailed_exec_rep(bp, sol, patterns)
            if TREE_VISUALIZATION:
                visualize_bnp_tree(bp.tree)
                print(f"[INFO] Branch-and-Price tree visualized.")
                plt.ioff()
                plt.show()
            print(f"\nObjective: {best:.2f} (elapsed {elapsed:.2f} s)\n")
    # save everything (including verbatim terminal)
    stdout_text = buf.getvalue()
    elapsed = time.perf_counter() - t0
    # If the solve failed, create a tiny stub so you still get a file.
    if sol is None:
        best = float("inf")
        bp = bp if "bp" in locals() else None
        patterns = {}
    out_txt = save_results(
        lot=lot,
        bp=bp,
        sol=sol,
        patterns=patterns,
        objective=best,
        elapsed_sec=elapsed,
        stdout_text=stdout_text,
        instance_path=INSTANCE_PATH,
        outdir="results",
    )
    print(f"[RESULTS] saved to: {out_txt.parent} (main report: {out_txt.name})")


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
            (4, 100.0, 5.0, 2.5, 0.4, (8, 50)),
            (5, 60.0, 5.0, 3.5, 0.7, (9, 29)),
            (6, 80.0, 5.0, 1.5, 0.2, (3, 93)),
            (7, 90.0, 5.0, 2.0, 0.3, (92, 100)),
            (8, 110.0, 5.0, 2.8, 0.4, (60, 90)),
            (9, 120.0, 5.0, 3.2, 0.6, (234, 365)),
            (10, 130.0, 5.0, 4.0, 0.8, (3, 5)),
            (11, 140.0, 5.0, 2.2, 0.3, (32, 53)),
            (12, 170.0, 5.0, 3.4, 0.7, (25, 29)),
            (13, 180.0, 5.0, 1.2, 0.2, (31, 52)),
            (14, 190.0, 5.0, 2.4, 0.3, (23, 78)),
            (15, 2000.0, 5.0, 3.6, 0.5, (30, 365)),
            (16, 210.0, 5.0, 1.8, 0.4, (3, 5)),
            (17, 220.0, 5.0, 2.6, 0.6, (3, 50)),
            (18, 230.0, 5.0, 3.8, 0.7, (3, 60)),
            (19, 240.0, 5.0, 2.1, 0.3, (3, 70)),
            (20, 250.0, 5.0, 4.2, 0.8, (3, 5)),
            (21, 260.0, 5.0, 1.4, 0.2, (3, 93)),
            (22, 270.0, 5.0, 2.9, 0.4, (92, 100)),
            (23, 280.0, 5.0, 3.1, 0.6, (60, 90)),
            (24, 290.0, 5.0, 2.7, 0.4, (234, 365)),
            (25, 300.0, 5.0, 3.3, 0.5, (3, 5)),
            (26, 310.0, 5.0, 2.3, 0.3, (32, 53)),
            (27, 320.0, 5.0, 3.7, 0.7, (25, 29)),
            (28, 330.0, 5.0, 1.1, 0.2, (31, 52)),
            (29, 340.0, 5.0, 2.5, 0.3, (23, 78)),
            (30, 3500.0, 5.0, 3.9, 0.5, (30, 365)),
            (31, 360.0, 5.0, 1.7, 0.4, (3, 5)),
            (32, 370.0, 5.0, 2.8, 0.6, (3, 50)),
            (33, 380.0, 5.0, 3.0, 0.7, (3, 60)),
            (34, 390.0, 5.0, 2.2, 0.3, (3, 70)),
            (35, 400.0, 5.0, 4.1, 0.8, (3, 5)),
            (36, 410.0, 5.0, 1.6, 0.2, (3, 93)),
            (37, 420.0, 5.0, 2.4, 0.4, (92, 100)),
            (38, 430.0, 5.0, 3.5, 0.6, (60, 90)),
            (39, 440.0, 5.0, 2.9, 0.4, (234, 365)),
        ]
        period = 365
        manual_caps = [100000] * period if USE_MANUAL_CAPACITY else None
        lot = build_lot(
            period=period,
            lb_dem=50,
            ub_dem=2000,
            capacity_pad=10,
            specs=specs,
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
