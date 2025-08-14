"""
Branch-and-price for a multi-item perishable lot-sizing problem
with setup costs (time-varying allowed), holding costs, optional back-orders,
dual stabilisation, order/no-order branching on (item,period), and options:
 - Per-item, per-time production capacities cap_{i,t}
 - Time-varying setups s_{i,t} (or scalar s_i)
 - Fixed warehouse holding capacity W (end-of-period)

Pricing order: Shortest-Path (SP) → FEFO DP → Perishability-aware MIP → Greedy.
Python ≥ 3.8, gurobipy installed, academic licence assumed.
"""

from __future__ import annotations
import random, sys, math
from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import asdict, dataclass, field
from pathlib import Path
import time, json
import numpy as np
import gurobipy as grb
import io, contextlib, os
from datetime import datetime

# ────────────────────────────────────────────────────────────────────────────
# Global switches & defaults (organized)
# ────────────────────────────────────────────────────────────────────────────
RANDOMIZE = True  # False → reuse last_instance.json
USE_MANUAL_CAPACITY = True  # if True, MANUAL_CAPACITY_VALUE is used
ALLOW_BACKORDER = False  # pricing assumes no backorders; kept for completeness
SEED = 0
INSTANCE_PATH = Path(__file__).with_name("last_instance.json")

# Instance-wide generation knobs (active when RANDOMIZE=True)
PERIOD = 60
DEMAND_RANGE = (5, 80)  # (lo, hi) for per-period demand sampling

# Global capacity: either manual or auto-from-demand (with buffer)
MANUAL_CAPACITY_VALUE = 10000  # used if USE_MANUAL_CAPACITY=True
AUTO_CAPACITY_BUFFER_FRAC = (
    0.2  # if manual disabled, add this buffer on top of total demand
)

# Warehouse capacity (end-of-period inventory) — set None to disable
WAREHOUSE_CAPACITY = None  # e.g., 50000

# Time-varying setup sequence generation
SETUP_SEQ_ENABLE = True  # if True, generate s_{i,t}; else use scalar s_i
SETUP_SEQ_AMPLITUDE = 0.10  # ± amplitude around base setup (e.g., 0.10 → ±10%)
SETUP_SEQ_PERIOD = 30.0  # seasonal period (days)
SETUP_SEQ_JITTER = 0.04  # extra iid jitter per period (0..0.1 recommended)

# Per-item capacity generation policy
#   "none"          → do not create cap_{i,t}
#   "demand_pad"    → cap_{i,t} = max(demand_t + ITEM_CAP_PAD, int(ITEM_CAP_MULT * demand_t))
#   "uniform_range" → cap_{i,t} ~ Uniform[lo, hi] (integers)
ITEM_CAP_SEQ_POLICY = "none"
ITEM_CAP_PAD = 50
ITEM_CAP_MULT = 1.5
ITEM_CAP_UNIFORM_RANGE = (400, 600)  # only used if POLICY == "uniform_range"

# Setup overrides (optional; applied AFTER build_lot())
# Priority: JSON sequences → scalar overrides → explicit sequences
SETUP_JSON_OVERRIDES = None  # e.g., "setup_sequences.json"
SETUP_SCALAR_OVERRIDES = {
    # 0: 150.0,
}
SETUP_SEQUENCE_OVERRIDES = {
    # 1: [120.0]*PERIOD,
}

# Dual stabilization
DUAL_STAB_ALPHA = 0.6


# ────────────────────────────────────────────────────────────────────────────
# Shortest path pricing (no backorders), supports variable m_{t} and s_{t}
# ────────────────────────────────────────────────────────────────────────────
def price_shortest_path(
    item_id: int,
    demand: List[int],
    c_var: float,
    h: float,
    setup_seq: Union[float, List[float]],  # setup can be scalar or list
    mu_eff: List[float],  # effective duals per t (global+item+warehouse)
    pi: float,
    m_seq: List[int] | None,
    k_max: int,
    order_fix: Dict[Tuple[int, int], Tuple[int, int]],
) -> Tuple[float, float, Optional[List[int]]]:
    """
    Layered DAG shortest path: nodes 0..T; arc (t -> u+1) = one order at t covering demand[t..u].
    Feasible if u <= t + m_seq[t] - 1 and cumulative qty ≤ k_max.
    Respects ub==0 (forbidden orders). Does NOT enforce lb==1; caller should skip SP if any lb==1.
    Reduced cost uses mu_eff[t] (already folded duals).
    """
    T = len(demand)
    dist = [float("inf")] * (T + 1)
    nxt_arc: List[Optional[Tuple[int, int]]] = [None] * T
    dist[T] = -pi
    forbidden = {
        tt for (ii, tt), (lb, ub) in order_fix.items() if ii == item_id and ub == 0
    }

    def s_at(t: int) -> float:
        return setup_seq[t] if isinstance(setup_seq, list) else float(setup_seq)

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
            red_arc = (
                (c_var - mu_eff[t]) * q_acc + h * hold_acc + (s_at(t) if q_acc else 0)
            )
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
        true_cost += s_at(t) + c_var * qty + h * hold
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
      - global capacity: sum_i sum_p q_{i,p,t} λ_{i,p} ≤ cap_t
      - per-item capacity: sum_p q_{i,p,t} λ_{i,p} ≤ cap_{i,t}        (optional)
      - warehouse cap: sum_i sum_p invTail_{i,p,u} λ_{i,p} ≤ W        (optional)
        where invTail_{i,p,u} := max(0, ∑_{τ≤u} q_{i,p,τ} − ∑_{τ≤u} d_{i,τ})

      - branching (fix y_{i,t}): if y=1 ⇒ sum_p δ_{i,p,t} λ_{i,p} ≥ 1
                                 if y=0 ⇒ sum_p δ_{i,p,t} λ_{i,p} ≤ 0
        where δ_{i,p,t} = 1 if pattern p orders at t (q_{i,p,t} > 0)
    """

    def __init__(
        self,
        items,
        T,
        capacity,
        cap_seq: Dict[int, Optional[List[float]]],
        W: Optional[float],
        demand: Dict[int, List[int]],
    ):
        self.items, self.T, self.capacity = items, T, capacity
        self.cap_seq, self.W, self.demand = cap_seq, W, demand

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
        self.cap_item_constr: Dict[Tuple[int, int], grb.Constr] = {}
        for i in items:
            seq = cap_seq.get(i) if cap_seq else None
            if seq is None:
                continue
            for t in range(T):
                self.cap_item_constr[(i, t)] = self.model.addConstr(
                    le() <= float(seq[t]), name=f"cap_i_{i}_{t}"
                )
        self.whcap_constr: List[grb.Constr] = []
        if W is not None:
            for u in range(T - 1):
                self.whcap_constr.append(
                    self.model.addConstr(le() <= float(W), name=f"whcap_{u}")
                )
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

    def _inv_tail_vector(self, i: int, q: List[int]) -> List[float]:
        """Compute invTail_{u} = max(0, cum_q[u] - cum_d[u]) for u=0..T-2."""
        T = self.T
        if not self.whcap_constr:
            return []
        D = self.demand[i]
        cq, cd = [0] * T, [0] * T
        s = 0
        for t in range(T):
            s += q[t]
            cq[t] = s
        s = 0
        for t in range(T):
            s += D[t]
            cd[t] = s
        inv = [max(0, cq[u] - cd[u]) for u in range(T - 1)]
        return inv

    def add_pattern(self, i: int, cost: float, q: List[int]):
        if any(p["q"] == q for p in self.patterns[i]):
            return
        y = [1 if qty > 0 else 0 for qty in q]
        col = grb.Column()
        col.addTerms(1.0, self.sel_constr[i])
        for t, qty in enumerate(q):
            if qty:
                col.addTerms(qty, self.cap_constr[t])
        for t, qty in enumerate(q):
            if qty and (i, t) in self.cap_item_constr:
                col.addTerms(qty, self.cap_item_constr[(i, t)])
        if self.whcap_constr:
            inv_tail = self._inv_tail_vector(i, q)
            for u, coeff in enumerate(inv_tail):
                if coeff:
                    col.addTerms(coeff, self.whcap_constr[u])
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
        mu_global = [c.Pi for c in self.cap_constr]
        mu_item: Dict[int, List[float]] = {i: [0.0] * self.T for i in self.items}
        for (i, t), row in self.cap_item_constr.items():
            mu_item[i][t] = row.Pi
        nu_wh = [row.Pi for row in self.whcap_constr] if self.whcap_constr else []
        lam = {i: [v.X for v in self.lambda_vars[i]] for i in self.items}
        return self.model.ObjVal, pi, mu_global, mu_item, nu_wh, lam

    def copy(self, order_fix=None):
        clone = MasterModel(
            self.items, self.T, self.capacity, self.cap_seq, self.W, self.demand
        )
        for i in self.items:
            for pat in self.patterns[i]:
                clone.add_pattern(i, pat["cost"], pat["q"])
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


# ────────────────────────────────────────────────────────────────────────────
# Branch-and-Price driver
# ────────────────────────────────────────────────────────────────────────────
class BranchPrice:
    def __init__(
        self,
        demand: Dict[int, List[int]],
        c_var: Dict[int, float],
        h: Dict[int, float],
        setup: Dict[int, Union[float, List[float]]],
        b_var: Dict[int, float],
        capacity: List[int],
        shelf_seq: Dict[int, List[int]],
        k_max: Dict[int, int],
        cap_seq: Dict[int, Optional[List[float]]],
        W: Optional[float],
    ):
        self.items = list(demand)
        self.dem, self.c_var, self.h = demand, c_var, h
        self.setup, self.b_var = setup, b_var
        self.cap, self.T = capacity, len(capacity)
        self.mseq = shelf_seq
        self.k_max = k_max
        self.cap_seq = cap_seq
        self.W = W
        self.order_fix: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.prev_mu_base = [0.0] * self.T
        self.alpha = DUAL_STAB_ALPHA

        self.master = MasterModel(
            self.items, self.T, self.cap, self.cap_seq, self.W, self.dem
        )

        BIG_M = 1e6
        for i in self.items:
            q = self.dem[i][:]
            setup_cost = sum(self._setup_at(i, t) for t, qty in enumerate(q) if qty > 0)
            cost = sum(self.c_var[i] * q_t for q_t in q) + setup_cost
            self.master.add_pattern(i, cost, q)
            self.master.add_pattern(i, BIG_M, [0] * self.T)
        self.master.model.update()

        self.tree = {}
        self.node_counter = 0
        self.parent_stack = []

    def _setup_at(self, i: int, t: int) -> float:
        s = self.setup[i]
        return s[t] if isinstance(s, list) else float(s)

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

    def price_mip_perishable(
        self, item_id: int, mu_eff: List[float], pi_i: float
    ) -> Tuple[float, float, Optional[List[int]]]:
        D = self.dem[item_id]
        T = len(D)
        m_seq = self.mseq[item_id]
        kmax = self.k_max[item_id]
        s_seq = self.setup[item_id]

        m = grb.Model(f"price_per_{item_id}")
        m.Params.OutputFlag = 0
        m.Params.Presolve = 2
        m.Params.Method = 2
        m.Params.Cuts = 2
        m.Params.Heuristics = 0.5
        m.Params.Threads = 1

        q = m.addVars(T, vtype=grb.GRB.INTEGER, lb=0, ub=kmax, name="q")
        y = m.addVars(T, vtype=grb.GRB.BINARY, name="y")
        x = {}
        for s in range(T):
            v_s = min(T - 1, s + m_seq[s] - 1)
            for t in range(s, v_s + 1):
                x[(s, t)] = m.addVar(
                    vtype=grb.GRB.CONTINUOUS, lb=0.0, name=f"x_{s}_{t}"
                )

        for t in range(T):
            m.addConstr(
                grb.quicksum(x[(s, t)] for s in range(t + 1) if (s, t) in x) == D[t],
                name=f"demand_{t}",
            )
        for s in range(T):
            out = grb.quicksum(x[(s, t)] for t in range(s, T) if (s, t) in x)
            m.addConstr(out <= q[s], name=f"cap_{s}")

        for s in range(T):
            m.addGenConstrIndicator(y[s], True, q[s] >= 1)
            m.addGenConstrIndicator(y[s], False, q[s] == 0)

        if isinstance(s_seq, list):
            setup_cost = grb.quicksum(float(s_seq[t]) * y[t] for t in range(T))
        else:
            setup_cost = float(s_seq) * y.sum()

        var_cost = self.c_var[item_id] * q.sum()
        hold_cost = grb.quicksum(self.h[item_id] * (t - s) * x[(s, t)] for (s, t) in x)
        dual_adj = grb.quicksum((-mu_eff[s]) * q[s] for s in range(T)) - pi_i

        m.setObjective(
            setup_cost + var_cost + hold_cost + dual_adj, sense=grb.GRB.MINIMIZE
        )

        for t in range(T):
            lb, ub = self.order_fix.get((item_id, t), (0, 1))
            y[t].LB, y[t].UB = lb, ub

        m.Params.TimeLimit = 30
        m.optimize()
        if m.Status != grb.GRB.OPTIMAL or m.ObjVal >= -1e-6:
            return float("inf"), None, None

        q_plan = [int(round(q[t].X)) for t in range(T)]
        setup_true = sum(
            (s_seq[t] if isinstance(s_seq, list) else float(s_seq))
            for t in range(T)
            if q_plan[t] > 0
        )
        var_true = self.c_var[item_id] * sum(q_plan)
        hold_true = sum(self.h[item_id] * (t - s) * x[(s, t)].X for (s, t) in x)
        true_val = setup_true + var_true + hold_true
        return m.ObjVal, true_val, q_plan

    def price_with_growth(
        self, i: int, mu_hat_base: List[float], mu_item_i: List[float], pi_i: float
    ):
        mu_eff = [mu_hat_base[t] + mu_item_i[t] for t in range(self.T)]
        s_seq = self.setup[i]
        must_order = {
            t
            for (ii, t), (lb, ub) in self.order_fix.items()
            if ii == i and lb == ub == 1
        }
        if not must_order:
            rc, cost, q = price_shortest_path(
                i,
                self.dem[i],
                self.c_var[i],
                self.h[i],
                s_seq,
                mu_eff,
                pi_i,
                self.mseq[i],
                self.k_max[i],
                self.order_fix,
            )
            if q is not None and rc < -1e-6:
                print(f" ✔ [SP] rc={rc:.2f}")
                return rc, cost, q
            print(f"[SP] item {i} ⇢ no improving column (or mandatory order present)")
        if not must_order:
            rc, cost, q = self.price_fefo_dp(i, mu_eff, pi_i)
            if q is not None and rc < -1e-6:
                print(f" ✔ [DP-FEFO] rc={rc:.2f}")
                return rc, cost, q
            print(f"[DP-FEFO] item {i} ⇢ no improving column")
        rc, cost, q = self.price_mip_perishable(i, mu_eff, pi_i)
        if q is not None and rc < -1e-6:
            print(f"[MIP] item {i} ✔ rc={rc:.2f}")
            return rc, cost, q
        print(f"[MIP] item {i} ⇢ no improving column")
        print(f"[HEUR] item {i} generating greedy pattern")
        return self.fallback_heuristic_pattern(i, mu_eff, pi_i)

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
            if self._f(m, b, mid) < self._f(cur_m, cur_b, mid):
                self.m, self.b, self.idx, m, b, idx = m, b, idx, cur_m, cur_b, cur_idx
            if l == r:
                return
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

    def price_fefo_dp(self, item_id: int, mu_eff: List[float], pi_i: float):
        D = self.dem[item_id]
        T = len(D)
        if T == 0:
            return float("inf"), float("inf"), None
        m_seq = self.mseq[item_id]
        s_seq = self.setup[item_id]
        cvar = self.c_var[item_id]
        h = self.h[item_id]

        def s_at(t):
            return s_seq[t] if isinstance(s_seq, list) else float(s_seq)

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

        U = [0] * (T + 1)
        V = [0] * (T + 1)
        for k in range(T):
            U[k + 1] = U[k] + D[k]
            V[k + 1] = V[k] + D[k] * (k)

        u_max = [min(T - 1, t + m_seq[t] - 1) for t in range(T)]
        B = [cvar - mu_eff[t] - h * t for t in range(T)]

        fmin = [[float("inf")] * T for _ in range(T)]
        argt = [[-1] * T for _ in range(T)]
        Xmin, Xmax = 0, U[T]

        for a in range(T - 1, -1, -1):
            root = self._LiChao(Xmin, Xmax)
            Ua, Va = U[a], V[a]
            for t in range(0, a + 1):
                if t in forbidden:
                    continue
                R_tau = u_max[t]
                if R_tau < a:
                    continue
                At = s_at(t) - (cvar - mu_eff[t]) * Ua - h * Va + h * t * Ua
                Lx = Ua
                Rx = U[R_tau + 1]
                root.add_segment(B[t], At, t, Lx, Rx)
            for tau in range(a, T):
                val, who = root.query(U[tau + 1])
                if who == -1:
                    break
                fmin[a][tau] = val + h * V[tau + 1]
                argt[a][tau] = who

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

        q = [0] * T
        true_cost = 0.0

        def cost_true_block(t, a, b):
            sumD = U[b + 1] - U[a]
            sumAge = V[b + 1] - V[a]
            return s_at(t) + cvar * sumD + h * (sumAge - t * sumD)

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

    def fallback_heuristic_pattern(self, i: int, mu_eff: List[float], pi_i: float):
        D = self.dem[i]
        T = len(D)
        q = [max(0, d) for d in D]
        s_seq = self.setup[i]
        setup_cost = sum(
            (s_seq[t] if isinstance(s_seq, list) else float(s_seq))
            for t in range(T)
            if q[t] > 0
        )
        true = setup_cost + self.c_var[i] * sum(q)
        red = true - sum(mu_eff[t] * q[t] for t in range(T)) - pi_i
        return red, true, q

    def column_generation(self):
        while True:
            res = self.master.optimize()
            if res is None:
                print("[FAIL] Master infeasible")
                return float("inf"), (None, None, None, None, None)
            obj, pi, mu_global, mu_item, nu_wh, lam = res
            print(
                f"[CG] iter {len(self.master.lambda_vars[self.items[0]])} obj={obj:.2f}"
            )

            # Fold warehouse duals into base per-t dual
            tail = [0.0] * self.T
            for t in range(self.T):
                tail[t] = sum(nu_wh[u] for u in range(t, self.T - 1)) if nu_wh else 0.0
            mu_base_now = [mu_global[t] + tail[t] for t in range(self.T)]
            mu_hat_base = [
                self.alpha * mu_base_now[t] + (1 - self.alpha) * self.prev_mu_base[t]
                for t in range(self.T)
            ]
            self.prev_mu_base = mu_hat_base

            added = False
            for i in self.items:
                rc, cost, q = self.price_with_growth(
                    i, mu_hat_base, mu_item.get(i, [0.0] * self.T), pi[i]
                )
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
        return obj, (pi, mu_global, mu_item, nu_wh, lam)

    def branch_and_price(self, best=float("inf"), best_sol=None, best_patterns=None):
        bound, (pi, mu_g, mu_i, nu, lam) = self.column_generation()
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
        res = self.master.optimize()
        if res is None:
            print("[FAIL] Final RMP infeasible.")
            return float("inf"), None, None
        *_, lam = res
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
        child = object.__new__(BranchPrice)
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
        child.cap_seq = self.cap_seq
        child.W = self.W
        child.prev_mu_base = list(self.prev_mu_base)
        child.alpha = self.alpha
        child.order_fix = dict(self.order_fix)
        child.order_fix[(i, t)] = fix
        child.master = self.master.copy(order_fix=child.order_fix)
        child.tree = self.tree
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
        cohorts: List[Tuple[int, int]] = []
        back = 0
        print(" t | dem | ord | inv+ | back")
        print("-" * 27)
        for t in range(bp.T):
            if orders[t] > 0:
                cohorts.append((m_seq[t], orders[t]))
            need = D[t] + back
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
    setup: Union[float, List[float]]  # scalar or list s_{t}
    b_var: float
    c_var: float
    h: float
    shelf_seq: List[int]  # m_{t}
    cap_seq: Optional[List[float]] = None  # optional cap_{i,t}


@dataclass
class Lot:
    period: int
    capacity_pad: int
    items: Dict[int, Item] = field(default_factory=dict)
    manual_capacity: List[int] = field(default=None)
    warehouse_capacity: Optional[float] = None

    @property
    def capacity(self) -> List[int]:
        if self.manual_capacity is not None:
            return self.manual_capacity
        cap_raw = [
            sum(it.demand[t] for it in self.items.values()) for t in range(self.period)
        ]
        max_cap = max(cap_raw)
        buffer = max(5, int(AUTO_CAPACITY_BUFFER_FRAC * max_cap))
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
        cap_seq = {i: it.cap_seq for i, it in self.items.items()}
        return (
            demand,
            c_var,
            h,
            setup,
            b_var,
            cap,
            mseq,
            kmax,
            cap_seq,
            self.warehouse_capacity,
        )

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serial = asdict(self)
        serial["items"] = {str(k): v for k, v in serial["items"].items()}
        if self.manual_capacity is None:
            serial.pop("manual_capacity", None)
        path.write_text(json.dumps(serial, indent=indent))

    def to_mip_json(self, path: str | Path, *, indent: int = 2) -> None:
        payload = {
            "period": self.period,
            "items": {
                str(i): {
                    "demand": it.demand,
                    "setup": it.setup,  # number or list
                    "c_var": it.c_var,
                    "h": it.h,
                    "b_var": it.b_var,
                    "shelf_seq": it.shelf_seq,
                    **({"cap_seq": it.cap_seq} if it.cap_seq is not None else {}),
                }
                for i, it in self.items.items()
            },
        }
        if self.manual_capacity is not None:
            payload["manual_capacity"] = list(self.manual_capacity)
        if self.warehouse_capacity is not None:
            payload["warehouse_capacity"] = float(self.warehouse_capacity)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Lot":
        data = json.loads(Path(path).read_text())
        items = {
            int(k): Item(
                id=int(k),
                demand=v["demand"],
                setup=v["setup"],
                b_var=v["b_var"],
                c_var=v["c_var"],
                h=v["h"],
                shelf_seq=v["shelf_seq"],
                cap_seq=v.get("cap_seq"),
            )
            for k, v in data["items"].items()
        }
        return cls(
            period=data["period"],
            capacity_pad=data.get("capacity_pad", 0),
            items=items,
            manual_capacity=data.get("manual_capacity", None),
            warehouse_capacity=data.get("warehouse_capacity", None),
        )


# ────────────────────────────────────────────────────────────────────────────
# Instance generation (organized, with knobs)
# ────────────────────────────────────────────────────────────────────────────
def _sample_setup_base(stp: Union[float, Tuple[float, float]]) -> float:
    """Allow a scalar or (lo, hi) range in specs for setup base."""
    if isinstance(stp, (list, tuple)) and len(stp) == 2:
        lo, hi = float(stp[0]), float(stp[1])
        return random.uniform(lo, hi)
    return float(stp)


def _build_setup_sequence(base: float, period: int) -> List[float]:
    """Build s_{t} around 'base' using sinusoidal seasonality and optional jitter."""
    seq = []
    amp = SETUP_SEQ_AMPLITUDE
    per = SETUP_SEQ_PERIOD
    jit = SETUP_SEQ_JITTER
    for t in range(period):
        seasonal = 1.0 + amp * math.sin(2 * math.pi * t / per)
        jitter = 1.0 + (random.uniform(-jit, jit) if jit > 0 else 0.0)
        seq.append(base * seasonal * jitter)
    return seq


def _build_item_cap_seq(demand: List[int], policy: str) -> Optional[List[int]]:
    if policy == "none":
        return None
    T = len(demand)
    if policy == "demand_pad":
        return [
            max(demand[t] + ITEM_CAP_PAD, int(ITEM_CAP_MULT * demand[t]))
            for t in range(T)
        ]
    if policy == "uniform_range":
        lo, hi = ITEM_CAP_UNIFORM_RANGE
        return [random.randint(int(lo), int(hi)) for _ in range(T)]
    raise ValueError(f"Unknown ITEM_CAP_SEQ_POLICY: {policy}")


def build_lot(
    period: int,
    demand_range: Tuple[int, int],
    capacity_pad: int,
    specs: List[tuple],
    manual_capacity: Optional[List[int]],
    default_shelf_rng: Tuple[int, int] = (3, 5),
    setup_seq_enable: bool = False,
    item_cap_seq_policy: str = "none",
    warehouse_capacity: Optional[float] = None,
) -> Lot:
    """
    specs format per item:
      (idx, setup_base_or_range, b, c, h)                           → default shelf range
      (idx, setup_base_or_range, b, c, h, (m_lo, m_hi))             → per-item shelf range
    where setup_base_or_range is either a float or a (lo, hi) tuple to sample from.
    """
    random.seed(SEED)
    lot = Lot(
        period=period,
        capacity_pad=capacity_pad,
        manual_capacity=manual_capacity,
        warehouse_capacity=warehouse_capacity,
    )
    lo_d, hi_d = map(int, demand_range)
    for rec in specs:
        if len(rec) == 5:
            idx, stp_raw, b, c, hold = rec
            shelf_rng = default_shelf_rng
        elif len(rec) == 6:
            idx, stp_raw, b, c, hold, shelf_rng = rec
            if not (isinstance(shelf_rng, tuple) and len(shelf_rng) == 2):
                raise ValueError("specs[5] must be (m_lo, m_hi)")
        else:
            raise ValueError(
                "spec must be (idx, stp, b, c, h) or (idx, stp, b, c, h, (m_lo,m_hi))"
            )

        demand = [random.randint(lo_d, hi_d) for _ in range(period)]

        # Ensure shelf_rng[0] is less than shelf_rng[1]
        if shelf_rng[0] == shelf_rng[1]:
            shelf_seq = [shelf_rng[0]] * period  # Use a constant value
        else:
            shelf_seq = [
                random.randint(shelf_rng[0], shelf_rng[1]) for _ in range(period)
            ]

        base_setup = _sample_setup_base(stp_raw)
        if setup_seq_enable:
            setup_val: Union[float, List[float]] = _build_setup_sequence(
                base_setup, period
            )
        else:
            setup_val = float(base_setup)

        cap_seq = _build_item_cap_seq(demand, item_cap_seq_policy)

        lot.items[idx] = Item(
            idx, demand, setup_val, float(b), float(c), float(hold), shelf_seq, cap_seq
        )
    return lot


# ────────────────────────────────────────────────────────────────────────────
# Run harness: save reports & JSON
# ────────────────────────────────────────────────────────────────────────────
class Tee(io.TextIOBase):
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
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    instance_copy = outdir / f"instance_{ts}.json"
    try:
        instance_copy.write_text(Path(instance_path).read_text())
    except Exception:
        pass

    demand, c_var, h, setup, b_var, cap, mseq, kmax, cap_seq, W = lot.to_dicts()
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
    lam = sol
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
        "warehouse_capacity": W,
        "items": {
            str(i): {
                "setup": setup[i],
                "b_var": b_var[i],
                "c_var": c_var[i],
                "h": h[i],
                "demand": demand[i],
                "perishability_seq": mseq[i],
                "kmax": kmax[i],
                "cap_seq": cap_seq[i],
            }
            for i in bp.items
        },
        "patterns": patterns_json,
        "lambda_solution": lam,
        "order_fix": order_fix,
        "tree": tree_list,
    }
    (outdir / f"run_{ts}.json").write_text(json.dumps(run_json, indent=2))

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
        f.write("\n-- Global Capacity --\n")
        f.write(", ".join(map(str, cap)) + "\n")
        if W is not None:
            f.write(f"\n-- Warehouse capacity W --\n{W}\n")
        f.write("\n-- Per-item: costs / perishability / cap_seq present --\n")
        for i in bp.items:
            mseq_i = mseq[i]
            has_seq = isinstance(setup[i], list)
            f.write(
                f"Item {i}: setup={'seq' if has_seq else setup[i]}, "
                f"c={c_var[i]}, h={h[i]}, b={b_var[i]}, "
                f"min(m_it)={min(mseq_i)}, max(m_it)={max(mseq_i)}, "
                f"cap_seq={'yes' if cap_seq[i] is not None else 'no'}\n"
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
                f"Item {i}: pattern {sel_idx}, cost={pat['cost']:.2f}, y={pat.get('y')}, q={pat['q']}\n"
            )
        f.write("\n-- Branch-and-Price Tree (nodes) --\n")
        for n in tree_list:
            f.write(json.dumps(n) + "\n")
        f.write("\n\n=== Terminal Output (verbatim) ===\n")
        f.write(stdout_text)
    return txt_path


def run_and_save(lot: "Lot") -> None:
    buf = io.StringIO()
    tee = Tee(sys.stdout, buf)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(tee):
        demand, c_var, h, setup, b_var, cap, mseq, k_max, cap_seq, W = lot.to_dicts()
        bp = BranchPrice(demand, c_var, h, setup, b_var, cap, mseq, k_max, cap_seq, W)
        best, sol, patterns = bp.branch_and_price()
        if sol is None:
            print("[❌ ERROR] No feasible solution — check capacity vs demand.")
            for t in range(lot.period):
                total_d = sum(it.demand[t] for it in lot.items.values())
                print(
                    f"Period {t:2d} → Total demand: {total_d}, Capacity: {lot.capacity[t]}"
                )
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
            print(f"\nObjective: {best:.2f} (elapsed {elapsed:.2f} s)\n")
    stdout_text = buf.getvalue()
    elapsed = time.perf_counter() - t0
    if sol is None:
        best = float("inf")
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


def apply_setup_overrides(lot: "Lot", period: int) -> None:
    """Priority: JSON file → SETUP_SCALAR_OVERRIDES → SETUP_SEQUENCE_OVERRIDES."""
    if SETUP_JSON_OVERRIDES:
        p = Path(SETUP_JSON_OVERRIDES)
        if not p.exists():
            raise FileNotFoundError(f"SETUP_JSON_OVERRIDES file not found: {p}")
        data = json.loads(p.read_text())
        for k, seq in data.items():
            i = int(k)
            if i not in lot.items:
                continue
            if len(seq) != period:
                raise ValueError(
                    f"setup seq for item {i} has len={len(seq)}; expected {period}"
                )
            lot.items[i].setup = [float(x) for x in seq]
    for i, val in (SETUP_SCALAR_OVERRIDES or {}).items():
        if i in lot.items:
            lot.items[i].setup = float(val)
    for i, seq in (SETUP_SEQUENCE_OVERRIDES or {}).items():
        if i not in lot.items:
            continue
        if len(seq) != period:
            raise ValueError(
                f"SETUP_SEQUENCE_OVERRIDES[{i}] len={len(seq)}; expected {period}"
            )
        lot.items[i].setup = [float(x) for x in seq]


# ────────────────────────────────────────────────────────────────────────────
# Main with organized knobs (edit here)
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    if (not RANDOMIZE) and INSTANCE_PATH.exists():
        lot = Lot.from_json(INSTANCE_PATH)
        print(f"[INFO] Loaded instance from {INSTANCE_PATH}")
    else:
        # specs: (item_id, setup_base_or_range (..,..), b, c, h, (m_min, m_max))
        # Tip: put a scalar for fixed setup OR a (lo, hi) tuple to randomize per item.
        specs = [
            (0, (50.0, 150.0), 5.0, 2.0, 0.4, (6, 50)),
            (1, (20.0, 50.0), 5.0, 3.0, 0.6, (6, 60)),
            (2, (45.0, 70.0), 5.0, 1.0, 0.3, (6, 70)),
            (3, 75.0, 5.0, 4.0, 0.5, (5, 6)),
            (4, 100.0, 5.0, 2.5, 0.4, (8, 50)),
            (5, (120.0, 150.0), 5.0, 3.5, 0.7, (9, 29)),
            (6, 80.0, 5.0, 1.5, 0.2, (6, 93)),
            (7, (120.0, 150.0), 5.0, 2.0, 0.3, (92, 100)),
            (8, (95.0, 130.0), 5.0, 2.8, 0.4, (60, 90)),
            (9, 120.0, 5.0, 3.2, 0.6, (234, 365)),
            (10, 110.0, 5.0, 2.0, 0.4, (1, 50)),
        ]

        manual_caps = [MANUAL_CAPACITY_VALUE] * PERIOD if USE_MANUAL_CAPACITY else None

        lot = build_lot(
            period=PERIOD,
            demand_range=DEMAND_RANGE,
            capacity_pad=10,
            specs=specs,
            manual_capacity=manual_caps,
            default_shelf_rng=(3, 5),
            setup_seq_enable=SETUP_SEQ_ENABLE,
            item_cap_seq_policy=ITEM_CAP_SEQ_POLICY,
            warehouse_capacity=WAREHOUSE_CAPACITY,
        )

        # Respect item-specific perishability ranges from specs (overwrite default)
        for (i, *_rest, rng) in specs:
            if isinstance(rng, tuple) and len(rng) == 2:
                lo, hi = rng
                lot.items[i].shelf_seq = [random.randint(lo, hi) for _ in range(PERIOD)]

        # Optional setup overrides (JSON / scalar / explicit sequence)
        apply_setup_overrides(lot, PERIOD)

        # Persist instance in unified JSON schema
        lot.to_mip_json(INSTANCE_PATH)
        print(f"[INFO] Generated instance → {INSTANCE_PATH}")

    run_and_save(lot)
