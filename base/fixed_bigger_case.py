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

    # add column obeying the current order_fix
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
        # order indicators
        for (ii, tt), (lb, ub) in order_fix.items():
            if (ii, tt) not in self.order_rows:
                sense = (
                    ">" if lb == 1 and ub == 1 else "<" if lb == 0 and ub == 0 else ">"
                )
                rhs = lb if lb == ub else 1
                expr = grb.LinExpr()  # Create an empty linear expression
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

    def copy(self):
        clone = MasterModel(self.items, self.T, self.capacity)
        for i in self.items:
            for pat in self.patterns[i]:
                clone.add_pattern(
                    i, pat["cost"], pat["q"], {}
                )  # order rows added later
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
        # add initial “order-exact-demand” pattern per item
        for i in self.items:
            q = self.dem[i]
            cost = sum(c_var[i] * q_t for q_t in q) + setup[i] * sum(
                1 for q_t in q if q_t > 0
            )
            self.master.add_pattern(i, cost, q, self.order_fix)
        self.master.model.update()

    # helper: compute y_it from current λ
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
                return float("inf")
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
                red, cost, q, _ = dp_pricing_general(
                    i,
                    self.dem[i],
                    self.c_var[i],
                    self.h[i],
                    self.setup[i],
                    self.b_var[i],
                    mu_hat,
                    pi[i],
                    self.shelf[i],
                    self.k_max[i],
                    self.order_fix,
                    allow_backorder=False,
                )
                if red < -1e-6:
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
        return obj

    def branch_and_price(self, best=float("inf"), best_sol=None):
        bound = self.column_generation()
        if bound >= best - 1e-6:
            return best, best_sol
        _, _, _, lam = self.master.optimize()
        y = self.compute_y(lam)
        frac = None
        for (i, t), val in y.items():
            if 1e-6 < val < 1 - 1e-6:
                frac = (i, t)
                print(f"[BRANCH] depth?  frac y[{i},{t}]={y[(i,t)]:.3f}")
                break
        if frac is None:
            print(f"[SOL] incumbent {bound:.2f}")
            return bound, lam
        i_b, t_b = frac
        # branch (no order) then (must order)
        best, best_sol = self.branch_child(i_b, t_b, (0, 0), best, best_sol)
        best, best_sol = self.branch_child(i_b, t_b, (1, 1), best, best_sol)
        return best, best_sol

    def branch_child(self, i, t, fix, best, best_sol):
        print(f"    |-- create child  fix y[{i},{t}]={fix}")
        child = deepcopy(self)
        child.order_fix[(i, t)] = fix
        child.master = child.master.copy()  # need separate model
        return child.branch_and_price(best, best_sol)

    def __deepcopy__(self, memo):
        # Create a shallow copy of the object
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy all attributes except the Gurobi model
        for k, v in self.__dict__.items():
            if k == "master":
                # Create a new MasterModel instance instead of deep copying
                result.master = self.master.copy()
            else:
                setattr(result, k, deepcopy(v, memo))

        return result


# -----------------------------------------------------------------------
#  Build instance + run
# -----------------------------------------------------------------------
def build_instance():
    random.seed(0)
    demand = {i: [random.randint(1, 7) for _ in range(30)] for i in range(3)}
    setup = {0: 7.5, 1: 9.0, 2: 6.0}
    b_var = {0: 5.0, 1: 5.0, 2: 5.0}
    c_var = {0: 2.0, 1: 3.0, 2: 1.8}
    h = {0: 0.4, 1: 0.6, 2: 0.3}
    cap = [sum(demand[i][t] for i in demand) + 2 for t in range(30)]
    shelf = {0: 3, 1: 3, 2: 3}
    kmax = {i: max(demand[i]) + 3 for i in demand}
    return demand, c_var, h, setup, b_var, cap, shelf, kmax


if __name__ == "__main__":
    dem, cvar, h, setup, bvar, cap, shelf, kmax = build_instance()
    bp = BranchPrice(dem, cvar, h, setup, bvar, cap, shelf, kmax)
    best, sol = bp.branch_and_price()
    print("Objective:", best)
    for i in bp.items:
        sel = [idx for idx, v in enumerate(sol[i]) if v > 0.9][0]
        print(f"Item {i}: pattern {sel}, orders={bp.master.patterns[i][sel]['q']}")
