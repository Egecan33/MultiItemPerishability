"""
Branch-and-price for a 3-item, 30-period perishable lot-sizing instance
using Gurobi to solve the restricted master problem.

Shelf life = 3 (inventory ages 1 and 2 are tracked; age-3 expires).
One global capacity constraint per period.
Ordering cost c_i, holding cost h_i per item.
"""

from __future__ import annotations
import itertools, random, sys
import numpy as np


# ---------------------------------------------------------------------------
# Dynamic-programming pricing routine with setup costs and optional back-orders
# ---------------------------------------------------------------------------
def dp_pricing_general(
    demand, c_var, h, setup, b_var, mu, pi, shelf_life, k_max, allow_backorder=False
):
    """
    Returns (reduced_cost , true_cost , q_plan , leftover_plan)
    setup   : fixed cost whenever q>0
    b_var   : per-unit back-order cost  (use 0 if not allowed)
    allow_backorder : if True we let state components be negative down to −k_max
    """
    T = len(demand)
    L = shelf_life
    rng = range(-k_max, k_max + 1) if allow_backorder else range(k_max + 1)
    start = tuple(0 for _ in range(L - 1))
    dp = [{} for _ in range(T + 2)]
    dp[T + 1][start] = (-pi, None)

    for t in range(T, 0, -1):
        d = demand[t - 1]
        mu_t = mu[t - 1]
        new_dp = {}
        for state in itertools.product(rng, repeat=L - 1):
            avail = sum(x for x in state if x > 0)  # only positive inventory counts
            need = d - avail
            min_q = max(0, need)
            best_val, best_dec = float("inf"), None
            for q in range(min_q, k_max + 1):
                total = avail + q
                # allow back-order by letting remaining demand go negative
                rem = d
                inv = list(state)
                for age in range(L - 1, 0, -1):
                    idx = age - 1
                    use = min(max(inv[idx], 0), rem)
                    inv[idx] -= use
                    rem -= use
                use_q = min(q, rem)
                rem -= use_q
                leftover_q = q - use_q
                # back-order (negative inventory of age1) if rem>0
                inv_neg = -rem if rem > 0 else 0
                age1 = leftover_q - inv_neg
                next_state = [age1] + inv[:-1]
                if max(next_state, key=abs) > k_max:
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
                new_dp[state] = (best_val, best_dec)
        dp[t] = new_dp
    if start not in dp[1]:
        return float("inf"), None, None, None
    red_cost, _ = dp[1][start]
    q_plan = []
    pattern_cost = 0.0
    leftover_plan = []
    state = start
    for t in range(1, T + 1):
        q, next_state, true = dp[t][state][1]
        q_plan.append(q)
        leftover_plan.append(next_state)
        pattern_cost += true
        state = next_state
    return red_cost, pattern_cost, q_plan, leftover_plan


# ---------------------------------------------------------------------------
# Gurobi master-problem helpers
# ---------------------------------------------------------------------------
try:
    import gurobipy as grb
except ImportError:
    sys.exit(
        "gurobipy is not installed on this machine.  Please install Gurobi "
        "and ensure you have a license before running this script."
    )


class MasterModel:
    """Restricted master problem handled by Gurobi; rebuilt when new columns appear."""

    def __init__(self, items, T, capacity):
        self.items = items
        self.T = T
        self.capacity = capacity
        self.model = grb.Model("RMP")
        self.model.Params.OutputFlag = 0  # silent
        # storage
        self.lambda_vars = {i: [] for i in items}
        self.patterns = {i: [] for i in items}
        # constraints
        from gurobipy import LinExpr

        self.sel_constr = {
            i: self.model.addConstr(LinExpr() == 1.0, name=f"select_{i}") for i in items
        }
        self.cap_constr = [
            self.model.addConstr(LinExpr() <= capacity[t], name=f"cap_{t}")
            for t in range(T)
        ]

    def add_pattern(self, i, cost, q):
        """Add a column (pattern) for item i."""
        col = grb.Column()
        # coefficient in item-selection constraint
        col.addTerms(1.0, self.sel_constr[i])
        # capacity coefficients
        for t, val in enumerate(q):
            if val != 0:
                col.addTerms(val, self.cap_constr[t])
        var = self.model.addVar(
            obj=cost, column=col, name=f"lam_{i}_{len(self.lambda_vars[i])}"
        )
        self.lambda_vars[i].append(var)
        self.patterns[i].append(dict(cost=cost, q=q))

    def optimize(self):
        self.model.optimize()
        if self.model.Status == grb.GRB.INFEASIBLE:
            return None, None, None, None  # signal infeasible
        if self.model.Status != grb.GRB.OPTIMAL:
            raise RuntimeError("Unexpected Gurobi status")
        pi = {i: self.sel_constr[i].Pi for i in self.items}
        mu = [c.Pi for c in self.cap_constr]
        lam_vals = {i: [v.X for v in self.lambda_vars[i]] for i in self.items}
        return self.model.ObjVal, pi, mu, lam_vals

    def fix_variable(self, i, idx, val):
        var = self.lambda_vars[i][idx]
        var.LB = var.UB = val

    def branch_copy(self):
        # return a deep copy of the model for branching
        copy = MasterModel(self.items, self.T, self.capacity)
        for i in self.items:
            for pat in self.patterns[i]:
                copy.add_pattern(i, pat["cost"], pat["q"])
        copy.model.update()
        return copy

    def pattern_feasible(self, item_idx, pat_idx):
        # quick check: q_t ≤ capacity_t minus min demand of others
        q = self.master.patterns[item_idx][pat_idx]["q"]
        residual = self.capacity.copy()
        for i in self.items:
            if i == item_idx:
                continue
            best = min(self.master.patterns[i], key=lambda p: sum(p["q"]))
            for t in range(self.T):
                residual[t] -= best["q"][t]
        return all(q[t] <= residual[t] for t in range(self.T))


# ---------------------------------------------------------------------------
# Branch-and-price driver with dual stabilisation
# ---------------------------------------------------------------------------
class BranchAndPrice:
    def __init__(self, demand, c_var, h, setup, b_var, capacity, shelf_life, k_max):
        self.prev_mu = [0.0] * len(capacity)  # initialise saved duals
        self.alpha = 0.6  # stabilisation weight
        self.items = list(demand.keys())
        self.demand = demand
        self.c_var = c_var
        self.h = h
        self.setup = setup
        self.b_var = b_var
        self.capacity = capacity
        self.T = len(next(iter(demand.values())))
        self.shelf = shelf_life
        self.k_max = k_max
        self.master = MasterModel(self.items, self.T, self.capacity)
        # initial direct-order patterns
        for i in self.items:
            q = list(demand[i])
            cost = sum(c_var[i] * x for x in q) + setup[i] * sum(1 for x in q if x > 0)
            self.master.add_pattern(i, cost, q)
        self.master.model.update()
        # Ensure each item has at least two patterns by adding a dummy high-cost pattern
        for i in self.items:
            if len(self.master.lambda_vars[i]) == 1:
                bigM = 1e6
                dummy_q = [0] * self.T  # or any capacity-safe vector
                self.master.add_pattern(i, bigM, dummy_q)

    def column_generation(self):
        while True:
            opt = self.master.optimize()
            if opt[0] is None:  # infeasible node, prune
                return float("inf")  # bound = +∞ so it will be pruned
            obj, pi, mu, lam = opt
            # ---------- dual stabilisation -----------------
            mu_hat = [
                self.alpha * m + (1 - self.alpha) * p for m, p in zip(mu, self.prev_mu)
            ]
            self.prev_mu = mu_hat[:]  # save for next iteration
            # -----------------------------------------------
            added = False
            for i in self.items:
                red, cost, q, _ = dp_pricing_general(
                    self.demand[i],
                    self.c_var[i],
                    self.h[i],
                    self.setup[i],
                    self.b_var[i],
                    mu_hat,
                    pi[i],
                    self.shelf[i],
                    self.k_max[i],
                    allow_backorder=False,
                )
                if red < -1e-6:
                    self.master.add_pattern(i, cost, q)
                    added = True
            if not added:
                break
        return obj

    def branch_and_price(self, best_obj=float("inf"), best_sol=None):
        bound = self.column_generation()
        if bound >= best_obj - 1e-6:
            return best_obj, best_sol
        # check integrality
        _, _, _, lam = self.master.optimize()
        frac = None
        for i, vlist in lam.items():
            for idx, val in enumerate(vlist):
                if 1e-6 < val < 1 - 1e-6:
                    frac = (i, idx)
                    break
            if frac:
                break
        if frac is None:  # integer solution
            if bound < best_obj:
                best_obj = bound
                best_sol = lam
            return best_obj, best_sol
        i_b, j_b = frac
        # Branch 1: lambda=1
        child1 = self.master.branch_copy()
        child1.fix_variable(i_b, j_b, 1)
        bp1 = BranchAndPrice(
            self.demand,
            self.c_var,
            self.h,
            self.setup,
            self.b_var,  # ← add these two
            self.capacity,
            self.shelf,
            self.k_max,
        )
        bp1.master = child1
        best_obj, best_sol = bp1.branch_and_price(best_obj, best_sol)
        # Branch 2: lambda=0
        if len(self.master.lambda_vars[i_b]) > 1:  # Ensure at least two patterns exist
            child2 = self.master.branch_copy()
            child2.fix_variable(i_b, j_b, 0)
            bp2 = BranchAndPrice(
                self.demand,
                self.c_var,
                self.h,
                self.setup,
                self.b_var,
                self.capacity,
                self.shelf,
                self.k_max,
            )
            bp2.master = child2
            return bp2.branch_and_price(best_obj, best_sol)
        return best_obj, best_sol


# ---------------------------------------------------------------------------
# Build 3-item, 30-period example with setup and back-order costs
# ---------------------------------------------------------------------------
def build_instance():
    random.seed(42)
    demand = {i: [random.randint(1, 7) for _ in range(30)] for i in range(3)}
    setup = {0: 7.5, 1: 9.0, 2: 6.0}  # fixed cost if any order in period t
    b_var = {0: 5.0, 1: 5.0, 2: 5.0}  # per-unit back-order cost (optional)
    c_var = {0: 2.0, 1: 3.0, 2: 1.8}
    h = {0: 0.4, 1: 0.6, 2: 0.3}
    # Parameters for capacity adjustment
    BUFFER = 2  # add this many "spare" units to each period’s capacity
    TIGHTNESS = 1.00  # multiply capacity by this factor (<1 tightens, >1 loosens)

    capacity = [
        int((sum(demand[i][t] for i in demand) + BUFFER) * TIGHTNESS) for t in range(30)
    ]
    shelf = {0: 3, 1: 3, 2: 3}
    k_max = {i: max(demand[i]) + 3 for i in demand}
    return demand, c_var, h, setup, b_var, capacity, shelf, k_max


if __name__ == "__main__":
    demand, c_var, h, setup, b_var, cap, shelf, k_max = build_instance()
    print("30-period demands (sample):")
    for i in demand:
        print(i, demand[i][:10], "...")
    bp = BranchAndPrice(demand, c_var, h, setup, b_var, cap, shelf, k_max)
    best, sol = bp.branch_and_price()
    print(f"Best integer objective = {best:.2f}")
    for i in bp.items:
        chosen = [
            idx for idx, v in enumerate(bp.master.lambda_vars[i]) if abs(v.X - 1) < 1e-6
        ]
        pat = bp.master.patterns[i][chosen[0]]
        print(f"Item {i}: orders={pat['q']}")
