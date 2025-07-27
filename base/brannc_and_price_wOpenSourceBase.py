"""
Advanced branch-and-price implementation for multi-item perishable lot-sizing.

This module generalises previous prototypes by allowing an arbitrary number of
items, a flexible planning horizon, and configurable shelf life.  It uses
a dynamic-programming pricing routine to generate replenishment patterns and
a restricted master problem solved via linear programming to select among
these patterns.  A simple branch-and-price method enforces integrality.

Key features:

* **General shelf life:** The dynamic-programming pricing routine can handle
  any shelf-life length L ≥ 2 by tracking inventory at each age from 1 to L−1.
* **Multiple items:** The solver accepts dictionaries of demands, costs, and
  holding costs for any number of items.  Each item may have its own shelf
  life and DP state bound (k_max).
* **Global capacity constraints:** A per-period capacity limit on total
  orders across all items is enforced in the master problem.
* **Branch-and-price:** The solver includes a simple branch-and-price
  mechanism to obtain integer solutions.

Limitations:

* **Single supplier:** Only one supplier with immediate delivery is
  implemented.  Extending to multiple suppliers with lead times would
  require augmenting the DP state and transition logic.
* **Linear costs:** Only unit ordering costs and holding costs are
  considered.  Setup costs and dynamic pricing/revenue are not included.
* **Open-source solver:** The code uses SciPy’s linear programming solver
  (`scipy.optimize.linprog`).  Commercial solvers like Gurobi are not
  available in this environment.

This module is designed to handle small to moderate instances for
illustrative purposes.  Larger instances may require additional
optimization and heuristics.
"""

from __future__ import annotations
import itertools
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linprog


def dp_pricing_general(
    demand: List[int],
    c_var: float,
    h: float,
    mu: List[float],
    pi: float,
    shelf_life: int,
    k_max: int,
) -> Tuple[float, float, List[int], List[Tuple[int, ...]]]:
    """Dynamic-programming pricing routine for arbitrary shelf life.

    Parameters
    ----------
    demand : list[int]
        Demand over planning horizon (length T).
    c_var : float
        Unit ordering cost.
    h : float
        Holding cost per unit inventory.
    mu : list[float]
        Dual variables for capacity constraints (length T).
    pi : float
        Dual variable for the pattern-selection constraint.
    shelf_life : int
        Shelf life (L ≥ 2).  Inventory can be kept for up to L−1 periods.
    k_max : int
        Maximum inventory allowed per age state.

    Returns
    -------
    reduced_cost : float
        Reduced cost of the best pattern.
    pattern_cost : float
        True cost of the best pattern (without dual terms).
    q_plan : list[int]
        Ordered quantities per period in the pattern.
    leftover_plan : list[tuple[int, ...]]
        Leftover inventory vectors (ages 1..L-1) for each period.
    """
    T = len(demand)
    L = shelf_life
    # DP state dimensions: period (1..T+1) x (inv1..invL-1)
    # We'll use a dictionary to store DP values to avoid constructing huge arrays
    # State is a tuple of length L-1 representing inventory at ages 1..L-1
    # Base case at period T+1: cost = -pi
    from collections import defaultdict

    # Using dictionary-of-dictionary: dp[t][state] -> value
    dp: List[Dict[Tuple[int, ...], float]] = [
        defaultdict(lambda: np.inf) for _ in range(T + 2)
    ]
    decision: List[Dict[Tuple[int, ...], Tuple[int, Tuple[int, ...], float]]] = [
        defaultdict(lambda: None) for _ in range(T + 1)
    ]
    # base case
    base_state = tuple(0 for _ in range(L - 1))
    dp[T + 1][base_state] = -pi

    # backward induction
    # generate all possible inventory states up to k_max per age
    # For performance, we enumerate states on demand
    for t in range(T, 0, -1):
        mu_t = mu[t - 1]
        d_t = demand[t - 1]
        # For each state reachable at period t+1, consider predecessor states
        # To avoid exploring huge state space, we iterate over states present in dp[t+1]
        for state_next, val_next in dp[t + 1].items():
            # Predecessor states can lead to state_next after ordering and consumption
            # We need to invert the inventory aging: state_next = (inv0_next, inv1_next, ..., invL-2_next)
            # At period t, we have state = (inv0, inv1, ..., invL-2).  After consuming demand and aging,
            # inv0_next = leftover_q, inv1_next = leftover_inv0, inv2_next = leftover_inv1, ..., invL-2_next = leftover_invL-3.
            # Let state = (a1, a2, ..., aL-1) where a1 is age1 inventory, a2 age2,...,aL-1 ageL-1 (oldest).  invL-1
            # After consumption, leftover_invL-1 becomes waste.  invL-2_next = leftover_invL-3.  state_next
            # We need to find all states and q such that transition yields state_next.
            # However, enumerating predecessors is complex.  Instead, we iterate over all possible state pre.
            pass
        # The above predecessor enumeration is expensive; instead we iterate over all possible states at period t
        # This yields O(k_max^(L-1)) states; feasible for small k_max and L.
        # Build dp[t] by iterating over all states and all feasible q.
        # We'll derive transitions directly from current state to next state.
        # Temporary dict for dp[t]
        dp_t = defaultdict(lambda: np.inf)
        decision_t = {}
        # Iterate over possible inventory states at period t
        # For each age a1..aL-1 with 0..k_max
        for state in itertools.product(range(k_max + 1), repeat=L - 1):
            # available inventory before ordering
            available = sum(state)
            min_q = max(0, d_t - available)
            best_val = np.inf
            best_decision = None
            # Consider q from min_q..k_max
            for q in range(min_q, k_max + 1):
                total_available = available + q
                if total_available < d_t:
                    continue
                # copy state to mutable list to represent leftover inventory per age
                leftover = list(state)
                remaining_d = d_t
                # consume from oldest age first (age L-1 down to 1)
                for age in range(L - 1, 0, -1):
                    idx = age - 1
                    use = min(leftover[idx], remaining_d)
                    leftover[idx] -= use
                    remaining_d -= use
                    if remaining_d == 0:
                        break
                # consume from new order q
                use_q = remaining_d
                if use_q > q:
                    continue
                leftover_q = q - use_q
                # Compute next state: age1_next = leftover_q, age2_next = leftover[0], age3_next = leftover[1], ...
                next_state_list = [leftover_q] + leftover[:-1]
                next_state = tuple(min(x, k_max) for x in next_state_list)
                # true cost and reduced cost
                true_cost = c_var * q + h * sum(next_state)
                reduced_cost = (
                    (c_var - mu_t) * q
                    + h * sum(next_state)
                    + dp[t + 1].get(next_state, np.inf)
                )
                if reduced_cost < best_val:
                    best_val = reduced_cost
                    best_decision = (q, next_state, true_cost)
            if best_decision is not None:
                dp_t[state] = best_val
                decision_t[state] = best_decision
        dp[t] = dp_t
        decision[t] = decision_t
    # Start state at period 1 is zero inventory
    start_state = tuple(0 for _ in range(L - 1))
    if start_state not in dp[1] or dp[1][start_state] == np.inf:
        return np.inf, None, None, None
    reduced_cost = dp[1][start_state]
    # Reconstruct pattern
    q_plan: List[int] = []
    leftover_plan: List[Tuple[int, ...]] = []
    pattern_cost = 0.0
    state = start_state
    for t in range(1, T + 1):
        q, next_state, true_cost = decision[t][state]
        q_plan.append(int(q))
        leftover_plan.append(next_state)
        pattern_cost += true_cost
        state = next_state
    return reduced_cost, pattern_cost, q_plan, leftover_plan


class MultiItemBP:
    """Branch-and-price solver for a general multi-item perishable lot-sizing problem."""

    def __init__(
        self,
        demand: Dict[int, List[int]],
        c_var: Dict[int, float],
        h: Dict[int, float],
        capacity: List[int],
        shelf_life: Dict[int, int],
        k_max: Optional[Dict[int, int]] = None,
    ):
        self.items = list(demand.keys())
        self.demand = demand
        self.c_var = c_var
        self.h = h
        self.capacity = capacity
        self.T = len(next(iter(demand.values())))
        self.shelf_life = shelf_life
        # Determine k_max per item
        self.k_max = {}
        for i in self.items:
            if k_max and i in k_max:
                self.k_max[i] = k_max[i]
            else:
                self.k_max[i] = max(demand[i]) + 2
        # patterns per item
        self.patterns: Dict[int, List[Dict[str, object]]] = {i: [] for i in self.items}
        # initialise with direct-order patterns
        for i in self.items:
            q_plan = list(self.demand[i])
            cost = sum(self.c_var[i] * q for q in q_plan)
            self.patterns[i].append({"cost": cost, "q": q_plan})

    def solve_master(
        self,
    ) -> Tuple[float, Dict[int, List[float]], Dict[int, float], List[float]]:
        num_vars = sum(len(self.patterns[i]) for i in self.items)
        c = []
        A_eq = []
        b_eq = []
        A_ub = [[] for _ in range(self.T)]
        b_ub = list(self.capacity)
        var_map = []
        # Build eq constraints
        col_idx = 0
        for i in self.items:
            row = [0.0] * num_vars
            for j, pattern in enumerate(self.patterns[i]):
                c.append(pattern["cost"])
                var_map.append((i, j))
                row[col_idx] = 1.0
                col_idx += 1
            A_eq.append(row)
            b_eq.append(1.0)
        # build capacity constraints
        col_idx = 0
        for i in self.items:
            for j, pattern in enumerate(self.patterns[i]):
                for t in range(self.T):
                    A_ub[t].append(pattern["q"][t])
                col_idx += 1
        bounds = [(0, None)] * num_vars
        c = np.array(c)
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        res = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
        )
        if not res.success:
            raise RuntimeError("Master problem infeasible: " + res.message)
        x = res.x
        dual_pi = {i: res.eqlin.marginals[idx] for idx, i in enumerate(self.items)}
        dual_mu = list(res.ineqlin.marginals)
        lambda_vals = {i: [] for i in self.items}
        for idx, (i, j) in enumerate(var_map):
            lambda_vals[i].append(x[idx])
        return res.fun, lambda_vals, dual_pi, dual_mu

    def column_generation(
        self, max_iters: int = 50, tol: float = 1e-6
    ) -> Tuple[float, Dict[int, List[float]], Dict[int, List[Dict[str, object]]]]:
        for iteration in range(max_iters):
            lp_val, lambda_vals, dual_pi, dual_mu = self.solve_master()
            print(f"CG iteration {iteration}: LP = {lp_val:.2f}")
            added = False
            for i in self.items:
                reduced_cost, pattern_cost, q_plan, leftover_plan = dp_pricing_general(
                    self.demand[i],
                    self.c_var[i],
                    self.h[i],
                    dual_mu,
                    dual_pi[i],
                    shelf_life=self.shelf_life[i],
                    k_max=self.k_max[i],
                )
                if reduced_cost < -tol and q_plan is not None:
                    self.patterns[i].append({"cost": pattern_cost, "q": q_plan})
                    print(
                        f"  Added column for item {i}: reduced cost {reduced_cost:.3f}, cost {pattern_cost:.2f}, q = {q_plan}"
                    )
                    added = True
            if not added:
                print("No improving columns found.")
                break
        final_lp, lambda_vals, dual_pi, dual_mu = self.solve_master()
        return final_lp, lambda_vals, self.patterns

    def branch_and_price(
        self,
        depth: int = 0,
        best_obj: float = np.inf,
        best_solution: Optional[Dict[int, int]] = None,
        patterns_override: Optional[Dict[int, List[Dict[str, object]]]] = None,
    ) -> Tuple[float, Optional[Dict[int, int]]]:
        # Optionally override patterns at this node
        if patterns_override is not None:
            original_patterns = self.patterns
            self.patterns = {i: list(patterns_override[i]) for i in self.items}
        # Solve CG at this node
        lp_val, lambda_vals, current_patterns = self.column_generation()
        # Check integrality
        integral = True
        chosen = {}
        branch_item = None
        branch_idx = None
        for i in self.items:
            # Look for a variable with lambda around 1
            idx_one = [idx for idx, v in enumerate(lambda_vals[i]) if v > 1 - 1e-6]
            if len(idx_one) == 1:
                chosen[i] = idx_one[0]
            else:
                integral = False
                # pick fractional var with largest fractional value
                max_frac = 0.0
                frac_index = None
                for idx, v in enumerate(lambda_vals[i]):
                    if 1e-6 < v < 1 - 1e-6 and v > max_frac:
                        max_frac = v
                        frac_index = idx
                if frac_index is not None:
                    branch_item = i
                    branch_idx = frac_index
                    break
        if integral:
            if lp_val < best_obj:
                best_obj = lp_val
                best_solution = chosen
            if patterns_override is not None:
                self.patterns = original_patterns
            return best_obj, best_solution
        # prune if bound worse than best
        if lp_val >= best_obj:
            if patterns_override is not None:
                self.patterns = original_patterns
            return best_obj, best_solution
        # Branch: Node1 fix var = 1, Node2 fix var = 0
        # Node1: only keep pattern branch_idx for item branch_item
        patterns1 = {i: list(self.patterns[i]) for i in self.items}
        patterns1[branch_item] = [patterns1[branch_item][branch_idx]]
        best_obj, best_solution = self.branch_and_price(
            depth + 1, best_obj, best_solution, patterns_override=patterns1
        )
        # Node2: remove pattern branch_idx
        patterns2 = {i: list(self.patterns[i]) for i in self.items}
        patterns2[branch_item] = [
            p for idx, p in enumerate(patterns2[branch_item]) if idx != branch_idx
        ]
        best_obj, best_solution = self.branch_and_price(
            depth + 1, best_obj, best_solution, patterns_override=patterns2
        )
        if patterns_override is not None:
            self.patterns = original_patterns
        return best_obj, best_solution


def example_usage():
    # Example with 3 items, 6 periods, shelf life 3
    import random

    random.seed(0)
    demand = {
        0: [random.randint(2, 5) for _ in range(6)],
        1: [random.randint(1, 4) for _ in range(6)],
        2: [random.randint(3, 6) for _ in range(6)],
    }
    c_var = {0: 2.0, 1: 3.5, 2: 1.8}
    h = {0: 0.4, 1: 0.6, 2: 0.3}
    # Set capacity high enough to make initial patterns feasible
    capacity = [20] * 6
    shelf_life = {0: 3, 1: 3, 2: 3}
    solver = MultiItemBP(demand, c_var, h, capacity, shelf_life)
    print("Demands:", demand)
    lp_val, lambda_vals, patterns = solver.column_generation()
    print("LP value after CG:", lp_val)
    for i in solver.items:
        print(f"Item {i} patterns ({len(patterns[i])}):")
        for idx, pat in enumerate(patterns[i]):
            print(f"  {idx}: cost {pat['cost']}, q={pat['q']}")
    best_obj, best_sol = solver.branch_and_price()
    print("Best integer objective:", best_obj)
    print("Selected patterns:", best_sol)
    for i in solver.items:
        idx = best_sol.get(i, None)
        if idx is not None:
            pat = solver.patterns[i][idx]
            print(f"Item {i}: q={pat['q']}, cost={pat['cost']}")


if __name__ == "__main__":
    example_usage()
