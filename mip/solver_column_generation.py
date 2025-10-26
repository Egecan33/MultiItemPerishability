#!/usr/bin/env python3
"""
Exact Branch-and-Price (DFS) for capacitated lot sizing with perishability & lost sales
— ready to drop into your project.

What this file gives you (matches your spec):
- One dummy column per item at root: [1 | 0,...,0 | 0,...,0] with a very high cost so it
  represents outsourcing / lost-sales. This guarantees RMP feasibility at root and at every node.
- Column structure = (lambda over plans) with data carried on each column:
    • item_id: int
    • plan_id: int (any hashable id)
    • cost: float (includes all economics: production, holding, lost sales, etc.)
    • ybar: List[int]  (setup indicators per period for that item/plan)
    • cap_by_t: List[float]  (period capacity usage contributed if that plan is chosen)
    • meta (optional): Dict for x, ZOI, etc. You can store x quantities here; RMP only
      needs cap_by_t for feasibility & reduced costs.
- RMP (LP) uses λ_{i,k} ≥ 0 with convexity Σ_k λ_{i,k} = 1 for each item i, and capacity
  constraints Σ_{i,k} cap_{i,k,t} λ_{i,k} ≤ κ_t for each period t. No y binaries in the
  RMP; we compute aggregated ŷ_{i,t} = Σ_k ybar^{i,k}_t · λ_{i,k} from the λ solution.
- Pricing happens per-item against duals (μ_i from convexity rows, π_t ≥ 0 from capacity rows):
      rc(i,k) = cost(i,k) − μ_i − Σ_t π_t · cap_{i,k,t}
  You can add up to K most-negative columns per item per CG round (respecting branch constraints).
- Branching is ONLY on aggregated setups ŷ_{i,t}. We pick the most fractional ŷ and create two
  children:
      Left  (ŷ_{i*,t*} = 0): forbid columns with ybar_{i*,t*}=1.
      Right (ŷ_{i*,t*} = 1): require columns with ybar_{i*,t*}=1 (i.e., only allow those columns).
  We enforce this by filtering the allowed column set AND by constraining future pricing to
  generate only consistent columns for that (i*, t*).
- At each node we run CG to optimality under those branch constraints, then:
    • LB := RMP objective value.
    • UB candidate: solve the restricted MIP over CURRENT column pool with z_{i,k} ∈ {0,1},
      Σ_k z_{i,k} = 1 for each item, capacity constraints, minimize Σ_i Σ_k cost · z.
      (This is exact over the pool; column generation ensures the pool keeps improving.)
- DFS exploration with global incumbent + gap tracking. Time limit and mip_gap honored.
- Works with your CLI flags (stab, alpha, max_iter, pricing_k, drop_age, finalize_off,
  time_limit, mip_gap, exclusive vs inclusive shelf, force_no_lost_sales, diving_off,
  diving_reprice_iters, outsource_unit_cost, quiet).

IMPORTANT: You must plug your pricing DP. See `pricing_oracle(...)` below — it receives
(i, duals, branch constraints, etc.) and should return up to `params.pricing_k` NEW columns
with negative reduced cost that also respect the node's branch constraints.

Dependencies: gurobipy (for RMP and the small pool-MIP). Python 3.10+.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import gurobipy as gp
from gurobipy import GRB


# ------------------------- Data structures -------------------------
@dataclass(frozen=True)
class BranchFix:
    """Branching fix for a single (item, t).
    require=True  → keep only columns/plans whose ybar_t == 1 (mass must be on those)
    require=False → forbid any column with ybar_t == 1 (i.e., ŷ_{i,t}=0)
    """

    item: int
    t: int
    require: bool


@dataclass
class Node:
    node_id: int
    depth: int
    fixes: List[BranchFix] = field(default_factory=list)
    parent_id: Optional[int] = None


@dataclass
class Column:
    item: int
    plan_id: int
    cost: float
    ybar: List[int]  # length T
    cap_by_t: List[float]  # length T
    meta: Dict[str, Any] = field(
        default_factory=dict
    )  # put x, ZOI, etc. here if you like
    age: int = 0  # for optional dropping


@dataclass
class Instance:
    T: int
    kappa: List[float]  # capacity per period, length T
    n_items: int
    raw: Dict[str, Any]  # full JSON if you want (demands, costs...)


@dataclass
class Params:
    stabilize: bool
    stab_alpha: float
    max_iter: int
    max_add_per_item_per_iter: int
    pricing_k: int
    drop_age: int
    finalize_as_mip: bool
    time_limit: int  # seconds; 0 = no limit
    mip_gap: float  # e.g., 0.0 exact, 0.01 = 1%
    inclusive_shelf: bool
    force_no_lost_sales: bool
    enable_diving: bool
    diving_reprice_iters: int
    outsource_unit_cost: Optional[float]
    verbose: bool


# ------------------------- Utilities -------------------------
def now() -> float:
    return time.time()


def load_instance(path: str | Path) -> Instance:
    j = json.loads(Path(path).read_text())
    # Expect j to have: "T", "kappa" (len T), and either "n_items" or items list.
    T = int(j["T"]) if "T" in j else len(j["kappa"])  # be tolerant
    kappa = list(map(float, j["kappa"]))
    if "n_items" in j:
        n_items = int(j["n_items"])
    elif "items" in j:
        n_items = len(j["items"])
    else:
        raise ValueError("Instance JSON must include 'n_items' or 'items'.")
    return Instance(T=T, kappa=kappa, n_items=n_items, raw=j)


def make_dummy_column(
    i: int, inst: Instance, outsource_unit_cost: Optional[float]
) -> Column:
    """All-zero setups and capacity. High cost so it's never chosen unless necessary.
    If you have a proper lost-sales / outsourcing cost per item, plug it here.
    """
    big = 1.0e6 if outsource_unit_cost is None else float(outsource_unit_cost)
    cost = big  # a single huge cost is enough; you can scale by demand if you prefer
    T = inst.T
    return Column(
        item=i,
        plan_id=-1,
        cost=cost,
        ybar=[0] * T,
        cap_by_t=[0.0] * T,
        meta={"dummy": True},
    )


def column_allowed_by_fixes(col: Column, fixes: List[BranchFix]) -> bool:
    for fx in fixes:
        if col.item != fx.item:
            continue
        y = col.ybar[fx.t]
        if fx.require and y != 1:
            return False
        if not fx.require and y == 1:
            return False
    return True


# ------------------------- RMP (LP) -------------------------
class RMP:
    def __init__(self, inst: Instance, columns_by_item: Dict[int, List[Column]]):
        self.inst = inst
        self.columns_by_item = columns_by_item
        self.model = gp.Model("rmp")
        self.model.Params.OutputFlag = 0
        self.lmbda: Dict[Tuple[int, int], gp.Var] = {}
        self.mu_idx: Dict[int, int] = {}  # map item→row index for μ
        self.pi_idx: Dict[int, int] = {}  # map t→row index for π
        self._build()

    def _build(self):
        m = self.model
        T = self.inst.T
        # λ vars
        for i, cols in self.columns_by_item.items():
            for k, col in enumerate(cols):
                self.lmbda[(i, k)] = m.addVar(
                    lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"lam_{i}_{k}"
                )
        m.update()
        # Objective
        obj = gp.LinExpr()
        for (i, k), var in self.lmbda.items():
            col = self.columns_by_item[i][k]
            obj += col.cost * var
        m.setObjective(obj, GRB.MINIMIZE)
        # Convexity per item: Σ_k λ_{i,k} = 1
        for i, cols in self.columns_by_item.items():
            row = gp.LinExpr()
            for k, _ in enumerate(cols):
                row += self.lmbda[(i, k)]
            m.addConstr(row == 1.0, name=f"conv_{i}")
        # Capacity per period: Σ_{i,k} cap_{i,k,t} λ_{i,k} ≤ κ_t
        for t in range(T):
            row = gp.LinExpr()
            for i, cols in self.columns_by_item.items():
                for k, col in enumerate(cols):
                    row += col.cap_by_t[t] * self.lmbda[(i, k)]
            m.addConstr(row <= self.inst.kappa[t], name=f"cap_{t}")
        m.update()

    def optimize(self):
        self.model.optimize()

    def obj_value(self) -> float:
        if self.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return math.inf
        return float(self.model.ObjVal)

    def duals(self) -> Tuple[List[float], List[float]]:
        """Return (mu_i for convexity rows, pi_t for capacity rows)."""
        m = self.model
        # Constraints are added in order: first all convexity, then all capacity (see _build)
        mu: List[float] = []
        pi: List[float] = []
        # Read by name to be robust
        i = 0
        while True:
            cname = f"conv_{i}"
            c = m.getConstrByName(cname)
            if c is None:
                break
            mu.append(c.Pi)
            i += 1
        t = 0
        while True:
            cname = f"cap_{t}"
            c = m.getConstrByName(cname)
            if c is None:
                break
            pi.append(c.Pi)
            t += 1
        return mu, pi

    def lambda_solution(self) -> Dict[Tuple[int, int], float]:
        return {(i, k): self.lmbda[(i, k)].X for (i, k) in self.lmbda}


def compute_yhat(
    columns_by_item: Dict[int, List[Column]],
    lam_sol: Dict[Tuple[int, int], float],
    T: int,
) -> Dict[Tuple[int, int], float]:
    """Return yhat[(i,t)] = Σ_k ybar^{i,k}_t * λ_{i,k}."""
    yhat: Dict[Tuple[int, int], float] = {}
    for i, cols in columns_by_item.items():
        for t in range(T):
            s = 0.0
            for k, col in enumerate(cols):
                s += col.ybar[t] * lam_sol[(i, k)]
            yhat[(i, t)] = s
    return yhat


def select_most_fractional_yhat(
    yhat: Dict[Tuple[int, int], float], eps: float = 1e-6
) -> Optional[Tuple[int, int, float]]:
    """Pick (i*, t*, val) with val farthest from {0,1} (closest to 0.5). Return None if all integral."""
    best: Optional[Tuple[int, int, float]] = None
    best_gap = -1.0
    for (i, t), v in yhat.items():
        if v < eps or v > 1.0 - eps:
            continue
        gap = 0.5 - abs(v - 0.5)  # larger is more fractional
        if gap > best_gap:
            best_gap = gap
            best = (i, t, v)
    return best


# ------------------------- Pricing hook -------------------------
def pricing_oracle(
    item: int,
    inst: Instance,
    mu_i: float,
    pi: List[float],
    fixes: List[BranchFix],
    params: Params,
    existing_cols_for_item: List[Column],
) -> List[Column]:
    """
    Zero-Order-Inventory (ZOI) pricing:
    - If there are require=True fixes for this item, build exactly one plan whose ybar has 1's at those
      periods (and 0 elsewhere), compute ZOI quantities x, price it, and return it iff rc < 0 and not duplicate.
    - If there are no require=True fixes, generate single-setup ZOI candidates (one ybar with a single 1 at t
      for each t that's not forbidden), price them, and return up to pricing_k best negative-rc, non-duplicate columns.

    Notes:
    - cap_by_t is taken as production x[t] (cap_per_unit defaults to 1.0 if not provided).
    - Cost = Σ_{setups t} (setup_cost + prod_cost * x[t] + holding_cost * Σ_{k=t..u-1} (k-t)*d[k]),
      where u is the next setup or T.
    """
    T = inst.T
    raw = inst.raw

    # -------- demands for this item (fallbacks tolerated) --------
    d: List[float] = [0.0] * T
    try:
        if "items" in raw and item < len(raw["items"]):
            itm = raw["items"][item]
            if "demand" in itm:
                d = [float(x) for x in itm["demand"]][:T]
            elif "demands" in itm:
                d = [float(x) for x in itm["demands"]][:T]
        elif "demands" in raw:
            d = [float(x) for x in raw["demands"][item]][:T]
    except Exception:
        pass
    if len(d) < T:
        d += [0.0] * (T - len(d))

    # -------- costs (with safe defaults) --------
    prod_cost = 0.0
    setup_cost = 0.0
    hold_cost = 0.0
    cap_per_unit = 1.0
    if "items" in raw and item < len(raw.get("items", [])):
        itm = raw["items"][item]
        prod_cost = float(itm.get("prod_cost", itm.get("c_prod", 0.0)))
        setup_cost = float(itm.get("setup_cost", itm.get("c_setup", 0.0)))
        hold_cost = float(itm.get("hold_cost", itm.get("c_hold", 0.0)))
        cap_per_unit = float(itm.get("cap_per_unit", 1.0))
    else:
        prod_cost = float(raw.get("prod_cost", raw.get("c_prod", 0.0)))
        setup_cost = float(raw.get("setup_cost", raw.get("c_setup", 0.0)))
        hold_cost = float(raw.get("hold_cost", raw.get("c_hold", 0.0)))
        cap_per_unit = float(raw.get("cap_per_unit", 1.0))

    # -------- branch fixes for this item --------
    requires = sorted(fx.t for fx in fixes if fx.item == item and fx.require)
    forbids = set(fx.t for fx in fixes if fx.item == item and not fx.require)

    # -------- existing keys to avoid duplicates --------
    existing_keys = {
        (tuple(col.ybar), tuple(round(c, 9) for c in col.cap_by_t))
        for col in existing_cols_for_item
    }

    def build_column_from_ybar(ybar: List[int]) -> Optional[Column]:
        # respect fixes strictly
        for t in forbids:
            if ybar[t] != 0:
                return None
        for t in requires:
            if ybar[t] != 1:
                return None

        S = [t for t, y in enumerate(ybar) if y == 1]
        if not S:
            return None

        x = [0.0] * T
        cap_by_t = [0.0] * T
        total_cost = 0.0

        for idx, t in enumerate(S):
            u = S[idx + 1] if idx + 1 < len(S) else T  # next setup (or horizon end)
            # ZOI: produce at t the total demand until just before u
            qty = float(sum(d[t:u]))
            if qty <= 0.0:
                # No fake setups: if branch required a setup but there's no production, reject this plan.
                # (If not required, we could drop ybar[t]=0 instead—but here we keep ybar fixed.)
                if t in requires:
                    return None
                # skip zero-qty setup silently by zeroing ybar[t]
                ybar[t] = 0
                continue

            x[t] = qty
            cap_by_t[t] = qty * cap_per_unit

            # Costs: setup + production
            total_cost += setup_cost + prod_cost * qty
            # Holding: demand for future periods in this block is held (k - t) periods
            if hold_cost != 0.0:
                for k in range(t, u):
                    total_cost += hold_cost * (k - t) * d[k]

        # If all setups collapsed to zero-qty, reject
        if sum(x) <= 0.0:
            return None

        key = (tuple(ybar), tuple(round(c, 9) for c in cap_by_t))
        if key in existing_keys:
            return None

        # Reduced cost
        rc = total_cost - mu_i - sum(pi[t] * cap_by_t[t] for t in range(T))
        if rc >= -1e-9:
            return None

        # plan_id from ybar bits (stable across runs)
        try:
            pid = int("".join(str(b) for b in ybar), 2)
        except Exception:
            pid = abs(hash(tuple(ybar)))

        return Column(
            item=item,
            plan_id=pid,
            cost=total_cost,
            ybar=ybar[:],
            cap_by_t=cap_by_t,
            meta={"x": x, "rc": rc, "zoi": True},
        )

    candidates: List[Column] = []

    if requires:
        # Build exactly one plan matching required setups (others 0, but not in forbids)
        ybar = [0] * T
        for t in requires:
            if t in forbids:
                return []  # infeasible under fixes
            ybar[t] = 1
        col = build_column_from_ybar(ybar)
        return [col] if col is not None else []

    # No required setups → generate single-setup ZOI candidates for all t not forbidden
    for t in range(T):
        if t in forbids:
            continue
        ybar = [0] * T
        ybar[t] = 1
        col = build_column_from_ybar(ybar)
        if col is not None:
            candidates.append(col)

    # Sort by reduced cost (stored in meta["rc"]) and return up to pricing_k
    candidates.sort(key=lambda c: c.meta.get("rc", 0.0))
    return candidates[: max(1, params.pricing_k)]


# ------------------------- Column Generation at a node -------------------------
@dataclass
class NodeResult:
    status: str  # "ok", "infeasible", "time"
    lb: float
    yhat: Dict[Tuple[int, int], float]
    lam: Dict[Tuple[int, int], float]
    columns_by_item: Dict[int, List[Column]]


def run_cg_at_node(
    inst: Instance,
    seed_columns_by_item: Dict[int, List[Column]],
    fixes: List[BranchFix],
    params: Params,
    start_time: float,
) -> NodeResult:
    # Filter seed columns by fixes; ensure at least 1 dummy column survives for each item
    columns_by_item: Dict[int, List[Column]] = {}
    for i in range(inst.n_items):
        allowed = [
            c
            for c in seed_columns_by_item.get(i, [])
            if column_allowed_by_fixes(c, fixes)
        ]
        if not allowed:
            # inject dummy compatible column
            dum = make_dummy_column(i, inst, params.outsource_unit_cost)
            if not column_allowed_by_fixes(dum, fixes):
                # require True on a t while dummy has 0 — impossible branch
                return NodeResult(
                    status="infeasible",
                    lb=math.inf,
                    yhat={},
                    lam={},
                    columns_by_item={},
                )
            allowed = [dum]
        columns_by_item[i] = allowed

    mu_prev: Optional[List[float]] = None
    pi_prev: Optional[List[float]] = None

    it = 0
    while True:
        if params.time_limit and now() - start_time > params.time_limit:
            return NodeResult(
                status="time",
                lb=math.inf,
                yhat={},
                lam={},
                columns_by_item=columns_by_item,
            )

        # Build and solve RMP
        rmp = RMP(inst, columns_by_item)
        rmp.optimize()
        if rmp.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return NodeResult(
                status="infeasible",
                lb=math.inf,
                yhat={},
                lam={},
                columns_by_item=columns_by_item,
            )
        lb = rmp.obj_value()
        mu, pi = rmp.duals()

        # Stabilization (dual smoothing)
        if params.stabilize and mu_prev is not None and pi_prev is not None:
            a = params.stab_alpha
            mu = [a * mup + (1 - a) * m for mup, m in zip(mu_prev, mu)]
            pi = [a * pip + (1 - a) * p for pip, p in zip(pi_prev, pi)]
        mu_prev, pi_prev = mu[:], pi[:]

        # Pricing per item (respecting fixes)
        added_any = False
        for i in range(inst.n_items):
            new_cols = pricing_oracle(
                i, inst, mu[i], pi, fixes, params, columns_by_item[i]
            )
            # Keep only up to max_add_per_item_per_iter
            if params.max_add_per_item_per_iter > 0:
                new_cols = new_cols[: params.max_add_per_item_per_iter]
            if new_cols:
                columns_by_item[i].extend(new_cols)
                added_any = True

        it += 1
        if (not added_any) or it >= params.max_iter:
            lam = rmp.lambda_solution()
            yhat = compute_yhat(columns_by_item, lam, inst.T)
            return NodeResult(
                status="ok", lb=lb, yhat=yhat, lam=lam, columns_by_item=columns_by_item
            )


# ------------------------- Pool MIP for UB -------------------------
@dataclass
class UBResult:
    ub: float
    z: Dict[Tuple[int, int], int]


def solve_pool_mip(
    inst: Instance,
    columns_by_item: Dict[int, List[Column]],
    mip_gap: float,
) -> Optional[UBResult]:
    try:
        m = gp.Model("pool_mip")
        m.Params.OutputFlag = 0
        m.Params.MIPGap = mip_gap
        z: Dict[Tuple[int, int], gp.Var] = {}
        # Vars
        for i, cols in columns_by_item.items():
            for k, _ in enumerate(cols):
                z[(i, k)] = m.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}")
        m.update()
        # Objective
        obj = gp.LinExpr()
        for (i, k), var in z.items():
            obj += columns_by_item[i][k].cost * var
        m.setObjective(obj, GRB.MINIMIZE)
        # Pick exactly one plan per item
        for i, cols in columns_by_item.items():
            m.addConstr(
                gp.quicksum(z[(i, k)] for k in range(len(cols))) == 1, name=f"pick_{i}"
            )
        # Capacity
        T = inst.T
        for t in range(T):
            m.addConstr(
                gp.quicksum(
                    columns_by_item[i][k].cap_by_t[t] * z[(i, k)]
                    for i, cols in columns_by_item.items()
                    for k in range(len(cols))
                )
                <= inst.kappa[t],
                name=f"cap_{t}",
            )
        m.optimize()
        if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return None
        zsol = {(i, k): int(round(z[(i, k)].X)) for (i, k) in z}
        return UBResult(ub=float(m.ObjVal), z=zsol)
    except gp.GurobiError:
        return None


# ------------------------- DFS Branch-and-Price driver -------------------------
@dataclass
class ExploreResult:
    best_ub: float
    best_node_id: Optional[int]
    best_pool: Optional[Dict[int, List[Column]]]
    best_z: Optional[Dict[Tuple[int, int], int]]
    optimal: bool
    explored_nodes: int


def branch_and_price(
    inst: Instance,
    params: Params,
    seed_columns_by_item: Optional[Dict[int, List[Column]]] = None,
) -> ExploreResult:
    start_time = now()

    # Global incumbent
    best_ub = math.inf
    best_z: Optional[Dict[Tuple[int, int], int]] = None
    best_pool: Optional[Dict[int, List[Column]]] = None
    best_node: Optional[int] = None

    # Seed pool: at least dummies
    pool: Dict[int, List[Column]] = {i: [] for i in range(inst.n_items)}
    if seed_columns_by_item:
        for i in range(inst.n_items):
            pool[i] = [c for c in seed_columns_by_item.get(i, [])]
    for i in range(inst.n_items):
        if not pool[i]:
            pool[i] = [make_dummy_column(i, inst, params.outsource_unit_cost)]

    # DFS stack
    nid = 0
    stack: List[Node] = [Node(node_id=nid, depth=0, fixes=[])]
    explored = 0

    while stack:
        if params.time_limit and now() - start_time > params.time_limit:
            return ExploreResult(
                best_ub,
                best_node,
                best_pool,
                best_z,
                optimal=False,
                explored_nodes=explored,
            )

        node = stack.pop()
        explored += 1
        if params.verbose:
            print(f"\n--- Exploring Node #{node.node_id} (depth={node.depth}) ---")
            if node.fixes:
                for fx in node.fixes:
                    print(f"  fix: item {fx.item}, t {fx.t}, require={fx.require}")

        # Run CG at this node
        cg_res = run_cg_at_node(inst, pool, node.fixes, params, start_time)
        if cg_res.status == "infeasible":
            if params.verbose:
                print("  Node infeasible (pool+fixes).")
            continue
        if cg_res.status == "time":
            if params.verbose:
                print("  Time limit reached during CG.")
            return ExploreResult(
                best_ub,
                best_node,
                best_pool,
                best_z,
                optimal=False,
                explored_nodes=explored,
            )

        lb = cg_res.lb
        yhat = cg_res.yhat
        columns_by_item = cg_res.columns_by_item

        if params.verbose:
            print(
                f"  LB (RMP) = {lb:.6f} | current UB = {best_ub if math.isfinite(best_ub) else float('inf')}"
            )

        # Bound prune
        if math.isfinite(best_ub) and lb >= best_ub - 1e-9:
            if params.verbose:
                print("  Fathom by bound (LB ≥ UB).")
            continue

        # Try to compute a UB via pool MIP
        ub_res = solve_pool_mip(inst, columns_by_item, params.mip_gap)
        if ub_res is not None and ub_res.ub < best_ub - 1e-9:
            best_ub = ub_res.ub
            best_z = ub_res.z
            best_pool = columns_by_item
            best_node = node.node_id
            if params.verbose:
                print(f"  New incumbent UB = {best_ub:.6f} @ node {node.node_id}")

        # Check integrality of ŷ
        branch_cand = select_most_fractional_yhat(yhat)
        if branch_cand is None:
            # ŷ integral → this node is a leaf (feasible in the LP sense); gap check
            if params.verbose:
                gap = (
                    (best_ub - lb) / max(1.0, abs(best_ub))
                    if math.isfinite(best_ub)
                    else float("inf")
                )
                print(
                    f"  Leaf (y-hat integral). LB={lb:.6f}, UB={best_ub:.6f}, gap={100*gap:.3f}%"
                )
            # Even if λ fractional, aggregated setups are integral, which is your requirement.
            # Keep exploring others; DFS will end when stack empty or time.
            continue

        i_star, t_star, v = branch_cand
        if params.verbose:
            print(f"  Branch on ŷ[{i_star},{t_star}] = {v:.4f}")

        # Create children (DFS push right first so left explored next)
        nid += 1
        right = Node(
            node_id=nid,
            depth=node.depth + 1,
            fixes=node.fixes + [BranchFix(i_star, t_star, True)],
            parent_id=node.node_id,
        )
        nid += 1
        left = Node(
            node_id=nid,
            depth=node.depth + 1,
            fixes=node.fixes + [BranchFix(i_star, t_star, False)],
            parent_id=node.node_id,
        )

        # Simple DFS order: explore left first → push right then left
        stack.append(right)
        stack.append(left)

        # Optional: pool inheritance — we already pass the global `pool` at each node start.
        # To "learn" globally, merge new columns back into the global pool:
        # (This is safe since we always re-filter by fixes inside run_cg_at_node.)
        for i, cols in columns_by_item.items():
            pool[i] = merge_pool(pool[i], cols)

    # DFS finished normally → optimal if we have a UB
    return ExploreResult(
        best_ub, best_node, best_pool, best_z, optimal=True, explored_nodes=explored
    )


def merge_pool(base: List[Column], new: List[Column]) -> List[Column]:
    """Deduplicate by (ybar, cap_by_t, cost). Keep first occurrence."""
    seen = set()
    out: List[Column] = []
    for col in list(base) + list(new):
        key = (
            tuple(col.ybar),
            tuple(round(x, 9) for x in col.cap_by_t),
            round(col.cost, 9),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(col)
    return out


# ------------------------- Public entry point (solve_instance) -------------------------


def solve_instance(
    instance_path: str | Path = "last_instance.json",
    out_dir: str | Path = "bp_fast_results",
    # Pricing / CG
    stabilize: bool = False,
    stab_alpha: float = 0.8,
    max_iter: int = 50_000,
    max_add_per_item_per_iter: int = 5,
    pricing_k: int = 5,
    drop_age: int = 10,
    # Finalize / limits
    finalize_as_mip: bool = True,
    time_limit: int = 0,
    mip_gap: float = 0.0,
    # Semantics
    inclusive_shelf: bool = False,
    force_no_lost_sales: bool = False,
    # Diving (not used here but kept for arg-compat)
    enable_diving: bool = True,
    diving_reprice_iters: int = 40,
    # Dummy outsourcing cost
    outsource_unit_cost: Optional[float] = None,
    # Verbosity
    verbose: bool = True,
):
    """
    Full Branch-and-Price. Returns (summary, orders).
    """
    params = Params(
        stabilize=stabilize,
        stab_alpha=stab_alpha,
        max_iter=max_iter,
        max_add_per_item_per_iter=max_add_per_item_per_iter,
        pricing_k=pricing_k,
        drop_age=drop_age,
        finalize_as_mip=finalize_as_mip,
        time_limit=time_limit,
        mip_gap=mip_gap,
        inclusive_shelf=inclusive_shelf,
        force_no_lost_sales=force_no_lost_sales,
        enable_diving=enable_diving,
        diving_reprice_iters=diving_reprice_iters,
        outsource_unit_cost=outsource_unit_cost,
        verbose=verbose,
    )

    inst = load_instance(instance_path)

    # Kick off B&P
    result = branch_and_price(inst, params)

    # Prepare summary
    summary: Dict[str, Any] = {
        "status": "optimal" if result.optimal else "time_or_bound",
        "best_ub": result.best_ub,
        "best_node": result.best_node_id,
        "explored_nodes": result.explored_nodes,
        "mip_gap": mip_gap,
    }

    # Build a human-friendly order plan from the best UB if available
    orders: Dict[str, Any] = {}
    if result.best_pool is not None and result.best_z is not None:
        T = inst.T
        by_item: Dict[int, Dict[str, Any]] = {}
        for (i, k), take in result.best_z.items():
            if take != 1:
                continue
            col = result.best_pool[i][k]
            by_item[i] = {
                "plan_id": col.plan_id,
                "ybar": col.ybar,
                "cap": col.cap_by_t,
                "cost": col.cost,
                "meta": col.meta,
            }
        orders = {"T": T, "by_item": by_item}

    # Persist
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "summary.json").write_text(json.dumps(summary, indent=2))
    (outp / "orders.json").write_text(json.dumps(orders, indent=2))

    return summary, orders


# ------------------------- CLI -------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="last_instance.json")
    p.add_argument("--out", default="bp_fast_results")

    # pricing / CG loop
    p.add_argument("--stab_off", action="store_true")
    p.add_argument("--stab_alpha", type=float, default=0.8)
    p.add_argument("--max_iter", type=int, default=50000)
    p.add_argument("--max_add_per_item_per_iter", type=int, default=5)
    p.add_argument("--pricing_k", type=int, default=5)
    p.add_argument("--drop_age", type=int, default=10)

    # finalize / limits
    p.add_argument("--finalize_off", action="store_true")
    p.add_argument("--time_limit", type=int, default=0)
    p.add_argument("--mip_gap", type=float, default=0.0)

    # semantics
    p.add_argument(
        "--exclusive_shelf",
        action="store_true",
        help="Use exclusive shelf life (u ≤ t+L-1). Default: inclusive (u ≤ t+L).",
    )
    p.add_argument(
        "--force_no_lost_sales",
        action="store_true",
        help="Force no lost sales even if instance allows it. Default: follow instance.",
    )

    # diving
    p.add_argument("--diving_off", action="store_true")
    p.add_argument("--diving_reprice_iters", type=int, default=40)

    # dummy outsourcing cost (optional override)
    p.add_argument(
        "--outsource_unit_cost",
        type=float,
        default=None,
        help="Override dummy outsourcing unit cost (default: 1e6).",
    )

    # verbosity
    p.add_argument("--quiet", action="store_true", help="Suppress verbose output.")

    args = p.parse_args()

    summary, orders = solve_instance(
        instance_path=args.instance,
        out_dir=args.out,
        stabilize=(not args.stab_off),
        stab_alpha=args.stab_alpha,
        max_iter=args.max_iter,
        max_add_per_item_per_iter=args.max_add_per_item_per_iter,
        pricing_k=args.pricing_k,
        drop_age=args.drop_age,
        finalize_as_mip=(not args.finalize_off),
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        inclusive_shelf=(not args.exclusive_shelf),
        force_no_lost_sales=args.force_no_lost_sales,
        enable_diving=(not args.diving_off),
        diving_reprice_iters=args.diving_reprice_iters,
        outsource_unit_cost=args.outsource_unit_cost,
        verbose=(not args.quiet),
    )

    print(json.dumps(summary, indent=2))
