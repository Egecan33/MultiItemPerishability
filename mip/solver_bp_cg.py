# from __future__ import annotations
# import time, json
# from typing import Dict, List, Tuple, Optional
# from pathlib import Path
# import gurobipy as gp
# from gurobipy import GRB


# # ---------- small helpers ----------
# def _as_len_T_vector(val, T: int) -> List[float]:
#     if isinstance(val, (int, float)):
#         return [float(val)] * T
#     if isinstance(val, list):
#         if len(val) != T:
#             raise ValueError(f"Expected length-{T} list, got {len(val)}")
#         return [float(x) for x in val]
#     raise TypeError("Expected number or length-T list")


# def _vectorize(item: dict, key: str, T: int, default: float = 0.0) -> List[float]:
#     v = item.get(key, default)
#     if isinstance(v, (int, float)):
#         return [float(v)] * T
#     if isinstance(v, list):
#         if len(v) != T:
#             raise ValueError(f"items[*]['{key}'] must have length {T}")
#         return [float(x) for x in v]
#     return [float(default)] * T


# def _hsum(h: List[float] | float, t: int, u: int) -> float:
#     """Holding cost from t to u-1 (inclusive)."""
#     if isinstance(h, list):
#         return float(sum(h[k] for k in range(t, u)))
#     return float(h) * max(0, u - t)


# def _pattern_carry_from_arcs(T: int, arcs: Dict[Tuple[int, int], float]) -> List[float]:
#     """inv_after_u[u] = inventory carried past u (i.e., available at end of u)."""
#     inv = [0.0] * T
#     for (t, w), val in arcs.items():
#         # For each arc t->w, it contributes to carry for all u in [t..w-1]
#         for u in range(t, w):
#             inv[u] += val
#     return inv


# class Pattern:
#     """Column for one item: exact feasibility for C2,C3,C4,C5,C1b and coefficients for C1,C1c."""

#     __slots__ = ("cost", "q", "delta", "arcs", "inv_after_u", "ls")

#     def __init__(
#         self,
#         cost: float,
#         q: List[float],
#         delta: List[int],
#         arcs: Dict[Tuple[int, int], float],
#         inv_after_u: List[float],
#         ls: List[float],
#     ):
#         self.cost = float(cost)  # master obj coeff
#         self.q = list(q)  # q_{i,t}
#         self.delta = list(delta)  # δ_{i,t}
#         self.arcs = dict(arcs)  # (t,u) → qty
#         self.inv_after_u = list(inv_after_u)  # carry past u (for whcap rows)
#         self.ls = list(ls)  # lost sales per u (for reporting)


# # ======================================================================
# #                         SOLVER: Branch & Price
# # ======================================================================
# def solve_instance_bp_cg_setups(
#     instance_path: str | Path = "last_instance.json",
#     time_limit: int = 0,  # wallclock seconds (0 = unlimited)
#     mip_gap: float = 0.0,  # unused (LP master; pricing is exact MIP)
#     out_dir: str | Path = "bp_cg_setups_results",
#     cg_tol: float = 1e-7,  # reduced-cost tolerance
#     integrality_tol: float = 1e-6,  # y-integrality check
#     max_nodes: int = 10_000,  # branch-and-price node cap
# ):
#     """
#     Exact Branch-and-Price on setups (δ), with column generation.
#     Enforces the identical core constraints C1–C5 as your MIP:

#     C1  : Global capacity              ∑_i q_{i,t} <= κ_t
#     C1b : Per-item cap                 ∑_u X_{i,t,u} <= p_{i,t}
#     C1c : Warehouse carry cap          ∑_i inv_after_u(i) <= W
#     C2  : Setup linking                ∑_u X_{i,t,u} <= μ_{i,t} * δ_{i,t}, μ_{i,t}=∑_{u∈Γ(i,t)} d_{i,u}
#     C3  : Demand (+lost sales if on)   ∑_{t<=u} X_{i,t,u} (+LS) = d_{i,u}
#     C4  : Arc activation               X_{i,t,u} <= d_{i,u} * Z_{i,t,u}
#     C5  : LEFO no-crossing             pairwise cuts on Z per Expiry

#     Returns (summary, orders_txt) and writes summary.json + orders.txt under out_dir.
#     """
#     t0 = time.time()
#     data = json.loads(Path(instance_path).read_text())
#     T = int(data["period"])
#     Periods = list(range(T))
#     items_raw: Dict[int, dict] = {int(k): v for k, v in data["items"].items()}

#     # ------------- capacity vector κ_t (C1) -------------
#     prod_cap = data.get("production_capacity") or data.get("manual_capacity")
#     if prod_cap is None:
#         cap_raw = [0.0] * T
#         for it in items_raw.values():
#             dem = it["demand"]
#             for t in Periods:
#                 cap_raw[t] += float(dem[t])
#         buf = max(5.0, 0.2 * max(cap_raw) if cap_raw else 0.0)
#         prod_cap = [float(c + buf) for c in cap_raw]
#     else:
#         prod_cap = _as_len_T_vector(prod_cap, T)

#     # ------------- per item vectors -------------
#     demand: Dict[int, List[float]] = {}
#     setup: Dict[int, List[float]] = {}
#     cvar: Dict[int, List[float]] = {}
#     hold: Dict[int, List[float]] = {}
#     bvar: Dict[int, List[float]] = {}

#     for i, it in items_raw.items():
#         demand[i] = _vectorize(it, "demand", T, 0.0)
#         setup[i] = _vectorize(it, "setup", T, 0.0)
#         cvar[i] = _vectorize(it, "c_var", T, 0.0)
#         hold[i] = _vectorize(it, "h", T, 0.0)
#         bvar[i] = _vectorize(it, "b_var", T, 0.0)

#     # ------------- Gamma (feasible arcs), Expiry, Triples, μ_it -------------
#     Gamma: Dict[Tuple[int, int], List[int]] = {}
#     Expiry: Dict[Tuple[int, int], int] = {}
#     Triples: List[Tuple[int, int, int]] = []
#     for i, it in items_raw.items():
#         mseq = list(it["shelf_seq"])
#         if len(mseq) != T:
#             raise ValueError(f"items[{i}]['shelf_seq'] must have length {T}")
#         for t in Periods:
#             m_it = int(mseq[t])
#             v_it = t + m_it
#             Expiry[(i, t)] = v_it
#             if m_it <= 0:
#                 Gamma[(i, t)] = []
#                 continue
#             u_max = min(T - 1, v_it - 1)
#             us = [u for u in range(t, u_max + 1)]
#             Gamma[(i, t)] = us
#             for u in us:
#                 Triples.append((i, t, u))

#     mu_bound: Dict[Tuple[int, int], float] = {}
#     for i in items_raw:
#         d = demand[i]
#         for t in Periods:
#             mu_bound[(i, t)] = float(sum(d[u] for u in Gamma.get((i, t), [])))

#     # ------------- per-item cap p_it (C1b) -------------
#     item_cap_raw = data.get("item_capacity")
#     per_item_cap: Dict[Tuple[int, int], float] = {}
#     if item_cap_raw is not None:
#         if isinstance(item_cap_raw, dict):
#             for i_str, cap in item_cap_raw.items():
#                 i0 = int(i_str)
#                 vec = _as_len_T_vector(cap, T)
#                 for t in Periods:
#                     per_item_cap[(i0, t)] = float(vec[t])
#         else:
#             vec = _as_len_T_vector(item_cap_raw, T)
#             for i in items_raw:
#                 for t in Periods:
#                     per_item_cap[(i, t)] = float(vec[t])

#     # ------------- warehouse cap W (C1c) -------------
#     W = data.get("warehouse_capacity", None)
#     W = float(W) if W is not None else None

#     # ------------- lost sales penalties (C3 soft) -------------
#     allow_lost_sales = bool(
#         data.get("allow_unmet_demand", False) or data.get("allow_lost_sales", False)
#     )
#     loss_penalty_global = data.get("lost_sales_penalty", None)
#     loss_penalty_factor = float(data.get("lost_sales_penalty_factor", 200.0))
#     if allow_lost_sales:
#         max_unit_var_cost = 0.0
#         for i, t, u in Triples:
#             max_unit_var_cost = max(
#                 max_unit_var_cost, cvar[i][t] + _hsum(hold[i], t, u)
#             )
#         if max_unit_var_cost <= 0.0:
#             max_unit_var_cost = 1.0
#         max_setup_seen = max(max(setup[i]) for i in items_raw)
#         default_loss_penalty = loss_penalty_global
#         if default_loss_penalty is None:
#             base = max_unit_var_cost + max_setup_seen
#             default_loss_penalty = max(
#                 10.0 * max_unit_var_cost, loss_penalty_factor * base
#             )
#             default_loss_penalty = float(min(default_loss_penalty + 1.0, 1e9))

#         loss_pen = {}
#         for i in items_raw:
#             lp = items_raw[i].get("lost_sales_penalty", None)
#             if lp is not None:
#                 vec = _as_len_T_vector(lp, T)
#                 for u in Periods:
#                     loss_pen[(i, u)] = float(vec[u])
#             else:
#                 for u in Periods:
#                     loss_pen[(i, u)] = float(default_loss_penalty)
#     else:
#         loss_pen = {}

#     # ------------- arc cost (var + hold) -------------
#     def arc_cost(i: int, t: int, u: int) -> float:
#         return cvar[i][t] + _hsum(hold[i], t, u)

#     # ------------- branching bookkeeping -------------
#     forced: Dict[int, set] = {i: set() for i in items_raw}  # δ_{i,t}=1
#     forbidden: Dict[int, set] = {i: set() for i in items_raw}  # δ_{i,t}=0

#     # ------------- initial pattern (on-demand) -------------
#     def initial_pattern(i: int) -> Pattern:
#         arcs = {}
#         q = [0.0] * T
#         delta = [0] * T
#         ls = [0.0] * T
#         cost = 0.0
#         for u in Periods:
#             d = demand[i][u]
#             if d > 0 and u in Gamma.get((i, u), []):
#                 arcs[(u, u)] = d
#                 q[u] += d
#                 delta[u] = 1
#                 cost += setup[i][u] + d * arc_cost(i, u, u)
#             elif d > 0 and allow_lost_sales:
#                 ls[u] = d
#                 cost += loss_pen[(i, u)] * d
#         inv_after_u = _pattern_carry_from_arcs(T, arcs)
#         return Pattern(cost, q, delta, arcs, inv_after_u, ls)

#     patterns: Dict[int, List[Pattern]] = {i: [initial_pattern(i)] for i in items_raw}

#     # ------------- MASTER (RMP) builder -------------
#     def build_rmp():
#         m = gp.Model("bp_rmp")
#         m.Params.OutputFlag = 1
#         if time_limit:
#             m.Params.TimeLimit = max(1, int(time_limit - (time.time() - t0)))

#         # choose exactly one pattern per item (LP relaxation allows mixing)
#         choose = {
#             i: m.addConstr(gp.LinExpr() == 1.0, name=f"choose_{i}") for i in items_raw
#         }

#         # C1: capacity rows (hard, no slack)
#         cap_rows = {
#             t: m.addConstr(gp.LinExpr() <= float(prod_cap[t]), name=f"cap_{t}")
#             for t in Periods
#         }

#         # C1c: warehouse carry rows (optional)
#         wh_rows = {}
#         if W is not None:
#             for u in Periods[:-1]:
#                 wh_rows[u] = m.addConstr(gp.LinExpr() <= float(W), name=f"whcap_{u}")
#         else:
#             wh_rows = None

#         # λ variables (by column)
#         lam: Dict[Tuple[int, int], gp.Var] = {}
#         for i in items_raw:
#             for p_idx, pat in enumerate(patterns[i]):
#                 col = gp.Column()
#                 col.addTerms(1.0, choose[i])
#                 for t, q_t in enumerate(pat.q):
#                     if q_t != 0.0:
#                         col.addTerms(float(q_t), cap_rows[t])
#                 if wh_rows is not None:
#                     for u in Periods[:-1]:
#                         iv = pat.inv_after_u[u]
#                         if iv != 0.0:
#                             col.addTerms(float(iv), wh_rows[u])
#                 v = m.addVar(
#                     lb=0.0, obj=float(pat.cost), name=f"lam_i{i}_p{p_idx}", column=col
#                 )
#                 lam[(i, p_idx)] = v

#         # y expressions for branching: y_{i,t} = ∑_p λ_{i,p} * δ_{i,t}^{(p)}
#         y_expr: Dict[Tuple[int, int], gp.LinExpr] = {}
#         for i in items_raw:
#             for t in Periods:
#                 expr = gp.LinExpr()
#                 for p_idx, pat in enumerate(patterns[i]):
#                     if pat.delta[t] != 0:
#                         expr.addTerms(float(pat.delta[t]), lam[(i, p_idx)])
#                 y_expr[(i, t)] = expr

#         # apply branching cuts accumulated so far
#         for i in items_raw:
#             for t in forced[i]:
#                 m.addConstr(y_expr[(i, t)] >= 1.0, name=f"force_i{i}_t{t}")
#             for t in forbidden[i]:
#                 m.addConstr(y_expr[(i, t)] <= 0.0, name=f"forbid_i{i}_t{t}")

#         m.ModelSense = GRB.MINIMIZE
#         m.update()
#         return m, lam, choose, cap_rows, wh_rows, y_expr

#     # ------------- PRICING (exact MIP) for one item: enforces C2,C1b,C4,C5,C3 -------------
#     def price_item(
#         i: int, mu: List[float], theta: Optional[List[float]]
#     ) -> Tuple[float, Optional[Pattern]]:
#         m = gp.Model(f"price_item_{i}")
#         m.Params.OutputFlag = 0
#         if time_limit:
#             remain = max(1, int(time_limit - (time.time() - t0)))
#             m.Params.TimeLimit = remain

#         X, Z, Delta = {}, {}, {}
#         LS = {} if allow_lost_sales else None

#         # variables
#         for t in Periods:
#             for u in Gamma.get((i, t), []):
#                 X[(t, u)] = m.addVar(lb=0.0, name=f"x_{t}_{u}")
#                 Z[(t, u)] = m.addVar(vtype=GRB.BINARY, name=f"z_{t}_{u}")
#             Delta[t] = m.addVar(vtype=GRB.BINARY, name=f"delta_{t}")
#         if allow_lost_sales:
#             for u in Periods:
#                 LS[u] = m.addVar(lb=0.0, name=f"ls_{u}")

#         # C2: setup linking
#         for t in Periods:
#             Us = Gamma.get((i, t), [])
#             if Us:
#                 m.addConstr(
#                     gp.quicksum(X[(t, u)] for u in Us) <= mu_bound[(i, t)] * Delta[t],
#                     name=f"link_{t}",
#                 )
#             else:
#                 m.addConstr(Delta[t] == 0.0, name=f"link_zero_{t}")

#         # C1b: per-item cap
#         for t in Periods:
#             if (i, t) in per_item_cap and Gamma.get((i, t)):
#                 m.addConstr(
#                     gp.quicksum(X[(t, u)] for u in Gamma[(i, t)])
#                     <= float(per_item_cap[(i, t)]),
#                     name=f"item_cap_{i}_{t}",
#                 )

#         # C4: arc activation
#         for t in Periods:
#             for u in Gamma.get((i, t), []):
#                 Ciu = float(demand[i][u])
#                 m.addConstr(X[(t, u)] <= Ciu * Z[(t, u)], name=f"arc_on_{t}_{u}")

#         # C5: LEFO no-crossing
#         prods = [t for t in Periods if Gamma.get((i, t))]
#         prods.sort(key=lambda t: Expiry[(i, t)])
#         for a in range(len(prods)):
#             t1 = prods[a]
#             v1 = Expiry[(i, t1)]
#             for b in range(a + 1, len(prods)):
#                 t2 = prods[b]
#                 v2 = Expiry[(i, t2)]
#                 if v1 >= v2:
#                     continue
#                 for up in Gamma[(i, t2)]:
#                     for u in [uu for uu in Gamma[(i, t1)] if t2 <= uu <= up - 1]:
#                         m.addConstr(
#                             Z[(t1, u)] + Z[(t2, up)] <= 1,
#                             name=f"nocross_{t1}_{t2}_{u}_{up}",
#                         )

#         # branching fixes on setups
#         for t in forbidden[i]:
#             m.addConstr(Delta[t] == 0.0, name=f"fix_forbid_{t}")
#         for t in forced[i]:
#             m.addConstr(Delta[t] == 1.0, name=f"fix_force_{t}")

#         # C3: demand satisfaction (soft if LS)
#         for u in Periods:
#             origins = [t for t in range(0, u + 1) if u in Gamma.get((i, t), [])]
#             if allow_lost_sales:
#                 m.addConstr(
#                     gp.quicksum(X[(t, u)] for t in origins) + LS[u] == demand[i][u],
#                     name=f"demand_{u}",
#                 )
#             else:
#                 m.addConstr(
#                     gp.quicksum(X[(t, u)] for t in origins) == demand[i][u],
#                     name=f"demand_{u}",
#                 )

#         # pricing objective: true cost - duals (mu on capacity; theta on carry)
#         obj = gp.LinExpr()
#         for t in Periods:
#             for u in Gamma.get((i, t), []):
#                 obj += (cvar[i][t] + _hsum(hold[i], t, u)) * X[(t, u)]
#         for t in Periods:
#             obj += setup[i][t] * Delta[t]
#         if allow_lost_sales:
#             for u in Periods:
#                 obj += loss_pen[(i, u)] * LS[u]

#         # subtract C1 dual contributions
#         for t in Periods:
#             Us = Gamma.get((i, t), [])
#             if Us:
#                 obj += -mu[t] * gp.quicksum(X[(t, u)] for u in Us)

#         # subtract C1c dual contributions (carry)
#         if theta is not None:
#             # each arc (t,w) contributes -sum_{u=t}^{w-1} theta[u]
#             for t in Periods:
#                 for w in Gamma.get((i, t), []):
#                     coef = 0.0
#                     for u in range(t, w):
#                         coef += theta[u]
#                     if coef != 0.0:
#                         obj += -coef * X[(t, w)]

#         m.setObjective(obj, GRB.MINIMIZE)
#         m.optimize()
#         if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
#             return 0.0, None

#         rc = float(m.ObjVal)  # this is reduced cost excluding convexity dual α_i
#         if rc >= -cg_tol:
#             return rc, None

#         # build pattern
#         arcs = {}
#         q = [0.0] * T
#         delta = [0] * T
#         ls = [0.0] * T if allow_lost_sales else [0.0] * T

#         for t in Periods:
#             delta[t] = int(round(float(Delta[t].X)))
#             for u in Gamma.get((i, t), []):
#                 val = float(X[(t, u)].X)
#                 if val > 1e-12:
#                     arcs[(t, u)] = val
#                     q[t] += val
#         if allow_lost_sales:
#             for u in Periods:
#                 ls[u] = float(LS[u].X)

#         # true master cost (no duals)
#         true_cost = 0.0
#         for (t, u), val in arcs.items():
#             true_cost += val * (cvar[i][t] + _hsum(hold[i], t, u))
#         for t in Periods:
#             true_cost += setup[i][t] * delta[t]
#         if allow_lost_sales:
#             for u in Periods:
#                 if ls[u] > 1e-12:
#                     true_cost += loss_pen[(i, u)] * ls[u]

#         inv_after_u = _pattern_carry_from_arcs(T, arcs)
#         return rc, Pattern(true_cost, q, delta, arcs, inv_after_u, ls)

#     # ------------- Column generation loop at a node -------------
#     def run_cg():
#         m, lam, choose, cap_rows, wh_rows, y_expr = build_rmp()
#         any_added = True
#         best_node_lb = float("inf")

#         while any_added:
#             if time_limit and (time.time() - t0) >= time_limit:
#                 break
#             m.optimize()
#             if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
#                 break
#             obj_val = float(m.ObjVal)
#             best_node_lb = min(best_node_lb, obj_val)

#             mu = [cap_rows[t].Pi for t in Periods]
#             theta = None
#             if wh_rows is not None:
#                 theta = [wh_rows[u].Pi if u in wh_rows else 0.0 for u in range(T)]

#             alpha = {i: choose[i].Pi for i in items_raw}

#             any_added = False
#             best_rc = 0.0
#             for i in items_raw:
#                 rc, pat = price_item(i, mu, theta)
#                 rc_master = rc - alpha[i]
#                 best_rc = min(best_rc, rc_master)
#                 if rc_master < -cg_tol and pat is not None:
#                     p_idx = len(patterns[i])
#                     patterns[i].append(pat)
#                     col = gp.Column()
#                     col.addTerms(1.0, choose[i])
#                     for t, q_t in enumerate(pat.q):
#                         if q_t != 0.0:
#                             col.addTerms(float(q_t), cap_rows[t])
#                     if wh_rows is not None:
#                         for u in Periods[:-1]:
#                             iv = pat.inv_after_u[u]
#                             if iv != 0.0:
#                                 col.addTerms(float(iv), wh_rows[u])
#                     v = m.addVar(
#                         lb=0.0,
#                         obj=float(pat.cost),
#                         name=f"lam_i{i}_p{p_idx}",
#                         column=col,
#                     )
#                     lam[(i, p_idx)] = v
#                     # keep y_expr updated (branching)
#                     for t in Periods:
#                         if pat.delta[t] != 0:
#                             y_expr[(i, t)].addTerms(float(pat.delta[t]), v)
#                     m.update()
#                     any_added = True

#             if best_rc >= -cg_tol:
#                 break

#         return m, lam, choose, cap_rows, wh_rows, y_expr, best_node_lb

#     # ------------- y-integrality check & branching -------------
#     def compute_y(rmp, y_expr) -> Dict[Tuple[int, int], float]:
#         y = {}
#         for i in items_raw:
#             for t in Periods:
#                 y[(i, t)] = float(y_expr[(i, t)].getValue())
#         return y

#     def pick_branch(y: Dict[Tuple[int, int], float]) -> Optional[Tuple[int, int]]:
#         best_key = None
#         best_gap = 1.0
#         for k, val in y.items():
#             if integrality_tol < val < 1.0 - integrality_tol:
#                 # most fractional (closest to 0.5)
#                 gap = abs(0.5 - val)
#                 if gap < best_gap:
#                     best_gap = gap
#                     best_key = k
#         return best_key

#     # ------------- DFS Branch-and-Price -------------
#     nodes_processed = 0
#     global_LB = float("inf")
#     incumbent_obj = float("inf")

#     def dfs():
#         nonlocal nodes_processed, global_LB, incumbent_obj
#         if nodes_processed >= max_nodes:
#             return
#         rmp, lam, choose, cap_rows, wh_rows, y_expr, node_lb = run_cg()
#         nodes_processed += 1
#         if node_lb < global_LB:
#             global_LB = node_lb
#         # bound prune
#         if node_lb >= incumbent_obj - 1e-12:
#             return

#         y = compute_y(rmp, y_expr)
#         k = pick_branch(y)
#         if k is None:
#             # integral in setups → accept incumbent
#             incumbent_obj = min(incumbent_obj, float(rmp.ObjVal))
#             return

#         i_b, t_b = k
#         # branch left: forbid δ=0
#         forbidden[i_b].add(t_b)
#         dfs()
#         forbidden[i_b].remove(t_b)
#         if time_limit and (time.time() - t0) >= time_limit:
#             return
#         # branch right: force δ=1
#         forced[i_b].add(t_b)
#         dfs()
#         forced[i_b].remove(t_b)

#     # run the tree
#     dfs()

#     # final solve at root for reporting λ (columns set already)
#     rmp_f, lam_f, choose_f, cap_rows_f, wh_rows_f, y_expr_f, _ = run_cg()

#     # aggregate flows and LS for output
#     X_flow: Dict[Tuple[int, int, int], float] = {}
#     q_it: Dict[Tuple[int, int], float] = {}
#     LS_it: Dict[Tuple[int, int], float] = {}

#     for i in items_raw:
#         for p_idx, pat in enumerate(patterns[i]):
#             v = lam_f.get((i, p_idx))
#             if v is None:
#                 continue
#             w = float(v.X)
#             if w <= 1e-9:
#                 continue
#             for (t, u), qty in pat.arcs.items():
#                 X_flow[(i, t, u)] = X_flow.get((i, t, u), 0.0) + w * qty
#             for t, qv in enumerate(pat.q):
#                 if qv != 0.0:
#                     q_it[(i, t)] = q_it.get((i, t), 0.0) + w * qv
#             if allow_lost_sales:
#                 for u, lsv in enumerate(pat.ls):
#                     if lsv != 0.0:
#                         LS_it[(i, u)] = LS_it.get((i, u), 0.0) + w * lsv

#     # write outputs
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     orders_txt: List[str] = []
#     for i in items_raw:
#         orders_txt.append(f"Item {i} — orders (t → qty)")
#         for t in Periods:
#             qty_t = q_it.get((i, t), 0.0)
#             if qty_t > 1e-6:
#                 orders_txt.append(f" {t:2d} → {qty_t:8.3f}")
#         if allow_lost_sales:
#             for u in Periods:
#                 val = LS_it.get((i, u), 0.0)
#                 if val > 1e-6:
#                     orders_txt.append(f" u={u:2d} → LOST {val:8.3f}")
#         orders_txt.append("")
#     (out_dir / "orders.txt").write_text("\n".join(orders_txt), encoding="utf-8")

#     UB = float(incumbent_obj if incumbent_obj < float("inf") else rmp_f.ObjVal)
#     LB = float(global_LB if global_LB < float("inf") else UB)
#     gap = max(0.0, (UB - LB) / (abs(UB) + 1e-12))

#     summary = {
#         "status": int(GRB.OPTIMAL if gap <= 1e-9 else GRB.SUBOPTIMAL),
#         "objective": UB,
#         "best_bound": LB,
#         "gap": gap,
#         "runtime_sec": float(time.time() - t0),
#         "solver_version": "bp_cg_setups_v1",
#         "n_items": len(items_raw),
#         "T": T,
#         "n_columns": sum(len(v) for v in patterns.values()),
#         "nodes_processed": nodes_processed,
#     }
#     (out_dir / "summary.json").write_text(
#         json.dumps(summary, indent=2), encoding="utf-8"
#     )

#     return summary, orders_txt
