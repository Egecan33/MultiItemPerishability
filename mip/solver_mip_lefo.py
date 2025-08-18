# mip/solver_mip_lefo.py
"""
Freshest-First Compact MIP matching your LaTeX:

Min   sum_{i,t} sum_{u in Γ(i,t)} (c_{it} + h_{it}(u-t)) X_{itu} + sum_{i,t} s_{it} Y_{it}
s.t.  Global capacity per t
      Setup linking with tight μ_{it}
      Warehouse capacity (optional)
      Demand balance (no backlog)
      LEFO freshest-first constraints with S_{iuτ} ≥ 0

X continuous ≥ 0, Y binary, S ≥ 0.

Supports scalar or time-varying setup s_{it}; scalar c_i, h_i (extendable to sequences).
"""

from __future__ import annotations
import argparse, json, csv, shutil
from pathlib import Path
from datetime import datetime
import gurobipy as gp
from gurobipy import GRB


def load_instance(path: str):
    data = json.loads(Path(path).read_text())
    T = int(data["period"])
    Periods = list(range(T))

    # capacities
    cap_global = data.get("manual_capacity")
    items_raw = data["items"]

    # if no manual capacity: derive from demand + buffer (20% of max per-period demand)
    if not cap_global:
        cap_raw = [0] * T
        for it in items_raw.values():
            dem = it["demand"]
            for t in Periods:
                cap_raw[t] += int(dem[t])
        max_cap = max(cap_raw) if cap_raw else 0
        buffer = max(5, int(0.2 * max_cap))
        cap_global = [c + buffer for c in cap_raw]

    warehouse_capacity = data.get("warehouse_capacity", None)

    # massage items
    items = {}
    for k, it in items_raw.items():
        i = int(k)
        demand = list(map(int, it["demand"]))
        m_seq = list(map(int, it["shelf_seq"]))  # m_{i,t}

        # setups can be scalar or list
        setup_raw = it["setup"]
        if isinstance(setup_raw, list):
            setup_seq = [float(x) for x in setup_raw]
            setup_scalar = None
        else:
            setup_scalar = float(setup_raw)
            setup_seq = None

        cap_seq = it.get("cap_seq")
        if cap_seq is not None:
            cap_seq = [float(v) for v in cap_seq]

        items[i] = dict(
            demand=demand,
            m_seq=m_seq,
            setup_scalar=setup_scalar,
            setup_seq=setup_seq,
            cap_seq=cap_seq,
            c_var=float(it["c_var"]),  # c_i
            h=float(it["h"]),  # h_i
        )

    return T, Periods, cap_global, warehouse_capacity, items


def setup_at(items, i: int, t: int) -> float:
    seq = items[i]["setup_seq"]
    if seq is not None:
        return float(seq[t])
    return (
        float(items[i]["setup_scalar"]) if items[i]["setup_scalar"] is not None else 0.0
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=str, default="last_instance.json")
    ap.add_argument("--time-limit", type=float, default=3600)
    ap.add_argument("--mipgap", type=float, default=0.01)
    ap.add_argument("--outdir", type=str, default="mip_results")
    ap.add_argument(
        "--x-as-integer",
        action="store_true",
        help="If set, make X integer instead of continuous (default is continuous).",
    )
    args = ap.parse_args()

    T, Periods, cap_global, W, items = load_instance(args.instance)

    m = gp.Model("perishable_FF_compact_LEFO")
    m.Params.OutputFlag = 1
    if args.time_limit:
        m.Params.TimeLimit = args.time_limit
    if args.mipgap is not None:
        m.Params.MIPGap = args.mipgap

    # feasible (i,t,u) with per-period perishability m_{i,t}
    Triples = []
    Gamma = {}  # Gamma[(i,t)] = feasible u for that (i,t)
    Origins = {}  # Origins[(i,u)] = feasible t that can serve u
    for i, it in items.items():
        mseq = it["m_seq"]
        for t in Periods:
            u_max = min(T - 1, t + int(mseq[t]) - 1)
            us = [u for u in range(t, u_max + 1)]
            Gamma[(i, t)] = us
        for u in Periods:
            origins = []
            for t in range(0, u + 1):
                if u in Gamma.get((i, t), []):
                    origins.append(t)
            Origins[(i, u)] = origins

    # μ_{it} = sum_{u in Γ(i,t)} d_{iu} (tight big-M)
    MU = {}
    for i, it in items.items():
        d = it["demand"]
        for t in Periods:
            MU[(i, t)] = sum(d[u] for u in Gamma.get((i, t), []))

    # Variables
    vtype_x = GRB.INTEGER if args.x_as_integer else GRB.CONTINUOUS
    X = m.addVars(Triples, vtype=vtype_x, lb=0.0, name="X")  # production→consumption
    Y = m.addVars([(i, t) for i in items for t in Periods], vtype=GRB.BINARY, name="Y")
    # LEFO slack S_{i,u,tau} for tau=0..u-1
    S = m.addVars(
        [(i, u, tau) for i in items for u in Periods for tau in range(0, u)],
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="S",
    )

    # Objective: sum g_{itu} X + sum s_{it} Y
    obj_terms = []
    for i, t, u in Triples:
        c_it = items[i]["c_var"]  # scalar c_i (extend to per-t if needed)
        h_it = items[i]["h"]  # scalar h_i
        g_itu = c_it + h_it * (u - t)
        obj_terms.append(g_itu * X[i, t, u])
    obj_terms += [setup_at(items, i, t) * Y[i, t] for (i, t) in Y.keys()]
    m.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)

    # (1) Global capacity per production period t
    for t in Periods:
        m.addConstr(
            gp.quicksum(X[i, t, u] for (i, tt, u) in Triples if tt == t)
            <= cap_global[t],
            name=f"cap_global_{t}",
        )

    # (2) Optional per-item capacity cap_{i,t} if present
    for i, it in items.items():
        cap_seq = it.get("cap_seq")
        if cap_seq is None:
            continue
        for t in Periods:
            m.addConstr(
                gp.quicksum(X[i, t, u] for u in Gamma[(i, t)]) <= float(cap_seq[t]),
                name=f"cap_item_{i}_{t}",
            )

    # (3) Setup linking with tight μ_{it}
    for i, t in Y.keys():
        m.addConstr(
            gp.quicksum(X[i, t, u] for u in Gamma.get((i, t), []))
            <= MU[(i, t)] * Y[i, t],
            name=f"setup_link_{i}_{t}",
        )

    # (4) Warehouse end-of-period capacity (if W provided)
    if W is not None:
        W = float(W)
        for u in Periods[:-1]:  # T-1 has no future consumption
            inv_u = gp.quicksum(
                X[i, t, w]
                for i in items
                for t in range(0, u + 1)
                for w in Gamma[(i, t)]
                if w > u
            )
            m.addConstr(inv_u <= W, name=f"whcap_end_{u}")

    # (5) Demand balance (no backorders)
    for i, it in items.items():
        d = it["demand"]
        for u in Periods:
            m.addConstr(
                gp.quicksum(X[i, t, u] for t in Origins[(i, u)]) == d[u],
                name=f"demand_{i}_{u}",
            )

    # (6) Freshest-first (LEFO) constraints
    # For each (i,u), for each threshold tau in [0..u-1]:
    #   S_{iuτ} >= d_{iu} - sum_{t=tau+1..u} X_{itu}
    #   sum_{t=0..tau} X_{itu} <= S_{iuτ}
    for i, it in items.items():
        d = it["demand"]
        for u in range(T):
            if u == 0:
                continue  # no tau in [0..-1]
            for tau in range(0, u):
                newer = [t for t in range(tau + 1, u + 1) if u in Gamma.get((i, t), [])]
                older = [t for t in range(0, tau + 1) if u in Gamma.get((i, t), [])]
                m.addConstr(
                    S[i, u, tau] >= d[u] - gp.quicksum(X[i, t, u] for t in newer),
                    name=f"lefo_slack_{i}_{u}_{tau}",
                )
                m.addConstr(
                    gp.quicksum(X[i, t, u] for t in older) <= S[i, u, tau],
                    name=f"lefo_older_{i}_{u}_{tau}",
                )

    # Solve
    m.optimize()

    # Reporting & artifacts
    out_root = (
        Path(args.outdir)
        / f"{Path(args.instance).stem}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    def _status_name(st: int) -> str:
        return {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
        }.get(st, str(st))

    # summary.json
    summary = {
        "instance": Path(args.instance).name,
        "run_id": out_root.name,
        "time_limit_sec": args.time_limit,
        "mip_gap_target": args.mipgap,
        "status_code": int(m.Status),
        "status_name": _status_name(m.Status),
        "objective": getattr(m, "ObjVal", None),
        "best_bound": getattr(m, "ObjBound", None),
        "mip_gap": getattr(m, "MIPGap", None),
        "runtime_sec": getattr(m, "Runtime", None),
        "n_periods": T,
        "n_arcs": len(Triples),
        "x_as_integer": args.x_as_integer,
        "warehouse_capacity": W,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # capacity.csv
    with open(out_root / "capacity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "cap_global_t"])
        for t, c in enumerate(cap_global):
            w.writerow([t, c])

    # model.lp
    try:
        m.write(str(out_root / "model.lp"))
    except Exception:
        pass

    # metrics.txt
    obj_txt = (
        f"{getattr(m, 'ObjVal', float('nan')):,.6f}"
        if getattr(m, "SolCount", 0)
        else "N/A"
    )
    bnd_txt = (
        f"{getattr(m, 'ObjBound', float('nan')):,.6f}"
        if getattr(m, "ObjBound", None) is not None
        else "N/A"
    )
    gap_attr = getattr(m, "MIPGap", None)
    gap_txt = f"{gap_attr*100:.4f} %" if gap_attr is not None else "N/A"
    rt_txt = f"{getattr(m, 'Runtime', float('nan')):,.3f} s"
    (out_root / "metrics.txt").write_text(
        "\n".join(
            [
                f"Status     : {_status_name(m.Status)}",
                f"Objective  : {obj_txt}",
                f"Best bound : {bnd_txt}",
                f"MIP gap    : {gap_txt}",
                f"Runtime    : {rt_txt}",
            ]
        ),
        encoding="utf-8",
    )

    # write X>0
    if getattr(m, "SolCount", 0):
        with open(out_root / "x_nonzero.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["i", "t", "u", "X"])
            for i, t, u in Triples:
                val = X[i, t, u].X if (i, t, u) in X else 0.0
                if val > 1e-9:
                    w.writerow([i, t, u, f"{val:.6f}"])
        with open(out_root / "y_setups.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["i", "t", "Y"])
            for i, t in Y.keys():
                w.writerow([i, t, int(round(Y[i, t].X))])

        # human-friendly orders by period t → qty
        lines = []
        for i in items:
            lines.append(f"Item {i} — orders (period → qty)")
            for t in Periods:
                qty = sum(X[i, t, u].X for u in Gamma.get((i, t), []) if (i, t, u) in X)
                if qty > 1e-9:
                    lines.append(f"  {t:2d} → {qty:4.0f}")
            lines.append("")
        (out_root / "orders.txt").write_text("\n".join(lines), encoding="utf-8")

    # copy instance
    shutil.copy(args.instance, out_root / Path(args.instance).name)
    print(f"\nArtifacts saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
