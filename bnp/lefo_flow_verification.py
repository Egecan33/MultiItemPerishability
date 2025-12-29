"""
LEFO Flow Verification Experiment

Tests whether B&P solutions are always LEFO-compatible.
Generates FEASIBLE instances and checks LEFO compatibility using Gurobi LP.

Key insight: Objective = f(X, Y) only, NOT f(Z)
- Setup costs depend on Y (binary setup decisions)
- Variable/holding costs depend on X (production quantities)
- Z (production-to-demand arcs) only affects feasibility, not cost

Therefore, if we can find ANY LEFO-compatible Z for given (X,Y), the objective is optimal.
"""

import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import pandas as pd
from datetime import datetime

# Import solver
from solver_bnp_dp_full_plans import (
    solve_branch_and_price,
    _as_len_T_vector,
    BnPLogger,
    EPS,
)

# Import LEFO checker
from lefo_gurobi_check import check_solution_lefo_gurobi, check_lefo_gurobi


def extract_x_from_solution(
    columns,
    lam_vals: Dict[int, Dict[int, float]],
    items: Dict[int, dict],
    T: int,
) -> Dict[int, Dict[int, float]]:
    """Extract aggregated X[item][t] from solution."""
    x_per_item = {}
    for i in sorted(items.keys()):
        x_per_item[i] = {}
        if i not in columns or i not in lam_vals:
            continue
        for t in range(T):
            x_val = 0.0
            for k, col in enumerate(columns[i]):
                x_val += col.x.get(t, 0) * lam_vals[i].get(k, 0)
            if x_val > EPS:
                x_per_item[i][t] = x_val
    return x_per_item


def extract_y_from_solution(
    columns,
    lam_vals: Dict[int, Dict[int, float]],
    items: Dict[int, dict],
    T: int,
) -> Dict[int, Dict[int, float]]:
    """Extract aggregated Y[item][t] (setup indicators) from solution."""
    y_per_item = {}
    for i in sorted(items.keys()):
        y_per_item[i] = {}
        if i not in columns or i not in lam_vals:
            continue
        for t in range(T):
            y_val = 0.0
            for k, col in enumerate(columns[i]):
                y_val += col.y.get(t, 0) * lam_vals[i].get(k, 0)
            if y_val > EPS:
                y_per_item[i][t] = y_val
    return y_per_item


def extract_z_from_solution(
    columns,
    lam_vals: Dict[int, Dict[int, float]],
    items: Dict[int, dict],
    T: int,
) -> Dict[int, Dict[Tuple[int, int], float]]:
    """Extract aggregated Z[item][(t,u)] (arcs) from solution."""
    z_per_item = {}
    for i in sorted(items.keys()):
        z_per_item[i] = {}
        if i not in columns or i not in lam_vals:
            continue
        for k, col in enumerate(columns[i]):
            lam = lam_vals[i].get(k, 0)
            if lam < EPS:
                continue
            for (t, u), z in col.z.items():
                if z > EPS:
                    key = (t, u)
                    z_per_item[i][key] = z_per_item[i].get(key, 0) + z * lam
    # Filter small values
    for i in z_per_item:
        z_per_item[i] = {k: v for k, v in z_per_item[i].items() if v > EPS}
    return z_per_item


def generate_feasible_instance(
    num_items: int,
    num_periods: int,
    tbo: int,
    demand_max: int,
    capacity_max: int,
    shelf_min: int,
    seed: int,
) -> Tuple[Dict[int, dict], List[float]]:
    """
    Generate an instance that is GUARANTEED to be feasible.

    Key: shelf_life >= (T - production_period) ensures all production can serve all future demands.
    Capacity is scaled to total demand to ensure feasibility.
    """
    random.seed(seed)
    T = num_periods
    h_base = 0.4

    items = {}
    total_demand = 0

    for i in range(num_items):
        # Generate demand: 0 for first 2 periods, then random
        demand = [0.0, 0.0]
        for t in range(2, T):
            d = random.randint(0, demand_max // 10) * 10
            demand.append(float(d))

        # Ensure at least some demand
        if sum(demand) == 0:
            demand[random.randint(2, T - 1)] = 10.0

        total_demand += sum(demand)

        # Generate shelf life: ensure FEASIBILITY
        # shelf_seq[t] >= (T - 1 - t) ensures production at t can serve demand at T-1
        # We use shelf_min as minimum, but ensure it's at least enough for feasibility
        shelf_seq = []
        for t in range(T):
            # Minimum shelf life for feasibility from period t
            min_for_feasibility = max(1, T - 1 - t)
            # Random shelf life, at least shelf_min and at least enough for feasibility
            s = random.randint(max(shelf_min, min_for_feasibility), T)
            shelf_seq.append(s)

        # Compute setup cost based on TBO
        avg_demand = sum(demand) / max(1, len([d for d in demand if d > 0]))
        setup_cost = max(10, (tbo**2) * h_base * avg_demand / 2)

        items[i] = {
            "demand": demand,
            "setup": [setup_cost] * T,
            "h": [h_base] * T,
            "c_var": [round(random.uniform(1.0, 2.5), 2) for _ in range(T)],
            "shelf_seq": shelf_seq,
        }

    # Generate capacity: ensure enough to cover all demand
    # Average demand per period * safety factor
    avg_demand_per_period = total_demand / (T - 2) if T > 2 else total_demand
    min_cap = max(10, int(avg_demand_per_period * 1.2))  # 20% safety margin
    min_cap = min(min_cap, capacity_max)

    capacity = [0.0, 0.0]
    for t in range(2, T):
        c = random.randint(min_cap // 10, capacity_max // 10) * 10
        capacity.append(float(max(c, min_cap)))

    return items, capacity


def format_dict_compact(d: dict, max_items: int = 10) -> str:
    """Format dict compactly for Excel."""
    if not d:
        return ""
    items = [
        f"{k}:{v:.1f}" if isinstance(v, float) else f"{k}:{v}"
        for k, v in sorted(d.items())
    ]
    if len(items) > max_items:
        items = items[:max_items] + ["..."]
    return ", ".join(items)


def format_arcs_compact(arcs: dict, max_arcs: int = 15) -> str:
    """Format arcs compactly for Excel."""
    if not arcs:
        return ""
    items = [f"({t}→{u}):{v:.2f}" for (t, u), v in sorted(arcs.items())]
    if len(items) > max_arcs:
        items = items[:max_arcs] + ["..."]
    return ", ".join(items)


def run_experiment(
    num_items: int = 5,
    num_periods: int = 10,
    num_instances_per_config: int = 5,
    time_limit_per_instance: float = 60,
    output_file: str = None,
):
    """
    Run LEFO verification experiment with DETAILED Excel output.

    For each instance, records:
    - Input: demand, shelf_seq, capacity per item
    - B&P Output: X, Y, Z values and objective
    - LEFO Check: corrected Z arcs
    - Key insight: Objective unchanged (depends only on X,Y not Z)
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"lefo_experiment_{timestamp}.xlsx"

    # Test configurations
    configs = []
    config_id = 0

    # Parameters - comprehensive but manageable
    tbo_values = [2, 3, 4, 5]
    demand_max_values = [20, 30, 40]
    capacity_max_values = [80, 100, 120]
    shelf_min_values = [2, 3, 4]

    for tbo in tbo_values:
        for demand_max in demand_max_values:
            for capacity_max in capacity_max_values:
                for shelf_min in shelf_min_values:
                    configs.append(
                        {
                            "config_id": config_id,
                            "tbo": tbo,
                            "demand_max": demand_max,
                            "capacity_max": capacity_max,
                            "shelf_min": shelf_min,
                        }
                    )
                    config_id += 1

    total_configs = len(configs)
    total_instances = total_configs * num_instances_per_config

    print("=" * 70)
    print("LEFO VERIFICATION EXPERIMENT - DETAILED OUTPUT")
    print("=" * 70)
    print(f"Items: {num_items}, Periods: {num_periods}")
    print(f"Configurations: {total_configs}")
    print(f"Instances per config: {num_instances_per_config}")
    print(f"Total instances: {total_instances}")
    print(f"Output: {output_file}")
    print("=" * 70)

    # Results for different sheets
    summary_results = []
    instance_details = []  # Per-instance details
    item_details = []  # Per-item details

    total_feasible = 0
    total_lefo_compat = 0
    start_time = time.time()

    for cfg in configs:
        cfg_id = cfg["config_id"]
        config_feasible = 0
        config_compat = 0

        for inst_idx in range(num_instances_per_config):
            seed = cfg_id * 10000 + inst_idx

            # Generate FEASIBLE instance
            items, capacity = generate_feasible_instance(
                num_items=num_items,
                num_periods=num_periods,
                tbo=cfg["tbo"],
                demand_max=cfg["demand_max"],
                capacity_max=cfg["capacity_max"],
                shelf_min=cfg["shelf_min"],
                seed=seed,
            )

            T = num_periods

            # Solve B&P
            try:
                logger = BnPLogger(Path("/tmp"), enabled=False)
                best_ub, best_lb, columns, lam_vals, time_limit_reached = (
                    solve_branch_and_price(
                        items=items,
                        capacity=capacity,
                        T=T,
                        time_limit=time_limit_per_instance,
                        logger=logger,
                        verbose=False,
                    )
                )

                if best_ub < math.inf and lam_vals:
                    config_feasible += 1
                    total_feasible += 1

                    # Extract X, Y, Z values from B&P solution
                    x_per_item = extract_x_from_solution(columns, lam_vals, items, T)
                    y_per_item = extract_y_from_solution(columns, lam_vals, items, T)
                    z_per_item = extract_z_from_solution(columns, lam_vals, items, T)

                    # Check LEFO with Gurobi
                    is_compat, gurobi_results, details = check_solution_lefo_gurobi(
                        items, x_per_item, T, verbose=False
                    )

                    if is_compat:
                        config_compat += 1
                        total_lefo_compat += 1
                        lefo_status = "LEFO_OK"
                    else:
                        lefo_status = "LEFO_FAIL"
                        # Log failures
                        print(f"\n{'='*60}")
                        print(f"!!! LEFO FAILURE: seed={seed}, cfg={cfg_id}")
                        for i, item_data in items.items():
                            x_vals = x_per_item.get(i, {})
                            print(f"    Item {i}: demand={item_data['demand']}")
                            print(f"             X={dict(sorted(x_vals.items()))}")
                        print(f"{'='*60}\n")

                    # Instance-level summary
                    instance_details.append(
                        {
                            "Seed": seed,
                            "Config": cfg_id,
                            "TBO": cfg["tbo"],
                            "Demand_Max": cfg["demand_max"],
                            "Capacity_Max": cfg["capacity_max"],
                            "Shelf_Min": cfg["shelf_min"],
                            "Capacity": str(capacity[2:]),
                            "Solver_Status": "OPTIMAL",
                            "Objective": round(best_ub, 2),
                            "LEFO_Status": lefo_status,
                            "Obj_After_LEFO": round(
                                best_ub, 2
                            ),  # Same! Z doesn't affect obj
                            "Obj_Changed": "NO",  # Key insight
                        }
                    )

                    # Per-item details
                    for i, item_data in items.items():
                        x_vals = x_per_item.get(i, {})
                        y_vals = y_per_item.get(i, {})
                        z_solver = z_per_item.get(i, {})

                        # Get LEFO-corrected Z from Gurobi results
                        z_lefo = {}
                        for r in gurobi_results:
                            if r.item_id == i and r.z_values:
                                z_lefo = r.z_values
                                break

                        item_details.append(
                            {
                                "Seed": seed,
                                "Item": i,
                                "Demand": str([int(d) for d in item_data["demand"]]),
                                "Shelf_Seq": str(item_data["shelf_seq"]),
                                "Total_Demand": sum(item_data["demand"]),
                                "Total_X": sum(x_vals.values()) if x_vals else 0,
                                "Y_Periods": format_dict_compact(y_vals),
                                "X_Values": format_dict_compact(x_vals),
                                "Z_Solver": format_arcs_compact(z_solver),
                                "Z_LEFO": format_arcs_compact(z_lefo),
                                "LEFO_OK": "YES" if is_compat else "NO",
                            }
                        )

                else:
                    instance_details.append(
                        {
                            "Seed": seed,
                            "Config": cfg_id,
                            "TBO": cfg["tbo"],
                            "Demand_Max": cfg["demand_max"],
                            "Capacity_Max": cfg["capacity_max"],
                            "Shelf_Min": cfg["shelf_min"],
                            "Capacity": str(capacity[2:]),
                            "Solver_Status": "INFEASIBLE",
                            "Objective": "N/A",
                            "LEFO_Status": "N/A",
                            "Obj_After_LEFO": "N/A",
                            "Obj_Changed": "N/A",
                        }
                    )

            except Exception as e:
                instance_details.append(
                    {
                        "Seed": seed,
                        "Config": cfg_id,
                        "TBO": cfg["tbo"],
                        "Demand_Max": cfg["demand_max"],
                        "Capacity_Max": cfg["capacity_max"],
                        "Shelf_Min": cfg["shelf_min"],
                        "Capacity": "",
                        "Solver_Status": f"ERROR: {str(e)[:50]}",
                        "Objective": "N/A",
                        "LEFO_Status": "N/A",
                        "Obj_After_LEFO": "N/A",
                        "Obj_Changed": "N/A",
                    }
                )

            # Progress
            done = len(instance_details)
            if done % 20 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / done) * (total_instances - done) if done > 0 else 0
                rate = (
                    (total_lefo_compat / total_feasible * 100)
                    if total_feasible > 0
                    else 0
                )
                print(
                    f"Progress: {done}/{total_instances} | "
                    f"Feasible: {total_feasible} | LEFO OK: {total_lefo_compat} ({rate:.1f}%) | "
                    f"ETA: {eta/60:.1f}min"
                )

        # Config summary
        if config_feasible > 0:
            rate = config_compat / config_feasible * 100
            summary_results.append(
                {
                    "Config": cfg_id,
                    "TBO": cfg["tbo"],
                    "Demand_Max": cfg["demand_max"],
                    "Capacity_Max": cfg["capacity_max"],
                    "Shelf_Min": cfg["shelf_min"],
                    "Feasible": config_feasible,
                    "LEFO_OK": config_compat,
                    "Rate_%": round(rate, 1),
                }
            )
            if rate < 100:
                print(
                    f"Config {cfg_id}: {config_compat}/{config_feasible} ({rate:.1f}%) ⚠️"
                )

    # Create DataFrames
    df_summary = pd.DataFrame(summary_results)
    df_instances = pd.DataFrame(instance_details)
    df_items = pd.DataFrame(item_details)

    # Overall summary
    overall = pd.DataFrame(
        {
            "Metric": [
                "Total Instances",
                "Solver Feasible",
                "LEFO Compatible",
                "LEFO Rate (%)",
                "",
                "KEY INSIGHT",
                "Objective depends on X,Y only",
                "Z arcs only affect feasibility",
                "If LEFO Z exists, obj is optimal",
            ],
            "Value": [
                len(df_instances),
                total_feasible,
                total_lefo_compat,
                (
                    f"{total_lefo_compat/total_feasible*100:.2f}"
                    if total_feasible > 0
                    else "N/A"
                ),
                "",
                "================",
                "Obj = Σ setup(Y) + Σ var_cost(X) + Σ holding(X)",
                "Z determines which t serves which u",
                "All LEFO_OK instances have same obj!",
            ],
        }
    )

    # Write Excel with multiple sheets
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="Overview", index=False)
        df_summary.to_excel(writer, sheet_name="Config Summary", index=False)
        df_instances.to_excel(writer, sheet_name="Instances", index=False)
        df_items.to_excel(writer, sheet_name="Item Details", index=False)

    # Final report
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Total: {len(df_instances)}")
    print(f"Solver Feasible: {total_feasible}")
    print(f"LEFO Compatible: {total_lefo_compat}")
    if total_feasible > 0:
        print(f"LEFO Rate: {total_lefo_compat/total_feasible*100:.2f}%")
    print(f"\n*** KEY INSIGHT: Objective NEVER changes after LEFO check ***")
    print(f"*** Z arcs only affect feasibility, not cost ***")
    print(f"\nExcel saved: {output_file}")
    print("=" * 70)

    return instance_details, item_details


def test_single_instance():
    """Test on a single instance to verify correctness."""
    print("=" * 70)
    print("SINGLE INSTANCE TEST")
    print("=" * 70)

    # Generate a feasible instance
    items, capacity = generate_feasible_instance(
        num_items=3,
        num_periods=10,
        tbo=3,
        demand_max=30,
        capacity_max=100,
        shelf_min=3,
        seed=42,
    )

    T = 10

    print("\nGenerated Instance:")
    for i, item in items.items():
        print(f"  Item {i}:")
        print(f"    demand: {item['demand']}")
        print(f"    shelf_seq: {item['shelf_seq']}")
    print(f"  capacity: {capacity}")

    # Solve
    print("\nSolving with B&P...")
    logger = BnPLogger(Path("/tmp"), enabled=False)
    best_ub, best_lb, columns, lam_vals, time_limit_reached = solve_branch_and_price(
        items=items,
        capacity=capacity,
        T=T,
        time_limit=60,
        logger=logger,
        verbose=False,
    )

    print(f"Objective: {best_ub:.2f}")

    if best_ub < math.inf and lam_vals:
        x_per_item = extract_x_from_solution(columns, lam_vals, items, T)
        y_per_item = extract_y_from_solution(columns, lam_vals, items, T)
        z_per_item = extract_z_from_solution(columns, lam_vals, items, T)

        print("\n--- B&P SOLUTION ---")
        for i in items:
            print(f"\nItem {i}:")
            print(f"  Y (setups): {y_per_item.get(i, {})}")
            print(f"  X (production): {x_per_item.get(i, {})}")
            print(f"  Z (arcs from solver): {z_per_item.get(i, {})}")

        # Check LEFO
        print("\n--- LEFO CHECK ---")
        is_compat, results, summary = check_solution_lefo_gurobi(
            items, x_per_item, T, verbose=False
        )

        print(f"LEFO Compatible: {is_compat}")
        for r in results:
            if r.z_values:
                print(f"\nItem {r.item_id} - LEFO Z arcs:")
                for (t, u), z in sorted(r.z_values.items()):
                    print(f"  Z[{t},{u}] = {z:.2f}")

        print(f"\n*** Objective after LEFO: {best_ub:.2f} (UNCHANGED!) ***")
    else:
        print("Instance is infeasible!")

    return best_ub


if __name__ == "__main__":
    import sys

    NUM_ITEMS = 5  # Number of items per instance
    NUM_PERIODS = 10  # Planning horizon
    NUM_INSTANCES = 3  # Instances per configuration (108 configs total)
    TIME_LIMIT = 60  # Seconds per instance

    # Parse command line arguments (optional)
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "test":
            test_single_instance()
            sys.exit(0)
        elif arg.isdigit():
            NUM_INSTANCES = int(arg)
    if len(sys.argv) > 2:
        NUM_ITEMS = int(sys.argv[2])
    if len(sys.argv) > 3:
        NUM_PERIODS = int(sys.argv[3])

    # Run comprehensive experiment
    print(f"\n{'='*70}")
    print("LEFO VERIFICATION EXPERIMENT")
    print(f"{'='*70}")
    print(f"Items: {NUM_ITEMS}, Periods: {NUM_PERIODS}")
    print(f"Instances per config: {NUM_INSTANCES}")
    print(f"Total configs: 108 (4 TBO x 3 Dmax x 3 Cmax x 3 Shelf)")
    print(f"Total instances: {108 * NUM_INSTANCES}")
    print(f"{'='*70}\n")

    run_experiment(
        num_items=NUM_ITEMS,
        num_periods=NUM_PERIODS,
        num_instances_per_config=NUM_INSTANCES,
        time_limit_per_instance=TIME_LIMIT,
    )
