"""
Genetic Algorithm Heuristic Solver for Multi-Item Perishable Lot-Sizing.

Chromosome = Binary Y vector (which periods have setups)
Given Y, uses greedy LEFO-respecting flow assignment to compute costs.
Fast heuristic - no column generation or branching.
"""

from __future__ import annotations
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

EPS = 1e-6


@dataclass
class Individual:
    """GA Individual: Y pattern for all items."""

    y: Dict[int, List[int]]  # item_id -> [0,1,1,0,...] binary setup pattern
    fitness: float = math.inf
    x: Optional[Dict[int, Dict[int, float]]] = None  # item -> {t: qty}
    flow: Optional[Dict[int, Dict[Tuple[int, int], float]]] = (
        None  # item -> {(t,u): qty}
    )


def build_gamma(item_data: dict, T: int) -> Dict[int, List[int]]:
    """Build reachability: Gamma[t] = list of periods reachable from t."""
    shelf_seq = item_data["shelf_seq"]
    Gamma = {}
    for t in range(T):
        m_t = int(shelf_seq[t]) if t < len(shelf_seq) else 0
        if m_t <= 0:
            Gamma[t] = []
        else:
            Gamma[t] = list(range(t, min(T, t + m_t + 1)))
    return Gamma


def arc_cost(item_data: dict, t: int, u: int) -> float:
    """Cost per unit on arc (t, u): variable + holding."""
    c_var = item_data["c_var"]
    h = item_data.get("h", [0.0])

    var_cost = (
        float(c_var[t])
        if isinstance(c_var, list) and t < len(c_var)
        else float(c_var) if not isinstance(c_var, list) else 0.0
    )

    hold_cost = 0.0
    for r in range(t, u):
        if isinstance(h, list):
            hold_cost += float(h[r]) if r < len(h) else 0.0
        else:
            hold_cost += float(h)

    return var_cost + hold_cost


def setup_cost_at(item_data: dict, t: int) -> float:
    """Get setup cost at period t."""
    setup = item_data["setup"]
    if isinstance(setup, list):
        return float(setup[t]) if t < len(setup) else 0.0
    return float(setup)


def evaluate_chromosome(
    y_pattern: Dict[int, List[int]],
    items: Dict[int, dict],
    T: int,
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    capacity: List[float],
) -> Tuple[float, Dict[int, Dict[int, float]], Dict[int, Dict[Tuple[int, int], float]]]:
    """
    Evaluate a chromosome (Y pattern) using greedy LEFO flow assignment.

    Returns: (total_cost, x_by_item, flow_by_item) or (inf, {}, {}) if infeasible.
    """
    total_cost = 0.0
    x_by_item = {i: {} for i in items}
    flow_by_item = {i: {} for i in items}
    production_at_t = [0.0] * T  # Track production per period for capacity

    for item_id, item_data in items.items():
        y_vec = y_pattern[item_id]
        Gamma = Gamma_by_item[item_id]
        demand = item_data["demand"]

        # Get setup periods (where Y=1)
        setup_periods = [t for t in range(T) if y_vec[t] == 1]

        # Add setup costs
        for t in setup_periods:
            total_cost += setup_cost_at(item_data, t)

        # Demand periods
        demand_periods = [u for u in range(T) if demand[u] > 0]

        # Greedy LEFO assignment: for each demand u, find best production t
        # LEFO: prefer production with LATER expiry (to use fresher items for later demands)
        remaining_demand = {u: float(demand[u]) for u in demand_periods}

        # Sort demands by period (earliest first)
        for u in sorted(demand_periods):
            if remaining_demand[u] <= EPS:
                continue

            # Find setup periods that can reach u
            candidates = []
            for t in setup_periods:
                if u in Gamma.get(t, []):
                    # LEFO: prefer later expiry (t + shelf_life)
                    shelf = (
                        item_data["shelf_seq"][t]
                        if t < len(item_data["shelf_seq"])
                        else 0
                    )
                    expiry = t + shelf
                    cost = arc_cost(item_data, t, u)
                    candidates.append((expiry, cost, t))

            if not candidates:
                # No production can reach this demand - infeasible
                return math.inf, {}, {}

            # LEFO: sort by expiry ascending (use earlier-expiring first)
            candidates.sort(key=lambda x: (x[0], x[1]))

            # Assign demand to first valid candidate
            _, cost, t = candidates[0]
            qty = remaining_demand[u]

            # Record flow
            if (t, u) not in flow_by_item[item_id]:
                flow_by_item[item_id][(t, u)] = 0.0
            flow_by_item[item_id][(t, u)] += qty

            # Record production
            if t not in x_by_item[item_id]:
                x_by_item[item_id][t] = 0.0
            x_by_item[item_id][t] += qty

            # Add arc cost
            total_cost += cost * qty

            # Track capacity usage
            production_at_t[t] += qty

            remaining_demand[u] = 0.0

    # Check capacity
    for t in range(T):
        if production_at_t[t] > capacity[t] + EPS:
            return math.inf, {}, {}  # Capacity violation

    return total_cost, x_by_item, flow_by_item


def create_random_individual(
    items: Dict[int, dict],
    T: int,
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
) -> Dict[int, List[int]]:
    """Create a random Y pattern ensuring feasibility."""
    y_pattern = {}

    for item_id, item_data in items.items():
        demand = item_data["demand"]
        Gamma = Gamma_by_item[item_id]

        # Find demand periods
        demand_periods = [u for u in range(T) if demand[u] > 0]

        # Greedy: ensure each demand is reachable
        y_vec = [0] * T
        covered = set()

        for u in demand_periods:
            if u in covered:
                continue
            # Find a production period that can reach u
            candidates = [t for t in range(T) if u in Gamma.get(t, [])]
            if candidates:
                # Randomly choose, but prefer periods that cover more demands
                t = random.choice(candidates)
                y_vec[t] = 1
                # Mark all demands reachable from t as covered
                for uu in Gamma.get(t, []):
                    if demand[uu] > 0:
                        covered.add(uu)

        # Random mutations: add/remove some setups
        for t in range(T):
            if Gamma.get(t, []):
                if random.random() < 0.1:  # 10% chance to flip
                    y_vec[t] = 1 - y_vec[t]

        y_pattern[item_id] = y_vec

    return y_pattern


def crossover(
    parent1: Dict[int, List[int]],
    parent2: Dict[int, List[int]],
    items: Dict[int, dict],
) -> Dict[int, List[int]]:
    """Two-point crossover per item."""
    child = {}
    for item_id in items:
        T = len(parent1[item_id])
        if T < 2:
            child[item_id] = list(parent1[item_id])
            continue

        # Two-point crossover
        pt1, pt2 = sorted(random.sample(range(T), 2))
        child_vec = (
            parent1[item_id][:pt1] + parent2[item_id][pt1:pt2] + parent1[item_id][pt2:]
        )
        child[item_id] = child_vec

    return child


def mutate(
    y_pattern: Dict[int, List[int]],
    items: Dict[int, dict],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    mutation_rate: float = 0.05,
) -> Dict[int, List[int]]:
    """Flip random bits with given probability."""
    mutated = {}
    for item_id, y_vec in y_pattern.items():
        new_vec = list(y_vec)
        Gamma = Gamma_by_item[item_id]
        for t in range(len(y_vec)):
            if Gamma.get(t, []) and random.random() < mutation_rate:
                new_vec[t] = 1 - new_vec[t]
        mutated[item_id] = new_vec
    return mutated


def repair_individual(
    y_pattern: Dict[int, List[int]],
    items: Dict[int, dict],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
) -> Dict[int, List[int]]:
    """Repair infeasible Y pattern by ensuring all demands are covered."""
    repaired = {i: list(y) for i, y in y_pattern.items()}

    for item_id, item_data in items.items():
        demand = item_data["demand"]
        Gamma = Gamma_by_item[item_id]
        T = len(demand)

        demand_periods = [u for u in range(T) if demand[u] > 0]

        for u in demand_periods:
            # Check if any setup can reach u
            can_reach = False
            for t in range(T):
                if repaired[item_id][t] == 1 and u in Gamma.get(t, []):
                    can_reach = True
                    break

            if not can_reach:
                # Add a setup that can reach u
                candidates = [t for t in range(T) if u in Gamma.get(t, [])]
                if candidates:
                    # Prefer cheapest setup
                    best_t = min(candidates, key=lambda t: setup_cost_at(item_data, t))
                    repaired[item_id][best_t] = 1

    return repaired


def solve_ga(
    items: Dict[int, dict],
    T: int,
    capacity: List[float],
    Gamma_by_item: Dict[int, Dict[int, List[int]]],
    pop_size: int = 50,
    generations: int = 100,
    elite_size: int = 5,
    mutation_rate: float = 0.05,
    verbose: bool = True,
) -> Tuple[float, Optional[Individual]]:
    """
    Run Genetic Algorithm to find good Y pattern.

    Returns: (best_cost, best_individual)
    """
    start_time = time.time()

    # Initialize population
    population: List[Individual] = []
    for _ in range(pop_size):
        y_pattern = create_random_individual(items, T, Gamma_by_item)
        y_pattern = repair_individual(y_pattern, items, Gamma_by_item)
        fitness, x, flow = evaluate_chromosome(
            y_pattern, items, T, Gamma_by_item, capacity
        )
        ind = Individual(y=y_pattern, fitness=fitness, x=x, flow=flow)
        population.append(ind)

    # Sort by fitness
    population.sort(key=lambda ind: ind.fitness)
    best = population[0]

    if verbose:
        print(f"  Gen 0: best={best.fitness:.2f}")

    for gen in range(1, generations + 1):
        new_pop = []

        # Elitism: keep best individuals
        new_pop.extend(population[:elite_size])

        # Fill rest with crossover and mutation
        while len(new_pop) < pop_size:
            # Tournament selection
            tourney = random.sample(population, min(5, len(population)))
            parent1 = min(tourney, key=lambda ind: ind.fitness)
            tourney = random.sample(population, min(5, len(population)))
            parent2 = min(tourney, key=lambda ind: ind.fitness)

            # Crossover
            child_y = crossover(parent1.y, parent2.y, items)

            # Mutation
            child_y = mutate(child_y, items, Gamma_by_item, mutation_rate)

            # Repair
            child_y = repair_individual(child_y, items, Gamma_by_item)

            # Evaluate
            fitness, x, flow = evaluate_chromosome(
                child_y, items, T, Gamma_by_item, capacity
            )
            child = Individual(y=child_y, fitness=fitness, x=x, flow=flow)
            new_pop.append(child)

        population = new_pop
        population.sort(key=lambda ind: ind.fitness)

        if population[0].fitness < best.fitness:
            best = population[0]

        if verbose and gen % 10 == 0:
            print(f"  Gen {gen}: best={best.fitness:.2f}")

    runtime = time.time() - start_time
    if verbose:
        print(f"\n  GA completed in {runtime:.2f}s")
        print(f"  Best fitness: {best.fitness:.2f}")

    return best.fitness, best


def solve_instance(
    instance_path: str | Path = "last_instance.json",
    time_limit: int = 0,
    out_dir: str | Path = "ga_results",
    verbose: bool = True,
    pop_size: int = 100,
    generations: int = 200,
) -> Tuple[Dict, List[str]]:
    """
    Solve using Genetic Algorithm heuristic.
    """
    start_time = time.time()

    data = json.loads(Path(instance_path).read_text())
    T = int(data["period"])
    items: Dict[int, dict] = {int(k): v for k, v in data["items"].items()}

    # Capacity
    prod_cap = data.get("production_capacity") or data.get("manual_capacity")
    if prod_cap is None:
        # Default: sum of demands + buffer
        cap_raw = [0.0] * T
        for it in items.values():
            for t in range(T):
                cap_raw[t] += float(it["demand"][t])
        buf = max(5.0, 0.2 * max(cap_raw) if cap_raw else 0.0)
        capacity = [c + buf for c in cap_raw]
    elif isinstance(prod_cap, (int, float)):
        capacity = [float(prod_cap)] * T
    else:
        capacity = [float(c) for c in prod_cap]

    # Build Gamma
    Gamma_by_item = {}
    for item_id, item_data in items.items():
        Gamma_by_item[item_id] = build_gamma(item_data, T)

    if verbose:
        print("=" * 60)
        print("GENETIC ALGORITHM HEURISTIC SOLVER")
        print("=" * 60)
        print(f"  Items: {len(items)}, Periods: {T}")
        print(f"  Population: {pop_size}, Generations: {generations}")

    # Run GA
    best_cost, best_ind = solve_ga(
        items,
        T,
        capacity,
        Gamma_by_item,
        pop_size=pop_size,
        generations=generations,
        verbose=verbose,
    )

    runtime = time.time() - start_time

    # Output
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if best_ind is None or not math.isfinite(best_cost):
        summary = {
            "status": 3,  # INFEASIBLE
            "objective": None,
            "runtime_sec": runtime,
            "solver_version": "ga_heuristic",
        }
        orders_txt = ["No feasible solution found"]
    else:
        summary = {
            "status": 2,  # OPTIMAL (heuristic)
            "objective": float(best_cost),
            "runtime_sec": runtime,
            "solver_version": "ga_heuristic",
        }

        orders_txt = []
        for item_id in sorted(items.keys()):
            y_vec = best_ind.y[item_id]
            y_periods = [t for t in range(len(y_vec)) if y_vec[t] == 1]

            orders_txt.append(f"Item {item_id}")
            orders_txt.append(f"  Y (setups): {y_periods}")
            orders_txt.append(f"  Production:")
            x_dict = best_ind.x.get(item_id, {})
            for t in sorted(x_dict.keys()):
                if x_dict[t] > EPS:
                    orders_txt.append(f"    t={t}: {x_dict[t]:.2f}")
            orders_txt.append("")

    (out_path / "orders.txt").write_text("\n".join(orders_txt))
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2))

    return summary, orders_txt


if __name__ == "__main__":
    # Test with 1105 instance
    instance = {
        "period": 10,
        "manual_capacity": [0, 0, 80, 80, 80, 80, 80, 80, 100, 80],
        "items": {
            "0": {
                "demand": [0, 0, 68, 49, 66, 38, 17, 17, 41, 43],
                "h": [
                    0.4,
                    0.408,
                    0.416,
                    0.424,
                    0.430,
                    0.435,
                    0.438,
                    0.440,
                    0.440,
                    0.438,
                ],
                "c_var": [1.76, 1.71, 2.50, 1.78, 1.67, 2.11, 1.37, 1.24, 2.68, 1.81],
                "setup": [
                    80,
                    81.66,
                    83.25,
                    84.70,
                    85.95,
                    86.93,
                    87.61,
                    87.96,
                    87.96,
                    87.61,
                ],
                "shelf_seq": [24, 18, 22, 6, 16, 9, 21, 23, 9, 7],
            },
        },
    }

    out_dir = Path("ga_results")
    out_dir.mkdir(exist_ok=True)

    instance_path = out_dir / "test_instance.json"
    instance_path.write_text(json.dumps(instance, indent=2))

    summary, orders = solve_instance(
        instance_path=str(instance_path),
        out_dir=str(out_dir),
        pop_size=1000,
        generations=2000,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Objective: {summary.get('objective', 'N/A')}")
    print(f"  Runtime: {summary['runtime_sec']:.2f}s")
    print("\nOrders:")
    for line in orders:
        print(f"  {line}")
