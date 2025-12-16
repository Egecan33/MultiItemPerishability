"""
Analyze ZIO column space for the demo instance.
Find the 3 columns that achieve optimal 1105.
"""
import math
from itertools import product
import gurobipy as gp
from gurobipy import GRB

def main():
    print("=" * 70)
    print("ANALYZING ZIO COLUMN SPACE FOR DEMO INSTANCE")
    print("=" * 70)
    
    T = 10
    demand = [0, 68, 52, 22, 33, 45, 56, 33, 25, 5]
    h = 2.0
    s = 250.0
    
    demand_periods = [u for u in range(T) if demand[u] > 0]
    print(f"Demand periods: {demand_periods}")
    print(f"Demands: {[demand[u] for u in demand_periods]}")
    print(f"Total demand: {sum(demand)}")
    print()
    
    # For each demand period, list possible serving periods (ZIO: must be <= demand period)
    print("Possible serving periods for each demand:")
    for u in demand_periods:
        serving = list(range(u + 1))  # Can serve from any period <= u
        holding_costs = [h * (u - t) * demand[u] for t in serving]
        print(f"  d[{u}]={demand[u]:3d}: serve from {serving}, holding={holding_costs}")
    print()
    
    # Enumerate ALL ZIO columns (arc assignments)
    # Each demand can be served from any period <= demand period
    options = [list(range(u + 1)) for u in demand_periods]
    total_cols = math.prod(len(o) for o in options)
    print(f"Total possible ZIO columns: {total_cols}")
    print()
    
    # Generate all columns
    columns = []
    for combo in product(*options):
        assignment = {u: t for u, t in zip(demand_periods, combo)}
        
        # Setup periods and cost
        setup_periods = set(assignment.values())
        setup_cost = len(setup_periods) * s
        
        # Holding cost
        holding_cost = sum(
            h * (u - t) * demand[u] 
            for u, t in assignment.items()
        )
        
        total_cost = setup_cost + holding_cost
        
        # Production at each period
        production = {}
        for u, t in assignment.items():
            production[t] = production.get(t, 0) + demand[u]
        
        # Z indicators (which arcs are active)
        z = {(t, u): 1 for u, t in assignment.items()}
        
        columns.append({
            'assignment': assignment,
            'setup_periods': sorted(setup_periods),
            'n_setups': len(setup_periods),
            'setup_cost': setup_cost,
            'holding_cost': holding_cost,
            'total_cost': total_cost,
            'production': production,
            'z': z,
        })
    
    # Sort by total cost
    columns.sort(key=lambda c: c['total_cost'])
    
    print("=" * 70)
    print("TOP 15 LOWEST-COST ZIO COLUMNS")
    print("=" * 70)
    for i, col in enumerate(columns[:15]):
        print(f"\nColumn {i+1}: total_cost={col['total_cost']:.0f} "
              f"(setup={col['setup_cost']:.0f}, hold={col['holding_cost']:.0f})")
        print(f"  Setups at periods: {col['setup_periods']}")
        print(f"  Arc assignment (demand_period -> prod_period):")
        for u in demand_periods:
            t = col['assignment'][u]
            hc = h * (u - t) * demand[u]
            print(f"    d[{u}]={demand[u]:2d} <- prod[{t}], hold_cost={hc:.0f}")
    
    # Now find optimal LP relaxation using ALL columns
    print("\n" + "=" * 70)
    print("FINDING OPTIMAL LP RELAXATION")
    print("=" * 70)
    
    m = gp.Model("ZIO_LP")
    m.Params.OutputFlag = 0
    
    n_cols = len(columns)
    lam = [m.addVar(lb=0, ub=1, name=f"lam_{i}") for i in range(n_cols)]
    
    # Convexity: Σλ = 1
    m.addConstr(gp.quicksum(lam) == 1, "convex")
    
    # Objective: min Σ cost_k * λ_k
    m.setObjective(
        gp.quicksum(columns[i]['total_cost'] * lam[i] for i in range(n_cols)),
        GRB.MINIMIZE
    )
    
    m.optimize()
    
    if m.Status == GRB.OPTIMAL:
        print(f"\n*** OPTIMAL LP VALUE: {m.ObjVal:.4f} ***\n")
        
        active_cols = [(i, lam[i].X) for i in range(n_cols) if lam[i].X > 1e-6]
        active_cols.sort(key=lambda x: -x[1])
        
        print(f"Number of active columns: {len(active_cols)}")
        print("\nActive columns:")
        
        for i, val in active_cols:
            col = columns[i]
            print(f"\n  λ[{i}] = {val:.6f} (cost={col['total_cost']:.0f})")
            print(f"    Setups: {col['setup_periods']}")
            print(f"    Arcs: ", end="")
            arcs = [f"({t}->{u})" for u, t in col['assignment'].items()]
            print(", ".join(arcs))
        
        # Compute aggregated production
        print("\n" + "-" * 50)
        print("AGGREGATED SOLUTION (convex combination):")
        print("-" * 50)
        
        x_agg = {}
        for i, val in active_cols:
            col = columns[i]
            for t, x_t in col['production'].items():
                x_agg[t] = x_agg.get(t, 0) + x_t * val
        
        print("\nProduction (X):")
        for t in sorted(x_agg.keys()):
            if x_agg[t] > 0.01:
                print(f"  x[{t}] = {x_agg[t]:.2f}")
        
        print(f"\nTotal production: {sum(x_agg.values()):.2f}")
        
        # Check which demands are split
        print("\nDemand splitting analysis:")
        for u in demand_periods:
            sources = {}
            for i, val in active_cols:
                t = columns[i]['assignment'][u]
                sources[t] = sources.get(t, 0) + val
            if len(sources) > 1:
                print(f"  d[{u}]={demand[u]} SPLIT: ", end="")
                for t, frac in sorted(sources.items()):
                    print(f"{frac*100:.1f}% from t={t}, ", end="")
                print()
            else:
                t = list(sources.keys())[0]
                print(f"  d[{u}]={demand[u]} from t={t} (100%)")


if __name__ == "__main__":
    main()

