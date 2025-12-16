"""
Verify Professor's ZIO Columns for the 1105 optimal.

This script constructs the 4 exact columns from the professor's diagram
and verifies that their convex combination achieves 1105.

Instance:
- demand = [0, 0, 68, 49, 66, 38, 17, 17, 41, 43]
- s = 250, h = 2

The 4 columns all have Y = [0,0,1,1,1,0,1,1,0,0] but differ in X patterns
based on which periods have "dummy setups" (Y=1, X=0).
"""

import gurobipy as gp
from gurobipy import GRB

def main():
    print("=" * 70)
    print("VERIFYING PROFESSOR'S ZIO COLUMNS")
    print("=" * 70)
    
    # Instance parameters (from solver_bnp_dp.py lines 1925-1958)
    T = 10
    demand = [0, 0, 68, 49, 66, 38, 17, 17, 41, 43]
    
    # Period-varying setup costs
    setup = [80, 81.66, 83.25, 84.70, 85.95, 86.93, 87.61, 87.96, 87.96, 87.61]
    
    # Period-varying holding costs
    h = [0.4, 0.408, 0.416, 0.424, 0.430, 0.435, 0.438, 0.440, 0.440, 0.438]
    
    # Variable production costs
    c_var = [1.76, 1.71, 2.50, 1.78, 1.67, 2.11, 1.37, 1.24, 2.68, 1.81]
    
    # Capacity constraints!
    capacity = [0, 0, 80, 80, 80, 80, 80, 80, 100, 80]
    
    total_demand = sum(demand)
    print(f"Demand: {demand}")
    print(f"Total demand: {total_demand}")
    print(f"Setup costs (by period): {setup}")
    print(f"Holding costs (by period): {h}")
    print(f"Variable costs (by period): {c_var}")
    print()
    
    # Define the 4 columns exactly as specified
    # Each column has:
    # - x: production quantities at each period
    # - y: setup indicators (all same: [0,0,1,1,1,0,1,1,0,0])
    # - z: arc indicators (which period serves which demand)
    # - dashed: periods with dummy setups (Y=1, X=0)
    
    columns = []
    
    # Column 1: No dummy setups
    # Covers: d[2] from t=2, d[3] from t=3, d[4]+d[5] from t=4, d[6] from t=6, d[7]+d[8]+d[9] from t=7
    col1 = {
        "name": "Column 1 (no dummy)",
        "x": [0, 0, 68, 49, 104, 0, 17, 101, 0, 0],
        "y": [0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        "z": {
            (2, 2): 1,  # t=2 serves d[2]=68
            (3, 3): 1,  # t=3 serves d[3]=49
            (4, 4): 1,  # t=4 serves d[4]=66
            (4, 5): 1,  # t=4 serves d[5]=38
            (6, 6): 1,  # t=6 serves d[6]=17
            (7, 7): 1,  # t=7 serves d[7]=17
            (7, 8): 1,  # t=7 serves d[8]=41
            (7, 9): 1,  # t=7 serves d[9]=43
        },
        "dashed": [],
    }
    columns.append(col1)
    
    # Column 2: Dummy at t=4
    # Covers: d[2] from t=2, d[3]+d[4]+d[5] from t=3, d[6] from t=6, d[7]+d[8]+d[9] from t=7
    col2 = {
        "name": "Column 2 (dummy at t=4)",
        "x": [0, 0, 68, 153, 0, 0, 17, 101, 0, 0],
        "y": [0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        "z": {
            (2, 2): 1,  # t=2 serves d[2]=68
            (3, 3): 1,  # t=3 serves d[3]=49
            (3, 4): 1,  # t=3 serves d[4]=66
            (3, 5): 1,  # t=3 serves d[5]=38
            (6, 6): 1,  # t=6 serves d[6]=17
            (7, 7): 1,  # t=7 serves d[7]=17
            (7, 8): 1,  # t=7 serves d[8]=41
            (7, 9): 1,  # t=7 serves d[9]=43
        },
        "dashed": [4],
    }
    columns.append(col2)
    
    # Column 3: Dummy at t=7
    # Covers: d[2] from t=2, d[3] from t=3, d[4]+d[5] from t=4, d[6]+d[7]+d[8]+d[9] from t=6
    col3 = {
        "name": "Column 3 (dummy at t=7)",
        "x": [0, 0, 68, 49, 104, 0, 118, 0, 0, 0],
        "y": [0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        "z": {
            (2, 2): 1,  # t=2 serves d[2]=68
            (3, 3): 1,  # t=3 serves d[3]=49
            (4, 4): 1,  # t=4 serves d[4]=66
            (4, 5): 1,  # t=4 serves d[5]=38
            (6, 6): 1,  # t=6 serves d[6]=17
            (6, 7): 1,  # t=6 serves d[7]=17
            (6, 8): 1,  # t=6 serves d[8]=41
            (6, 9): 1,  # t=6 serves d[9]=43
        },
        "dashed": [7],
    }
    columns.append(col3)
    
    # Column 4: Dummy at t=4 and t=7
    # Covers: d[2] from t=2, d[3]+d[4]+d[5] from t=3, d[6]+d[7]+d[8]+d[9] from t=6
    col4 = {
        "name": "Column 4 (dummy at t=4, t=7)",
        "x": [0, 0, 68, 153, 0, 0, 118, 0, 0, 0],
        "y": [0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        "z": {
            (2, 2): 1,  # t=2 serves d[2]=68
            (3, 3): 1,  # t=3 serves d[3]=49
            (3, 4): 1,  # t=3 serves d[4]=66
            (3, 5): 1,  # t=3 serves d[5]=38
            (6, 6): 1,  # t=6 serves d[6]=17
            (6, 7): 1,  # t=6 serves d[7]=17
            (6, 8): 1,  # t=6 serves d[8]=41
            (6, 9): 1,  # t=6 serves d[9]=43
        },
        "dashed": [4, 7],
    }
    columns.append(col4)
    
    # Calculate cost of each column
    print("=" * 70)
    print("COLUMN COSTS")
    print("=" * 70)
    
    for col in columns:
        # Setup cost: sum of setup[t] for periods with Y=1
        setup_cost = sum(setup[t] for t in range(T) if col["y"][t] == 1)
        
        # Variable production cost: sum of c_var[t] * x[t]
        var_cost = sum(c_var[t] * col["x"][t] for t in range(T))
        
        # Holding cost: for each arc (t, u), cost = h[t] * (u - t) * demand[u]
        # Note: holding cost uses h[t] from the production period
        holding_cost = 0.0
        for (t, u), z_val in col["z"].items():
            if z_val == 1:
                hc = h[t] * (u - t) * demand[u]
                holding_cost += hc
        
        col["setup_cost"] = setup_cost
        col["var_cost"] = var_cost
        col["holding_cost"] = holding_cost
        col["total_cost"] = setup_cost + var_cost + holding_cost
        
        # Verify total production = total demand
        total_prod = sum(col["x"])
        
        print(f"\n{col['name']}:")
        print(f"  X = {col['x']}")
        print(f"  Total production = {total_prod}")
        print(f"  Setups = {sum(col['y'])} at periods {[t for t in range(T) if col['y'][t] == 1]}")
        print(f"  Dummy setups = {col['dashed']}")
        print(f"  Setup cost = {setup_cost:.2f}")
        print(f"  Variable cost = {var_cost:.2f}")
        print(f"  Holding cost = {holding_cost:.2f}")
        print(f"  TOTAL COST = {col['total_cost']:.2f}")
    
    # Now solve LP with convexity constraint
    print("\n" + "=" * 70)
    print("LP WITH CONVEXITY ONLY (baseline)")
    print("=" * 70)
    
    m1 = gp.Model("ZIO_Convex")
    m1.Params.OutputFlag = 0
    
    lam = [m1.addVar(lb=0, ub=1, name=f"lam_{i}") for i in range(4)]
    m1.addConstr(gp.quicksum(lam) == 1, "convex")
    m1.setObjective(
        gp.quicksum(columns[i]["total_cost"] * lam[i] for i in range(4)),
        GRB.MINIMIZE
    )
    m1.optimize()
    
    print(f"\nOptimal LP value: {m1.ObjVal:.4f}")
    print("Active columns:")
    for i in range(4):
        if lam[i].X > 1e-6:
            print(f"  λ[{i+1}] = {lam[i].X:.6f}")
    
    # Now solve LP with X-linking constraints
    # The professor's formulation links aggregate X to specific values
    print("\n" + "=" * 70)
    print("LP WITH X-LINKING CONSTRAINTS")
    print("=" * 70)
    
    m2 = gp.Model("ZIO_XLink")
    m2.Params.OutputFlag = 0
    
    lam2 = [m2.addVar(lb=0, ub=1, name=f"lam_{i}") for i in range(4)]
    X_agg = [m2.addVar(lb=0, name=f"X_{t}") for t in range(T)]
    
    # Convexity
    m2.addConstr(gp.quicksum(lam2) == 1, "convex")
    
    # X-linking: aggregate X at each period
    for t in range(T):
        x_sum = gp.quicksum(columns[i]["x"][t] * lam2[i] for i in range(4))
        m2.addConstr(x_sum == X_agg[t], f"x_link_{t}")
    
    # Objective: minimize cost
    m2.setObjective(
        gp.quicksum(columns[i]["total_cost"] * lam2[i] for i in range(4)),
        GRB.MINIMIZE
    )
    m2.optimize()
    
    print(f"\nOptimal LP value: {m2.ObjVal:.4f}")
    print("\nActive columns:")
    for i in range(4):
        if lam2[i].X > 1e-6:
            print(f"  λ[{i+1}] = {lam2[i].X:.6f} (cost={columns[i]['total_cost']:.0f})")
    
    print("\nAggregate X values:")
    for t in range(T):
        if X_agg[t].X > 1e-6:
            print(f"  X[{t}] = {X_agg[t].X:.2f}")
    
    # Now let's try what the professor's LP structure suggests:
    # The constraints force specific X values at key periods
    print("\n" + "=" * 70)
    print("LP WITH PROFESSOR'S STRUCTURE")
    print("=" * 70)
    print("Adding constraints: X[3]=73, X[4]=80, X[6]=38, X[7]=80")
    
    m3 = gp.Model("ZIO_Professor")
    m3.Params.OutputFlag = 0
    
    lam3 = [m3.addVar(lb=0, ub=1, name=f"lam_{i}") for i in range(4)]
    
    # Convexity
    m3.addConstr(gp.quicksum(lam3) == 1, "convex")
    
    # Professor's constraints (from the LP structure)
    # X[3] = 73: 49*λ1 + 153*λ2 + 49*λ3 + 153*λ4 = 73
    m3.addConstr(49*lam3[0] + 153*lam3[1] + 49*lam3[2] + 153*lam3[3] == 73, "x3")
    
    # X[4] = 80: 104*λ1 + 0*λ2 + 104*λ3 + 0*λ4 = 80
    m3.addConstr(104*lam3[0] + 0*lam3[1] + 104*lam3[2] + 0*lam3[3] == 80, "x4")
    
    # X[6] = 38: 17*λ1 + 17*λ2 + 118*λ3 + 118*λ4 = 38
    m3.addConstr(17*lam3[0] + 17*lam3[1] + 118*lam3[2] + 118*lam3[3] == 38, "x6")
    
    # X[7] = 80: 101*λ1 + 101*λ2 + 0*λ3 + 0*λ4 = 80
    m3.addConstr(101*lam3[0] + 101*lam3[1] + 0*lam3[2] + 0*lam3[3] == 80, "x7")
    
    # Objective: minimize cost
    m3.setObjective(
        gp.quicksum(columns[i]["total_cost"] * lam3[i] for i in range(4)),
        GRB.MINIMIZE
    )
    m3.optimize()
    
    if m3.Status == GRB.OPTIMAL:
        print(f"\nOptimal LP value: {m3.ObjVal:.4f}")
        print("\nλ values:")
        for i in range(4):
            print(f"  λ[{i+1}] = {lam3[i].X:.6f}")
        
        # Verify aggregate X
        print("\nAggregate X values:")
        for t in [2, 3, 4, 6, 7]:
            x_t = sum(columns[i]["x"][t] * lam3[i].X for i in range(4))
            print(f"  X[{t}] = {x_t:.2f}")
        
        # Verify total production
        total_x = sum(
            sum(columns[i]["x"][t] * lam3[i].X for i in range(4))
            for t in range(T)
        )
        print(f"\nTotal aggregate production: {total_x:.2f}")
        
        # Compare with expected λ values
        print("\n" + "-" * 50)
        print("COMPARISON WITH PROFESSOR'S VALUES")
        print("-" * 50)
        expected_lam = [0.561310, 0.230769, 0.207921, 0.0]
        for i in range(4):
            diff = abs(lam3[i].X - expected_lam[i])
            status = "✓" if diff < 0.001 else "✗"
            print(f"  λ[{i+1}]: got {lam3[i].X:.6f}, expected {expected_lam[i]:.6f} {status}")
    else:
        print(f"LP Status: {m3.Status} (not optimal)")
    
    # LP with CAPACITY constraints - this is the key!
    print("\n" + "=" * 70)
    print("LP WITH CAPACITY CONSTRAINTS")
    print("=" * 70)
    print(f"Capacity: {capacity}")
    
    m4 = gp.Model("ZIO_Capacity")
    m4.Params.OutputFlag = 0
    
    lam4 = [m4.addVar(lb=0, ub=1, name=f"lam_{i}") for i in range(4)]
    
    # Convexity
    m4.addConstr(gp.quicksum(lam4) == 1, "convex")
    
    # Capacity constraints: aggregate X[t] <= capacity[t]
    for t in range(T):
        if capacity[t] > 0:
            x_sum = gp.quicksum(columns[i]["x"][t] * lam4[i] for i in range(4))
            m4.addConstr(x_sum <= capacity[t], f"cap_{t}")
    
    # Objective: minimize cost
    m4.setObjective(
        gp.quicksum(columns[i]["total_cost"] * lam4[i] for i in range(4)),
        GRB.MINIMIZE
    )
    m4.optimize()
    
    if m4.Status == GRB.OPTIMAL:
        print(f"\nOptimal LP value: {m4.ObjVal:.4f}")
        print("\nλ values:")
        for i in range(4):
            if lam4[i].X > 1e-6:
                print(f"  λ[{i+1}] = {lam4[i].X:.6f} (cost={columns[i]['total_cost']:.2f})")
        
        print("\nAggregate X values vs Capacity:")
        for t in range(T):
            if capacity[t] > 0:
                x_t = sum(columns[i]["x"][t] * lam4[i].X for i in range(4))
                status = "≤" if x_t <= capacity[t] + 0.01 else "VIOLATES"
                print(f"  X[{t}] = {x_t:.2f} {status} {capacity[t]}")
    else:
        print(f"LP Status: {m4.Status}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Convexity only:      {m1.ObjVal:.4f} (may violate capacity)")
    print(f"With X-linking:      {m2.ObjVal:.4f}")
    if m3.Status == GRB.OPTIMAL:
        print(f"Professor's struct:  {m3.ObjVal:.4f}")
    if m4.Status == GRB.OPTIMAL:
        print(f"WITH CAPACITY:       {m4.ObjVal:.4f}")
    print(f"Target optimal:      1105.0000")


if __name__ == "__main__":
    main()

