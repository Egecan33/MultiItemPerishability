#!/usr/bin/env python3
"""Test the B&P solver with Streamlit format instance."""

import json
import random
import math
from pathlib import Path
from solver_bnp_dp_full_plans import solve_instance

# Set random seed for reproducibility
random.seed(42)

# Create a Streamlit-format instance
# Based on typical Streamlit parameters: 10 items, 30 periods, shelf life 1-10
T = 30
N_ITEMS = 10

instance = {
    "period": T,
    "manual_capacity": [],
    "items": {}
}

# Generate capacity (Streamlit style: larger values)
for t in range(T):
    if t < 3:  # zero_head = 3
        instance["manual_capacity"].append(0)
    else:
        # Capacity between 100-150 (scaled down for testing)
        instance["manual_capacity"].append(random.randint(100, 150))

# Generate items (Streamlit format)
for i in range(N_ITEMS):
    demand = []
    setup = []
    h = []
    c_var = []
    shelf_seq = []
    
    # TBO (Time Between Orders)
    tbo = random.choice([2, 4, 6])
    base_setup = 100 + i * 10
    
    for t in range(T):
        # Demand: 0 for first 3 periods (zero_head), then random
        if t < 3:
            demand.append(0)
        else:
            demand.append(random.randint(20, 60))
        
        # Setup cost: TBO pattern
        setup.append(base_setup * (1 + 0.1 * (t % tbo)))
        
        # Holding cost: random between 0.3-1.0
        h.append(random.uniform(0.3, 1.0))
        
        # Variable cost: random between 1.5-3.0
        c_var.append(random.uniform(1.5, 3.0))
        
        # Shelf life: CRITICAL - Streamlit can have very short shelf lives (1-10)
        # This is what causes node explosion!
        shelf_seq.append(random.randint(1, 10))  # Can be as low as 1!
    
    instance["items"][str(i)] = {
        "demand": demand,
        "setup": setup,
        "h": h,
        "c_var": c_var,
        "shelf_seq": shelf_seq,
        "b_var": 0.0,
    }

# Save instance
instance_path = Path("test_streamlit_instance.json")
instance_path.write_text(json.dumps(instance, indent=2))

print(f"Generated Streamlit-format instance:")
print(f"  Items: {N_ITEMS}, Periods: {T}")
print(f"  Shelf life range: {min(min(instance['items'][str(i)]['shelf_seq']) for i in range(N_ITEMS))}-{max(max(instance['items'][str(i)]['shelf_seq']) for i in range(N_ITEMS))}")
print(f"  Capacity: {instance['manual_capacity'][:10]}...")
print(f"\nRunning solver...\n")

# Run solver
try:
    summary, orders = solve_instance(
        instance_path=str(instance_path),
        time_limit=120,  # 2 minutes
        verbose=True,
    )
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Status: {summary.get('status', 'UNKNOWN')}")
    print(f"  Objective: {summary.get('objective', 'N/A')}")
    print(f"  Gap: {summary.get('gap', 'N/A')}")
    runtime = summary.get('runtime', 'N/A')
    if isinstance(runtime, (int, float)):
        print(f"  Runtime: {runtime:.2f}s")
    else:
        print(f"  Runtime: {runtime}")
    print(f"{'='*60}")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

