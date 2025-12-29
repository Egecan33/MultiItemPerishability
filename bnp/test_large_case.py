#!/usr/bin/env python3
"""Test the B&P solver with a larger case."""

import random
import math
from solver_bnp_dp_full_plans import solve_branch_and_price, BnPLogger
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Parameters
T = 20  # 20 periods (slightly smaller for faster testing)
N_ITEMS = 15  # 15 items (more items)

# Generate instance
items = {}
capacity = []

# Capacity: start with 0, then ramp up
# Make capacity tighter to make problem harder
for t in range(T):
    if t < 2:
        capacity.append(0)
    else:
        # Capacity between 80-120 (tighter than before)
        capacity.append(random.randint(80, 120))

# Generate items
for i in range(N_ITEMS):
    demand = []
    setup = []
    h = []
    c_var = []
    shelf_seq = []
    
    # TBO (Time Between Orders) - determines setup cost pattern
    tbo = random.choice([2, 4, 6])
    
    for t in range(T):
        # Demand: 0 for first 2 periods, then random (higher demand)
        if t < 2:
            demand.append(0)
        else:
            demand.append(random.randint(30, 70))  # Increased from 20-60
        
        # Setup cost: increases with TBO pattern
        base_setup = 100 + i * 10
        setup.append(base_setup * (1 + 0.1 * (t % tbo)))
        
        # Holding cost: random between 0.3-1.0
        h.append(random.uniform(0.3, 1.0))
        
        # Variable cost: random between 1.5-3.0
        c_var.append(random.uniform(1.5, 3.0))
        
        # Shelf life: random between 3-10 (shorter, makes problem harder)
        shelf_seq.append(random.randint(3, 10))
    
    items[i] = {
        'demand': demand,
        'setup': setup,
        'h': h,
        'c_var': c_var,
        'shelf_seq': shelf_seq,
    }

print(f"Generated instance: {N_ITEMS} items, {T} periods")
print(f"Total demand per period: {[sum(items[i]['demand'][t] for i in range(N_ITEMS)) for t in range(T)]}")
print(f"Capacity per period: {capacity}")
print(f"\nRunning B&P solver...\n")

# Run solver
logger = BnPLogger(Path('/tmp'), enabled=False)
best_ub, best_lb, columns, lam_vals = solve_branch_and_price(
    items=items,
    capacity=capacity,
    T=T,
    time_limit=300,  # 5 minutes
    logger=logger,
    verbose=True,
)

print(f"\n{'='*60}")
print(f"Final Results:")
print(f"  Best Upper Bound: {best_ub:.2f}")
print(f"  Best Lower Bound: {best_lb:.2f}")
if best_ub < math.inf and best_lb > -math.inf:
    gap = ((best_ub - best_lb) / best_lb) * 100 if best_lb > 0 else 0
    print(f"  Gap: {gap:.2f}%")
print(f"{'='*60}")

