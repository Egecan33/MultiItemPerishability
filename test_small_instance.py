import json
import sys
sys.path.insert(0, ".")
from bnp.solver_bnp_dp_arc_based import solve_instance
from mip.solver_mip_no_cross import solve_instance as mip_solve

# Load and convert the small instance
small_data = json.load(open("single_item_instance.json"))

# Convert to proper format
instance = {
    "period": small_data["T"],
    "items": {
        "0": {
            "demand": small_data["demand"],
            "h": small_data["h"],
            "c_var": small_data["c_var"],
            "setup": small_data["setup"],
            "shelf_seq": small_data["shelf_seq"],
        }
    }
}

from pathlib import Path
instance_path = Path("small_test_instance.json")
instance_path.write_text(json.dumps(instance, indent=2))

print("=" * 60)
print("SMALL INSTANCE TEST (expected ~ 1105)")
print("=" * 60)

# Run B&P
print("\n--- Running B&P with LEFO ---")
summary_bnp, orders_bnp = solve_instance(str(instance_path), out_dir="bnp_small_out", verbose=True)
print(f"\nB&P Result: {summary_bnp.get('objective')}")

# Run MIP for comparison
print("\n--- Running MIP with LEFO for comparison ---")
summary_mip, orders_mip = mip_solve(str(instance_path), out_dir="mip_small_out")
print(f"MIP Result: {summary_mip.get('objective')}")

print(f"\n{'='*60}")
print(f"COMPARISON:")
print(f"  B&P: {summary_bnp.get('objective')}")
print(f"  MIP: {summary_mip.get('objective')}")
print(f"{'='*60}")
