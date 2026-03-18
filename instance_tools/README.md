# instance_tools

Standalone (no Streamlit) toolkit for perishable lot-sizing instance generation and feasibility mending.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Gurobi must be installed and licensed (needed only for fix_instances.py)
```

## Two generation modes

### Lib mode — real demands from the 540 dataset (recommended)

```bash
# Step 1 (one-time): parse the 540 folder into a demand library
python parse_demands_and_insert.py \
    --dataset-root ../540 \
    --demand-lib-root ./demands_lib

# Step 2: generate one instance
python generator.py \
    --preset X111112 --j 0 \
    --demand-lib-root ./demands_lib \
    --out instance.json

# Generate ALL presets, 5 replicas each
python generator.py \
    --all --replicas 5 \
    --demand-lib-root ./demands_lib \
    --out-dir ./instances
```

### RNG mode — fully synthetic demands (no 540 folder needed)

```bash
python generator.py --preset X111112 --j 0 --out instance.json
python generator.py --all --replicas 5 --out-dir ./instances
```

### List all preset codes

```bash
python generator.py --list-presets   # prints all 1080 X-codes
```

## Inserting into Supabase (optional)

```bash
# Insert classes + instances directly (lib mode, real demands)
python parse_demands_and_insert.py \
    --demand-lib-root ./demands_lib \
    --create-all \
    --instances-per-class 5 \
    --L-list 2,5,7,9,11,12 \
    -y
```

## Fixing infeasible instances in Supabase

```bash
DRY_RUN=1 python fix_instances.py   # dry run
python fix_instances.py             # live run

# Optional env vars:
#   SUPABASE_URL, SUPABASE_ANON_KEY
#   PAGE_SIZE=200  SOLVER_TIME_LIMIT_SEC=60  DRY_RUN=1
```

## Files

| File | Purpose |
|---|---|
| `generator.py` | Instance generator — lib mode (540 demands) or RNG mode |
| `class_gen.py` | Class config presets (1080 X-codes). Writes `local_presets.json`. |
| `parse_demands_and_insert.py` | Parses 540 → demand library; inserts classes+instances to Supabase |
| `fix_instances.py` | Fetches instances from Supabase and fixes infeasible ones |
| `mip/` | Gurobi MIP solvers (used by `fix_instances.py`) |

## X-code legend

```
X  A  B  C  V  M  L
   |  |  |  |  |  |
   |  |  |  |  |  └── TBO L parameter (from L-list: 2,5,7,9,11,12)
   |  |  |  |  └───── shelf-life group: A=[0,T/2], B=[0,3T/4], C=[5,T/2]
   |  |  |  └──────── demand CV: L=low, H=high
   |  |  └─────────── capacity tightness: 1=Loose, 2=Medium
   |  └────────────── #items: 1→10, 2→20, 3→30
   └───────────────── periods: 1→20, 2→30, 3→40
```

## Dependencies

- `numpy` — RNG and array operations
- `supabase` — Supabase read/write (`parse_demands_and_insert.py`, `fix_instances.py`)
- `gurobipy` — MIP solver (`fix_instances.py`); requires a valid Gurobi license
