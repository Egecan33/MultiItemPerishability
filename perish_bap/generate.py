#!/usr/bin/env python3
# generator.py
# No-arg, dead-simple specs.json generator.

import json, random
from pathlib import Path

# ---- tweak here if ever needed ----
PERIOD = 60
CAPACITY = 1870
CAPACITY_PAD = 10
SEED = 0
DEMAND_MIN, DEMAND_MAX = 1, 150
OUT = Path("specs.json")

ITEMS = [
    {
        "id": 0,
        "setup": 127.5,
        "b": 5.0,
        "c": 2.0,
        "h": 0.4,
        "shelf": [1, 50],
        "agency": {"name": "Acme Foods", "sku": "ACM-001", "region": "EMEA"},
    },
    {
        "id": 1,
        "setup": 29.0,
        "b": 5.0,
        "c": 3.0,
        "h": 0.6,
        "shelf": [1, 60],
        "agency": {"name": "BlueBay", "sku": "BBY-244", "region": "NA"},
    },
    {
        "id": 2,
        "setup": 50.0,
        "b": 5.0,
        "c": 1.0,
        "h": 0.3,
        "shelf": [3, 70],
        "agency": {"name": "BlueBay", "sku": "BBY-993", "region": "NA"},
    },
    {
        "id": 3,
        "setup": 75.0,
        "b": 5.0,
        "c": 4.0,
        "h": 0.5,
        "shelf": [3, 5],
        "agency": {"name": "Acme Foods", "sku": "ACM-440", "region": "EMEA"},
    },
    {
        "id": 4,
        "setup": 100000.0,
        "b": 5.0,
        "c": 2.5,
        "h": 0.4,
        "shelf": [3, 50],
        "agency": {"name": "Orchard", "sku": "ORC-777", "region": "APAC"},
    },
    {
        "id": 5,
        "setup": 60000.0,
        "b": 5.0,
        "c": 3.5,
        "h": 0.7,
        "shelf": [1, 50],
        "agency": {"name": "Orchard", "sku": "ORC-123", "region": "APAC"},
    },
    {
        "id": 6,
        "setup": 80.0,
        "b": 5.0,
        "c": 2.2,
        "h": 0.4,
        "shelf": [1, 60],
        "agency": {"name": "Acme Foods", "sku": "ACM-999", "region": "EMEA"},
    },
]
# -----------------------------------


def make_demands(T: int, lo: int, hi: int, seed: int, item_id: int):
    rnd = random.Random((seed << 16) ^ item_id)
    return [rnd.randint(lo, hi) for _ in range(T)]


def main():
    items_out = []
    for rec in ITEMS:
        i = int(rec["id"])
        base = {
            "id": i,
            "setup": float(rec["setup"]),
            "b": float(rec["b"]),
            "c": float(rec["c"]),
            "h": float(rec["h"]),
            # keep 2-length range as 'shelf' (loader will expand to sequence)
            "shelf": rec["shelf"],
            "agency": rec.get("agency"),
            # full per-period demand (so loader never complains)
            "demand": make_demands(PERIOD, DEMAND_MIN, DEMAND_MAX, SEED, i),
        }
        items_out.append(base)

    doc = {
        "period": PERIOD,
        "capacity_pad": CAPACITY_PAD,
        "manual_capacity": [CAPACITY] * PERIOD,  # exact length = period
        "random_seed": SEED,
        "demand_min": DEMAND_MIN,
        "demand_max": DEMAND_MAX,
        "items": items_out,  # list; your loader accepts list or dict
    }

    OUT.write_text(json.dumps(doc, indent=2))
    print(f"[OK] wrote {OUT.resolve()}")


if __name__ == "__main__":
    main()
