from __future__ import annotations
import json
import hashlib
from typing import Dict, Tuple, Iterable

# ---------------------------
# Encoding:
#   X{P}{N}{C}{V}{M}{L}
#   P: 1→20, 2→30, 3→40  (zero_head: 2,3,4)
#   N: 1→10, 2→20, 3→30
#   C: 1→Loose, 2→Medium
#   V: 1→CV low (0–125), 2→CV high (0–200)
#   M: 1→(1,10), 2→(5,15), 3→(10,20), 4→(5,25), 5→(10,30)
#   L: 1..10  (TBO L as float)
# ---------------------------

PERIOD_MAP: Dict[int, int] = {1: 20, 2: 30, 3: 40}
ZERO_HEAD_BY_T: Dict[int, int] = {20: 2, 30: 3, 40: 4}
NITEMS_MAP: Dict[int, int] = {1: 10, 2: 20, 3: 30}
CAP_TIGHT_MAP: Dict[int, str] = {1: "Loose", 2: "Medium"}
CV_MAP: Dict[int, Tuple[int, int]] = {1: (0, 125), 2: (0, 200)}
SHELF_MAP: Dict[int, Tuple[int, int]] = {
    1: (1, 10),
    2: (5, 15),
    3: (10, 20),
    4: (5, 25),
    5: (10, 30),
}


def seed_from_code(code: str) -> int:
    """Deterministic, varied seed per code (stable across runs)."""
    h = int(hashlib.md5(code.encode("utf-8")).hexdigest()[:8], 16)
    return 10000 + (h % 80000)  # 10000..89999


def make_config(P: int, N: int, C: int, V: int, M: int, L: int) -> dict:
    """Build one config dict from mode digits."""
    T = PERIOD_MAP[P]
    zero_head = ZERO_HEAD_BY_T[T]
    n_items = NITEMS_MAP[N]
    cap_tight = CAP_TIGHT_MAP[C]
    dem_lo, dem_hi = CV_MAP[V]
    m_lo, m_hi = SHELF_MAP[M]
    name = f"X{P}{N}{C}{V}{M}{L}"

    return {
        "name": name,
        "period": T,
        "cap_mode": "DemandBased",
        "cap_params": {"jitter_pct": 10.0},
        "cap_tight": cap_tight,
        "n_items": n_items,
        "dem_lo": dem_lo,
        "dem_hi": dem_hi,
        "m_lo": m_lo,
        "m_hi": m_hi,
        "c_mode": "uniform",
        "c_params": {"hi": 4.0, "lo": 2.0},
        "h_mode": "uniform",
        "h_params": {"hi": 1.0, "lo": 0.3},
        "s_mode": "tbo",
        "s_params": {"L": float(L), "jitter_pct": 10.0},
        "zero_head": zero_head,
        "batch_size": 1,  # as requested
        "seed_base": seed_from_code(name),
        "allow_unmet_demand": False,
        "lost_sales_penalty_factor": 200.0,
    }


def generate_all_presets() -> Iterable[dict]:
    """All combinations: P(1..3) × N(1..3) × C(1..2) × V(1..2) × M(1..5) × L(1..10)."""
    for P in (1, 2, 3):
        for N in (1, 2, 3):
            for C in (1, 2):
                for V in (1, 2):
                    for M in (1, 2, 3, 4, 5):
                        for L in [l for l in range(1, 13) if l in (2, 5, 7, 9, 11, 12)]:
                            yield make_config(P, N, C, V, M, L)


# Public: local presets list you can import directly
LOCAL_PRESETS = list(generate_all_presets())

if __name__ == "__main__":
    # Dump to a JSON file, if you want a static artifact:
    with open("local_presets.json", "w", encoding="utf-8") as f:
        json.dump(LOCAL_PRESETS, f, indent=2, ensure_ascii=False)

    # Quick eye-check example (X232236: P=2,N=3,C=2,V=2,M=3,L=6)
    ex = next(cfg for cfg in LOCAL_PRESETS if cfg["name"] == "X2322311")
    print(json.dumps(ex, indent=2, ensure_ascii=False))
