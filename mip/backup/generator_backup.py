from __future__ import annotations
import random, math, json
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path

# --------------------- knobs (same spirit as your old code) -----------------
AUTO_CAPACITY_BUFFER_FRAC = 0.20
SETUP_SEQ_AMPLITUDE = 0.10
SETUP_SEQ_PERIOD = 30.0
SETUP_SEQ_JITTER = 0.04

# Per-item capacity policies (kept for compatibility; not used by solver by default)
ITEM_CAP_SEQ_POLICY = "none"  # "none" | "demand_pad" | "uniform_range"
ITEM_CAP_PAD = 50
ITEM_CAP_MULT = 1.5
ITEM_CAP_UNIFORM_RANGE = (400, 600)

# ---------------------------------------------------------------------------


@dataclass
class Item:
    id: int
    demand: List[int]
    setup: Union[float, List[float]]
    c_var: float
    h: float
    b_var: float = 0.0  # kept for compatibility; solver ignores
    shelf_seq: List[int] = field(default_factory=list)  # m_{i,t}
    cap_seq: Optional[List[float]] = None  # optional cap_{i,t}


@dataclass
class Lot:
    period: int
    capacity_pad: int = 0
    items: Dict[int, Item] = field(default_factory=dict)
    manual_capacity: Optional[List[int]] = None
    warehouse_capacity: Optional[float] = None

    @property
    def capacity(self) -> List[int]:
        if self.manual_capacity is not None:
            return list(self.manual_capacity)
        T = self.period
        cap_raw = [0] * T
        for it in self.items.values():
            for t in range(T):
                cap_raw[t] += it.demand[t]
        max_cap = max(cap_raw) if cap_raw else 0
        buf = max(5, int(AUTO_CAPACITY_BUFFER_FRAC * max_cap))
        return [c + buf for c in cap_raw]

    def to_mip_json(self, path: str | Path, *, indent: int = 2) -> None:
        payload = {
            "period": self.period,
            "items": {
                str(i): {
                    "demand": it.demand,
                    "setup": it.setup,  # scalar or list s_{i,t}
                    "c_var": it.c_var,  # can be scalar; if you want c_{i,t} pass list instead
                    "h": it.h,  # can be scalar or list h_{i,t} (solver accepts both)
                    "b_var": it.b_var,
                    "shelf_seq": it.shelf_seq,
                    **({"cap_seq": it.cap_seq} if it.cap_seq is not None else {}),
                }
                for i, it in self.items.items()
            },
            "manual_capacity": self.manual_capacity,
            "warehouse_capacity": self.warehouse_capacity,
        }
        Path(path).write_text(json.dumps(payload, indent=indent), encoding="utf-8")


# --------------------- helpers ---------------------
def _sample_setup_base(stp: Union[float, Tuple[float, float]]) -> float:
    if isinstance(stp, (tuple, list)) and len(stp) == 2:
        lo, hi = float(stp[0]), float(stp[1])
        return random.uniform(lo, hi)
    return float(stp)


def _build_setup_sequence(
    base: float,
    period: int,
    amp=SETUP_SEQ_AMPLITUDE,
    per=SETUP_SEQ_PERIOD,
    jit=SETUP_SEQ_JITTER,
) -> List[float]:
    seq = []
    for t in range(period):
        seasonal = 1.0 + amp * math.sin(2 * math.pi * t / per)
        jitter = 1.0 + (random.uniform(-jit, jit) if jit > 0 else 0.0)
        seq.append(base * seasonal * jitter)
    return seq


def _build_item_cap_seq(demand: List[int], policy: str) -> Optional[List[int]]:
    if policy == "none":
        return None
    T = len(demand)
    if policy == "demand_pad":
        return [
            max(demand[t] + ITEM_CAP_PAD, int(ITEM_CAP_MULT * demand[t]))
            for t in range(T)
        ]
    if policy == "uniform_range":
        lo, hi = ITEM_CAP_UNIFORM_RANGE
        return [random.randint(int(lo), int(hi)) for _ in range(T)]
    raise ValueError(f"Unknown ITEM_CAP_SEQ_POLICY: {policy}")


# --------------------- main API ---------------------
def build_lot(
    *,
    period: int,
    demand_range: Tuple[int, int],
    specs: List[tuple],
    manual_capacity: Optional[List[int]] = None,
    default_shelf_rng: Tuple[int, int] = (3, 5),
    setup_seq_enable: bool = True,
    item_cap_seq_policy: str = "none",
    warehouse_capacity: Optional[float] = None,
    seed: int = 0,
) -> Lot:
    """
    specs: (item_id, setup_base_or_range, c, h, (m_min, m_max))
           Put a scalar for fixed setup or a (lo,hi) tuple to randomize.
    """
    random.seed(seed)
    lot = Lot(
        period=period,
        manual_capacity=manual_capacity,
        warehouse_capacity=warehouse_capacity,
    )

    lo_d, hi_d = demand_range
    for rec in specs:
        if len(rec) == 5:
            idx, stp_raw, c, h, shelf_rng = rec
        else:
            raise ValueError(
                "spec must be (idx, setup_base|(lo,hi), c, h, (m_lo,m_hi))"
            )

        demand = [random.randint(lo_d, hi_d) for _ in range(period)]
        if shelf_rng[0] == shelf_rng[1]:
            shelf_seq = [int(shelf_rng[0])] * period
        else:
            shelf_seq = [
                random.randint(int(shelf_rng[0]), int(shelf_rng[1]))
                for _ in range(period)
            ]

        base_setup = _sample_setup_base(stp_raw)
        setup_val = (
            _build_setup_sequence(base_setup, period)
            if setup_seq_enable
            else float(base_setup)
        )
        cap_seq = _build_item_cap_seq(demand, item_cap_seq_policy)

        lot.items[idx] = Item(
            id=idx,
            demand=demand,
            setup=setup_val,
            c_var=float(c),
            h=float(h),
            shelf_seq=shelf_seq,
            cap_seq=cap_seq,
        )
    return lot
