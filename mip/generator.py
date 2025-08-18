---

# `mip/generator.py`

```python
# mip/generator.py
from __future__ import annotations
import json, math, random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# -------- Global knobs (same spirit as your original generator) ----------
SEED = 0
SETUP_SEQ_ENABLE = True
SETUP_SEQ_AMPLITUDE = 0.10
SETUP_SEQ_PERIOD = 30.0
SETUP_SEQ_JITTER = 0.04

ITEM_CAP_SEQ_POLICY = "none"  # "none" | "demand_pad" | "uniform_range"
ITEM_CAP_PAD = 50
ITEM_CAP_MULT = 1.5
ITEM_CAP_UNIFORM_RANGE = (400, 600)

AUTO_CAPACITY_BUFFER_FRAC = 0.20  # used when manual capacity not provided


@dataclass
class Item:
    id: int
    demand: List[int]
    setup: Union[float, List[float]]       # s_i or s_{i,t}
    c_var: float                           # c_i (can treat as c_{i,t} if desired)
    h: float                               # h_i (per age period)
    shelf_seq: List[int]                   # m_{i,t}
    cap_seq: Optional[List[float]] = None  # optional cap_{i,t}


@dataclass
class Lot:
    period: int
    capacity_pad: int = 10
    items: Dict[int, Item] = field(default_factory=dict)
    manual_capacity: Optional[List[int]] = None
    warehouse_capacity: Optional[float] = None

    @property
    def capacity(self) -> List[int]:
        if self.manual_capacity is not None:
            return self.manual_capacity
        cap_raw = [
            sum(it.demand[t] for it in self.items.values())
            for t in range(self.period)
        ]
        max_cap = max(cap_raw) if cap_raw else 0
        buffer = max(5, int(AUTO_CAPACITY_BUFFER_FRAC * max_cap))
        return [c + buffer for c in cap_raw]

    def to_mip_json(self, path: str | Path, *, indent: int = 2) -> None:
        payload = {
            "period": self.period,
            "items": {
                str(i): {
                    "demand": it.demand,
                    "setup": it.setup,
                    "c_var": it.c_var,
                    "h": it.h,
                    "shelf_seq": it.shelf_seq,
                    **({"cap_seq": it.cap_seq} if it.cap_seq is not None else {}),
                }
                for i, it in self.items.items()
            },
        }
        if self.manual_capacity is not None:
            payload["manual_capacity"] = list(self.manual_capacity)
        if self.warehouse_capacity is not None:
            payload["warehouse_capacity"] = float(self.warehouse_capacity)
        Path(path).write_text(json.dumps(payload, indent=indent), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Lot":
        data = json.loads(Path(path).read_text())
        T = int(data["period"])
        items = {
            int(k): Item(
                id=int(k),
                demand=list(v["demand"]),
                setup=v["setup"],
                c_var=float(v["c_var"]),
                h=float(v["h"]),
                shelf_seq=list(v["shelf_seq"]),
                cap_seq=list(v["cap_seq"]) if "cap_seq" in v and v["cap_seq"] is not None else None,
            ) for k, v in data["items"].items()
        }
        return cls(
            period=T,
            items=items,
            manual_capacity=list(data.get("manual_capacity")) if data.get("manual_capacity") else None,
            warehouse_capacity=float(data.get("warehouse_capacity")) if data.get("warehouse_capacity") is not None else None,
        )


def _sample_setup_base(stp: Union[float, Tuple[float, float]]) -> float:
    if isinstance(stp, (list, tuple)) and len(stp) == 2:
        lo, hi = float(stp[0]), float(stp[1])
        return random.uniform(lo, hi)
    return float(stp)


def _build_setup_sequence(base: float, period: int) -> List[float]:
    seq = []
    amp = SETUP_SEQ_AMPLITUDE
    per = SETUP_SEQ_PERIOD
    jit = SETUP_SEQ_JITTER
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


def build_lot(
    *,
    period: int,
    demand_range: Tuple[int, int],
    specs: List[tuple],
    manual_capacity: Optional[List[int]],
    setup_seq_enable: bool = True,
    item_cap_seq_policy: str = "none",
    default_shelf_rng: Tuple[int, int] = (3,5),
    warehouse_capacity: Optional[float] = None,
    seed: int = 0,
) -> Lot:
    """
    specs per item:
      (i, setup_base_or_range, c, h)                         -> default shelf range (3,5)
      (i, setup_base_or_range, c, h, (m_lo, m_hi))           -> custom shelf range
    """
    random.seed(seed)
    lo_d, hi_d = map(int, demand_range)

    lot = Lot(period=period, manual_capacity=manual_capacity, warehouse_capacity=warehouse_capacity)
    for rec in specs:
        if len(rec) == 4:
            idx, stp_raw, c, hold = rec
            shelf_rng = default_shelf_rng
        elif len(rec) == 5:
            idx, stp_raw, c, hold, shelf_rng = rec
        else:
            raise ValueError("spec must be (i, setup, c, h) or (i, setup, c, h, (m_lo,m_hi))")

        demand = [random.randint(lo_d, hi_d) for _ in range(period)]
        if shelf_rng[0] == shelf_rng[1]:
            shelf_seq = [int(shelf_rng[0])] * period
        else:
            shelf_seq = [random.randint(int(shelf_rng[0]), int(shelf_rng[1])) for _ in range(period)]

        base_setup = _sample_setup_base(stp_raw)
        setup_val: Union[float, List[float]] = (
            _build_setup_sequence(base_setup, period) if setup_seq_enable else float(base_setup)
        )
        cap_seq = _build_item_cap_seq(demand, item_cap_seq_policy)

        lot.items[int(idx)] = Item(
            id=int(idx),
            demand=demand,
            setup=setup_val,
            c_var=float(c),
            h=float(hold),
            shelf_seq=shelf_seq,
            cap_seq=cap_seq,
        )
    return lot