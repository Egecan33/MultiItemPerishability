from __future__ import annotations
import random, math, json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path

# --------------------- knobs ---------------------
AUTO_CAPACITY_BUFFER_FRAC = 0.20
SETUP_SEQ_AMPLITUDE = 0.10
SETUP_SEQ_PERIOD = 30.0
SETUP_SEQ_JITTER = 0.04

# Per-item capacity policies (generator-side)
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
    c_var: Union[float, List[float]]
    h: Union[float, List[float]]
    b_var: float = 0.0  # kept for compatibility; solver ignores
    shelf_seq: List[int] = field(default_factory=list)  # m_{i,t}
    cap_seq: Optional[List[float]] = None  # generator-side convenience


@dataclass
class Lot:
    period: int
    items: Dict[int, Item] = field(default_factory=dict)
    manual_capacity: Optional[List[int]] = None  # production cap κ_t alias
    warehouse_capacity: Optional[float] = None

    # Lost-sales controls (passed straight to solver JSON)
    allow_unmet_demand: bool = False  # or allow_lost_sales
    lost_sales_penalty: Optional[float] = None  # scalar; None => auto
    lost_sales_penalty_factor: float = 200.0

    @property
    def capacity(self) -> List[int]:
        """Auto κ_t = sum demand_t + buffer, if manual_capacity not provided."""
        if self.manual_capacity is not None:
            return list(self.manual_capacity)
        T = self.period
        cap_raw = [0] * T
        for it in self.items.values():
            for t in range(T):
                cap_raw[t] += int(it.demand[t])
        max_cap = max(cap_raw) if cap_raw else 0
        buf = max(5, int(AUTO_CAPACITY_BUFFER_FRAC * max_cap))
        return [int(c + buf) for c in cap_raw]

    def to_mip_json(self, path: str | Path, *, indent: int = 2) -> None:
        """
        Writes exactly what mip/solver_mip_lefo.py expects:
          - top-level "item_capacity": {item_id: [cap_t]}  (if any provided)
          - "manual_capacity" (solver treats as production capacity)
          - lost-sales knobs when enabled
        """
        # Collect per-item caps into top-level dict as the solver expects
        item_capacity: Dict[str, List[float]] = {}
        for i, it in self.items.items():
            if it.cap_seq is not None:
                item_capacity[str(i)] = [float(x) for x in it.cap_seq]

        payload = {
            "period": int(self.period),
            "items": {
                str(i): {
                    "demand": [int(x) for x in it.demand],
                    "setup": it.setup,  # scalar or list
                    "c_var": it.c_var,  # scalar or list
                    "h": it.h,  # scalar or list
                    "b_var": float(it.b_var),
                    "shelf_seq": [int(x) for x in it.shelf_seq],
                }
                for i, it in self.items.items()
            },
            # production capacity (alias name kept for solver compatibility)
            "manual_capacity": [
                int(x) for x in (self.manual_capacity or self.capacity)
            ],
            "warehouse_capacity": (
                float(self.warehouse_capacity)
                if self.warehouse_capacity is not None
                else None
            ),
        }

        if item_capacity:
            payload["item_capacity"] = item_capacity

        # Lost-sales controls
        if self.allow_unmet_demand:
            payload["allow_unmet_demand"] = True
            if self.lost_sales_penalty is not None:
                payload["lost_sales_penalty"] = float(self.lost_sales_penalty)
            if self.lost_sales_penalty_factor is not None:
                payload["lost_sales_penalty_factor"] = float(
                    self.lost_sales_penalty_factor
                )

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
    seq: List[float] = []
    for t in range(period):
        seasonal = 1.0 + amp * math.sin(2 * math.pi * t / per)
        jitter = 1.0 + (random.uniform(-jit, jit) if jit > 0 else 0.0)
        seq.append(float(base) * seasonal * jitter)
    return seq


def _build_item_cap_seq(demand: List[int], policy: str) -> Optional[List[int]]:
    if policy == "none":
        return None
    T = len(demand)
    if policy == "demand_pad":
        return [
            int(max(demand[t] + ITEM_CAP_PAD, ITEM_CAP_MULT * demand[t]))
            for t in range(T)
        ]
    if policy == "uniform_range":
        lo, hi = ITEM_CAP_UNIFORM_RANGE
        return [int(random.randint(int(lo), int(hi))) for _ in range(T)]
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
    zero_head: int = 0,  # NEW: force first Z periods to 0 demand
    allow_unmet_demand: bool = False,  # NEW: pass through to solver JSON
    lost_sales_penalty: Optional[float] = None,  # NEW
    lost_sales_penalty_factor: float = 200.0,  # NEW
    seed: int = 0,
) -> Lot:
    """
    specs: (item_id, setup_base_or_range, c, h, (m_min, m_max))
           Put a scalar for fixed setup or a (lo,hi) tuple to randomize.
    """
    random.seed(seed)
    lot = Lot(
        period=int(period),
        manual_capacity=[int(x) for x in manual_capacity] if manual_capacity else None,
        warehouse_capacity=(
            float(warehouse_capacity) if warehouse_capacity is not None else None
        ),
        allow_unmet_demand=bool(allow_unmet_demand),
        lost_sales_penalty=(
            float(lost_sales_penalty) if lost_sales_penalty is not None else None
        ),
        lost_sales_penalty_factor=float(lost_sales_penalty_factor),
    )

    lo_d, hi_d = demand_range
    for rec in specs:
        if len(rec) != 5:
            raise ValueError(
                "spec must be (idx, setup_base|(lo,hi), c, h, (m_lo,m_hi))"
            )
        idx, stp_raw, c, h, shelf_rng = rec

        # demand with optional zero-head
        demand = [int(random.randint(int(lo_d), int(hi_d))) for _ in range(period)]
        if zero_head > 0:
            for z in range(min(int(zero_head), period)):
                demand[z] = 0

        # shelf life sequence
        m_lo, m_hi = int(shelf_rng[0]), int(shelf_rng[1])
        if m_lo == m_hi:
            shelf_seq = [m_lo] * period
        else:
            shelf_seq = [int(random.randint(m_lo, m_hi)) for _ in range(period)]

        # setup (scalar or seasonal list)
        base_setup = _sample_setup_base(stp_raw)
        setup_val: Union[float, List[float]] = (
            _build_setup_sequence(float(base_setup), period)
            if setup_seq_enable
            else float(base_setup)
        )

        cap_seq = _build_item_cap_seq(demand, item_cap_seq_policy)

        lot.items[int(idx)] = Item(
            id=int(idx),
            demand=[int(x) for x in demand],
            setup=setup_val,
            c_var=float(c),
            h=float(h),
            shelf_seq=[int(x) for x in shelf_seq],
            cap_seq=cap_seq,
        )
    return lot
