"""
generator.py
============
Standalone (no Streamlit) instance generator for perishable lot-sizing.

Two modes:
  1. RNG mode  – demands generated from [dem_lo, dem_hi] range (no demand library needed)
  2. Lib mode  – demands come from a pre-built demand library (parse the 540 folder first)

Usage (CLI):

  # --- Lib mode (real 540 demands) ---

  # Step 1: build the demand library from the 540 folder (one-time)
  python parse_demands_and_insert.py \\
      --dataset-root ../540 \\
      --demand-lib-root ./demands_lib

  # Step 2: generate instances using real demands
  python generator.py --preset X111112 --j 0 --demand-lib-root ./demands_lib --out inst.json
  python generator.py --all --replicas 5 --demand-lib-root ./demands_lib --out-dir ./instances

  # --- RNG mode (no demand library needed) ---
  python generator.py --preset X111112 --j 0 --out inst.json
  python generator.py --all --replicas 5 --out-dir ./instances

  # List all preset codes
  python generator.py --list-presets

Usage (library):
  from generator import generate_instance          # auto-picks lib or rng based on args
  from generator import generate_instance_from_class   # RNG-only
  from parse_demands_and_insert import build_instance_from_class  # lib-only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAP_TIGHT_BETAS: Dict[str, float] = {
    "Loose": 1.10,
    "Medium": 1.02,
    "Tight": 0.96,
    "Tighter": 0.90,
    "Ultra": 0.80,
}
SETUP_TBO_JITTER_DEFAULT: float = 10.0


# ---------------------------------------------------------------------------
# RNG / array helpers
# ---------------------------------------------------------------------------

def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def clip_list(xs, lo=None, hi=None):
    if lo is not None:
        xs = np.maximum(xs, lo)
    if hi is not None:
        xs = np.minimum(xs, hi)
    return xs


def make_series(mode: str, T: int, params: dict, seed: int) -> List[float]:
    """
    Returns list[float] of length T.
    mode in {"scalar", "uniform", "normal", "linear", "seasonal", "manual"}
    """
    g = rng(seed)
    if mode == "scalar":
        return [float(params.get("value", 0.0))] * T
    if mode == "uniform":
        lo = float(params.get("lo", 0.0))
        hi = float(params.get("hi", 1.0))
        return list(g.uniform(lo, hi, size=T))
    if mode == "normal":
        mu = float(params.get("mean", 0.0))
        sd = float(params.get("std", 1.0))
        lo = params.get("clip_lo", None)
        hi = params.get("clip_hi", None)
        arr = clip_list(g.normal(mu, sd, size=T), lo, hi)
        return list(arr)
    if mode == "linear":
        base = float(params.get("base", 0.0))
        slope = float(params.get("slope", 0.0))
        noise = float(params.get("noise_std", 0.0))
        arr = base + slope * np.arange(T) + g.normal(0.0, noise, size=T)
        lo = params.get("clip_lo", None)
        hi = params.get("clip_hi", None)
        arr = clip_list(arr, lo, hi)
        return list(arr)
    if mode == "seasonal":
        base = float(params.get("base", 1.0))
        amp = float(params.get("amp", 0.0))
        per = float(params.get("period", max(1, T)))
        phase = float(params.get("phase", 0.0))
        noise = float(params.get("noise_std", 0.0))
        t = np.arange(T)
        arr = base * (1.0 + amp * np.sin(2 * np.pi * (t + phase) / per)) + g.normal(
            0.0, noise, size=T
        )
        lo = params.get("clip_lo", None)
        hi = params.get("clip_hi", None)
        arr = clip_list(arr, lo, hi)
        return list(arr)
    if mode == "manual":
        lst = params.get("list", [])
        if len(lst) != T:
            raise ValueError(f"Manual list length must equal T={T}.")
        return [float(x) for x in lst]
    raise ValueError(f"Unknown mode: {mode!r}")


def generate_cap_series(period: int, mode: str, params: dict, seed: int) -> List[int]:
    g = rng(seed)
    if mode == "Constant":
        return [int(params.get("value", 10000))] * period
    if mode == "Uniform":
        lo, hi = int(params.get("lo", 9000)), int(params.get("hi", 11000))
        return [int(x) for x in g.integers(lo, hi + 1, size=period)]
    mu = float(params.get("mean", 10000))
    sd = float(params.get("std", 500))
    clip_lo = float(params.get("clip_lo", 0))
    clip_hi = float(params.get("clip_hi", 2 * mu))
    arr = clip_list(g.normal(mu, sd, size=period), clip_lo, clip_hi).round().astype(int)
    return [int(x) for x in arr]


# ---------------------------------------------------------------------------
# RNG-based generator (no demand library needed)
# ---------------------------------------------------------------------------

def generate_instance_from_class(
    cls: dict,
    j: int = 0,
    warehouse_capacity: Optional[float] = None,
) -> dict:
    """
    Generate instance #j from a class config using pure RNG demands.

    Parameters
    ----------
    cls : dict
        Class config (from class_gen.LOCAL_PRESETS or class_gen.make_config).
    j : int
        Replica index. Each j gives a different random draw.
    warehouse_capacity : float | None
        Optional inventory capacity. None = unconstrained.
    """
    period = int(cls["period"])
    zero_head = int(cls.get("zero_head", 0))

    items: dict = {}
    total_by_t = np.zeros(period, dtype=float)

    for i in range(cls["n_items"]):
        seed_k = int(cls["seed_base"] + 1000 * j + 17 * i)

        D = list(
            np.random.default_rng(seed_k).integers(
                int(cls["dem_lo"]), int(cls["dem_hi"]) + 1, size=period
            )
        )
        if zero_head > 0:
            for z in range(min(zero_head, period)):
                D[z] = 0

        Mseq = list(
            np.random.default_rng(seed_k + 1).integers(
                int(cls["m_lo"]), int(cls["m_hi"]) + 1, size=period
            )
        )

        c_seq = make_series(cls["c_mode"], period, dict(cls["c_params"] or {}), seed_k + 2)
        h_seq = make_series(cls["h_mode"], period, dict(cls["h_params"] or {}), seed_k + 3)
        c_out = (
            float(cls["c_params"].get("value", 0.0))
            if cls["c_mode"] == "scalar"
            else [float(x) for x in c_seq]
        )
        h_out = (
            float(cls["h_params"].get("value", 0.0))
            if cls["h_mode"] == "scalar"
            else [float(x) for x in h_seq]
        )

        if cls["s_mode"] == "tbo":
            h_avg = float(h_out) if isinstance(h_out, float) else float(np.mean(h_out))
            d_avg = float(np.mean(D)) if D else 0.0
            L_raw = float((cls.get("s_params") or {}).get("L", 2.0))
            L = max(0.1, L_raw if math.isfinite(L_raw) else 2.0)
            jitter_pct = float(
                (cls.get("s_params") or {}).get("jitter_pct", SETUP_TBO_JITTER_DEFAULT)
            )
            per_period = bool((cls.get("s_params") or {}).get("per_period", True))
            base_s = 0.5 * h_avg * d_avg * (L ** 2)
            if base_s <= 0:
                base_s = 1e-6
            g_s = np.random.default_rng(seed_k + 4)
            if per_period:
                eps = g_s.uniform(-jitter_pct / 100.0, jitter_pct / 100.0, size=period)
                s_out = list((base_s * (1.0 + eps)).astype(float))
            else:
                jit = 1.0 + g_s.uniform(-jitter_pct / 100.0, jitter_pct / 100.0)
                s_out: float | list = float(base_s * jit)
        else:
            s_out = make_series(
                cls["s_mode"], period, dict(cls.get("s_params") or {}), seed_k + 4
            )

        items[str(i)] = {
            "demand": [int(x) for x in D],
            "setup": s_out,
            "c_var": c_out,
            "h": h_out,
            "b_var": 0.0,
            "shelf_seq": [int(x) for x in Mseq],
        }
        total_by_t += np.array(D, dtype=float)

    cap_mode = cls.get("cap_mode", "Uniform")
    if cap_mode == "DemandBased":
        beta = CAP_TIGHT_BETAS.get(cls.get("cap_tight") or "Medium", 1.02)
        mean_total = float(np.mean(total_by_t)) if period > 0 else 0.0
        base_cap = max(1, int(round(beta * mean_total)))
        jit_pct = float((cls.get("cap_params") or {}).get("jitter_pct", 3.0))
        if jit_pct > 0:
            g = np.random.default_rng(int(cls["seed_base"] + 31 * j + 7))
            noise = g.uniform(-jit_pct / 100.0, jit_pct / 100.0, size=period)
            cap = np.maximum(0, np.round(base_cap * (1.0 + noise)).astype(int)).tolist()
        else:
            cap = [base_cap] * period
    else:
        cap = generate_cap_series(
            period, cap_mode, cls.get("cap_params") or {}, cls["seed_base"] + 31 * j
        )

    return {
        "period": period,
        "items": items,
        "manual_capacity": [int(x) for x in cap],
        "warehouse_capacity": warehouse_capacity,
        "allow_unmet_demand": bool(cls.get("allow_unmet_demand", False)),
        "lost_sales_penalty_factor": float(cls.get("lost_sales_penalty_factor", 200.0)),
        "meta": {"origin": "class_rng", "class_key": cls["name"], "class_params": cls},
    }


# ---------------------------------------------------------------------------
# Lib-based generator (uses real demands from the demands_lib folder)
# ---------------------------------------------------------------------------

def generate_instance_from_lib(
    cls: dict,
    demand_lib_root: str | Path,
    j: int = 0,
    pick_mode: str = "first",
    warehouse_capacity: Optional[float] = None,
) -> dict:
    """
    Generate instance #j using real demands from the demand library.

    The demand library is populated by running:
        python parse_demands_and_insert.py --dataset-root ../540 --demand-lib-root ./demands_lib

    Parameters
    ----------
    cls : dict
        Class config (from class_gen.LOCAL_PRESETS or parse_demands_and_insert.make_config_cv).
    demand_lib_root : str | Path
        Path to the demands_lib folder.
    j : int
        Replica index. j>0 picks a random file from the bucket.
    pick_mode : str
        "first" (deterministic) or "random".
    warehouse_capacity : float | None
        Optional inventory capacity.
    """
    from parse_demands_and_insert import build_instance_from_class

    inst = build_instance_from_class(
        cls,
        Path(demand_lib_root),
        j=j,
        pick_mode=pick_mode,
    )
    inst["warehouse_capacity"] = warehouse_capacity
    inst["meta"]["origin"] = "class_lib"
    return inst


# ---------------------------------------------------------------------------
# Unified entry point (picks mode automatically)
# ---------------------------------------------------------------------------

def generate_instance(
    cls: dict,
    j: int = 0,
    demand_lib_root: Optional[str | Path] = None,
    pick_mode: str = "first",
    warehouse_capacity: Optional[float] = None,
) -> dict:
    """
    Generate one instance. Uses lib mode if demand_lib_root is given, else RNG mode.
    """
    if demand_lib_root:
        return generate_instance_from_lib(
            cls, demand_lib_root, j=j, pick_mode=pick_mode,
            warehouse_capacity=warehouse_capacity,
        )
    return generate_instance_from_class(cls, j=j, warehouse_capacity=warehouse_capacity)


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------

def _dumps_safe(obj) -> str:
    def _conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    return json.dumps(obj, default=_conv, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        from class_gen import LOCAL_PRESETS
    except ImportError:
        LOCAL_PRESETS = []

    p = argparse.ArgumentParser(
        description=(
            "Generate perishable lot-sizing instances.\n\n"
            "Lib mode  (real 540 demands): provide --demand-lib-root\n"
            "RNG mode  (synthetic demands): omit --demand-lib-root\n\n"
            "To build the demand library from the 540 folder first:\n"
            "  python parse_demands_and_insert.py \\\n"
            "      --dataset-root ../540 \\\n"
            "      --demand-lib-root ./demands_lib"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--preset", type=str, default=None,
        help="Class code to generate (e.g. X111112). Use --list-presets to see all.",
    )
    p.add_argument("--j", type=int, default=0, help="Replica index (default 0).")
    p.add_argument(
        "--out", type=str, default=None,
        help="Output JSON file (default: <preset>_j<j>.json).",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Generate ALL presets into --out-dir.",
    )
    p.add_argument(
        "--replicas", type=int, default=1,
        help="Number of replicas per preset for --all (default 1).",
    )
    p.add_argument(
        "--out-dir", type=str, default="instances",
        help="Output directory for --all (default: instances/).",
    )
    p.add_argument(
        "--demand-lib-root", type=str, default=None,
        metavar="PATH",
        help=(
            "Path to the demand library folder (lib mode). "
            "Build it first with parse_demands_and_insert.py --dataset-root ../540."
        ),
    )
    p.add_argument(
        "--pick", choices=["first", "random"], default="first",
        help="How to pick a demand file per bucket in lib mode (default: first).",
    )
    p.add_argument(
        "--warehouse-capacity", type=float, default=None,
        help="Optional warehouse/inventory capacity (default: unconstrained).",
    )
    p.add_argument(
        "--list-presets", action="store_true",
        help="Print all preset codes and exit.",
    )
    args = p.parse_args()

    if args.list_presets:
        if not LOCAL_PRESETS:
            print("No presets found. Ensure class_gen.py is in the same directory.")
            sys.exit(1)
        for cfg in LOCAL_PRESETS:
            print(cfg["name"])
        sys.exit(0)

    mode_label = "lib" if args.demand_lib_root else "rng"

    if args.all:
        if not LOCAL_PRESETS:
            print("No presets found. Ensure class_gen.py is in the same directory.")
            sys.exit(1)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        errors = 0
        for cls in LOCAL_PRESETS:
            for j in range(max(1, args.replicas)):
                try:
                    inst = generate_instance(
                        cls,
                        j=j,
                        demand_lib_root=args.demand_lib_root,
                        pick_mode=args.pick,
                        warehouse_capacity=args.warehouse_capacity,
                    )
                    fname = out_dir / f"{cls['name']}_j{j}.json"
                    fname.write_text(_dumps_safe(inst), encoding="utf-8")
                    count += 1
                except Exception as e:
                    print(f"[WARN] {cls['name']} j={j}: {e}", file=sys.stderr)
                    errors += 1
        print(f"Generated {count} instance files in {out_dir}/  (mode={mode_label}, errors={errors})")
        sys.exit(0)

    if args.preset:
        match = next((c for c in LOCAL_PRESETS if c["name"] == args.preset), None)
        if match is None:
            print(f"Preset '{args.preset}' not found. Run --list-presets to see all.")
            sys.exit(1)
        inst = generate_instance(
            match,
            j=args.j,
            demand_lib_root=args.demand_lib_root,
            pick_mode=args.pick,
            warehouse_capacity=args.warehouse_capacity,
        )
        out_path = args.out or f"{args.preset}_j{args.j}.json"
        Path(out_path).write_text(_dumps_safe(inst), encoding="utf-8")
        print(f"Written to {out_path}  (mode={mode_label})")
        sys.exit(0)

    p.print_help()


if __name__ == "__main__":
    main()
