#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse perishable lot-sizing files (items are COLUMNS), build a demand library,
synthesize any missing (CV,T,I) buckets from existing ones, create classes with
CV letters (L/H) while PRESERVING the capacity digit (C), and insert
classes + instances into Supabase.

X-code format used here (matches your legend, with CV letter):
    X A B C V E F  -> example: X132H412
      | | | | | |
      | | | | | └─ F: TBO parameter L (1..12 default set)
      | | | | └─── E: shelf-life group (1..5) → (m_lo, m_hi)
      | | | └───── V: demand CV (L=low, H=high)
      | | └─────── C: capacity tightness (1=Loose, 2=Medium)
      | └───────── B: #items (1→10, 2→20, 3→30)
      └─────────── A: periods (1→20, 2→30, 3→40)  (zero_head {20:2,30:3,40:4})

Demand library folder structure:
    <demand_lib_root>/<low|high>/<periods>/<items>/<original_or_synth>.txt
Each file stores an I×T integer matrix (rows=items, cols=periods).

CLI highlights:
  - Parse raw dataset into the library: --dataset-root <540-root> --demand-lib-root <lib>
  - Insert ONE example class+instance: --create-example  (name becomes X111L11)
  - Insert ALL combos (P∈1..3, N∈1..3, C∈{1,2}, V∈{L,H}, M∈1..5, L in list):
      --create-all --instances-per-class 5 --L-list 2,5,7,9,11,12 -y
  - No missing buckets: script synthesizes from 10×20 donors on the fly.

Default instances per class = 5 (as requested).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# ------------------ Supabase defaults (can be overridden) ------------------
DEFAULT_SUPABASE_URL = "https://btqqbsnjcsgjvgpuutiw.supabase.co"
DEFAULT_SUPABASE_ANON = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ0cXFic25q"
    "Y3NnanZncHV1dGl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU1NDMwODMsImV4cCI6MjA3MTExOTA4M30."
    "gissvSrKruPsYJOHOoLqfzQGLrB4oFVckVwhrUpGJXU"
)

try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None  # type: ignore
    Client = None  # type: ignore


# =============================================================================
# Parsing (items are columns; supports multi-page and extra-period waves)
# =============================================================================


def parse_demand_file(file_path: Path) -> Tuple[int, int, List[List[int]]]:
    """
    Parse a problem file and return (periods, items, demands[I][T]).

    - First non-empty line gives: <items> <periods>.
    - Demand appears as one or more numeric sections.
    - All sections with row count == <periods> are horizontally concatenated.
      If combined width exceeds <items>, extra columns are dropped.
      If short, pad zeros to reach <items>.
    - Resulting matrix is (periods × items) then transposed to (items × periods).
    """
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{file_path}: empty file")

    hdr = re.findall(r"-?\d+", lines[0])
    if len(hdr) < 2:
        raise ValueError(f"{file_path}: cannot read header line '{lines[0]}'")
    items = int(hdr[0])
    periods = int(hdr[1])

    numeric_groups: List[List[List[float]]] = []
    cur: List[List[float]] = []
    cur_len: Optional[int] = None
    for ln in lines[1:]:
        if re.search(r"[A-Za-z]", ln):
            if cur:
                numeric_groups.append(cur)
                cur = []
                cur_len = None
            continue
        nums = re.findall(r"-?\d+\.?\d*", ln)
        if not nums:
            if cur:
                numeric_groups.append(cur)
                cur = []
                cur_len = None
            continue
        row = [float(x) for x in nums]
        if cur and cur_len is not None and len(row) != cur_len:
            numeric_groups.append(cur)
            cur = []
            cur_len = None
        cur.append(row)
        cur_len = len(row)
    if cur:
        numeric_groups.append(cur)

    slices = [g for g in numeric_groups if len(g) == periods]
    if not slices:
        # Fallback: accept I rows × T cols and transpose
        alt = [g for g in numeric_groups if len(g) == items]
        if alt and len(alt[0][0]) == periods:
            matrix_TxI = [[alt[0][i][t] for i in range(items)] for t in range(periods)]
        else:
            raise ValueError(
                f"{file_path}: no usable demand blocks (expected {periods} rows)"
            )
    else:
        matrix_TxI: List[List[float]] = []
        for r in range(periods):
            row: List[float] = []
            for sec in slices:
                if len(row) >= items:
                    break
                need = items - len(row)
                row.extend(sec[r][:need])
            if len(row) < items:
                row.extend([0.0] * (items - len(row)))
            matrix_TxI.append(row)

    # transpose periods×items -> items×periods, cast to nonnegative ints
    demands: List[List[int]] = []
    for i in range(items):
        series = []
        for t in range(periods):
            v = matrix_TxI[t][i] if i < len(matrix_TxI[t]) else 0.0
            iv = max(0, int(round(v)))
            series.append(iv)
        demands.append(series)
    return periods, items, demands


# =============================================================================
# Demand library I/O
# =============================================================================


def cv_from_name(file_name: str) -> Optional[str]:
    """Infer CV from raw dataset name: 4th digit after X (1→low, 2→high)."""
    m = re.match(r"^[Xx](\d)(\d)(\d)(\d)", file_name)
    if not m:
        return None
    return "low" if m.group(4) == "1" else ("high" if m.group(4) == "2" else None)


def lib_bucket_dir(root: Path, cv: str, periods: int, items: int) -> Path:
    return root / cv.lower() / str(periods) / str(items)


def save_demand_matrix(
    root: Path, cv: str, periods: int, items: int, stem: str, demands: List[List[int]]
) -> Path:
    folder = lib_bucket_dir(root, cv, periods, items)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{stem}.txt"
    with dest.open("w", encoding="utf-8") as f:
        for row in demands:
            f.write(" ".join(str(x) for x in row) + "\n")
    return dest


def list_bucket_files(root: Path, cv: str, periods: int, items: int) -> List[Path]:
    folder = lib_bucket_dir(root, cv, periods, items)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix == ".txt")


def pick_demand_file(
    lib_root: Path, cv: str, periods: int, items: int, mode: str = "first"
) -> Optional[Path]:
    files = list_bucket_files(lib_root, cv, periods, items)
    if not files:
        return None
    if mode == "random":
        rng = np.random.default_rng(12345)
        return files[int(rng.integers(0, len(files)))]
    return files[0]


def load_demand_matrix_full(path: Path) -> List[List[int]]:
    out: List[List[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            vals = [int(float(x)) for x in ln.split()]
            out.append(vals)
    return out  # rows=items, cols=periods


def load_demand_matrix_crop(path: Path, n_items: int, periods: int) -> List[List[int]]:
    mat = load_demand_matrix_full(path)
    out: List[List[int]] = []
    for r in mat[:n_items]:
        vals = (r + [0] * periods)[:periods]
        out.append(vals)
    while len(out) < n_items:
        out.append([0] * periods)
    return out


def process_dataset(dataset_root: Path, demand_lib_root: Path) -> None:
    if not dataset_root or not dataset_root.is_dir():
        raise ValueError(f"dataset-root {dataset_root} is not a directory")
    count = 0
    for dirpath, _, filenames in os.walk(dataset_root):
        for fname in filenames:
            if not fname.lower().startswith("x"):
                continue
            src = Path(dirpath) / fname
            try:
                periods, items, demands = parse_demand_file(src)
            except Exception as e:
                print(f"[WARN] {src}: {e}")
                continue
            cv = cv_from_name(fname) or "low"
            dest = save_demand_matrix(
                demand_lib_root, cv, periods, items, Path(fname).stem, demands
            )
            print(f"Parsed {fname}: I={items}, T={periods} → {dest}")
            count += 1
    print(f"Done. Parsed {count} files.")


# =============================================================================
# Synthesis: fill missing (cv, T, I) by duplicating/concatenating/shuffling
# =============================================================================


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed or 123456789)


def _permute_rows(mat: List[List[int]], g: np.random.Generator) -> List[List[int]]:
    idx = list(range(len(mat)))
    g.shuffle(idx)
    return [mat[i] for i in idx]


def _circular_shift_row(row: List[int], k: int) -> List[int]:
    if not row:
        return []
    k %= len(row)
    return row[-k:] + row[:-k] if k else row[:]


def _append_time(mat: List[List[int]], extra_cols: List[List[int]]) -> List[List[int]]:
    return [r + c for r, c in zip(mat, extra_cols)]


def _sample_rows(
    mat: List[List[int]], count: int, g: np.random.Generator
) -> List[List[int]]:
    if count <= 0:
        return []
    I = len(mat)
    if I == 0:
        return [[0]] * count
    if count >= I:
        out = mat[:]
        while len(out) < count:
            out.extend(_permute_rows(mat, g))
        return out[:count]
    idx = np.arange(I)
    g.shuffle(idx)
    return [mat[i] for i in idx[:count]]


def _slice_time(mat: List[List[int]], start: int, length: int) -> List[List[int]]:
    return [row[start : start + length] for row in mat]


def _other_cv(cv: str) -> str:
    return "high" if cv.lower() == "low" else "low"


def synthesize_matrix_from_library(
    lib_root: Path,
    cv: str,
    target_T: int,
    target_I: int,
    *,
    donor_I_options: Tuple[int, ...] = (10, 20),
    seed: int = 0,
) -> List[List[int]]:
    """
    Create an (I×T) matrix by stitching donors from the library:
      - Build an I×20 base by sampling donors with T=20 and I in {10,20}
      - Extend time to target_T by appending contiguous windows from donors
      - Shuffle items and apply small circular time shifts for variety
    If no donors exist for the requested CV, borrow from the other CV.
    """
    g = _rng(seed or (target_T * 100 + target_I))

    # Collect 20-period donors for the requested CV
    donors_20: List[List[List[int]]] = []
    for I0 in donor_I_options:
        for p in list_bucket_files(lib_root, cv, 20, I0):
            donors_20.append(load_demand_matrix_full(p))

    # Fallback: borrow donors from the other CV bucket if needed
    if not donors_20:
        alt_cv = _other_cv(cv)
        for I0 in donor_I_options:
            for p in list_bucket_files(lib_root, alt_cv, 20, I0):
                donors_20.append(load_demand_matrix_full(p))

    if not donors_20:
        raise RuntimeError(
            f"No donors found in library for CV={cv} (or {_other_cv(cv)}) with T=20."
        )

    # 1) Build I×20 base by sampling rows across donors
    rows: List[List[int]] = []
    while len(rows) < target_I:
        d = donors_20[int(g.integers(0, len(donors_20)))]
        need = target_I - len(rows)
        take = _sample_rows(d, min(need, len(d)), g)
        rows.extend(take)
    if len(rows) > target_I:
        rows = _sample_rows(rows, target_I, g)
    rows = _permute_rows(rows, g)  # shuffle items

    base_T = 20
    if target_T == base_T:
        max_shift = max(1, base_T // 10)
        return [_circular_shift_row(r, int(g.integers(0, max_shift + 1))) for r in rows]

    # 2) Extend time horizon
    cur = [r[:] for r in rows]
    remaining = target_T - base_T
    while remaining > 0:
        w = 20 if remaining >= 20 else remaining
        d = donors_20[int(g.integers(0, len(donors_20)))]
        d_rows = _sample_rows(d, target_I, g)
        start = int(g.integers(0, 20 - w + 1))
        d_slice = _slice_time(d_rows, start, w)
        max_shift = max(0, w // 6)
        if max_shift > 0:
            d_slice = [
                _circular_shift_row(r, int(g.integers(0, max_shift + 1)))
                for r in d_slice
            ]
        cur = _append_time(cur, d_slice)
        remaining -= w

    # Mild global circular shift
    Ttot = target_T
    max_shift_all = max(1, Ttot // 12)
    cur = [_circular_shift_row(r, int(g.integers(0, max_shift_all + 1))) for r in cur]
    return cur


def ensure_bucket_with_synthesis(
    lib_root: Path,
    cv: str,
    T: int,
    I: int,
    *,
    synth_variants: int = 3,
    seed: int = 0,
) -> List[Path]:
    """
    Ensure at least one file exists for (cv, T, I). If none, synthesize `synth_variants`
    different matrices and save them. Returns the list of available files after.
    """
    existing = list_bucket_files(lib_root, cv, T, I)
    if existing:
        return existing

    folder = lib_bucket_dir(lib_root, cv, T, I)
    folder.mkdir(parents=True, exist_ok=True)
    made: List[Path] = []
    for k in range(synth_variants):
        mat = synthesize_matrix_from_library(
            lib_root, cv, T, I, seed=(seed + k + T * 1000 + I * 17)
        )
        fname = f"synth_{cv}_T{T}_I{I}_{k+1}.txt"
        path = save_demand_matrix(lib_root, cv, T, I, fname[:-4], mat)
        made.append(path)
    return made


# =============================================================================
# Class + instance generation (CV uses L/H letter; NAME includes C!)
# =============================================================================

PERIOD_MAP: Dict[int, int] = {1: 20, 2: 30, 3: 40}
NITEMS_MAP: Dict[int, int] = {1: 10, 2: 20, 3: 30}
CAP_TIGHT_MAP: Dict[int, str] = {1: "Loose", 2: "Medium"}
SHELF_MAP: Dict[int, Tuple[int, int]] = {
    1: (1, 10),
    2: (5, 15),
    3: (10, 20),
    4: (5, 25),
    5: (10, 30),
}
ZERO_HEAD_BY_T: Dict[int, int] = {20: 0, 30: 1, 40: 2}


def seed_from_code(code: str) -> int:
    import hashlib

    h = int(hashlib.md5(code.encode("utf-8")).hexdigest()[:8], 16)
    return 10000 + (h % 80000)


def make_config_cv(
    P: int,
    N: int,
    C: int,
    cv_letter: str,
    M: int,
    L: int,
    *,
    demand_lib_root: Optional[str] = None,
    zero_head_map: Optional[Dict[int, int]] = None,
) -> Dict:
    """
    Build a class spec. NAME includes capacity digit C to avoid collisions:
        name = X{P}{N}{C}{V}{M}{L}  where V is 'L' or 'H'
    """
    cv = "low" if cv_letter.lower().startswith("l") else "high"
    T = PERIOD_MAP[P]
    n_items = NITEMS_MAP[N]
    cap_tight = CAP_TIGHT_MAP[C]
    m_lo, m_hi = SHELF_MAP[M]
    zero_head = (zero_head_map or ZERO_HEAD_BY_T).get(T, 0)
    # >>> FIXED HERE: include C in the code string <<<
    name = f"X{P}{N}{C}{cv_letter.upper()}{M}{L}"

    return {
        "name": name,
        "period": T,
        "cap_mode": "DemandBased",
        "cap_params": {"jitter_pct": 10.0},
        "cap_tight": cap_tight,
        "n_items": n_items,
        "dem_lo": 0,
        "dem_hi": 0,
        "m_lo": m_lo,
        "m_hi": m_hi,
        "c_mode": "uniform",
        "c_params": {"hi": 4.0, "lo": 2.0},
        "h_mode": "uniform",
        "h_params": {"hi": 1.0, "lo": 0.3},
        "s_mode": "tbo",
        "s_params": {"L": float(L), "jitter_pct": 10.0, "per_period": True},
        "zero_head": int(zero_head),
        "batch_size": 1,
        "seed_base": seed_from_code(name),
        "allow_unmet_demand": False,
        "lost_sales_penalty_factor": 200.0,
        "dem_source": {
            "type": "files",
            "cv": cv,
            **({"root": str(demand_lib_root)} if demand_lib_root else {}),
        },
    }


def _series(mode: str, T: int, params: Dict, seed: int) -> List[float]:
    g = np.random.default_rng(seed)
    if mode == "scalar":
        return [float(params.get("value", 0.0))] * T
    if mode == "uniform":
        lo = float(params.get("lo", 0.0))
        hi = float(params.get("hi", 1.0))
        return list(g.uniform(lo, hi, size=T))
    if mode == "normal":
        mu = float(params.get("mean", 0.0))
        sd = float(params.get("std", 1.0))
        arr = g.normal(mu, sd, size=T)
        lo = params.get("clip_lo")
        hi = params.get("clip_hi")
        if lo is not None or hi is not None:
            arr = np.clip(
                arr, lo if lo is not None else -np.inf, hi if hi is not None else np.inf
            )
        return list(arr)
    if mode == "linear":
        base = float(params.get("base", 0.0))
        slope = float(params.get("slope", 0.0))
        noise = float(params.get("noise_std", 0.0))
        arr = base + slope * np.arange(T) + g.normal(0.0, noise, size=T)
        lo = params.get("clip_lo")
        hi = params.get("clip_hi")
        if lo is not None or hi is not None:
            arr = np.clip(
                arr, lo if lo is not None else -np.inf, hi if hi is not None else np.inf
            )
        return list(arr)
    if mode == "seasonal":
        base = float(params.get("base", 1.0))
        amp = float(params.get("amp", 0.0))
        per = float(params.get("period", max(1, T)))
        phase = float(params.get("phase", 0.0))
        noise = float(params.get("noise_std", 0.0))
        t = np.arange(T)
        arr = base * (1 + amp * np.sin(2 * np.pi * (t + phase) / per)) + g.normal(
            0.0, noise, size=T
        )
        lo = params.get("clip_lo")
        hi = params.get("clip_hi")
        if lo is not None or hi is not None:
            arr = np.clip(
                arr, lo if lo is not None else -np.inf, hi if hi is not None else np.inf
            )
        return list(arr)
    return [float(params.get("value", 0.0))] * T


def build_instance_from_class(
    cls: Dict, demand_lib_root: Path, *, j: int = 0, pick_mode: str = "first"
) -> Dict:
    T = int(cls["period"])
    I = int(cls["n_items"])
    zero_head = int(cls.get("zero_head", 0))
    dem_source = cls.get("dem_source") or {}
    cv = dem_source.get("cv", "low")
    lib_root = Path(dem_source.get("root") or demand_lib_root)

    # Ensure bucket exists (synthesize if missing), then pick a file
    ensure_bucket_with_synthesis(
        lib_root, cv, T, I, synth_variants=3, seed=1000 + T + I
    )
    pick_mode_eff = "random" if j > 0 else pick_mode
    file_path = pick_demand_file(lib_root, cv, T, I, mode=pick_mode_eff)
    file_demands = load_demand_matrix_crop(file_path, I, T) if file_path else None

    items: Dict[str, Dict] = {}
    total_by_t = np.zeros(T, dtype=float)

    for i in range(I):
        seed_k = int(cls.get("seed_base", 0) + 1000 * j + 17 * i)
        D = list(file_demands[i]) if file_demands else [0] * T
        if zero_head > 0:
            for z in range(min(zero_head, T)):
                D[z] = 0
        m_lo = int(cls.get("m_lo", 1))
        m_hi = int(cls.get("m_hi", m_lo))
        Mseq = (
            [m_lo] * T
            if m_lo == m_hi
            else list(
                np.random.default_rng(seed_k + 1).integers(m_lo, m_hi + 1, size=T)
            )
        )
        c_seq = _series(
            cls.get("c_mode", "scalar"), T, dict(cls.get("c_params", {})), seed_k + 2
        )
        h_seq = _series(
            cls.get("h_mode", "scalar"), T, dict(cls.get("h_params", {})), seed_k + 3
        )
        c_out = (
            float(cls.get("c_params", {}).get("value", 0.0))
            if cls.get("c_mode") == "scalar"
            else [float(x) for x in c_seq]
        )
        h_out = (
            float(cls.get("h_params", {}).get("value", 0.0))
            if cls.get("h_mode") == "scalar"
            else [float(x) for x in h_seq]
        )
        # TBO setup
        Lval = max(0.1, float(cls.get("s_params", {}).get("L", 2.0)))
        jitter = float(cls.get("s_params", {}).get("jitter_pct", 10.0))
        h_avg = float(h_out) if isinstance(h_out, float) else float(np.mean(h_out))
        d_avg = float(np.mean(D)) if D else 0.0
        base_s = 0.5 * h_avg * d_avg * (Lval**2)
        if base_s <= 0:
            base_s = 1e-6
        eps = np.random.default_rng(seed_k + 4).uniform(
            -jitter / 100.0, jitter / 100.0, size=T
        )
        s_out = list((base_s * (1.0 + eps)).astype(float))

        items[str(i)] = {
            "demand": [int(x) for x in D],
            "setup": s_out,
            "c_var": c_out,
            "h": h_out,
            "b_var": 0.0,
            "shelf_seq": [int(x) for x in Mseq],
        }
        total_by_t += np.array(D, dtype=float)

    betas = {
        "Loose": 1.10,
        "Medium": 1.02,
        "Tight": 0.96,
        "Tighter": 0.90,
        "Ultra": 0.80,
    }
    beta = betas.get(cls.get("cap_tight", "Medium"), 1.02)
    base_cap = max(1, int(round(beta * float(np.mean(total_by_t))))) if T > 0 else 1
    jit_pct = float(cls.get("cap_params", {}).get("jitter_pct", 3.0))
    g = np.random.default_rng(int(cls.get("seed_base", 0) + 31 * j + 7))
    noise = g.uniform(-jit_pct / 100.0, jit_pct / 100.0, size=T)
    cap = np.maximum(0, np.round(base_cap * (1.0 + noise)).astype(int)).tolist()

    return {
        "period": T,
        "items": items,
        "manual_capacity": [int(x) for x in cap],
        "warehouse_capacity": None,
        "allow_unmet_demand": bool(cls.get("allow_unmet_demand", False)),
        "lost_sales_penalty_factor": float(cls.get("lost_sales_penalty_factor", 200.0)),
        "meta": {"origin": "class", "class_key": cls["name"], "class_params": cls},
    }


# =============================================================================
# Supabase helpers
# =============================================================================


def ensure_class_row(sb: Client, cls: Dict) -> str:
    spec = json.loads(json.dumps(cls))
    name = spec["name"]
    sel = sb.table("classes").select("id").eq("name", name).limit(1).execute()
    if sel.data:
        cid = sel.data[0]["id"]
        sb.table("classes").update({"spec": spec}).eq("id", cid).execute()
        return cid
    ins = sb.table("classes").insert({"name": name, "spec": spec}).execute()
    return ins.data[0]["id"]


def insert_instance(sb: Client, instance: Dict, class_id: Optional[str] = None) -> str:
    payload = {
        "period": int(instance["period"]),
        "manual_capacity": instance.get("manual_capacity"),
        "warehouse_capacity": instance.get("warehouse_capacity"),
        "data": instance,
    }
    if class_id:
        payload["class_id"] = class_id
    ins = sb.table("instances").insert(payload).execute()
    return ins.data[0]["id"]


# =============================================================================
# Bulk generation helpers (all combinations)
# =============================================================================


def generate_all_class_specs(
    demand_lib_root: Path, L_values: List[int], *, zero_head: bool = True
) -> List[Dict]:
    specs: List[Dict] = []
    zh_map = ZERO_HEAD_BY_T if zero_head else {20: 0, 30: 0, 40: 0}
    for P in (1, 2, 3):
        for N in (1, 2, 3):
            for C in (1, 2):
                for cv_letter in ("L", "H"):
                    for M in (1, 2, 3, 4, 5):
                        for L in L_values:
                            specs.append(
                                make_config_cv(
                                    P,
                                    N,
                                    C,
                                    cv_letter,
                                    M,
                                    L,
                                    demand_lib_root=str(demand_lib_root),
                                    zero_head_map=zh_map,
                                )
                            )
    return specs


# =============================================================================
# CLI
# =============================================================================


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Parse demands (items=columns), synthesize missing buckets, build classes (L/H with capacity digit), insert into Supabase."
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=False,
        help="Folder with raw files (e.g., 540-1/540-2/540-3). If omitted, no parsing is done.",
    )
    p.add_argument(
        "--demand-lib-root",
        type=Path,
        required=True,
        help="Where parsed/synthesized matrices are stored: <root>/<cv>/<periods>/<items>/*.txt",
    )
    p.add_argument(
        "--create-example",
        action="store_true",
        help="Create X111L11 (P=1,N=1,C=1,V=L,M=1,L=1), build one instance, and insert after confirmation.",
    )
    p.add_argument(
        "--create-all",
        action="store_true",
        help="Generate ALL combinations (P:1..3, N:1..3, C:{1,2}, V:{L,H}, M:1..5, L: from --L-list).",
    )
    p.add_argument(
        "--instances-per-class",
        type=int,
        default=5,  # default 5 as requested
        help="Instances per class for --create-all (default 5).",
    )
    p.add_argument(
        "--L-list",
        type=str,
        default="2,5,7,9,11,12",
        help="Comma-separated TBO L values to use for --create-all (default '2,5,7,9,11,12').",
    )
    p.add_argument(
        "--pick",
        choices=["first", "random"],
        default="first",
        help="How to pick a demand file for a (cv,T,I) bucket (default: first).",
    )
    p.add_argument(
        "--no-zero-head",
        action="store_true",
        help="Disable zero_head-by-T (defaults are {20:2,30:3,40:4}).",
    )
    p.add_argument(
        "--synth-variants-per-bucket",
        type=int,
        default=3,
        help="If a (cv,T,I) bucket is missing, create this many synthetic variants (default 3).",
    )
    p.add_argument(
        "--supabase-url",
        type=str,
        default=os.environ.get("SUPABASE_URL", DEFAULT_SUPABASE_URL),
    )
    p.add_argument(
        "--supabase-key",
        type=str,
        default=os.environ.get("SUPABASE_ANON_KEY", DEFAULT_SUPABASE_ANON),
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts and insert directly.",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    # 1) Parse raw dataset → demand library
    if args.dataset_root:
        if not args.dataset_root.is_dir():
            print(f"[ERROR] dataset-root not found: {args.dataset_root}")
            return 1
        print(f"Processing dataset at {args.dataset_root} ...")
        process_dataset(args.dataset_root, args.demand_lib_root)

    # If nothing else requested, stop here
    if not (args.create_example or args.create_all):
        return 0

    # Supabase availability
    if create_client is None or Client is None:
        print(
            "Supabase client not installed. Install with `pip install supabase` and retry."
        )
        return 1
    if not args.supabase_url or not args.supabase_key:
        print(
            "Supabase URL/key missing. Provide via CLI or SUPABASE_URL / SUPABASE_ANON_KEY."
        )
        return 1

    sb = create_client(args.supabase_url, args.supabase_key)

    # 2a) Example path (ensures bucket via synthesis too)
    if args.create_example:
        cls = make_config_cv(
            P=1,
            N=1,
            C=1,
            cv_letter="L",
            M=1,
            L=1,
            demand_lib_root=str(args.demand_lib_root),
            zero_head_map=(
                ZERO_HEAD_BY_T if not args.no_zero_head else {20: 0, 30: 0, 40: 0}
            ),
        )
        T, I, cv = cls["period"], cls["n_items"], cls["dem_source"]["cv"]
        ensure_bucket_with_synthesis(
            args.demand_lib_root,
            cv,
            T,
            I,
            synth_variants=args.synth_variants_per_bucket,
            seed=4242,
        )
        inst = build_instance_from_class(
            cls, args.demand_lib_root, j=0, pick_mode=args.pick
        )

        print("\n--- Class preview ---")
        print(json.dumps(cls, indent=2))
        print("\n--- Instance preview (first item only) ---")
        first_key = sorted(inst["items"].keys(), key=int)[0]
        preview = {k: v for k, v in inst.items() if k != "items"}
        preview["items"] = {first_key: inst["items"][first_key]}
        print(json.dumps(preview, indent=2)[:1200])

        if not args.yes:
            try:
                resp = (
                    input("\nInsert this class and one instance into Supabase? [y/N]: ")
                    .strip()
                    .lower()
                )
            except EOFError:
                resp = ""
            if resp != "y":
                print("Aborted.")
                return 0

        class_id = ensure_class_row(sb, cls)
        inst_id = insert_instance(sb, inst, class_id=class_id)
        print(f"\n✅ Inserted class {class_id} and instance {inst_id}.")
        return 0

    # 2b) All combinations (no missing — synth where needed)
    L_vals = [int(x) for x in args.L_list.split(",") if x.strip()]
    specs = generate_all_class_specs(
        args.demand_lib_root, L_vals, zero_head=(not args.no_zero_head)
    )

    # Prime library: ensure every (cv,T,I) bucket exists
    touched = set()
    for spec in specs:
        T = spec["period"]
        I = spec["n_items"]
        cv = spec["dem_source"]["cv"]
        key = (cv, T, I)
        if key in touched:
            continue
        ensure_bucket_with_synthesis(
            args.demand_lib_root,
            cv,
            T,
            I,
            synth_variants=args.synth_variants_per_bucket,
            seed=2025 + T * 100 + I,
        )
        touched.add(key)

    total_instances = len(specs) * max(1, args.instances_per_class)
    if not args.yes:
        try:
            resp = (
                input(
                    f"Insert {len(specs)} classes and {total_instances} instances (no missing; synthesized where needed)? [y/N]: "
                )
                .strip()
                .lower()
            )
        except EOFError:
            resp = ""
        if resp != "y":
            print("Aborted.")
            return 0

    ins_classes = 0
    ins_instances = 0
    for cls in specs:
        cid = ensure_class_row(sb, cls)
        ins_classes += 1
        for j in range(max(1, args.instances_per_class)):
            inst = build_instance_from_class(
                cls, args.demand_lib_root, j=j, pick_mode=args.pick
            )
            insert_instance(sb, inst, class_id=cid)
            ins_instances += 1
        if ins_classes % 50 == 0:
            print(
                f"... inserted {ins_classes} classes / {ins_instances} instances so far"
            )

    print(f"\n✅ Done. Inserted {ins_classes} classes and {ins_instances} instances.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
