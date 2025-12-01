import os, json, time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client, Client
import plotly.express as px
import plotly.graph_objects as go
import re

# from mip.solver_mip_lefo import solve_instance
import math

# register alternative solver backends that share the same I/O

try:
    from mip.solver_mip_no_cross_no_shelf import (
        solve_instance as _solve_nocross_no_shelf_v1,
    )
except Exception:
    _solve_nocross_no_shelf_v1 = None  # optional backend

try:
    from mip.solver_mip_lefo import solve_instance as _solve_lefo_v2
except Exception:
    _solve_lefo_v2 = None  # optional backend

try:
    from mip.solver_mip_no_cross import solve_instance as _solve_nocross_v1
except Exception:
    _solve_nocross_v1 = None  # optional backend

try:
    from mip.solver_bnp import solve_instance as _solve_bnp
except Exception:
    _solve_bnp = None  # optional backend

try:
    from mip. import SOLVER_VERSION as SOLVER_VERSION_LEFO_V2

SOLVER_REGISTRY = {
    "No-Crossing (no shelf) v1": {
        "fn": _solve_nocross_no_shelf_v1,
        "tag": "lefo_mip_no_shelf_v1",  # the DB signature you wanted
        "desc": "No-crossing model, ignore cap(v1).",
    },
    "LEFO v2 (permission-based)": {
        "fn": _solve_lefo_v2,
        "tag": "lefo_mip_v2",
        "desc": "LEFO permission-based model (v2).",
    },
    "No-Crossing v1": {
        "fn": _solve_nocross_v1,
        "tag": "lefo_mip_v1",
        "desc": "No-crossing model (v1).",
    },
    "BNP": {
        "fn": _solve_bnp,
        "tag": "bnp_v1",
        "desc": "BNP (v1).",
    },
}

HOUR_GAP_FOR_BATCH = 1  # hours gap in created_at to start a new batch

# ==== X-code helpers (shared) ====
_x_pat = re.compile(r"^[Xx]([A-Za-z0-9]+)$")


def _x_digits(name: str):
    m = _x_pat.match(str(name))
    if not m:
        return None
    s = m.group(1)
    if len(s) < 6:
        return None
    A, B, C, D, E = s[0], s[1], s[2], s[3], s[4]
    F = s[5:]
    if A not in {"1", "2", "3"}:
        return None
    if B not in {"1", "2", "3"}:
        return None
    if C not in {"1", "2"}:
        return None
    if D not in {"1", "2", "L", "H"}:
        return None
    if E not in {"1", "2", "3", "4", "5"}:
        return None
    if not (F.isdigit() and 1 <= int(F) <= 12):
        return None
    return (A, B, C, D, E, F)


def _matches_x_filters(
    name: str, inc: dict[str, set[str]], exc: dict[str, set[str]]
) -> bool:
    tup = _x_digits(name)
    if not tup:
        return False
    POS = ["A", "B", "C", "D", "E", "F"]
    for pos_idx, p in enumerate(POS):
        d = tup[pos_idx]
        if inc.get(p) and d not in inc[p]:
            return False
        if d in exc.get(p, set()):
            return False
    return True


def parse_orders_lines(orders_txt):
    """
    Parse solver order lines robustly. Handles lines like:
      "Item 3", "Item i=3", "Item(3)" and t lines like "10 → 5", "t=10 → 5", "u=10 -> 5"
    Accepts separators: "→", "->", "=>".
    Returns list of dicts: {"item_id": int, "t": int, "qty": float}
    """
    rows, cur_item = [], None
    for ln in (ln.strip() for ln in orders_txt if ln and ln.strip()):
        low = ln.lower()
        if low.startswith("item"):
            m = re.search(r"(-?\d+)", ln)  # first integer anywhere on the line
            cur_item = int(m.group(1)) if m else None
            continue

        # find an arrow-like separator
        if "→" in ln:
            left, right = ln.split("→", 1)
        elif "->" in ln:
            left, right = ln.split("->", 1)
        elif "=>" in ln:
            left, right = ln.split("=>", 1)
        else:
            continue  # not an order line

        if cur_item is None:
            continue  # no current item context yet

        mt = re.search(r"(-?\d+)", left)  # period index anywhere on the left
        mq = re.search(
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", right
        )  # first number on the right
        if not (mt and mq):
            continue

        t = int(mt.group(1))
        qty = float(mq.group(1))
        rows.append({"item_id": cur_item, "t": t, "qty": _safe_float(qty)})
    return rows


def get_solver_backend():
    """Return (fn, tag). Falls back to LEFO v2 if a backend is missing."""
    label = st.session_state.get("solver_backend_label") or "LEFO v2 (permission-based)"
    entry = SOLVER_REGISTRY.get(label) or SOLVER_REGISTRY["LEFO v2 (permission-based)"]
    fn = entry["fn"] or SOLVER_REGISTRY["LEFO v2 (permission-based)"]["fn"]
    tag = (
        entry["tag"]
        if entry["fn"]
        else SOLVER_REGISTRY["LEFO v2 (permission-based)"]["tag"]
    )
    return fn, tag


def _safe_float(x):
    """Return a finite float or None."""
    try:
        f = float(x)
    except Exception:
        return None
    return f if math.isfinite(f) else None


def sanitize_json(obj):
    """
    Recursively replace NaN/±Inf with None and convert numpy scalars/arrays.
    Safe for any dict/list you are about to send to Supabase.
    """
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


# ======================= App Config =======================
DEFAULT_URL = "https://btqqbsnjcsgjvgpuutiw.supabase.co"
DEFAULT_ANON = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ0cXFic25qY3NnanZncHV1dGl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU1NDMwODMsImV4cCI6MjA3MTExOTA4M30.gissvSrKruPsYJOHOoLqfzQGLrB4oFVckVwhrUpGJXU"

st.set_page_config(page_title="Perishable Lot-Sizing (LEFO MIP)", layout="wide")
st.title("Perishable Lot-Sizing — Generator • Classes • Batches • Visualizer")

# ----------------------------------------------------------------------------
CAP_TIGHT_BETAS = {
    "Loose": 1.10,
    "Medium": 1.02,
    "Tight": 0.96,
    "Tighter": 0.90,
    "Ultra": 0.80,
}  # demand-based cap
DEFAULT_TBO_CHOICES = [1, 2, 4]  # target TBO set
SETUP_TBO_JITTER_DEFAULT = 10.0  # +/- percent


# ======================= Helpers ==========================
def rng(seed: int):
    return np.random.default_rng(int(seed))


def clip_list(xs, lo=None, hi=None):
    if lo is not None:
        xs = np.maximum(xs, lo)
    if hi is not None:
        xs = np.minimum(xs, hi)
    return xs


def make_series(mode: str, T: int, params: dict, seed: int) -> List[float]:
    """
    Returns list[float] of length T given mode in {"scalar","uniform","normal","linear","seasonal","manual"}.
    If "scalar", returns T copies; caller may store the scalar itself if preferred.
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
            raise ValueError("Manual list length must equal T.")
        return [float(x) for x in lst]
    raise ValueError(f"Unknown mode: {mode}")


def parse_manual_list(raw: str, T: int) -> List[float]:
    raw = (raw or "").strip()
    if not raw:
        return []
    parts = raw.replace(",", " ").split()
    vals = [float(x) for x in parts]
    if len(vals) != T:
        raise ValueError(f"Manual list length {len(vals)} != T={T}")
    return vals


def supabase_client() -> Client | None:
    if not st.session_state.get("use_supabase"):
        return None
    url = (
        st.session_state.get("supabase_url") or os.getenv("SUPABASE_URL") or DEFAULT_URL
    )
    key = (
        st.session_state.get("supabase_key")
        or os.getenv("SUPABASE_ANON_KEY")
        or DEFAULT_ANON
    )
    try:
        return create_client(url, key)
    except Exception as e:
        st.warning(f"Supabase client not created: {e}")
        return None


# --- Safe JSON dumper that converts NumPy types ---
def dumps_safe(obj) -> str:
    def _conv(o):
        import numpy as _np

        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    return json.dumps(obj, default=_conv, indent=2)


def fetch_classes(sb: Client | None) -> list[dict]:
    if sb is None:
        return []
    try:
        return (
            sb.table("classes")
            .select("id,name,spec,created_at")
            .order("created_at", desc=False)
            .execute()
            .data
        )
    except Exception as e:
        st.warning(f"Could not load classes: {e}")
        return []


# --- helpers for json/supabase ---


def fetch_runs_keyset_filtered(
    sb: Client,
    n_wanted: int,
    class_id_or_flag: str | None,
    only_optimal: bool,
    hide_infeasible: bool,
    hide_interrupted: bool,
    page_size: int = 500,  # small pages avoid timeouts
) -> list[dict]:
    """
    Pull newest runs using keyset pagination on created_at DESC.
    Applies simple server-side filters only (eq/neq), everything else can be client-side.
    """
    cols = (
        "id,created_at,instance_id,class_id,status,"
        "objective,best_bound,gap,runtime_sec,solver_version"
    )

    out: list[dict] = []
    last_seen_created: str | None = None

    while len(out) < int(n_wanted):
        need = min(page_size, int(n_wanted) - len(out))
        q = sb.table("runs").select(cols).order("created_at", desc=True).limit(need)

        if last_seen_created:
            q = q.lt("created_at", last_seen_created)

        # class filter
        if class_id_or_flag == "__ADHOC__":
            q = q.is_("class_id", "null")
        elif class_id_or_flag and class_id_or_flag != "__ALL__":
            q = q.eq("class_id", class_id_or_flag)

        # status filters
        if only_optimal:
            q = q.eq("status", 2)  # OPTIMAL
        else:
            if hide_infeasible:
                q = q.neq("status", 3)  # INFEASIBLE
            if hide_interrupted:
                q = q.neq("status", 11)  # INTERRUPTED

        batch = (q.execute().data) or []
        if not batch:
            break

        out.extend(batch)
        last_seen_created = batch[-1]["created_at"]

    return out


def fetch_instances_meta_map(sb, ids, cols="id,ins_id", chunk=50):
    """
    Return {instance_id: row} using tiny IN() chunks to avoid 414.
    If a chunk still fails (e.g. Cloudflare 414), fall back to per-ID selects.
    """
    out = {}
    if not ids:
        return out

    CH = max(10, int(chunk))  # keep small (10–50)
    for k in range(0, len(ids), CH):
        sub = ids[k : k + CH]
        rows = []
        try:
            rows = (
                sb.table("instances_enriched")
                .select(cols)
                .in_("id", sub)
                .execute()
                .data
                or []
            )
        except Exception:
            # Fallback: fetch one-by-one for this small slice
            for iid in sub:
                try:
                    r = (
                        sb.table("instances_enriched")
                        .select(cols)
                        .eq("id", iid)
                        .limit(1)
                        .execute()
                        .data
                        or []
                    )
                    if r:
                        rows.append(r[0])
                except Exception:
                    pass

        for r in rows:
            out[r["id"]] = r
    return out


def to_py(o):
    """Recursively convert NumPy scalars/arrays to plain Python types."""
    if isinstance(o, dict):
        return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(x) for x in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def ensure_class_row(sb: Client | None, cls: dict) -> str | None:
    """
    Upsert one row in 'classes' by name. Returns class_id (uuid) or None.
    """
    if sb is None:
        return None
    payload = {"name": cls["name"], "spec": to_py(cls)}
    try:
        sel = (
            sb.table("classes")
            .select("id")
            .eq("name", payload["name"])
            .limit(1)
            .execute()
        )
        if sel.data:
            cid = sel.data[0]["id"]
            sb.table("classes").update({"spec": payload["spec"]}).eq(
                "id", cid
            ).execute()
            return cid
        ins = sb.table("classes").insert(payload).execute()
        return ins.data[0]["id"]
    except Exception as e:
        st.warning(f"Could not save class '{payload['name']}': {e}")
        return None


def dedupe_queue_by_name():
    seen, out = set(), []
    for c in st.session_state.get("classes", []):
        nm = c.get("name")
        if nm not in seen:
            out.append(c)
            seen.add(nm)
    st.session_state["classes"] = out


def _ins_label(v):
    try:
        return f"ins#{int(v)}"
    except Exception:
        return "—"


# ======================= Sidebar: global ======================
with st.sidebar:
    st.header("Global Settings")
    T = st.number_input(
        "Periods T",
        min_value=1,
        value=60,
        step=1,
        help="Default periods for ad-hoc Items tab and Classes.",
    )
    SEED = st.number_input("Global random seed", min_value=0, value=0, step=1)

    st.markdown("---")
    st.subheader("Default Demand (Items tab only)")
    dem_lo = st.number_input("Demand min", min_value=0, value=5, step=1)
    dem_hi = st.number_input("Demand max", min_value=1, value=80, step=1)

    st.markdown("---")
    st.subheader("Warehouse capacity W")
    W_txt = st.text_input("W (blank = None)", value="")

    st.markdown("---")
    st.subheader("Solver")

    # global time/gap stay the same
    time_limit = st.number_input("Time limit (sec)", min_value=0, value=0, step=10)
    mip_gap = st.number_input(
        "MIPGap (0.0 = default)", min_value=0.0, value=0.0, step=0.01, format="%.4f"
    )

    # new: backend picker
    solver_labels = list(SOLVER_REGISTRY.keys())
    default_idx = (
        solver_labels.index("LEFO v2 (permission-based)")
        if "LEFO v2 (permission-based)" in solver_labels
        else 0
    )
    solver_label = st.selectbox(
        "Solver backend",
        solver_labels,
        index=default_idx,
        help="Which solver .py to use for all runs.",
    )
    st.session_state["solver_backend_label"] = solver_label

    # small heads-up if a backend couldn't be imported
    if SOLVER_REGISTRY[solver_label]["fn"] is None:
        st.warning(
            f"Backend '{solver_label}' is not available (module not importable). Falling back to LEFO v2."
        )

    allow_unmet_default = st.checkbox("Allow unmet demand (lost sales)", value=False)
    ls_penalty_factor_default = st.number_input(
        "Lost-sales penalty factor (× mean demand)",
        min_value=10.0,
        value=200.0,
        step=10.0,
        format="%.1f",
    )

    st.markdown("---")
    st.subheader("Supabase (optional)")
    st.session_state["use_supabase"] = st.checkbox("Log to Supabase", value=True)
    st.session_state["supabase_url"] = st.text_input("Supabase URL", value=DEFAULT_URL)
    st.session_state["supabase_key"] = st.text_input(
        "Supabase anon key", value=DEFAULT_ANON, type="password"
    )


# ===== CLASS SOURCE OF TRUTH: Supabase + optional local presets =====
st.session_state.setdefault("classes", [])  # queue to run
st.session_state.setdefault("db_classes", {})  # classes from DB
st.session_state.setdefault("local_presets", {})  # small, in-code stash

# 1) Load from Supabase (source of truth)
_sb_boot = supabase_client()
_rows = fetch_classes(_sb_boot)
st.session_state["db_classes"] = {r["name"]: r for r in _rows}  # name -> row

# 2) (Optional) keep a FEW local presets here for convenience.
#    They are NOT auto-queued and NOT auto-saved; you'll pick them in the UI.

LOCAL_PRESETS = [
    {
        "name": "X111112",
        "period": 20,
        "cap_mode": "DemandBased",
        "cap_params": {"jitter_pct": 10.0},
        "cap_tight": "Loose",
        "n_items": 10,
        "dem_lo": 0,
        "dem_hi": 125,
        "m_lo": 1,
        "m_hi": 10,
        "c_mode": "uniform",
        "c_params": {"hi": 4.0, "lo": 2.0},
        "h_mode": "uniform",
        "h_params": {"hi": 1.0, "lo": 0.3},
        "s_mode": "tbo",
        "s_params": {"L": 2.0, "jitter_pct": 10.0},
        "zero_head": 2,
        "batch_size": 1,
        "seed_base": 36812,
        "allow_unmet_demand": False,
        "lost_sales_penalty_factor": 200.0,
    },
]

# Add a light capacity jitter (3%) to all DemandBased presets
for _p in LOCAL_PRESETS:
    if _p.get("cap_mode") == "DemandBased":
        _p.setdefault("cap_params", {})
        _p["cap_params"].setdefault("jitter_pct", 3.0)
st.session_state["local_presets"] = {p["name"]: p for p in LOCAL_PRESETS}
# ===== END =====


# ==== INSTANCE GENERATION HELPERS (shared) ====
def generate_cap_series(period: int, mode: str, params: dict, seed: int) -> List[int]:
    g = rng(seed)
    if mode == "Constant":
        return [int(params.get("value", 10000))] * period
    if mode == "Uniform":
        lo, hi = int(params.get("lo", 9000)), int(params.get("hi", 11000))
        return [int(x) for x in g.integers(lo, hi + 1, size=period)]
    mu, sd = float(params.get("mean", 10000)), float(params.get("std", 500))
    clip_lo, clip_hi = float(params.get("clip_lo", 0)), float(
        params.get("clip_hi", 2 * mu)
    )
    arr = clip_list(g.normal(mu, sd, size=period), clip_lo, clip_hi).round().astype(int)
    return [int(x) for x in arr]


def generate_instance_from_class(cls: dict, j: int) -> dict:
    period = int(cls["period"])
    zero_head = int(cls.get("zero_head", 0))
    rng_local = np.random.default_rng(int(cls["seed_base"] + 31 * j))

    items = {}
    total_by_t = np.zeros(period, dtype=float)

    for i in range(cls["n_items"]):
        seed_k = int(cls["seed_base"] + 1000 * j + 17 * i)
        # demand
        D = list(
            np.random.default_rng(seed_k).integers(
                int(cls["dem_lo"]), int(cls["dem_hi"]) + 1, size=period
            )
        )
        if zero_head > 0:
            for z in range(min(zero_head, period)):
                D[z] = 0
        # shelf life
        Mseq = list(
            np.random.default_rng(seed_k + 1).integers(
                int(cls["m_lo"]), int(cls["m_hi"]) + 1, size=period
            )
        )
        # c, h
        c_seq = make_series(
            cls["c_mode"], period, dict(cls["c_params"] or {}), seed_k + 2
        )
        h_seq = make_series(
            cls["h_mode"], period, dict(cls["h_params"] or {}), seed_k + 3
        )
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

        # setup (TBO or generator)
        # setup (TBO or generator)
        if cls["s_mode"] == "tbo":
            # --- base from TBO formula ---
            # use mean(h) and mean(d) like before
            h_avg = float(h_out) if isinstance(h_out, float) else float(np.mean(h_out))
            d_avg = float(np.mean(D)) if D else 0.0

            L_raw = float((cls.get("s_params") or {}).get("L", 2.0))
            L = max(0.1, L_raw if math.isfinite(L_raw) else 2.0)
            jitter_pct = float(
                (cls.get("s_params") or {}).get("jitter_pct", SETUP_TBO_JITTER_DEFAULT)
            )
            per_period = bool(
                (cls.get("s_params") or {}).get("per_period", True)
            )  # ✅ default True

            base_s = 0.5 * h_avg * d_avg * (L**2)
            if base_s <= 0:
                base_s = 1e-6

            # use an item-specific RNG so items get different jitters
            g_s = np.random.default_rng(seed_k + 4)

            if per_period:
                # one jittered value per period
                eps = g_s.uniform(-jitter_pct / 100.0, jitter_pct / 100.0, size=period)
                s_out = list((base_s * (1.0 + eps)).astype(float))
            else:
                # single jittered scalar for the whole horizon
                jit = 1.0 + g_s.uniform(-jitter_pct / 100.0, jitter_pct / 100.0)
                s_out = float(base_s * jit)

        items[str(i)] = {
            "demand": [int(x) for x in D],
            "setup": s_out,
            "c_var": c_out,
            "h": h_out,
            "b_var": 0.0,
            "shelf_seq": [int(x) for x in Mseq],
        }
        total_by_t += np.array(D, dtype=float)

    # capacity
    cap_mode = cls.get("cap_mode", "Uniform")
    if cap_mode == "DemandBased":
        beta = CAP_TIGHT_BETAS.get(cls.get("cap_tight") or "Medium", 0.60)
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

    inst = {
        "period": period,
        "items": items,
        "manual_capacity": [int(x) for x in cap],
        "warehouse_capacity": (float(W_txt) if W_txt.strip() != "" else None),
        "allow_unmet_demand": bool(cls.get("allow_unmet_demand", False)),
        "lost_sales_penalty_factor": float(cls.get("lost_sales_penalty_factor", 200.0)),
        "meta": {"origin": "class", "class_key": cls["name"], "class_params": cls},
    }
    return inst


# ======================= Tabs ======================
cap_tab, items_tab, classes_tab, batch_tab, inspect_tab, saved_run_tab, viz_tab = (
    st.tabs(
        [
            "Capacity",
            "Items",
            "Classes",
            "Batch Runner",
            "Run Inspector",
            "Saved Instances Runner",
            "Explore & Visualize",
        ]
    )
)

# ----------------- Capacity tab -----------------

with cap_tab:
    st.header("Global Capacity κ_t")
    cap_mode = st.radio(
        "Capacity series mode",
        ["Constant", "Uniform", "Normal", "Manual list/grid"],
        horizontal=True,
    )
    cap_series = None

    if cap_mode == "Constant":
        v = st.number_input(
            "cap value", min_value=0, value=10000, step=100, key="cap_tab_val"
        )
        cap_series = [int(v)] * T

    elif cap_mode == "Uniform":
        lo = st.number_input("lo", min_value=0, value=9000, step=100, key="cap_tab_lo")
        hi = st.number_input("hi", min_value=0, value=11000, step=100, key="cap_tab_hi")
        cap_series = list(
            np.random.default_rng(SEED + 101).integers(lo, hi + 1, size=T)
        )

    elif cap_mode == "Normal":
        mu = st.number_input(
            "mean", min_value=0, value=10000, step=100, key="cap_tab_mean"
        )
        sd = st.number_input("std", min_value=0, value=500, step=10, key="cap_tab_std")
        clip_lo = st.number_input(
            "clip_lo", min_value=0, value=0, step=100, key="cap_tab_clo"
        )
        clip_hi = st.number_input(
            "clip_hi", min_value=0, value=20000, step=100, key="cap_tab_chi"
        )
        arr = np.random.default_rng(SEED + 102).normal(mu, sd, size=T)
        arr = clip_list(arr, clip_lo, clip_hi).round().astype(int)
        cap_series = list(arr)

    else:
        df = pd.DataFrame({"t": list(range(T)), "cap_t": [10000] * T})
        edited = st.data_editor(
            df, use_container_width=True, hide_index=True, num_rows="fixed"
        )
        cap_series = [int(x) for x in edited["cap_t"].tolist()]

    st.line_chart(cap_series, height=140)
    st.caption("Preview of κ_t")


# ----------------- Items tab (ad-hoc instance builder) -----------------
@dataclass
class ItemSpec:
    item_id: int
    c_mode: str = "scalar"
    c_params: dict = None
    h_mode: str = "scalar"
    h_params: dict = None
    s_mode: str = "seasonal"
    s_params: dict = None
    m_min: int = 6
    m_max: int = 50
    dem_lo: int = 5
    dem_hi: int = 80
    remove: bool = False


def default_item(i=0):
    return ItemSpec(
        item_id=i,
        c_mode="scalar",
        c_params={"value": 2.0},
        h_mode="scalar",
        h_params={"value": 0.4},
        s_mode="seasonal",
        s_params={
            "base": 80.0,
            "amp": 0.10,
            "period": 30.0,
            "phase": 0.0,
            "noise_std": 0.0,
            "clip_lo": 0.0,
        },
        m_min=6,
        m_max=50,
        dem_lo=dem_lo,
        dem_hi=dem_hi,
    )


if "item_specs" not in st.session_state:
    st.session_state["item_specs"] = [default_item(0), default_item(1), default_item(2)]

with items_tab:
    st.header("Ad-hoc Items (quick instance)")
    cols = st.columns(3)
    with cols[0]:
        if st.button("➕ Add item"):
            next_id = (
                (max([it.item_id for it in st.session_state["item_specs"]]) + 1)
                if st.session_state["item_specs"]
                else 0
            )
            st.session_state["item_specs"].append(default_item(next_id))
    with cols[1]:
        if st.button("🧹 Clear items"):
            st.session_state["item_specs"] = []
    with cols[2]:
        if st.button("♻️ Reset demo"):
            st.session_state["item_specs"] = [
                default_item(0),
                default_item(1),
                default_item(2),
            ]

    for idx, it in enumerate(st.session_state["item_specs"]):
        with st.expander(f"Item {it.item_id}", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                it.item_id = st.number_input(
                    "id", value=int(it.item_id), key=f"id_{idx}"
                )
            with c2:
                it.dem_lo = st.number_input(
                    "dem_lo", value=int(it.dem_lo), key=f"dlo_{idx}"
                )
            with c3:
                it.dem_hi = st.number_input(
                    "dem_hi", value=int(it.dem_hi), key=f"dhi_{idx}"
                )
            with c4:
                it.remove = st.checkbox("Remove", key=f"rm_{idx}", value=False)

            st.markdown("**c_{i,t} generator**")
            it.c_mode = st.selectbox(
                "mode (c)",
                ["scalar", "uniform", "normal", "linear", "seasonal", "manual"],
                index=0,
                key=f"c_mode_{idx}",
            )
            it.c_params = it.c_params or {}
            if it.c_mode == "scalar":
                it.c_params["value"] = st.number_input(
                    "c value",
                    value=float(it.c_params.get("value", 2.0)),
                    key=f"c_val_{idx}",
                )
            elif it.c_mode == "uniform":
                it.c_params["lo"] = st.number_input(
                    "c lo", value=float(it.c_params.get("lo", 1.0)), key=f"c_lo_{idx}"
                )
                it.c_params["hi"] = st.number_input(
                    "c hi", value=float(it.c_params.get("hi", 3.0)), key=f"c_hi_{idx}"
                )
            elif it.c_mode == "normal":
                it.c_params["mean"] = st.number_input(
                    "c mean",
                    value=float(it.c_params.get("mean", 2.0)),
                    key=f"c_mu_{idx}",
                )
                it.c_params["std"] = st.number_input(
                    "c std", value=float(it.c_params.get("std", 0.2)), key=f"c_sd_{idx}"
                )
                it.c_params["clip_lo"] = st.number_input(
                    "clip_lo",
                    value=float(it.c_params.get("clip_lo", 0.0)),
                    key=f"c_clo_{idx}",
                )
                it.c_params["clip_hi"] = st.number_input(
                    "clip_hi",
                    value=float(it.c_params.get("clip_hi", 10.0)),
                    key=f"c_chi_{idx}",
                )
            elif it.c_mode == "linear":
                it.c_params["base"] = st.number_input(
                    "base", value=float(it.c_params.get("base", 2.0)), key=f"c_b_{idx}"
                )
                it.c_params["slope"] = st.number_input(
                    "slope",
                    value=float(it.c_params.get("slope", 0.0)),
                    key=f"c_sl_{idx}",
                )
                it.c_params["noise_std"] = st.number_input(
                    "noise_std",
                    value=float(it.c_params.get("noise_std", 0.0)),
                    key=f"c_ns_{idx}",
                )
                it.c_params["clip_lo"] = st.number_input(
                    "clip_lo",
                    value=float(it.c_params.get("clip_lo", 0.0)),
                    key=f"c_lcl_{idx}",
                )
                it.c_params["clip_hi"] = st.number_input(
                    "clip_hi",
                    value=float(it.c_params.get("clip_hi", 10.0)),
                    key=f"c_lch_{idx}",
                )
            elif it.c_mode == "seasonal":
                it.c_params["base"] = st.number_input(
                    "base", value=float(it.c_params.get("base", 2.0)), key=f"c_sb_{idx}"
                )
                it.c_params["amp"] = st.number_input(
                    "amp",
                    value=float(it.c_params.get("amp", 0.1)),
                    step=0.01,
                    key=f"c_sa_{idx}",
                )
                it.c_params["period"] = st.number_input(
                    "period",
                    value=float(it.c_params.get("period", 30.0)),
                    key=f"c_sp_{idx}",
                )
                it.c_params["phase"] = st.number_input(
                    "phase",
                    value=float(it.c_params.get("phase", 0.0)),
                    key=f"c_sph_{idx}",
                )
                it.c_params["noise_std"] = st.number_input(
                    "noise_std",
                    value=float(it.c_params.get("noise_std", 0.0)),
                    key=f"c_sn_{idx}",
                )
                it.c_params["clip_lo"] = st.number_input(
                    "clip_lo",
                    value=float(it.c_params.get("clip_lo", 0.0)),
                    key=f"c_scl_{idx}",
                )
                it.c_params["clip_hi"] = st.number_input(
                    "clip_hi",
                    value=float(it.c_params.get("clip_hi", 10.0)),
                    key=f"c_sch_{idx}",
                )
            else:
                st.caption("Enter exactly T numbers separated by commas or spaces.")
                raw = st.text_area(
                    "c list", value=it.c_params.get("raw", ""), key=f"c_raw_{idx}"
                )
                it.c_params["raw"] = raw

            st.markdown("---")
            st.markdown("**h_{i,t} generator**")
            it.h_mode = st.selectbox(
                "mode (h)",
                ["scalar", "uniform", "normal", "linear", "seasonal", "manual"],
                index=0,
                key=f"h_mode_{idx}",
            )
            it.h_params = it.h_params or {}
            if it.h_mode == "scalar":
                it.h_params["value"] = st.number_input(
                    "h value",
                    value=float(it.h_params.get("value", 0.4)),
                    key=f"h_val_{idx}",
                )
            elif it.h_mode == "uniform":
                it.h_params["lo"] = st.number_input(
                    "h lo", value=float(it.h_params.get("lo", 0.2)), key=f"h_lo_{idx}"
                )
                it.h_params["hi"] = st.number_input(
                    "h hi", value=float(it.h_params.get("hi", 0.8)), key=f"h_hi_{idx}"
                )
            elif it.h_mode == "normal":
                it.h_params["mean"] = st.number_input(
                    "h mean",
                    value=float(it.h_params.get("mean", 0.4)),
                    key=f"h_mu_{idx}",
                )
                it.h_params["std"] = st.number_input(
                    "h std",
                    value=float(it.h_params.get("std", 0.05)),
                    key=f"h_sd_{idx}",
                )
                it.h_params["clip_lo"] = st.number_input(
                    "clip_lo",
                    value=float(it.h_params.get("clip_lo", 0.0)),
                    key=f"h_clo_{idx}",
                )
                it.h_params["clip_hi"] = st.number_input(
                    "clip_hi",
                    value=float(it.h_params.get("clip_hi", 10.0)),
                    key=f"h_chi_{idx}",
                )
            elif it.h_mode == "linear":
                it.h_params["base"] = st.number_input(
                    "base", value=float(it.h_params.get("base", 0.4)), key=f"h_b_{idx}"
                )
                it.h_params["slope"] = st.number_input(
                    "slope",
                    value=float(it.h_params.get("slope", 0.0)),
                    key=f"h_sl_{idx}",
                )
                it.h_params["noise_std"] = st.number_input(
                    "noise_std",
                    value=float(it.h_params.get("noise_std", 0.0)),
                    key=f"h_ns_{idx}",
                )
                it.h_params["clip_lo"] = st.number_input(
                    "clip_lo",
                    value=float(it.h_params.get("clip_lo", 0.0)),
                    key=f"h_lcl_{idx}",
                )
                it.h_params["clip_hi"] = st.number_input(
                    "clip_hi",
                    value=float(it.h_params.get("clip_hi", 10.0)),
                    key=f"h_lch_{idx}",
                )
            elif it.h_mode == "seasonal":
                it.h_params["base"] = st.number_input(
                    "base", value=float(it.h_params.get("base", 0.4)), key=f"h_sb_{idx}"
                )
                it.h_params["amp"] = st.number_input(
                    "amp",
                    value=float(it.h_params.get("amp", 0.1)),
                    step=0.01,
                    key=f"h_sa_{idx}",
                )
                it.h_params["period"] = st.number_input(
                    "period",
                    value=float(it.h_params.get("period", 30.0)),
                    key=f"h_sp_{idx}",
                )
                it.h_params["phase"] = st.number_input(
                    "phase",
                    value=float(it.h_params.get("phase", 0.0)),
                    key=f"h_sph_{idx}",
                )
                it.h_params["noise_std"] = st.number_input(
                    "noise_std",
                    value=float(it.h_params.get("noise_std", 0.0)),
                    key=f"h_sn_{idx}",
                )
                it.h_params["clip_lo"] = st.number_input(
                    "clip_lo",
                    value=float(it.h_params.get("clip_lo", 0.0)),
                    key=f"h_scl_{idx}",
                )
                it.h_params["clip_hi"] = st.number_input(
                    "clip_hi",
                    value=float(it.h_params.get("clip_hi", 10.0)),
                    key=f"h_sch_{idx}",
                )
            else:
                raw = st.text_area(
                    "h list", value=it.h_params.get("raw", ""), key=f"h_raw_{idx}"
                )
                it.h_params["raw"] = raw

            st.markdown("---")
            st.markdown("**s_{i,t} (setup) generator**")
            it.s_mode = st.selectbox(
                "mode (s)",
                ["scalar", "uniform", "normal", "linear", "seasonal", "manual"],
                index=4,
                key=f"s_mode_{idx}",
            )
            it.s_params = it.s_params or {}
            if it.s_mode == "scalar":
                it.s_params["value"] = st.number_input(
                    "s value",
                    value=float(it.s_params.get("value", 80.0)),
                    key=f"s_val_{idx}",
                )
            elif it.s_mode == "uniform":
                it.s_params["lo"] = st.number_input(
                    "s lo", value=float(it.s_params.get("lo", 50.0)), key=f"s_lo_{idx}"
                )
                it.s_params["hi"] = st.number_input(
                    "s hi", value=float(it.s_params.get("hi", 150.0)), key=f"s_hi_{idx}"
                )
            elif it.s_mode == "normal":
                it.s_params["mean"] = st.number_input(
                    "s mean",
                    value=float(it.s_params.get("mean", 80.0)),
                    key=f"s_mu_{idx}",
                )
                it.s_params["std"] = st.number_input(
                    "s std",
                    value=float(it.s_params.get("std", 10.0)),
                    key=f"s_sd_{idx}",
                )
                it.s_params["clip_lo"] = st.number_input(
                    "clip_lo",
                    value=float(it.s_params.get("clip_lo", 0.0)),
                    key=f"s_clo_{idx}",
                )
                it.s_params["clip_hi"] = st.number_input(
                    "clip_hi",
                    value=float(it.s_params.get("clip_hi", 1e6)),
                    key=f"s_chi_{idx}",
                )
            elif it.s_mode == "linear":
                it.s_params["base"] = st.number_input(
                    "base", value=float(it.s_params.get("base", 80.0)), key=f"s_b_{idx}"
                )
                it.s_params["slope"] = st.number_input(
                    "slope",
                    value=float(it.s_params.get("slope", 0.0)),
                    key=f"s_sl_{idx}",
                )
                it.s_params["noise_std"] = st.number_input(
                    "noise_std",
                    value=float(it.s_params.get("noise_std", 0.0)),
                    key=f"s_ns_{idx}",
                )
                it.s_params["clip_lo"] = st.number_input(
                    "clip_lo",
                    value=float(it.s_params.get("clip_lo", 0.0)),
                    key=f"s_lcl_{idx}",
                )
                it.s_params["clip_hi"] = st.number_input(
                    "clip_hi",
                    value=float(it.s_params.get("clip_hi", 1e6)),
                    key=f"s_lch_{idx}",
                )
            elif it.s_mode == "seasonal":
                it.s_params["base"] = st.number_input(
                    "base",
                    value=float(it.s_params.get("base", 80.0)),
                    key=f"s_sb_{idx}",
                )
                it.s_params["amp"] = st.number_input(
                    "amp",
                    value=float(it.s_params.get("amp", 0.10)),
                    step=0.01,
                    key=f"s_sa_{idx}",
                )
                it.s_params["period"] = st.number_input(
                    "period",
                    value=float(it.s_params.get("period", 30.0)),
                    key=f"s_sp_{idx}",
                )
                it.s_params["phase"] = st.number_input(
                    "phase",
                    value=float(it.s_params.get("phase", 0.0)),
                    key=f"s_sph_{idx}",
                )
                it.s_params["noise_std"] = st.number_input(
                    "noise_std",
                    value=float(it.s_params.get("noise_std", 0.0)),
                    key=f"s_sn_{idx}",
                )
                it.s_params["clip_lo"] = st.number_input(
                    "clip_lo",
                    value=float(it.s_params.get("clip_lo", 0.0)),
                    key=f"s_scl_{idx}",
                )
                it.s_params["clip_hi"] = st.number_input(
                    "clip_hi",
                    value=float(it.s_params.get("clip_hi", 1e6)),
                    key=f"s_sch_{idx}",
                )
            else:
                raw = st.text_area(
                    "s list", value=it.s_params.get("raw", ""), key=f"s_raw_{idx}"
                )
                it.s_params["raw"] = raw

            st.markdown("---")
            c5, c6 = st.columns(2)
            with c5:
                it.m_min = st.number_input(
                    "shelf life min", value=int(it.m_min), key=f"mlo_{idx}"
                )
            with c6:
                it.m_max = st.number_input(
                    "shelf life max", value=int(it.m_max), key=f"mhi_{idx}"
                )

    st.session_state["item_specs"] = [
        it for it in st.session_state["item_specs"] if not it.remove
    ]

    if st.button("🛠️ Generate instance (ad-hoc)"):
        items = {}
        for k, it in enumerate(st.session_state["item_specs"]):
            seed_k = int(SEED + 1000 + 17 * k)
            D = list(
                np.random.default_rng(seed_k).integers(
                    int(it.dem_lo), int(it.dem_hi) + 1, size=T
                )
            )
            Mseq = (
                [int(it.m_min)] * T
                if it.m_min == it.m_max
                else list(
                    np.random.default_rng(seed_k + 1).integers(
                        int(it.m_min), int(it.m_max) + 1, size=T
                    )
                )
            )
            # c
            if it.c_mode == "manual":
                c_list = parse_manual_list(it.c_params.get("raw", ""), T)
                c = c_list if c_list else [float(it.c_params.get("value", 2.0))] * T
            else:
                c = make_series(it.c_mode, T, dict(it.c_params or {}), seed_k + 2)
            c_out = (
                float(it.c_params.get("value", 0.0))
                if it.c_mode == "scalar"
                else [float(x) for x in c]
            )
            # h
            if it.h_mode == "manual":
                h_list = parse_manual_list(it.h_params.get("raw", ""), T)
                h = h_list if h_list else [float(it.h_params.get("value", 0.4))] * T
            else:
                h = make_series(it.h_mode, T, dict(it.h_params or {}), seed_k + 3)
            h_out = (
                float(it.h_params.get("value", 0.0))
                if it.h_mode == "scalar"
                else [float(x) for x in h]
            )
            # s
            if it.s_mode == "manual":
                s_list = parse_manual_list(it.s_params.get("raw", ""), T)
                s = s_list if s_list else [float(it.s_params.get("value", 80.0))] * T
            else:
                s = make_series(it.s_mode, T, dict(it.s_params or {}), seed_k + 4)
            s_out = (
                float(it.s_params.get("value", 0.0))
                if it.s_mode == "scalar"
                else [float(x) for x in s]
            )

            items[str(int(it.item_id))] = {
                "demand": [int(x) for x in D],
                "setup": s_out,
                "c_var": c_out,
                "h": h_out,
                "b_var": 0.0,
                "shelf_seq": [int(x) for x in Mseq],
            }

        inst = {
            "period": int(T),
            "items": items,
            "manual_capacity": [int(x) for x in cap_series],
            "warehouse_capacity": (float(W_txt) if W_txt.strip() != "" else None),
            "allow_unmet_demand": bool(allow_unmet_default),
            "lost_sales_penalty_factor": float(ls_penalty_factor_default),
            "meta": {"origin": "adhoc"},
        }
        Path("last_instance.json").write_text(
            json.dumps(to_py(inst), indent=2), encoding="utf-8"
        )
        st.success("Instance saved → last_instance.json")
        st.code(Path("last_instance.json").read_text()[:3000], language="json")

    if st.button("🚀 Solve (ad-hoc)"):
        if not Path("last_instance.json").exists():
            st.error("Generate an instance first.")
        else:
            with st.spinner("Solving..."):
                _solve_fn, _solver_tag = get_solver_backend()
                summary, orders_txt = _solve_fn(
                    "last_instance.json", time_limit=time_limit, mip_gap=mip_gap
                )
            st.subheader("Summary")
            st.json(summary)
            st.subheader("Orders")
            st.text("\n".join(orders_txt))
            sb = supabase_client()
            if sb is not None:
                try:
                    inst_json = json.loads(Path("last_instance.json").read_text())
                    inst_res = (
                        sb.table("instances")
                        .insert(
                            {
                                "period": int(inst_json["period"]),
                                "manual_capacity": inst_json.get("manual_capacity"),
                                "warehouse_capacity": inst_json.get(
                                    "warehouse_capacity"
                                ),
                                "data": to_py(inst_json),
                            }
                        )
                        .execute()
                    )
                    instance_id = inst_res.data[0]["id"]
                    run_res = (
                        sb.table("runs")
                        .insert(
                            {
                                "instance_id": instance_id,
                                "time_limit_sec": int(time_limit),
                                "mip_gap": float(mip_gap),
                                "status": int(summary.get("status")),
                                "objective": summary.get("objective"),
                                "best_bound": summary.get("best_bound"),
                                "gap": summary.get("gap"),
                                "runtime_sec": summary.get("runtime_sec"),
                                "solver_version": _solver_tag,
                            }
                        )
                        .execute()
                    )
                    run_id = run_res.data[0]["id"]
                    # parse orders
                    rows = [
                        {"run_id": run_id, **r} for r in parse_orders_lines(orders_txt)
                    ]
                    if rows:
                        CHUNK = 500
                        for k in range(0, len(rows), CHUNK):
                            sb.table("orders").insert(rows[k : k + CHUNK]).execute()
                    st.success(
                        f"Logged to Supabase. instance_id={instance_id}, run_id={run_id}"
                    )
                except Exception as e:
                    st.warning(f"Supabase logging failed: {e}")

# ----------------- Classes tab -----------------
with classes_tab:
    st.header("Solution Classes (define and queue)")

    # ---------- Full-table fetch (keyset pagination; no 1k cap) ----------
    def fetch_all_classes_by_keyset(
        sb_client: Client, page_size: int = 1000
    ) -> list[dict]:
        """
        Return ALL rows from public.classes using keyset pagination on 'name' ASC.
        Assumes 'name' is unique/sortable.
        """
        if sb_client is None:
            return []
        out: list[dict] = []
        cursor: str | None = None
        select_cols = "id,name,spec,created_at"
        while True:
            q = sb_client.table("classes").select(select_cols).order("name", desc=False)
            if cursor is not None:
                q = q.gt("name", cursor)  # strictly after last seen name
            batch = (q.limit(page_size).execute().data) or []
            if not batch:
                break
            out.extend(batch)
            cursor = batch[-1]["name"]
            if len(batch) < page_size:
                break
        return out

    # ---- Compat shim: ensure fetch_classes() does full fetch without rendering banners
    if "fetch_classes" not in globals():

        def fetch_classes(sb_client: Client) -> list[dict]:
            return fetch_all_classes_by_keyset(sb_client)

        # mark so we don't overwrite elsewhere
        fetch_classes._full_fetch = True  # type: ignore[attr-defined]
    else:
        # if someone defined a 1k-limited fetch, override it
        try:
            if not getattr(fetch_classes, "_full_fetch", False):  # type: ignore[name-defined]

                def fetch_classes(sb_client: Client) -> list[dict]:  # type: ignore[no-redef]
                    return fetch_all_classes_by_keyset(sb_client)

                fetch_classes._full_fetch = True  # type: ignore[attr-defined]
        except Exception:

            def fetch_classes(sb_client: Client) -> list[dict]:  # fallback
                return fetch_all_classes_by_keyset(sb_client)

    # ---------- Always load full set on render (prevents shrink-on-rerun) ----------
    sb = supabase_client()
    if not sb:
        st.info("Configure Supabase in the sidebar.")
        st.stop()

    rows = fetch_all_classes_by_keyset(sb)  # <- always full set
    st.session_state["db_classes"] = {r["name"]: r for r in rows}
    db_map: dict[str, dict] = st.session_state["db_classes"]

    st.subheader("Classes in Supabase")
    st.caption(f"Loaded **{len(db_map)}** classes from Supabase.")

    # ---------- X-code legend & helpers ----------
    with st.expander("X-code legend (A..F)", expanded=False):
        st.markdown(
            """
        **Format:** `X A B C D E F`  (e.g., `X232237`)  

        | pos | meaning | codes → value |
        |---|---|---|
        | A | period **T** (also sets `zero_head`) | 1→20 (`zero_head`=2), 2→30 (3), 3→40 (4) |
        | B | **#items** (`n_items`) | 1→10, 2→20, 3→30 |
        | C | capacity tightness (`cap_tight`) | 1→Loose, 2→Tight |
        | D | demand CV level (drives demand range) | 1→Low CV → `dem_hi`=125, 2→High CV → `dem_hi`=200 (both `dem_lo`=0) , L low from folders / H high from folders |
        | E | shelf-life set (`m_lo`,`m_hi`) | 1→(1,10), 2→(5,15), 3→(10,20), 4→(5,25), 5→(10,30),  **A**: m∈[0, ⌊T/2⌋], **B**: m∈[0, ⌊3T/4⌋], **C**: m∈[5, ⌊T/2⌋]  |
        | F | TBO parameter (`s_params.L`) | integer **1..12** |
        """
        )

    # ---------- Table with checkboxes ----------
    st.session_state.setdefault("db_table_checked", set())

    def _safe_spec(row: dict, key: str, default=None):
        try:
            return (row.get("spec") or {}).get(key, default)
        except Exception:
            return default

    df_db = pd.DataFrame(
        [
            {
                "name": r["name"],
                "period": _safe_spec(r, "period"),
                "#items": _safe_spec(r, "n_items"),
                "cap_mode": _safe_spec(r, "cap_mode"),
                "created_at": r.get("created_at"),
            }
            for r in db_map.values()
        ]
    ).sort_values("name", kind="mergesort")

    # Keep existing checks
    df_db["✓"] = df_db["name"].isin(st.session_state["db_table_checked"])

    edited = st.data_editor(
        df_db,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "name": st.column_config.TextColumn(disabled=True),
            "period": st.column_config.NumberColumn(format="%d", disabled=True),
            "#items": st.column_config.NumberColumn(format="%d", disabled=True),
            "cap_mode": st.column_config.TextColumn(disabled=True),
            "created_at": st.column_config.TextColumn(disabled=True),
            "✓": st.column_config.CheckboxColumn(
                "✓", help="Check to include in actions below.", default=False
            ),
        },
        key="db_classes_editor",
        height=560,
    )

    # Sync back checks (and drop any stale names)
    try:
        checked_now = set(edited.loc[edited["✓"], "name"].astype(str).tolist())
        st.session_state["db_table_checked"] = checked_now & set(db_map.keys())
    except Exception:
        pass

    # ---------- X-code bulk selection (include/exclude by A..F) ----------
    with st.expander("Bulk select X-coded classes by legend filters", expanded=False):
        POS = ["A", "B", "C", "D", "E", "F"]
        CHOICES = {
            "A": [str(i) for i in (1, 2, 3)],
            "B": [str(i) for i in (1, 2, 3)],
            "C": [str(i) for i in (1, 2)],
            "D": ["1", "2", "L", "H"],
            "E": ["A", "B", "C", "1", "2", "3", "4", "5"],  # letters primary now
            "F": [str(i) for i in range(1, 13)],
        }

        # Persist selections across reruns
        for p in POS:
            st.session_state.setdefault(f"x_inc_{p}", [])
            st.session_state.setdefault(f"x_exc_{p}", [])

        ccols = st.columns(6)
        for idx, p in enumerate(POS):
            with ccols[idx]:
                st.session_state[f"x_inc_{p}"] = st.multiselect(
                    f"Include {p}",
                    options=CHOICES[p],
                    default=st.session_state[f"x_inc_{p}"],
                    key=f"x_inc_ms_{p}",
                    help=f"Leave empty to allow any {p}",
                )
                st.session_state[f"x_exc_{p}"] = st.multiselect(
                    f"Exclude {p}",
                    options=CHOICES[p],
                    default=st.session_state[f"x_exc_{p}"],
                    key=f"x_exc_ms_{p}",
                    help=f"Digits of {p} to exclude",
                )

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("✨ Check all X*"):
                x_names = [n for n in db_map.keys() if _x_digits(n)]
                if not x_names:
                    st.info("No X-coded classes in DB.")
                else:
                    st.session_state["db_table_checked"].update(x_names)
                    st.rerun()
        with b2:
            if st.button("➕ Check matches (keep current)"):
                matches = [n for n in db_map.keys() if _matches_x_filters(n)]
                if not matches:
                    st.info("No X-coded classes match current filters.")
                else:
                    st.session_state["db_table_checked"].update(matches)
                    st.rerun()
        with b3:
            if st.button("📌 Replace checks with matches"):
                matches = [n for n in db_map.keys() if _matches_x_filters(n)]
                st.session_state["db_table_checked"] = set(matches)
                st.rerun()
        with b4:
            if st.button("🧹 Clear all checks"):
                st.session_state["db_table_checked"].clear()
                st.rerun()

        st.caption(
            "Examples: **B=2** → include B=[2]. **F=9** → include F=[9]. "
            "**A∈{2,3} & F∉{1,2,3,4}** → include A=[2,3], exclude F=[1,2,3,4]."
        )

    # ---------- Actions ----------
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("➕ Queue checked (DB)"):
            chosen = list(st.session_state["db_table_checked"])
            if not chosen:
                st.info("No rows are checked.")
            else:
                st.session_state.setdefault("classes", [])
                st.session_state["classes"] += [
                    db_map[name]["spec"] for name in chosen if name in db_map
                ]
                try:
                    dedupe_queue_by_name()
                except Exception:
                    pass
                st.success(f"Queued {len(chosen)} DB class(es).")

    with c2:
        if st.button("📥 Replace queue with ALL DB"):
            st.session_state["classes"] = [r["spec"] for r in db_map.values()]
            try:
                dedupe_queue_by_name()
            except Exception:
                pass
            st.success(
                f"Queue replaced with {len(st.session_state['classes'])} DB class(es)."
            )

    with c3:
        if st.button("🔄 Full refresh now"):
            # just rerun; full fetch happens at top every render
            st.rerun()

    st.markdown("### Generate instances only (no solve) for checked classes")
    gen_copies = st.number_input(
        "Copies per class to generate",
        min_value=1,
        value=5,
        step=1,
        key="gen_only_copies",
    )

    if st.button("🧪 Generate & save (no solve) for CHECKED"):
        sbx = supabase_client()
        if sbx is None:
            st.error("Supabase not configured.")
        else:
            # which classes? — the currently checked rows in the table
            chosen = list(st.session_state.get("db_table_checked", set()))
            if not chosen:
                st.info("No classes are checked.")
            else:
                prog = st.progress(0.0, text="Generating…")
                done = 0
                total = len(chosen) * int(gen_copies)

                # prebuild name->spec and name->class_id
                name_to_spec = {nm: db_map[nm]["spec"] for nm in chosen if nm in db_map}
                name_to_cid = {nm: db_map[nm]["id"] for nm in chosen if nm in db_map}

                for nm in chosen:
                    cls_spec = name_to_spec[nm]
                    cid = name_to_cid.get(nm)
                    for j in range(int(gen_copies)):
                        inst = generate_instance_from_class(cls_spec, j)
                        payload = sanitize_json(
                            {
                                "period": int(inst["period"]),
                                "manual_capacity": inst.get("manual_capacity"),
                                "warehouse_capacity": inst.get("warehouse_capacity"),
                                "data": to_py(inst),
                                "class_id": cid,
                            }
                        )
                        try:
                            sbx.table("instances").insert(payload).execute()
                        except Exception as e:
                            st.warning(f"Insert failed for {nm} copy {j+1}: {e}")
                        done += 1
                        prog.progress(done / total, text=f"Generating… {done}/{total}")

                st.success(f"Generated {total} instance(s) with no solve.")

    # ---------- Local presets (in code) ----------
    lp_map = st.session_state.get("local_presets", {})
    st.subheader("Local presets (in code)")
    if lp_map:
        df_lp = pd.DataFrame(
            [
                {
                    "name": p["name"],
                    "period": p["period"],
                    "#items": p["n_items"],
                    "cap_mode": p["cap_mode"],
                }
                for p in lp_map.values()
            ]
        ).sort_values("name", kind="mergesort")
        st.dataframe(df_lp, use_container_width=True)

        pick_local = st.multiselect(
            "Select local presets", options=sorted(lp_map.keys()), key="pick_local"
        )

        cL1, cL2 = st.columns(2)
        with cL1:
            if st.button("➕ Queue selected local"):
                for name in pick_local:
                    st.session_state.setdefault("classes", []).append(lp_map[name])
                try:
                    dedupe_queue_by_name()
                except Exception:
                    pass
                st.success("Queued selected local preset(s).")

        with cL2:
            if st.button("⬆️ Save selected local to Supabase"):
                sbx = supabase_client()
                saved = 0
                for name in pick_local:
                    if ensure_class_row(sbx, lp_map[name]):
                        saved += 1
                # refresh DB list so they appear immediately
                st.session_state["db_classes"] = {
                    r["name"]: r for r in fetch_classes(sbx)
                }
                st.success(f"Saved/updated {saved} class(es) in Supabase.")
    else:
        st.caption(
            "No local presets defined. Edit LOCAL_PRESETS near the top to add some."
        )

    # ---------- New class form ----------

    c_name = st.text_input("Class name", value="X...", key="cls_name")

    st.markdown("**Periods**")
    pchoice = st.radio(
        "Choose T", ["20", "40", "60", "Custom"], horizontal=True, key="cls_Tpick"
    )
    if pchoice == "Custom":
        c_period = st.number_input(
            "Custom T", min_value=1, value=int(T), step=1, key="cls_period_custom"
        )
    else:
        c_period = int(pchoice)

    st.markdown("**Capacity generator (κ_t)**")
    c_cap_mode = st.selectbox(
        "mode",
        ["Constant", "Uniform", "Normal", "Demand-based (L/M/T)"],
        index=1,
        key="cls_cap_mode",
    )

    cap_params: Dict[str, Any] = {}
    cap_tight = None
    if c_cap_mode == "Constant":
        cap_params["value"] = st.number_input(
            "cap value", min_value=0, value=10000, step=100, key="cls_cap_val"
        )
    elif c_cap_mode == "Uniform":
        cap_params["lo"] = st.number_input(
            "lo", min_value=0, value=9000, step=100, key="cls_cap_lo"
        )
        cap_params["hi"] = st.number_input(
            "hi", min_value=0, value=11000, step=100, key="cls_cap_hi"
        )
    elif c_cap_mode == "Normal":
        cap_params["mean"] = st.number_input(
            "mean", min_value=0, value=10000, step=100, key="cls_cap_mean"
        )
        cap_params["std"] = st.number_input(
            "std", min_value=0, value=500, step=10, key="cls_cap_std"
        )
        cap_params["clip_lo"] = st.number_input(
            "clip_lo", min_value=0, value=0, step=100, key="cls_cap_clo"
        )
        cap_params["clip_hi"] = st.number_input(
            "clip_hi", min_value=0, value=20000, step=100, key="cls_cap_chi"
        )
    else:  # Demand-based
        cap_tight = st.selectbox(
            "Tightness",
            ["Loose", "Medium", "Tight", "Tighter", "Ultra"],
            index=1,
            key="cls_cap_tight",
        )
        cap_jitter_pct = st.number_input(
            "Jitter κ_t (±%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            key="cls_cap_jitter",
        )
        cap_params["jitter_pct"] = float(cap_jitter_pct)
        st.caption("Capacity per period = β × mean(total demand).")
    # -----------------------------------------------------------------------------
    st.markdown("**Item count**")
    use_custom_n = st.checkbox("Use custom item count", value=False, key="use_custom_n")
    bucket_map = {
        "Tiny(3)": 3,
        "Small(5)": 5,
        "Medium(8)": 8,
        "Large(15)": 15,
        "XL(40)": 40,
        "XXL(100)": 100,
    }
    if use_custom_n:
        n_items = st.number_input("n_items (custom)", min_value=1, value=8, step=1)
    else:
        bucket = st.selectbox(
            "bucket",
            list(bucket_map.keys()),
            index=2,
            key="bucket_pick",
        )
        n_items = bucket_map[bucket]

    st.markdown("**Demand and Shelf-life per item**")
    c_dem_lo = st.number_input("demand lo", min_value=0, value=5, step=1)
    c_dem_hi = st.number_input("demand hi", min_value=1, value=80, step=1)
    c_m_lo = st.number_input("m min", min_value=1, value=6, step=1)
    c_m_hi = st.number_input("m max", min_value=1, value=50, step=1)

    zero_head = st.number_input(
        "Force first Z periods demand=0",
        min_value=0,
        max_value=int(c_period),
        value=0,
        step=1,
        key="cls_zero_head",
    )

    cls_allow_unmet = st.checkbox(
        "Allow unmet demand (lost sales)", value=False, key="cls_allow_unmet"
    )
    cls_ls_factor = st.number_input(
        "Lost-sales penalty factor (× mean demand)",
        min_value=10.0,
        value=200.0,
        step=10.0,
        format="%.1f",
        key="cls_ls_factor",
    )
    # -----------------------------------------------------------------------------

    st.markdown("**c_it / h_it / s_it generators (applied to all items)**")
    gen_modes = ["scalar", "uniform", "normal", "linear", "seasonal"]
    c_mode = st.selectbox("mode (c_it)", gen_modes, index=0, key="cls_c_mode")
    h_mode = st.selectbox("mode (h_it)", gen_modes, index=0, key="cls_h_mode")
    s_mode = st.selectbox("mode (s_it)", gen_modes + ["tbo"], index=4, key="cls_s_mode")

    def ui_params(prefix: str, mode: str, defaults: dict, key_base: str) -> dict:
        # key_base differentiates groups (cls_c / cls_h / cls_s)
        p = {}
        kb = f"{key_base}_{prefix}"

        if mode == "scalar":
            p["value"] = st.number_input(
                f"{prefix} value",
                value=float(defaults.get("value", 1.0)),
                key=f"{kb}_value",
            )

        elif mode == "uniform":
            p["lo"] = st.number_input(
                f"{prefix} lo",
                value=float(defaults.get("lo", 1.0)),
                key=f"{kb}_lo",
            )
            p["hi"] = st.number_input(
                f"{prefix} hi",
                value=float(defaults.get("hi", 3.0)),
                key=f"{kb}_hi",
            )

        elif mode == "normal":
            p["mean"] = st.number_input(
                f"{prefix} mean",
                value=float(defaults.get("mean", 1.0)),
                key=f"{kb}_mean",
            )
            p["std"] = st.number_input(
                f"{prefix} std",
                value=float(defaults.get("std", 0.2)),
                key=f"{kb}_std",
            )
            p["clip_lo"] = st.number_input(
                f"{prefix} clip_lo",
                value=float(defaults.get("clip_lo", 0.0)),
                key=f"{kb}_cliplo",
            )
            p["clip_hi"] = st.number_input(
                f"{prefix} clip_hi",
                value=float(defaults.get("clip_hi", 10.0)),
                key=f"{kb}_cliphi",
            )

        elif mode == "linear":
            p["base"] = st.number_input(
                f"{prefix} base",
                value=float(defaults.get("base", 1.0)),
                key=f"{kb}_base",
            )
            p["slope"] = st.number_input(
                f"{prefix} slope",
                value=float(defaults.get("slope", 0.0)),
                key=f"{kb}_slope",
            )
            p["noise_std"] = st.number_input(
                f"{prefix} noise_std",
                value=float(defaults.get("noise_std", 0.0)),
                key=f"{kb}_noise",
            )
            p["clip_lo"] = st.number_input(
                f"{prefix} clip_lo",
                value=float(defaults.get("clip_lo", 0.0)),
                key=f"{kb}_lcliplo",
            )
            p["clip_hi"] = st.number_input(
                f"{prefix} clip_hi",
                value=float(defaults.get("clip_hi", 10.0)),
                key=f"{kb}_lcliphi",
            )

        elif mode == "seasonal":
            p["base"] = st.number_input(
                f"{prefix} base",
                value=float(defaults.get("base", 1.0)),
                key=f"{kb}_sbase",
            )
            p["amp"] = st.number_input(
                f"{prefix} amp",
                value=float(defaults.get("amp", 0.10)),
                step=0.01,
                key=f"{kb}_samp",
            )
            p["period"] = st.number_input(
                f"{prefix} period",
                value=float(defaults.get("period", 30.0)),
                key=f"{kb}_speriod",
            )
            p["phase"] = st.number_input(
                f"{prefix} phase",
                value=float(defaults.get("phase", 0.0)),
                key=f"{kb}_sphase",
            )
            p["noise_std"] = st.number_input(
                f"{prefix} noise_std",
                value=float(defaults.get("noise_std", 0.0)),
                key=f"{kb}_snoise",
            )
            p["clip_lo"] = st.number_input(
                f"{prefix} clip_lo",
                value=float(defaults.get("clip_lo", 0.0)),
                key=f"{kb}_scliplo",
            )
            p["clip_hi"] = st.number_input(
                f"{prefix} clip_hi",
                value=float(defaults.get("clip_hi", 1e6)),
                key=f"{kb}_scliphi",
            )

        return p

    # use keyed ui_params
    c_params = ui_params("c", c_mode, {"value": 2.0}, key_base="cls_c")
    h_params = ui_params("h", h_mode, {"value": 0.4}, key_base="cls_h")

    # s_params special case for "tbo"

    if s_mode == "tbo":
        st.markdown("**Setup from TBO target**")

        # pick from quick presets or Custom
        tbo_opts = [("L = 1", 1.0), ("L = 2", 2.0), ("L = 4", 4.0), ("Custom", None)]
        tbo_label = st.selectbox(
            "Target TBO (L)",
            [lbl for lbl, _ in tbo_opts],
            index=1,  # default to L=2 like before
            key="cls_tbo_pick",
            help="Choose a preset or 'Custom' to enter any positive value",
        )
        pick_val = dict(tbo_opts)[tbo_label]
        if pick_val is None:
            tbo_L = st.number_input(
                "Custom TBO (L)",
                min_value=0.1,
                value=6.0,
                step=0.1,
                key="cls_tbo_L",
                help="Typical range 1–10; higher L ⇒ larger setup cost via 0.5·h·d·L²",
            )
        else:
            tbo_L = float(pick_val)

        tbo_jit = st.number_input(
            "Jitter (±%)",
            min_value=0.0,
            max_value=100.0,
            value=SETUP_TBO_JITTER_DEFAULT,
            step=1.0,
            key="cls_tbo_jit",
        )

        per_period = st.checkbox(
            "Per-period jitter",
            value=st.session_state.get("cls_tbo_per_period", True),  # ✅ default True
            key="cls_tbo_per_period",
            help="If on, TBO setup is stored as an array with one jittered value per period.",
        )

        s_params = {
            "L": float(tbo_L),
            "jitter_pct": float(tbo_jit),
            "per_period": bool(per_period),
        }
    else:
        s_params = ui_params(
            "s", s_mode, {"base": 80.0, "amp": 0.10, "period": 30.0}, key_base="cls_s"
        )

    batch_size = st.number_input(
        "Instances per class (batch size)",
        min_value=1,
        value=50,
        step=1,
        key="cls_batch",
    )
    seed_base = st.number_input(
        "Base seed for this class",
        min_value=0,
        value=int(SEED + 10000),
        step=1,
        key="cls_seed",
    )

    if st.button("Add class to queue"):
        st.session_state["classes"].append(
            {
                "name": c_name,
                "period": int(c_period),
                "cap_mode": (
                    "DemandBased"
                    if c_cap_mode.startswith("Demand-based")
                    else c_cap_mode
                ),
                "cap_params": cap_params,
                "cap_tight": cap_tight,  # None unless DemandBased
                "n_items": int(n_items),
                "dem_lo": int(c_dem_lo),
                "dem_hi": int(c_dem_hi),
                "m_lo": int(c_m_lo),
                "m_hi": int(c_m_hi),
                "zero_head": int(zero_head),  # <-- new
                "c_mode": c_mode,
                "c_params": c_params,
                "h_mode": h_mode,
                "h_params": h_params,
                "s_mode": s_mode,
                "s_params": s_params,
                "batch_size": int(batch_size),
                "seed_base": int(seed_base),
                "allow_unmet_demand": bool(cls_allow_unmet),
                "lost_sales_penalty_factor": float(cls_ls_factor),
            }
        )

    dedupe_queue_by_name()

    st.subheader("Queue & Order")

    # stable labels (no numeric prefixes)
    labels = [
        f"{c['name']} · T={c['period']} · items={c['n_items']} · batch={c['batch_size']}"
        for c in st.session_state["classes"]
    ]
    label_to_class = {lbl: cls for lbl, cls in zip(labels, st.session_state["classes"])}

    try:
        from streamlit_sortables import sort_items  # type: ignore

        # dynamic key forces refresh when labels change
        sort_key = f"class_sort_{hash(tuple(labels))}"
        ordered_labels = sort_items(labels, direction="vertical", key=sort_key)

        # rebuild queue from the *labels* we just got back
        st.session_state["classes"] = [
            label_to_class[lbl] for lbl in ordered_labels if lbl in label_to_class
        ]
        st.caption("Drag-and-drop ordering active.")
    except Exception:
        st.info(
            "Install streamlit-sortables for drag-and-drop: pip install streamlit-sortables"
        )

    def _reset_sortables_state():
        for k in list(st.session_state.keys()):
            if k.startswith("class_sort_"):
                st.session_state.pop(k, None)

    # ---------- Batch override (this run only) ----------
    st.session_state.setdefault("run_batch_override_enabled", False)
    st.session_state.setdefault("run_batch_override_value", 1)

    oc1, oc2 = st.columns([1, 1])
    with oc1:
        # Do NOT assign to st.session_state here; just use a key
        st.checkbox(
            "Use batch override (this run only)",
            help="Apply one batch size to every queued class for the next run, without changing class specs.",
            key="run_batch_override_enabled",
        )
    with oc2:
        # Same: widget controls the state via the key
        st.number_input(
            "Override batch size",
            min_value=1,
            step=1,
            help="How many instances per class to run (temporary).",
            key="run_batch_override_value",
        )

    override_enabled = bool(st.session_state["run_batch_override_enabled"])
    override_value = int(st.session_state["run_batch_override_value"])

    # Quick table view  ✅ editable when override is OFF; locked when ON
    if st.session_state.get("classes"):
        dfq = pd.DataFrame(
            [
                {
                    "name": c["name"],
                    "T": c["period"],
                    "items": c["n_items"],
                    "batch": int(c.get("batch_size", 1)),
                }
                for c in st.session_state["classes"]
            ]
        )

        # When override is ON, show the effective value in the table (read-only)
        data_to_show = dfq.assign(batch=override_value) if override_enabled else dfq

        edited_dfq = st.data_editor(
            data_to_show,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "name": st.column_config.TextColumn(disabled=True),
                "T": st.column_config.NumberColumn(format="%d", disabled=True),
                "items": st.column_config.NumberColumn(format="%d", disabled=True),
                "batch": st.column_config.NumberColumn(
                    "batch",
                    help=(
                        "Instances per class (batch size). "
                        "This column is temporarily ignored when the override is enabled."
                    ),
                    min_value=1,
                    step=1,
                    format="%d",
                    disabled=override_enabled,  # lock when override is active
                ),
            },
            key="queue_editor",
        )

        # ---- Persist edits or apply override ----
        if override_enabled:
            # Mutate every queued class's batch_size, then rerun so UI reflects it
            changed = False
            for c in st.session_state["classes"]:
                if int(c.get("batch_size", 1)) != override_value:
                    c["batch_size"] = override_value
                    changed = True
            # Handy mapping for downstream code (if any uses it)
            st.session_state["effective_batch_by_name"] = {
                c["name"]: int(c.get("batch_size", 1))
                for c in st.session_state["classes"]
            }
            if changed:
                st.toast(
                    f"Applied batch size {override_value} to all queued classes.",
                    icon="✅",
                )
                st.rerun()
            else:
                st.info(
                    f"Batch override active: running **{override_value}** instance(s) per class."
                )
        else:
            # Persist user edits from the table back to class specs
            try:
                batches_by_name = {
                    row["name"]: int(row["batch"])
                    for _, row in edited_dfq.iterrows()
                    if pd.notnull(row["batch"])
                }
                for c in st.session_state["classes"]:
                    if c["name"] in batches_by_name:
                        c["batch_size"] = max(1, batches_by_name[c["name"]])
            except Exception:
                pass

            # Reflect the persisted values as the effective mapping
            st.session_state["effective_batch_by_name"] = {
                c["name"]: int(c.get("batch_size", 1))
                for c in st.session_state["classes"]
            }
    # Remove/clear controls
    rm_names = st.multiselect(
        "Select queued classes to remove",
        options=[c["name"] for c in st.session_state.get("classes", [])],
        key="rm_from_queue",
    )
    c_rm, c_clr, c_save = st.columns(3)
    with c_rm:
        if st.button("🗑️ Remove selected from queue"):
            before = len(st.session_state.get("classes", []))
            st.session_state["classes"] = [
                c
                for c in st.session_state.get("classes", [])
                if c["name"] not in rm_names
            ]
            _reset_sortables_state()
            st.success(
                f"Removed {before - len(st.session_state['classes'])} class(es)."
            )

    with c_clr:
        if st.button("🧹 Clear queue"):
            st.session_state["classes"] = []
            _reset_sortables_state()
            st.success("Queue cleared.")

    with c_save:
        sb_for_classes = supabase_client()
        if sb_for_classes and st.button(
            "💾 Save queued classes to Supabase", key="save_to_db"
        ):
            dedupe_queue_by_name()
            saved = 0
            for cls in st.session_state.get("classes", []):
                if ensure_class_row(sb_for_classes, cls):  # upsert by name
                    saved += 1
            _reset_sortables_state()
            st.success(f"Saved/updated {saved} unique class spec(s) to Supabase.")


# ----------------- Batch Runner tab -----------------
with batch_tab:
    st.header("Batch Runner (generate → solve → log)")
    sb = supabase_client()
    if sb is None:
        st.warning("Supabase not configured (see sidebar). Logging will be skipped.")

    # Prepare class id cache so we can stamp instances/runs
    class_id_cache = {}
    if sb is not None:
        for cls in st.session_state["classes"]:
            class_id_cache[cls["name"]] = ensure_class_row(sb, cls)

    # -----------------------------------------------------------------------------

    if st.button("Run queued classes"):
        total_runs = sum(
            int(c.get("batch_size", 0)) for c in st.session_state["classes"]
        )
        if total_runs <= 0:
            st.error(
                "No classes queued (or batch sizes are zero). Add presets or queue classes first."
            )
            st.stop()

        prog = st.progress(0.0, text="Running batches...")
        done = 0

        # sanity log
        st.write(f"Queued classes: {[c['name'] for c in st.session_state['classes']]}")
        st.write(f"Total runs: {total_runs}")

        for cls_idx, cls in enumerate(st.session_state["classes"], start=1):
            st.write(
                f"### Class {cls_idx}/{len(st.session_state['classes'])}: {cls['name']} (T={cls['period']}, items={cls['n_items']})"
            )
            for j in range(int(cls["batch_size"])):
                st.write(f"- Instance {j+1}/{cls['batch_size']} for '{cls['name']}'")
                inst = generate_instance_from_class(cls, j)

                # write instance (debug visibility)
                Path("last_instance.json").write_text(
                    json.dumps(to_py(inst), indent=2), encoding="utf-8"
                )

                # solve
                _solve_fn, _solver_tag = get_solver_backend()
                summary, orders_txt = _solve_fn(
                    "last_instance.json", time_limit=time_limit, mip_gap=mip_gap
                )

                # log to Supabase (if configured)
                sb = supabase_client()
                if sb is not None:
                    try:
                        cid = class_id_cache.get(cls["name"])

                        # use the in-memory dict 'inst', not 'inst_json'
                        inst_payload = {
                            "period": int(inst["period"]),
                            "manual_capacity": inst.get("manual_capacity"),
                            "warehouse_capacity": inst.get("warehouse_capacity"),
                            "data": to_py(inst),
                        }
                        if cid:
                            inst_payload["class_id"] = cid
                            inst["meta"]["class_id"] = cid  # optional

                        inst_payload = sanitize_json(inst_payload)
                        inst_res = sb.table("instances").insert(inst_payload).execute()
                        instance_id = inst_res.data[0]["id"]

                        run_payload = {
                            "instance_id": instance_id,
                            "time_limit_sec": int(time_limit),
                            "mip_gap": _safe_float(mip_gap),
                            "status": (
                                int(summary.get("status"))
                                if summary.get("status") is not None
                                else None
                            ),
                            "objective": _safe_float(summary.get("objective")),
                            "best_bound": _safe_float(summary.get("best_bound")),
                            "gap": _safe_float(summary.get("gap")),
                            "runtime_sec": _safe_float(summary.get("runtime_sec")),
                            "solver_version": _solver_tag,
                        }
                        if cid:
                            run_payload["class_id"] = cid

                        run_payload = sanitize_json(run_payload)
                        run_res = sb.table("runs").insert(run_payload).execute()
                        if run_res.data and len(run_res.data):
                            run_id = run_res.data[0]["id"]
                        else:
                            raise RuntimeError("Insert returned no data")
                        run_id = run_res.data[0]["id"]

                        # orders
                        rows = [
                            {"run_id": run_id, **r}
                            for r in parse_orders_lines(orders_txt)
                        ]
                        if rows:
                            CHUNK = 500
                            for k in range(0, len(rows), CHUNK):
                                sb.table("orders").insert(rows[k : k + CHUNK]).execute()

                    except Exception as e:
                        st.warning(f"Supabase logging failed: {e}")
                else:
                    st.info("Supabase disabled; skipping logging.")

                done += 1
                prog.progress(
                    done / total_runs, text=f"Running batches... {done}/{total_runs}"
                )

        st.success("Batch run complete.")

# ----------------- Run Inspector tab -----------------

with inspect_tab:
    st.header("Run Inspector (single run viewer)")
    sb = supabase_client()
    if sb is None:
        st.info("Configure Supabase in sidebar to inspect DB runs.")
    else:
        # ---- Class filter (NULL -> adhoc) ----
        db_map = st.session_state.get("db_classes", {})  # {name: row}
        class_opts = [
            ("All classes", "__ALL__"),
            ("adhoc (NULL class)", "__ADHOC__"),
        ]
        for nm in sorted(db_map.keys()):
            class_opts.append((nm, db_map[nm]["id"]))  # label -> class_id

        chosen_label = st.selectbox(
            "Filter by class", [x[0] for x in class_opts], index=0
        )
        label_to_val = {lbl: val for lbl, val in class_opts}
        chosen_val = label_to_val[chosen_label]

        limit_runs = st.number_input(
            "Fetch last N runs", min_value=10, value=300, step=50
        )

        # Fetch runs (filtered)
        select_cols = "id,created_at,instance_id,class_id,status,objective,best_bound,gap,runtime_sec,solver_version"
        q = (
            sb.table("runs")
            .select(select_cols)
            .order("created_at", desc=True)
            .limit(int(limit_runs))
        )
        if chosen_val == "__ADHOC__":
            q = q.is_("class_id", "null")
        elif chosen_val != "__ALL__":
            q = q.eq("class_id", chosen_val)
        runs_rows = q.execute().data or []

        # Status mapping (Gurobi-style)
        STATUS_MAP = {
            1: "LOADED",
            2: "OPTIMAL",
            3: "INFEASIBLE",
            4: "INF_OR_UNBD",
            5: "UNBOUNDED",
            6: "CUTOFF",
            7: "ITERATION_LIMIT",
            8: "NODE_LIMIT",
            9: "TIME_LIMIT",
            10: "SOLUTION_LIMIT",
            11: "INTERRUPTED",
            12: "NUMERIC",
            13: "SUBOPTIMAL",
            14: "INPROGRESS",
            15: "USER_OBJ_LIMIT",
        }

        def class_name_of(row):
            cid = row.get("class_id")
            if not cid:
                return "adhoc"
            for nm, r in db_map.items():
                if r.get("id") == cid:
                    return nm
            return f"{cid[:8]}…"

        recs = []
        for r in runs_rows:
            recs.append(
                {
                    "run_id": r.get("id"),
                    "created_at": r.get("created_at"),
                    "class_name": class_name_of(r),
                    "instance_id": r.get("instance_id"),
                    "status": r.get("status"),
                    "status_label": STATUS_MAP.get(
                        r.get("status"), str(r.get("status"))
                    ),
                    "objective": r.get("objective"),
                    "best_bound": r.get("best_bound"),
                    "gap": r.get("gap"),
                    "runtime_sec": r.get("runtime_sec"),
                    "solver_version": r.get("solver_version"),
                }
            )
        df_runs = pd.DataFrame(recs)
        if df_runs.empty:
            st.info("No runs found for the selected filter.")
            st.stop()

        # fetch ins_id for the displayed runs → add pretty 'instance' label (chunked to avoid 414)
        inst_ids = [x for x in df_runs["instance_id"].dropna().unique().tolist() if x]
        if inst_ids:
            meta_map = fetch_instances_meta_map(
                sb, inst_ids, cols="id,ins_id", chunk=150
            )
            id_to_insid = {
                iid: (meta_map[iid].get("ins_id") if iid in meta_map else None)
                for iid in inst_ids
            }
            df_runs["ins_id"] = df_runs["instance_id"].map(id_to_insid)
        else:
            df_runs["ins_id"] = None

        df_runs["instance"] = df_runs["ins_id"].apply(_ins_label)

        # -------- Add batch inference (≥ 1h gap → new batch) + batch filter --------
        try:
            df_sorted = df_runs.sort_values("created_at").copy()
            ts_sorted = pd.to_datetime(
                df_sorted["created_at"], utc=True, errors="coerce"
            )
            boundaries = ts_sorted.diff() > pd.Timedelta(hours=HOUR_GAP_FOR_BATCH)
            df_sorted["run_batch"] = (boundaries.cumsum() + 1).astype(int)
            batch_map = df_sorted.set_index("run_id")["run_batch"]
            df_runs["run_batch"] = df_runs["run_id"].map(batch_map)
        except Exception:
            df_runs["run_batch"] = 1  # fallback: single batch

        batch_values = sorted(df_runs["run_batch"].dropna().unique().tolist())
        batch_labels = ["ALL"] + [f"batch {b}" for b in batch_values]
        selected_batch_label = st.selectbox(
            "Filter by run batch", options=batch_labels, index=0
        )
        if selected_batch_label != "ALL":
            try:
                selected_batch = int(selected_batch_label.split()[-1])
                df_view = df_runs[df_runs["run_batch"] == selected_batch].copy()
            except Exception:
                df_view = df_runs.copy()
        else:
            df_view = df_runs.copy()

        if df_view.empty:
            st.info("No runs after applying the batch filter.")
            st.stop()

        # Single, clean table (no duplicate)
        runs_cols = [
            "run_id",
            "created_at",
            "instance",
            "class_name",
            "status_label",
            "solver_version",
            "objective",
            "best_bound",
            "gap",
            "runtime_sec",
            "run_batch",
        ]
        st.subheader("Filtered runs")
        st.dataframe(df_view[runs_cols], use_container_width=True)

        # ---- Pick a specific run ----
        pretty_opts = [
            f"{row['run_id']} • {row['created_at']} • {row['status_label']} • {row['solver_version']} • {row['instance']} • obj={row['objective']}"
            for _, row in df_view.iterrows()
        ]
        sel = st.selectbox("Select a run to inspect", pretty_opts, index=0)
        sel_idx = pretty_opts.index(sel)
        selected_run_id = df_view.iloc[sel_idx]["run_id"]

        # ---- Fetch the selected run, instance, and orders ----
        run_row = (
            sb.table("runs")
            .select("*")
            .eq("id", selected_run_id)
            .limit(1)
            .execute()
            .data
        )
        run_row = run_row[0] if run_row else None

        inst_row = None
        if run_row and run_row.get("instance_id"):
            inst_row = (
                sb.table("instances_enriched")
                .select("id,ins_id,period,manual_capacity,data,class_id,created_at")
                .eq("id", run_row["instance_id"])
                .limit(1)
                .execute()
                .data
            )
            inst_row = inst_row[0] if inst_row else None

        orders_rows = (
            sb.table("orders")
            .select("item_id,t,qty")
            .eq("run_id", selected_run_id)
            .order("item_id")
            .order("t")
            .execute()
            .data
            or []
        )
        df_orders = pd.DataFrame(orders_rows)
        if not df_orders.empty:
            # Ensure numeric types
            df_orders["item_id"] = pd.to_numeric(df_orders["item_id"], errors="coerce")
            df_orders["t"] = pd.to_numeric(df_orders["t"], errors="coerce")
            df_orders["qty"] = pd.to_numeric(df_orders["qty"], errors="coerce")

        # ---- Show run / instance details ----
        st.subheader("Run details")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Run ID", selected_run_id)
            st.metric(
                "Status",
                STATUS_MAP.get(run_row.get("status"), run_row.get("status")),
            )
            st.metric("Runtime (sec)", f"{run_row.get('runtime_sec')}")
        with c2:
            st.metric("Objective", f"{run_row.get('objective')}")
            st.metric("Best bound", f"{run_row.get('best_bound')}")
            st.metric("MIP gap", f"{run_row.get('gap')}")
            st.metric("Solver", run_row.get("solver_version", "—"))
        with c3:
            if inst_row:
                T_ = inst_row.get("period")
                cap = inst_row.get("manual_capacity") or (
                    inst_row.get("data") or {}
                ).get("manual_capacity")
                cap_mean = float(np.mean(cap)) if cap else None
                items_dict = (inst_row.get("data") or {}).get("items") or {}
                st.metric("Period (T)", f"{T_}")
                st.metric("Items", f"{len(items_dict)}")
                st.metric(
                    "Mean cap (κ̄)", f"{cap_mean if cap_mean is not None else '—'}"
                )
            else:
                st.caption("No instance row found.")

    # ==== Instance snapshot ====
    st.subheader("Instance snapshot")

    if not inst_row:
        st.info("No instance data tied to this run.")
    else:
        inst_data = inst_row.get("data") or {}
        T_inst = inst_row.get("period") or inst_data.get("period")
        items_dict = inst_data.get("items") or {}
        cap = (
            inst_row.get("manual_capacity")
            or inst_data.get("manual_capacity")
            or inst_data.get("production_capacity")
            or []
        )

        # --- Top chips / quick facts ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Items (|I|)", f"{len(items_dict)}")
        with c2:
            st.metric("Horizon (T)", f"{T_inst}")
        with c3:
            st.metric("Cap vector?", "Yes" if cap else "No")
        with c4:
            wh = inst_data.get("warehouse_capacity", None)
            st.metric("Warehouse cap W", "—" if wh is None else f"{wh}")

        # --- Build demand & shelf dataframes (items × t) ---
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go
        import plotly.express as px

        if T_inst is None or not items_dict:
            st.caption("Instance has no items or missing T.")
        else:
            # Demand matrix
            dem_rows = {}
            for i_str, it in items_dict.items():
                i = int(i_str)
                d = list(it.get("demand", [0] * int(T_inst)))
                d = (d + [0] * int(T_inst))[: int(T_inst)]
                dem_rows[i] = d
            df_dem = pd.DataFrame.from_dict(dem_rows, orient="index")
            df_dem.index.name = "item_id"
            df_dem.columns = list(range(int(T_inst)))

            # Shelf-life matrix
            shelf_rows = {}
            for i_str, it in items_dict.items():
                i = int(i_str)
                mseq = list(it.get("shelf_seq", [0] * int(T_inst)))
                mseq = (mseq + [0] * int(T_inst))[: int(T_inst)]
                shelf_rows[i] = mseq
            df_shelf = pd.DataFrame.from_dict(shelf_rows, orient="index")
            df_shelf.index.name = "item_id"
            df_shelf.columns = list(range(int(T_inst)))

            # Total demand per period and capacity series
            total_dem = df_dem.sum(axis=0)
            cap_series = pd.Series(cap, index=list(range(len(cap)))) if cap else None

            # Demand vs Capacity
            st.markdown("**Total demand vs capacity (per period)**")
            fig_dc = go.Figure()
            fig_dc.add_trace(
                go.Bar(
                    x=list(total_dem.index),
                    y=list(total_dem.values),
                    name="Total demand",
                )
            )
            if cap_series is not None and len(cap_series) >= len(total_dem):
                fig_dc.add_trace(
                    go.Scatter(
                        x=list(range(len(cap_series))),
                        y=list(cap_series.values),
                        mode="lines+markers",
                        name="Capacity κₜ",
                    )
                )
            fig_dc.update_layout(
                height=380,
                xaxis_title="t",
                yaxis_title="qty",
                legend_title="Series",
                barmode="overlay",
            )
            st.plotly_chart(fig_dc, use_container_width=True)

            # Per-item demand heatmap
            st.markdown("**Per-item demand heatmap**")
            if not df_dem.empty:
                fig_h1 = px.imshow(
                    df_dem,
                    labels=dict(x="t", y="item_id", color="demand"),
                    aspect="auto",
                )
                fig_h1.update_layout(height=420)
                st.plotly_chart(fig_h1, use_container_width=True)
            else:
                st.caption("No demand matrix to display.")

            # Per-item shelf-life heatmap
            st.markdown("**Per-item shelf-life (m_it) heatmap**")
            if not df_shelf.empty:
                fig_h2 = px.imshow(
                    df_shelf,
                    labels=dict(x="t", y="item_id", color="m_it"),
                    aspect="auto",
                )
                fig_h2.update_layout(height=420)
                st.plotly_chart(fig_h2, use_container_width=True)
            else:
                st.caption("No shelf-life matrix to display.")

            # Items overview
            st.markdown("**Items overview (quick stats)**")

            def _avg(x):
                x = np.asarray(x, dtype=float)
                return float(np.mean(x)) if x.size else np.nan

            rows = []
            for i_str, it in items_dict.items():
                i = int(i_str)
                d = df_dem.loc[i].values if i in df_dem.index else np.zeros(int(T_inst))
                m = (
                    df_shelf.loc[i].values
                    if i in df_shelf.index
                    else np.zeros(int(T_inst))
                )
                setup = it.get("setup", 0)
                cvar = it.get("c_var", 0)
                h = it.get("h", 0)

                def _mean_or_scalar(v):
                    if isinstance(v, list):
                        return _avg(v)
                    try:
                        return float(v)
                    except Exception:
                        return np.nan

                rows.append(
                    dict(
                        item_id=i,
                        dem_sum=float(np.sum(d)),
                        dem_max=float(np.max(d)) if d.size else 0.0,
                        dem_mean=_avg(d),
                        m_mean=_avg(m),
                        m_min=float(np.min(m)) if m.size else 0.0,
                        m_max=float(np.max(m)) if m.size else 0.0,
                        setup_mean=_mean_or_scalar(setup),
                        cvar_mean=_mean_or_scalar(cvar),
                        h_mean=_mean_or_scalar(h),
                    )
                )
            df_items = pd.DataFrame(rows).sort_values("dem_sum", ascending=False)
            st.dataframe(df_items, use_container_width=True)

            # Downloads
            colA, colB = st.columns([1, 1])
            with colA:
                if inst_data:
                    st.download_button(
                        "⬇️ Download instance JSON",
                        data=json.dumps(inst_data, indent=2),
                        file_name=f"instance_{inst_row.get('id','unknown')}.json",
                        mime="application/json",
                    )
            with colB:
                try:
                    import io

                    xbuf = io.BytesIO()
                    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as xlw:
                        df_items.to_excel(xlw, sheet_name="items_overview", index=False)
                        df_dem.to_excel(xlw, sheet_name="demand_matrix")
                        df_shelf.to_excel(xlw, sheet_name="shelf_life_mit")
                        if cap_series is not None:
                            pd.DataFrame({"kappa_t": cap_series}).to_excel(
                                xlw, sheet_name="capacity_kappa"
                            )
                    st.download_button(
                        "⬇️ Download instance snapshot (Excel)",
                        data=xbuf.getvalue(),
                        file_name=f"instance_{inst_row.get('id','unknown')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except Exception as _e:
                    st.caption(f"Excel export not available: {_e}")

            with st.expander("Raw `instances` row"):
                st.json(inst_row)

        # Orders: table & quick plots
        st.subheader("Order plan (per item)")
        if df_orders.empty:
            st.info("This run has no orders recorded.")
        else:
            pivot = df_orders.pivot_table(
                index="t", columns="item_id", values="qty", fill_value=0
            ).sort_index()
            st.dataframe(pivot, use_container_width=True)

            totals = (
                df_orders.groupby("item_id")["qty"].sum().sort_values(ascending=False)
            )
            default_items = list(totals.head(8).index.astype(int))
            items_to_plot = st.multiselect(
                "Plot selected item(s)",
                options=list(totals.index.astype(int)),
                default=default_items,
            )

            if items_to_plot:
                fig = go.Figure()
                for iid in items_to_plot:
                    ser = pivot.get(iid)
                    if ser is None:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=ser.index,
                            y=ser.values,
                            mode="lines+markers",
                            name=f"Item {iid}",
                        )
                    )
                fig.update_layout(
                    height=420,
                    title="Order quantities by period (selected items)",
                    xaxis_title="t",
                    yaxis_title="qty",
                    legend_title="Item",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Nonzero orders (long form)**")
            st.dataframe(
                df_orders[df_orders["qty"] > 0].sort_values(["item_id", "t"]),
                use_container_width=True,
            )


# ----------------- Saved Instances Runner tab -----------------

with saved_run_tab:
    st.header("Saved Instances Runner (Supabase)")

    sb = supabase_client()
    if sb is None:
        st.info("Configure Supabase in sidebar.")
        st.stop()

    # --- class filter (including NULL/adhoc) ---
    db_map = st.session_state.get("db_classes", {})
    class_choices = ["<ALL>", "<ADHOC (NULL)>"] + sorted(db_map.keys())
    pick_cls = st.selectbox("Filter by class", class_choices, index=0)

    # limit + refresh + page size + ALL
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        inst_limit = st.number_input(
            "Fetch last N instances",
            min_value=10,
            value=500,
            step=50,
            key="saved_run_limit",
        )
    with c2:
        page_size = st.number_input(
            "Page size",
            min_value=50,
            value=200,
            step=50,
            help="Smaller = safer; larger = faster",
            key="saved_run_pagesize",
        )
    with c3:
        load_all = st.checkbox(
            "Load ALL (keyset)",
            value=False,
            help="Fetches every matching instance by paging until empty.",
        )
    with c4:
        if st.button("🔄 Refresh instances"):
            st.rerun()

    # -------- helpers --------
    def _class_of(row):
        if row.get("class_id"):
            return next(
                (n for n, v in db_map.items() if v["id"] == row["class_id"]), "adhoc"
            )
        return "adhoc"

    def _n_items(row):
        v = row.get("n_items")
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return None

    # Fast keyset pagination over base table; then backfill ins_id/n_items from the view.
    def fetch_instances_keyset(
        sb_client: Client, total: int | None, class_name: str | None, page_sz: int = 200
    ):
        out, last_id = [], None

        # map class name -> id once
        cls_id = None
        if class_name and class_name in db_map:
            cls_id = db_map[class_name]["id"]

        base_select = "id,created_at,period,class_id"
        fetched = 0
        while True:
            need = page_sz if (total is None) else min(page_sz, max(0, total - fetched))
            if need == 0:
                break

            q = sb_client.table("instances").select(base_select).order("id", desc=True)
            if last_id is not None:
                q = q.lt("id", last_id)  # keyset by id

            if class_name == "<ADHOC (NULL)>":
                q = q.is_("class_id", None)
            elif cls_id:
                q = q.eq("class_id", cls_id)

            batch = q.limit(need).execute().data or []
            if not batch:
                break
            out.extend(batch)
            fetched += len(batch)
            last_id = batch[-1]["id"]

            # if total is None we keep going until empty; otherwise stop when we hit the cap
            if total is not None and fetched >= total:
                break

        # backfill ins_id & n_items cheaply from the view
        if out:
            ids = [r["id"] for r in out]
            CH = 400
            meta_map = {}
            for k in range(0, len(ids), CH):
                part = ids[k : k + CH]
                vv = (
                    sb_client.table("instances_enriched")
                    .select("id,ins_id")  # ← drop n_items here
                    .in_("id", part)
                    .execute()
                    .data
                    or []
                )
                for v in vv:
                    meta_map[v["id"]] = {"ins_id": v.get("ins_id")}

            for r in out:
                m = meta_map.get(r["id"], {})
                r["ins_id"] = m.get("ins_id")
                r["n_items"] = None  # ← safe default (not shown)
        return out

    # tiny helper: fetch JSON for a list of instance ids (chunked)
    # def fetch_instances_json_map(sb_client: Client, ids: list[str]) -> dict[str, dict]:
    #     m: dict[str, dict] = {}
    #     if not ids:
    #         return m
    #     CH = 200
    #     for k in range(0, len(ids), CH):
    #         sub = ids[k : k + CH]
    #         part = (
    #             sb_client.table("instances")
    #             .select("id,data")
    #             .in_("id", sub)
    #             .execute()
    #             .data
    #             or []
    #         )
    #         for row in part:
    #             m[row["id"]] = row.get("data") or {}
    #     return m
    def fetch_instances_json_map(sb_client: Client, ids: list[str]) -> dict[str, dict]:
        """
        Safer fetch for instance JSON:
        - For very small selections (<=5), fetch one-by-one with eq('id', ...) to avoid slow IN(...) plans.
        - For larger selections, use small chunks (50).
        """
        out: dict[str, dict] = {}
        if not ids:
            return out

        # very small: fetch individually (fast, avoids statement_timeout)
        if len(ids) <= 5:
            for iid in ids:
                try:
                    row = (
                        sb_client.table("instances")
                        .select("id,data")
                        .eq("id", iid)
                        .limit(1)
                        .execute()
                        .data
                        or []
                    )
                    if row:
                        out[iid] = row[0].get("data") or {}
                except Exception as e:
                    st.warning(f"JSON fetch failed for {iid[:8]}…: {e}")
            return out

        # larger: tiny IN() batches
        CH = 50
        for k in range(0, len(ids), CH):
            sub = ids[k : k + CH]
            try:
                part = (
                    sb_client.table("instances")
                    .select("id,data")
                    .in_("id", sub)
                    .execute()
                    .data
                    or []
                )
                for row in part:
                    out[row["id"]] = row.get("data") or {}
            except Exception as e:
                # fall back to per-id if a chunk still times out
                for iid in sub:
                    try:
                        row = (
                            sb_client.table("instances")
                            .select("id,data")
                            .eq("id", iid)
                            .limit(1)
                            .execute()
                            .data
                            or []
                        )
                        if row:
                            out[iid] = row[0].get("data") or {}
                    except Exception as e2:
                        st.warning(f"JSON fetch failed for {iid[:8]}…: {e2}")
        return out

    # -------- fetch rows (ALL or limited) --------
    class_name_filter = None if pick_cls == "<ALL>" else pick_cls
    with st.spinner("Loading instances..."):
        if load_all:
            instances = fetch_instances_keyset(
                sb, None, class_name_filter, page_sz=int(page_size)
            )
        else:
            instances = fetch_instances_keyset(
                sb, int(inst_limit), class_name_filter, page_sz=int(page_size)
            )

    if not instances:
        st.info("No instances match the filter.")
        st.stop()

    # -------- present table --------
    dfI = pd.DataFrame(
        [
            {
                "instance_id": r["id"],
                "ins_id": r.get("ins_id"),
                "instance": _ins_label(r.get("ins_id")),
                "created_at": r["created_at"],
                "class": _class_of(r),
                "period": r.get("period"),
                "n_items": _n_items(r),
            }
            for r in instances
        ]
    )
    st.dataframe(
        dfI[["instance", "created_at", "class", "period", "n_items"]],
        use_container_width=True,
        height=320,
    )

    # -------- choose which instances to run (pretty labels) --------
    opt_pairs = [
        (f"{_ins_label(r.get('ins_id'))} × {_class_of(r)}", r["id"]) for r in instances
    ]
    label_to_id = {lbl: iid for lbl, iid in opt_pairs}

    pick_labels = st.multiselect(
        "Pick specific instances (leave empty to use the per-class limit below)",
        options=[lbl for lbl, _ in opt_pairs],
    )

    # NEW: per-class cap (only used when none are explicitly selected)
    per_class_mode = st.radio(
        "When none are explicitly selected, run…",
        ["All loaded", "1 per class", "2 per class"],
        horizontal=True,
        key="saved_run_per_class_mode",
    )

    all_ids = dfI["instance_id"].tolist()
    pick_ids = [label_to_id[lbl] for lbl in pick_labels] if pick_labels else []

    # choose one or more solvers to run
    multi_solver_labels = st.multiselect(
        "Solvers to run",
        options=list(SOLVER_REGISTRY.keys()),
        default=[
            st.session_state.get("solver_backend_label") or "LEFO v2 (permission-based)"
        ],
    )
    if not multi_solver_labels:
        st.warning("Pick at least one solver.")
        st.stop()

    run_btn = st.button("🚀 Run selected instances")

    if run_btn:
        if pick_ids:
            ids_to_run = pick_ids
        else:
            mode = st.session_state.get("saved_run_per_class_mode", "All loaded")
            if mode == "All loaded":
                ids_to_run = all_ids
            else:
                n = 1 if mode.startswith("1") else 2
                # newest first per class; then take top-n per class
                df_sorted = dfI.sort_values("created_at", ascending=False)
                ids_to_run = []
                for _, grp in df_sorted.groupby("class", sort=False):
                    ids_to_run.extend(grp["instance_id"].head(n).tolist())

        id_set = set(ids_to_run)
        rows = [r for r in instances if r["id"] in id_set]
        total = len(rows) * len(multi_solver_labels)
        if total <= 0:
            st.info("Nothing to run with current selections.")
            st.stop()

        # fetch JSON for selected instances once (chunked)
        inst_json_map = fetch_instances_json_map(sb, ids_to_run)

        prog = st.progress(0.0, text="Running saved instances...")
        done = 0
        created_run_ids: list[str] = []

        from datetime import datetime, timezone, timedelta

        for r in rows:
            inst_json = inst_json_map.get(r["id"]) or {}
            # write temp json for solver I/O
            tmp_path = Path("tmp_instance_saved.json")
            tmp_path.write_text(
                json.dumps(to_py(inst_json), indent=2), encoding="utf-8"
            )

            for label in multi_solver_labels:
                entry = SOLVER_REGISTRY.get(label) or SOLVER_REGISTRY.get(
                    "LEFO v2 (permission-based)", {}
                )
                solve_fn = (entry or {}).get("fn")
                solver_tag = (entry or {}).get("tag", "unknown")
                if solve_fn is None:
                    st.warning(f"Skipping '{label}': backend not importable.")
                    done += 1
                    prog.progress(
                        done / total, text=f"Running saved instances... {done}/{total}"
                    )
                    continue

                # run solver
                summary, orders_txt = solve_fn(
                    str(tmp_path), time_limit=time_limit, mip_gap=mip_gap
                )

                # log run + orders
                try:
                    run_payload = sanitize_json(
                        {
                            "instance_id": r["id"],
                            "class_id": r.get("class_id"),
                            "time_limit_sec": int(time_limit),
                            "mip_gap": _safe_float(mip_gap),
                            "status": (
                                int(summary.get("status"))
                                if summary.get("status") is not None
                                else None
                            ),
                            "objective": _safe_float(summary.get("objective")),
                            "best_bound": _safe_float(summary.get("best_bound")),
                            "gap": _safe_float(summary.get("gap")),
                            "runtime_sec": _safe_float(summary.get("runtime_sec")),
                            "solver_version": solver_tag,
                        }
                    )

                    # lower bound on created_at (UTC) BEFORE insert
                    t0 = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()

                    # insert + robust fallback to select just-inserted id
                    run_res = None
                    run_id = None
                    try:
                        run_res = (
                            sb.table("runs")
                            .insert(run_payload, returning="representation")
                            .execute()
                        )
                    except TypeError:
                        run_res = sb.table("runs").insert(run_payload).execute()

                    if run_res and getattr(run_res, "data", None):
                        try:
                            run_id = run_res.data[0]["id"]
                        except Exception:
                            run_id = None

                    if not run_id:
                        try:
                            sel = (
                                sb.table("runs")
                                .select("id,created_at")
                                .eq("instance_id", r["id"])
                                .eq("solver_version", solver_tag)
                                .eq("time_limit_sec", int(time_limit))
                                .eq("mip_gap", _safe_float(mip_gap))
                                .gte("created_at", t0)
                                .order("created_at", desc=True)
                                .limit(1)
                                .execute()
                            )
                            if sel and sel.data:
                                run_id = sel.data[0]["id"]
                        except Exception:
                            run_id = None

                    if not run_id:
                        raise RuntimeError(
                            "Insert succeeded but no run_id returned; could not resolve via fallback select."
                        )

                    created_run_ids.append(run_id)

                    # parse orders and chunk insert
                    rows_ord = [
                        {"run_id": run_id, **row}
                        for row in parse_orders_lines(orders_txt)
                    ]
                    if rows_ord:
                        CH = 500
                        for k in range(0, len(rows_ord), CH):
                            sb.table("orders").insert(rows_ord[k : k + CH]).execute()

                except Exception as e:
                    st.warning(
                        f"Supabase logging failed (instance {r['id']} / {label}): {e}"
                    )

                done += 1
                prog.progress(
                    done / total, text=f"Running saved instances... {done}/{total}"
                )

        st.success("Saved instances run complete.")

        # ============  “batch” logic copied from Explore (≥ 1h gaps) ============
        try:
            if created_run_ids:
                # 1) Pull just-created runs
                select_cols = (
                    "id,created_at,instance_id,status,objective,best_bound,gap,"
                    "runtime_sec,solver_version"
                )
                runs_rows = []
                CH = 500
                for k in range(0, len(created_run_ids), CH):
                    sub = created_run_ids[k : k + CH]
                    part = (
                        sb.table("runs")
                        .select(select_cols)
                        .in_("id", sub)
                        .order("created_at", desc=False)
                        .execute()
                        .data
                        or []
                    )
                    runs_rows.extend(part)

                if runs_rows:
                    # 2) Bring in instance metadata (for class_key, ins_id)
                    inst_ids = list(
                        {
                            r.get("instance_id")
                            for r in runs_rows
                            if r.get("instance_id")
                        }
                    )
                    inst_map = {}
                    if inst_ids:
                        for k in range(0, len(inst_ids), CH):
                            part = inst_ids[k : k + CH]
                            inst_rows = (
                                sb.table("instances_enriched")
                                .select(
                                    "id,ins_id,period,manual_capacity,data,class_id"
                                )
                                .in_("id", part)
                                .execute()
                                .data
                                or []
                            )
                            for row in inst_rows:
                                inst_map[row["id"]] = row

                    # 3) Build df like Explore
                    STATUS_MAP = {
                        1: "LOADED",
                        2: "OPTIMAL",
                        3: "INFEASIBLE",
                        4: "INF_OR_UNBD",
                        5: "UNBOUNDED",
                        6: "CUTOFF",
                        7: "ITERATION_LIMIT",
                        8: "NODE_LIMIT",
                        9: "TIME_LIMIT",
                        10: "SOLUTION_LIMIT",
                        11: "INTERRUPTED",
                        12: "NUMERIC",
                        13: "SUBOPTIMAL",
                        14: "INPROGRESS",
                        15: "USER_OBJ_LIMIT",
                    }

                    recs = []
                    for rrow in runs_rows:
                        I = inst_map.get(rrow.get("instance_id"))
                        data = (I or {}).get("data") or {}
                        meta = data.get("meta") or {}
                        class_key = meta.get("class_key", "adhoc")
                        ins_id = (I or {}).get("ins_id")
                        recs.append(
                            {
                                "run_id": rrow.get("id"),
                                "created_at": rrow.get("created_at"),
                                "instance_id": rrow.get("instance_id"),
                                "instance": _ins_label(ins_id),
                                "class_key": class_key,
                                "status": rrow.get("status"),
                                "status_label": STATUS_MAP.get(
                                    rrow.get("status"), str(rrow.get("status"))
                                ),
                                "solver_version": rrow.get("solver_version")
                                or "unknown",
                                "objective": rrow.get("objective"),
                                "best_bound": rrow.get("best_bound"),
                                "gap": rrow.get("gap"),
                                "runtime_sec": rrow.get("runtime_sec"),
                            }
                        )
                    df_new = pd.DataFrame(recs)

                    # 4) Infer run_batch EXACTLY like Explore (≥2h gap → new batch)
                    df_sorted = df_new.sort_values("created_at").copy()
                    ts_sorted = pd.to_datetime(
                        df_sorted["created_at"], utc=True, errors="coerce"
                    )
                    boundaries = ts_sorted.diff() > pd.Timedelta(hours=2)
                    df_sorted["run_batch"] = (boundaries.cumsum() + 1).astype(int)
                    batch_map = df_sorted.set_index("run_id")["run_batch"]
                    df_new["run_batch"] = df_new["run_id"].map(batch_map)

                    # 5) Show quick batch summary for just-created runs
                    st.subheader("Batch summary (this run)")
                    cols_show = [
                        "run_id",
                        "created_at",
                        "instance",
                        "class_key",
                        "status_label",
                        "solver_version",
                        "runtime_sec",
                        "run_batch",
                    ]
                    st.dataframe(
                        df_new[cols_show].sort_values(["run_batch", "created_at"]),
                        use_container_width=True,
                    )

                    st.markdown("**Status counts by solver & batch**")
                    cnt = (
                        df_new.groupby(
                            ["solver_version", "status_label", "run_batch"],
                            dropna=False,
                        )
                        .size()
                        .reset_index(name="runs")
                        .sort_values(["run_batch", "solver_version", "status_label"])
                    )
                    st.dataframe(cnt, use_container_width=True)
        except Exception as e:
            st.caption(f"Batch summary unavailable: {e}")

    # === NEW: X-coded tools (generate only, queue & run, download) ===
    with st.expander(
        "X-coded tools: generate instances (no solve), queue & run, download",
        expanded=False,
    ):
        import re, io, zipfile

        # --- Local, non-intrusive helper: parse X-codes X A B C D E F (F can be 1..12) ---
        _x_pat_local = re.compile(r"^[Xx]([A-Za-z0-9]+)$")

        def _x_digits_local(name: str):
            m = _x_pat_local.match(str(name))
            if not m:
                return None
            s = m.group(1)
            if len(s) < 6:
                return None
            A, B, C, D, E = s[0], s[1], s[2], s[3], s[4]
            F = s[5:]
            if A not in {"1", "2", "3"}:
                return None
            if B not in {"1", "2", "3"}:
                return None
            if C not in {"1", "2"}:
                return None
            if D not in {"1", "2", "L", "H"}:
                return None
            if E not in {"A", "B", "C", "1", "2", "3", "4", "5"}:
                return None
            if not (F.isdigit() and 1 <= int(F) <= 12):
                return None
            return (A, B, C, D, E, F)

        # --- Gather all X-coded class names present in DB ---
        x_names_all = [n for n in db_map.keys() if _x_digits_local(n)]
        st.caption(f"Found **{len(x_names_all)}** X-coded classes in Supabase.")

        # --- A..F include/exclude filters (persist in session) ---
        POS = ["A", "B", "C", "D", "E", "F"]
        CHOICES = {
            "A": [str(i) for i in (1, 2, 3)],
            "B": [str(i) for i in (1, 2, 3)],
            "C": [str(i) for i in (1, 2)],
            "D": ["1", "2", "L", "H"],
            "E": ["A", "B", "C", "1", "2", "3", "4", "5"],
            "F": [str(i) for i in range(1, 13)],
        }
        for p in POS:
            st.session_state.setdefault(f"x_inc_{p}_saved", [])
            st.session_state.setdefault(f"x_exc_{p}_saved", [])

        ccols = st.columns(6)
        for idx, p in enumerate(POS):
            with ccols[idx]:
                st.session_state[f"x_inc_{p}_saved"] = st.multiselect(
                    f"Include {p}",
                    options=CHOICES[p],
                    default=st.session_state[f"x_inc_{p}_saved"],
                    key=f"x_inc_ms_saved_{p}",
                )
                st.session_state[f"x_exc_{p}_saved"] = st.multiselect(
                    f"Exclude {p}",
                    options=CHOICES[p],
                    default=st.session_state[f"x_exc_{p}_saved"],
                    key=f"x_exc_ms_saved_{p}",
                )

        def _match_x_filters(name: str) -> bool:
            tup = _x_digits_local(name)
            if not tup:
                return False
            for pos_idx, p in enumerate(POS):
                d = tup[pos_idx]
                inc = set(st.session_state.get(f"x_inc_{p}_saved", []))
                exc = set(st.session_state.get(f"x_exc_{p}_saved", []))
                if inc and d not in inc:
                    return False
                if d in exc:
                    return False
            return True

        x_selected = [n for n in x_names_all if _match_x_filters(n)]
        st.write(f"**Selected X classes:** {len(x_selected)}")
        if x_selected:
            st.caption(
                ", ".join(sorted(x_selected)[:12])
                + (" …" if len(x_selected) > 12 else "")
            )

        st.markdown("---")

        # ========== 1) Generate instances for selected X-classes (no solve) ==========
        gen_n = st.number_input(
            "Instances per selected class to GENERATE (no solve)",
            min_value=1,
            value=5,
            step=1,
            key="x_gen_n",
        )
        if st.button("➕ Generate to Supabase (no solve)", key="btn_x_gen"):
            if not x_selected:
                st.warning("No X classes selected by filters.")
            else:
                total = len(x_selected) * int(gen_n)
                prog = st.progress(0.0, text="Generating...")
                done, saved = 0, 0
                for cname in x_selected:
                    cls_row = db_map.get(cname)
                    if not cls_row:
                        continue
                    cls_spec = cls_row.get("spec") or {}
                    class_id = cls_row.get("id")
                    # use time-based j offset to avoid deterministic duplicates on repeated clicks
                    j_base = int(time.time())
                    for j in range(int(gen_n)):
                        try:
                            inst = generate_instance_from_class(cls_spec, j_base + j)
                            payload = sanitize_json(
                                {
                                    "period": int(inst["period"]),
                                    "manual_capacity": inst.get("manual_capacity"),
                                    "warehouse_capacity": inst.get(
                                        "warehouse_capacity"
                                    ),
                                    "data": to_py(inst),
                                    "class_id": class_id,
                                }
                            )
                            sb.table("instances").insert(payload).execute()
                            saved += 1
                        except Exception as e:
                            st.warning(f"[{cname}] insert failed (j={j}): {e}")
                        finally:
                            done += 1
                            prog.progress(
                                done / total, text=f"Generating... {done}/{total}"
                            )
                st.success(
                    f"Generated and saved **{saved}** instance(s). Click 'Refresh instances' above to see them."
                )

        st.markdown("---")

        # Helper: fetch ALL instances for a list of class_ids (keyset pagination)
        def _fetch_all_instances_for_class_ids(
            sb_client: Client, class_ids: list[str]
        ) -> list[dict]:
            out = []
            for cid in class_ids:
                last_seen = None
                while True:
                    q = (
                        sb_client.table("instances_enriched")
                        .select("id,ins_id,created_at,period,class_id,data")
                        .eq("class_id", cid)
                        .order("created_at", desc=True)
                        .limit(1000)
                    )
                    if last_seen is not None:
                        q = q.lt("created_at", last_seen)
                    batch = (q.execute().data) or []
                    if not batch:
                        break
                    out.extend(batch)
                    last_seen = batch[-1]["created_at"]
            return out

        # ========== 2) Queue & run all instances for selected X-classes ==========
        st.subheader("Run a batch for selected X-classes")
        x_solver_labels = st.multiselect(
            "Solvers to run (X-batch)",
            options=list(SOLVER_REGISTRY.keys()),
            default=[
                st.session_state.get("solver_backend_label")
                or "LEFO v2 (permission-based)"
            ],
            key="x_solver_labels",
        )
        x_run_source = st.radio(
            "Which instances to include?",
            [
                "Use instances loaded above",
                "Fetch ALL from Supabase for selected X classes",
            ],
            index=0,
            key="x_run_source",
        )

        # NEW: cap how many instances per class to run when nothing is explicitly picked
        x_run_sample_mode = st.radio(
            "When none are explicitly selected, run…",
            ["All loaded", "1 per class", "2 per class"],
            index=0,
            horizontal=True,
            key="x_run_sample_mode",
        )

        if st.button(
            "🚀 Run ALL instances for selected X classes", key="btn_x_run_all"
        ):
            if not x_selected:
                st.warning("No X classes selected by filters.")
            elif not x_solver_labels:
                st.warning("Pick at least one solver.")
            else:
                # --- How many per class should we run (when none explicitly selected)?
                cap_label = st.session_state.get("x_run_sample_mode", "All loaded")
                cap_per_class = {
                    "All loaded": 0,
                    "1 per class": 1,
                    "2 per class": 2,
                }.get(cap_label, 0)

                # --- Build rows_to_run, but DON'T fetch JSON yet ---
                if x_run_source.startswith("Use instances loaded"):
                    # Filter the already-loaded list by the selected X class names
                    rows_to_run = [
                        r for r in instances if _class_of(r) in set(x_selected)
                    ]
                    inst_json_map = None  # we'll fetch after we cap
                else:
                    # Pull from DB. If we only need 1/2 per class, fetch just that much per class.
                    class_ids = [db_map[n]["id"] for n in x_selected if n in db_map]
                    if cap_per_class:
                        rows_to_run = []
                        select_cols = "id,ins_id,created_at,period,class_id,data"
                        for cid in class_ids:
                            try:
                                batch = (
                                    sb.table("instances_enriched")
                                    .select(select_cols)
                                    .eq("class_id", cid)
                                    .order("created_at", desc=True)
                                    .limit(cap_per_class)
                                    .execute()
                                    .data
                                    or []
                                )
                                rows_to_run.extend(batch)
                            except Exception as e:
                                st.warning(f"Fetch failed for class_id={cid[:8]}…: {e}")
                        inst_json_map = {}  # rows already have 'data'
                    else:
                        # No cap → fetch all for those classes (could be large)
                        rows_to_run = _fetch_all_instances_for_class_ids(sb, class_ids)
                        inst_json_map = {}  # rows already have 'data'

                if not rows_to_run:
                    st.info(
                        "No instances found for the selected X classes with current source."
                    )
                else:
                    # --- Apply cap for the 'loaded' source (DB source was capped above) ---
                    if (
                        x_run_source.startswith("Use instances loaded")
                        and cap_per_class
                    ):
                        # newest first within each class (fallback to "" if missing)
                        try:
                            rows_to_run.sort(
                                key=lambda r: r.get("created_at") or "", reverse=True
                            )
                        except Exception:
                            pass
                        seen = {}
                        limited = []
                        for r in rows_to_run:
                            cname = _class_of(r)
                            if seen.get(cname, 0) < cap_per_class:
                                limited.append(r)
                                seen[cname] = seen.get(cname, 0) + 1
                        rows_to_run = limited

                    # --- Now that rows_to_run is SMALL, fetch JSON only for these (loaded-source) ---
                    if x_run_source.startswith("Use instances loaded"):
                        ids = [r["id"] for r in rows_to_run]
                        inst_json_map = fetch_instances_json_map(sb, ids)  # small, fast

                    # --- Run the solvers ---
                    total = len(rows_to_run) * len(x_solver_labels)
                    prog = st.progress(0.0, text="Running X-batch...")
                    done = 0
                    created_run_ids_x: list[str] = []
                    from datetime import datetime, timezone, timedelta

                    for r in rows_to_run:
                        # Prefer the (possibly None) map; DB source rows already carry 'data'
                        inst_json = (
                            (inst_json_map or {}).get(r["id"]) or r.get("data") or {}
                        )
                        if "period" not in inst_json:
                            st.warning(
                                f"Skipping instance {r.get('id')} (no JSON/period)."
                            )
                            done += 1
                            prog.progress(
                                done / total, text=f"Running X-batch... {done}/{total}"
                            )
                            continue

                        tmp_path = Path("tmp_instance_saved_x.json")
                        tmp_path.write_text(
                            json.dumps(to_py(inst_json), indent=2), encoding="utf-8"
                        )

                        for label in x_solver_labels:
                            entry = SOLVER_REGISTRY.get(label) or SOLVER_REGISTRY.get(
                                "LEFO v2 (permission-based)", {}
                            )
                            solve_fn = (entry or {}).get("fn")
                            solver_tag = (entry or {}).get("tag", "unknown")
                            if solve_fn is None:
                                st.warning(
                                    f"Skipping '{label}': backend not importable."
                                )
                                done += 1
                                prog.progress(
                                    done / total,
                                    text=f"Running X-batch... {done}/{total}",
                                )
                                continue

                            summary, orders_txt = solve_fn(
                                str(tmp_path), time_limit=time_limit, mip_gap=mip_gap
                            )

                            try:
                                run_payload = sanitize_json(
                                    {
                                        "instance_id": r["id"],
                                        "class_id": r.get("class_id"),
                                        "time_limit_sec": int(time_limit),
                                        "mip_gap": _safe_float(mip_gap),
                                        "status": (
                                            int(summary.get("status"))
                                            if summary.get("status") is not None
                                            else None
                                        ),
                                        "objective": _safe_float(
                                            summary.get("objective")
                                        ),
                                        "best_bound": _safe_float(
                                            summary.get("best_bound")
                                        ),
                                        "gap": _safe_float(summary.get("gap")),
                                        "runtime_sec": _safe_float(
                                            summary.get("runtime_sec")
                                        ),
                                        "solver_version": solver_tag,
                                    }
                                )

                                t0 = (
                                    datetime.now(timezone.utc) - timedelta(seconds=5)
                                ).isoformat()
                                run_res = None
                                run_id = None
                                try:
                                    run_res = (
                                        sb.table("runs")
                                        .insert(run_payload, returning="representation")
                                        .execute()
                                    )
                                except TypeError:
                                    run_res = (
                                        sb.table("runs").insert(run_payload).execute()
                                    )

                                if run_res and getattr(run_res, "data", None):
                                    try:
                                        run_id = run_res.data[0]["id"]
                                    except Exception:
                                        run_id = None

                                if not run_id:
                                    try:
                                        sel = (
                                            sb.table("runs")
                                            .select("id,created_at")
                                            .eq("instance_id", r["id"])
                                            .eq("solver_version", solver_tag)
                                            .eq("time_limit_sec", int(time_limit))
                                            .eq("mip_gap", _safe_float(mip_gap))
                                            .gte("created_at", t0)
                                            .order("created_at", desc=True)
                                            .limit(1)
                                            .execute()
                                        )
                                        if sel and sel.data:
                                            run_id = sel.data[0]["id"]
                                    except Exception:
                                        run_id = None

                                if not run_id:
                                    raise RuntimeError(
                                        "Insert succeeded but no run_id returned; fallback select failed."
                                    )

                                created_run_ids_x.append(run_id)

                                # orders insert
                                rows_ord = [
                                    {"run_id": run_id, **row}
                                    for row in parse_orders_lines(orders_txt)
                                ]
                                if rows_ord:
                                    CH = 500
                                    for k in range(0, len(rows_ord), CH):
                                        sb.table("orders").insert(
                                            rows_ord[k : k + CH]
                                        ).execute()

                            except Exception as e:
                                st.warning(
                                    f"Supabase logging failed (instance {r['id']} / {label}): {e}"
                                )

                            done += 1
                            prog.progress(
                                done / total, text=f"Running X-batch... {done}/{total}"
                            )

                    st.success("X-batch run complete.")

        st.markdown("---")

        # ========== 3) Download all chosen X instances as ZIP (CHOSEN/<class>/<file>.json) ==========
        st.subheader("Download selected X instances")

        # Extra options for saving locally and progress feedback
        save_to_disk = st.checkbox(
            "Also save CHOSEN.zip to local disk",
            value=False,
            help="Writes the ZIP beside your Streamlit script on the host machine.",
            key="x_zip_save_disk",
        )
        zip_filename_input = st.text_input(
            "Output filename",
            value="CHOSEN.zip",
            disabled=not save_to_disk,
            key="x_zip_filename",
        )

        x_dl_source = st.radio(
            "Which instances to include in ZIP?",
            [
                "Use instances loaded above",
                "Fetch ALL from Supabase for selected X classes",
            ],
            index=0,
            key="x_dl_source",
        )

        if st.button("⬇️ Build ZIP (CHOSEN.zip)", key="btn_x_zip"):
            import re, io, zipfile, time, unicodedata

            def _safe_name(s: str) -> str:
                # ASCII-ish and filesystem-safe (and short)
                s = unicodedata.normalize("NFKD", str(s))
                s = s.encode("ascii", "ignore").decode("ascii")
                s = re.sub(r"[^A-Za-z0-9_.-]", "_", s)
                return s[:120] or "file"

            if not x_selected:
                st.warning("No X classes selected by filters.")
            else:
                # Determine rows to include
                if x_dl_source.startswith("Use instances loaded"):
                    rows_src = [r for r in instances if _class_of(r) in set(x_selected)]
                else:
                    class_ids = [db_map[n]["id"] for n in x_selected if n in db_map]
                    rows_src = _fetch_all_instances_for_class_ids(sb, class_ids)

                if not rows_src:
                    st.info(
                        "No instances found for the selected X classes with current source."
                    )
                else:
                    # Prefetch missing JSON in bulk (avoid N round-trips)
                    missing_ids = [r["id"] for r in rows_src if not r.get("data")]
                    data_map = (
                        fetch_instances_json_map(sb, missing_ids) if missing_ids else {}
                    )

                    total = len(rows_src)
                    prog = st.progress(0.0, text="Zipping instances...")
                    added = 0
                    by_class = {}

                    buf = io.BytesIO()
                    with zipfile.ZipFile(
                        buf, "w", compression=zipfile.ZIP_DEFLATED
                    ) as zf:
                        for idx, r in enumerate(rows_src, start=1):
                            cls_name = _class_of(r)
                            data = r.get("data") or data_map.get(r["id"])
                            if not data:
                                # skip rows with no data
                                prog.progress(
                                    idx / total, text=f"Zipping… {idx}/{total}"
                                )
                                continue

                            ins_label = _ins_label(r.get("ins_id"))
                            base = (
                                ins_label
                                if ins_label and ins_label != "—"
                                else str(r["id"])[:8]
                            )

                            rel_path = (
                                f"CHOSEN/{_safe_name(cls_name)}/{_safe_name(base)}.json"
                            )
                            zf.writestr(rel_path, json.dumps(to_py(data), indent=2))
                            added += 1
                            by_class[cls_name] = by_class.get(cls_name, 0) + 1
                            prog.progress(idx / total, text=f"Zipping… {idx}/{total}")

                    buf.seek(0)

                    # Optional: persist to disk on the Streamlit host
                    if save_to_disk:
                        try:
                            with open(zip_filename_input or "CHOSEN.zip", "wb") as f:
                                f.write(buf.getbuffer())
                            st.success(
                                f"Saved to **{zip_filename_input or 'CHOSEN.zip'}** on disk."
                            )
                        except Exception as e:
                            st.warning(f"Could not save to disk: {e}")

                    # Always offer browser download (unique key avoids stale caching)
                    st.download_button(
                        "Download ZIP (CHOSEN)",
                        data=buf.getvalue(),
                        file_name="CHOSEN.zip",
                        mime="application/zip",
                        key=f"dl_zip_{int(time.time())}",
                    )

                    # Tiny summary
                    if added == 0:
                        st.info(
                            "ZIP created but no JSON files were added (no data found)."
                        )
                    else:
                        parts = ", ".join(
                            f"{k}: {v}" for k, v in sorted(by_class.items())
                        )
                        st.caption(
                            f"ZIP includes **{added}** JSON files across classes → {parts}"
                        )


def fetch_instances_details_map(sb: Client, inst_ids: list[str]) -> dict[str, dict]:
    """
    Return {instance_id: row} from instances_enriched with the minimal columns
    needed for the Explore tab, fetched in safe-sized chunks to avoid HTTP 414
    and statement timeouts.
    """
    out: dict[str, dict] = {}
    if not inst_ids:
        return out
    CH = 150  # keep pages small; safer for big JSON rows
    sel = "id,ins_id,period,manual_capacity,data,class_id"
    for k in range(0, len(inst_ids), CH):
        sub = inst_ids[k : k + CH]
        part = (
            sb.table("instances_enriched").select(sel).in_("id", sub).execute().data
            or []
        )
        for row in part:
            out[row["id"]] = row
    return out


# ----------------- Explore & Visualize tab -----------------
with viz_tab:
    st.header("Explore & Visualize (Supabase)")
    sb = supabase_client()
    if sb is None:
        st.info("Configure Supabase in the sidebar.")
        st.stop()

    # ------------ filters ------------
    class_opts = [("__ALL__", "All classes"), ("__ADHOC__", "adhoc (NULL class)")]
    for nm, row in st.session_state.get("db_classes", {}).items():
        class_opts.append((row["id"], nm))
    label_to_val = {label: val for val, label in class_opts}

    chosen_label = st.selectbox(
        "Filter by class", [lbl for _, lbl in class_opts], index=0
    )
    class_filter = label_to_val[chosen_label]

    n_runs = st.number_input("Fetch last N runs", min_value=50, value=1000, step=50)
    only_opt = st.checkbox("Show only OPTIMAL runs", value=False)
    hide_inf = st.checkbox("Hide INFEASIBLE runs", value=True)
    hide_int = st.checkbox("Hide INTERRUPTED runs", value=True)

    if st.button("🔄 Refresh data"):
        st.rerun()

    # ------------ fetch runs with keyset pagination (safe) ------------
    with st.spinner("Loading runs..."):
        runs_rows = fetch_runs_keyset_filtered(
            sb,
            n_wanted=int(n_runs),
            class_id_or_flag=class_filter,
            only_optimal=bool(only_opt),
            hide_infeasible=bool(hide_inf),
            hide_interrupted=bool(hide_int),
            page_size=500,
        )

    if not runs_rows:
        st.info("No runs match the filters.")
        st.stop()

    # ------------ fetch instance details (chunked) ------------
    inst_ids = sorted(
        list({r["instance_id"] for r in runs_rows if r.get("instance_id")})
    )
    inst_map = fetch_instances_details_map(sb, inst_ids)

    # ------------ assemble dataframe ------------
    STATUS_MAP = {
        1: "LOADED",
        2: "OPTIMAL",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        6: "CUTOFF",
        7: "ITERATION_LIMIT",
        8: "NODE_LIMIT",
        9: "TIME_LIMIT",
        10: "SOLUTION_LIMIT",
        11: "INTERRUPTED",
        12: "NUMERIC",
        13: "SUBOPTIMAL",
        14: "INPROGRESS",
        15: "USER_OBJ_LIMIT",
    }

    recs = []
    for r in runs_rows:
        I = inst_map.get(r.get("instance_id"))
        data = (I or {}).get("data") or {}
        items = data.get("items") or {}
        n_items = len(items) if isinstance(items, dict) else None
        cap = (I or {}).get("manual_capacity") or data.get("manual_capacity") or []
        cap_mean = float(np.mean(cap)) if cap else None
        meta = data.get("meta") or {}
        class_key = meta.get("class_key", "adhoc")
        period = (I or {}).get("period") or data.get("period")
        recs.append(
            {
                "run_id": r.get("id"),
                "created_at": r.get("created_at"),
                "instance_id": r.get("instance_id"),
                "ins_id": (I or {}).get("ins_id"),
                "class_key": class_key,
                "period": period,
                "n_items": n_items,
                "cap_mean": cap_mean,
                "status": r.get("status"),
                "status_label": STATUS_MAP.get(r.get("status"), str(r.get("status"))),
                "objective": r.get("objective"),
                "best_bound": r.get("best_bound"),
                "gap": r.get("gap"),
                "runtime_sec": r.get("runtime_sec"),
                "solver_version": r.get("solver_version") or "unknown",
            }
        )

    df = pd.DataFrame(recs)
    if df.empty:
        st.info("No runs after enrichment.")
        st.stop()

    # Pretty instance label
    try:
        _ = _ins_label  # use your global helper if present
    except NameError:

        def _ins_label(v):
            try:
                return f"ins#{int(v)}"
            except Exception:
                return "—"

    df["instance"] = df["ins_id"].apply(_ins_label)

    # Ensure numerics
    for col in [
        "runtime_sec",
        "cap_mean",
        "n_items",
        "objective",
        "best_bound",
        "gap",
        "period",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["gap_num"] = df["gap"]

    # Infer run batches from created_at (≥1 hour gap → new batch)
    try:
        df_sorted = df.sort_values("created_at").copy()
        ts_sorted = pd.to_datetime(df_sorted["created_at"], utc=True, errors="coerce")
        boundaries = ts_sorted.diff() > pd.Timedelta(hours=HOUR_GAP_FOR_BATCH)
        df_sorted["run_batch"] = (boundaries.cumsum() + 1).astype(int)
        df["run_batch"] = df["run_id"].map(df_sorted.set_index("run_id")["run_batch"])
    except Exception:
        df["run_batch"] = 1

    # ---- Solver filter ----
    solver_values = sorted([s for s in df["solver_version"].fillna("unknown").unique()])
    chosen_solvers = st.multiselect(
        "Filter by solver_version",
        options=solver_values,
        default=solver_values,
        key="vis_solver_filter",
    )
    if chosen_solvers:
        df = df[df["solver_version"].isin(chosen_solvers)]

    # ---- Batch filter ----
    batch_values = sorted(df["run_batch"].dropna().unique().tolist())
    batch_labels = ["ALL"] + [f"batch {b}" for b in batch_values]
    selected_batch_label = st.selectbox(
        "Filter by run batch",
        options=batch_labels,
        index=0,
        key="vis_batch_filter",
    )
    if selected_batch_label != "ALL":
        try:
            selected_batch = int(selected_batch_label.split()[-1])
            df = df[df["run_batch"] == selected_batch]
        except Exception:
            pass

    # ---- Status filters (client-side mirror of UI) ----
    df_filtered = df.copy()
    if hide_inf:
        df_filtered = df_filtered[df_filtered["status"] != 3]
    if hide_int:
        df_filtered = df_filtered[df_filtered["status"] != 11]
    if only_opt:
        df_filtered = df_filtered[df_filtered["status"] == 2]

    # ------------ tables & charts (same as before) ------------
    cols_order = [
        "run_id",
        "created_at",
        "instance",
        "class_key",
        "period",
        "n_items",
        "cap_mean",
        "status_label",
        "solver_version",
        "objective",
        "best_bound",
        "gap",
        "runtime_sec",
        "run_batch",
    ]
    show_cols = [c for c in cols_order if c in df_filtered.columns]
    st.subheader("Summary (runs table)")
    st.dataframe(df_filtered[show_cols], use_container_width=True)

    st.markdown("**Status breakdown (counts)**")
    cnt = (
        df_filtered.groupby(
            ["solver_version", "class_key", "status_label"], dropna=False
        )
        .size()
        .reset_index(name="runs")
    )
    st.dataframe(cnt, use_container_width=True)

    color_by_main = st.selectbox(
        "Color series by",
        options=["solver_version", "class_key", "status_label"],
        index=0,
        key="vis_color_by",
    )

    c1, c2 = st.columns(2)
    with c1:
        df_sc1 = df_filtered.dropna(subset=["n_items", "runtime_sec"])
        if not df_sc1.empty:
            fig = px.scatter(
                df_sc1,
                x="n_items",
                y="runtime_sec",
                color=color_by_main,
                symbol="status_label",
                hover_data=[
                    "instance",
                    "run_id",
                    "instance_id",
                    "status_label",
                    "solver_version",
                    "class_key",
                    "run_batch",
                ],
                title="Runtime vs #Items",
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Not enough data for Runtime vs #Items.")

    with c2:
        df_sc2 = df_filtered.dropna(subset=["cap_mean", "runtime_sec"])
        if not df_sc2.empty:
            fig = px.scatter(
                df_sc2,
                x="cap_mean",
                y="runtime_sec",
                color=color_by_main,
                symbol="status_label",
                hover_data=[
                    "instance",
                    "run_id",
                    "instance_id",
                    "status_label",
                    "solver_version",
                    "class_key",
                    "run_batch",
                ],
                title="Runtime vs Mean Capacity",
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Not enough data for Runtime vs Mean Capacity.")

    st.subheader("3D scatter")
    x_axis = st.selectbox("X", ["n_items", "period", "cap_mean"], key="x3d")
    y_axis = st.selectbox(
        "Y", ["gap", "runtime_sec", "objective", "best_bound"], key="y3d"
    )
    z_axis = st.selectbox(
        "Z", ["runtime_sec", "gap", "objective", "best_bound"], key="z3d"
    )
    color_choice = st.selectbox(
        "Color", ["solver_version", "class_key", "status_label"], key="c3d"
    )
    symbol_choice = st.selectbox(
        "Symbol", ["(none)", "class_key", "status_label"], index=1, key="s3d"
    )

    color_col = color_choice
    try:
        nunique = df_filtered[color_col].nunique(dropna=False)
    except Exception:
        nunique = 0
    if nunique <= 1:
        fallback = "class_key" if color_col != "class_key" else "status_label"
        if df_filtered[fallback].nunique(dropna=False) > 1:
            color_col = fallback
            st.caption(f"Only one '{color_choice}' present → coloring by '{fallback}'.")

    symbol_col = None if symbol_choice == "(none)" else symbol_choice

    plot_df = df_filtered.dropna(subset=[x_axis, y_axis, z_axis])
    if not plot_df.empty:
        fig3d = px.scatter_3d(
            plot_df,
            x=x_axis,
            y=y_axis,
            z=z_axis,
            color=color_col,
            symbol=symbol_col,
            hover_data=[
                "instance",
                "run_id",
                "instance_id",
                "status_label",
                "solver_version",
                "class_key",
                "run_batch",
            ],
        )
        fig3d.update_layout(
            height=520,
            legend_title_text=(
                f"{color_col}" if not symbol_col else f"{color_col}, {symbol_col}"
            ),
            scene=dict(xaxis_title=x_axis, yaxis_title=y_axis, zaxis_title=z_axis),
        )
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.caption("Not enough data for 3D scatter.")

    st.subheader("Class aggregates (including current filter)")

    def _p90(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        return float(np.percentile(x, 90)) if len(x) else None

    def _p95(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        return float(np.percentile(x, 95)) if len(x) else None

    agg = (
        df_filtered.groupby(["class_key", "solver_version"])
        .agg(
            runs=("run_id", "count"),
            n_items_mean=("n_items", "mean"),
            runtime_p50=("runtime_sec", "median"),
            runtime_p90=("runtime_sec", _p90),
            gap_p95=("gap_num", _p95),
            infeasible=("status", lambda s: int(np.sum(s == 3))),
            time_limit=("status", lambda s: int(np.sum(s == 9))),
            suboptimal=("status", lambda s: int(np.sum(s == 13))),
            optimal=("status", lambda s: int(np.sum(s == 2))),
        )
        .reset_index()
    )
    st.dataframe(agg, use_container_width=True)

    st.subheader("Status counts per class (faceted by solver)")
    if not cnt.empty:
        fig_bar = px.bar(
            cnt,
            x="class_key",
            y="runs",
            color="status_label",
            barmode="stack",
            text_auto=True,
            facet_col="solver_version",
        )
        fig_bar.update_layout(height=420, xaxis_title="", yaxis_title="# runs")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.caption("No data for the selected filter.")

    st.subheader("Runtime distribution per class (all statuses in filter)")
    df_rt = df_filtered.dropna(subset=["runtime_sec"])
    if not df_rt.empty:
        fig_box = px.box(
            df_rt,
            x="class_key",
            y="runtime_sec",
            color="status_label",
            points="all",
            hover_data=[
                "instance",
                "run_id",
                "instance_id",
                "status_label",
                "solver_version",
                "class_key",
                "run_batch",
            ],
            facet_col="solver_version",
            title="Runtime distribution per class",
        )
        fig_box.update_layout(height=420)
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.caption("No runtimes available for box plot.")

    st.subheader("Median runtime heatmap (class × #items bin)")
    df_hm = df_filtered.copy()
    bins = [0, 5, 10, 20, 50, 100, np.inf]
    labels = ["≤5", "6–10", "11–20", "21–50", "51–100", "100+"]
    df_hm["items_bin"] = pd.cut(df_hm["n_items"], bins=bins, labels=labels)

    solvers_for_hm = chosen_solvers if chosen_solvers else ["(all solvers)"]
    for sv in solvers_for_hm:
        sub = df_hm if sv == "(all solvers)" else df_hm[df_hm["solver_version"] == sv]
        pt = (
            sub.dropna(subset=["runtime_sec"])
            .groupby(["class_key", "items_bin"])["runtime_sec"]
            .median()
            .unstack("items_bin")
            .reindex(columns=labels)
        )
        st.markdown(f"**Median runtime (solver = {sv})**")
        if pt.size > 0:
            fig_hm = px.imshow(
                pt,
                labels=dict(x="items_bin", y="class_key", color="median runtime (sec)"),
                aspect="auto",
            )
            fig_hm.update_layout(height=420)
            fig_hm.update_traces(hoverongaps=False)
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.caption("No data for heatmap.")

    st.subheader("Objective vs Runtime")
    df_bub = df_filtered.dropna(subset=["objective", "runtime_sec"])
    if not df_bub.empty:
        fig_bub = px.scatter(
            df_bub,
            x="runtime_sec",
            y="objective",
            size="n_items",
            color=color_by_main,
            symbol="status_label",
            hover_data=[
                "instance",
                "run_id",
                "instance_id",
                "status_label",
                "solver_version",
                "class_key",
                "run_batch",
            ],
            size_max=20,
            title="Objective vs Runtime (bubble size = #items)",
        )
        fig_bub.update_layout(height=420)
        st.plotly_chart(fig_bub, use_container_width=True)
    else:
        st.caption("No data for objective vs runtime.")
