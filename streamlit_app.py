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
    from mip.solver_mip_lefo import solve_instance as _solve_lefo_v2
except Exception:
    _solve_lefo_v2 = None  # optional backend

try:
    from mip.solver_mip_no_cross import solve_instance as _solve_nocross_v1
except Exception:
    _solve_nocross_v1 = None  # optional backend

SOLVER_REGISTRY = {
    "LEFO v2 (permission-based)": {
        "fn": _solve_lefo_v2,
        "tag": "lefo_mip_v2",
        "desc": "LEFO permission-based model (v2).",
    },
    "No-Crossing v1": {
        "fn": _solve_nocross_v1,
        "tag": "lefo_mip_v1",  # the DB signature you wanted
        "desc": "No-crossing model (v1).",
    },
}


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
        "name": "2ndclass_T60_N35_DBMedium_TBO6_CVhigh_Z4_m8-20",
        "period": 60,
        "cap_mode": "DemandBased",
        "cap_params": {"jitter_pct": 3.0},
        "cap_tight": "Medium",
        "n_items": 35,
        "dem_lo": 0,
        "dem_hi": 160,  # CV high
        "m_lo": 8,
        "m_hi": 20,
        "c_mode": "scalar",
        "c_params": {"value": 2.0},
        "h_mode": "scalar",
        "h_params": {"value": 0.4},
        "s_mode": "tbo",
        "s_params": {"L": 6.0, "jitter_pct": 10.0},
        "zero_head": 4,
        "batch_size": 4,
        "seed_base": 31700,
        "allow_unmet_demand": False,
        "lost_sales_penalty_factor": 200.0,
    },
    {
        "name": "2ndclass_T60_N45_DBLoose_TBO8_CVhigh_Z4_m10-30",
        "period": 60,
        "cap_mode": "DemandBased",
        "cap_params": {"jitter_pct": 3.0},
        "cap_tight": "Loose",
        "n_items": 45,
        "dem_lo": 0,
        "dem_hi": 160,  # CV high
        "m_lo": 10,
        "m_hi": 30,
        "c_mode": "scalar",
        "c_params": {"value": 2.0},
        "h_mode": "scalar",
        "h_params": {"value": 0.4},
        "s_mode": "tbo",
        "s_params": {"L": 8.0, "jitter_pct": 10.0},
        "zero_head": 4,
        "batch_size": 4,
        "seed_base": 31720,
        "allow_unmet_demand": False,
        "lost_sales_penalty_factor": 200.0,
    },
    {
        "name": "2ndclass_T60_N30_DBMedium_TBO8_CVhigh_Z5_m10-30",
        "period": 60,
        "cap_mode": "DemandBased",
        "cap_params": {"jitter_pct": 3.0},
        "cap_tight": "Medium",
        "n_items": 30,
        "dem_lo": 0,
        "dem_hi": 160,  # CV high
        "m_lo": 10,
        "m_hi": 30,
        "c_mode": "scalar",
        "c_params": {"value": 2.0},
        "h_mode": "scalar",
        "h_params": {"value": 0.4},
        "s_mode": "tbo",
        "s_params": {"L": 8.0, "jitter_pct": 12.0},
        "zero_head": 5,
        "batch_size": 4,
        "seed_base": 31740,
        "allow_unmet_demand": False,
        "lost_sales_penalty_factor": 200.0,
    },
    {
        "name": "2ndclass_T60_N30_DBMedium_TBO6_CVhigh_Z4_m8-12",
        "period": 60,
        "cap_mode": "DemandBased",
        "cap_params": {"jitter_pct": 3.0},
        "cap_tight": "Medium",
        "n_items": 30,
        "dem_lo": 0,
        "dem_hi": 170,  # slightly higher dem_hi to increase difficulty
        "m_lo": 8,
        "m_hi": 12,
        "c_mode": "scalar",
        "c_params": {"value": 2.0},
        "h_mode": "scalar",
        "h_params": {"value": 0.4},
        "s_mode": "tbo",
        "s_params": {"L": 6.0, "jitter_pct": 10.0},
        "zero_head": 4,
        "batch_size": 4,
        "seed_base": 31760,
        "allow_unmet_demand": False,
        "lost_sales_penalty_factor": 200.0,
    },
    {
        "name": "2ndclass_T20_N10_DBMedium_TBO6_CVhigh_Z4_m7-12",
        "period": 20,
        "cap_mode": "DemandBased",
        "cap_params": {"jitter_pct": 3.0},
        "cap_tight": "Medium",
        "n_items": 10,
        "dem_lo": 0,
        "dem_hi": 170,  # slightly higher dem_hi to increase difficulty
        "m_lo": 7,
        "m_hi": 12,
        "c_mode": "scalar",
        "c_params": {"value": 2.0},
        "h_mode": "scalar",
        "h_params": {"value": 0.4},
        "s_mode": "tbo",
        "s_params": {"L": 6.0, "jitter_pct": 10.0},
        "zero_head": 3,
        "batch_size": 4,
        "seed_base": 31760,
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
st.session_state.setdefault("classes", [])
st.session_state.setdefault("class_queue", [])

with classes_tab:
    st.header("Solution Classes (define and queue)")

    # --- Classes in Supabase ---
    db_map = st.session_state.get("db_classes", {})
    st.subheader("Classes in Supabase")

    # Reload DB
    cdb0 = st.columns(1)[0]
    if cdb0.button("🔄 Refresh from Supabase"):
        rows = fetch_classes(supabase_client())
        st.session_state["db_classes"] = {r["name"]: r for r in rows}
        st.success("Reloaded classes from DB.")

    if db_map:
        df_db = pd.DataFrame(
            [
                {
                    "name": r["name"],
                    "period": r["spec"].get("period"),
                    "#items": r["spec"].get("n_items"),
                    "cap_mode": r["spec"].get("cap_mode"),
                    "created_at": r.get("created_at"),
                }
                for r in db_map.values()
            ]
        )
        st.dataframe(df_db, use_container_width=True)

        pick_db = st.multiselect(
            "Select DB classes", options=sorted(db_map.keys()), key="pick_db"
        )

        cdb1, cdb2 = st.columns(2)
        with cdb1:
            if st.button("➕ Queue selected (DB)"):
                # Add chosen specs to queue (no duplicates)
                st.session_state["classes"] += [
                    db_map[name]["spec"] for name in pick_db
                ]
                dedupe_queue_by_name()
                st.success("Queued selected DB class(es).")

        with cdb2:
            if st.button("📥 Replace queue with ALL DB"):
                st.session_state["classes"] = [r["spec"] for r in db_map.values()]
                dedupe_queue_by_name()
                st.success(
                    f"Queue replaced with {len(st.session_state['classes'])} DB class(es)."
                )
    else:
        st.info("No classes in Supabase yet. Save some or push local presets.")

    # --- Local presets (in code) ---
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
        )
        st.dataframe(df_lp, use_container_width=True)

        pick_local = st.multiselect(
            "Select local presets", options=sorted(lp_map.keys()), key="pick_local"
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ Queue selected local"):
                for name in pick_local:
                    st.session_state["classes"].append(lp_map[name])
                dedupe_queue_by_name()
                st.success("Queued selected local preset(s).")

        with c2:
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

    c_name = st.text_input(
        "Class name", value="demo_60_uniformcap_meditems", key="cls_name"
    )

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
        s_params = {"L": float(tbo_L), "jitter_pct": float(tbo_jit)}
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

    # Quick table view  ✅ now editable for batch sizes
    if st.session_state["classes"]:
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

        edited_dfq = st.data_editor(
            dfq,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "name": st.column_config.TextColumn(disabled=True),
                "T": st.column_config.NumberColumn(format="%d", disabled=True),
                "items": st.column_config.NumberColumn(format="%d", disabled=True),
                "batch": st.column_config.NumberColumn(
                    "batch",
                    help="Instances per class (batch size)",
                    min_value=1,
                    step=1,
                    format="%d",
                ),
            },
            key="queue_editor",
        )

    # Write edited batch sizes back into the queued class specs
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

    # Remove/clear controls
    rm_names = st.multiselect(
        "Select queued classes to remove",
        options=[c["name"] for c in st.session_state["classes"]],
        key="rm_from_queue",
    )
    c_rm, c_clr, c_save = st.columns(3)
    with c_rm:
        if st.button("🗑️ Remove selected from queue"):
            before = len(st.session_state["classes"])
            st.session_state["classes"] = [
                c for c in st.session_state["classes"] if c["name"] not in rm_names
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
            for cls in st.session_state["classes"]:
                if ensure_class_row(sb_for_classes, cls):  # upsert by name
                    saved += 1
            _reset_sortables_state()
            st.success(f"Saved/updated {saved} unique class spec(s) to Supabase.")

    # ---cascade delete portion in this tab ----

    def _delete_in_chunks(
        sb: Client, table: str, col: str, ids: list[str], chunk: int = 500
    ) -> int:
        total = 0
        for k in range(0, len(ids), chunk):
            part = ids[k : k + chunk]
            if not part:
                continue
            try:
                res = sb.table(table).delete().in_(col, part).execute()
                total += len(part) if (res.data is None) else len(res.data)
            except Exception:
                # bazı kurulumlarda returning kapalı veya RLS uyarısı baskılanmış olabilir
                total += len(part)
        return total

    def _select_ids_eq(
        sb: Client, table: str, col: str, val: str, step: int = 1000
    ) -> list[str]:
        """eq ile sayfa sayfa id topla (RLS varsa erişilebilenleri döner)."""
        out, start = [], 0
        while True:
            res = (
                sb.table(table)
                .select("id")
                .eq(col, val)
                .range(start, start + step - 1)
                .order("id")
                .execute()
            )
            rows = res.data or []
            if not rows:
                break
            out.extend([r["id"] for r in rows])
            if len(rows) < step:
                break
            start += len(rows)
        return out

    def _fetch_ids_or(
        sb: Client, table: str, or_expr: str, step: int = 1000
    ) -> list[str]:
        """
        or_expr: PostgREST or= ifadesi, ör: "class_id.eq.<uuid>,data->meta->>class_key.eq.<name>"
        """
        out, start = [], 0
        while True:
            q = (
                sb.table(table)
                .select("id")
                .or_(or_expr)
                .order("id", desc=False)
                .range(start, start + step - 1)
            )
            res = q.execute()
            rows = res.data or []
            if not rows:
                break
            out.extend([r["id"] for r in rows])
            if len(rows) < step:
                break
            start += len(rows)
        return out

    def delete_class_everywhere(sb: Client, class_name: str) -> Dict[str, int]:
        # 0) class_id çek
        cls_res = (
            sb.table("classes").select("id").eq("name", class_name).limit(1).execute()
        )
        cls_rows = cls_res.data or []
        if not cls_rows:
            return {"classes": 0, "instances": 0, "runs": 0, "orders": 0}
        cid = cls_rows[0]["id"]

        deleted_orders = 0
        deleted_runs = 0
        deleted_insts = 0

        # 1) Döngü: class_id=cid olan INSTANCES bitene kadar silme adımlarını tekrarla
        while True:
            inst_ids = _select_ids_eq(sb, "instances", "class_id", cid, step=1000)
            if not inst_ids:
                break

            # Bu instance'lara bağlı RUN id'lerini topla
            run_ids = []
            for k in range(0, len(inst_ids), 500):
                part = inst_ids[k : k + 500]
                r = (
                    sb.table("runs")
                    .select("id")
                    .in_("instance_id", part)
                    .order("id")
                    .execute()
                )
                run_ids.extend([x["id"] for x in (r.data or [])])

            # Orders → Runs → Instances sırayla sil
            if run_ids:
                deleted_orders += _delete_in_chunks(
                    sb, "orders", "run_id", run_ids, chunk=500
                )
                deleted_runs += _delete_in_chunks(sb, "runs", "id", run_ids, chunk=500)

            deleted_insts += _delete_in_chunks(
                sb, "instances", "id", inst_ids, chunk=500
            )

            # Döngü başına dönüp kalan var mı tekrar bakacağız

        # 2) Emniyet: class_id=cid bağlı RUN varsa (nadiren) onları da temizle
        # (ör. biri instance_id=NULL, class_id=cid kalmış olabilir)
        extra_run_ids = _select_ids_eq(sb, "runs", "class_id", cid, step=1000)
        if extra_run_ids:
            deleted_orders += _delete_in_chunks(
                sb, "orders", "run_id", extra_run_ids, chunk=500
            )
            deleted_runs += _delete_in_chunks(
                sb, "runs", "id", extra_run_ids, chunk=500
            )

        # 3) Hâlâ class_id=cid'li instance var mı? Varsa NULL'la ve bir kez daha dene
        try:
            chk = (
                sb.table("instances")
                .select("id", count="exact")
                .eq("class_id", cid)
                .execute()
            )
            remain = getattr(chk, "count", 0) or 0
        except Exception:
            remain = 0

        if remain > 0:
            # class_id NULL'la
            try:
                sb.table("instances").update({"class_id": None}).eq(
                    "class_id", cid
                ).execute()
            except Exception:
                pass
            # tekrar dene
            inst_ids = _select_ids_eq(sb, "instances", "class_id", cid, step=1000)
            if inst_ids:
                deleted_insts += _delete_in_chunks(
                    sb, "instances", "id", inst_ids, chunk=500
                )

        # 4) Son kontrol: sınıf bağlı instance kaldıysa, RLS/policy engelliyordur → hata göster
        try:
            chk2 = (
                sb.table("instances")
                .select("id", count="exact")
                .eq("class_id", cid)
                .execute()
            )
            remain2 = getattr(chk2, "count", 0) or 0
        except Exception:
            remain2 = 0

        if remain2 > 0:
            # Bu durumda sınıfı silmeye kalkarsan tekrar 23503 alırsın.
            # UI'da kullanıcıya net hata gösterebilmen için Exception fırlatıyorum.
            raise RuntimeError(
                f"Cannot delete class '{class_name}': {remain2} instance(s) still reference it. "
                "Check RLS/Policies or use a service key / ON DELETE CASCADE."
            )

        # 5) Artık class'ı sil
        sb.table("classes").delete().eq("id", cid).execute()

        # (İSTEĞE BAĞLI) meta.class_key eşleşen yetim 'instances' varsa onları da temizlemek istersen:
        #   NOT: class silmek için şart değil; sadece temizlik amaçlı.
        # try:
        #     # JSON path ile: data->meta->>class_key = class_name
        #     # Bu kısım RLS'e takılabilir; takılırsa görmezden gel.
        #     orphans = sb.table("instances").select("id").or_(f"data->meta->>class_key.eq.{class_name}").execute().data or []
        #     orphan_ids = [x["id"] for x in orphans]
        #     if orphan_ids:
        #         _delete_in_chunks(sb, "instances", "id", orphan_ids, chunk=500)
        # except Exception:
        #     pass

        return {
            "classes": 1,
            "instances": deleted_insts,
            "runs": deleted_runs,
            "orders": deleted_orders,
        }

    # ### deletion
    st.markdown("### Danger zone")
    to_del = st.multiselect(
        "Select DB classes to DELETE (cascade)",
        options=sorted(db_map.keys()),
        key="pick_db_del",
    )

    if st.button("🗑️ Delete selected classes (DB + instances + runs + orders)"):
        sbx = supabase_client()
        if not sbx:
            st.error("Supabase not configured.")
        else:
            total = {"classes": 0, "instances": 0, "runs": 0, "orders": 0}
            for nm in to_del:
                cnt = delete_class_everywhere(sbx, nm)
                for k in total:
                    total[k] += cnt.get(k, 0)
            st.success(
                f"Deleted: classes={total['classes']} instances={total['instances']} runs={total['runs']} orders={total['orders']}"
            )
            # refresh DB list
            rows = fetch_classes(sbx)
            st.session_state["db_classes"] = {r["name"]: r for r in rows}
    # -----------------------------------------------------------------------------

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

    def generate_cap_series(
        period: int, mode: str, params: dict, seed: int
    ) -> List[int]:
        g = rng(seed)
        if mode == "Constant":
            return [int(params.get("value", 10000))] * period

        if mode == "Uniform":
            lo, hi = int(params.get("lo", 9000)), int(params.get("hi", 11000))
            # cast each element to builtin int
            return [int(x) for x in g.integers(lo, hi + 1, size=period)]

        # Normal
        mu, sd = float(params.get("mean", 10000)), float(params.get("std", 500))
        clip_lo, clip_hi = float(params.get("clip_lo", 0)), float(
            params.get("clip_hi", 2 * mu)
        )
        arr = (
            clip_list(g.normal(mu, sd, size=period), clip_lo, clip_hi)
            .round()
            .astype(int)
        )
        return [int(x) for x in arr]  # cast to builtin ints

    # --- generate_instance_from_class
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

            # c, h sequences or scalar
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

            # setup (support TBO)
            if cls["s_mode"] == "tbo":
                # h average for this item
                h_avg = (
                    float(h_out) if isinstance(h_out, float) else float(np.mean(h_out))
                )
                d_avg = float(np.mean(D)) if len(D) else 0.0
                L_raw = float(cls["s_params"].get("L", 2.0))
                L = max(0.1, L_raw if math.isfinite(L_raw) else 2.0)
                jitter_pct = float(
                    cls["s_params"].get("jitter_pct", SETUP_TBO_JITTER_DEFAULT)
                )
                base_s = 0.5 * h_avg * d_avg * (L**2)
                if base_s <= 0:
                    base_s = 1e-6
                jitter = 1.0 + rng_local.uniform(
                    -jitter_pct / 100.0, jitter_pct / 100.0
                )
                s_out = float(base_s * jitter)  # scalar setup
            else:
                s_seq = make_series(
                    cls["s_mode"], period, dict(cls["s_params"] or {}), seed_k + 4
                )
                s_out = (
                    float(cls["s_params"].get("value", 0.0))
                    if cls["s_mode"] == "scalar"
                    else [float(x) for x in s_seq]
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

        # capacity
        cap_mode = cls.get("cap_mode", "Uniform")
        if cap_mode == "DemandBased":
            beta = CAP_TIGHT_BETAS.get(cls.get("cap_tight") or "Medium", 0.60)
            mean_total = float(np.mean(total_by_t)) if period > 0 else 0.0
            base_cap = max(0, int(round(beta * mean_total)))

            # NEW: small per-period jitter (defaults to 3% if not specified)
            jit_pct = float((cls.get("cap_params") or {}).get("jitter_pct", 3.0))
            if jit_pct > 0:
                g = np.random.default_rng(int(cls["seed_base"] + 31 * j + 7))
                noise = g.uniform(-jit_pct / 100.0, jit_pct / 100.0, size=period)
                cap = np.maximum(
                    0, np.round(base_cap * (1.0 + noise)).astype(int)
                ).tolist()
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
            "lost_sales_penalty_factor": float(
                cls.get("lost_sales_penalty_factor", 200.0)
            ),
            "meta": {
                "origin": "class",
                "class_key": cls["name"],
                "class_params": cls,
            },
        }
        return inst

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

        st.subheader("Filtered runs")
        st.dataframe(df_runs, use_container_width=True)

        # ---- Pick a specific run ----
        pretty_opts = [
            f"{row['run_id']} • {row['created_at']} • {row['status_label']} • {row['solver_version']} • obj={row['objective']}"
            for _, row in df_runs.iterrows()
        ]
        sel = st.selectbox("Select a run to inspect", pretty_opts, index=0)
        sel_idx = pretty_opts.index(sel)
        selected_run_id = df_runs.iloc[sel_idx]["run_id"]

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
                sb.table("instances")
                .select("id,period,manual_capacity,data")
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
                # pad/trim to T_inst defensively
                d = (d + [0] * int(T_inst))[: int(T_inst)]
                dem_rows[i] = d
            df_dem = pd.DataFrame.from_dict(dem_rows, orient="index")
            df_dem.index.name = "item_id"
            df_dem.columns = list(range(int(T_inst)))

            # Shelf-life (m_it) matrix (same shape)
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

            # ---- Demand vs Capacity (bar + line) ----
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

            # ---- Per-item demand heatmap ----
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

            # ---- Per-item shelf-life heatmap (m_it) ----
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

            # ---- Items overview table (quick stats per item) ----
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

            # ---- Raw JSON + downloads ----
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
                # Export a compact Excel with the matrices and items overview
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

        # ---- Orders: table & quick plots ----
        st.subheader("Order plan (per item)")
        if df_orders.empty:
            st.info("This run has no orders recorded.")
        else:
            # Wide pivot for eye-check (t rows, item columns)
            pivot = df_orders.pivot_table(
                index="t", columns="item_id", values="qty", fill_value=0
            ).sort_index()
            st.dataframe(pivot, use_container_width=True)

            # Quick per-item plot selector (default to up to 8 most active items)
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
                # Combine selected items into one figure (lines)
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

            # Also show nonzero long-form table for quick scanning
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
    # fetch classes once (you already cached db_classes at boot)
    db_map = st.session_state.get("db_classes", {})
    class_choices = ["<ALL>", "<ADHOC (NULL)>"] + sorted(db_map.keys())
    pick_cls = st.selectbox("Filter by class", class_choices, index=0)

    # limit + refresh
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        inst_limit = st.number_input(
            "Fetch last N instances",
            min_value=10,
            value=500,
            step=50,
            key="saved_run_limit",
        )
    with c2:
        if st.button("🔄 Refresh instances"):
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    with c3:
        pass

    # query instances with optional class filter (keyset-like paging)
    def fetch_instances_keyset(sb_client: Client, total: int, class_name: str | None):
        out, last_seen = [], None
        page_size = 1000
        # map class name -> id
        cls_id = None
        if class_name and class_name in db_map:
            cls_id = db_map[class_name]["id"]
        while len(out) < total:
            need = min(page_size, total - len(out))
            q = (
                sb_client.table("instances")
                .select("id,created_at,period,class_id,manual_capacity,data")
                .order("created_at", desc=True)
            )
            if last_seen is not None:
                q = q.lt("created_at", last_seen)
            if class_name == "<ADHOC (NULL)>":
                q = q.is_("class_id", None)
            elif cls_id:
                q = q.eq("class_id", cls_id)
            batch = q.limit(need).execute().data
            if not batch:
                break
            out.extend(batch)
            last_seen = batch[-1]["created_at"]
        return out

    class_name_filter = None if pick_cls in ("<ALL>",) else pick_cls
    instances = fetch_instances_keyset(sb, int(inst_limit), class_name_filter)

    if not instances:
        st.info("No instances match the filter.")
        st.stop()

    # small table
    def _safe_len_items(row):
        try:
            return len((row.get("data") or {}).get("items") or {})
        except Exception:
            return None

    dfI = pd.DataFrame(
        [
            {
                "instance_id": r["id"],
                "created_at": r["created_at"],
                "class": (
                    next(
                        (n for n, v in db_map.items() if v["id"] == r.get("class_id")),
                        "adhoc",
                    )
                    if r.get("class_id")
                    else "adhoc"
                ),
                "period": r.get("period"),
                "n_items": _safe_len_items(r),
            }
            for r in instances
        ]
    )

    st.dataframe(dfI, use_container_width=True, height=300)

    # choose which instances to run
    all_ids = dfI["instance_id"].tolist()
    pick_ids = st.multiselect(
        "Pick specific instance IDs (leave empty to use ALL loaded above)",
        options=all_ids,
    )

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
        ids_to_run = pick_ids if pick_ids else all_ids
        id_set = set(ids_to_run)
        rows = [r for r in instances if r["id"] in id_set]
        total = len(rows) * len(multi_solver_labels)
        prog = st.progress(0.0, text="Running saved instances...")
        done = 0

        for r in rows:
            inst_json = r.get("data") or {}
            # write temp json for solver I/O
            tmp_path = Path("tmp_instance_saved.json")
            tmp_path.write_text(
                json.dumps(to_py(inst_json), indent=2), encoding="utf-8"
            )

            for label in multi_solver_labels:
                entry = (
                    SOLVER_REGISTRY.get(label)
                    or SOLVER_REGISTRY["LEFO v2 (permission-based)"]
                )
                solve_fn = (
                    entry["fn"] or SOLVER_REGISTRY["LEFO v2 (permission-based)"]["fn"]
                )
                solver_tag = (
                    entry["tag"]
                    if entry["fn"]
                    else SOLVER_REGISTRY["LEFO v2 (permission-based)"]["tag"]
                )

                # run
                summary, orders_txt = solve_fn(
                    str(tmp_path), time_limit=time_limit, mip_gap=mip_gap
                )

                # log run + orders
                try:
                    run_payload = {
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
                    run_payload = sanitize_json(run_payload)
                    run_res = sb.table("runs").insert(run_payload).execute()
                    run_id = run_res.data[0]["id"]

                    # parse orders
                    rows_ord = [
                        {"run_id": run_id, **r} for r in parse_orders_lines(orders_txt)
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

# ----------------- Explore & Visualize tab -----------------

with viz_tab:
    st.header("Explore & Visualize (Supabase)")
    sb = supabase_client()
    if sb is None:
        st.info("Configure Supabase in sidebar.")
    else:
        # ---- Controls: limit / status / refresh ----
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            limit = st.number_input(
                "Fetch last N runs",
                min_value=10,
                value=3000,
                step=100,
                key="vis_limit",
            )
        with c2:
            only_optimal = st.checkbox(
                "Show only OPTIMAL runs", value=False, key="vis_only_opt"
            )
            hide_infeasible = st.checkbox(
                "Hide INFEASIBLE runs", value=False, key="vis_hide_inf"
            )
            hide_interrupted = st.checkbox(  # NEW
                "Hide INTERRUPTED runs", value=False, key="vis_hide_int"
            )
        with c3:
            if st.button("🔄 Refresh data", key="vis_refresh"):
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

        # ---- Helper: paginate Supabase fetch to bypass 1000-row caps ----
        def fetch_runs_keyset(sb_client: Client, total: int, page_size: int = 1000):
            out, last_seen = [], None
            select_cols = (
                "id,created_at,instance_id,status,objective,best_bound,gap,"
                "runtime_sec,solver_version"
            )
            while len(out) < total:
                need = min(page_size, total - len(out))
                q = (
                    sb_client.table("runs")
                    .select(select_cols)
                    .order("created_at", desc=True)
                )
                if last_seen is not None:
                    q = q.lt("created_at", last_seen)  # keyset pagination
                batch = q.limit(need).execute().data
                if not batch:
                    break
                out.extend(batch)
                last_seen = batch[-1]["created_at"]
            return out

        # ---- Status mapping (Gurobi) ----
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
            11: "INTERrupted",
            12: "NUMERIC",
            13: "SUBOPTIMAL",
            14: "INPROGRESS",
            15: "USER_OBJ_LIMIT",
        }

        try:
            runs = fetch_runs_keyset(sb, int(limit))

            # Pull related instances
            inst_ids = list({r["instance_id"] for r in runs if r.get("instance_id")})
            inst = []
            if inst_ids:
                CH = 500
                for k in range(0, len(inst_ids), CH):
                    chunk = inst_ids[k : k + CH]
                    inst += (
                        sb.table("instances")
                        .select("id,period,manual_capacity,data")
                        .in_("id", chunk)
                        .execute()
                        .data
                    )
            inst_map = {row["id"]: row for row in inst}

            # Build unified dataframe of runs (+ instance metadata)
            recs = []
            for r in runs:
                I = inst_map.get(r["instance_id"])
                period = None
                cap = None
                class_key = "adhoc"
                n_items = None
                if I:
                    data = I.get("data", {}) or {}
                    meta = data.get("meta") or {}
                    items = data.get("items") or {}
                    n_items = len(items)
                    cap = I.get("manual_capacity") or data.get("manual_capacity") or []
                    class_key = meta.get("class_key", "adhoc")
                    period = I.get("period") or data.get("period")

                cap_mean = float(np.mean(cap)) if cap else None
                status_code = r.get("status")
                status_label = STATUS_MAP.get(status_code, str(status_code))
                solver_version = r.get("solver_version") or "unknown"

                recs.append(
                    {
                        "run_id": r.get("id"),
                        "created_at": r.get("created_at"),
                        "instance_id": r.get("instance_id"),
                        "class_key": class_key,
                        "period": period,
                        "n_items": n_items,
                        "cap_mean": cap_mean,
                        "status": status_code,
                        "status_label": status_label,
                        "objective": r.get("objective"),
                        "best_bound": r.get("best_bound"),
                        "gap": r.get("gap"),
                        "runtime_sec": r.get("runtime_sec"),
                        "has_instance": bool(I),
                        "solver_version": solver_version,
                    }
                )

            df = pd.DataFrame(recs)
            if df.empty:
                st.info("No runs found.")
                st.stop()

            # Ensure numeric columns are numeric (for plots/aggregations)
            for col in [
                "runtime_sec",
                "cap_mean",
                "n_items",
                "objective",
                "best_bound",
                "gap",
            ]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Numeric gap alias (for robust aggregations)
            df["gap_num"] = df["gap"]

            # ---- Infer run batches from created_at (≥2 hour gap starts a new batch) ----
            try:
                df_sorted = df.sort_values("created_at").copy()
                ts_sorted = pd.to_datetime(
                    df_sorted["created_at"], utc=True, errors="coerce"
                )
                boundaries = ts_sorted.diff() > pd.Timedelta(hours=2)
                df_sorted["run_batch"] = (boundaries.cumsum() + 1).astype(int)
                batch_map = df_sorted.set_index("run_id")["run_batch"]
                df["run_batch"] = df["run_id"].map(batch_map)
            except Exception:
                # Fallback if timestamps are missing/bad: treat everything as one batch
                df["run_batch"] = 1

            # ---- Solver filter & coloring options ----
            solver_values = sorted(
                [s for s in df["solver_version"].fillna("unknown").unique()]
            )
            default_solvers = solver_values[:]  # select all by default
            chosen_solvers = st.multiselect(
                "Filter by solver_version",
                options=solver_values,
                default=default_solvers,
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
                    pass  # ignore parsing issues and show all

            # ---- Status filters ----
            df_filtered = df.copy()
            if hide_infeasible:
                df_filtered = df_filtered[df_filtered["status"] != 3]
            if hide_interrupted:  # NEW
                df_filtered = df_filtered[df_filtered["status"] != 11]
            if only_optimal:
                df_filtered = df_filtered[df_filtered["status"] == 2]

            # How to color points/lines across plots
            color_by_main = st.selectbox(
                "Color series by",
                options=["solver_version", "class_key", "status_label"],
                index=0,
                key="vis_color_by",
            )

            st.subheader("Summary (runs table)")
            st.dataframe(df_filtered, use_container_width=True)

            st.markdown("**Status breakdown (counts)**")
            cnt = (
                df_filtered.groupby(
                    ["solver_version", "class_key", "status_label"], dropna=False
                )
                .size()
                .reset_index(name="runs")
            )
            st.dataframe(cnt, use_container_width=True)

            # --- Plots ---
            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(
                    df_filtered.dropna(subset=["n_items", "runtime_sec"]),
                    x="n_items",
                    y="runtime_sec",
                    color=color_by_main,
                    symbol="status_label",
                    hover_data=[
                        "gap",
                        "objective",
                        "best_bound",
                        "solver_version",
                        "class_key",
                        "run_batch",
                    ],
                )
                fig.update_layout(height=380, title="Runtime vs #Items")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.scatter(
                    df_filtered.dropna(subset=["cap_mean", "runtime_sec"]),
                    x="cap_mean",
                    y="runtime_sec",
                    color=color_by_main,
                    symbol="status_label",
                    hover_data=[
                        "gap",
                        "objective",
                        "best_bound",
                        "solver_version",
                        "class_key",
                        "run_batch",
                    ],
                )
                fig.update_layout(height=380, title="Runtime vs Mean Capacity")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("3D scatter")

            x_axis = st.selectbox("X", ["n_items", "period", "cap_mean"], key="x3d")
            y_axis = st.selectbox(
                "Y", ["gap", "runtime_sec", "objective", "best_bound"], key="y3d"
            )
            z_axis = st.selectbox(
                "Z", ["runtime_sec", "gap", "objective", "best_bound"], key="z3d"
            )

            # Let user pick color + symbol independently (with a 'none' option for symbols)
            color_choice = st.selectbox(
                "Color",
                ["solver_version", "class_key", "status_label"],
                key="c3d",
            )
            symbol_choice = st.selectbox(
                "Symbol",
                ["(none)", "class_key", "status_label"],
                index=1,  # default to class_key like before
                key="s3d",
            )

            # If the chosen color dimension has only a single level, fall back so colors still vary.
            color_col = color_choice
            try:
                nunique = df_filtered[color_col].nunique(dropna=False)
            except Exception:
                nunique = 0
            if nunique <= 1:
                fallback = "class_key" if color_col != "class_key" else "status_label"
                if df_filtered[fallback].nunique(dropna=False) > 1:
                    color_col = fallback
                    st.caption(
                        f"Only one '{color_choice}' present → coloring by '{fallback}'."
                    )

            symbol_col = None if symbol_choice == "(none)" else symbol_choice

            plot_df = df_filtered.dropna(subset=[x_axis, y_axis, z_axis])

            fig3d = px.scatter_3d(
                plot_df,
                x=x_axis,
                y=y_axis,
                z=z_axis,
                color=color_col,  # <- may be the fallback
                symbol=symbol_col,  # <- can be None
                hover_data=[
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

            st.subheader("Class aggregates (including current filter)")

            # P50/P90 runtime, P95 gap; plus status counts
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
                        "n_items",
                        "period",
                        "gap",
                        "objective",
                        "solver_version",
                    ],
                    facet_col="solver_version",
                )
                fig_box.update_layout(
                    height=420, title="Runtime distribution per class"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.caption("No runtimes available for box plot.")

            st.subheader("Median runtime heatmap (class × #items bin)")
            df_hm = df_filtered.copy()
            bins = [0, 5, 10, 20, 50, 100, np.inf]
            labels = ["≤5", "6–10", "11–20", "21–50", "51–100", "100+"]
            df_hm["items_bin"] = pd.cut(df_hm["n_items"], bins=bins, labels=labels)

            # one heatmap per solver_version (imshow has no facet_col)
            for sv in chosen_solvers if chosen_solvers else ["(all solvers)"]:
                sub = (
                    df_hm
                    if sv == "(all solvers)"
                    else df_hm[df_hm["solver_version"] == sv]
                )
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
                        labels=dict(
                            x="items_bin", y="class_key", color="median runtime (sec)"
                        ),
                        aspect="auto",
                    )
                    fig_hm.update_layout(height=420)
                    fig_hm.update_traces(hoverongaps=False)
                    st.plotly_chart(fig_hm, use_container_width=True)
                else:
                    st.caption("No data for heatmap.")

            st.subheader("Runs over time (daily)")
            df_time = df_filtered.copy()
            dt_parsed = pd.to_datetime(df_time["created_at"], utc=True, errors="coerce")
            df_time["date"] = dt_parsed.dt.date
            color_by_time = st.selectbox(
                "Time series color by",
                ["solver_version", "class_key", "status_label"],
                key="vis_ts_color",
            )
            ts = (
                df_time.groupby(
                    ["date", color_by_time, "status_label"], as_index=False
                )["run_id"]
                .count()
                .rename(columns={"run_id": "runs"})
            )
            if not ts.empty:
                fig_ts = px.line(
                    ts,
                    x="date",
                    y="runs",
                    color=color_by_time,
                    line_dash="status_label",
                    markers=True,
                )
                fig_ts.update_layout(height=380, title="Runs per day")
                st.plotly_chart(fig_ts, use_container_width=True)
            else:
                st.caption("No time series data.")

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
                        "gap",
                        "cap_mean",
                        "period",
                        "solver_version",
                        "class_key",
                        "run_batch",
                    ],
                    size_max=20,
                )
                fig_bub.update_layout(
                    height=420, title="Objective vs Runtime (bubble size = #items)"
                )
                st.plotly_chart(fig_bub, use_container_width=True)
            else:
                st.caption("No data for objective vs runtime.")
        except Exception as e:
            st.error(f"Supabase query failed: {e}")
