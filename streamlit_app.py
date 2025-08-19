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

from mip.solver_mip_lefo import solve_instance

# ======================= App Config =======================
DEFAULT_URL = "https://btqqbsnjcsgjvgpuutiw.supabase.co"
DEFAULT_ANON = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ0cXFic25qY3NnanZncHV1dGl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU1NDMwODMsImV4cCI6MjA3MTExOTA4M30.gissvSrKruPsYJOHOoLqfzQGLrB4oFVckVwhrUpGJXU"

st.set_page_config(page_title="Perishable Lot-Sizing (LEFO MIP)", layout="wide")
st.title("Perishable Lot-Sizing — Generator • Classes • Batches • Visualizer")


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
    """Keep first occurrence; drop later duplicates by 'name'."""
    seen = set()
    out = []
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
    time_limit = st.number_input("Time limit (sec)", min_value=0, value=0, step=10)
    mip_gap = st.number_input(
        "MIPGap (0.0 = default)", min_value=0.0, value=0.0, step=0.01, format="%.4f"
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
        "name": "baseline_T60_uniformCap_medItems",
        "period": 60,
        "cap_mode": "Uniform",
        "cap_params": {"lo": 9000, "hi": 11000},
        "n_items": 8,
        "dem_lo": 5,
        "dem_hi": 80,
        "m_lo": 6,
        "m_hi": 50,
        "c_mode": "scalar",
        "c_params": {"value": 2.0},
        "h_mode": "scalar",
        "h_params": {"value": 0.4},
        "s_mode": "seasonal",
        "s_params": {
            "base": 80.0,
            "amp": 0.10,
            "period": 30.0,
            "phase": 0.0,
            "noise_std": 0.0,
        },
        "batch_size": 50,
        "seed_base": 10000,
    },
    {
        "name": "baseline_T60_normalCap_medItems",
        "period": 60,
        "cap_mode": "Normal",
        "cap_params": {"mean": 10000, "std": 500, "clip_lo": 8000, "clip_hi": 12000},
        "n_items": 8,
        "dem_lo": 5,
        "dem_hi": 80,
        "m_lo": 6,
        "m_hi": 50,
        "c_mode": "scalar",
        "c_params": {"value": 2.0},
        "h_mode": "scalar",
        "h_params": {"value": 0.4},
        "s_mode": "seasonal",
        "s_params": {
            "base": 80.0,
            "amp": 0.10,
            "period": 30.0,
            "phase": 0.0,
            "noise_std": 0.0,
        },
        "batch_size": 50,
        "seed_base": 10100,
    },
]
st.session_state["local_presets"] = {p["name"]: p for p in LOCAL_PRESETS}
# ===== END =====

# ======================= Tabs ======================
cap_tab, items_tab, classes_tab, batch_tab, viz_tab = st.tabs(
    ["Capacity", "Items", "Classes", "Batch Runner", "Explore & Visualize"]
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
                summary, orders_txt = solve_instance(
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
                                "solver_version": "lefo_mip_v2",
                            }
                        )
                        .execute()
                    )
                    run_id = run_res.data[0]["id"]
                    # parse orders
                    lines = [ln.strip() for ln in orders_txt if ln.strip()]
                    rows, cur_item = [], None
                    for ln in lines:
                        if ln.startswith("Item"):
                            cur_item = int(ln.split()[1])
                        elif "→" in ln:
                            t_str, qty_str = ln.split("→")
                            t = int(t_str.strip())
                            qty = float(qty_str.strip())
                            rows.append(
                                {
                                    "run_id": run_id,
                                    "item_id": cur_item,
                                    "t": t,
                                    "qty": qty,
                                }
                            )
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

        cdb1, cdb2, cdb3 = st.columns(3)
        with cdb1:
            if st.button("➕ Queue selected DB"):
                existing = {c["name"] for c in st.session_state["classes"]}
                added = 0
                for name in pick_db:
                    spec = db_map[name]["spec"]
                    if spec["name"] not in existing:
                        st.session_state["classes"].append(spec)
                        added += 1
                dedupe_queue_by_name()
                st.success(f"Queued {added} DB class(es).")

        with cdb2:
            if st.button("📥 Queue ALL DB (replace queue)"):
                # replace queue with everything in DB, then dedupe just in case
                st.session_state["classes"] = [r["spec"] for r in db_map.values()]
                dedupe_queue_by_name()
                st.success(
                    f"Queued {len(st.session_state['classes'])} class(es) from DB."
                )

        with cdb3:
            if st.button("🔄 Refresh classes from Supabase"):
                rows = fetch_classes(supabase_client())
                st.session_state["db_classes"] = {r["name"]: r for r in rows}
                # convenience: if queue is empty, auto-fill from DB
                if not st.session_state.get("classes"):
                    st.session_state["classes"] = [r["spec"] for r in rows]
                    dedupe_queue_by_name()
                    st.success("Reloaded & filled queue from DB.")
                else:
                    st.success("Reloaded.")
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
    c_period = st.number_input(
        "Periods", min_value=1, value=int(T), step=1, key="cls_period"
    )

    st.markdown("**Capacity generator (κ_t)**")
    c_cap_mode = st.selectbox(
        "mode", ["Constant", "Uniform", "Normal"], index=1, key="cls_cap_mode"
    )
    cap_params: Dict[str, Any] = {}
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
    else:
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

    st.markdown("**Item count bucket**")
    bucket = st.selectbox(
        "bucket",
        ["Tiny(3)", "Small(5)", "Medium(8)", "Large(15)", "XL(40)", "XXL(100)"],
        index=2,
    )
    bucket_map = {
        "Tiny(3)": 3,
        "Small(5)": 5,
        "Medium(8)": 8,
        "Large(15)": 15,
        "XL(40)": 40,
        "XXL(100)": 100,
    }
    n_items = bucket_map[bucket]

    st.markdown("**Demand and Shelf-life per item**")
    c_dem_lo = st.number_input("demand lo", min_value=0, value=5, step=1)
    c_dem_hi = st.number_input("demand hi", min_value=1, value=80, step=1)
    c_m_lo = st.number_input("m min", min_value=1, value=6, step=1)
    c_m_hi = st.number_input("m max", min_value=1, value=50, step=1)

    st.markdown("**c_it / h_it / s_it generators (applied to all items)**")
    gen_modes = ["scalar", "uniform", "normal", "linear", "seasonal"]
    c_mode = st.selectbox("mode (c_it)", gen_modes, index=0, key="cls_c_mode")
    h_mode = st.selectbox("mode (h_it)", gen_modes, index=0, key="cls_h_mode")
    s_mode = st.selectbox("mode (s_it)", gen_modes, index=4, key="cls_s_mode")

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
                "cap_mode": c_cap_mode,
                "cap_params": cap_params,
                "n_items": int(n_items),
                "dem_lo": int(c_dem_lo),
                "dem_hi": int(c_dem_hi),
                "m_lo": int(c_m_lo),
                "m_hi": int(c_m_hi),
                "c_mode": c_mode,
                "c_params": c_params,
                "h_mode": h_mode,
                "h_params": h_params,
                "s_mode": s_mode,
                "s_params": s_params,
                "batch_size": int(batch_size),
                "seed_base": int(seed_base),
            }
        )

    st.subheader("Queue & Order")
    # try to use drag-and-drop if available
    try:
        from streamlit_sortables import sort_items

        labels = [
            f"{i+1}. {c['name']} (T={c['period']}, items={c['n_items']}, batch={c['batch_size']})"
            for i, c in enumerate(st.session_state["classes"])
        ]
        order = sort_items(labels, direction="vertical", key="class_sort")
        # rebuild order
        new_classes = []
        for lbl in order:
            idx = int(lbl.split(".")[0]) - 1
            new_classes.append(st.session_state["classes"][idx])
        st.session_state["classes"] = new_classes
        st.success("Drag-and-drop ordering active.")
    except Exception:
        # fallback: numeric order
        if st.session_state["classes"]:
            dfq = pd.DataFrame(
                {
                    "order": list(range(1, len(st.session_state["classes"]) + 1)),
                    "name": [c["name"] for c in st.session_state["classes"]],
                    "T": [c["period"] for c in st.session_state["classes"]],
                    "items": [c["n_items"] for c in st.session_state["classes"]],
                    "batch": [c["batch_size"] for c in st.session_state["classes"]],
                }
            )
            edited = st.data_editor(dfq, use_container_width=True, hide_index=True)
            # re-order by 'order'
            edited = edited.sort_values("order")
            new_order = edited["name"].tolist()
            st.session_state["classes"] = sorted(
                st.session_state["classes"], key=lambda c: new_order.index(c["name"])
            )
            st.info(
                "Install streamlit-sortables for drag-and-drop: pip install streamlit-sortables"
            )

    st.dataframe(pd.DataFrame(st.session_state["classes"]))
    sb_for_classes = supabase_client()
    if sb_for_classes and st.button("💾 Save queued classes to Supabase"):
        dedupe_queue_by_name()
        saved = 0
        for cls in st.session_state["classes"]:
            if ensure_class_row(sb_for_classes, cls):  # updates if name exists
                saved += 1
        st.success(f"Saved/updated {saved} class spec(s) in Supabase (unique by name).")

    if st.button("🔄 Reload classes from Supabase"):
        rows = fetch_classes(supabase_client())
        st.session_state["db_classes"] = {r["name"]: r for r in rows}
        st.success("Reloaded.")


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

    def generate_instance_from_class(cls: dict, j: int) -> dict:
        period = int(cls["period"])
        cap = generate_cap_series(
            period, cls["cap_mode"], cls["cap_params"], cls["seed_base"] + 31 * j
        )
        items = {}
        for i in range(cls["n_items"]):
            seed_k = int(cls["seed_base"] + 1000 * j + 17 * i)
            D = list(
                np.random.default_rng(seed_k).integers(
                    int(cls["dem_lo"]), int(cls["dem_hi"]) + 1, size=period
                )
            )
            Mseq = list(
                np.random.default_rng(seed_k + 1).integers(
                    int(cls["m_lo"]), int(cls["m_hi"]) + 1, size=period
                )
            )
            # c,h,s as sequences (or scalar if mode==scalar)
            c_seq = make_series(
                cls["c_mode"], period, dict(cls["c_params"] or {}), seed_k + 2
            )
            h_seq = make_series(
                cls["h_mode"], period, dict(cls["h_params"] or {}), seed_k + 3
            )
            s_seq = make_series(
                cls["s_mode"], period, dict(cls["s_params"] or {}), seed_k + 4
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
        inst = {
            "period": period,
            "items": items,
            # was: "manual_capacity": cap,
            "manual_capacity": [int(x) for x in cap],
            "warehouse_capacity": (float(W_txt) if W_txt.strip() != "" else None),
            "meta": {"origin": "class", "class_key": cls["name"], "class_params": cls},
        }
        return inst

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
                summary, orders_txt = solve_instance(
                    "last_instance.json", time_limit=time_limit, mip_gap=mip_gap
                )

                # log to Supabase (if configured)
                sb = supabase_client()
                if sb is not None:
                    try:
                        cid = class_id_cache.get(cls["name"])
                        inst_payload = {
                            "period": int(inst["period"]),
                            "manual_capacity": inst.get("manual_capacity"),
                            "warehouse_capacity": inst.get("warehouse_capacity"),
                            "data": to_py(inst),
                        }
                        if cid:
                            inst_payload["class_id"] = cid
                            inst["meta"][
                                "class_id"
                            ] = cid  # optional, keeps your meta in sync

                        inst_res = sb.table("instances").insert(inst_payload).execute()
                        instance_id = inst_res.data[0]["id"]

                        run_payload = {
                            "instance_id": instance_id,
                            "time_limit_sec": int(time_limit),
                            "mip_gap": float(mip_gap),
                            "status": int(summary.get("status")),
                            "objective": summary.get("objective"),
                            "best_bound": summary.get("best_bound"),
                            "gap": summary.get("gap"),
                            "runtime_sec": summary.get("runtime_sec"),
                            "solver_version": "lefo_mip_v2",
                        }
                        if cid:
                            run_payload["class_id"] = cid

                        run_res = sb.table("runs").insert(run_payload).execute()
                        run_id = run_res.data[0]["id"]

                        # orders
                        lines = [ln.strip() for ln in orders_txt if ln.strip()]
                        rows, cur_item = [], None
                        for ln in lines:
                            if ln.startswith("Item"):
                                cur_item = int(ln.split()[1])
                            elif "→" in ln:
                                t_str, qty_str = ln.split("→")
                                t = int(t_str.strip())
                                qty = float(qty_str.strip())
                                rows.append(
                                    {
                                        "run_id": run_id,
                                        "item_id": cur_item,
                                        "t": t,
                                        "qty": qty,
                                    }
                                )
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

# ----------------- Explore & Visualize tab -----------------
with viz_tab:
    st.header("Explore & Visualize (Supabase)")
    sb = supabase_client()
    if sb is None:
        st.info("Configure Supabase in sidebar.")
    else:
        # fetch last N runs and their instances, then join in pandas
        limit = st.number_input("Fetch last N runs", min_value=10, value=1000, step=10)
        try:
            runs = (
                sb.table("runs")
                .select("*")
                .order("created_at", desc=True)
                .limit(int(limit))
                .execute()
                .data
            )
            inst_ids = list({r["instance_id"] for r in runs})
            inst = []
            if inst_ids:
                # fetch in chunks to avoid URL size limits
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

            # Build dataframe
            recs = []
            for r in runs:
                I = inst_map.get(r["instance_id"])
                if not I:
                    continue
                data = I.get("data", {})
                meta = data.get("meta") or {}
                items = data.get("items") or {}
                n_items = len(items)
                cap = I.get("manual_capacity") or data.get("manual_capacity") or []
                cap_mean = float(np.mean(cap)) if cap else None
                class_key = meta.get("class_key", "adhoc")
                period = I.get("period") or data.get("period")
                recs.append(
                    {
                        "run_id": r["id"],
                        "created_at": r["created_at"],
                        "instance_id": r["instance_id"],
                        "class_key": class_key,
                        "period": period,
                        "n_items": n_items,
                        "cap_mean": cap_mean,
                        "status": r.get("status"),
                        "objective": r.get("objective"),
                        "best_bound": r.get("best_bound"),
                        "gap": r.get("gap"),
                        "runtime_sec": r.get("runtime_sec"),
                    }
                )
            df = pd.DataFrame(recs)
            if df.empty:
                st.info("No runs found.")
            else:
                st.subheader("Summary table")
                st.dataframe(df)

                c1, c2 = st.columns(2)
                with c1:
                    fig = px.scatter(
                        df,
                        x="n_items",
                        y="runtime_sec",
                        color="class_key",
                        hover_data=["gap", "objective", "status"],
                    )
                    fig.update_layout(height=380, title="Runtime vs #Items")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.scatter(
                        df,
                        x="cap_mean",
                        y="runtime_sec",
                        color="class_key",
                        hover_data=["gap", "objective", "status"],
                    )
                    fig.update_layout(height=380, title="Runtime vs Mean Capacity")
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("3D scatter")
                # choose axes
                x_axis = st.selectbox("X", ["n_items", "period", "cap_mean"])
                y_axis = st.selectbox("Y", ["runtime_sec", "gap", "objective"])
                z_axis = st.selectbox("Z", ["runtime_sec", "gap", "objective"])
                color_by = st.selectbox("Color", ["class_key", "status"])
                fig3d = px.scatter_3d(
                    df.dropna(subset=[x_axis, y_axis, z_axis]),
                    x=x_axis,
                    y=y_axis,
                    z=z_axis,
                    color=color_by,
                    symbol="class_key",
                    hover_data=["run_id", "instance_id"],
                )
                fig3d.update_layout(
                    height=520,
                    scene=dict(
                        xaxis_title=x_axis, yaxis_title=y_axis, zaxis_title=z_axis
                    ),
                )
                st.plotly_chart(fig3d, use_container_width=True)

                st.subheader("Class aggregates")
                agg = (
                    df.groupby("class_key")
                    .agg(
                        runs=("run_id", "count"),
                        n_items_mean=("n_items", "mean"),
                        runtime_p50=("runtime_sec", "median"),
                        runtime_p90=(
                            "runtime_sec",
                            lambda x: (
                                np.percentile(x.dropna(), 90)
                                if len(x.dropna())
                                else None
                            ),
                        ),
                        gap_p95=(
                            "gap",
                            lambda x: (
                                np.percentile(
                                    pd.to_numeric(x, errors="coerce").dropna(), 95
                                )
                                if len(pd.to_numeric(x, errors="coerce").dropna())
                                else None
                            ),
                        ),
                    )
                    .reset_index()
                )
                st.dataframe(agg)
        except Exception as e:
            st.error(f"Supabase query failed: {e}")
