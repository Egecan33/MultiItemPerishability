import os, json
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import streamlit as st
from supabase import create_client, Client

from mip.solver_mip_lefo import solve_instance

# ======================= App Config =======================
DEFAULT_URL = "https://btqqbsnjcsgjvgpuutiw.supabase.co"
DEFAULT_ANON = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ0cXFic25qY3NnanZncHV1dGl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU1NDMwODMsImV4cCI6MjA3MTExOTA4M30.gissvSrKruPsYJOHOoLqfzQGLrB4oFVckVwhrUpGJXU"

st.set_page_config(page_title="Perishable Lot-Sizing (LEFO MIP)", layout="wide")
st.title("Perishable Lot-Sizing — Advanced Generator + LEFO MIP")


# ======================= Helpers ==========================
def rng(seed: int):
    return np.random.default_rng(int(seed))


def clip_list(xs, lo=None, hi=None):
    if lo is not None:
        xs = np.maximum(xs, lo)
    if hi is not None:
        xs = np.minimum(xs, hi)
    return xs


def make_series(mode: str, T: int, params: dict, seed: int):
    """
    Returns a list[float] of length T given:
      mode in {"scalar","uniform","normal","linear","seasonal","manual"}
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
        arr = g.normal(mu, sd, size=T)
        arr = clip_list(arr, lo, hi)
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


# ======================= UI: Sidebar ======================
with st.sidebar:
    st.header("Global Settings")
    T = st.number_input("Periods T", min_value=1, value=60, step=1)
    SEED = st.number_input("Global seed", min_value=0, value=0, step=1)

    st.markdown("---")
    st.subheader("Demand")
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

# ======================= UI: Capacity tab =================
st.header("Global Capacity cap_t")
cap_tab, items_tab, run_tab = st.tabs(["Capacity", "Items", "Generate & Solve"])

with cap_tab:
    cap_mode = st.radio(
        "Capacity series mode",
        ["Constant", "Uniform", "Normal", "Manual list/grid"],
        horizontal=True,
    )
    cap_series = None
    if cap_mode == "Constant":
        v = st.number_input("cap value", min_value=0, value=10000, step=100)
        cap_series = [int(v)] * T
    elif cap_mode == "Uniform":
        lo = st.number_input("lo", min_value=0, value=9000, step=100)
        hi = st.number_input("hi", min_value=0, value=11000, step=100)
        cap_series = list(
            np.random.default_rng(SEED + 101).integers(lo, hi + 1, size=T)
        )
    elif cap_mode == "Normal":
        mu = st.number_input("mean", min_value=0, value=10000, step=100)
        sd = st.number_input("std", min_value=0, value=500, step=10)
        clip_lo = st.number_input("clip_lo", min_value=0, value=0, step=100)
        clip_hi = st.number_input("clip_hi", min_value=0, value=20000, step=100)
        arr = np.random.default_rng(SEED + 102).normal(mu, sd, size=T)
        arr = clip_list(arr, clip_lo, clip_hi).round().astype(int)
        cap_series = list(arr)
    else:
        # Manual editor
        import pandas as pd

        df = pd.DataFrame({"t": list(range(T)), "cap_t": [10000] * T})
        edited = st.data_editor(
            df, use_container_width=True, hide_index=True, num_rows="fixed"
        )
        cap_series = [int(x) for x in edited["cap_t"].tolist()]

    st.line_chart(cap_series, height=140)
    st.caption("Preview of cap_t")


# ======================= Item dataclass ===================
@dataclass
class ItemSpec:
    item_id: int
    # costs can be scalar or series generated from chosen mode
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


# ======================= UI: Items tab ====================
if "item_specs" not in st.session_state:
    st.session_state["item_specs"] = [default_item(0), default_item(1), default_item(2)]

with items_tab:
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
                rcol = st.container()
                it.remove = rcol.checkbox("Remove", key=f"rm_{idx}", value=False)

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
            else:  # manual
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

    # Remove checked items
    st.session_state["item_specs"] = [
        it for it in st.session_state["item_specs"] if not it.remove
    ]

# ======================= Generate & Solve =================
with run_tab:
    left, right = st.columns([1, 1])

    def parse_manual_list(raw: str, T: int):
        if not raw.strip():
            return []
        # allow commas/spaces/newlines
        parts = raw.replace(",", " ").split()
        vals = [float(x) for x in parts]
        if len(vals) != T:
            st.error(f"Manual list length {len(vals)} != T={T}")
            return []
        return vals

    if st.button("🛠️ Generate instance"):
        items = {}
        for k, it in enumerate(st.session_state["item_specs"]):
            seed_k = int(SEED + 1000 + 17 * k)

            # demand (integer uniform)
            D = list(
                np.random.default_rng(seed_k).integers(
                    int(it.dem_lo), int(it.dem_hi) + 1, size=T
                )
            )

            # shelf life m_{i,t}
            if it.m_min == it.m_max:
                Mseq = [int(it.m_min)] * T
            else:
                Mseq = list(
                    np.random.default_rng(seed_k + 1).integers(
                        int(it.m_min), int(it.m_max) + 1, size=T
                    )
                )

            # c_{i,t}
            c_params = dict(it.c_params or {})
            if it.c_mode == "manual":
                c_list = parse_manual_list(c_params.get("raw", ""), T)
                c = c_list if c_list else [float(c_params.get("value", 2.0))] * T
            else:
                c = make_series(it.c_mode, T, c_params, seed_k + 2)
            # if scalar mode: store scalar; else list
            c_out = (
                float(c_params.get("value", 0.0))
                if it.c_mode == "scalar"
                else [float(x) for x in c]
            )

            # h_{i,t}
            h_params = dict(it.h_params or {})
            if it.h_mode == "manual":
                h_list = parse_manual_list(h_params.get("raw", ""), T)
                h = h_list if h_list else [float(h_params.get("value", 0.4))] * T
            else:
                h = make_series(it.h_mode, T, h_params, seed_k + 3)
            h_out = (
                float(h_params.get("value", 0.0))
                if it.h_mode == "scalar"
                else [float(x) for x in h]
            )

            # s_{i,t}
            s_params = dict(it.s_params or {})
            if it.s_mode == "manual":
                s_list = parse_manual_list(s_params.get("raw", ""), T)
                s = s_list if s_list else [float(s_params.get("value", 80.0))] * T
            else:
                s = make_series(it.s_mode, T, s_params, seed_k + 4)
            s_out = (
                float(s_params.get("value", 0.0))
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
        }
        Path("last_instance.json").write_text(
            json.dumps(inst, indent=2), encoding="utf-8"
        )
        st.success("Instance saved → last_instance.json")
        st.code(Path("last_instance.json").read_text()[:3000], language="json")
        with right:
            st.line_chart(cap_series, height=140)

    st.markdown("---")
    if st.button("🚀 Solve with LEFO MIP"):
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

            # Optional logging
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
                                "data": inst_json,
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

                    # parse orders into rows (item, t, qty)
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
