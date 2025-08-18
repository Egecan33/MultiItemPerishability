# streamlit_app.py
import json
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd

from mip.generator import build_lot, Lot, SETUP_SEQ_ENABLE, ITEM_CAP_SEQ_POLICY

st.set_page_config(page_title="Perishable Lot-Sizing — Instance Builder", layout="wide")

st.title("Perishable Lot-Sizing — Instance Generator (Freshest-First MIP)")

with st.sidebar:
    st.header("Global Settings")
    seed = st.number_input("Random Seed", min_value=0, value=0, step=1)
    period = st.number_input("Planning Periods (T)", min_value=1, value=60, step=1)
    n_items = st.number_input("# Items (I)", min_value=1, value=6, step=1)
    dem_lo = st.number_input("Demand min", min_value=0, value=5, step=1)
    dem_hi = st.number_input("Demand max", min_value=1, value=80, step=1)

    st.subheader("Capacity")
    use_manual_cap = st.checkbox("Use manual global capacity", value=True)
    manual_cap_val = st.number_input(
        "Manual capacity value per period", min_value=1, value=10000, step=1
    )
    warehouse_capacity = st.text_input("Warehouse capacity W (blank = None)", value="")

    st.subheader("Setup & Item Caps")
    setup_seq_enable = st.checkbox(
        "Enable time-varying setups s_{i,t}", value=SETUP_SEQ_ENABLE
    )
    cap_policy = st.selectbox(
        "Per-item cap_{i,t} policy",
        options=["none", "demand_pad", "uniform_range"],
        index=["none", "demand_pad", "uniform_range"].index(ITEM_CAP_SEQ_POLICY),
    )

st.markdown("### Item Specs")
st.caption(
    "Each row: (item_id, setup_base_or_range, c, h, (m_lo, m_hi))\n- `setup_base_or_range` can be a number or a JSON pair, e.g., `80` or `[50,120]`"
)

default_rows = []
for i in range(int(n_items)):
    default_rows.append(
        [
            i,
            "[80, 120]" if i % 2 == 0 else 100,
            2.0 + (i % 3),
            0.4 + 0.1 * (i % 2),
            "[6, 50]",
        ]
    )

df = pd.DataFrame(
    default_rows,
    columns=["item_id", "setup_base_or_range", "c", "h", "shelf_range_(m_lo,m_hi)"],
)
edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)


def parse_range(v):
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    try:
        arr = json.loads(s)
        if isinstance(arr, list) and len(arr) == 2:
            return (float(arr[0]), float(arr[1]))
    except Exception:
        pass
    # fallback: try as scalar float string
    try:
        return float(s)
    except Exception:
        return float(100.0)


def parse_shelf(v):
    s = str(v).strip()
    try:
        arr = json.loads(s)
        if isinstance(arr, list) and len(arr) == 2:
            return (int(arr[0]), int(arr[1]))
    except Exception:
        pass
    return (3, 5)


btn = st.button("Generate Instance", type="primary")

if btn:
    specs = []
    for _, row in edited.iterrows():
        i = int(row["item_id"])
        setup_rng = parse_range(row["setup_base_or_range"])
        c = float(row["c"])
        h = float(row["h"])
        m_rng = parse_shelf(row["shelf_range_(m_lo,m_hi)"])
        specs.append((i, setup_rng, c, h, m_rng))

    manual_capacity = [int(manual_cap_val)] * int(period) if use_manual_cap else None
    W = float(warehouse_capacity) if warehouse_capacity.strip() != "" else None

    lot = build_lot(
        period=int(period),
        demand_range=(int(dem_lo), int(dem_hi)),
        specs=specs,
        manual_capacity=manual_capacity,
        setup_seq_enable=bool(setup_seq_enable),
        item_cap_seq_policy=str(cap_policy),
        default_shelf_rng=(3, 5),
        warehouse_capacity=W,
        seed=int(seed),
    )

    # Preview tables
    st.success("Instance generated.")
    with st.expander("Global capacity (κ_t)"):
        st.dataframe(
            pd.DataFrame({"t": list(range(lot.period)), "kappa_t": lot.capacity})
        )

    for i, it in lot.items.items():
        with st.expander(f"Item {i}"):
            df_i = pd.DataFrame(
                {
                    "t": list(range(lot.period)),
                    "demand": it.demand,
                    "shelf_m_it": it.shelf_seq,
                    "setup_s_it": (
                        it.setup
                        if isinstance(it.setup, list)
                        else [it.setup] * lot.period
                    ),
                    "cap_i_t": (
                        it.cap_seq if it.cap_seq is not None else [None] * lot.period
                    ),
                }
            )
            st.dataframe(df_i)

    # Save
    fname = st.text_input("Output file name", value="last_instance.json")
    if st.button("Save JSON"):
        Path(fname).write_text("", encoding="utf-8")  # create if missing perms
        lot.to_mip_json(fname)
        st.success(f"Saved → {Path(fname).resolve()}")

st.divider()
st.caption(
    "Tip: After saving JSON, run the solver: `python -m mip.solver_mip_lefo --instance last_instance.json`"
)
