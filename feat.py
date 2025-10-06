# report.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

# --------- STRICT schema ----------
REQUIRED_COLS = ("class_key", "solver_version", "gap")


# --------- IO ----------
def read_any(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1254", "iso-8859-9", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            if len(df.columns) >= 1:
                return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read {path}: {last_err}")


def require_columns(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available: {list(df.columns)}"
        )


# --------- X-code parsing ----------
def parse_xcode(s: str):
    """
    Parse class_key of form X A B C D E F... (A..E single chars, F digits).
    Examples: X232237, X132H412, X332HB12
    """
    s = str(s)
    m = re.search(
        r"X\s*([A-Za-z0-9])([A-Za-z0-9])([A-Za-z0-9])([A-Za-z0-9])([A-Za-z0-9])(\d+)", s
    )
    if m:
        A, B, C, D, E, F = m.groups()
        return {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}
    m2 = re.search(r"X([A-Za-z0-9_\-\s]+)", s)
    if not m2:
        return dict(A=np.nan, B=np.nan, C=np.nan, D=np.nan, E=np.nan, F=np.nan)
    payload = re.sub(r"[^A-Za-z0-9]", "", m2.group(1))
    if len(payload) < 5:
        A = payload[0] if len(payload) > 0 else np.nan
        B = payload[1] if len(payload) > 1 else np.nan
        C = payload[2] if len(payload) > 2 else np.nan
        D = payload[3] if len(payload) > 3 else np.nan
        E = payload[4] if len(payload) > 4 else np.nan
        F = np.nan
        return {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}
    A, B, C, D, E = payload[:5]
    rest = payload[5:]
    mF = re.match(r"(\d+)", rest)
    F = mF.group(1) if mF else np.nan
    return {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}


# --------- Mappings -> codes only (keep it simple) ----------
MAP_A = {"1": 20, "2": 30, "3": 40}
MAP_B = {"1": 10, "2": 20, "3": 30}


def map_E_ord(e):
    if pd.isna(e):
        return np.nan
    s = str(e)
    if s in ("A", "a"):
        return 1
    if s in ("B", "b"):
        return 2
    if s in ("C", "c"):
        return 3
    if s.isdigit():
        try:
            v = int(s)
            return v if 1 <= v <= 10 else np.nan
        except Exception:
            return np.nan
    return np.nan


# --------- Enrichment (only what's needed) ----------
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, REQUIRED_COLS)
    out = df.copy()
    out.rename(columns={"class_key": "X_class", "gap": "GAP"}, inplace=True)
    out["GAP"] = pd.to_numeric(out["GAP"], errors="coerce")
    out["solver_bucket"] = out["solver_version"].astype(str)

    parsed = out["X_class"].astype(str).map(parse_xcode)
    xdf = pd.DataFrame(list(parsed))
    out = pd.concat([out, xdf], axis=1)

    # Build ordinal codes A..F
    out["A_T"] = out["A"].map(MAP_A)  # 20/30/40
    out["B_n_items"] = out["B"].map(MAP_B)  # 10/20/30
    out["A_code"] = out["A_T"].map({20: 1, 30: 2, 40: 3})  # 1..3
    out["B_code"] = out["B_n_items"].map({10: 1, 20: 2, 30: 3})
    out["C_code"] = out["C"].map({"1": 1, "2": 2})

    D_map = {"1": "Low", "2": "High", "L": "Low", "H": "High", "l": "Low", "h": "High"}
    out["D_val"] = out["D"].map(D_map)
    out["D_code"] = out["D_val"].map({"Low": 0, "High": 1})

    out["E_code"] = out["E"].map(map_E_ord)
    out["F_code"] = out["F"].apply(lambda v: float(v) if str(v).isdigit() else np.nan)

    return out


# --------- Simple effects ----------
def _effect_by_code(y: pd.Series, xcode: pd.Series) -> dict:
    """
    Simple effect = mean(GAP at highest code) - mean(GAP at lowest code).
    """
    t = pd.concat({"code": xcode, "gap": y}, axis=1).dropna()
    if t.empty or t["code"].nunique() < 2:
        return {
            "n_rows": int(len(t)),
            "n_levels": int(t["code"].nunique()),
            "level_min": np.nan,
            "level_max": np.nan,
            "mean_lowest": np.nan,
            "mean_highest": np.nan,
            "effect": np.nan,
        }
    # group by code (sorted)
    g = (
        t.groupby("code")["gap"]
        .agg(n="count", mean_gap="mean")
        .reset_index()
        .sort_values("code")
    )
    level_min = float(g["code"].min())
    level_max = float(g["code"].max())
    mean_lowest = float(g.loc[g["code"].idxmin(), "mean_gap"])
    mean_highest = float(g.loc[g["code"].idxmax(), "mean_gap"])
    effect = float(mean_highest - mean_lowest)
    return {
        "n_rows": int(len(t)),
        "n_levels": int(g.shape[0]),
        "level_min": level_min,
        "level_max": level_max,
        "mean_lowest": mean_lowest,
        "mean_highest": mean_highest,
        "effect": effect,
    }


def compute_simple_effects(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - effects summary (one row per scope x feature_code)
      - level means (all scopes stacked)
    """
    code_cols = ["A_code", "B_code", "C_code", "D_code", "E_code", "F_code"]
    scopes = ["ALL"] + list(
        pd.Series(df["solver_bucket"].unique()).astype(str).sort_values()
    )

    effects_rows = []
    level_means_rows = []

    for scope in scopes:
        dff = df if scope == "ALL" else df[df["solver_bucket"] == scope]
        for code_col in code_cols:
            stats = _effect_by_code(dff["GAP"], dff[code_col])
            row = {"scope": scope, "feature_code": code_col, **stats}
            effects_rows.append(row)

            # also record per-level means for visibility
            s = dff[[code_col, "GAP"]].dropna()
            if not s.empty:
                g = (
                    s.groupby(code_col)["GAP"]
                    .agg(n="count", mean_gap="mean")
                    .reset_index()
                )
                g = g.rename(columns={code_col: "level"})
                g.insert(0, "feature_code", code_col)
                g.insert(0, "scope", scope)
                level_means_rows.append(g)

    effects = (
        pd.DataFrame(effects_rows)[
            [
                "scope",
                "feature_code",
                "n_rows",
                "n_levels",
                "level_min",
                "level_max",
                "mean_lowest",
                "mean_highest",
                "effect",
            ]
        ]
        .sort_values(["scope", "feature_code"])
        .reset_index(drop=True)
    )

    level_means = (
        pd.concat(level_means_rows, ignore_index=True)
        if level_means_rows
        else pd.DataFrame(columns=["scope", "feature_code", "level", "n", "mean_gap"])
    )
    return effects, level_means


# --------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Simple feature effects: A..F effect on GAP (highest level mean − lowest level mean)."
    )
    ap.add_argument(
        "--csv",
        required=True,
        help="Input CSV (must contain: class_key, solver_version, gap)",
    )
    ap.add_argument("--out", default="out", help="Output directory (default: out/)")
    ap.add_argument("--xlsx", action="store_true", help="Also write an XLSX workbook")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df_raw = read_any(Path(args.csv))
    require_columns(df_raw, REQUIRED_COLS)
    df = enrich(df_raw)

    effects, level_means = compute_simple_effects(df)

    # Write CSV
    effects_path = outdir / "feature_effects_simple.csv"
    effects.to_csv(effects_path, index=False)

    # Optional XLSX
    if args.xlsx:
        with pd.ExcelWriter(
            outdir / "feature_effects_simple.xlsx", engine="xlsxwriter"
        ) as xw:
            effects.to_excel(xw, sheet_name="effects", index=False)
            level_means.to_excel(xw, sheet_name="level_means", index=False)

    # Console summary
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(effects)

    print(f"\nOK → {effects_path}")
    if args.xlsx:
        print(f"OK → {outdir/'feature_effects_simple.xlsx'}")


if __name__ == "__main__":
    main()
