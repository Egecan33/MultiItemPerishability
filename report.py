# report.py
from __future__ import annotations

import argparse
import re
import zlib
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- IO helpers (STRICT schema) ----------

REQUIRED_COLS = ("class_key", "solver_version", "gap")


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
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


# ---------- X-code parsing & maps (from class_key) ----------


def parse_xcode(s: str):
    """
    Parse X-codes of the form X A B C D E F... where:
      - A..E are SINGLE characters (digits or letters)
      - F is one or more DIGITS (e.g., 8, 10, 12)
    Examples:
      X232237, X132H412, X332HB12
    """
    s = str(s)

    # Strict: 5 single alnum chars then 1+ digits
    m = re.search(
        r"X\s*([A-Za-z0-9])([A-Za-z0-9])([A-Za-z0-9])([A-Za-z0-9])([A-Za-z0-9])(\d+)", s
    )
    if m:
        A, B, C, D, E, F = m.groups()
        return {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}

    # Fallback: clean after 'X', take first 5 as A..E, then leading digits as F
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
    mF = re.match(r"(\d+)", rest)  # only the leading digits after E
    F = mF.group(1) if mF else np.nan
    return {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}


# Maps
MAP_A = {"1": 20, "2": 30, "3": 40}
MAP_B = {"1": 10, "2": 20, "3": 30}
MAP_C = {"1": "Loose", "2": "Medium"}  # Medium is tighter
MAP_D = {"1": "Low", "2": "High", "L": "Low", "H": "High", "l": "Low", "h": "High"}


def map_E_label(e):
    if pd.isna(e):
        return np.nan
    s = str(e)
    if s in ("A", "a"):
        return "A(0–T/2)"
    if s in ("B", "b"):
        return "B(0–3T/4)"
    if s in ("C", "c"):
        return "C(5–T/2)"
    if s.isdigit():
        return {
            "1": "E1 (1,10)",
            "2": "E2 (5,15)",
            "3": "E3 (10,20)",
            "4": "E4 (5,25)",
            "5": "E5 (10,30)",
        }.get(s, f"E{s}")
    return s


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


# ---------- Enrichment (STRICT: class_key, solver_version, gap) ----------


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, REQUIRED_COLS)

    out = df.copy()
    out.rename(columns={"class_key": "X_class", "gap": "GAP"}, inplace=True)
    out["GAP"] = pd.to_numeric(out["GAP"], errors="coerce")
    out["solver_bucket"] = out["solver_version"].astype(str)

    # Parse features from class name (X-code)
    parsed = out["X_class"].astype(str).map(parse_xcode)
    xdf = pd.DataFrame(list(parsed))
    out = pd.concat([out, xdf], axis=1)

    out["A_T"] = out["A"].map(MAP_A)
    out["B_n_items"] = out["B"].map(MAP_B)
    out["C_cap"] = out["C"].map(MAP_C)
    out["D_cv"] = out["D"].map(MAP_D)
    out["E_bucket"] = out["E"].map(map_E_label)
    out["F_tbo"] = out["F"].apply(lambda v: int(v) if str(v).isdigit() else np.nan)

    # Ordinals for monotonicity questions
    out["A_code"] = out["A_T"].map({20: 1, 30: 2, 40: 3})
    out["B_code"] = out["B_n_items"].map({10: 1, 20: 2, 30: 3})
    out["C_code"] = out["C"].map({"1": 1, "2": 2})
    out["D_code"] = out["D_cv"].map({"Low": 0, "High": 1})
    out["E_code"] = out["E"].map(map_E_ord)
    out["F_code"] = out["F_tbo"].astype("float")
    return out


# ---------- Analytics ----------


def summarize_levels(df: pd.DataFrame, feature: str, y="GAP"):
    return (
        df.groupby(feature, dropna=False)[y]
        .agg(
            n="count",
            gap_mean="mean",
            gap_median="median",
            gap_std="std",
            gap_min="min",
            gap_max="max",
        )
        .reset_index()
        .sort_values("gap_mean", ascending=False)
    )


def adjacent_deltas(df: pd.DataFrame, feature_code: str, y="GAP"):
    s = df[[feature_code, y]].dropna()
    if s.empty:
        return pd.DataFrame(columns=[feature_code, "gap_mean", "delta_from_prev"])
    tbl = (
        s.groupby(feature_code)[y]
        .mean()
        .sort_index()
        .reset_index()
        .rename(columns={y: "gap_mean"})
    )
    tbl["delta_from_prev"] = tbl["gap_mean"].diff()
    return tbl


def spearman_monotone(df: pd.DataFrame, code_cols, y="GAP"):
    rows = []
    for col in code_cols:
        s = df[[col, y]].dropna()
        if s.empty or s[col].nunique() <= 1 or s[y].nunique() <= 1:
            rho, n = np.nan, len(s)
        else:
            rho = s[col].rank().corr(s[y].rank())
            n = len(s)
        rows.append({"feature_code": col, "spearman_rho": rho, "n": n})
    return pd.DataFrame(rows)


# ---------- OLS (hardened) ----------


def _to_numeric_matrix(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(float)
    X = (
        X.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    if not X.empty:
        variances = X.var(axis=0)
        keep = variances.index[variances > 0]
        X = X[keep]
    return X.astype("float64")


def fit_ols(X: pd.DataFrame, y: pd.Series):
    X = _to_numeric_matrix(X)
    y = (
        pd.to_numeric(y, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float64")
    )
    X = pd.concat([pd.Series(1.0, index=X.index, name="Intercept"), X], axis=1)
    X_arr, y_arr = X.to_numpy(np.float64), y.to_numpy(np.float64)
    XtX = X_arr.T @ X_arr
    beta = np.linalg.pinv(XtX) @ (X_arr.T @ y_arr)
    coef = pd.Series(beta, index=X.columns, name="coef")
    y_hat = X_arr @ beta
    ss_tot = float(((y_arr - y_arr.mean()) ** 2).sum())
    ss_res = float(((y_arr - y_hat) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return coef, r2


def ols_views(df: pd.DataFrame):
    dd = df.dropna(subset=["GAP"]).copy()

    # One-hot for C,D,E (keep ALL levels so nothing “skips”); ordinal A,B,F
    X1 = pd.concat(
        [
            dd[["A_code", "B_code", "F_code"]].astype(float),
            pd.get_dummies(dd["C_cap"], prefix="C_cap", dtype=float),
            pd.get_dummies(dd["D_cv"], prefix="D_cv", dtype=float),
            pd.get_dummies(dd["E_bucket"], prefix="E_bucket", dtype=float),
        ],
        axis=1,
    )
    coef1, r21 = fit_ols(X1, dd["GAP"].astype(float))
    t1 = coef1.reset_index().rename(columns={"index": "feature", "coef": "coef"})
    t1["model"] = "OLS_onehot_all_CDE"
    t1["R2"] = r21

    # Pure ordinal (answers “increase/lower”)
    X2 = dd[["A_code", "B_code", "C_code", "D_code", "E_code", "F_code"]]
    coef2, r22 = fit_ols(X2, dd["GAP"].astype(float))
    t2 = coef2.reset_index().rename(columns={"index": "feature", "coef": "coef"})
    t2["model"] = "OLS_ordinal"
    t2["R2"] = r22
    return t1, t2


# ---------- Uniform XLSX helpers ----------


def _sanitize_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s))


def _sheet_name(prefix: str, raw_tag: str, suffix: str) -> str:
    """
    Build an Excel-safe sheet name (<=31 chars).
    If truncation is needed, keep a short CRC to avoid collisions.
    """
    tag = _sanitize_tag(raw_tag)
    max_len = 31 - len(prefix) - len(suffix)
    if max_len < 1:
        h = format(zlib.crc32(tag.encode()), "x")[:6]
        return f"{prefix}{h[:1]}{suffix}"
    if len(tag) <= max_len:
        return f"{prefix}{tag}{suffix}"
    h = format(zlib.crc32(tag.encode()), "x")[:6]
    keep = max_len - 7  # room for "_"+6 hash
    keep = max(1, keep)
    tag_short = f"{tag[:keep]}_{h}"
    return f"{prefix}{tag_short}{suffix}"


def _levels_stacked(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    rows = []
    for feat in feat_cols:
        t = summarize_levels(df, feat)
        level_col = t.columns[0]  # the feature column name (e.g., 'A_T')
        t = t.rename(columns={level_col: "feature_level"})
        t.insert(0, "feature", feat)
        t = t[
            [
                "feature",
                "feature_level",
                "n",
                "gap_mean",
                "gap_median",
                "gap_std",
                "gap_min",
                "gap_max",
            ]
        ]
        rows.append(t)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(
        columns=[
            "feature",
            "feature_level",
            "n",
            "gap_mean",
            "gap_median",
            "gap_std",
            "gap_min",
            "gap_max",
        ]
    )


def _deltas_stacked(df: pd.DataFrame, code_cols: list[str]) -> pd.DataFrame:
    rows = []
    for code in code_cols:
        t = adjacent_deltas(df, code)
        t = t.rename(columns={code: "level"})
        t.insert(0, "feature_code", code)
        t = t[["feature_code", "level", "gap_mean", "delta_from_prev"]]
        rows.append(t)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(
        columns=["feature_code", "level", "gap_mean", "delta_from_prev"]
    )


def _spearman_uniform(df: pd.DataFrame, code_cols: list[str]) -> pd.DataFrame:
    t = spearman_monotone(df, code_cols)
    return t[["feature_code", "spearman_rho", "n"]]


def _ols_uniform(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["feature", "coef", "model", "R2"])
    t1, t2 = ols_views(df)
    out = pd.concat([t1, t2], ignore_index=True)
    return out[["feature", "coef", "model", "R2"]]


# ---------- Main ----------


def main():
    ap = argparse.ArgumentParser(
        description="GAP drivers from X(A..F) (in class_key) on a single CSV, split by solver_version, with uniform tabs."
    )
    ap.add_argument(
        "--csv", default="data.csv", help="Input CSV path (default: data.csv)"
    )
    ap.add_argument("--out", default="out", help="Output directory (default: out/)")
    ap.add_argument("--xlsx", action="store_true", help="Also write an XLSX workbook")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load + enforce schema + enrich
    df_raw = read_any(Path(args.csv))
    require_columns(df_raw, REQUIRED_COLS)
    df = enrich(df_raw)

    # Buckets: per solver_version + ALL
    solver_values = list(
        pd.Series(df["solver_bucket"].unique()).astype(str).sort_values()
    )
    buckets = solver_values + ["ALL"]

    # Feature lists
    code_cols = ["A_code", "B_code", "C_code", "D_code", "E_code", "F_code"]
    feat_cols = ["A_T", "B_n_items", "C_cap", "D_cv", "E_bucket", "F_tbo"]

    # Prepare uniform tables for each bucket
    levels_by_solver: dict[str, pd.DataFrame] = {}
    deltas_by_solver: dict[str, pd.DataFrame] = {}
    spearman_by_solver: dict[str, pd.DataFrame] = {}
    ols_by_solver: dict[str, pd.DataFrame] = {}

    for bucket in buckets:
        dff = df if bucket == "ALL" else df[df["solver_bucket"] == bucket]
        levels_by_solver[bucket] = _levels_stacked(dff, feat_cols)
        deltas_by_solver[bucket] = _deltas_stacked(dff, code_cols)
        spearman_by_solver[bucket] = _spearman_uniform(dff, code_cols)
        ols_by_solver[bucket] = _ols_uniform(dff)

    # Write CSV exports (uniform + sanitized filenames)
    df.to_csv(outdir / "parsed_enriched.csv", index=False)
    for bucket in buckets:
        tag = _sanitize_tag(bucket)
        levels_by_solver[bucket].to_csv(outdir / f"levels_{tag}.csv", index=False)
        deltas_by_solver[bucket].to_csv(outdir / f"deltas_{tag}.csv", index=False)
        spearman_by_solver[bucket].to_csv(outdir / f"spearman_{tag}.csv", index=False)
        ols_by_solver[bucket].to_csv(outdir / f"ols_{tag}.csv", index=False)

    # Write XLSX with identical tab structures per solver (sheet names <= 31 chars)
    if args.xlsx:
        with pd.ExcelWriter(
            outdir / "gap_feature_effect_report.xlsx", engine="xlsxwriter"
        ) as xw:
            df.to_excel(xw, sheet_name="parsed_enriched", index=False)

            for bucket in buckets:
                xw_sheet = _sheet_name("01_", bucket, "_levels")
                levels_by_solver[bucket].to_excel(xw, sheet_name=xw_sheet, index=False)

                xw_sheet = _sheet_name("02_", bucket, "_deltas")
                deltas_by_solver[bucket].to_excel(xw, sheet_name=xw_sheet, index=False)

                xw_sheet = _sheet_name("03_", bucket, "_spearman")
                spearman_by_solver[bucket].to_excel(
                    xw, sheet_name=xw_sheet, index=False
                )

                xw_sheet = _sheet_name("04_", bucket, "_ols")
                ols_by_solver[bucket].to_excel(xw, sheet_name=xw_sheet, index=False)

    print(f"OK → {outdir/'parsed_enriched.csv'}")
    if args.xlsx:
        print(f"OK → {outdir/'gap_feature_effect_report.xlsx'}")


if __name__ == "__main__":
    main()
