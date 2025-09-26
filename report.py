# report.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- IO helpers ----------


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


def guess_class_col(df: pd.DataFrame) -> str:
    for cand in [
        "X_class",
        "class",
        "Class",
        "CLASS",
        "name",
        "Name",
        "instance",
        "Instance",
        "X",
    ]:
        if cand in df.columns:
            return cand
    best, best_ratio = None, -1.0
    for col in df.columns:
        try:
            ratio = df[col].astype(str).str.strip().str.startswith("X").mean()
        except Exception:
            ratio = 0.0
        if ratio > best_ratio:
            best_ratio, best = ratio, col
    return best or df.columns[0]


def guess_gap_col(df: pd.DataFrame) -> str:
    for cand in [
        "GAP",
        "gap",
        "Gap",
        "best_gap",
        "obj_gap",
        "relative_gap",
        "rel_gap",
        "GAP%",
    ]:
        if cand in df.columns:
            return cand
    for col in df.columns:
        if "gap" in col.lower():
            return col
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[-1] if nums else df.columns[-1]


# ---------- X-code parsing & maps ----------


def parse_xcode(s: str):
    """
    Parse X-codes of the form X A B C D E F... where:
      - A..E are SINGLE characters (digits or letters)
      - F is one or more DIGITS (e.g., 8, 10, 12)
    Examples:
      X232237     -> A=2,B=3,C=2,D=2,E=3,F=7
      X132H412    -> A=1,B=3,C=2,D=H,E=4,F=12
      X23 2 2 H4 10 -> A=2,B=3,C=2,D=H,E=4,F=10
    Anything after F's digits is ignored.
    """
    s = str(s)

    # Strict: 5 single alnum chars then 1+ digits
    m = re.search(
        r"X\s*([A-Za-z0-9])([A-Za-z0-9])([A-Za-z0-9])([A-Za-z0-9])([A-Za-z0-9])(\d+)", s
    )
    if m:
        A, B, C, D, E, F = m.groups()
        return {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}

    # Fallback: clean payload; take first 5 as A..E, then leading digits as F
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


# ---------- Enrichment ----------


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    cls = guess_class_col(df)
    gapc = guess_gap_col(df)
    out = df.copy()
    out.rename(columns={cls: "X_class", gapc: "GAP"}, inplace=True)
    out["GAP"] = pd.to_numeric(out["GAP"], errors="coerce")

    parsed = out["X_class"].astype(str).map(parse_xcode)
    xdf = pd.DataFrame(list(parsed))
    out = pd.concat([out, xdf], axis=1)

    out["A_T"] = out["A"].map(MAP_A)
    out["B_n_items"] = out["B"].map(MAP_B)
    out["C_cap"] = out["C"].map(MAP_C)
    out["D_cv"] = out["D"].map(MAP_D)
    out["E_bucket"] = out["E"].map(map_E_label)
    out["F_tbo"] = out["F"].apply(lambda v: int(v) if str(v).isdigit() else np.nan)

    # Ordinal codes for monotone “increase/lower” questions
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
            pd.get_dummies(
                dd["E_bucket"], prefix="E_bucket", dtype=float
            ),  # A/B/C + E1..E5
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


# ---------- Report writers ----------


def write_consolidated_csv(
    out_path: Path,
    per_feat: dict[str, pd.DataFrame],
    adj_deltas: dict[str, pd.DataFrame],
    spear: pd.DataFrame,
    ols_tbl: pd.DataFrame,
):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "# GAP feature effect report v3 (A..F; E_bucket fixed incl. A/B/C & 1..5)\n"
        )
        for name, tbl in per_feat.items():
            f.write(f"\n## {name}\n")
            tbl.to_csv(f, index=False)
        f.write("\n## Adjacent deltas (monotone)\n")
        for name, tbl in adj_deltas.items():
            f.write(f"\n### {name}\n")
            tbl.to_csv(f, index=False)
        f.write("\n## Spearman (codes vs GAP)\n")
        spear.to_csv(f, index=False)
        f.write("\n## OLS (onehot-all & ordinal)\n")
        ols_tbl.to_csv(f, index=False)


def write_xlsx(
    out_xlsx: Path,
    per_feat: dict[str, pd.DataFrame],
    adj_deltas: dict[str, pd.DataFrame],
    spear: pd.DataFrame,
    ols_tbl: pd.DataFrame,
    enr1: pd.DataFrame,
    enr10: pd.DataFrame,
):
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
        enr1.to_excel(xw, sheet_name="parsed_1min", index=False)
        enr10.to_excel(xw, sheet_name="parsed_10min", index=False)
        for name, tbl in per_feat.items():
            sheet = name[:31] if len(name) > 31 else name
            tbl.to_excel(xw, sheet_name=sheet, index=False)
        spear.to_excel(xw, sheet_name="spearman", index=False)
        ols_tbl.to_excel(xw, sheet_name="ols", index=False)
        for name, tbl in adj_deltas.items():
            sheet = name[:31] if len(name) > 31 else name
            tbl.to_excel(xw, sheet_name=sheet, index=False)


# ---------- Main ----------


def main():
    ap = argparse.ArgumentParser(
        description="GAP drivers from X(A..F) on 1 & 10 datasets."
    )
    ap.add_argument("--one", default="1.csv", help="1-minute CSV path (default: 1.csv)")
    ap.add_argument(
        "--ten", default="10.csv", help="10-minute CSV path (default: 10.csv)"
    )
    ap.add_argument("--out", default="out", help="Output directory (default: out/)")
    ap.add_argument("--xlsx", action="store_true", help="Also write an XLSX workbook")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load + enrich
    df1 = enrich(read_any(Path(args.one)))
    df10 = enrich(read_any(Path(args.ten)))

    # Per-feature summaries + deltas
    per_feat, adj_deltas = {}, {}
    for key, df in [("1min", df1), ("10min", df10)]:
        for feat in ["A_T", "B_n_items", "C_cap", "D_cv", "E_bucket", "F_tbo"]:
            per_feat[f"{key}__{feat}"] = summarize_levels(df, feat)
        for code in ["A_code", "B_code", "C_code", "D_code", "E_code", "F_code"]:
            adj_deltas[f"{key}__{code}_adjacent_deltas"] = adjacent_deltas(df, code)

    # Spearman + OLS
    spear = pd.concat(
        [
            spearman_monotone(
                df1, ["A_code", "B_code", "C_code", "D_code", "E_code", "F_code"]
            ).assign(dataset="1min"),
            spearman_monotone(
                df10, ["A_code", "B_code", "C_code", "D_code", "E_code", "F_code"]
            ).assign(dataset="10min"),
        ],
        ignore_index=True,
    )[["dataset", "feature_code", "spearman_rho", "n"]]

    ols_parts = []
    for key, df in [("1min", df1), ("10min", df10)]:
        t1, t2 = ols_views(df)
        t1.insert(0, "dataset", key)
        t2.insert(0, "dataset", key)
        ols_parts.extend([t1, t2])
    ols_tbl = pd.concat(ols_parts, ignore_index=True)

    # Write outputs
    df1.to_csv(outdir / "parsed_1min_enriched.csv", index=False)
    df10.to_csv(outdir / "parsed_10min_enriched.csv", index=False)
    write_consolidated_csv(
        outdir / "gap_feature_effect_report_v3.csv",
        per_feat,
        adj_deltas,
        spear,
        ols_tbl,
    )

    if args.xlsx:
        write_xlsx(
            outdir / "gap_feature_effect_report_v3.xlsx",
            per_feat,
            adj_deltas,
            spear,
            ols_tbl,
            df1,
            df10,
        )

    print(f"OK → {outdir/'gap_feature_effect_report_v3.csv'}")
    print(f"OK → {outdir/'parsed_1min_enriched.csv'}")
    print(f"OK → {outdir/'parsed_10min_enriched.csv'}")
    if args.xlsx:
        print(f"OK → {outdir/'gap_feature_effect_report_v3.xlsx'}")


if __name__ == "__main__":
    main()
