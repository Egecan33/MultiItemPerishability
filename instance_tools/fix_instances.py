#!/usr/bin/env python3
"""
Ensure feasibility of 'instances' in Supabase by period-wise zeroing of demands.

PRIORITY + SAFETY (NEW):
- Processes NEWEST rows first by keysetting on created_at DESC.
- Cursor now stores created_at (ISO string). If an old UUID cursor is found,
  it is ignored automatically (so you won't see "invalid input syntax for timestamp").
- We still fetch *all* rows (including previously seen) but skip ones present in
  checked_ids.txt so reruns don't churn.

Feasible if ANY holds:
  1) summary['gap_seen'] is True
  2) status not in {3,4,5} (not INFEASIBLE/INF_OR_UNBD/UNBOUNDED)
  3) summary['gap'] is a finite number

If infeasible, set all items' demand[t] = 0 starting at t=0; re-check each t; on first
feasible result, update ONLY the 'data' JSON in Supabase (same row, same id).

Persistence (crash-safe appends + fsync):
  - checked_ids.txt : every row id fully processed (feasible OR mutated)
  - cursor.txt      : last_seen_created_at ISO string; pagination uses lt(created_at, cursor)

Env vars (optional):
  SUPABASE_URL
  SUPABASE_ANON_KEY
  PAGE_SIZE               (default 200)
  DRY_RUN                 ("1"/"true" to enable; default False)
  START_BEFORE_CREATED_AT (ISO timestamptz; start with rows created before this)
  SOLVER_TIME_LIMIT_SEC   (default 60 sec)
  CHECKED_IDS_PATH        (default "checked_ids.txt")
  CURSOR_PATH             (default "cursor.txt")
  FETCH_RETRIES           (default 6)
  UPDATE_RETRIES          (default 3)   # keep light for Supabase
  VERIFY_AFTER_UPDATE     ("0" to skip readback verify; default verify enabled)
"""

from __future__ import annotations
import json
import os
import time
import random
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

from supabase import create_client, Client

# ---- Prefer patched solver (supports stop_on_first_gap & gap_seen); fallback
try:
    from mip.solver_mip_no_cross_feasible import solve_instance as _solve_base

    _PATCHED = True
except Exception:
    from mip.solver_mip_no_cross import solve_instance as _solve_base  # type: ignore

    _PATCHED = False

# ====================== CONFIG ======================
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://btqqbsnjcsgjvgpuutiw.supabase.co")
SUPABASE_ANON = os.getenv(
    "SUPABASE_ANON_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ0cXFic25qY3NnanZncHV1dGl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU1NDMwODMsImV4cCI6MjA3MTExOTA4M30.gissvSrKruPsYJOHOoLqfzQGLrB4oFVckVwhrUpGJXU",
)
TABLE_NAME = "instances"

PAGE_SIZE = int(os.getenv("PAGE_SIZE", "200"))
DRY_RUN = os.getenv("DRY_RUN", "").lower() in {"1", "true", "yes", "y"}

# Newest-first starting boundary for created_at keyset (ISO string).
START_BEFORE_CREATED_AT = os.getenv("START_BEFORE_CREATED_AT")

SOLVER_TIME_LIMIT_SEC = max(0, int(os.getenv("SOLVER_TIME_LIMIT_SEC", "60")))
CHECKED_IDS_PATH = Path(os.getenv("CHECKED_IDS_PATH", "checked_ids.txt"))
CURSOR_PATH = Path(os.getenv("CURSOR_PATH", "cursor.txt"))
FETCH_RETRIES = max(1, int(os.getenv("FETCH_RETRIES", "6")))
UPDATE_RETRIES = max(1, int(os.getenv("UPDATE_RETRIES", "3")))
VERIFY_AFTER_UPDATE = os.getenv("VERIFY_AFTER_UPDATE", "1").lower() not in {
    "0",
    "false",
    "no",
}

TMP_INSTANCE_JSON = Path("tmp_instance.json")
_NOT_FEASIBLE = {3, 4, 5}  # INFEASIBLE, INF_OR_UNBD/UNBOUNDED
# ====================================================


# -------------------- Helpers --------------------
def _normalize_json(x):
    """Deterministic, tolerant normalization (Decimal->float, order-insensitive)."""
    if isinstance(x, dict):
        return {
            str(k): _normalize_json(v)
            for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(x, list):
        return [_normalize_json(v) for v in x]
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, float):
        return round(x, 9)
    return x


def _is_iso_timestamp(s: Optional[str]) -> bool:
    if not s:
        return False
    t = s.strip().replace("Z", "+00:00")
    try:
        datetime.fromisoformat(t)
        return True
    except Exception:
        return False


def _fetch_data_by_id(sb: Client, inst_id: str) -> Optional[Dict[str, Any]]:
    """Read back JSON 'data' for a row."""
    try:
        res = sb.table(TABLE_NAME).select("data").eq("id", inst_id).limit(1).execute()
        rows = res.data or []
        if not rows:
            return None
        stored = rows[0].get("data")
        if isinstance(stored, str):
            try:
                stored = json.loads(stored)
            except Exception:
                return {"__raw_string__": stored}
        return dict(stored or {})
    except Exception as e:
        print(f"[{inst_id}] Verify read failed: {e}", flush=True)
        return None


def load_checked_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        with path.open("r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except Exception:
        return set()


def append_checked_id(path: Path, inst_id: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(inst_id + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_cursor(path: Path) -> Optional[str]:
    """Returns an ISO timestamp string or None (auto-ignores old UUID cursors)."""
    if not path.exists():
        return None
    try:
        txt = path.read_text(encoding="utf-8").strip()
        if not txt:
            return None
        if _is_iso_timestamp(txt):
            return txt
        # Old cursor held an id (UUID). Ignore it safely.
        print(f"[cursor] Ignoring legacy non-timestamp cursor value: {txt}", flush=True)
        return None
    except Exception:
        return None


def save_cursor(path: Path, created_at_iso: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(created_at_iso)
        f.flush()
        os.fsync(f.fileno())


def _exp_backoff_sleep(attempt: int, base: float = 1.5, cap: float = 12.0) -> None:
    delay = min(cap, (base**attempt)) * (1.0 + 0.2 * random.random())
    time.sleep(delay)


# -------------------- Supabase fetch/update --------------------
def fetch_page_with_retry(
    sb: Client, last_seen_created_at: Optional[str], page_size: int
) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch newest-first, paged by created_at DESC.
    If last_seen_created_at is set, fetch rows with created_at < last_seen_created_at.
    """
    for attempt in range(FETCH_RETRIES):
        try:
            q = (
                sb.table(TABLE_NAME)
                .select("id, data, created_at")
                .order("created_at", desc=True)
                .limit(page_size)
            )
            if last_seen_created_at:
                q = q.lt("created_at", last_seen_created_at)
            res = q.execute()
            return res.data or []
        except Exception as e:
            print(f"Fetch error (attempt {attempt+1}/{FETCH_RETRIES}): {e}", flush=True)
            if attempt + 1 >= FETCH_RETRIES:
                return None
            _exp_backoff_sleep(attempt)
    return None


def update_with_verify(
    sb: Client,
    inst_id: str,
    new_data: Dict[str, Any],
    must_have_zeroed_up_to_t: Optional[int],
) -> bool:
    """Write JSONB and verify by a separate SELECT (no .select() on update due to client semantics)."""
    if DRY_RUN:
        return True
    for attempt in range(UPDATE_RETRIES):
        try:
            sb.table(TABLE_NAME).update({"data": new_data}).eq("id", inst_id).execute()
            if not VERIFY_AFTER_UPDATE:
                return True
            stored = _fetch_data_by_id(sb, inst_id)
            if stored is None:
                print(f"[{inst_id}] Verify read returned no row.", flush=True)
            else:
                if must_have_zeroed_up_to_t is None:
                    if _normalize_json(stored) == _normalize_json(new_data):
                        return True
                else:
                    if _verify_zeroed_prefix(stored, must_have_zeroed_up_to_t):
                        return True
                    print(
                        f"[{inst_id}] Post-update verify mismatch up to t={must_have_zeroed_up_to_t}.",
                        flush=True,
                    )
        except Exception as e:
            print(
                f"[{inst_id}] Update failed (attempt {attempt+1}/{UPDATE_RETRIES}): {e}",
                flush=True,
            )
        _exp_backoff_sleep(attempt)
    print(f"[{inst_id}] Update verification failed after retries.", flush=True)
    return False


def _verify_zeroed_prefix(data: Dict[str, Any], t_last: int) -> bool:
    """Confirm data['items'][*]['demand'][0..t_last] == 0 after write."""
    items = (data or {}).get("items") or {}
    for _, it in items.items():
        dem = it.get("demand", [])
        for t in range(min(len(dem), t_last + 1)):
            if dem[t] != 0:
                return False
    return True


# -------------------- Feasibility / solver --------------------
def feasible_from_summary(summary: Dict[str, Any]) -> bool:
    if summary.get("gap_seen"):
        return True
    st = summary.get("status")
    try:
        if st is not None and int(st) not in _NOT_FEASIBLE:
            return True
    except Exception:
        pass
    try:
        g = summary.get("gap")
        if g is not None and float(g) >= 0.0 and float(g) < 1e12:
            return True
    except Exception:
        pass
    return False


def run_solver(data: Dict[str, Any]) -> Dict[str, Any]:
    TMP_INSTANCE_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        summary, _ = _solve_base(
            str(TMP_INSTANCE_JSON),
            time_limit=SOLVER_TIME_LIMIT_SEC,
            mip_gap=0.0,
            stop_on_first_gap=True,
        )
    except TypeError:
        summary, _ = _solve_base(
            str(TMP_INSTANCE_JSON), time_limit=SOLVER_TIME_LIMIT_SEC, mip_gap=0.0
        )
    except Exception as e:
        print(f"Solver error: {e}", flush=True)
        return {"status": None, "gap": None, "gap_seen": False, "solver_error": str(e)}
    return dict(summary or {})


# -------------------- JSON mutation helpers --------------------
def coerce_data_object(data: Any) -> Dict[str, Any]:
    if isinstance(data, str):
        return json.loads(data)
    return dict(data or {})


def infer_horizon(data: Dict[str, Any]) -> int:
    try:
        T = int(data.get("period", 0))
        if T > 0:
            return T
    except Exception:
        pass
    items = (data or {}).get("items") or {}
    T = 0
    for _, it in items.items():
        dem = it.get("demand", [])
        if isinstance(dem, list):
            T = max(T, len(dem))
    return T


def zero_period_demands_inplace(data: Dict[str, Any], t: int) -> bool:
    items = (data or {}).get("items") or {}
    changed = False
    for _, it in items.items():
        dem = it.get("demand", [])
        if isinstance(dem, list) and t < len(dem) and dem[t] != 0:
            dem[t] = 0
            changed = True
    return changed


# -------------------- Row processing --------------------
def process_row(row: Dict[str, Any], sb: Client) -> Tuple[bool, bool, str, str]:
    """
    Returns (was_mutated, was_checked, id_str, created_at_iso)
    """
    inst_id = str(row.get("id"))
    created_at = str(row.get("created_at") or "")
    data = coerce_data_object(row.get("data"))

    if not data:
        print(f"[{inst_id}] No 'data' JSON → skip.", flush=True)
        return False, True, inst_id, created_at

    summary = run_solver(data)
    if "solver_error" in summary and summary["solver_error"]:
        print(f"[{inst_id}] Solver failed; leaving for a future retry.", flush=True)
        return False, False, inst_id, created_at

    if feasible_from_summary(summary):
        print(
            f"[{inst_id}] Feasible (status={summary.get('status')}, gap_seen={summary.get('gap_seen')}, gap={summary.get('gap')}). Skip.",
            flush=True,
        )
        return False, True, inst_id, created_at

    print(
        f"[{inst_id}] Infeasible (status={summary.get('status')}, gap_seen={summary.get('gap_seen')}, gap={summary.get('gap')}). Zeroing demands period-by-period…",
        flush=True,
    )

    T = infer_horizon(data)
    if T <= 0:
        print(f"[{inst_id}] Could not infer horizon; skip.", flush=True)
        return False, True, inst_id, created_at

    last_zeroed_t_for_verify: Optional[int] = None

    for t in range(T):
        changed = zero_period_demands_inplace(data, t)
        if not changed:
            print(f"[{inst_id}] t={t}: nothing to zero. Continue.", flush=True)
            continue

        last_zeroed_t_for_verify = t
        print(f"[{inst_id}] t={t}: zeroed all items → re-check…", flush=True)
        summary2 = run_solver(data)

        if "solver_error" in summary2 and summary2["solver_error"]:
            print(
                f"[{inst_id}] Solver failed after zeroing t={t}. Will retry later.",
                flush=True,
            )
            return False, False, inst_id, created_at

        if feasible_from_summary(summary2):
            print(
                f"[{inst_id}] Feasible after zeroing through t={t} (status={summary2.get('status')}, gap_seen={summary2.get('gap_seen')}, gap={summary2.get('gap')}). Writing JSON…",
                flush=True,
            )
            ok = update_with_verify(sb, inst_id, data, last_zeroed_t_for_verify)
            if not ok:
                print(
                    f"[{inst_id}] Update ultimately failed; will retry later.",
                    flush=True,
                )
                return False, False, inst_id, created_at
            return True, True, inst_id, created_at

        print(
            f"[{inst_id}] Still infeasible after t={t} (status={summary2.get('status')}, gap_seen={summary2.get('gap_seen')}, gap={summary2.get('gap')}). Continuing…",
            flush=True,
        )

    print(
        f"[{inst_id}] Could not achieve feasibility after zeroing all periods → left unchanged.",
        flush=True,
    )
    return False, True, inst_id, created_at


def count_total(sb: Client) -> int | None:
    try:
        res = sb.table(TABLE_NAME).select("id", count="exact").limit(0).execute()
        total = getattr(res, "count", None)
        return int(total) if total is not None else None
    except Exception:
        return None


# -------------------- Main --------------------
def main() -> None:
    sb: Client = create_client(SUPABASE_URL, SUPABASE_ANON)

    checked_ids = load_checked_ids(CHECKED_IDS_PATH)

    # Determine starting boundary (created_at ISO).
    start_cursor = START_BEFORE_CREATED_AT or load_cursor(CURSOR_PATH)
    if start_cursor and not _is_iso_timestamp(start_cursor):
        # extra guard (shouldn't happen due to load_cursor)
        print(f"[cursor] Resetting invalid cursor value: {start_cursor}", flush=True)
        start_cursor = None

    total_to_check = count_total(sb)
    if total_to_check is not None:
        print(f"Planned to check: {total_to_check} instances.", flush=True)
    else:
        print("Planned to check: (unknown – count not available).", flush=True)

    mutated_ids_this_run: List[str] = []
    processed = 0
    mutated = 0
    last_seen_created_at = start_cursor  # DESC pagination boundary

    while True:
        rows = fetch_page_with_retry(sb, last_seen_created_at, PAGE_SIZE)
        if rows is None:
            print("Giving up on fetch after retries.", flush=True)
            break
        if not rows:
            break

        for row in rows:
            row_id = str(row.get("id"))
            created_at_iso = str(row.get("created_at") or "")

            # We *fetch* everything, but skip already-checked ids quickly.
            if row_id in checked_ids:
                if created_at_iso:
                    last_seen_created_at = created_at_iso
                    save_cursor(CURSOR_PATH, last_seen_created_at)
                continue

            changed, was_checked, id_str, row_created_at = process_row(row, sb)
            processed += 1

            # Advance keyset cursor toward older rows (DESC).
            if row_created_at:
                last_seen_created_at = row_created_at
                save_cursor(CURSOR_PATH, last_seen_created_at)

            if was_checked:
                checked_ids.add(id_str)
                append_checked_id(CHECKED_IDS_PATH, id_str)

            if changed:
                mutated += 1
                mutated_ids_this_run.append(id_str)

            if total_to_check is not None:
                remaining = max(total_to_check - len(checked_ids), 0)
                print(
                    f"Progress: checked {len(checked_ids)}/{total_to_check} | remaining {remaining} | mutated {mutated}",
                    flush=True,
                )
            else:
                print(
                    f"Progress: processed {processed} | mutated {mutated}", flush=True
                )

        if len(rows) < PAGE_SIZE:
            break

    print("\n================== SUMMARY ==================", flush=True)
    if total_to_check is not None:
        print(
            f"Checked this/previous runs: {len(checked_ids)}/{total_to_check}",
            flush=True,
        )
    else:
        print(f"Checked this/previous runs: {len(checked_ids)}", flush=True)
    print(f"Mutated (JSON only) this run: {mutated}", flush=True)
    if mutated_ids_this_run:
        print("Mutated instance IDs (this run):", flush=True)
        for mid in mutated_ids_this_run:
            print(f"  - {mid}", flush=True)
    else:
        print("No instances mutated in this run.", flush=True)
    print("Patched-solver in use:", _PATCHED, flush=True)
    print("DRY_RUN:", DRY_RUN, flush=True)
    print("SOLVER_TIME_LIMIT_SEC:", SOLVER_TIME_LIMIT_SEC, flush=True)
    print(f"Checked IDs file: {CHECKED_IDS_PATH.resolve()}", flush=True)
    print(
        f"Cursor file:      {CURSOR_PATH.resolve()}  (created_at DESC, ISO)", flush=True
    )
    print("=============================================\n", flush=True)


if __name__ == "__main__":
    print("=============================================", flush=True)
    print("Fix Instances – Feasibility Enforcer", flush=True)
    print("=============================================", flush=True)
    print("Patched-solver available:", _PATCHED, flush=True)
    print("DRY_RUN:", DRY_RUN, flush=True)
    print("PAGE_SIZE:", PAGE_SIZE, flush=True)
    print("SOLVER_TIME_LIMIT_SEC:", SOLVER_TIME_LIMIT_SEC, flush=True)
    print("=============================================\n", flush=True)
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        raise
