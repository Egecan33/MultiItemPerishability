# db/runner_supabase.py
"""
Minimal worker:
- poll Supabase 'runs' table for next queued run
- download instance JSON from 'instances.raw'
- call solver
- upload summary + x/y rows
"""

import os, json, time, csv, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get(
    "SUPABASE_ANON_KEY"
)
assert (
    SUPABASE_URL and SUPABASE_KEY
), "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or ANON_KEY)"

sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_next_run():
    # returns (run, instance)
    res = (
        sb.table("runs")
        .select("*")
        .eq("status", "queued")
        .order("created_at", desc=False)
        .limit(1)
        .execute()
    )
    if not res.data:
        return None, None
    run = res.data[0]
    inst_id = run["instance_id"]
    inst = sb.table("instances").select("*").eq("id", inst_id).single().execute().data
    return run, inst


def mark_status(run_id, status, patch=None):
    payload = {"status": status}
    if patch:
        payload.update(patch)
    sb.table("runs").update(payload).eq("id", run_id).execute()


def upsert_xy(run_id, out_dir: Path):
    # X
    xp = out_dir / "x_nonzero.csv"
    if xp.exists():
        rows = []
        with xp.open() as f:
            rd = csv.DictReader(f)
            for r in rd:
                rows.append(
                    dict(
                        run_id=run_id,
                        item_id=int(r["i"]),
                        t=int(r["t"]),
                        u=int(r["u"]),
                        x=float(r["X"]),
                    )
                )
        if rows:
            sb.table("run_x").upsert(rows, on_conflict="run_id,item_id,t,u").execute()
    # Y
    yp = out_dir / "y_setups.csv"
    if yp.exists():
        rows = []
        with yp.open() as f:
            rd = csv.DictReader(f)
            for r in rd:
                rows.append(
                    dict(
                        run_id=run_id, item_id=int(r["i"]), t=int(r["t"]), y=int(r["Y"])
                    )
                )
        if rows:
            sb.table("run_y").upsert(rows, on_conflict="run_id,item_id,t").execute()


def run_once():
    run, inst = fetch_next_run()
    if not run:
        print("[runner] queue empty")
        return False
    run_id = run["id"]
    print(f"[runner] picked run {run_id}")
    mark_status(
        run_id, "running", {"started_at": datetime.now(timezone.utc).isoformat()}
    )

    # write instance locally
    inst_json = inst["raw"]
    inst_path = Path(f"tmp_instance_{run_id}.json")
    inst_path.write_text(json.dumps(inst_json, indent=2), encoding="utf-8")

    # call solver
    outdir = Path("mip_results")
    time_limit = run.get("time_limit_sec", 3600)
    mipgap = run.get("mip_gap_target", 0.01)
    xint = "--x-as-integer" if bool(run.get("x_as_integer", False)) else ""
    cmd = f"{sys.executable} -m mip.solver_mip_lefo --instance {inst_path} --time-limit {time_limit} --mipgap {mipgap} --outdir {outdir} {xint}"
    print("[runner] exec:", cmd)
    proc = subprocess.run(cmd, shell=True)
    # find newest result dir
    latest = max(outdir.glob(f"{inst_path.stem}__*"), key=lambda p: p.stat().st_mtime)

    # parse summary.json
    summary = json.loads((latest / "summary.json").read_text())
    patch = {
        "status_name": summary.get("status_name"),
        "objective": summary.get("objective"),
        "best_bound": summary.get("best_bound"),
        "mip_gap": summary.get("mip_gap"),
        "runtime_sec": summary.get("runtime_sec"),
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }
    mark_status(run_id, "done", patch)
    upsert_xy(run_id, latest)

    # cleanup instance file
    try:
        inst_path.unlink()
    except Exception:
        pass

    print(f"[runner] completed run {run_id} → {latest}")
    return True


if __name__ == "__main__":
    while True:
        did = run_once()
        if not did:
            time.sleep(5)
