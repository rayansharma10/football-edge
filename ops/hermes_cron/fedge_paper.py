"""Shared runner for football-edge paper-desk cron jobs (zero-LLM).

Records each run in <repo>/data/last_run.json (read by the dashboard) and provides a
catch-up mode: the PC is often off at the scheduled time, so a frequent cheap job calls
main("catchup"), which runs picks/settle once if their last success is too old.
"""
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = pathlib.Path("C:/Users/Rayan/dev/football-edge")
UV = "C:/Users/Rayan/AppData/Local/hermes/bin/uv.exe"
STATE = REPO / "data" / "last_run.json"
# job -> repo script. Betting jobs (picks/settle/weekly) are PAUSED since the accuracy-tracker
# pivot but still runnable by hand; the live jobs are predict (schedule refresh + predictions
# logged to prediction_log) and results (finished scores).
SCRIPTS = {
    "picks": "paper_picks.py", "settle": "paper_settle.py",
    "predict": "predict_upcoming.py", "results": "update_results.py",
}
# max age (hours) of the last successful run before catch-up fires (PC is often off before ~11am)
MAX_AGE_H = {"predict": 12, "results": 26}


def _load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _record(job: str, ok: bool) -> None:
    st = _load_state()
    e = st.setdefault(job, {})
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    e["last_attempt_utc"] = now
    if ok:
        e["last_success_utc"] = now
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")


def catchup() -> int:
    rc = 0
    st = _load_state()
    for job, hrs in MAX_AGE_H.items():
        ts = st.get(job, {}).get("last_success_utc")
        age = None
        if ts:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() / 3600
        if age is None or age > hrs:
            when = "never" if age is None else f"{age:.0f}h ago"
            print(f"(catch-up) football-edge {job}: last success {when}, running now")
            rc |= main(job)
    return rc


def _run(job: str):
    script = REPO / "scripts" / SCRIPTS[job]
    if not script.exists():
        return None
    env = {k: v for k, v in os.environ.items()
           if k.upper() not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP")}
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [UV, "run", "python", str(script)],
        env=env, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=1500,
    )


def main(job: str) -> int:
    if job == "catchup":
        return catchup()
    p = _run(job)
    if p is None:
        return 0  # desk script missing: stay quiet
    if p.returncode != 0:
        time.sleep(20)  # one retry: a venv mid-sync gives a transient ImportError
        p = _run(job)
    if p.stdout.strip():
        print(p.stdout.strip())
    _record(job, p.returncode == 0)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout).strip().splitlines()[-5:]
        print(f"⚠️ football-edge {job} failed (exit {p.returncode}):\n" + "\n".join(tail))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "picks"))
