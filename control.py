"""
Tiny cooperative pause/resume control, shared by optimize.py,
policy/train.py, live_driver.py, and dashboard.py. A "control file" is just
a small JSON blob; the long-running scripts poll it between units of work
(a generation, a lap) rather than being signalled, so a pause always lands
cleanly instead of killing mid-computation.
"""
import json
import os
import time

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def control_path(kind: str, circuit: str) -> str:
    return os.path.join(RUNS_DIR, f"{kind}_{circuit}.control.json")


def read_control(path: str) -> dict:
    if not os.path.exists(path):
        return {"paused": False, "stop": False}
    try:
        with open(path) as f:
            data = json.load(f)
            data.setdefault("paused", False)
            data.setdefault("stop", False)
            return data
    except (OSError, json.JSONDecodeError):
        return {"paused": False, "stop": False}


def write_control(path: str, **kwargs) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cur = read_control(path)
    cur.update(kwargs)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cur, f)
    os.replace(tmp, path)


def wait_while_paused(path: str, on_tick=None, poll_s: float = 0.5) -> bool:
    """Blocks while `paused` is set. Returns True if a stop was requested
    (either while running or while paused). `on_tick(control_dict)` is
    called every poll_s while paused, if given (e.g. to keep a live
    snapshot's "paused" flag fresh for the dashboard)."""
    while True:
        c = read_control(path)
        if c.get("stop"):
            return True
        if not c.get("paused"):
            return False
        if on_tick:
            on_tick(c)
        time.sleep(poll_s)
