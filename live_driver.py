"""
Drives the REAL, live game (gluefold.co.uk/quali-prix) with the current
best genome, in Practice mode only, and streams screenshots to
runs/<circuit>_live_frame.jpg + runs/<circuit>_live_status.json for
dashboard.py's "live game" panel.

Safety, by design, not just by convention:
  - start_practice_lap() (live_common.py) refuses to send any input unless
    it can read the page's own text confirming Practice mode is on, every
    single attempt.
  - Idle (paused) by default -- it only drives the page while the
    dashboard's Play button for this panel is on (see control.py), so it
    never hits the real site unless you've explicitly asked to watch.
  - Self-pauses after MAX_ATTEMPTS_PER_SESSION attempts or
    MAX_MINUTES_PER_SESSION minutes of continuous driving, so leaving it
    running overnight by accident doesn't turn into unattended, indefinite
    automated traffic against someone else's small site. Press Play again
    to do another session.

Usage:
    python3 live_driver.py --circuit lasvegas
    (then toggle it on from the dashboard, or see --autostart)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim
from control import control_path, read_control, wait_while_paused, write_control
from live_common import (events_to_schedule, make_interrupt_checker, read_hud,
                          replay_schedule_sync, restart_lap, start_practice_lap)

MAX_ATTEMPTS_PER_SESSION = 40
MAX_MINUTES_PER_SESSION = 30
URL = "https://gluefold.co.uk/quali-prix"


def save_status(path, **kwargs):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(kwargs, f)
    os.replace(tmp, path)


def load_best(best_path):
    if not os.path.exists(best_path):
        return None
    with open(best_path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="lasvegas")
    ap.add_argument("--data-dir", default="extract/raw_data")
    ap.add_argument("--best", default=None, help="default: runs/<circuit>_best.json")
    ap.add_argument("--frame-out", default=None, help="default: runs/<circuit>_live_frame.jpg")
    ap.add_argument("--status-out", default=None, help="default: runs/<circuit>_live_status.json")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--headed", dest="headless", action="store_false")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    best_path = args.best or f"runs/{args.circuit}_best.json"
    frame_path = args.frame_out or f"runs/{args.circuit}_live_frame.jpg"
    status_path = args.status_out or f"runs/{args.circuit}_live_status.json"
    ctrl_path = control_path("live", args.circuit)
    # Start paused: never drive the real site until a human presses Play.
    if not os.path.exists(ctrl_path):
        write_control(ctrl_path, paused=True, stop=False)

    attempts = 0
    session_start = None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        page = browser.new_page(viewport={"width": 1180, "height": 760})
        page_loaded = False

        def snapshot(note=""):
            try:
                page.screenshot(path=frame_path + ".tmp", type="jpeg", quality=70)
                os.replace(frame_path + ".tmp", frame_path)
            except Exception as e:
                save_status(status_path, circuit=args.circuit, attempts=attempts,
                            note=f"screenshot failed: {e}", updated_at=time.time())
            save_status(status_path, circuit=args.circuit, attempts=attempts,
                        note=note, updated_at=time.time())

        def on_pause_tick(_c):
            nonlocal page_loaded
            save_status(status_path, circuit=args.circuit, attempts=attempts,
                        note="paused", updated_at=time.time(), paused=True)

        while True:
            if wait_while_paused(ctrl_path, on_tick=on_pause_tick):
                break

            if not page_loaded:
                page.goto(args.url, wait_until="load", timeout=45000)
                time.sleep(1.0)
                try:
                    # DRIVE only exists on the pre-race screen -- click it
                    # exactly once per page load. Every later attempt in
                    # this same session restarts via Space (restart_lap),
                    # not by clicking DRIVE again (see live_common.py).
                    start_practice_lap(page)
                except Exception as e:
                    save_status(status_path, circuit=args.circuit, attempts=attempts,
                                note=f"could not confirm Practice mode, stopping for safety: {e}",
                                updated_at=time.time())
                    write_control(ctrl_path, paused=True)
                    continue
                page_loaded = True
                session_start = time.time()
                attempts = 0

            if attempts >= MAX_ATTEMPTS_PER_SESSION or (time.time() - session_start) > MAX_MINUTES_PER_SESSION * 60:
                write_control(ctrl_path, paused=True)
                save_status(status_path, circuit=args.circuit, attempts=attempts,
                            note=f"auto-paused after {attempts} attempts / "
                                 f"{(time.time()-session_start)/60:.0f} min -- press Play to continue",
                            updated_at=time.time(), paused=True)
                page_loaded = False  # reload fresh next time it's resumed
                continue

            best = load_best(best_path)
            if best is None:
                save_status(status_path, circuit=args.circuit, attempts=attempts,
                            note="waiting for optimize.py to produce a first result...",
                            updated_at=time.time())
                time.sleep(3.0)
                continue

            schedule = events_to_schedule(best["events"], best["dt"])
            attempts += 1
            snapshot(f"attempt {attempts}: replaying {best.get('lap_time') or 'a crash'} genome")

            def extra_stop():
                c = read_control(ctrl_path)
                return bool(c.get("paused") or c.get("stop"))

            interrupt_check = make_interrupt_checker(page, extra_stop=extra_stop)
            replay_schedule_sync(page, schedule, on_tick=lambda _t: snapshot(
                f"attempt {attempts}/{MAX_ATTEMPTS_PER_SESSION}"), tick_s=0.2, stop_check=interrupt_check)

            reason = interrupt_check.reason()
            if reason == "paused":
                continue
            if reason:
                # The page itself stopped the car (spin/crash) before our
                # schedule finished -- restarting now avoids cascading a
                # now-desynced schedule into a repeat spin.
                snapshot(f"attempt {attempts}: {reason.splitlines()[0]} -- restarting")
                restart_lap(page)
                time.sleep(0.5)
                continue

            time.sleep(0.6)
            hud = read_hud(page)
            snapshot(f"attempt {attempts} finished -- {hud.get('attempts')}")
            restart_lap(page)
            time.sleep(0.5)

        browser.close()


if __name__ == "__main__":
    main()
