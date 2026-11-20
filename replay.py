"""
Replays a saved genome's exact press/release events against the LIVE game
and reports what happened for comparison against the simulator's
prediction.

Runs in **real/ranked mode by default** -- a finished lap counts and can
appear in the live site's public times listing. Pass --practice to use
Practice mode instead. See live_common.py's docstring for why real mode
is the default (short version: the Practice toggle defaults OFF, not on
as an earlier version of this script incorrectly assumed).

How this was wired up: a short, limited inspection of the live page (a
handful of page loads/screenshots, not full lap attempts -- see
extract/NOTES.md) found that the control is "hold anywhere to go / Up
arrow works too / Space restarts", and that the Practice toggle's real
live state is its `aria-pressed` attribute, which this script checks
every time and refuses to send input if it can't confirm.

Usage:
    pip install playwright && playwright install chromium
    python3 replay.py --best runs/lasvegas_best.json
    python3 replay.py --best runs/lasvegas_best.json --practice
"""
import argparse
import json
import time

from live_common import (events_to_schedule, make_interrupt_checker, read_hud,
                          replay_schedule_sync, restart_lap, start_lap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="runs/lasvegas_best.json")
    ap.add_argument("--url", default="https://gluefold.co.uk/quali-prix")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--pointer", action="store_true", help="use pointer events instead of the Up arrow key")
    ap.add_argument("--practice", action="store_true", help="use Practice mode instead of real/ranked mode")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    best = json.load(open(args.best))
    schedule = events_to_schedule(best["events"], best["dt"])
    print(f"replaying {len(schedule)} press/release events; simulator predicted "
          f"{'lap time ' + format(best['lap_time'], '.3f') + 's' if best['finished'] else 'a crash'} "
          f"({best.get('dist_reached', '?')} m reached if it crashed)")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        page = browser.new_page(viewport={"width": 1180, "height": 760})
        page.goto(args.url, wait_until="load", timeout=45000)
        time.sleep(1.0)

        start_lap(page, practice=args.practice)
        print(f"Confirmed {'Practice' if args.practice else 'real/ranked'} mode. Replaying now...")

        interrupt_check = make_interrupt_checker(page)
        replay_schedule_sync(page, schedule, use_pointer=args.pointer, stop_check=interrupt_check)
        reason = interrupt_check.reason()
        if reason:
            print(f"The page interrupted the attempt before the schedule finished: "
                  f"{reason.splitlines()[0]!r} (likely a spin -- live timing drifted from the "
                  "simulator's prediction here; see README's Fidelity section)")
            restart_lap(page)

        time.sleep(0.8)
        hud = read_hud(page)
        print(f"Live page attempts counter: {hud.get('attempts')}")
        print("Read the lap/crash the page itself shows and compare it to the simulator's "
              "prediction above. Leave the browser open to look, or pass --headless to skip it.")
        if not args.headless:
            print("Press Enter to close the browser.")
            input()
        browser.close()


if __name__ == "__main__":
    main()
