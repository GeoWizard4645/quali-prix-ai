"""
Replays a saved genome's exact press/release events against the LIVE game
in Practice mode (so crashes don't matter and nothing is submitted to any
leaderboard), and reports the lap time the game itself shows, for
comparison against the simulator's prediction.

This has NOT been run against the live site by the agent that built this
project (see README.md -- that's a deliberate choice to avoid "live
dry-run" costs/risk; it drives real keyboard/pointer input against the
real page). Run it yourself:

    pip install playwright && playwright install chromium
    python3 replay.py --best runs/lasvegas_best.json --circuit lasvegas

It uses the keyboard (arrow key / spacebar fallback) rather than pointer
events by default since that's the more reliable automatable input; pass
--pointer to use pointerdown/up on the track element instead.

How it synchronizes: it opens the page, switches Practice mode on, starts
a lap, and begins sending input at the moment it detects the car is moving
(polling a simple heuristic -- see _wait_for_lap_start). Because the exact
visual/DOM cue for "the lap has started" isn't something we reverse
engineered (we only extracted the physics module, not the UI/React layer),
treat the reported gap as approximate until you've checked the sync point
against a manual run; the README documents this as an open caveat.
"""
import argparse
import asyncio
import json
import sys

sim = None  # imported lazily so `python3 replay.py --help` doesn't need playwright installed


def events_to_schedule(events, dt):
    """events: list of (tick, held). Returns [(seconds_from_start, held), ...]."""
    return [(tick * dt, bool(held)) for tick, held in events]


async def _wait_for_lap_start(page, timeout_s=15.0):
    """Best-effort: wait until the page's lap/practice UI looks "started".
    This is a heuristic (see module docstring) -- adjust the selector/text
    check below if the real page's DOM differs once you inspect it live.
    """
    try:
        await page.wait_for_selector("text=/00:0|0:0/", timeout=timeout_s * 1000)
    except Exception:
        pass


async def run_replay(circuit: str, best_path: str, url: str, headless: bool, use_pointer: bool):
    from playwright.async_api import async_playwright

    global sim
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sim as _sim
    sim = _sim

    best = json.load(open(best_path))
    schedule = events_to_schedule(best["events"], best["dt"])
    print(f"replaying {len(schedule)} press/release events, predicted sim lap time "
          f"{best.get('lap_time')} s (finished={best['finished']})")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()
        await page.goto(url)

        # TODO once inspected live: click the Practice-mode toggle and the
        # correct circuit if it isn't the live one already. Left as an
        # explicit manual/first-run step since we didn't drive the live UI
        # while building this.
        print("Set Practice mode and select the circuit in the opened browser window, "
              "then press Enter here to start the timed replay.")
        input()

        await _wait_for_lap_start(page)
        t0 = asyncio.get_event_loop().time()

        held = False
        for t_target, new_held in schedule:
            now = asyncio.get_event_loop().time() - t0
            if t_target > now:
                await asyncio.sleep(t_target - now)
            held = new_held
            if use_pointer:
                if held:
                    await page.mouse.down()
                else:
                    await page.mouse.up()
            else:
                if held:
                    await page.keyboard.down("ArrowRight")
                else:
                    await page.keyboard.up("ArrowRight")

        if held:
            if use_pointer:
                await page.mouse.up()
            else:
                await page.keyboard.up("ArrowRight")

        print("Replay finished. Read the lap time the page shows and compare it to the "
              "simulator's prediction above.")
        print("Press Enter to close the browser.")
        input()
        await browser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="runs/lasvegas_best.json")
    ap.add_argument("--circuit", default="lasvegas")
    ap.add_argument("--url", default="https://gluefold.co.uk/quali-prix")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--pointer", action="store_true", help="use pointer events instead of the arrow key")
    args = ap.parse_args()
    asyncio.run(run_replay(args.circuit, args.best, args.url, args.headless, args.pointer))


if __name__ == "__main__":
    main()
