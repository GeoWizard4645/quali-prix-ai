"""
Shared logic for driving the real, live game with Playwright -- used by
both replay.py (one-shot, for fidelity checking) and live_driver.py
(continuous, for the dashboard's "watch it play live" view).

Found by a short, limited inspection of the live page (a handful of page
loads/screenshots, not full lap attempts -- see extract/NOTES.md): the
plain "DRIVE" button enters Practice mode by default (confirmed via its
screen-reader text, which this module checks every time before sending any
input, and refuses to proceed if it doesn't say so -- this project never
submits a real time). The control is "hold anywhere to go / Up arrow works
too / Space restarts".
"""
import time

DRIVE_BUTTON_TEXT = "text=DRIVE"
PRACTICE_SR_SELECTOR = ".qp-practice-btn + .sr-only, .sr-only"
HOLD_KEY = "ArrowUp"
RESTART_KEY = "Space"


def events_to_schedule(events, dt):
    """events: [(tick, held), ...] as recorded by sim.simulate(record=True).

    simulate() always starts held=True at tick 0 but only RECORDS a change
    when held flips, so tick 0's implicit "hold" is never itself in the
    list. Callers must press at t=0 regardless -- this prepends that.
    """
    schedule = [(0.0, True)]
    for tick, held in events:
        schedule.append((tick * dt, bool(held)))
    return schedule


def start_practice_lap(page, timeout_s=20.0):
    """Clicks DRIVE, waits for the page's own "hold anywhere to go" prompt
    (rather than a guessed fixed delay -- this is the car's actual "ready
    and waiting for your first input" signal, so timing the replay
    schedule from the moment this returns is as tight a sync point as we
    can get without the game's own source for its UI layer), and verifies
    (via the page's own screen-reader text, not a guess) that Practice
    mode is on. Raises if it can't confirm that -- we never want to risk
    sending a real timed lap.
    """
    page.click(DRIVE_BUTTON_TEXT, timeout=timeout_s * 1000)
    try:
        page.wait_for_selector("text=/Hold anywhere to go/i", timeout=timeout_s * 1000)
    except Exception:
        time.sleep(0.8)  # fallback if the page's wording changes
    practice_text = page.evaluate("""
        () => {
            const el = document.querySelector('.qp-practice-btn');
            const sr = document.querySelector('.sr-only');
            return (el ? el.getAttribute('aria-pressed') : null) + '|' +
                   (sr ? sr.innerText : '');
        }
    """)
    if "Practice on" not in practice_text and "true" not in practice_text:
        raise RuntimeError(
            f"could not confirm Practice mode is on (page said: {practice_text!r}) "
            "-- refusing to send input, to avoid submitting a real timed lap"
        )


def restart_lap(page, timeout_s=10.0):
    """Space restarts the CURRENT race session (no need to click DRIVE
    again -- that button only exists on the pre-race screen, and clicking
    it again mid-session fails)."""
    page.keyboard.press(RESTART_KEY)
    try:
        page.wait_for_selector("text=/Hold anywhere to go/i", timeout=timeout_s * 1000)
    except Exception:
        time.sleep(0.8)


def replay_schedule_sync(page, schedule, use_pointer=False, on_tick=None, tick_s=0.15,
                          stop_check=None):
    """Sends the (seconds_from_start, held) schedule via real input events,
    timed against wall clock from the moment this is called.

    `on_tick(elapsed_s)` is called roughly every `tick_s` seconds throughout
    (not just at schedule points) so a caller can capture a steady stream of
    screenshots while a long hold/brake segment is in progress.
    `stop_check()` is polled the same way; returning True aborts early
    (releasing the key first) -- used to cut a live attempt short on pause.
    """
    t0 = time.monotonic()
    held = False
    for t_target, new_held in schedule:
        while True:
            now = time.monotonic() - t0
            remaining = t_target - now
            if remaining <= 0:
                break
            if on_tick:
                on_tick(now)
            if stop_check and stop_check():
                if held:
                    (page.mouse.up() if use_pointer else page.keyboard.up(HOLD_KEY))
                return
            time.sleep(min(tick_s, remaining))
        if new_held == held:
            continue
        held = new_held
        if use_pointer:
            page.mouse.down() if held else page.mouse.up()
        else:
            page.keyboard.down(HOLD_KEY) if held else page.keyboard.up(HOLD_KEY)
    if held:
        if use_pointer:
            page.mouse.up()
        else:
            page.keyboard.up(HOLD_KEY)


def check_interrupt(page) -> str:
    """Detects the live page's own "Spun out / Tap to restart your lap" (or
    similarly-worded crash) overlay, found during live inspection (see
    extract/NOTES.md) -- our scripted input can't tell a spin happened any
    other way, and continuing to send a schedule timed for a car that the
    page has already stopped and is waiting to restart just cascades into
    more spins. Returns the overlay's text, or "" if nothing is showing.
    """
    return page.evaluate("""
        () => {
            const els = document.querySelectorAll('div, p, span');
            for (const e of els) {
                const t = e.innerText;
                if (t && /restart your lap/i.test(t) && e.children.length <= 2) return t.trim();
            }
            return '';
        }
    """) or ""


def make_interrupt_checker(page, extra_stop=None, poll_every_s: float = 1.0):
    """Builds a `stop_check()` for replay_schedule_sync() that bails out
    early (and reports why, via .reason) the moment the live page shows
    its own spin/crash-restart overlay, or `extra_stop()` says to -- so we
    never keep sending a schedule timed for a car the page has already
    stopped and is waiting on a tap to restart.
    """
    state = {"last_check": 0.0, "reason": None}

    def check():
        if extra_stop and extra_stop():
            state["reason"] = "paused"
            return True
        now = time.monotonic()
        if now - state["last_check"] >= poll_every_s:
            state["last_check"] = now
            text = check_interrupt(page)
            if text:
                state["reason"] = text
                return True
        return False

    check.reason = lambda: state["reason"]
    return check


def read_hud(page) -> dict:
    """Best-effort read of the in-race HUD text (lap/sector times, attempts)."""
    return page.evaluate("""
        () => {
            const q = (sel) => { const e = document.querySelector(sel); return e ? e.innerText : null; };
            const attempts = document.querySelector('.qp-attempts');
            return {
                attempts: attempts ? attempts.innerText : null,
            };
        }
    """)
