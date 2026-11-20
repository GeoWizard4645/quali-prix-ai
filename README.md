# Quali Prix AI

A self-improving lap optimizer for [Quali Prix](https://gluefold.co.uk/quali-prix),
a one-button browser racing game. It runs an evolutionary search offline,
against a from-scratch port of the game's own vehicle physics, to find a
sequence of hold/release points that drives a near-optimal lap -- the "AI
learns to drive, lap after lap" idea, but using an offline simulator
instead of the live page so it can run millions of laps overnight instead
of ~600.

**Read this before trusting any lap time in this README**: a time is only
"real" once it's been checked against the actual game. The
[Fidelity](#fidelity-what-is-and-isnt-verified) section below says exactly
what has and hasn't been checked, and how.

## Quick start

```bash
cd quali-prix-ai
source .venv/bin/activate          # one-time setup: see Setup below
python3 optimize.py --circuit lasvegas        # terminal 1: let it learn (runs until you stop it)
python3 policy/train.py --circuit lasvegas    # terminal 2: optional, trains the neural-net panel
python3 dashboard.py                           # terminal 3: watch it
```
Open the URL `dashboard.py` prints. There's no `--hours` budget to set --
each one runs until you pause or stop it. **Pause/Resume** from the
dashboard itself (a button per panel) any time, with no need to touch a
terminal; `Ctrl+C` in a terminal fully stops that engine, and running the
same command again resumes from its last checkpoint. That's the whole
thing -- everything past this point is detail/reference. The dashboard
also has its own "How to run this" button with this same cheat sheet.

On macOS, if you want the laptop to keep running overnight without going
to sleep, prefix a command with `caffeinate -s`.

## What's actually going on

The game's only control is one button: hold to accelerate, release to
brake. Everything else -- steering, racing line -- is autonomous. That
makes a lap fully determined by *when* you press and release, and because
the physics turned out to be a deterministic, offline-runnable JS module
(see [`extract/NOTES.md`](extract/NOTES.md) for how it was found and why
it's a rewrite rather than a copy), the standard move is: port the physics,
then search over press/release sequences instead of training against the
live page.

The genome is a sorted list of **switch distances in metres** -- not
seconds -- which is much better conditioned for an evolutionary search than
raw timing (a given brake *point* matters, not exactly when you reach it).
One subtlety: distances are measured from the start of the simulated
attempt, which begins **100 m before the start/finish line** (the game
places you there already at a scripted "flying lap" entry speed, exactly
like a real qualifying lap) -- not from the line itself. `Track.length` is
`lap_length + runup_m`, and genomes are clamped to `[0, Track.length)`. See
the docstring at the top of `sim.py` for the full rationale.

## Layout

```
sim.py             track loading + physics + simulate() -- the core port
optimize.py        Phase 1: evolutionary sequence optimizer (per circuit)
policy/
  net.py           tiny MLP policy + rollout (feeds sim.step() live)
  train.py         Phase 2: trains the policy with a simple evolution strategy
  eval.py          zero-shot-generalization check (train on one circuit, test on another)
control.py         shared pause/resume file used by optimize.py/train.py/live_driver.py/dashboard.py
plot.py            learning curve + speed/throttle track plots
dashboard.py       live local web dashboard (watch it learn, with friends)
live_common.py     shared real-page driving logic (used by replay.py and live_driver.py)
live_driver.py     continuously plays the LIVE page in Practice mode for the dashboard's live view
replay.py          one-shot Playwright replay against the LIVE page, Practice mode only
extract/
  NOTES.md         how the physics and circuit data were found
  raw_data/        the circuit metadata/binary/terrain files (see NOTES.md)
  verify/          the live-bundle fidelity-check harness (fetches fresh, not vendored)
tests/             determinism/sanity tests + the live fidelity cross-check
runs/              best.json / history.csv / live state per circuit (gitignored)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

(`node` on PATH is only needed for the live fidelity test in `tests/`; a
recent LTS is fine, nothing project-specific is installed globally.)

## Watch it learn: the live dashboard

```bash
python3 dashboard.py
```

This starts a small local web server (stdlib only -- no new dependency)
that reads `runs/*.json`/`*.csv` and serves a live page:

- **The best lap, animated** on the track, coloured by speed, with small
  markers where it's slid/spun/saved itself ("pain" events) along the way.
- **A learning curve**, gold/silver/bronze medal comparisons, and a
  gamified **Training Score** + reward feed ("+420m further", "🏁 LAP
  COMPLETE -- 46.3s", "⚡ -0.09s faster") that ticks up every time the car
  actually improves, plus a **Pain** meter (% of the track still
  unconquered) and a **Reward** meter (progress from Bronze toward Gold)
  -- a game-score framing of the same fitness numbers `optimize.py` is
  already climbing.
- **A live node-link view of the policy network's weights** (once
  `policy/train.py` is running) plus its own mini-track showing its
  current best attempt -- it's a separate, usually-slower learner than the
  sequence search, see Phase 2 below.
- **⏸ Pause / ▶ Resume buttons**, one per engine (sequence optimizer,
  policy, and the live-game view) -- no terminal needed, and nothing is
  lost: the underlying process just idles until you resume it.
- **"🏎️ Watch it play live"** -- see the next section.
- A **"📖 How to run this"** button with the exact commands, any time.

It prints two URLs: one for `localhost`, and one on your LAN
(`http://<your-ip>:8743`) that anyone on the same Wi-Fi can open to watch
along. That's LAN-only by design (no cloud, nothing exposed to the wider
internet) -- ask if you want it reachable from outside your network too.

The first time the car completes a clean lap, the dashboard names it
**VIV** with a little reveal popup, and labels it that way from then on.

`dashboard.py` only reads files and writes tiny pause/resume control files
-- it never touches the learning processes directly, so start/stop/restart
`optimize.py`, `policy/train.py`, and `live_driver.py` independently of it,
in any order.

## Watch it play the REAL game, live

Click **"Watch it play live"** on the dashboard (off by default) to start
`live_driver.py`'s loop: a background Playwright browser loads the actual
`gluefold.co.uk/quali-prix` page, confirms (via the page's own text, every
single time -- it refuses to send input otherwise) that **Practice mode**
is on, and replays the current best genome's exact press/release timing
with the real Up-arrow key, continuously, restarting itself after a lap or
a spin. Screenshots stream into the dashboard a few times a second.

It never submits a time to any leaderboard, and it **self-pauses after
~40 attempts or ~30 live minutes** so leaving it on by accident doesn't
turn into unattended, indefinite traffic against someone else's small
site -- press the button again for another session. You can also run it
standalone: `python3 live_driver.py --circuit lasvegas` (then toggle it
from the dashboard, same as any other engine).

**Be aware of the gap this surfaces**: live timing sync with the real page
is looser than the simulator, which is tick-for-tick verified against the
real physics code (see Fidelity below). In testing, the live car sometimes
spins out earlier than the simulator predicted for the exact same genome.
That's real, and it's specific to the *input-replay* layer (exactly when
"t=0" should be relative to the page's own readiness, which isn't
something we could extract from the physics bundle -- it's React/UI code)
-- not a sign the physics model itself is wrong. `live_common.py` detects
the page's own "Spun out / Tap to restart" prompt and recovers cleanly
rather than cascading into repeated spins, which is as far as this session
took it; tightening the sync further is a reasonable follow-up.

## Running it overnight

```bash
python3 optimize.py --circuit lasvegas
```

- Runs until you pause or stop it (pass `--hours N` if you'd rather it
  stop itself after N hours).
- Checkpoints `runs/lasvegas_best.json` every time it improves, and appends
  a row (generation, elapsed seconds, best fitness) to
  `runs/lasvegas_history.csv`.
- **Resuming**: just run the same command again -- it loads the existing
  checkpoint and keeps going. Pass `--no-resume` to start over.
- Uses all CPU cores by default (`--workers`); population/elite size and
  the mutation step size (`sigma`, auto-shrinks when progress stalls) are
  tunable flags -- see `python3 optimize.py --help`.
- Fitness: a finished lap scores its lap time; a crash scores
  `1000 + (track.length - distance_reached)`, so there's always a gradient
  toward getting further before crashing, and any finished lap beats any
  crash. This is also exactly what the dashboard's reward/pain framing is
  computed from.
- Run `--circuit saopaulo` the same way for the other shipped circuit.

Plot progress any time with:
```bash
python3 plot.py --circuit lasvegas
```

## Phase 2: a generalizing policy

`optimize.py` finds a sequence tuned to one specific circuit. `policy/`
trains a small neural net (speed, upcoming-curvature lookahead, grade,
previous action -> hold/release) with a simple evolution strategy
(mirrored-sampling ES with rank-based fitness shaping -- no extra
dependency beyond numpy; CMA-ES wasn't needed at ~113 parameters), on the
idea that a *policy* might generalize to next week's circuit, since the
game swaps circuits weekly and a from-scratch per-circuit search starts
from nothing each time.

```bash
python3 policy/train.py --circuit lasvegas
python3 policy/eval.py --policy runs/policy_lasvegas.npy --circuit saopaulo
```

### A real bug this surfaced, and the fix

Early on, the policy got stuck crashing in the same early corner for ~30
minutes with almost no improvement. The cause: a freshly-initialized
network has small random weights, so its output (a `tanh` near zero) flips
between hold/release almost randomly tick to tick -- that's noisy
button-mashing, which crashes immediately almost every time regardless of
what the network is "looking at." There's no usable fitness gradient to
climb out of that; every random policy looks equally bad.

The fix (`policy.net.initial_weights`): bias the output layer so the net
starts out equivalent to the known full-throttle baseline (a clean,
consistent ~500 m before a real crash) -- i.e. start from "always hold"
and let evolution learn *where to deviate*, exactly like the sequence
optimizer's own empty-genome starting point. Also added an adaptive
mutation step (widens when stuck for 15 generations, eases back down once
moving again -- the opposite schedule from `optimize.py`'s, since a stuck
*policy* is usually stuck in a bad region of weight space and needs to
search more broadly, not fine-tune).

Result: the same setup that spent 30 minutes barely past the first corner
went from crash to a **clean finished lap (82.4s) in 2 generations** after
restarting with the fix. Still far slower to converge than the sequence
optimizer (expected -- ES over ~113 weights has no targeted "add a brake
zone near the crash point" mutation the way the sequence search does), but
no longer stuck. The dashboard's policy panel shows this directly now: its
own mini-track and a `training/paused` badge, so "is it stuck or just
slower" is visible at a glance instead of a mystery.

Zero-shot transfer to Sao Paulo (train on Las Vegas, test cold) is the
real experiment this phase exists for, and is still open -- run
`policy/train.py` for a good while, then `policy/eval.py --circuit
saopaulo`, and compare its time against `optimize.py --circuit saopaulo`'s
own from-scratch result.

## Fidelity: what is and isn't verified

**What's verified, and how:** `tests/test_fidelity_vs_real_bundle.py`
fetches the game's own physics module fresh (same request any visitor's
browser makes) and runs it standalone under Node, then diffs it tick-by-
tick against `sim.py` on a fixed set of genomes across both shipped
circuits. Run it yourself: `python3 -m pytest tests/ -v`.

Measured results from that test, on this machine, on 2026-11-20:
- **13/13** outcome matches (finished/crashed/timed-out) across random and
  hand-picked genomes on both circuits, including 80+ second timeout runs.
- Position agreement of **~1e-4 m** (sub-millimetre) for ordinary genomes,
  even over long runs -- after reproducing the game's own `sin`/`cos`/
  `atan`/`atan2` polynomial approximations exactly instead of using
  Python's `math` module's (more accurate, but *differently* rounded)
  versions, which turned out to matter (see `extract/NOTES.md`).
- **One known caveat**, found deliberately: a hand-picked genome that
  survives a hairpin by a hair (a recorded "save" -- a near-spin recovery)
  then crashes ~2.5 s later diverges to ~5 ticks / ~7 cm of position error
  by the time it crashes. Both implementations still agree on *crash vs.
  finish* and the crash happens within ~40 ms of each other; this looks
  like irreducible floating-point-ordering sensitivity right at a
  bifurcation point (JS and CPython don't guarantee identical rounding for
  the same formula evaluated in a different operation order), not a
  modelling bug -- but it means genomes that win by *barely* surviving a
  near-spin should be treated with a few ticks of suspicion on their exact
  crash/finish boundary.

**The live-game side, now that it's been run:** `replay.py` was run a
handful of times against the real page in Practice mode (never the
leaderboard) to wire up `live_driver.py` and the dashboard's live view.
Practice mode was confirmed on every time via the page's own text before
any input was sent. Findings:
- The input mechanism is exactly as documented on the page: hold anywhere
  (or the Up arrow) to accelerate, release to brake, Space restarts.
- The physics-level fidelity work above holds up -- the live car follows
  the same track, takes the same corners, and visibly drives most of a lap
  correctly.
- **What doesn't yet match**: the *exact* press/release timing a genome
  was tuned for doesn't always survive live -- the live car has spun out
  earlier than the simulator predicted for the same genome, more than
  once. This is a sync-point problem (exactly when the live page's "t=0"
  begins relative to our script's first input, which depends on the
  React/UI layer we never extracted, not the physics module we did) --
  not evidence the physics port itself is wrong. `live_common.py` detects
  the page's own "Spun out / Tap to restart" prompt and recovers instead
  of cascading into repeated spins, which is as far as this went; tighter
  sync is an open, honest gap, not a claimed fix.

**Bottom line: treat every lap time in this README as a simulator
prediction until you've watched (or run) a live replay yourself and seen
it survive the whole lap.** The dashboard's live panel shows this
honestly too -- when the live car spins out, that's what's actually
happening on the real site, not a mocked-up animation.

## Results

Best lap found by `optimize.py` on Las Vegas so far (still running --
these numbers will already be stale by the time you read them; check the
dashboard or `runs/lasvegas_best.json` for the current one):

| | time |
|---|---|
| **Sequence optimizer (simulated)** | **0:45.000**, improving |
| **Policy network, phase 2 (simulated)** | 1:22.4, 2 generations after an init fix (see Phase 2) |
| Gold medal | 0:43.000 |
| Silver medal | 0:43.800 |
| Bronze medal | 0:47.200 (**beaten**) |

It went from crashing 500 m into the lap to a clean, bronze-beating lap in
about 10 minutes, then kept shaving tenths off since. See
`runs/lasvegas_plot.png` (`python3 plot.py --circuit lasvegas`) or the
live dashboard's learning curve and Training Score. These are simulator
predictions, cross-checked tick-for-tick against the real physics code and
spot-checked live a handful of times -- see Fidelity for exactly what that
does and doesn't cover.

**The sequence optimizer, the policy trainer, the live-game view, and the
dashboard are all running right now** on this machine, pause/resume-able
from the dashboard at any time.

## To start it yourself (next time)

```bash
cd quali-prix-ai
source .venv/bin/activate
python3 optimize.py --circuit lasvegas         # terminal 1
python3 policy/train.py --circuit lasvegas     # terminal 2 (optional, for the NN panel)
python3 dashboard.py                            # terminal 3
```
Then click "Watch it play live" on the dashboard whenever you want the
fourth panel. Pause/Resume everything from the dashboard; `Ctrl+C` a
terminal to fully stop that engine, and re-run the same command to resume
from its checkpoint.

## Known weaknesses / honest TODOs

- **Throughput**: this is pure Python (plus numpy for track geometry at
  load time), parallelized with `multiprocessing` across cores -- it is
  *not* "thousands of laps per second," because the real physics (full
  bicycle-model dynamics, Pacejka tyres, a real-time racing-line QP solve
  every 1/20 s) is far more expensive than the toy placeholder physics in
  the original brief assumed. On this machine, a single core does on the
  order of 10-20 attempts/second depending on how far the car gets before
  crashing; across cores that's still enough for tens of thousands of
  generations overnight, which is plenty for a single-circuit sequence
  search to converge, but it's not the eye-popping throughput number a
  simpler physics model would give. The honest path to faster-than-that
  would be a numba-JIT'd or Rust/Cython port of `step()`, or vectorizing
  the simulation across the whole population with numpy (looping over
  ticks instead of over genomes) -- both are plausible follow-ups, neither
  is done here.
- The evolutionary loop in `optimize.py` is the same elitist mutate-and-
  select design as the original brief's starter code (with a shrinking
  mutation sigma on stalls); it is not CMA-ES, and can still get stuck in
  local optima on a technical circuit. Restarts or a proper CMA-ES would be
  a reasonable upgrade for phase 1 if the overnight run plateaus.
- `policy/train.py`'s ES is a simple one (mirrored sampling + rank
  shaping), not CMA-ES -- fine for ~113 parameters, probably not for a much
  bigger network.
- The live-game view's input-timing sync is looser than the simulator (see
  Fidelity) -- it's a real, open gap in exactly when "t=0" should be
  relative to the live page's own readiness, not something fixed in this
  session, just detected and recovered from cleanly instead of cascading.
- The gamified Training Score/reward feed on the dashboard is derived
  purely from existing fitness history (no new backend instrumentation) --
  it's a reframing of the same number `optimize.py` already optimizes, not
  a separately-tuned reward function.
- Only Las Vegas and Sao Paulo are extracted (the two circuits live so
  far). `sim.py get_track()` and `extract/raw_data` generalize to any
  future circuit using the same "v2" handling model without code changes,
  as long as its `<id>.json` / `.bin` / `-terrain.json` are fetched into
  `extract/raw_data/` the same way.
