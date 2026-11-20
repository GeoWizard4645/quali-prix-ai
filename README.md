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
python3 optimize.py --hours 8 --circuit lasvegas   # terminal 1: let it learn
python3 dashboard.py                                # terminal 2: watch it
```
Open the URL `dashboard.py` prints. `Ctrl+C` either one to stop; re-run the
same `optimize.py` command any time to pick up where it left off. That's
the whole thing -- everything past this point is detail/reference.

On macOS, if you want the laptop to keep running overnight without going
to sleep, prefix the optimize command with `caffeinate -s`.

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
plot.py            learning curve + speed/throttle track plots
dashboard.py       live local web dashboard (watch it learn, with friends)
replay.py          Playwright replay against the LIVE page, Practice mode only
extract/
  NOTES.md         how the physics and circuit data were found
  raw_data/        the circuit metadata/binary/terrain files (see NOTES.md)
  verify/          the live-bundle fidelity-check harness (fetches fresh, not vendored)
tests/             determinism/sanity tests + the live fidelity cross-check
runs/              best.json / history.csv per circuit (gitignored)
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
that reads `runs/*.json`/`*.csv` and serves a live page: the track with
the current best lap animated on it (colour = speed), a learning-curve
chart, gold/silver/bronze medal comparisons, and -- once `policy/train.py`
is also running -- a live node-link view of the policy network's weights.
It prints two URLs: one for `localhost`, and one on your LAN
(`http://<your-ip>:8743`) that anyone on the same Wi-Fi can open to watch
along. That's LAN-only by design (no cloud, nothing exposed to the wider
internet) -- ask if you want it reachable from outside your network too.

The first time the car completes a clean lap, the dashboard names it
**VIV** with a little reveal popup, and labels it that way from then on.

`dashboard.py` only reads files -- it doesn't touch the optimizer process,
so start/stop/restart `optimize.py` and `policy/train.py` independently of
it, in any order.

## Running it overnight

```bash
python3 optimize.py --hours 8 --circuit lasvegas
```

- Checkpoints `runs/lasvegas_best.json` every time it improves, and appends
  a row (generation, elapsed seconds, best fitness) to
  `runs/lasvegas_history.csv`.
- **Resuming**: just run the same command again (optionally with a
  different `--hours`) -- it loads the existing checkpoint and keeps going.
  Pass `--no-resume` to start over.
- Uses all CPU cores by default (`--workers`); population/elite size and
  the mutation step size (`sigma`, auto-shrinks when progress stalls) are
  tunable flags -- see `python3 optimize.py --help`.
- Fitness: a finished lap scores its lap time; a crash scores
  `1000 + (track.length - distance_reached)`, so there's always a gradient
  toward getting further before crashing, and any finished lap beats any
  crash.
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
python3 policy/train.py --hours 4 --circuit lasvegas
python3 policy/eval.py --policy runs/policy_lasvegas.npy --circuit saopaulo
```

### Early result (not a full overnight run)

A **2-minute** smoke-test training run (`pop=32`, 10 cores) already found a
policy that completes a clean Las Vegas lap in 102.3 s -- nowhere near the
tuned sequence optimizer's time, but a genuine full lap from a generalist
controller after 19 generations, which is a good sign the approach works.
Tested zero-shot on Sao Paulo, that same lightly-trained policy did **not**
complete a lap (ran to the 150 s cutoff, 2555/4400 m covered) -- expected
from 2 minutes of training, not yet evidence either way about whether a
properly-trained policy would generalize. That's the real experiment to
run overnight; `policy/train.py --hours 4` (or more) followed by
`policy/eval.py` is the next step, and genuinely hasn't been done yet in
this session.

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

**What is NOT verified:** no lap recorded from the live page has been
replayed back through it. `replay.py` is written and ready
(`python3 replay.py --best runs/lasvegas_best.json`), but running it drives
the actual page with real input, and the brief that this was built from
is explicit that doing that (or an agent-driven "dry run" of the live game
with screenshots) should wait until asked for, to avoid the cost of
repeated live/visual runs. **Treat every lap time in this README as a
simulator prediction, not a confirmed real time, until you've run
`replay.py` (in Practice mode -- it never submits to any leaderboard) and
checked the gap.** If you ask for that dry run, it'll be limited to 3-5
live attempts.

## Results

Best lap found by `optimize.py` on Las Vegas, after a 24-minute test run
(not the full overnight run) on 10 cores, 47 improving generations:

| | time |
|---|---|
| **This optimizer (simulated)** | **0:46.342** |
| Gold medal | 0:43.000 |
| Silver medal | 0:43.800 |
| Bronze medal | 0:47.200 (**beaten**) |

It went from crashing 500 m into the lap to a clean, bronze-beating lap in
about 10 minutes, then kept shaving tenths off for the rest of the run --
see `runs/lasvegas_plot.png` (`python3 plot.py --circuit lasvegas`) or the
live dashboard's learning curve. As above, this is a simulator prediction;
the live-replay gap is unmeasured (see Fidelity).

**The real 8-hour overnight run is running right now** (started
2026-11-20, resumed from that 46.342s checkpoint), alongside
`policy/train.py` and `dashboard.py` -- see the next section to watch it.

## To start it yourself (next time)

```bash
cd quali-prix-ai
source .venv/bin/activate
python3 optimize.py --hours 8 --circuit lasvegas   # terminal 1
python3 policy/train.py --hours 8 --circuit lasvegas   # terminal 2 (optional, for the NN panel)
python3 dashboard.py   # terminal 3
```

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
- `replay.py`'s synchronization with the live page's lap start is a
  best-effort heuristic (see its docstring) -- it hasn't been run live, so
  treat the sync point as unverified until you've checked it against a
  manual attempt.
- Only Las Vegas and Sao Paulo are extracted (the two circuits live so
  far). `sim.py get_track()` and `extract/raw_data` generalize to any
  future circuit using the same "v2" handling model without code changes,
  as long as its `<id>.json` / `.bin` / `-terrain.json` are fetched into
  `extract/raw_data/` the same way.
