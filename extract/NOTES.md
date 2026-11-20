# Where the physics and data came from

## The physics

The game's page (`https://gluefold.co.uk/quali-prix`) is a Next.js/Turbopack
app. Its HTML references about a dozen JS chunks under
`/_next/static/immutable/chunks/`. Those are ordinary static assets served
to every visitor -- fetching them is no different from loading the page in
a browser.

One of those chunks (identified by searching the downloaded chunks for
physics-sounding keywords -- `accel`, `brake`, `grip`, `lasvegas`,
`terrain`, `spinout`...) turned out to contain the entire vehicle-physics
and track-loading module, minified. Running it through `prettier` (a
formatter, not a decompiler -- it only re-indents and expands the existing
AST, it doesn't rename anything or recover any information that wasn't
already there) made it readable enough to trace through by hand: constants,
the tyre model, the autonomous steering/racing-line planner, lap timing,
crash detection, circuit loading.

From there, `sim.py` is a **rewrite**, not a copy: same formulas, same
constants, same control flow, new (English, documented) names, written from
scratch. Two reasons for doing it that way instead of just vendoring the
extracted file:

1. The obfuscated bundle is the game author's own code. Reverse-engineering
   it to understand and reproduce its *behaviour* (interoperability -- so a
   personal tool can predict what the real game will do) is one thing;
   redistributing their literal source in a repo that's headed to GitHub is
   a different thing, and not one we should do.
2. A rewrite with real names is the only way a human (or a future session)
   can actually verify, maintain, or extend this.

That said, "rewrite it and hope it's equivalent" isn't good enough on its
own, which is why `tests/test_fidelity_vs_real_bundle.py` exists: it
fetches that same chunk fresh at test time, runs it standalone under Node
via a ~30-line shim (`extract/verify/shim.js` -- it only intercepts the two
calls the bundle's module system makes to register exports and import other
modules; it doesn't patch or reinterpret any of the physics), and diffs its
output tick-by-tick against `sim.py` on the same genomes. See
`README.md`'s Fidelity section for the numbers that test produces, and the
"known caveat" about one adversarial, knife-edge genome.

One concrete thing that mattered: the real game computes `sin`/`cos`/`atan`/
`atan2` with its own fixed-iteration polynomial approximations rather than
the platform's `Math.sin` etc (presumably for speed and/or determinism in
the hot per-tick loop). `sim.py` reproduces those exact approximations
(`fsin`/`fcos`/`fatan`/`fatan2` near the top of the file) rather than using
`math.sin` etc -- using the "more accurate" real trig functions actually
made fidelity *worse* on long runs, because the two implementations'
different rounding slowly diverges in a system with feedback (the
autonomous steering) and occasional near-threshold events (barely surviving
a slide). Swapping in the exact same approximations removed nearly all of
that drift.

## The circuit data

The page itself fetches, per circuit:
- `/race/circuits/<id>.json` -- metadata: lap length, sample spacing,
  track width, wall clearance, DRS zones, which binary/terrain file to load.
- `/race/circuits/<id>.bin` -- `3*M` float64s: the centreline's X, Y, and a
  static baseline racing-line lateral offset, at `DS`-metre spacing.
- `/race/circuits/<id>-terrain.json` -- heights, road edges, wall
  clearances, banking, the circuit's "flying lap" entry speed, and any
  per-circuit overrides to the "v2" handling model.

These are plain data files any visitor's browser downloads to render/drive
the track; `extract/raw_data/` has local copies of the two shipped circuits
(Las Vegas, the live one; Sao Paulo, the previous one) fetched the same way,
each with an `_licence`/`_credit` field crediting the real source:
circuit outlines from `bacinger/f1-circuits` (MIT), heights from USGS 3DEP
via Open Topo Data.

We did **not** fetch `-map.json` or `-ground.json` (scenery/rendering data)
-- they aren't needed for physics, only for the game's own visuals.

## What we deliberately did not do

- No probing of the submission/leaderboard API (`/api/quali-challenge/...`).
- No scraping beyond the handful of static files above.
- No live-browser "dry run" of the actual game (driving it with
  screenshots) while building this -- see README.md's Verification section
  for what that means for how confident you should be in the live-replay
  number, and run `replay.py` yourself (or ask the agent to, a few times)
  to get it.
