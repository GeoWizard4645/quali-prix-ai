"""
Overnight evolutionary optimizer for a single circuit.

A genome is a sorted list of switch distances in metres, in the
cumulative-distance domain documented in sim.py (0 = the start of the
simulated attempt, 100 m before the line; track.length = lap_length +
runup_m). The car holds the button from 0 to switches[0], releases to
switches[1], and so on, starting held.

Fitness: a finished lap is scored by lap time (seconds). A crash is scored
1000 + (track.length - dist_reached), so there is a gradient toward getting
further before crashing, and any finished lap beats any crash.

Usage:
    python3 optimize.py --circuit lasvegas
    python3 optimize.py --circuit lasvegas --hours 8   # optional safety cap

Runs until paused/stopped from dashboard.py (or --hours elapses, if given).
Checkpoints runs/<circuit>_best.json on every improvement and appends one
row to runs/<circuit>_history.csv per improvement (generation, elapsed
seconds, best score); writes runs/<circuit>_live.json every generation for
the dashboard.
"""
import argparse
import csv
import json
import os
import time
from collections import deque
from multiprocessing import Pool

import numpy as np

import sim
from control import control_path, read_control, wait_while_paused
from sim import DT, get_track

TRACK = None

# Medal benchmark times shown on the game's own circuit page, where known.
MEDALS = {
    "lasvegas": {"gold": 43.000, "silver": 43.800, "bronze": 47.200},
}


def _init_worker(circuit: str, data_dir: str):
    global TRACK
    TRACK = get_track(circuit, data_dir)


def evaluate(genome):
    r = sim.simulate(TRACK, genome)
    fit = r.time if r.finished else 1000.0 + (TRACK.length - r.dist)
    return fit, r.dist, r.finished


def mutate(g, L, rng, sigma, focus):
    g = list(g)
    roll = rng.random()
    if roll < 0.50 and g:
        i = rng.integers(len(g))
        g[i] += rng.normal(0, sigma)
    elif roll < 0.72:
        a = rng.uniform(max(0.0, focus - 400), max(focus, 1.0))
        g += [a, a + rng.uniform(5, 150)]
    elif roll < 0.85 and len(g) >= 2:
        i = rng.integers(len(g) - 1)
        del g[i:i + 2]
    elif roll < 0.93 and g:
        i = rng.integers(len(g))
        del g[i]
    else:
        g = [x + rng.normal(0, sigma * 0.5) for x in g]
    g = sorted(min(max(x, 0.0), L) for x in g)
    # collapse near-duplicate switches (they're a no-op pair anyway)
    cleaned = []
    for x in g:
        if not cleaned or x - cleaned[-1] > 0.05:
            cleaned.append(x)
    return cleaned


def score_population(pool, genomes):
    results = pool.map(evaluate, genomes)
    return [(f, d, fin, g) for (f, d, fin), g in zip(results, genomes)]


def save_best(path, track, genome, fit, finished, dist):
    r = sim.simulate(track, genome, record=True)
    events_out = [(tick, bool(held)) for tick, held in r.events]
    with open(path, "w") as f:
        json.dump({
            "circuit": track.circuit_id,
            "fitness": fit,
            "lap_time": r.time if r.finished else None,
            "finished": r.finished,
            "dist_reached": r.dist,
            "lap_length": track.lap_length,
            "sim_length": track.length,
            "runup_m": track.runup_m,
            "dt": DT,
            "switches_m": genome,
            "events": events_out,
        }, f, indent=1)


def save_live(path, track, gen, elapsed, sigma, pop, elite_n, workers,
              best_fit, best_fin, best_dist, trajectory=None, paused=False,
              recent_success_rate=None, total_evaluated=0, total_finished=0):
    payload = {
        "circuit": track.circuit_id,
        "generation": gen,
        "elapsed_s": round(elapsed, 1),
        "pop": pop, "elite": elite_n, "workers": workers, "sigma": round(sigma, 3),
        "best_fitness": best_fit,
        "best_finished": best_fin,
        "best_lap_time": best_fit if best_fin else None,
        "best_dist": best_dist,
        "lap_length": track.lap_length,
        "track_length": track.length,
        "medals": MEDALS.get(track.circuit_id),
        "paused": paused,
        "recent_success_rate": recent_success_rate,
        "total_attempts": total_evaluated,
        "total_finished": total_finished,
    }
    if trajectory is not None:
        payload["trajectory"] = trajectory
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=None,
                     help="optional safety cap in hours; omit to run until paused/stopped from the dashboard")
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--circuit", default="lasvegas")
    ap.add_argument("--data-dir", default="extract/raw_data")
    ap.add_argument("--out", default=None, help="default: runs/<circuit>_best.json")
    ap.add_argument("--live-out", default=None, help="default: runs/<circuit>_live.json (for dashboard.py)")
    ap.add_argument("--no-resume", action="store_true", help="ignore any existing checkpoint and start fresh")
    args = ap.parse_args()

    out_path = args.out or f"runs/{args.circuit}_best.json"
    live_path = args.live_out or f"runs/{args.circuit}_live.json"
    hist_path = os.path.join(os.path.dirname(out_path) or ".", f"{args.circuit}_history.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    track = get_track(args.circuit, args.data_dir)
    rng = np.random.default_rng()

    seed = []
    if not args.no_resume and os.path.exists(out_path):
        with open(out_path) as f:
            seed = json.load(f).get("switches_m", [])
        print(f"resuming from {out_path} ({len(seed)} switches)", flush=True)

    sigma, stale, gen = 25.0, 0, 0
    best_fit = float("inf")
    best_dist = 0.0
    best_fin = False
    best_trajectory = None
    t0 = time.time()
    deadline = t0 + args.hours * 3600 if args.hours else float("inf")
    ctrl_path = control_path("optimize", args.circuit)
    # How often a freshly-mutated attempt actually finishes a lap, not just
    # the single best one -- the headline fitness can look "stuck early"
    # for a while even once the population as a whole is finishing most
    # attempts, since only strict improvements get checkpointed/printed.
    recent_results = deque(maxlen=20 * args.pop)
    total_evaluated, total_finished = 0, 0

    with Pool(args.workers, initializer=_init_worker, initargs=(args.circuit, args.data_dir)) as pool, \
         open(hist_path, "a", newline="") as hf:
        hist = csv.writer(hf)
        elite = score_population(pool, [seed])

        def _on_pause_tick(_c):
            rate = (sum(recent_results) / len(recent_results)) if recent_results else None
            save_live(live_path, track, gen, time.time() - t0, sigma, args.pop, args.elite,
                      args.workers, best_fit, best_fin, best_dist, best_trajectory, paused=True,
                      recent_success_rate=rate, total_evaluated=total_evaluated, total_finished=total_finished)

        while True:
            if wait_while_paused(ctrl_path, on_tick=_on_pause_tick):
                break
            if time.time() >= deadline:
                break
            gen += 1
            kids = []
            for _ in range(args.pop):
                f, d, fin, g = elite[rng.integers(len(elite))]
                focus = track.length if fin else d
                kids.append(mutate(g, track.length, rng, sigma, focus))
            kid_results = score_population(pool, kids)
            for f, d, fin, g in kid_results:
                recent_results.append(fin)
                total_evaluated += 1
                total_finished += int(fin)
            scored = elite + kid_results
            scored.sort(key=lambda x: x[0])
            dedup, seen = [], set()
            for f, d, fin, g in scored:
                key = tuple(np.round(g, 1))
                if key not in seen:
                    seen.add(key)
                    dedup.append((f, d, fin, g))
            elite = dedup[:args.elite]

            if elite[0][0] < best_fit - 1e-9:
                best_fit, best_dist, best_fin, best_g = elite[0]
                save_best(out_path, track, best_g, best_fit, best_fin, best_dist)
                hist.writerow([gen, round(time.time() - t0, 1), round(best_fit, 4)])
                hf.flush()
                best_trajectory = sim.simulate_trajectory(track, best_g, stride=8)
                status = "FINISHED" if best_fin else f"dist={best_dist:.1f}/{track.length:.1f}"
                print(f"gen {gen}: fitness={best_fit:.3f} {status} sigma={sigma:.2f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
                stale = 0
            else:
                stale += 1
                if stale >= 40:
                    sigma = max(0.25, sigma * 0.7)
                    stale = 0

            recent_rate = sum(recent_results) / len(recent_results)
            save_live(live_path, track, gen, time.time() - t0, sigma, args.pop, args.elite,
                      args.workers, best_fit, best_fin, best_dist, best_trajectory, paused=False,
                      recent_success_rate=recent_rate, total_evaluated=total_evaluated,
                      total_finished=total_finished)

    print(f"done (stopped or --hours elapsed). best fitness={best_fit:.3f} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
