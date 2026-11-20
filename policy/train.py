"""
Phase 2: trains a small generalizing policy (see net.py) with a simple
evolution strategy (mirrored-sampling OpenAI-ES, rank-based fitness
shaping -- no CMA-ES dependency needed for a ~113-parameter network).

Train on one circuit, then run eval.py to test zero-shot on another.

Usage:
    python3 policy/train.py --hours 4 --circuit lasvegas
"""
import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sim
from control import control_path, wait_while_paused
from policy.net import N_HIDDEN, N_INPUTS, N_OUTPUTS, N_WEIGHTS, initial_weights, rollout, rollout_traced, unpack_weights

TRACK = None


def _init_worker(circuit, data_dir):
    global TRACK
    TRACK = sim.get_track(circuit, data_dir)


def _fitness(weights):
    r = rollout(TRACK, weights)
    return r.time if r.finished else 1000.0 + (TRACK.length - r.dist)


def _rank_shaped_weights(fitnesses: np.ndarray) -> np.ndarray:
    """Centered rank transform (lower fitness = better, so invert rank)."""
    n = len(fitnesses)
    order = np.argsort(fitnesses)  # ascending: best (lowest) first
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    # best gets the highest weight
    shaped = (n - 1 - ranks) / (n - 1) - 0.5
    return shaped


def save_live(path, track, gen, elapsed, sigma, lr, pop, workers, best_fit, weights,
              trajectory=None, paused=False):
    w1, b1, w2, b2 = unpack_weights(weights)
    payload = {
        "circuit": track.circuit_id,
        "generation": gen,
        "elapsed_s": round(elapsed, 1),
        "sigma": sigma, "lr": lr, "pop": pop, "workers": workers,
        "best_fitness": best_fit,
        "architecture": {"n_inputs": N_INPUTS, "n_hidden": N_HIDDEN, "n_outputs": N_OUTPUTS},
        "weights": {"w1": w1.tolist(), "b1": b1.tolist(), "w2": w2.tolist(), "b2": b2.tolist()},
        "paused": paused,
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
    ap.add_argument("--pop", type=int, default=64, help="must be even (mirrored sampling)")
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--circuit", default="lasvegas")
    ap.add_argument("--data-dir", default="extract/raw_data")
    ap.add_argument("--out", default=None, help="default: runs/policy_<circuit>.npy")
    ap.add_argument("--live-out", default=None, help="default: runs/policy_<circuit>_live.json")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    if args.pop % 2 != 0:
        args.pop += 1

    out_path = args.out or f"runs/policy_{args.circuit}.npy"
    live_path = args.live_out or f"runs/policy_{args.circuit}_live.json"
    hist_path = f"runs/policy_{args.circuit}_history.csv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    track_for_live = sim.get_track(args.circuit, args.data_dir)

    rng = np.random.default_rng()
    mu = initial_weights(rng)
    if not args.no_resume and os.path.exists(out_path):
        mu = np.load(out_path)
        print(f"resuming from {out_path}", flush=True)

    t0 = time.time()
    deadline = t0 + args.hours * 3600 if args.hours else float("inf")
    best_fit = float("inf")
    best_weights = mu.copy()
    best_trajectory = None
    gen = 0
    sigma, stale = args.sigma, 0
    ctrl_path = control_path("policy", args.circuit)

    with Pool(args.workers, initializer=_init_worker, initargs=(args.circuit, args.data_dir)) as pool, \
         open(hist_path, "a", newline="") as hf:
        hist = csv.writer(hf)
        half = args.pop // 2

        def _on_pause_tick(_c):
            save_live(live_path, track_for_live, gen, time.time() - t0, sigma, args.lr,
                      args.pop, args.workers, best_fit, best_weights, best_trajectory, paused=True)

        while True:
            if wait_while_paused(ctrl_path, on_tick=_on_pause_tick):
                break
            if time.time() >= deadline:
                break
            gen += 1
            eps = rng.normal(0, 1, (half, N_WEIGHTS))
            candidates = np.concatenate([mu + sigma * eps, mu - sigma * eps], axis=0)
            fitnesses = np.array(pool.map(_fitness, list(candidates)))

            cur_best = fitnesses.min()
            if cur_best < best_fit - 1e-6:
                best_fit = cur_best
                best_weights = candidates[fitnesses.argmin()]
                np.save(out_path, best_weights)
                hist.writerow([gen, round(time.time() - t0, 1), round(best_fit, 4)])
                hf.flush()
                traced = rollout_traced(track_for_live, best_weights, stride=8)
                best_trajectory = traced["trajectory"]
                print(f"gen {gen}: best_fitness={best_fit:.3f} sigma={sigma:.2f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
                stale = 0
                sigma = max(args.sigma, sigma * 0.9)  # ease back toward baseline once moving again
            else:
                # Unlike optimize.py's sequence search (which SHRINKS its
                # mutation step when it's close to a good answer and just
                # needs to fine-tune), a stuck policy is usually stuck in a
                # genuinely bad region of weight space -- so widen the
                # search instead, then settle back down once it escapes.
                stale += 1
                if stale >= 15:
                    sigma = min(args.sigma * 4, sigma * 1.3)
                    stale = 0

            # Standard mirrored-sampling ES update: step mu along the
            # fitness-weighted average perturbation direction. The best
            # individual ever seen is checkpointed separately above --
            # mu itself is only ever moved by this gradient step, never
            # snapped to a point sample, so the next generation's mirrored
            # perturbations stay centred on a consistent point.
            shaped = _rank_shaped_weights(fitnesses[:half]) - _rank_shaped_weights(fitnesses[half:])
            grad = (eps * shaped[:, None]).mean(axis=0)
            mu = mu + args.lr * grad

            save_live(live_path, track_for_live, gen, time.time() - t0, sigma, args.lr,
                      args.pop, args.workers, best_fit, best_weights, best_trajectory, paused=False)

    print(f"done (stopped or --hours elapsed). best_fitness={best_fit:.3f} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
