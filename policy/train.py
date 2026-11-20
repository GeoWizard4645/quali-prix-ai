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
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sim
from policy.net import N_WEIGHTS, rollout

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=4)
    ap.add_argument("--pop", type=int, default=64, help="must be even (mirrored sampling)")
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--circuit", default="lasvegas")
    ap.add_argument("--data-dir", default="extract/raw_data")
    ap.add_argument("--out", default=None, help="default: runs/policy_<circuit>.npy")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    if args.pop % 2 != 0:
        args.pop += 1

    out_path = args.out or f"runs/policy_{args.circuit}.npy"
    hist_path = f"runs/policy_{args.circuit}_history.csv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    rng = np.random.default_rng()
    mu = rng.normal(0, 0.3, N_WEIGHTS)
    if not args.no_resume and os.path.exists(out_path):
        mu = np.load(out_path)
        print(f"resuming from {out_path}", flush=True)

    t0 = time.time()
    deadline = t0 + args.hours * 3600
    best_fit = float("inf")
    gen = 0

    with Pool(args.workers, initializer=_init_worker, initargs=(args.circuit, args.data_dir)) as pool, \
         open(hist_path, "a", newline="") as hf:
        hist = csv.writer(hf)
        half = args.pop // 2
        while time.time() < deadline:
            gen += 1
            eps = rng.normal(0, 1, (half, N_WEIGHTS))
            candidates = np.concatenate([mu + args.sigma * eps, mu - args.sigma * eps], axis=0)
            fitnesses = np.array(pool.map(_fitness, list(candidates)))

            cur_best = fitnesses.min()
            if cur_best < best_fit:
                best_fit = cur_best
                np.save(out_path, candidates[fitnesses.argmin()])
                hist.writerow([gen, round(time.time() - t0, 1), round(best_fit, 4)])
                hf.flush()
                print(f"gen {gen}: best_fitness={best_fit:.3f} elapsed={time.time()-t0:.0f}s", flush=True)

            # Standard mirrored-sampling ES update: step mu along the
            # fitness-weighted average perturbation direction. The best
            # individual ever seen is checkpointed separately above --
            # mu itself is only ever moved by this gradient step, never
            # snapped to a point sample, so the next generation's mirrored
            # perturbations stay centred on a consistent point.
            shaped = _rank_shaped_weights(fitnesses[:half]) - _rank_shaped_weights(fitnesses[half:])
            grad = (eps * shaped[:, None]).mean(axis=0)
            mu = mu + args.lr * grad

    print(f"done. best_fitness={best_fit:.3f} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
