"""
Evaluates a trained policy (from train.py) on a circuit -- typically one it
was NOT trained on, to test zero-shot generalization -- and compares it
against the per-circuit sequence optimizer's result on the same circuit,
if a runs/<circuit>_best.json checkpoint exists.

Usage:
    python3 policy/eval.py --policy runs/policy_lasvegas.npy --circuit saopaulo
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sim
from policy.net import rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--circuit", default="saopaulo")
    ap.add_argument("--data-dir", default="extract/raw_data")
    ap.add_argument("--sequence-best", default=None, help="default: runs/<circuit>_best.json")
    args = ap.parse_args()

    weights = np.load(args.policy)
    track = sim.get_track(args.circuit, args.data_dir)
    r = rollout(track, weights)

    print(f"policy {args.policy} on {args.circuit}:")
    if r.finished:
        print(f"  FINISHED in {r.time:.3f}s ({r.time/60:.2f} min)")
    else:
        print(f"  crashed/timed out after {r.time:.2f}s, {r.dist:.1f}/{track.length:.1f} m covered")

    seq_path = args.sequence_best or f"runs/{args.circuit}_best.json"
    if os.path.exists(seq_path):
        seq = json.load(open(seq_path))
        print(f"\nper-circuit sequence optimizer on {args.circuit} ({seq_path}):")
        if seq["finished"]:
            print(f"  FINISHED in {seq['lap_time']:.3f}s")
            if r.finished:
                gap = r.time - seq["lap_time"]
                print(f"\n  policy is {gap:+.3f}s vs the circuit-specific optimizer "
                      f"({'slower' if gap > 0 else 'faster'})")
        else:
            print(f"  crashed, dist_reached={seq['dist_reached']:.1f} m")
    else:
        print(f"\n(no sequence-optimizer checkpoint found at {seq_path} to compare against)")


if __name__ == "__main__":
    main()
