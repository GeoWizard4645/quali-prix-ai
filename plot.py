"""
Plots:
  1. The learning curve from runs/<circuit>_history.csv (best fitness vs
     wall-clock time).
  2. The track outline for the best saved lap, coloured by speed, and a
     second panel coloured by throttle/brake.

Usage:
    python3 plot.py --circuit lasvegas
    python3 plot.py --circuit lasvegas --best runs/lasvegas_best.json --out runs/lasvegas_plot.png
"""
import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import sim


def plot_learning_curve(ax, history_path):
    if not os.path.exists(history_path):
        ax.text(0.5, 0.5, "no history yet", ha="center", va="center", transform=ax.transAxes)
        return
    gens, elapsed, fitness = [], [], []
    with open(history_path) as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            gens.append(int(row[0]))
            elapsed.append(float(row[1]))
            fitness.append(float(row[2]))
    ax.plot(elapsed, fitness, marker=".", linewidth=1)
    ax.set_xlabel("elapsed seconds")
    ax.set_ylabel("best fitness (lap time, s, once finished; else 1000+remaining m)")
    ax.set_title("Learning curve")
    ax.grid(alpha=0.3)
    if fitness and min(fitness) < 1000:
        ax.axhline(1000, color="gray", linestyle="--", linewidth=0.8, label="finish threshold")
        ax.legend()


def plot_track(fig, gs_row, track, switches_m, title_suffix=""):
    traj = sim.simulate_trajectory(track, switches_m, stride=1)
    xs, ys, speeds, throttle = map(np.array, (traj["x"], traj["y"], traj["speed"], traj["throttle"]))

    ax1 = fig.add_subplot(gs_row[0])
    sc = ax1.scatter(xs, ys, c=speeds, cmap="viridis", s=4)
    ax1.plot(track.X, track.Y, color="lightgray", linewidth=0.5, zorder=0)
    ax1.set_aspect("equal")
    ax1.set_title(f"Speed (m/s){title_suffix}")
    fig.colorbar(sc, ax=ax1, shrink=0.7)

    ax2 = fig.add_subplot(gs_row[1])
    sc2 = ax2.scatter(xs, ys, c=throttle, cmap="RdYlGn", vmin=-1, vmax=1, s=4)
    ax2.plot(track.X, track.Y, color="lightgray", linewidth=0.5, zorder=0)
    ax2.set_aspect("equal")
    ax2.set_title(f"Throttle(green)/Brake(red){title_suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="lasvegas")
    ap.add_argument("--data-dir", default="extract/raw_data")
    ap.add_argument("--best", default=None, help="default: runs/<circuit>_best.json")
    ap.add_argument("--history", default=None, help="default: runs/<circuit>_history.csv")
    ap.add_argument("--out", default=None, help="default: runs/<circuit>_plot.png")
    args = ap.parse_args()

    best_path = args.best or f"runs/{args.circuit}_best.json"
    history_path = args.history or f"runs/{args.circuit}_history.csv"
    out_path = args.out or f"runs/{args.circuit}_plot.png"

    track = sim.get_track(args.circuit, args.data_dir)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    plot_learning_curve(fig.add_subplot(gs[0, :]), history_path)

    if os.path.exists(best_path):
        best = json.load(open(best_path))
        plot_track(fig, [gs[1, 0], gs[1, 1]], track, best["switches_m"],
                   title_suffix=f" -- {'FINISHED ' + format(best['lap_time'], '.3f') + 's' if best['finished'] else 'crash'}")
    else:
        fig.add_subplot(gs[1, :]).text(0.5, 0.5, "no best.json yet", ha="center", va="center")

    fig.suptitle(f"Quali Prix -- {track.circuit_id}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
