"""Mirrors run_real_physics.js exactly, but runs sim.py's port. See test_fidelity_vs_real_bundle.py."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sim


def main():
    circuit, data_dir, switches_csv, max_ticks = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    switches = [float(x) for x in switches_csv.split(",") if x]

    track = sim.get_track(circuit, data_dir)
    state = sim.init_state(track)
    held, idx, cum = True, 0, 0.0
    trace = []
    outcome = "timeout"
    for tick in range(max_ticks):
        while idx < len(switches) and cum >= switches[idx]:
            held = not held
            idx += 1
        s_before = state["s"]
        ev = sim.step(track, state, held)
        s_after = state["s"]
        delta = s_after - s_before
        if delta < -track.lap_length / 2:
            delta += track.lap_length
        cum += delta
        trace.append([tick, state["s"], state["e"], state["dpsi"], state["vx"], state["vy"], state["r"]])
        if ev == "crash":
            outcome = "crash"
            break
        if ev in ("best", "lap"):
            outcome = "finished"
            break

    print(json.dumps({"outcome": outcome, "t": state["t"], "trace": trace}))


if __name__ == "__main__":
    main()
