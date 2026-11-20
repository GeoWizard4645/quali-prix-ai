"""
A tiny MLP policy mapping (speed, upcoming-curvature lookahead, grade,
previous action) to a hold/release decision, plus a rollout function that
drives sim.py's real step() function with it. Shared by train.py and
eval.py.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sim

LOOKAHEADS_M = [0.0, 20.0, 40.0, 80.0, 160.0, 320.0]
N_INPUTS = 2 + len(LOOKAHEADS_M) + 2 + 1 + 1  # speed(2) + curvature samples + grade(2) + prev_action + bias
N_HIDDEN = 8
N_OUTPUTS = 1
N_WEIGHTS = (N_INPUTS * N_HIDDEN + N_HIDDEN) + (N_HIDDEN * N_OUTPUTS + N_OUTPUTS)

SPEED_SCALE = 150.0
CURVATURE_SCALE = 0.05
GRADE_SCALE = 0.2


def feature_vector(track: sim.Track, state: dict, prev_action: float) -> np.ndarray:
    feats = [state["vx"] / SPEED_SCALE, state["vy"] / SPEED_SCALE]
    for d in LOOKAHEADS_M:
        feats.append(track.samp(track.K, state["s"] + d) / CURVATURE_SCALE)
    if track.grade is not None:
        feats.append(track.samp(track.grade, state["s"]) / GRADE_SCALE)
        feats.append(track.samp(track.grade, state["s"] + 40.0) / GRADE_SCALE)
    else:
        feats.extend([0.0, 0.0])
    feats.append(prev_action)
    feats.append(1.0)
    return np.array(feats, dtype=np.float64)


def unpack_weights(w: np.ndarray):
    i = 0
    w1 = w[i:i + N_INPUTS * N_HIDDEN].reshape(N_INPUTS, N_HIDDEN); i += N_INPUTS * N_HIDDEN
    b1 = w[i:i + N_HIDDEN]; i += N_HIDDEN
    w2 = w[i:i + N_HIDDEN * N_OUTPUTS].reshape(N_HIDDEN, N_OUTPUTS); i += N_HIDDEN * N_OUTPUTS
    b2 = w[i:i + N_OUTPUTS]
    return w1, b1, w2, b2


def forward(weights: np.ndarray, features: np.ndarray) -> float:
    w1, b1, w2, b2 = unpack_weights(weights)
    hidden = np.tanh(features @ w1 + b1)
    out = np.tanh(hidden @ w2 + b2)
    return float(out[0])


def rollout(track: sim.Track, weights: np.ndarray, max_time: float = sim.MAX_LAP_TIME):
    """Drives sim.step() with the policy instead of a pre-computed switch list."""
    state = sim.init_state(track)
    prev_action = 1.0
    cum_dist = 0.0
    max_ticks = int(max_time / sim.DT)
    for _ in range(max_ticks):
        score = forward(weights, feature_vector(track, state, prev_action))
        held = score > 0.0
        prev_action = 1.0 if held else -1.0

        s_before = state["s"]
        ev = sim.step(track, state, held)
        s_after = state["s"]
        delta = s_after - s_before
        if delta < -track.lap_length / 2:
            delta += track.lap_length
        cum_dist += delta

        if ev == "crash":
            return sim.Result(time=state["t"], finished=False, dist=min(cum_dist, track.length), crashed=True)
        if ev in ("best", "lap"):
            return sim.Result(time=state["last"], finished=True, dist=track.length)
    return sim.Result(time=state["t"], finished=False, dist=min(cum_dist, track.length))
