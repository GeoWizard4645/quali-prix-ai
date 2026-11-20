"""
Quali Prix physics simulator.

This is a faithful rewrite (not a literal copy) of the real game's vehicle
physics and autonomous steering/racing-line planner, reverse engineered from
the public, unminified-via-prettier JS the site serves to every visitor
(see extract/NOTES.md for exactly how, and extract/raw_data/ for the circuit
data files the page itself downloads). The formulas, constants and control
structure below are the same as the game's; the code itself has been
rewritten from scratch with descriptive names instead of copied verbatim,
both for clarity and because the obfuscated bundle is the game author's own
code, not something we should redistribute byte-for-byte.

Key facts this file encodes (see extract/NOTES.md for how each was found):
  - Tick rate: 120 Hz (DT = 1/120 s), with 4 physics sub-steps of 1/480 s
    inside every input tick.
  - The player's only control is a single boolean "held" (accelerate when
    held, brake when released); everything else -- steering, line choice --
    is an autonomous controller (pure-pursuit + a small real-time QP that
    picks a smooth, track-legal racing line each 1/20 s).
  - Both circuits shipped so far (Las Vegas, Sao Paulo) use the "v2"/
    "natural2" handling model: tyre forces via the Pacejka "magic formula",
    aero drag + downforce, elevation/grade coupling, kerbs/grass/walls, and
    a spin-and-stop recovery (rather than an instant "crash") for most
    traction loss, with a hard "crash" state reserved for a real wall/
    barrier hit at speed.
  - A lap begins 100 m before the start/finish line (a scripted "flying lap"
    entry at a per-circuit entry speed) and the timed lap is the distance
    from the line back to the line.

Distance-domain convention used by this whole project (see optimize.py):
  switch distances are metres of *cumulative distance travelled since the
  simulation started* (i.e. since 100 m before the line), not metres since
  the start/finish line. Track.length == lap_length + runup_m is the total
  distance a clean lap covers, and is what genomes are clamped against.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
HZ = 120
DT = 1.0 / HZ
N_SUBSTEPS = 4
SUBDT = DT / N_SUBSTEPS

# --------------------------------------------------------------------------
# Core vehicle physics constants ("C" in the game's own code).
# Units are whatever the game's internal tuning uses -- they do not all
# correspond to real SI quantities (e.g. `power` is not literally watts),
# but the formulas that consume them are reproduced exactly, so the
# resulting dynamics match.
# --------------------------------------------------------------------------
CAR = dict(
    m=800.0, wb_front=1.98, wb_rear=1.62, cg_height=0.22, Iz=1200.0, g=9.81,
    rho=1.2, CdA=0.95, ClA=6.5, aero_front_frac=0.45, drs_cd_mult=0.66,
    mu0=3.1, kmu=3e-5,
    Bf=18.0, Cf=1.6, Ef=0.5,
    Br=16.0, Cr=1.7, Er=0.0,
    grip_floor_front=0.85, grip_floor_rear=0.8, slip_relax=0.4,
    bite=0.07, bite_time=0.1, bite_fade=0.06, catch_rate=3.5,
    unwind_w=27.0, unwind_zeta=0.55, unwind_time=0.3,
    rear_grip_mult=1.18, rear_brake_cap=0.42, kbeta=1.0,
    power=1.6e6, coast=4500.0, brake_max=6.0e4, brake_front_frac=0.7,
    long_cap=0.72,
    steer_max=0.42, steer_rate=2.2, kff=0.6, kr=0.08, ks=2.5,
    throttle_ramp=0.06, brake_build=0.2, brake_delay=0.1, engine_ramp=0.12,
    ax_tau=0.08,
    float_gain=0.55, float_max=2.5, edge_guard=4.0,
    heat_gain=0.015, heat_cool=0.17, heat_loss=0.08,
    width=17.0, kerb=1.5, line_margin=2.4, car_half=1.0,
    kerb_mu=0.93, off_track_mu=1.0, off_track_drag=3.5e4,
    spin_beta=0.6, off_spin_beta=0.4, dirt_loss=0.12, dirt_clean=0.4,
    scrape_angle=0.22, crash_speed=40.0,
    fly_time=1.25, marshal_time=1.0, backup_m=25.0,
    plan_back=40.0, replan_deviation=2.0, replan_every=6,
)
WHEELBASE = CAR["wb_front"] + CAR["wb_rear"]

# "HANDLING_V2" defaults, overridden per-circuit by terrain["handling2"].
HANDLING_V2_DEFAULTS = dict(
    grass_mu=0.3, grass_drag=6000.0, grass_drag_max=4e4, grass_spin_beta=0.35,
    grass_power=0.7, grass_tug=5000.0,
    brake_cap2=0.8, rear_brake_cap2=0.46,
    look_max2=30.0, push_from2=40.0, push_to2=60.0, push_assist2=0.65,
    assist2=1.0,
    ClA2=3.5, kmu2=3e-5, steer_rate2=3.0, rear_grip2=1.18, brake_max2=65000.0,
    mu02=4.25, power2=1.7e6, CdA2=0.6,
    float2=0.55, float_max2=2.5, h2=0.18,
    exit_room2=1.5, exit_lag2=35.0, runup2=100.0, runupV2=122.0,
    pull2=0.0, pull_from2=20.0, pull_to2=150.0,
    natural2=False, bot_lift2=False,
)
# Map from the terrain JSON's camelCase handling2 keys to our names above.
_HANDLING2_KEY_MAP = {
    "mu02": "mu02", "ClA2": "ClA2", "power2": "power2", "CdA2": "CdA2",
    "brakeMax2": "brake_max2", "steerRate2": "steer_rate2",
    "pushAssist2": "push_assist2", "grassMu": "grass_mu",
    "grassDrag": "grass_drag", "h2": "h2", "pull2": "pull2",
    "pullFrom2": "pull_from2", "pullTo2": "pull_to2", "assist2": "assist2",
    "pushFrom2": "push_from2", "pushTo2": "push_to2",
    "natural2": "natural2", "botLift2": "bot_lift2",
}

# Geometry smoothing constants ("REAL"/old simple constants in the game's
# code; only SMOOTH and LINE are used, purely for track-geometry prep).
GEOM_SMOOTH_M = 22.0
GEOM_LINE_DAMP = 0.72

PLAN_POINTS = 50
PLAN_SPACING_IDX = 10  # control points are 10 track-samples apart
MAX_LAP_TIME = 150.0


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# --------------------------------------------------------------------------
# The game computes sin/cos/atan/atan2 with its own fixed-iteration
# polynomial approximations rather than calling into the platform's libm
# (presumably for speed/determinism in the hot per-tick loop). Early
# cross-validation against the real game's own (unmodified) physics module
# showed sub-ppm agreement for most of a lap but tick-level drift late in
# long, marginal (near-crash-recovery) runs -- exactly what you'd expect
# from a chaotic system fed two different approximations of the same
# function. Reproducing these approximations exactly (rather than using
# math.sin/cos/atan/atan2) removes that drift; see extract/NOTES.md and
# tests/test_fidelity_vs_real_bundle.py for the before/after numbers.
# --------------------------------------------------------------------------
_HALF_PI = 1.5707963267948966
_SQRT3 = 1.7320508075688772
_TAN15 = 0.2679491924311227
_PI6 = 0.5235987755982988


def _poly_sin(t: float) -> float:
    """sin(t) Taylor series, valid for |t| <= pi/4 (as used after range reduction)."""
    e = t * t
    return t * (1 + e * (-1 / 6 + e * (1 / 120 + e * (-1 / 5040 + e * (
        1 / 362880 + e * (-1 / 39916800 + e * (1 / 6227020800 + (-1 / 1307674368000) * e)))))))


def _poly_cos(t: float) -> float:
    e = t * t
    return 1 + e * (-0.5 + e * (1 / 24 + e * (-1 / 720 + e * (
        1 / 40320 + e * (-1 / 3628800 + e * (1 / 479001600 + (-1 / 87178291200) * e))))))


def _quadrant_reduce(t: float):
    k = math.floor(t / _HALF_PI + 0.5)
    return int((k % 4 + 4) % 4), t - _HALF_PI * k


def fsin(t: float) -> float:
    q, rem = _quadrant_reduce(t)
    if q == 0:
        return _poly_sin(rem)
    if q == 1:
        return _poly_cos(rem)
    if q == 2:
        return -_poly_sin(rem)
    return -_poly_cos(rem)


def fcos(t: float) -> float:
    q, rem = _quadrant_reduce(t)
    if q == 0:
        return _poly_cos(rem)
    if q == 1:
        return -_poly_sin(rem)
    if q == 2:
        return -_poly_cos(rem)
    return _poly_sin(rem)


def _atan_series(t: float) -> float:
    e = t * t
    acc = 0.0
    r = t
    for k in range(11):
        acc += (-r if k % 2 else r) / (2 * k + 1)
        r *= e
    return acc


def fatan(t: float) -> float:
    sign = 1.0
    if t < 0:
        t = -t
        sign = -1.0
    reciprocal = False
    if t > 1:
        t = 1 / t
        reciprocal = True
    offset = 0.0
    if t > _TAN15:
        t = (_SQRT3 * t - 1) / (_SQRT3 + t)
        offset = _PI6
    val = offset + _atan_series(t)
    if reciprocal:
        val = _HALF_PI - val
    return sign * val


def fatan2(y: float, x: float) -> float:
    if x > 0:
        return fatan(y / x)
    if x < 0:
        return fatan(y / x) + math.pi if y >= 0 else fatan(y / x) - math.pi
    if y > 0:
        return _HALF_PI
    if y < 0:
        return -_HALF_PI
    return 0.0


def _box_filter_circular(arr: np.ndarray, half_window: int) -> np.ndarray:
    """Circular moving-average filter, matching the game's running-sum box filter."""
    n = len(arr)
    half_window = max(1, half_window)
    out = np.empty(n, dtype=np.float64)
    acc = 0.0
    for k in range(-half_window, half_window + 1):
        acc += arr[k % n]
    for t in range(n):
        out[t] = acc / (2 * half_window + 1)
        acc += arr[(t + half_window + 1) % n] - arr[(t - half_window) % n]
    return out


def _find_peak_slip_angle(B: float, C: float, E: float) -> float:
    """Slip angle at which the Pacejka curve below peaks (used as a saturation threshold)."""
    best_val = 0.0
    best_x = 0.0
    for i in range(1, 400):
        x = 5e-4 * i
        v = _pacejka(x, B, C, E)
        if v > best_val:
            best_val = v
            best_x = x
    return best_x


def _pacejka(slip: float, B: float, C: float, E: float) -> float:
    Bx = B * slip
    return fsin(C * fatan(Bx - E * (Bx - fatan(Bx))))


def _saturate(slip: float, B: float, C: float, E: float, peak_slip: float,
              floor: float) -> float:
    val = _pacejka(slip, B, C, E)
    if abs(slip) > peak_slip and abs(val) < floor:
        return math.copysign(floor, slip)
    return val


_PEAK_SLIP_FRONT = _find_peak_slip_angle(CAR["Bf"], CAR["Cf"], CAR["Ef"])
_PEAK_SLIP_REAR = _find_peak_slip_angle(CAR["Br"], CAR["Cr"], CAR["Er"])


# --------------------------------------------------------------------------
# Track
# --------------------------------------------------------------------------
@dataclass
class Track:
    circuit_id: str
    M: int
    DS: float
    lap_length: float
    width: float
    wall_clearance: float
    drs_zones: list
    runup_m: float
    runup_speed: float
    handling: dict

    X: np.ndarray
    Y: np.ndarray
    H: np.ndarray          # centreline heading
    K: np.ndarray          # centreline curvature (smoothed, damped)
    Kc: np.ndarray         # centreline curvature (raw, short window) -- steering lead term
    PNX: np.ndarray        # left-normal unit vector x = -sin(H)
    PNY: np.ndarray        # left-normal unit vector y =  cos(H)
    baseline_offset: np.ndarray  # static baseline racing-line lateral offset ("E")
    RX: np.ndarray
    RY: np.ndarray
    RH: np.ndarray
    RK: np.ndarray         # static baseline racing-line curvature

    road_width: Optional[np.ndarray]   # None => constant `width`
    crown_offset: Optional[np.ndarray]
    apex_bias: Optional[np.ndarray]    # "LO": exit-room apex pull bias
    grade: Optional[np.ndarray]        # dz/ds
    grade_curv: Optional[np.ndarray]   # d(grade)/ds
    bank: Optional[np.ndarray]         # banking angle term
    wall_l: Optional[np.ndarray]
    wall_r: Optional[np.ndarray]

    @property
    def length(self) -> float:
        """Total distance one clean attempt covers (run-up + the timed lap)."""
        return self.lap_length + self.runup_m

    def idx(self, s: float) -> int:
        i = int(math.floor(s / self.DS)) % self.M
        return i + self.M if i < 0 else i

    def samp(self, arr: np.ndarray, s: float) -> float:
        r = s / self.DS
        r = (r % self.M + self.M) % self.M
        lo = int(math.floor(r))
        hi = (lo + 1) % self.M
        return arr[lo] + (arr[hi] - arr[lo]) * (r - lo)

    def samp_angle(self, arr: np.ndarray, s: float) -> float:
        r = s / self.DS
        r = (r % self.M + self.M) % self.M
        lo = int(math.floor(r))
        hi = (lo + 1) % self.M
        d = _wrap_pi(arr[hi] - arr[lo])
        return arr[lo] + d * (r - lo)

    def road_width_at(self, s: float) -> float:
        if self.road_width is not None:
            return self.samp(self.road_width, s)
        return self.width

    def crown_at(self, s: float) -> float:
        if self.crown_offset is not None:
            return self.samp(self.crown_offset, s)
        return 0.0

    def off_track_frac(self, lateral_e: float, s: float) -> float:
        half = self.road_width_at(s) / 2 + CAR["kerb"]
        dist_past_centre = abs(lateral_e - self.crown_at(s))
        if self.handling is not None:
            frac = (dist_past_centre - (half - CAR["car_half"])) / (2 * CAR["car_half"])
            return min(1.0, max(0.0, frac))
        return 1.0 if dist_past_centre > half else 0.0


def _load_circuit_bin(path: str, M: int):
    data = open(path, "rb").read()
    expected = 3 * M * 8
    if len(data) != expected:
        raise ValueError(f"circuit binary length {len(data)} != {expected} (M={M})")
    arr = np.frombuffer(data, dtype="<f8")
    X = arr[0:M].copy()
    Y = arr[M:2 * M].copy()
    E = arr[2 * M:3 * M].copy()
    return X, Y, E


def _centre_geometry(X: np.ndarray, Y: np.ndarray, DS: float, smooth_m: float):
    M = len(X)
    H = np.empty(M)
    for i in range(M):
        lo = (i - 1) % M
        hi = (i + 1) % M
        H[i] = fatan2(Y[hi] - Y[lo], X[hi] - X[lo])
    raw_k = np.empty(M)
    for i in range(M):
        d = _wrap_pi(H[(i + 1) % M] - H[(i - 1) % M])
        raw_k[i] = d / (2 * DS)
    half_window = max(1, round(smooth_m / DS))
    K = _box_filter_circular(raw_k, half_window)
    return H, K


def _apply_racing_line(track_arrays: dict, offset: np.ndarray):
    """Computes PNX/PNY/Kc and the static baseline racing line (RX/RY/RH/RK)."""
    X, Y, H, DS = track_arrays["X"], track_arrays["Y"], track_arrays["H"], track_arrays["DS"]
    M = len(X)
    Kc = np.empty(M)
    for i in range(M):
        d = _wrap_pi(H[(i + 4) % M] - H[(i - 4) % M])
        Kc[i] = d / (8 * DS)
    PNX = np.array([-fsin(h) for h in H])
    PNY = np.array([fcos(h) for h in H])
    RX = X + offset * PNX
    RY = Y + offset * PNY
    RH = np.empty(M)
    for i in range(M):
        lo, hi = (i - 2) % M, (i + 2) % M
        RH[i] = fatan2(RY[hi] - RY[lo], RX[hi] - RX[lo])
    raw_rk = np.empty(M)
    for i in range(M):
        d = _wrap_pi(RH[(i + 12) % M] - RH[(i - 12) % M])
        raw_rk[i] = d / (24 * DS)
    RK = _box_filter_circular(raw_rk, 10)
    return PNX, PNY, Kc, RX, RY, RH, RK


def _sample_terrain_linear(values: np.ndarray, step: float, scale: float, s: float) -> float:
    n = len(values)
    r = s / step
    r = (r % n + n) % n
    lo = int(math.floor(r))
    hi = (lo + 1) % n
    return (values[lo] + (values[hi] - values[lo]) * (r - lo)) * scale


def _apply_terrain(track_arrays: dict, handling: dict, terrain: dict):
    """Derives ZE/road_width/crown/grade/grade_curv/bank/walls/apex_bias from terrain JSON."""
    X, Y, H, DS = track_arrays["X"], track_arrays["Y"], track_arrays["H"], track_arrays["DS"]
    M = len(X)

    raw_k = np.empty(M)
    for i in range(M):
        raw_k[i] = _wrap_pi(H[(i + 4) % M] - H[(i - 4) % M]) / (8 * DS)

    step = terrain["step"]
    z = np.array(terrain["z"], dtype=np.float64)
    n_samples = len(z)

    elevation = np.array([_sample_terrain_linear(z, step, 0.01, i * DS) for i in range(M)])
    if terrain.get("w"):
        w_arr = np.array(terrain["w"], dtype=np.float64)
        road_width = np.array([_sample_terrain_linear(w_arr, step, 0.1, i * DS) for i in range(M)])
    else:
        road_width = np.zeros(M)

    crown_offset = None
    if terrain.get("wl") and terrain.get("wr"):
        wl = np.array(terrain["wl"], dtype=np.float64)
        wr = np.array(terrain["wr"], dtype=np.float64)
        crown_offset = np.empty(M)
        for i in range(M):
            el = _sample_terrain_linear(wl, step, 0.1, i * DS)
            er = _sample_terrain_linear(wr, step, 0.1, i * DS)
            road_width[i] = el + er
            crown_offset[i] = (el - er) / 2
        crown_offset = _box_filter_circular(crown_offset, max(1, round(10 / DS)))

    elevation = _box_filter_circular(_box_filter_circular(elevation, max(1, round(8 / DS))), max(1, round(8 / DS)))
    road_width = _box_filter_circular(road_width, max(1, round(10 / DS)))

    half_c = max(1, round(5 / DS))
    grade = np.empty(M)
    for i in range(M):
        grade[i] = (elevation[(i + half_c) % M] - elevation[(i - half_c) % M]) / (2 * half_c * DS)

    half_k = max(1, round(10 / DS))
    grade_curv = np.empty(M)
    for i in range(M):
        grade_curv[i] = (grade[(i + half_k) % M] - grade[(i - half_k) % M]) / (2 * half_k * DS)
    grade_curv = _box_filter_circular(grade_curv, max(1, round(10 / DS)))

    smoothed_base_k = _box_filter_circular(raw_k, max(1, round(30 / DS)))

    bank = np.zeros(M)
    for seg in terrain.get("bank", []):
        s_from, s_to, deg = seg
        i_from = round((s_from - 40) / DS)
        i_to = round((s_to + 40) / DS)
        for i in range(i_from, i_to + 1):
            idx = i % M
            pos = i * DS
            if pos < s_from:
                ramp = (pos - (s_from - 40)) / 40
            elif pos > s_to:
                ramp = (s_to + 40 - pos) / 40
            else:
                ramp = 1.0
            ang = deg * ramp * math.pi / 180
            sign = 1.0 if smoothed_base_k[idx] > 0 else -1.0
            bank[idx] = -sign * (ang - ang ** 3 / 6)
    bank = _box_filter_circular(bank, max(1, round(15 / DS)))

    apex_bias = None
    if crown_offset is not None:
        lag = max(1, round(handling["exit_lag2"] / DS))
        apex_bias_raw = np.empty(M)
        for i in range(M):
            val = 300 * smoothed_base_k[(i - lag) % M]
            apex_bias_raw[i] = handling["exit_room2"] * min(1.0, max(-1.0, val))
        apex_bias = _box_filter_circular(apex_bias_raw, max(1, round(15 / DS)))

    wall_l = wall_r = None
    if terrain.get("walls"):
        walls = terrain["walls"]
        wstep = walls["step"]
        wl_arr = np.array(walls["l"], dtype=np.float64)
        wr_arr = np.array(walls["r"], dtype=np.float64)
        wall_l = np.array([_sample_terrain_linear(wl_arr, wstep, 0.1, i * DS) for i in range(M)])
        wall_r = np.array([_sample_terrain_linear(wr_arr, wstep, 0.1, i * DS) for i in range(M)])

    return dict(
        ZE=elevation, road_width=road_width, crown_offset=crown_offset,
        apex_bias=apex_bias, grade=grade, grade_curv=grade_curv, bank=bank,
        wall_l=wall_l, wall_r=wall_r,
    )


def load_circuit(meta_path: str, bin_path: str, terrain_path: Optional[str] = None) -> Track:
    meta = json.load(open(meta_path))
    M = meta["M"]
    DS = meta["DS"]
    L = meta["L"]
    if M * DS != L:
        raise ValueError(f"bad circuit meta: L={L} DS={DS} M={M}")

    X, Y, baseline_offset = _load_circuit_bin(bin_path, M)
    H, K = _centre_geometry(X, Y, DS, GEOM_SMOOTH_M)
    K = K * GEOM_LINE_DAMP

    handling = None
    terrain = None
    if meta.get("handling") == "v2":
        handling = dict(HANDLING_V2_DEFAULTS)
        if terrain_path:
            terrain = json.load(open(terrain_path))
            for json_key, our_key in _HANDLING2_KEY_MAP.items():
                if json_key in terrain.get("handling2", {}):
                    handling[our_key] = terrain["handling2"][json_key]

    track_arrays = dict(X=X, Y=Y, H=H, DS=DS)
    PNX, PNY, Kc, RX, RY, RH, RK = _apply_racing_line(track_arrays, baseline_offset)

    terr_fields = dict(road_width=None, crown_offset=None, apex_bias=None,
                       grade=None, grade_curv=None, bank=None, wall_l=None, wall_r=None)
    if handling is not None and terrain is not None:
        terr_fields = _apply_terrain(track_arrays, handling, terrain)

    runup_m = handling["runup2"] if handling else 260.0
    runup_speed = 48.0
    if handling and terrain:
        runup_speed = terrain.get("runupV2") or handling["runupV2"]

    return Track(
        circuit_id=meta["id"], M=M, DS=DS, lap_length=L,
        width=meta.get("width", CAR["width"]),
        wall_clearance=meta.get("wall", 99.0),
        drs_zones=[tuple(z) for z in meta.get("drs", [])],
        runup_m=runup_m, runup_speed=runup_speed, handling=handling,
        X=X, Y=Y, H=H, K=K, Kc=Kc, PNX=PNX, PNY=PNY,
        baseline_offset=baseline_offset, RX=RX, RY=RY, RH=RH, RK=RK,
        road_width=terr_fields["road_width"], crown_offset=terr_fields["crown_offset"],
        apex_bias=terr_fields["apex_bias"], grade=terr_fields["grade"],
        grade_curv=terr_fields["grade_curv"], bank=terr_fields["bank"],
        wall_l=terr_fields["wall_l"], wall_r=terr_fields["wall_r"],
    )


def get_track(circuit: str = "lasvegas", data_dir: str = "extract/raw_data") -> Track:
    import os
    meta = os.path.join(data_dir, f"{circuit}.json")
    terrain = os.path.join(data_dir, f"{circuit}-terrain.json")
    binp = os.path.join(data_dir, json.load(open(meta))["data"].replace(".bin", "") + ".bin")
    return load_circuit(meta, binp, terrain if os.path.exists(terrain) else None)


# --------------------------------------------------------------------------
# Per-attempt mutable car state
# --------------------------------------------------------------------------
def init_state(track: Track, s0: Optional[float] = None) -> dict:
    s = track.lap_length - track.runup_m if s0 is None else s0
    return dict(
        s=s, e=track.samp(track.baseline_offset, s), dpsi=0.0,
        vx=track.runup_speed, vy=0.0, r=0.0,
        th=1.0, br=0.0, delta=0.0, axf=0.0, ay=0.0, ax=0.0,
        lamF=0.0, lamR=0.0, alphaF=0.0, alphaR=0.0,
        mode=0, mt=0.0, drs=0, clean=0.0,
        aFl=0.0, aRl=0.0, heatF=0.0, heatR=0.0,
        slideT=-9.0, bigSlide=0.0, biteT=0.0, unwindT=0.0, dv=0.0,
        saves=0, t=0.0, lapStart=0.0, laps=0, last=0.0, best=None, crashes=0,
        sector=0, secStart=0.0, secs=[0.0, 0.0, 0.0], started=False,
        offTrack=0.0, wall=0, dirt=0.0,
        offT=0.0, tc=1.0, ul=1.0, scrape=0, spinV=0.0,
        turnT=0, spins=0,
        pn=0, pFresh=True, pE=None, pK=None, pS=0.0,
    )


class _PlanScratch:
    """Reusable scratch buffers for the racing-line QP (mirrors the game's reused buffers)."""

    def __init__(self):
        n = PLAN_POINTS
        self.cx = np.zeros(n); self.cy = np.zeros(n)
        self.nx = np.zeros(n); self.ny = np.zeros(n)
        self.e = np.zeros(n); self.tE = np.zeros(n); self.fix = np.zeros(n, dtype=bool)
        self.lo = np.zeros(n); self.hi = np.zeros(n)
        self.A0 = np.zeros(n); self.A1 = np.zeros(n); self.A2 = np.zeros(n)
        self.r = np.zeros(n); self.w = np.zeros(n)
        self.x = np.zeros(n)


def _banded_ldl_solve(A0, A1, A2, rhs, out):
    """Solve a symmetric pentadiagonal (bandwidth 2) SPD system A0/A1/A2 @ x = rhs into `out`."""
    n = PLAN_POINTS
    d = np.zeros(n); l1 = np.zeros(n); l2 = np.zeros(n); y = np.zeros(n)
    for i in range(n):
        l2[i] = A2[i - 2] / d[i - 2] if i >= 2 else 0.0
        l1[i] = ((A1[i - 1] - (l2[i] * d[i - 2] * l1[i - 1] if i >= 2 else 0.0)) / d[i - 1]) if i >= 1 else 0.0
        d[i] = A0[i] - (l1[i] * l1[i] * d[i - 1] if i >= 1 else 0.0) - (l2[i] * l2[i] * d[i - 2] if i >= 2 else 0.0)
    for i in range(n):
        y[i] = rhs[i] - (l1[i] * y[i - 1] if i >= 1 else 0.0) - (l2[i] * y[i - 2] if i >= 2 else 0.0)
    for i in range(n):
        y[i] /= d[i]
    for i in range(n - 1, -1, -1):
        out[i] = y[i] - (l1[i + 1] * out[i + 1] if i + 1 < n else 0.0) - (l2[i + 2] * out[i + 2] if i + 2 < n else 0.0)


_SCRATCH = _PlanScratch()


def _plan_offset_at(track: Track, state: dict, s: float) -> float:
    if state["pE"] is None:
        return track.samp(track.baseline_offset, s)
    rel = ((s - state["pS"]) % track.lap_length + track.lap_length) % track.lap_length
    t = rel / 5.0
    if t >= PLAN_POINTS - 1:
        return track.samp(track.baseline_offset, s)
    i = int(math.floor(t))
    pE = state["pE"]
    return pE[i] + (pE[i + 1] - pE[i]) * (t - i)


def _plan_curvature_at(track: Track, state: dict, s: float) -> float:
    if state["pE"] is None:
        return track.samp(track.RK, s)
    rel = ((s - state["pS"]) % track.lap_length + track.lap_length) % track.lap_length
    t = rel / 5.0
    if t >= PLAN_POINTS - 1:
        return track.samp(track.RK, s)
    i = int(math.floor(t))
    pK = state["pK"]
    return pK[i] + (pK[i + 1] - pK[i]) * (t - i)


def _replan_racing_line(track: Track, state: dict, full: bool):
    I = _SCRATCH
    n = PLAN_POINTS
    base_idx = track.idx(state["s"])

    for k in range(n):
        a = (base_idx + PLAN_SPACING_IDX * k) % track.M
        I.cx[k] = track.X[a]; I.cy[k] = track.Y[a]
        I.nx[k] = track.PNX[a]; I.ny[k] = track.PNY[a]
        I.e[k] = track.baseline_offset[a]; I.tE[k] = track.baseline_offset[a]
        I.fix[k] = False

    handling = track.handling
    pull2 = handling["pull2"] if handling else 0.0
    for k in range(n):
        if pull2:
            frac = min(1.0, max(0.0, (5 * k - handling["pull_from2"]) / (handling["pull_to2"] - handling["pull_from2"])))
        else:
            frac = 0.0
        I.w[k] = pull2 * frac * frac

    if full:
        base_curv = track.Kc[base_idx]
        track_frame_vy = state["vx"] * fcos(state["dpsi"]) - state["vy"] * fsin(state["dpsi"])
        lateral_rate = state["vx"] * fsin(state["dpsi"]) + state["vy"] * fcos(state["dpsi"])
        slope = max(-0.5, min(0.5, lateral_rate * (1 - base_curv * state["e"]) / max(track_frame_vy, 5.0)))
        r0 = state["e"] - slope * (state["s"] - base_idx * track.DS)
    else:
        r0 = _plan_offset_at(track, state, base_idx * track.DS)
        slope = (_plan_offset_at(track, state, base_idx * track.DS + 5) - r0) / 5.0

    I.e[0] = r0
    I.e[1] = r0 + 5 * slope
    I.fix[0] = I.fix[1] = True
    I.fix[n - 2] = I.fix[n - 1] = True

    for k in range(n):
        a = (base_idx + PLAN_SPACING_IDX * k) % track.M
        half_w = (track.road_width[a] / 2 - CAR["line_margin"]) if track.road_width is not None else \
            (track.width / 2 - CAR["line_margin"])
        crown = track.crown_offset[a] if track.crown_offset is not None else 0.0
        bias = track.apex_bias[a] if track.apex_bias is not None else 0.0
        fade = max(0.0, 1 - (5 * k) / CAR["plan_back"])
        squeeze = (abs(r0 - crown) - half_w) * fade * fade
        I.lo[k] = crown - half_w + (bias if bias > 0 else 0.0) - (max(0.0, squeeze) if r0 < crown else 0.0)
        I.hi[k] = crown + half_w + (bias if bias < 0 else 0.0) + (max(0.0, squeeze) if r0 > crown else 0.0)

    for _ in range(16):
        I.A0[:] = 0.0; I.A1[:] = 0.0; I.A2[:] = 0.0; I.r[:] = 0.0
        for k in range(1, n - 1):
            ex = I.cx[k - 1] - 2 * I.cx[k] + I.cx[k + 1]
            ey = I.cy[k - 1] - 2 * I.cy[k] + I.cy[k + 1]
            for off in range(3):
                j = k - 1 + off
                coeff = -2.0 if off == 1 else 1.0
                if I.fix[j]:
                    ex += coeff * I.nx[j] * I.e[j]
                    ey += coeff * I.ny[j] * I.e[j]
            for off in range(3):
                j = k - 1 + off
                if I.fix[j]:
                    continue
                coeff = -2.0 if off == 1 else 1.0
                nxj = coeff * I.nx[j]
                nyj = coeff * I.ny[j]
                I.r[j] -= nxj * ex + nyj * ey
                for off2 in range(off, 3):
                    jj = k - 1 + off2
                    if I.fix[jj]:
                        continue
                    coeff2 = -2.0 if off2 == 1 else 1.0
                    contrib = nxj * coeff2 * I.nx[jj] + nyj * coeff2 * I.ny[jj]
                    if jj == j:
                        I.A0[j] += contrib
                    elif jj == j + 1:
                        I.A1[j] += contrib
                    else:
                        I.A2[j] += contrib
        if pull2:
            for k in range(n):
                if not I.fix[k] and I.w[k]:
                    I.A0[k] += I.w[k]
                    I.r[k] += I.w[k] * I.tE[k]
        for k in range(n):
            if I.fix[k]:
                I.A0[k] = 1.0; I.A1[k] = 0.0; I.A2[k] = 0.0
                if k >= 1:
                    I.A1[k - 1] = 0.0
                if k >= 2:
                    I.A2[k - 2] = 0.0
                I.r[k] = I.e[k]

        _banded_ldl_solve(I.A0, I.A1, I.A2, I.r, I.x)

        worst_idx, worst_val = -1, 1e-6
        for k in range(n):
            if I.fix[k]:
                continue
            viol = max(I.x[k] - I.hi[k], I.lo[k] - I.x[k])
            if viol > worst_val:
                worst_val, worst_idx = viol, k
        for k in range(n):
            if not I.fix[k]:
                I.e[k] = I.x[k]
        if worst_idx < 0:
            break
        I.e[worst_idx] = min(I.hi[worst_idx], max(I.lo[worst_idx], I.x[worst_idx]))
        I.fix[worst_idx] = True

    for k in range(n):
        I.e[k] = min(I.hi[k], max(I.lo[k], I.e[k]))

    if state["pE"] is None:
        state["pE"] = np.zeros(n)
        state["pK"] = np.zeros(n)
    state["pE"][:] = I.e

    curv = np.zeros(n)
    for k in range(1, n - 1):
        px0 = I.cx[k - 1] + I.nx[k - 1] * I.e[k - 1]; py0 = I.cy[k - 1] + I.ny[k - 1] * I.e[k - 1]
        px1 = I.cx[k] + I.nx[k] * I.e[k]; py1 = I.cy[k] + I.ny[k] * I.e[k]
        px2 = I.cx[k + 1] + I.nx[k + 1] * I.e[k + 1]; py2 = I.cy[k + 1] + I.ny[k + 1] * I.e[k + 1]
        dx = px2 - px0; dy = py2 - py0
        seg = math.hypot(dx, dy) or 1.0
        ddx = px0 - 2 * px1 + px2; ddy = py0 - 2 * py1 + py2
        denom = (seg * seg) / 4
        curv[k] = (dx * ddy - dy * ddx) / seg / denom
    curv[0] = curv[1]
    curv[n - 1] = curv[n - 2]
    for k in range(n):
        lo = max(0, k - 1); hi = min(n - 1, k + 1)
        state["pK"][k] = (curv[lo] + 2 * curv[k] + curv[hi]) / 4
    state["pS"] = base_idx * track.DS


# --------------------------------------------------------------------------
# One 1/120s tick
# --------------------------------------------------------------------------
def step(track: Track, state: dict, held: bool) -> Optional[str]:
    """Advance the car by one DT tick. Returns an event string or None.

    Events: "go" (crossed the line to start the timed lap), "lap"/"best"
    (crossed it again to finish -- state["last"] holds the elapsed lap
    time), "crash" (hit a barrier hard enough to spin/stop), "scrape"
    (grazed a barrier, lost some speed, kept going), "spin" (natural2 mode:
    lost control at low speed and did an in-place recovery turn),
    "sector" (crossed a sector boundary).
    """
    C = CAR
    v2 = track.handling
    state["t"] += DT
    event: Optional[str] = None

    if state["mode"] != 0:
        if state["mode"] == 1:
            state["mt"] += DT
            state["spinV"] *= math.exp(-0.021666666666666667)
            state["s"] += DT * state["spinV"] * 0.6
            if state["mt"] >= C["fly_time"]:
                state["mode"] = 2
                state["mt"] = 0.0
            return None
        if state["mt"] >= C["marshal_time"]:
            state["mode"] = 0
            state["mt"] = 0.0
            state["s"] -= C["backup_m"]
            if state["s"] < 0:
                state["s"] += track.lap_length
            state["e"] = track.samp(track.baseline_offset, state["s"])
            for k in ("dpsi", "vy", "r", "delta", "dv", "aFl", "aRl", "biteT",
                      "unwindT", "bigSlide", "axf", "br", "th", "drs", "clean"):
                state[k] = 0.0 if not isinstance(state[k], int) else 0
            state["pn"] = 0
            state["pFresh"] = True
            max_k = 0.0
            dist = 0.0
            while dist < 80:
                max_k = max(max_k, abs(track.samp(track.RK, state["s"] + dist)))
                dist += 4
            state["vx"] = min(55.0, 0.5 * math.sqrt((C["mu0"] * C["g"]) / max(max_k, 1e-4)))
            return "back"
        state["mt"] += DT
        return None

    natural = bool(v2 and v2["natural2"])
    if natural and state["turnT"]:
        state["vx"] = state["vy"] = state["r"] = state["th"] = state["delta"] = state["dv"] = 0.0
        state["lamF"] = state["lamR"] = 0.0
        step_amt = min(abs(state["dpsi"]), 0.02)
        state["dpsi"] -= (1 if state["dpsi"] > 0 else -1) * step_amt
        if abs(state["dpsi"]) < 0.12:
            state["turnT"] = 0
            state["pFresh"] = True
        return None

    state["pn"] = (state["pn"] or 0) + 1
    if state["pn"] % C["replan_every"] == 1:
        need_full = (state["pE"] is None or state["pFresh"] or
                     abs(state["e"] - _plan_offset_at(track, state, state["s"])) > C["replan_deviation"])
        state["pFresh"] = False
        _replan_racing_line(track, state, need_full)

    if held:
        state["br"] = 0.0
        state["offT"] = 0.0
        state["th"] = min(1.0, state["th"] + DT / C["throttle_ramp"])
    else:
        state["th"] = 0.0
        state["offT"] = state.get("offT", 0.0) + DT
        state["br"] = min(1.0, max(0.0, (state["offT"] - C["brake_delay"]) / C["brake_build"]))

    in_drs_zone = any(a <= state["s"] <= b for a, b in track.drs_zones)
    state["drs"] = 1 if (in_drs_zone and held and (state["drs"] or state["clean"] > 1)) else 0

    for _ in range(N_SUBSTEPS):
        speed = math.hypot(state["vx"], state["vy"])
        ref_speed = max(abs(state["vx"]), 2.0) if natural else max(state["vx"], 5.0)
        apex_curv = track.samp(track.Kc, state["s"])
        lead = fatan(WHEELBASE * abs(track.samp_angle(track.RK, state["s"] + 8))) + 1.5 * _PEAK_SLIP_FRONT
        steer_limit = min(C["steer_max"], lead + 0.02)
        steer_target = _steer_target(track, state, ref_speed)
        steer_target = max(-steer_limit, min(steer_limit, steer_target))

        if state["unwindT"] > 0:
            w = C["unwind_w"]
            acc = w * w * (steer_target - state["delta"]) - 2 * C["unwind_zeta"] * w * state["dv"]
            state["dv"] += acc * SUBDT
            state["delta"] += state["dv"] * SUBDT
        else:
            rate = C["catch_rate"] if state["lamR"] > 1.2 else (v2["steer_rate2"] if v2 else C["steer_rate"])
            d_delta = max(-rate * SUBDT, min(rate * SUBDT, steer_target - state["delta"]))
            state["dv"] = d_delta / SUBDT
            state["delta"] += d_delta

        width = track.road_width_at(state["s"])
        off_frac = track.off_track_frac(state["e"], state["s"]) if v2 else (
            1.0 if abs(state["e"]) > width / 2 + C["kerb"] else 0.0)
        downforce = 0.5 * C["rho"] * (v2["ClA2"] if v2 else C["ClA"]) * speed * speed
        dist_from_centre = abs(state["e"] - track.crown_at(state["s"]))
        on_kerb = (dist_from_centre > width / 2 and off_frac < 0.5 and
                   dist_from_centre <= width / 2 + C["kerb"] + C["car_half"]) if v2 else \
            (abs(state["e"]) > width / 2 and abs(state["e"]) <= width / 2 + C["kerb"])
        fully_off = (off_frac >= 0.5) if v2 else (abs(state["e"]) > width / 2 + C["kerb"])
        cg_h = v2["h2"] if v2 else C["cg_height"]

        grade_load, g_long, g_lat = 1.0, 0.0, 0.0
        if track.grade is not None:
            grade = track.samp(track.grade, state["s"])
            gcurv = track.samp(track.grade_curv, state["s"])
            bank = track.samp(track.bank, state["s"])
            cos_dpsi, sin_dpsi = fcos(state["dpsi"]), fsin(state["dpsi"])
            grade_load = max(0.25, math.sqrt(max(0.0, (1 - grade * grade) * (1 - bank * bank))) +
                              speed * speed * gcurv / C["g"])
            g_long = -C["g"] * (grade * cos_dpsi + bank * sin_dpsi)
            g_lat = C["g"] * (grade * sin_dpsi - bank * cos_dpsi)

        front_load = max(500.0, C["m"] * C["g"] * grade_load * C["wb_rear"] / WHEELBASE +
                          C["aero_front_frac"] * downforce - C["m"] * state["axf"] * cg_h / WHEELBASE)
        rear_load = max(500.0, C["m"] * C["g"] * grade_load * C["wb_front"] / WHEELBASE +
                         (1 - C["aero_front_frac"]) * downforce + C["m"] * state["axf"] * cg_h / WHEELBASE)
        kerb_mult = C["kerb_mu"] if on_kerb else 1.0
        dirt_mult = 1 - C["dirt_loss"] * state["dirt"]
        front_heat_mult = dirt_mult * kerb_mult * (1 - C["heat_loss"] * min(1.5, max(0.0, state["heatF"] - 0.15)))
        rear_heat_mult = dirt_mult * kerb_mult * (1 - C["heat_loss"] * min(1.5, max(0.0, state["heatR"] - 0.15)))

        if v2:
            mu_f = ((v2["mu02"] - v2["kmu2"] * front_load) * (1 - off_frac) + v2["grass_mu"] * off_frac) * front_heat_mult
            mu_r = ((v2["mu02"] - v2["kmu2"] * rear_load) * (1 - off_frac) + v2["grass_mu"] * off_frac) * v2["rear_grip2"] * rear_heat_mult
        else:
            base_mu = C["off_track_mu"] if fully_off else C["mu0"]
            mu_f = (base_mu - C["kmu"] * front_load * (0 if fully_off else 1)) * front_heat_mult
            mu_r = (base_mu - C["kmu"] * rear_load * (0 if fully_off else 1)) * C["rear_grip_mult"] * rear_heat_mult

        traction_ctrl = max(0.3, min(1.0, (1.35 - state["lamR"]) / 0.35))
        if v2:
            traction_ctrl = 1 - v2["assist2"] * (1 - traction_ctrl)
        state["tc"] = traction_ctrl

        edge_push = max(0.0, min(1.0, (dist_from_centre - (width / 2 - 1.5)) / 2))
        understeer_lift = max(0.2, 1 - max(0.0, state["lamF"] - 1.05) * (1.2 + 2.5 * edge_push))
        if v2:
            assist_fade = (speed - v2["push_from2"]) / (v2["push_to2"] - v2["push_from2"])
            assist_fade = min(1.0, max(0.0, assist_fade))
            understeer_lift = 1 - (1 - (1 - v2["push_assist2"]) * assist_fade) * (1 - understeer_lift)
        state["ul"] = understeer_lift

        rear_long_cap = (C["long_cap"] * (1 - off_frac) + v2["grass_power"] * off_frac) if v2 else \
            (0.5 if fully_off else C["long_cap"])

        drive_force = brake_front = brake_rear = 0.0
        if state["th"] > 0:
            power_limit = (v2["power2"] if v2 else C["power"]) / max(speed, 4.0)
            drive_force = state["th"] * traction_ctrl * understeer_lift * min(rear_long_cap * mu_r * rear_load, power_limit)
        else:
            coast_drag = C["coast"] * min(1.0, state.get("offT", 0.0) / C["engine_ramp"]) * (1 - state["br"])
            brake_cap = state["br"] * (v2["brake_max2"] if v2 else C["brake_max"])
            brake_front = -min((v2["brake_cap2"] if v2 else C["long_cap"]) * mu_f * front_load,
                                brake_cap * C["brake_front_frac"])
            brake_rear = -min((v2["rear_brake_cap2"] if v2 else C["rear_brake_cap"]) * mu_r * rear_load,
                               coast_drag + brake_cap * (1 - C["brake_front_frac"])) * \
                max(0.2, min(1.0, (1.3 - state["lamR"]) / 0.4))
            if natural and state["vx"] < 0:
                brake_front, brake_rear = -brake_front, -brake_rear
            drive_force = brake_rear

        front_long = brake_front
        rear_force = drive_force

        if speed < 0.5 and not state["th"]:
            front_long = 0.0
            rear_force = 0.0
            state["vx"] = state["vy"] = state["r"] = 0.0

        front_lateral = rear_lateral = 0.0
        moving_fast = (speed > 4) if natural else (state["vx"] > 8)
        if moving_fast:
            state["alphaF"] = (0.0 if (natural and state["vx"] < 0) else state["delta"]) - \
                fatan((state["vy"] + C["wb_front"] * state["r"]) / ref_speed)
            state["alphaR"] = -fatan((state["vy"] - C["wb_rear"] * state["r"]) / ref_speed)
            relax = min(1.0, (ref_speed / C["slip_relax"]) * SUBDT)
            state["aFl"] += (state["alphaF"] - state["aFl"]) * relax
            state["aRl"] += (state["alphaR"] - state["aRl"]) * relax
            bite_mult = 1 + C["bite"] * min(1.0, state["biteT"] / C["bite_fade"]) if state["biteT"] > 0 else 1.0

            front_lateral = mu_f * front_load * _saturate(state["aFl"], C["Bf"], C["Cf"], C["Ef"], _PEAK_SLIP_FRONT, C["grip_floor_front"])
            rear_lateral = mu_r * bite_mult * rear_load * _saturate(state["aRl"], C["Br"], C["Cr"], C["Er"], _PEAK_SLIP_REAR, C["grip_floor_rear"])

            front_circle = mu_f * front_load
            rear_circle = mu_r * rear_load
            front_long_room = math.sqrt(max(0.0, front_circle ** 2 - front_long ** 2))
            rear_long_room = math.sqrt(max(0.0, rear_circle ** 2 - rear_force ** 2))
            front_lateral = max(-front_long_room, min(front_long_room, front_lateral))
            rear_lateral = max(-rear_long_room, min(rear_long_room, rear_lateral))

            state["lamF"] = abs(state["aFl"]) / _PEAK_SLIP_FRONT
            state["lamR"] = abs(state["aRl"]) / _PEAK_SLIP_REAR
            state["heatF"] += (C["heat_gain"] * max(0.0, state["lamF"] - 0.9) * speed - C["heat_cool"] * state["heatF"]) * SUBDT
            state["heatR"] += (C["heat_gain"] * max(0.0, state["lamR"] - 0.9) * speed - C["heat_cool"] * state["heatR"]) * SUBDT
        else:
            state["alphaF"] = state["alphaR"] = 0.0
            state["lamF"] = state["lamR"] = 0.0

        drag = (0.5 * C["rho"] * (v2["CdA2"] if v2 else C["CdA"]) * (C["drs_cd_mult"] if state["drs"] else 1.0) * speed * speed)
        if v2:
            drag += off_frac * min(v2["grass_drag_max"], v2["grass_drag"] * (speed / 40) ** 2)
        elif fully_off:
            drag += C["off_track_drag"] * min(1.0, speed / 40)

        sin_d, cos_d = fsin(state["delta"]), fcos(state["delta"])
        long_accel = (rear_force + front_long * cos_d - front_lateral * sin_d -
                      (drag * state["vx"] / max(speed, 1.0) if natural else drag * math.copysign(1.0, state["vx"] or 1.0))) / C["m"] + g_long

        if moving_fast:
            lat_accel = (front_lateral * cos_d + rear_lateral - (drag * state["vy"] / max(speed, 1.0) if natural else 0.0)) / C["m"] + g_lat
            grass_tug = 0.0
            if v2 and 0 < off_frac < 1:
                grass_tug = 4 * off_frac * (1 - off_frac) * math.copysign(1.0, (state["e"] - track.crown_at(state["s"])) or 1.0) * \
                    v2["grass_tug"] * C["car_half"] * min(1.0, speed / 30)
            yaw_accel = (C["wb_front"] * front_lateral * cos_d - C["wb_rear"] * rear_lateral + grass_tug) / C["Iz"]
            state["vx"] += (long_accel + state["r"] * state["vy"]) * SUBDT
            state["vy"] += (lat_accel - state["r"] * state["vx"]) * SUBDT
            state["r"] += yaw_accel * SUBDT
            state["ay"] = lat_accel
        else:
            state["vx"] = max(0.0, state["vx"] + long_accel * SUBDT)
            state["vy"] *= 0.8
            tan_d = sin_d / cos_d if cos_d != 0 else 0.0
            state["r"] = state["vx"] * tan_d / WHEELBASE
            state["ay"] = state["vx"] * state["r"]

        state["ax"] = long_accel
        state["axf"] += (long_accel - state["axf"]) * (SUBDT / C["ax_tau"])

        line_shrink = 1 - apex_curv * state["e"]
        unnormalized_forward_rate = (state["vx"] * fcos(state["dpsi"]) - state["vy"] * fsin(state["dpsi"]))
        forward_rate = unnormalized_forward_rate / (line_shrink if line_shrink > 0.2 else 0.2)
        state["e"] += (state["vx"] * fsin(state["dpsi"]) + state["vy"] * fcos(state["dpsi"])) * SUBDT
        state["dpsi"] += (state["r"] - apex_curv * forward_rate) * SUBDT
        state["s"] += forward_rate * SUBDT

    if state["biteT"] > 0:
        state["biteT"] -= DT
    if state["unwindT"] > 0:
        state["unwindT"] -= DT
    if state["lamR"] > 1.5:
        state["slideT"] = state["t"]
        state["bigSlide"] = max(state["bigSlide"], state["lamR"])
    if state["lamR"] < 1 and state["t"] - state["slideT"] < 0.25 and state["bigSlide"] > 0:
        state["biteT"] = C["bite_time"] + C["bite_fade"]
        state["unwindT"] = C["unwind_time"]
        state["saves"] += 1
        state["bigSlide"] = 0.0
        event = "save"
    if state["t"] - state["slideT"] >= 0.25:
        state["bigSlide"] = 0.0

    width = track.road_width_at(state["s"])
    sideslip = fatan2(state["vy"], max(state["vx"], 1.0))
    state["clean"] = 0.0 if (state["lamR"] > 1 or state["lamF"] > 1.1) else state["clean"] + DT
    state["scrape"] = 0

    crown = track.crown_at(state["s"])
    side_sign = 1 if state["e"] - crown >= 0 else -1
    if track.wall_l is not None and track.wall_r is not None:
        wall_dist = track.samp(track.wall_l if side_sign > 0 else track.wall_r, state["s"])
    else:
        wall_dist = max(C["kerb"], track.wall_clearance)
    clearance = width / 2 + wall_dist - (C["car_half"] + 0.25 if v2 else 0.0)

    if abs(state["e"] - crown) > clearance:
        signed_dpsi = state["dpsi"] * side_sign
        speed = math.hypot(state["vx"], state["vy"])
        hard_hit = (speed > 3) if v2 else (signed_dpsi > C["scrape_angle"] and speed > C["crash_speed"])
        if hard_hit:
            state["mode"] = 1
            state["mt"] = 0.0
            state["crashes"] += 1
            state["spinV"] = speed
            state["wall"] = 1
            event = "crash"
        else:
            state["e"] = crown + side_sign * clearance
            if signed_dpsi > 0:
                state["dpsi"] -= side_sign * min(signed_dpsi, 0.05)
            state["vy"] *= 0.5
            state["r"] *= 0.6
            state["vx"] *= 0.985
            state["scrape"] = 1
            event = event or "scrape"

    is_off = track.off_track_frac(state["e"], state["s"]) >= 0.5 if v2 else abs(state["e"]) > width / 2 + C["kerb"]
    state["dirt"] = 1.0 if is_off else max(0.0, state["dirt"] - DT * C["dirt_clean"])

    if natural and not state["mode"]:
        if abs(state["dpsi"]) > 100:
            state["dpsi"] = state["dpsi"] % (2 * math.pi)
        while state["dpsi"] > math.pi:
            state["dpsi"] -= 2 * math.pi
        while state["dpsi"] < -math.pi:
            state["dpsi"] += 2 * math.pi
        if math.hypot(state["vx"], state["vy"]) < 3 and abs(state["dpsi"]) > 1.3:
            state["turnT"] = 1
            state["spins"] += 1
            event = event or "spin"

    if not natural and state["mode"] == 0:
        speed = math.hypot(state["vx"], state["vy"])
        spin_threshold = (C["off_spin_beta"] if is_off else C["spin_beta"])
        if (abs(sideslip) > spin_threshold and abs(state["r"]) > 0.5) or \
           (abs(state["dpsi"]) > (1.6 if is_off else 1.2) and speed > 20):
            state["mode"] = 1
            state["mt"] = 0.0
            state["crashes"] += 1
            state["spinV"] = speed
            state["wall"] = 0
            event = "crash"

    while state["s"] >= track.lap_length:
        state["s"] -= track.lap_length
        if not state["started"] or state["t"] - state["lapStart"] < 30:
            state["started"] = True
            state["lapStart"] = state["t"]
            state["secStart"] = state["t"]
            state["sector"] = 0
            state["secs"] = [0.0, 0.0, 0.0]
            event = "go"
        else:
            state["laps"] += 1
            state["secs"][2] = state["t"] - state["secStart"]
            state["secStart"] = state["t"]
            state["sector"] = 0
            lap_time = state["t"] - state["lapStart"]
            state["last"] = lap_time
            state["lapStart"] = state["t"]
            is_best = state["best"] is None or lap_time < state["best"]
            if is_best:
                state["best"] = lap_time
            event = "best" if is_best else "lap"

    if state["s"] < 0:
        state["s"] += track.lap_length
    if state["started"]:
        sector = min(2, int(state["s"] / (track.lap_length / 3)))
        if sector > state["sector"]:
            state["secs"][state["sector"]] = state["t"] - state["secStart"]
            state["secStart"] = state["t"]
            state["sector"] = sector
            event = event or "sector"

    return event


def _steer_target(track: Track, state: dict, ref_speed: float) -> float:
    C = CAR
    v2 = track.handling
    lookahead = max(10.0, min(v2["look_max2"] if v2 else 30.0, 0.32 * ref_speed))
    heading_here = track.samp_angle(track.H, state["s"])
    car_x = track.samp(track.X, state["s"]) - fsin(heading_here) * state["e"]
    car_y = track.samp(track.Y, state["s"]) + fcos(heading_here) * state["e"]
    car_heading = heading_here + state["dpsi"]

    half_width = track.road_width_at(state["s"]) / 2 - C["car_half"] - 0.15
    deviation = state["e"] - _plan_offset_at(track, state, state["s"])
    safety = max(0.0, min(1.0, (half_width - abs(state["e"] - track.crown_at(state["s"]))) / C["edge_guard"]))

    float_gain = v2["float2"] if v2 else C["float_gain"]
    float_max = v2["float_max2"] if v2 else C["float_max"]
    blend = float_gain * safety * min(1.0, 45 / max(ref_speed, 1.0))

    look_s = state["s"] + lookahead
    look_heading = track.samp_angle(track.H, look_s)
    target_offset = _plan_offset_at(track, state, look_s) + blend * max(-float_max, min(float_max, deviation))
    target_x = track.samp(track.X, look_s) - fsin(look_heading) * target_offset
    target_y = track.samp(track.Y, look_s) + fcos(look_heading) * target_offset

    bearing_err = _wrap_pi(fatan2(target_y - car_y, target_x - car_x) - car_heading)
    pure_pursuit = fatan((2 * WHEELBASE * fsin(bearing_err)) / lookahead)

    cross_track = _plan_offset_at(track, state, state["s"]) - state["e"]
    curvature_ahead = _plan_curvature_at(track, state, state["s"] + 0.3 * lookahead)
    beta = fatan2(state["vy"], max(state["vx"], 1.0))
    oversteer_comp = C["kbeta"] * beta * max(0.0, min(1.0, (state["lamR"] - 0.95) / 0.3))

    lateral_rate = state["vx"] * fsin(state["dpsi"]) + state["vy"] * fcos(state["dpsi"])
    pd_term = ((1 - 0.6 * safety * min(1.0, 45 / max(ref_speed, 1.0))) * WHEELBASE *
               (2 * cross_track - 0.6 * lateral_rate * lookahead / max(ref_speed, 5.0)) / (lookahead ** 2))
    feedforward = C["kff"] * WHEELBASE * curvature_ahead
    soft_correction = (0.25 + 0.35 * (1 - safety)) * fatan(C["ks"] * cross_track / (ref_speed + 2))
    yaw_damping = C["kr"] * (ref_speed * _plan_curvature_at(track, state, state["s"]) - state["r"])

    return pure_pursuit + pd_term + feedforward + soft_correction + yaw_damping + oversteer_comp


# --------------------------------------------------------------------------
# simulate(): the public API optimize.py / tests use
# --------------------------------------------------------------------------
@dataclass
class Result:
    time: float
    finished: bool
    dist: float
    events: list = field(default_factory=list)
    crashed: bool = False


def simulate(track: Track, switches_m, record: bool = False) -> Result:
    """Run one attempt. `switches_m` are sorted cumulative-distance switch
    points in metres from the start of the simulation (100 m before the
    line) -- see the module docstring. Hold from 0 to switches[0], release
    to switches[1], and so on, starting held.
    """
    state = init_state(track)
    held = True
    idx = 0
    n = len(switches_m)
    cum_dist = 0.0
    events = []
    last_held = held
    max_ticks = int(MAX_LAP_TIME / DT)

    for tick in range(max_ticks):
        while idx < n and cum_dist >= switches_m[idx]:
            held = not held
            idx += 1
        if record and held != last_held:
            events.append((tick, held))
            last_held = held

        s_before = state["s"]
        ev = step(track, state, held)
        s_after = state["s"]
        delta = s_after - s_before
        if delta < -track.lap_length / 2:
            delta += track.lap_length
        cum_dist += delta

        if ev == "crash":
            return Result(time=state["t"], finished=False, dist=min(cum_dist, track.length),
                          events=events, crashed=True)
        if ev in ("best", "lap"):
            return Result(time=state["last"], finished=True, dist=track.length, events=events)

    return Result(time=state["t"], finished=False, dist=min(cum_dist, track.length), events=events)
