"""
The real fidelity check: runs the game's OWN, unmodified physics module
(fetched fresh from the live site, same as any visitor's browser would)
under Node, and the sim.py port, on the same genomes, then diffs the
per-tick trajectories.

This does not play the live game -- it fetches static JS the page already
serves to every visitor (see extract/verify/fetch_physics_chunk.py), runs
it standalone, and compares numbers locally. No gameplay, no screenshots,
no leaderboard/submission traffic.

Skips (does not fail) if `node` isn't on PATH or the network is
unavailable, so the rest of the test suite still runs offline. If you see
it skip and want the real fidelity numbers, run it with network access.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFY_DIR = os.path.join(ROOT, "extract", "verify")
DATA_DIR = os.path.join(ROOT, "extract", "raw_data")

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not found on PATH")


def _get_chunk_path():
    sys.path.insert(0, VERIFY_DIR)
    from fetch_physics_chunk import get_physics_chunk_path
    try:
        return get_physics_chunk_path()
    except Exception as e:  # network unavailable, site layout changed, etc.
        pytest.skip(f"could not fetch the live physics chunk: {e}")


def _run_case(chunk_path, circuit, switches, max_ticks=10000):
    sw_csv = ",".join(str(x) for x in switches)
    meta = os.path.join(DATA_DIR, f"{circuit}.json")
    binp = os.path.join(DATA_DIR, f"{circuit}.bin")
    terr = os.path.join(DATA_DIR, f"{circuit}-terrain.json")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as jf, \
         tempfile.NamedTemporaryFile(suffix=".json", delete=False) as pf:
        js_path, py_path = jf.name, pf.name

    try:
        js = subprocess.run(
            ["node", os.path.join(VERIFY_DIR, "run_real_physics.js"), chunk_path, meta, binp, terr, sw_csv, str(max_ticks)],
            stdout=open(js_path, "w"), stderr=subprocess.PIPE, text=True, timeout=120,
        )
        py = subprocess.run(
            [sys.executable, os.path.join(VERIFY_DIR, "trace_py.py"), circuit, DATA_DIR, sw_csv, str(max_ticks)],
            stdout=open(py_path, "w"), stderr=subprocess.PIPE, text=True, timeout=120,
        )
        assert js.returncode == 0, f"node harness failed: {js.stderr}"
        assert py.returncode == 0, f"python harness failed: {py.stderr}"

        js_result = json.load(open(js_path))
        py_result = json.load(open(py_path))
    finally:
        os.remove(js_path)
        os.remove(py_path)

    js_trace, py_trace = js_result["trace"], py_result["trace"]
    n = min(len(js_trace), len(py_trace))
    max_abs_s = max(abs(js_trace[i][1] - py_trace[i][1]) for i in range(n)) if n else 0.0
    return js_result["outcome"], py_result["outcome"], js_result["t"], py_result["t"], max_abs_s, len(js_trace), len(py_trace)


# A fixed, varied set of genomes (not adversarially chosen) across both
# shipped circuits: empty, a few random switch lists, and a hand-picked
# one that gets well into the lap via real braking.
LASVEGAS_GENOMES = [
    [],
    [280.0, 475.0],
    [303.8, 592.3, 2484.4, 5167.9, 5964.4],
    [372.0, 540.8, 1351.0, 1514.4, 2631.3, 3467.5, 3558.1, 5961.8],
]
SAOPAULO_GENOMES = [
    [],
    [400.0, 900.0, 1600.0, 2200.0],
]


@pytest.mark.parametrize("genome", LASVEGAS_GENOMES)
def test_lasvegas_matches_real_physics(genome):
    chunk = _get_chunk_path()
    js_o, py_o, js_t, py_t, max_abs_s, len_js, len_py = _run_case(chunk, "lasvegas", genome)
    assert js_o == py_o, f"outcome mismatch: real={js_o} ours={py_o}"
    assert abs(len_js - len_py) <= 5, f"tick-count drift too large: {len_js} vs {len_py}"
    # Position agreement: sub-metre even on long/marginal runs (see README
    # for the one hand-picked near-miss case where this can reach ~7cm).
    assert max_abs_s < 1.0, f"position diverged by {max_abs_s:.4f} m"


@pytest.mark.parametrize("genome", SAOPAULO_GENOMES)
def test_saopaulo_matches_real_physics(genome):
    chunk = _get_chunk_path()
    js_o, py_o, js_t, py_t, max_abs_s, len_js, len_py = _run_case(chunk, "saopaulo", genome)
    assert js_o == py_o, f"outcome mismatch: real={js_o} ours={py_o}"
    assert abs(len_js - len_py) <= 5
    assert max_abs_s < 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
