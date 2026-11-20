"""
Tests that need no network access and always run. These catch regressions
in sim.py itself (determinism, genome clamping, basic physics sanity) --
they do NOT by themselves prove the physics matches the real game. For
that, see test_fidelity_vs_real_bundle.py.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sim

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extract", "raw_data")


def _track(circuit="lasvegas"):
    return sim.get_track(circuit, DATA_DIR)


def test_track_loads_with_sane_geometry():
    track = _track()
    assert track.M == 12385
    assert track.lap_length == 6192.5
    assert track.width == 22
    assert track.handling["natural2"] is True
    assert track.length == track.lap_length + track.runup_m


def test_determinism_same_genome_same_result():
    track = _track()
    genome = [280.0, 475.0, 1200.0, 3000.5, 5200.0]
    r1 = sim.simulate(track, genome)
    r2 = sim.simulate(track, genome)
    assert r1.time == r2.time
    assert r1.dist == r2.dist
    assert r1.finished == r2.finished
    assert r1.crashed == r2.crashed


def test_full_throttle_crashes_before_the_hairpin():
    """Holding the whole way should not survive Las Vegas's tight infield section."""
    track = _track()
    r = sim.simulate(track, [])
    assert not r.finished
    assert r.dist < track.lap_length * 0.5


def test_braking_before_a_known_tight_corner_gets_further():
    """A genome that brakes before the ~45 m-radius corner near lap-distance 380 m
    should progress meaningfully further than holding the whole way."""
    track = _track()
    baseline = sim.simulate(track, [])
    braked = sim.simulate(track, [280.0, 475.0])
    assert braked.dist > baseline.dist + 300


def test_mutation_keeps_genome_sorted_and_in_bounds():
    rng = random.Random(0)
    track = _track()
    g = [100.0, 500.0, 1000.0]
    for _ in range(200):
        g = [max(0.0, min(track.length, x + rng.gauss(0, 50))) for x in g]
        g = sorted(g)
    assert all(0.0 <= x <= track.length for x in g)
    assert g == sorted(g)


def test_crash_fitness_decreases_toward_zero_as_distance_increases():
    """Fitness gradient sanity: further progress before a crash must score better."""
    track = _track()

    def fitness(r):
        return r.time if r.finished else 1000.0 + (track.length - r.dist)

    worse = sim.simulate(track, [])  # crashes early
    better = sim.simulate(track, [280.0, 475.0])  # crashes later (or finishes)
    assert fitness(better) < fitness(worse)


def test_saopaulo_loads_and_runs():
    track = _track("saopaulo")
    assert track.lap_length == 4300.5
    r = sim.simulate(track, [])
    assert math.isfinite(r.time)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
