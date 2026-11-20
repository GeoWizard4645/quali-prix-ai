"""
Finds and downloads the game's current physics JS chunk, for the fidelity
cross-check in tests/test_fidelity_vs_real_bundle.py.

This is the same kind of request an ordinary visitor's browser makes
(fetch the page, fetch the JS chunks it references) -- no gameplay, no
leaderboard/submission endpoints, nothing beyond what loading the page
already does. The chunk's filename hash changes whenever the game is
redeployed, so rather than hardcoding a URL we fetch the page and search
its chunks for one exporting a name ("HANDLING_V2") that's distinctive to
the physics module.

We do NOT ship the downloaded file in the repo (it's the game author's own
code, not something we should redistribute) -- it's cached in a gitignored
directory purely for local, repeatable test runs.
"""
import os
import re
import urllib.request

BASE = "https://gluefold.co.uk"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
MARKER = b"HANDLING_V2"
UA = "Mozilla/5.0 (compatible; quali-prix-ai fidelity test; local use only)"


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


def get_physics_chunk_path() -> str:
    """Returns a local path to the current physics chunk, downloading/caching it if needed."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cached = os.path.join(CACHE_DIR, "physics_chunk.js")
    if os.path.exists(cached):
        with open(cached, "rb") as f:
            if MARKER in f.read():
                return cached

    html = _fetch(f"{BASE}/quali-prix").decode("utf-8", "replace")
    chunk_paths = sorted(set(re.findall(r'src="(/_next/static/immutable/chunks/[^"]+\.js)"', html)))
    if not chunk_paths:
        raise RuntimeError("could not find any JS chunks on the page -- site layout may have changed")

    for path in chunk_paths:
        data = _fetch(BASE + path)
        if MARKER in data:
            with open(cached, "wb") as f:
                f.write(data)
            return cached

    raise RuntimeError(
        "none of the page's JS chunks export HANDLING_V2 -- the game may have "
        "been redeployed with a different module layout; see extract/NOTES.md"
    )


if __name__ == "__main__":
    print(get_physics_chunk_path())
