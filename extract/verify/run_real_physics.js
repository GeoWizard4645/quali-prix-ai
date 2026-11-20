// Runs the game's own (unmodified) physics module against a genome and
// prints a per-tick trace as JSON, for comparison against sim.py.
// Usage: node run_real_physics.js <chunk.js> <meta.json> <circuit.bin> <terrain.json|-> <switches_csv> <max_ticks>
const fs = require("fs");
const { loadPhysicsExports } = require("./shim.js");

const [chunkPath, metaPath, binPath, terrainPath, switchesCsv, maxTicksArg] = process.argv.slice(2);
const api = loadPhysicsExports(chunkPath);

const meta = JSON.parse(fs.readFileSync(metaPath, "utf8"));
const bin = fs.readFileSync(binPath);
const terrain = terrainPath && terrainPath !== "-" ? JSON.parse(fs.readFileSync(terrainPath, "utf8")) : undefined;
const switches = switchesCsv ? switchesCsv.split(",").filter((s) => s.length).map(Number) : [];
const maxTicks = maxTicksArg ? parseInt(maxTicksArg, 10) : 20000;

const track = api.loadCircuit(meta, bin, terrain);
const state = api.init(track);
const runupM = track.v2 ? track.v2.runup2 : 260;

let held = true, idx = 0, cumDist = 0.0;
const trace = [];
for (let tick = 0; tick < maxTicks; tick++) {
  while (idx < switches.length && cumDist >= switches[idx]) {
    held = !held;
    idx++;
  }
  const sBefore = state.s;
  const ev = api.step(track, state, held);
  const sAfter = state.s;
  let delta = sAfter - sBefore;
  if (delta < -track.L / 2) delta += track.L;
  cumDist += delta;

  trace.push([tick, state.s, state.e, state.dpsi, state.vx, state.vy, state.r]);

  if (ev === "crash") {
    console.log(JSON.stringify({ outcome: "crash", t: state.t, dist: Math.min(cumDist, track.L + runupM), trace }));
    process.exit(0);
  }
  if (ev === "best" || ev === "lap") {
    console.log(JSON.stringify({ outcome: "finished", t: state.last, dist: track.L + runupM, trace }));
    process.exit(0);
  }
}
console.log(JSON.stringify({ outcome: "timeout", t: state.t, dist: Math.min(cumDist, track.L + runupM), trace }));
