"""
A local, live dashboard for watching optimize.py / policy/train.py learn --
no cloud, no new dependencies (stdlib http.server only). Shows the
learning curve, the current best lap animated on the track (coloured by
speed), and a live view of the policy network's weights when
policy/train.py is also running.

Usage:
    python3 dashboard.py --circuit lasvegas
    # then open the printed URL yourself, and/or share the LAN URL with
    # anyone on the same Wi-Fi/network to watch along -- this does NOT
    # expose anything to the internet, only your local network.

It only reads runs/*.json / *.csv (written by optimize.py / policy/train.py)
and the circuit files in extract/raw_data/ -- it doesn't touch the
optimizer process at all, so you can start/stop/restart either
independently.
"""
import argparse
import csv
import json
import os
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "extract", "raw_data")
RUNS_DIR = os.path.join(ROOT, "runs")

_track_cache = {}


def available_circuits():
    circuits = []
    if os.path.isdir(DATA_DIR):
        for name in os.listdir(DATA_DIR):
            if name.endswith(".json") and not name.endswith("-terrain.json"):
                circuits.append(name[:-5])
    return sorted(circuits)


def get_track_summary(circuit):
    if circuit not in _track_cache:
        track = sim.get_track(circuit, DATA_DIR)
        stride = max(1, track.M // 1500)
        _track_cache[circuit] = {
            "x": track.X[::stride].tolist(),
            "y": track.Y[::stride].tolist(),
            "width": track.width,
            "lap_length": track.lap_length,
            "sim_length": track.length,
            "runup_m": track.runup_m,
        }
    return _track_cache[circuit]


def read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None  # being written concurrently by the optimizer; try again next poll


def read_history_tail(path, max_rows=4000):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                try:
                    rows.append([int(row[0]), float(row[1]), float(row[2])])
                except ValueError:
                    continue
    return rows[-max_rows:]


def build_state(circuit):
    live = read_json(os.path.join(RUNS_DIR, f"{circuit}_live.json"))
    history = read_history_tail(os.path.join(RUNS_DIR, f"{circuit}_history.csv"))
    best = read_json(os.path.join(RUNS_DIR, f"{circuit}_best.json"))
    return {"live": live, "history": history, "best_meta": best}


def build_policy_state(circuit):
    live = read_json(os.path.join(RUNS_DIR, f"policy_{circuit}_live.json"))
    history = read_history_tail(os.path.join(RUNS_DIR, f"policy_{circuit}_history.csv"))
    return {"live": live, "history": history}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep the console quiet

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        circuit = qs.get("circuit", ["lasvegas"])[0]

        if parsed.path == "/":
            body = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/circuits":
            self._json({"circuits": available_circuits()})
        elif parsed.path == "/api/track":
            try:
                self._json(get_track_summary(circuit))
            except Exception as e:
                self._json({"error": str(e)}, status=404)
        elif parsed.path == "/api/state":
            self._json(build_state(circuit))
        elif parsed.path == "/api/policy":
            self._json(build_policy_state(circuit))
        else:
            self.send_response(404)
            self.end_headers()


def _lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Quali Prix AI</title>
<style>
  :root {
    --bg: #0b0f14; --panel: #121821; --panel2: #161e2a; --line: #233040;
    --text: #e7edf3; --muted: #8aa0b4; --accent: #37e2c4; --accent2: #ff8a3d;
    --red: #ff5a6e; --green: #46e07a; --gold: #ffd451; --silver: #d7dee6; --bronze: #e0a05b;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; }
  header { display:flex; align-items:center; gap:16px; padding:14px 20px; border-bottom:1px solid var(--line); background:var(--panel); flex-wrap:wrap; }
  header h1 { font-size:18px; margin:0; letter-spacing:0.5px; }
  header h1 span { color: var(--accent); }
  select, button { background:var(--panel2); color:var(--text); border:1px solid var(--line); border-radius:6px; padding:6px 10px; font-size:13px; }
  .stat { display:flex; flex-direction:column; min-width:88px; }
  .stat .v { font-size:18px; font-weight:600; font-variant-numeric: tabular-nums; }
  .stat .l { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; }
  .stats-row { display:flex; gap:22px; margin-left:auto; flex-wrap:wrap; }
  main { display:grid; grid-template-columns: 1.3fr 1fr; gap:16px; padding:16px; grid-template-rows: auto auto; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
  .panel h2 { font-size:13px; text-transform:uppercase; letter-spacing:0.8px; color:var(--muted); margin:0 0 10px 0; }
  canvas { width:100%; display:block; }
  #trackCanvas { height:460px; }
  #curveCanvas { height:220px; }
  #nnCanvas { height:220px; }
  .medal-row { display:flex; gap:10px; margin-top:8px; font-size:12px; flex-wrap:wrap; }
  .medal { padding:3px 9px; border-radius:12px; background:var(--panel2); }
  .medal.hit { font-weight:700; }
  .gold { color:var(--gold); } .silver { color:var(--silver); } .bronze { color:var(--bronze); }
  #vivOverlay { position:fixed; inset:0; background:rgba(5,8,12,0.88); display:none; align-items:center; justify-content:center; z-index:50; }
  #vivCard { background:linear-gradient(160deg, var(--panel2), var(--panel)); border:1px solid var(--accent); border-radius:16px; padding:34px 44px; text-align:center; max-width:420px; box-shadow:0 0 60px rgba(55,226,196,0.25); }
  #vivCard h2 { font-size:28px; margin:6px 0; color:var(--accent); }
  #vivCard p { color:var(--muted); font-size:14px; }
  #vivCard button { margin-top:14px; background:var(--accent); color:#06231d; font-weight:700; border:none; padding:8px 18px; border-radius:8px; cursor:pointer; }
  .badge { font-size:11px; padding:2px 8px; border-radius:10px; background:var(--panel2); color:var(--muted); }
  .phase2note { font-size:12px; color:var(--muted); margin-top:8px; }
  footer { padding:10px 20px; color:var(--muted); font-size:12px; }
  a { color: var(--accent); }
</style>
</head>
<body>

<header>
  <h1>Quali&nbsp;Prix <span>AI</span></h1>
  <select id="circuitSelect"></select>
  <div class="stats-row" id="statsRow"></div>
</header>

<main>
  <div class="panel">
    <h2 id="trackTitle">Best lap so far</h2>
    <canvas id="trackCanvas"></canvas>
    <div class="medal-row" id="medalRow"></div>
  </div>

  <div class="panel">
    <h2>Learning curve</h2>
    <canvas id="curveCanvas"></canvas>
    <div class="panel2note" style="font-size:12px;color:var(--muted);margin-top:6px;">
      y-axis: lap time once a clean lap is found; before that, a crash-distance score (lower is better, &gt;1000 = hasn't finished a lap yet).
    </div>
  </div>

  <div class="panel" style="grid-column: 1 / -1;">
    <h2>Policy network (phase 2) <span class="badge" id="policyBadge">not running</span></h2>
    <canvas id="nnCanvas"></canvas>
    <div class="phase2note">Nodes: speed (x/y), upcoming track curvature at six lookahead distances, elevation grade (now / 40m ahead), previous action, bias &rarr; 8 hidden &rarr; hold/release. Green edges push toward "hold", red toward "release"; thicker = stronger. This only populates once <code>policy/train.py</code> is running.</div>
  </div>
</main>

<footer>Served locally from your laptop -- <span id="urls"></span>. Nothing here touches the internet.</footer>

<div id="vivOverlay">
  <div id="vivCard">
    <div style="font-size:40px;">🏎️</div>
    <h2>Say hello to VIV.</h2>
    <p id="vivText">VIV just completed its first clean lap.</p>
    <button onclick="document.getElementById('vivOverlay').style.display='none'">Nice</button>
  </div>
</div>

<script>
const statsRow = document.getElementById('statsRow');
const medalRow = document.getElementById('medalRow');
const trackCanvas = document.getElementById('trackCanvas');
const curveCanvas = document.getElementById('curveCanvas');
const nnCanvas = document.getElementById('nnCanvas');
const circuitSelect = document.getElementById('circuitSelect');

let currentCircuit = 'lasvegas';
let trackSummary = null;
let lastTrajectory = null;
let vivRevealed = {};

function fmtTime(t) {
  if (t == null) return '--';
  const m = Math.floor(t / 60);
  return (m > 0 ? m + ':' : '') + (t - m*60).toFixed(3).padStart(m>0?6:5,'0');
}

function stat(label, value) {
  const d = document.createElement('div');
  d.className = 'stat';
  d.innerHTML = `<div class="v">${value}</div><div class="l">${label}</div>`;
  return d;
}

function vivName(circuit) {
  return vivRevealed[circuit] ? 'VIV' : 'the car';
}

async function loadCircuits() {
  const r = await fetch('/api/circuits').then(r => r.json());
  circuitSelect.innerHTML = '';
  for (const c of r.circuits) {
    const opt = document.createElement('option');
    opt.value = c; opt.textContent = c;
    circuitSelect.appendChild(opt);
  }
  if (r.circuits.length) currentCircuit = r.circuits[0];
  circuitSelect.value = currentCircuit;
}
circuitSelect.addEventListener('change', () => {
  currentCircuit = circuitSelect.value;
  trackSummary = null;
  lastTrajectory = null;
  curveHistory = [];
  drawTrack(null);
  drawCurve([]);
});

async function ensureTrack() {
  if (trackSummary && trackSummary._circuit === currentCircuit) return trackSummary;
  const r = await fetch('/api/track?circuit=' + currentCircuit).then(r => r.json());
  r._circuit = currentCircuit;
  trackSummary = r;
  return r;
}

function speedColor(v, vmin, vmax) {
  const t = Math.max(0, Math.min(1, (v - vmin) / Math.max(1, (vmax - vmin))));
  // viridis-ish: dark purple -> teal -> yellow
  const stops = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  const idx = t * (stops.length - 1);
  const i0 = Math.floor(idx), i1 = Math.min(stops.length-1, i0+1), f = idx - i0;
  const c0 = stops[i0], c1 = stops[i1];
  const r = Math.round(c0[0]+(c1[0]-c0[0])*f), g = Math.round(c0[1]+(c1[1]-c0[1])*f), b = Math.round(c0[2]+(c1[2]-c0[2])*f);
  return `rgb(${r},${g},${b})`;
}

function fitTransform(xs, ys, w, h, pad) {
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const scale = Math.min((w - 2*pad) / (maxX - minX || 1), (h - 2*pad) / (maxY - minY || 1));
  return (x, y) => [pad + (x - minX) * scale, h - pad - (y - minY) * scale];
}

let carAnimStart = null;

function drawTrack(traj) {
  const ctx = trackCanvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = trackCanvas.clientWidth, h = trackCanvas.clientHeight;
  trackCanvas.width = w * dpr; trackCanvas.height = h * dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#0b0f14';
  ctx.fillRect(0,0,w,h);
  if (!trackSummary) return;
  const tf = fitTransform(trackSummary.x, trackSummary.y, w, h, 30);

  ctx.strokeStyle = '#2a3a4c';
  ctx.lineWidth = 10;
  ctx.beginPath();
  trackSummary.x.forEach((x, i) => {
    const [px, py] = tf(x, trackSummary.y[i]);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.stroke();

  if (traj && traj.x && traj.x.length > 1) {
    const vmin = Math.min(...traj.speed), vmax = Math.max(...traj.speed);
    for (let i = 1; i < traj.x.length; i++) {
      const [x0,y0] = tf(traj.x[i-1], traj.y[i-1]);
      const [x1,y1] = tf(traj.x[i], traj.y[i]);
      ctx.strokeStyle = speedColor(traj.speed[i], vmin, vmax);
      ctx.lineWidth = 4;
      ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.stroke();
    }
    // animated car marker, looping over the recorded lap time
    const t = traj.t && traj.t.length ? traj.t : traj.x.map((_,i)=>i/30);
    const totalT = t[t.length-1] || 1;
    if (carAnimStart === null) carAnimStart = performance.now();
    const elapsed = ((performance.now() - carAnimStart) / 1000) % totalT;
    let idx = 0;
    while (idx < t.length - 1 && t[idx] < elapsed) idx++;
    const [cx, cy] = tf(traj.x[idx], traj.y[idx]);
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(cx, cy, 6, 0, 2*Math.PI); ctx.fill();
    ctx.strokeStyle = '#37e2c4'; ctx.lineWidth = 2; ctx.stroke();
  }
}

let curveHistory = [];
function drawCurve(history) {
  const ctx = curveCanvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = curveCanvas.clientWidth, h = curveCanvas.clientHeight;
  curveCanvas.width = w*dpr; curveCanvas.height = h*dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  if (!history.length) { ctx.fillStyle = '#8aa0b4'; ctx.fillText('no data yet', 10, 20); return; }
  const pad = 36;
  const xs = history.map(r => r[1]), ys = history.map(r => r[2]);
  const minX = 0, maxX = Math.max(...xs, 1);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const tf = (x,y) => [pad + (x-minX)/(maxX-minX||1)*(w-pad-10), h-pad - (y-minY)/((maxY-minY)||1)*(h-pad-20)];

  ctx.strokeStyle = '#233040'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, 10); ctx.lineTo(pad, h-pad); ctx.lineTo(w-10, h-pad); ctx.stroke();
  ctx.fillStyle = '#8aa0b4'; ctx.font = '11px monospace';
  ctx.fillText(maxY.toFixed(1), 2, 14);
  ctx.fillText(minY.toFixed(1), 2, h-pad);
  ctx.fillText(maxX.toFixed(0)+'s', w-50, h-pad+16);

  ctx.strokeStyle = '#37e2c4'; ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((r, i) => {
    const [px, py] = tf(r[1], r[2]);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.stroke();
}

function drawNN(policy) {
  const ctx = nnCanvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = nnCanvas.clientWidth, h = nnCanvas.clientHeight;
  nnCanvas.width = w*dpr; nnCanvas.height = h*dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  document.getElementById('policyBadge').textContent = policy && policy.live ? 'training' : 'not running';
  if (!policy || !policy.live) return;
  const {weights, architecture} = policy.live;
  const labels = ['speed_x','speed_y','curv@0m','curv@20m','curv@40m','curv@80m','curv@160m','curv@320m','grade@0m','grade@40m','prev_action','bias'];
  const nIn = architecture.n_inputs, nHid = architecture.n_hidden, nOut = architecture.n_outputs;
  const colX = [60, w/2, w-60];
  const layerY = (n, i) => 20 + (h-40) * (n<=1 ? 0.5 : i/(n-1));

  const maxW1 = Math.max(1e-6, ...weights.w1.flat().map(Math.abs));
  for (let i = 0; i < nIn; i++) {
    for (let j = 0; j < nHid; j++) {
      const v = weights.w1[i][j];
      const [x0,y0] = [colX[0], layerY(nIn, i)];
      const [x1,y1] = [colX[1], layerY(nHid, j)];
      ctx.strokeStyle = v >= 0 ? `rgba(70,224,122,${0.15+0.5*Math.abs(v)/maxW1})` : `rgba(255,90,110,${0.15+0.5*Math.abs(v)/maxW1})`;
      ctx.lineWidth = 0.5 + 2.5 * Math.abs(v)/maxW1;
      ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.stroke();
    }
  }
  const maxW2 = Math.max(1e-6, ...weights.w2.flat().map(Math.abs));
  for (let j = 0; j < nHid; j++) {
    for (let k = 0; k < nOut; k++) {
      const v = weights.w2[j][k];
      const [x0,y0] = [colX[1], layerY(nHid, j)];
      const [x1,y1] = [colX[2], layerY(nOut, k)];
      ctx.strokeStyle = v >= 0 ? `rgba(70,224,122,${0.2+0.6*Math.abs(v)/maxW2})` : `rgba(255,90,110,${0.2+0.6*Math.abs(v)/maxW2})`;
      ctx.lineWidth = 0.5 + 3 * Math.abs(v)/maxW2;
      ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.stroke();
    }
  }
  for (let i = 0; i < nIn; i++) {
    const [x,y] = [colX[0], layerY(nIn, i)];
    ctx.fillStyle = '#161e2a'; ctx.beginPath(); ctx.arc(x,y,5,0,2*Math.PI); ctx.fill();
    ctx.strokeStyle = '#37e2c4'; ctx.stroke();
    ctx.fillStyle = '#8aa0b4'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
    ctx.fillText(labels[i] || ('in'+i), x-10, y+3);
  }
  ctx.textAlign = 'left';
  for (let j = 0; j < nHid; j++) {
    const [x,y] = [colX[1], layerY(nHid, j)];
    ctx.fillStyle = '#161e2a'; ctx.beginPath(); ctx.arc(x,y,6,0,2*Math.PI); ctx.fill();
    ctx.strokeStyle = '#ff8a3d'; ctx.stroke();
  }
  for (let k = 0; k < nOut; k++) {
    const [x,y] = [colX[2], layerY(nOut, k)];
    ctx.fillStyle = '#161e2a'; ctx.beginPath(); ctx.arc(x,y,7,0,2*Math.PI); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.stroke();
    ctx.fillStyle = '#e7edf3'; ctx.fillText('hold / release', x+12, y+3);
  }
}

function maybeRevealViv(live, circuit) {
  if (!live || !live.best_finished) return;
  const key = 'quali_prix_viv_revealed_' + circuit;
  if (localStorage.getItem(key)) { vivRevealed[circuit] = true; return; }
  localStorage.setItem(key, '1');
  vivRevealed[circuit] = true;
  document.getElementById('vivText').textContent =
    `VIV just completed its first clean lap of ${circuit} in ${fmtTime(live.best_lap_time)}.`;
  document.getElementById('vivOverlay').style.display = 'flex';
}

async function poll() {
  try {
    await ensureTrack();
    const state = await fetch('/api/state?circuit=' + currentCircuit).then(r => r.json());
    const policy = await fetch('/api/policy?circuit=' + currentCircuit).then(r => r.json());
    const live = state.live;

    statsRow.innerHTML = '';
    if (live) {
      statsRow.appendChild(stat('driver', vivName(currentCircuit)));
      statsRow.appendChild(stat('generation', live.generation));
      statsRow.appendChild(stat('elapsed', Math.floor(live.elapsed_s/60) + 'm ' + Math.floor(live.elapsed_s%60) + 's'));
      statsRow.appendChild(stat(live.best_finished ? 'best lap' : 'best (crashed)',
        live.best_finished ? fmtTime(live.best_lap_time) : (live.best_dist.toFixed(0) + '/' + live.track_length.toFixed(0) + ' m')));
      statsRow.appendChild(stat('workers', live.workers));
      statsRow.appendChild(stat('mutation σ', live.sigma));
    } else {
      statsRow.appendChild(stat('status', 'waiting for optimize.py...'));
    }

    medalRow.innerHTML = '';
    if (live && live.medals) {
      for (const [name, cls] of [['gold','gold'],['silver','silver'],['bronze','bronze']]) {
        const t = live.medals[name];
        const hit = live.best_finished && live.best_lap_time <= t;
        const el = document.createElement('div');
        el.className = 'medal ' + cls + (hit ? ' hit' : '');
        el.textContent = `${name} ${fmtTime(t)}` + (hit ? ' ✓' : '');
        medalRow.appendChild(el);
      }
    }

    if (live && live.trajectory) { lastTrajectory = live.trajectory; carAnimStart = null; }
    drawTrack(lastTrajectory);
    document.getElementById('trackTitle').textContent =
      (live && live.best_finished) ? `${vivName(currentCircuit)}'s best lap -- ${fmtTime(live.best_lap_time)}` : 'Best attempt so far (not yet a clean lap)';

    curveHistory = state.history || [];
    drawCurve(curveHistory);

    drawNN(policy);

    maybeRevealViv(live, currentCircuit);
  } catch (e) {
    console.error(e);
  }
}

function animLoop() {
  drawTrack(lastTrajectory);
  requestAnimationFrame(animLoop);
}

(async () => {
  await loadCircuits();
  document.getElementById('urls').textContent = window.location.host;
  await poll();
  requestAnimationFrame(animLoop);
  setInterval(poll, 1500);
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8743)
    ap.add_argument("--circuit", default="lasvegas", help="only affects the default dropdown selection")
    ap.add_argument("--lan", action="store_true", default=True,
                     help="bind 0.0.0.0 so others on your network can watch (default on)")
    ap.add_argument("--localhost-only", action="store_true", help="bind 127.0.0.1 only")
    args = ap.parse_args()

    host = "127.0.0.1" if args.localhost_only else "0.0.0.0"
    server = ThreadingHTTPServer((host, args.port), Handler)
    print(f"Dashboard running.")
    print(f"  on this machine: http://localhost:{args.port}/")
    if not args.localhost_only:
        lan_ip = _lan_ip()
        if lan_ip:
            print(f"  on your network (share with friends on the same Wi-Fi): http://{lan_ip}:{args.port}/")
        print("  (this is LAN-only -- nothing here is exposed to the wider internet)")
    print("Reads runs/<circuit>_live.json etc; start optimize.py / policy/train.py separately.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
