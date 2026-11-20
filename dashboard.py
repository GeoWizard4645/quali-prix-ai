"""
A local, live dashboard for watching optimize.py / policy/train.py /
live_driver.py -- no cloud, no new dependencies (stdlib http.server only).

Shows: the best lap animated on the track (speed-coloured, with pain/
reward markers), a learning curve, a gamified reward/pain readout, the
policy network's weights and its own attempt on a mini-track, a live view
of the bot actually driving the real game, and pause/play control for all
three engines -- reachable on the LAN so others can watch along.

Usage:
    python3 dashboard.py --circuit lasvegas
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
from control import control_path, write_control

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
        return None  # being written concurrently; try again next poll


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


def build_live_game_state(circuit):
    status = read_json(os.path.join(RUNS_DIR, f"{circuit}_live_status.json")) or {}
    ctrl = read_json(control_path("live", circuit)) or {"paused": True, "stop": False}
    has_frame = os.path.exists(os.path.join(RUNS_DIR, f"{circuit}_live_frame.jpg"))
    return {"status": status, "paused": ctrl.get("paused", True), "has_frame": has_frame}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

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
        elif parsed.path == "/api/live_game":
            self._json(build_live_game_state(circuit))
        elif parsed.path == "/api/live_frame":
            frame_path = os.path.join(RUNS_DIR, f"{circuit}_live_frame.jpg")
            if os.path.exists(frame_path):
                with open(frame_path, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        circuit = qs.get("circuit", ["lasvegas"])[0]
        kind = qs.get("kind", [""])[0]
        action = qs.get("action", [""])[0]
        if parsed.path == "/api/control" and kind in ("optimize", "policy", "live") and \
           action in ("pause", "resume"):
            write_control(control_path(kind, circuit), paused=(action == "pause"))
            self._json({"ok": True, "kind": kind, "paused": action == "pause"})
        else:
            self.send_response(400)
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
    --page:        #0d0d0d;
    --surface:     #1a1a19;
    --surface-2:   #202423;
    --border:      rgba(255,255,255,0.10);
    --text:        #ffffff;
    --text-2:      #c3c2b7;
    --muted:       #898781;
    --grid:        #2c2c2a;
    --baseline:    #383835;
    --accent:      #3987e5;   /* categorical slot 1 (dark) -- used as the brand accent */
    --accent-glow: rgba(57,135,229,0.35);
    --good:        #0ca30c;
    --warning:     #fab219;
    --critical:    #e66767;
    --gold:        #ffd451;
    --silver:      #d7dee6;
    --bronze:      #e0a05b;
  }
  * { box-sizing: border-box; }
  body {
    margin:0; background:
      radial-gradient(1200px 500px at 20% -10%, rgba(57,135,229,0.10), transparent),
      radial-gradient(900px 400px at 100% 0%, rgba(230,103,103,0.06), transparent),
      var(--page);
    color:var(--text); font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  header {
    display:flex; align-items:center; gap:16px; padding:16px 22px;
    border-bottom:1px solid var(--border); background:rgba(26,26,25,0.85);
    backdrop-filter: blur(6px); position:sticky; top:0; z-index:20; flex-wrap:wrap;
  }
  header h1 {
    font-size:20px; margin:0; letter-spacing:0.3px; font-weight:800;
    text-shadow: 0 0 18px var(--accent-glow);
  }
  header h1 span { color: var(--accent); }
  select, button {
    background:var(--surface-2); color:var(--text); border:1px solid var(--border);
    border-radius:8px; padding:7px 12px; font-size:13px; cursor:pointer; font-family:inherit;
  }
  button:hover { border-color: var(--accent); }
  button.primary { background:var(--accent); color:#06131f; font-weight:700; border:none; }
  button.ghost { background:transparent; }
  button.small { padding:4px 9px; font-size:12px; border-radius:6px; }
  .pill-btn.pause { color: var(--warning); border-color: var(--warning); }
  .pill-btn.play { color: var(--good); border-color: var(--good); }
  .stats-row { display:flex; gap:20px; margin-left:auto; flex-wrap:wrap; align-items:center; }
  .stat { display:flex; flex-direction:column; min-width:80px; }
  .stat .v { font-size:19px; font-weight:700; }
  .stat .l { font-size:10.5px; color:var(--muted); text-transform:uppercase; letter-spacing:0.6px; }
  .live-dot {
    width:9px; height:9px; border-radius:50%; background:var(--good); display:inline-block;
    box-shadow:0 0 8px var(--good); animation: pulse 1.4s infinite;
  }
  .live-dot.paused { background:var(--muted); box-shadow:none; animation:none; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }

  main { display:grid; grid-template-columns: 1.3fr 1fr; gap:16px; padding:16px; max-width:1600px; margin:0 auto; }
  .panel {
    background:linear-gradient(180deg, var(--surface), var(--surface-2));
    border:1px solid var(--border); border-radius:14px; padding:16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
  }
  .panel.wide { grid-column: 1 / -1; }
  .panel-head { display:flex; align-items:center; gap:10px; margin-bottom:10px; flex-wrap:wrap; }
  .panel-head h2 { font-size:12.5px; text-transform:uppercase; letter-spacing:0.9px; color:var(--muted); margin:0; flex:1; }
  canvas { width:100%; display:block; }
  #trackCanvas { height:440px; }
  #curveCanvas { height:180px; }
  #nnCanvas { height:210px; }
  #policyTrackCanvas { height:190px; }
  #liveImg { width:100%; border-radius:10px; background:#000; display:block; aspect-ratio: 1180/760; object-fit:contain; }

  .medal-row { display:flex; gap:8px; margin-top:10px; font-size:12px; flex-wrap:wrap; }
  .medal { padding:3px 10px; border-radius:12px; background:var(--surface-2); border:1px solid var(--border); }
  .medal.hit { font-weight:800; box-shadow: 0 0 10px currentColor inset; }
  .gold { color:var(--gold); } .silver { color:var(--silver); } .bronze { color:var(--bronze); }

  .badge { font-size:10.5px; padding:2px 9px; border-radius:10px; background:var(--surface-2); color:var(--muted); border:1px solid var(--border); }
  .badge.on { color:var(--good); border-color: var(--good); }
  .badge.off { color:var(--muted); }
  .badge.warn { color: var(--warning); border-color: var(--warning); }

  .reward-pain-row { display:flex; gap:16px; margin-top:10px; flex-wrap:wrap; }
  .meter-box { flex:1; min-width:180px; }
  .meter-box .meter-label { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px; display:flex; justify-content:space-between;}
  .meter-track { height:10px; border-radius:6px; background:var(--surface-2); border:1px solid var(--border); overflow:hidden; }
  .meter-fill { height:100%; border-radius:6px; transition: width 0.6s ease; }
  .meter-fill.pain { background: linear-gradient(90deg, var(--critical), var(--warning)); }
  .meter-fill.reward { background: linear-gradient(90deg, var(--accent), var(--good)); }

  #scoreBig { font-size:34px; font-weight:800; text-shadow: 0 0 18px var(--accent-glow); }
  #feed { margin-top:10px; display:flex; flex-direction:column; gap:5px; max-height:110px; overflow-y:auto; }
  .feed-item { font-size:12px; padding:5px 9px; border-radius:7px; background:var(--surface-2); border:1px solid var(--border); font-variant-numeric: tabular-nums; animation: slidein 0.4s ease; }
  .feed-item.good { color:var(--good); }
  .feed-item.great { color:var(--gold); font-weight:700; }
  @keyframes slidein { from { opacity:0; transform:translateY(-6px);} to {opacity:1; transform:none;} }

  .note { font-size:12px; color:var(--muted); margin-top:8px; line-height:1.5; }
  .note code { color: var(--text-2); }
  footer { padding:14px 22px; color:var(--muted); font-size:12px; text-align:center; }
  a { color: var(--accent); }

  #helpOverlay, #vivOverlay {
    position:fixed; inset:0; background:rgba(5,8,12,0.88); display:none;
    align-items:center; justify-content:center; z-index:50; padding:20px;
  }
  #helpCard, #vivCard {
    background:linear-gradient(160deg, var(--surface-2), var(--surface));
    border:1px solid var(--accent); border-radius:16px; padding:28px 32px;
    max-width:560px; box-shadow:0 0 60px var(--accent-glow); max-height:85vh; overflow-y:auto;
  }
  #helpCard h2, #vivCard h2 { color: var(--accent); margin-top:0; }
  #helpCard pre { background:#000; padding:10px 12px; border-radius:8px; overflow-x:auto; font-size:12.5px; border:1px solid var(--border); }
  #vivCard { text-align:center; }
  #vivCard p { color:var(--text-2); font-size:14px; }
  .close-x { position:absolute; top:14px; right:18px; cursor:pointer; color:var(--muted); font-size:20px; }
  .card-wrap { position:relative; }
</style>
</head>
<body>

<header>
  <h1>Quali&nbsp;Prix <span>AI</span></h1>
  <select id="circuitSelect"></select>
  <button class="ghost small" onclick="document.getElementById('helpOverlay').style.display='flex'">📖 How to run this</button>
  <div class="stats-row" id="statsRow"></div>
</header>

<main>
  <div class="panel">
    <div class="panel-head">
      <span class="live-dot" id="optLiveDot"></span>
      <h2 id="trackTitle">Best lap so far</h2>
      <button class="small pill-btn" id="optPauseBtn" onclick="toggleEngine('optimize')">⏸ Pause</button>
    </div>
    <canvas id="trackCanvas"></canvas>
    <div class="medal-row" id="medalRow"></div>
  </div>

  <div class="panel">
    <div class="panel-head"><h2>Learning curve</h2></div>
    <canvas id="curveCanvas"></canvas>
    <div class="note">Lower is better: lap time once a clean lap is found, otherwise a crash-distance score (&gt;1000 = hasn't finished a lap yet).</div>

    <div class="panel-head" style="margin-top:14px;"><h2>🎮 Training score</h2></div>
    <div id="scoreBig">0</div>
    <div class="reward-pain-row">
      <div class="meter-box">
        <div class="meter-label"><span>💥 Pain (track left to conquer)</span><span id="painPct">--</span></div>
        <div class="meter-track"><div class="meter-fill pain" id="painFill" style="width:0%"></div></div>
      </div>
      <div class="meter-box">
        <div class="meter-label"><span>🏆 Reward (progress to Gold)</span><span id="rewardPct">--</span></div>
        <div class="meter-track"><div class="meter-fill reward" id="rewardFill" style="width:0%"></div></div>
      </div>
    </div>
    <div id="feed"></div>
  </div>

  <div class="panel wide">
    <div class="panel-head">
      <span class="live-dot" id="liveGameDot"></span>
      <h2>🏎️ Live: the bot actually playing the real game</h2>
      <span class="badge" id="liveGameBadge">off</span>
      <button class="small pill-btn" id="liveGamePauseBtn" onclick="toggleEngine('live')">▶ Watch it play live</button>
    </div>
    <div style="display:grid; grid-template-columns: 1.4fr 1fr; gap:16px;">
      <div>
        <img id="liveImg" src="" alt="not watching live right now">
        <div class="note" id="liveGameNote">Click "Watch it play live" to open a real browser against the live game (Practice mode only -- never submits a time) and replay the current best genome there, continuously.</div>
      </div>
      <div class="note">
        <b>How this works:</b> a background Playwright browser loads the real page, confirms Practice mode is on (it refuses to send any input otherwise), and replays the current best genome's exact press/release timing with the real Up-arrow key. It restarts automatically after a lap or a spin, and <b>auto-pauses after ~30 live minutes or 40 attempts</b> so it never runs unattended all night against the real site -- press the button again to watch another session.
        <br><br>Live timing sync with the real page is a bit looser than the simulator (which is tick-for-tick verified against the real physics code, see README) -- the live car sometimes spins out earlier than the simulator predicted. That's an open, honest gap in the input-replay layer, not the physics model.
        <div id="liveGameStatus" class="note" style="margin-top:8px;"></div>
      </div>
    </div>
  </div>

  <div class="panel wide">
    <div class="panel-head">
      <span class="live-dot" id="policyLiveDot"></span>
      <h2>🧠 Policy network (phase 2) <span class="badge" id="policyBadge">not running</span></h2>
      <button class="small pill-btn" id="policyPauseBtn" onclick="toggleEngine('policy')">⏸ Pause</button>
    </div>
    <div style="display:grid; grid-template-columns: 1fr 1.2fr; gap:16px;">
      <div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:4px;text-transform:uppercase;">Its own current best attempt</div>
        <canvas id="policyTrackCanvas"></canvas>
      </div>
      <div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:4px;text-transform:uppercase;">Its brain (green = push toward hold, red = toward release)</div>
        <canvas id="nnCanvas"></canvas>
      </div>
    </div>
    <div class="note">This is a <i>separate</i>, from-scratch learner: a tiny neural net reading (speed, upcoming track curvature at 6 lookahead distances, elevation grade, its own last action) and training with evolution, not the sequence search above -- so it typically takes much longer to get past the same early corners, especially right after being (re)started. That's expected, not broken; see README's Phase 2 section.</div>
  </div>
</main>

<footer>Served locally from your laptop at <span id="urls"></span> -- nothing here touches the internet beyond the live-game panel, which only talks to the real quali-prix page in Practice mode.</footer>

<div id="vivOverlay">
  <div id="vivCard" class="card-wrap">
    <div style="font-size:40px;">🏎️</div>
    <h2>Say hello to VIV.</h2>
    <p id="vivText">VIV just completed its first clean lap.</p>
    <button class="primary" onclick="document.getElementById('vivOverlay').style.display='none'">Nice</button>
  </div>
</div>

<div id="helpOverlay">
  <div id="helpCard" class="card-wrap">
    <span class="close-x" onclick="document.getElementById('helpOverlay').style.display='none'">✕</span>
    <h2>How to run this, any time</h2>
    <p>From the project folder, in up to three terminal tabs:</p>
    <pre>source .venv/bin/activate
python3 optimize.py --circuit lasvegas        # terminal 1: let it learn (runs until paused)
python3 policy/train.py --circuit lasvegas    # terminal 2: optional, trains the NN panel
python3 dashboard.py                          # terminal 3: this page</pre>
    <p>Then open <code id="helpUrl">this page's URL</code>. Use the <b>⏸ Pause / ▶ Resume</b> buttons on each panel any time -- no need to stop the terminal. <code>Ctrl+C</code> in a terminal fully stops that engine; running the same command again resumes from its last checkpoint automatically.</p>
    <p>Share <code id="helpLanUrl">the LAN URL</code> with anyone on the same Wi-Fi to watch along -- nothing here leaves your network.</p>
    <p>The "Watch it play live" button drives the <b>real</b> game in Practice mode. It's off by default and self-pauses after a while, on purpose.</p>
  </div>
</div>

<script>
const statsRow = document.getElementById('statsRow');
const medalRow = document.getElementById('medalRow');
const trackCanvas = document.getElementById('trackCanvas');
const curveCanvas = document.getElementById('curveCanvas');
const nnCanvas = document.getElementById('nnCanvas');
const policyTrackCanvas = document.getElementById('policyTrackCanvas');
const circuitSelect = document.getElementById('circuitSelect');

let currentCircuit = 'lasvegas';
let trackSummary = null;
let lastTrajectory = null;
let lastPolicyTrajectory = null;
let vivRevealed = {};
let engineState = {optimize: {paused:false}, policy: {paused:false}, live: {paused:true}};
let scoreSeen = new Set();
let cumulativeScore = 0;

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
function vivName(circuit) { return vivRevealed[circuit] ? 'VIV' : 'the car'; }

async function toggleEngine(kind) {
  const willPause = !engineState[kind].paused;
  await fetch(`/api/control?circuit=${currentCircuit}&kind=${kind}&action=${willPause ? 'pause' : 'resume'}`, {method:'POST'});
  engineState[kind].paused = willPause;
  updateEngineButtons();
}
function updateEngineButtons() {
  const map = {optimize: ['optPauseBtn','optLiveDot'], policy: ['policyPauseBtn','policyLiveDot'], live: ['liveGamePauseBtn','liveGameDot']};
  for (const [kind, [btnId, dotId]] of Object.entries(map)) {
    const paused = engineState[kind].paused;
    const btn = document.getElementById(btnId), dot = document.getElementById(dotId);
    if (kind === 'live') {
      btn.textContent = paused ? '▶ Watch it play live' : '⏸ Stop watching';
    } else {
      btn.textContent = paused ? '▶ Resume' : '⏸ Pause';
    }
    btn.className = 'small pill-btn ' + (paused ? 'play' : 'pause');
    dot.className = 'live-dot' + (paused ? ' paused' : '');
  }
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
  trackSummary = null; lastTrajectory = null; lastPolicyTrajectory = null;
  curveHistory = []; scoreSeen = new Set(); cumulativeScore = 0;
});

async function ensureTrack() {
  if (trackSummary && trackSummary._circuit === currentCircuit) return trackSummary;
  const r = await fetch('/api/track?circuit=' + currentCircuit).then(r => r.json());
  r._circuit = currentCircuit;
  trackSummary = r;
  return r;
}

function speedColor(v, vmin, vmax) {
  // single-hue sequential ramp (blue), dark surface: low recedes into the
  // surface, high pops bright -- per the dataviz skill's sequential rule.
  const t = Math.max(0, Math.min(1, (v - vmin) / Math.max(1, (vmax - vmin))));
  const dark = [24,79,149], light = [205,226,251];
  const r = Math.round(dark[0]+(light[0]-dark[0])*t), g = Math.round(dark[1]+(light[1]-dark[1])*t), b = Math.round(dark[2]+(light[2]-dark[2])*t);
  return `rgb(${r},${g},${b})`;
}
function fitTransform(xs, ys, w, h, pad) {
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const scale = Math.min((w - 2*pad) / (maxX - minX || 1), (h - 2*pad) / (maxY - minY || 1));
  return (x, y) => [pad + (x - minX) * scale, h - pad - (y - minY) * scale];
}

let carAnimStart = {};
function drawTrackOnCanvas(canvas, traj, animKey) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  if (!trackSummary) return;
  const tf = fitTransform(trackSummary.x, trackSummary.y, w, h, 26);

  ctx.strokeStyle = '#2c2c2a';
  ctx.lineWidth = 9;
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
    // pain/reward markers from recorded events
    if (traj.events) {
      for (const [ex, ey, ev] of traj.events) {
        const [px, py] = tf(ex, ey);
        if (ev === 'crash') {
          ctx.fillStyle = '#e66767';
          ctx.font = 'bold 15px system-ui'; ctx.textAlign='center';
          ctx.fillText('💥', px, py+5);
        } else if (ev === 'save' || ev === 'spin' || ev === 'scrape') {
          ctx.fillStyle = '#fab219';
          ctx.beginPath(); ctx.arc(px, py, 4, 0, 2*Math.PI); ctx.fill();
        }
      }
    }
    const t = traj.t && traj.t.length ? traj.t : traj.x.map((_,i)=>i/30);
    const totalT = t[t.length-1] || 1;
    if (!(animKey in carAnimStart)) carAnimStart[animKey] = performance.now();
    const elapsed = ((performance.now() - carAnimStart[animKey]) / 1000) % totalT;
    let idx = 0;
    while (idx < t.length - 1 && t[idx] < elapsed) idx++;
    const [cx, cy] = tf(traj.x[idx], traj.y[idx]);
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(cx, cy, 6, 0, 2*Math.PI); ctx.fill();
    ctx.strokeStyle = '#3987e5'; ctx.lineWidth = 2; ctx.stroke();
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
  if (!history.length) { ctx.fillStyle = '#898781'; ctx.font='12px system-ui'; ctx.fillText('no data yet', 10, 20); return; }
  const pad = 38;
  const xs = history.map(r => r[1]), ys = history.map(r => r[2]);
  const minX = 0, maxX = Math.max(...xs, 1);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const tf = (x,y) => [pad + (x-minX)/(maxX-minX||1)*(w-pad-10), h-pad - (y-minY)/((maxY-minY)||1)*(h-pad-18)];

  ctx.strokeStyle = '#2c2c2a'; ctx.lineWidth = 1;
  for (let gy = 0; gy <= 4; gy++) {
    const y = 10 + gy*(h-pad-10)/4;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w-10, y); ctx.stroke();
  }
  ctx.fillStyle = '#898781'; ctx.font = '10.5px system-ui'; ctx.textAlign = 'right';
  ctx.font = '10.5px system-ui'; ctx.textAlign = 'right';
  ctx.fillText(maxY.toFixed(1), pad-4, 14);
  ctx.fillText(minY.toFixed(1), pad-4, h-pad);
  ctx.textAlign = 'left';
  ctx.fillText(maxX.toFixed(0)+'s', w-46, h-pad+14);

  ctx.strokeStyle = '#3987e5'; ctx.lineWidth = 2;
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
  if (!policy || !policy.live) return;
  const {weights, architecture} = policy.live;
  const labels = ['speed_x','speed_y','curv@0m','curv@20m','curv@40m','curv@80m','curv@160m','curv@320m','grade@0m','grade@40m','prev_action','bias'];
  const nIn = architecture.n_inputs, nHid = architecture.n_hidden, nOut = architecture.n_outputs;
  const colX = [56, w/2, w-50];
  const layerY = (n, i) => 16 + (h-32) * (n<=1 ? 0.5 : i/(n-1));

  const maxW1 = Math.max(1e-6, ...weights.w1.flat().map(Math.abs));
  for (let i = 0; i < nIn; i++) {
    for (let j = 0; j < nHid; j++) {
      const v = weights.w1[i][j];
      const [x0,y0] = [colX[0], layerY(nIn, i)];
      const [x1,y1] = [colX[1], layerY(nHid, j)];
      ctx.strokeStyle = v >= 0 ? `rgba(12,163,12,${0.15+0.5*Math.abs(v)/maxW1})` : `rgba(230,103,103,${0.15+0.5*Math.abs(v)/maxW1})`;
      ctx.lineWidth = 0.5 + 2.2 * Math.abs(v)/maxW1;
      ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.stroke();
    }
  }
  const maxW2 = Math.max(1e-6, ...weights.w2.flat().map(Math.abs));
  for (let j = 0; j < nHid; j++) {
    for (let k = 0; k < nOut; k++) {
      const v = weights.w2[j][k];
      const [x0,y0] = [colX[1], layerY(nHid, j)];
      const [x1,y1] = [colX[2], layerY(nOut, k)];
      ctx.strokeStyle = v >= 0 ? `rgba(12,163,12,${0.2+0.6*Math.abs(v)/maxW2})` : `rgba(230,103,103,${0.2+0.6*Math.abs(v)/maxW2})`;
      ctx.lineWidth = 0.5 + 2.8 * Math.abs(v)/maxW2;
      ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.stroke();
    }
  }
  for (let i = 0; i < nIn; i++) {
    const [x,y] = [colX[0], layerY(nIn, i)];
    ctx.fillStyle = '#202423'; ctx.beginPath(); ctx.arc(x,y,4.5,0,2*Math.PI); ctx.fill();
    ctx.strokeStyle = '#3987e5'; ctx.stroke();
    ctx.fillStyle = '#898781'; ctx.font = '9.5px system-ui'; ctx.textAlign = 'right';
    ctx.fillText(labels[i] || ('in'+i), x-9, y+3);
  }
  ctx.textAlign = 'left';
  for (let j = 0; j < nHid; j++) {
    const [x,y] = [colX[1], layerY(nHid, j)];
    ctx.fillStyle = '#202423'; ctx.beginPath(); ctx.arc(x,y,5.5,0,2*Math.PI); ctx.fill();
    ctx.strokeStyle = '#ffd451'; ctx.stroke();
  }
  for (let k = 0; k < nOut; k++) {
    const [x,y] = [colX[2], layerY(nOut, k)];
    ctx.fillStyle = '#202423'; ctx.beginPath(); ctx.arc(x,y,6.5,0,2*Math.PI); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.stroke();
    ctx.fillStyle = '#fff'; ctx.font='10.5px system-ui'; ctx.fillText('hold/release', x+11, y+3);
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

function pushFeed(text, cls) {
  const feed = document.getElementById('feed');
  const el = document.createElement('div');
  el.className = 'feed-item ' + cls;
  el.textContent = text;
  feed.insertBefore(el, feed.firstChild);
  while (feed.children.length > 12) feed.removeChild(feed.lastChild);
}

function updateScoreAndFeed(history, trackLength, medals) {
  let prevFit = null, newBest = Infinity;
  let painFit = history.length ? history[history.length-1][2] : 7000;
  for (const row of history) {
    const [gen, elapsed, fit] = row;
    const key = gen + ':' + fit;
    if (prevFit !== null && !scoreSeen.has(key)) {
      scoreSeen.add(key);
      if (prevFit >= 1000 && fit >= 1000) {
        const distGain = prevFit - fit;
        if (distGain > 0.5) { cumulativeScore += distGain; pushFeed(`+${distGain.toFixed(0)}m further before crashing`, 'good'); }
      } else if (prevFit >= 1000 && fit < 1000) {
        cumulativeScore += 5000;
        pushFeed(`🏁 LAP COMPLETE -- ${fmtTime(fit)}!`, 'great');
      } else if (prevFit < 1000 && fit < 1000) {
        const timeGain = prevFit - fit;
        if (timeGain > 0.0005) { cumulativeScore += timeGain*1000; pushFeed(`⚡ -${timeGain.toFixed(3)}s faster -- ${fmtTime(fit)}`, 'good'); }
      }
    } else if (prevFit === null && !scoreSeen.has(key)) {
      scoreSeen.add(key);
    }
    prevFit = fit;
  }
  document.getElementById('scoreBig').textContent = Math.round(cumulativeScore).toLocaleString();

  const lastFit = history.length ? history[history.length-1][2] : null;
  const painPct = lastFit == null ? 100 : (lastFit >= 1000 ? Math.min(100, (lastFit-1000)/trackLength*100) : 0);
  document.getElementById('painFill').style.width = painPct.toFixed(0) + '%';
  document.getElementById('painPct').textContent = painPct.toFixed(0) + '% left';

  let rewardPct = 0;
  if (lastFit != null && lastFit < 1000 && medals) {
    const bronze = medals.bronze, gold = medals.gold;
    rewardPct = Math.max(0, Math.min(100, (bronze - lastFit) / (bronze - gold) * 100));
  }
  document.getElementById('rewardFill').style.width = rewardPct.toFixed(0) + '%';
  document.getElementById('rewardPct').textContent = (lastFit != null && lastFit < 1000) ? rewardPct.toFixed(0) + '% to Gold' : 'not finished yet';
}

async function pollLiveGame() {
  const dot = document.getElementById('liveGameDot');
  const badge = document.getElementById('liveGameBadge');
  const img = document.getElementById('liveImg');
  const note = document.getElementById('liveGameStatus');
  try {
    const r = await fetch('/api/live_game?circuit=' + currentCircuit).then(r => r.json());
    engineState.live.paused = r.paused;
    updateEngineButtons();
    if (!r.paused && r.has_frame) {
      img.src = '/api/live_frame?circuit=' + currentCircuit + '&t=' + Date.now();
      badge.textContent = 'live'; badge.className = 'badge on';
    } else {
      badge.textContent = r.paused ? 'off' : 'starting...'; badge.className = 'badge off';
    }
    if (r.status && r.status.note) note.textContent = r.status.note +
      (r.status.attempts != null ? ` (attempt ${r.status.attempts})` : '');
  } catch (e) { /* live_driver.py probably isn't running */ }
}

async function poll() {
  try {
    await ensureTrack();
    const state = await fetch('/api/state?circuit=' + currentCircuit).then(r => r.json());
    const policy = await fetch('/api/policy?circuit=' + currentCircuit).then(r => r.json());
    const live = state.live;

    statsRow.innerHTML = '';
    if (live) {
      engineState.optimize.paused = !!live.paused;
      statsRow.appendChild(stat('driver', vivName(currentCircuit)));
      statsRow.appendChild(stat('generation', live.generation));
      statsRow.appendChild(stat('elapsed', Math.floor(live.elapsed_s/60) + 'm ' + Math.floor(live.elapsed_s%60) + 's'));
      statsRow.appendChild(stat(live.best_finished ? 'best lap' : 'best (crashed)',
        live.best_finished ? fmtTime(live.best_lap_time) : (live.best_dist.toFixed(0) + '/' + live.track_length.toFixed(0) + ' m')));
      statsRow.appendChild(stat('workers', live.workers));
      updateScoreAndFeed(state.history || [], live.track_length, live.medals);
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

    if (live && live.trajectory) { lastTrajectory = live.trajectory; delete carAnimStart['main']; }
    drawTrackOnCanvas(trackCanvas, lastTrajectory, 'main');
    document.getElementById('trackTitle').textContent =
      (live && live.best_finished) ? `${vivName(currentCircuit)}'s best lap -- ${fmtTime(live.best_lap_time)}` : 'Best attempt so far (not yet a clean lap)';

    curveHistory = state.history || [];
    drawCurve(curveHistory);

    const policyBadge = document.getElementById('policyBadge');
    if (policy && policy.live) {
      policyBadge.textContent = policy.live.paused ? 'paused' : 'training';
      policyBadge.className = 'badge ' + (policy.live.paused ? 'off' : 'on');
      engineState.policy.paused = !!policy.live.paused;
      if (policy.live.trajectory) { lastPolicyTrajectory = policy.live.trajectory; delete carAnimStart['policy']; }
    } else {
      policyBadge.textContent = 'not running'; policyBadge.className = 'badge off';
    }
    drawTrackOnCanvas(policyTrackCanvas, lastPolicyTrajectory, 'policy');
    drawNN(policy);

    await pollLiveGame();
    updateEngineButtons();
    maybeRevealViv(live, currentCircuit);
  } catch (e) {
    console.error(e);
  }
}

function animLoop() {
  drawTrackOnCanvas(trackCanvas, lastTrajectory, 'main');
  drawTrackOnCanvas(policyTrackCanvas, lastPolicyTrajectory, 'policy');
  requestAnimationFrame(animLoop);
}

(async () => {
  await loadCircuits();
  const host = window.location.host;
  document.getElementById('urls').textContent = host;
  document.getElementById('helpUrl').textContent = window.location.href;
  document.getElementById('helpLanUrl').textContent = window.location.href;
  await poll();
  requestAnimationFrame(animLoop);
  setInterval(poll, 1500);
  setInterval(pollLiveGame, 700);
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8743)
    ap.add_argument("--circuit", default="lasvegas", help="only affects the default dropdown selection")
    ap.add_argument("--localhost-only", action="store_true", help="bind 127.0.0.1 only")
    args = ap.parse_args()

    host = "127.0.0.1" if args.localhost_only else "0.0.0.0"
    server = ThreadingHTTPServer((host, args.port), Handler)
    print("Dashboard running.")
    print(f"  on this machine: http://localhost:{args.port}/")
    if not args.localhost_only:
        lan_ip = _lan_ip()
        if lan_ip:
            print(f"  on your network (share with friends on the same Wi-Fi): http://{lan_ip}:{args.port}/")
        print("  (this is LAN-only -- nothing here is exposed to the wider internet)")
    print("Reads runs/<circuit>_live.json etc; start optimize.py / policy/train.py / live_driver.py separately.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
