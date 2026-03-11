"""
Evolution replay visualizer
----------------------------
For each generation snapshot, plays TWO phases:
  1. BEFORE TRAINING  — same robot shape, no trained control params
                        (random neural net → robot barely moves)
  2. AFTER TRAINING   — same robot shape, trained control params loaded
                        (learned gait → robot locomotes)

Then advances to the next generation and repeats.

Usage:
    python evolution_visualizer.py --config config.yaml
"""

from flask import Flask, render_template_string, jsonify, request
from argparse import ArgumentParser
from simulator import Simulator
from utils import load_config
import threading, queue, time, json, numpy as np, os, glob
import taichi as ti

app = Flask(__name__)

TARGET_FPS     = 60.0
STEPS_PER_GEN  = 400   # steps shown in each phase

frame_lock   = threading.Lock()
latest_frame = None
latest_meta  = None

cmd_queue = queue.Queue()
snapshots = []
_config   = None

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Evolution Replay</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #111; color: #eee; font-family: monospace;
    display: flex; flex-direction: column; align-items: center;
    padding: 16px; gap: 12px;
  }
  h2 { font-size: 15px; color: #aaa; letter-spacing: 1px; }
  #wrap { position: relative; }
  canvas { background: #1a1a2e; border: 1px solid #333; display: block; }

  /* Phase banner overlaid on canvas */
  #phase-banner {
    position: absolute;
    top: 10px; left: 50%; transform: translateX(-50%);
    padding: 5px 20px; border-radius: 20px;
    font-size: 15px; font-weight: bold; letter-spacing: 1px;
    pointer-events: none;
    transition: background 0.3s, color 0.3s;
  }
  #phase-banner.before {
    background: rgba(180, 60, 60, 0.85);
    color: #fff;
    border: 1px solid #f88;
  }
  #phase-banner.after {
    background: rgba(40, 160, 80, 0.85);
    color: #fff;
    border: 1px solid #6f6;
  }

  #controls {
    display: flex; align-items: center; gap: 14px;
    background: #222; border: 1px solid #444; border-radius: 8px;
    padding: 10px 20px; font-size: 14px;
    flex-wrap: wrap; justify-content: center; width: 900px;
  }
  button {
    background: #333; color: #eee; border: 1px solid #555;
    border-radius: 5px; padding: 5px 14px; cursor: pointer;
    font-size: 14px; font-family: monospace;
  }
  button:hover { background: #555; }
  #gen-slider { width: 220px; cursor: pointer; }
  .lbl { color: #888; }
  .val { color: #7df; font-weight: bold; }
  #debug {
    background: #1a1a1a; border: 1px solid #555; border-radius: 6px;
    padding: 8px 14px; font-size: 12px; color: #aaa;
    width: 900px; line-height: 1.9;
  }
  #debug span { color: #7df; }
</style>
</head>
<body>
<h2>EVOLUTION REPLAY — shape evolves across generations &amp; training teaches locomotion</h2>

<div id="wrap">
  <canvas id="c" width="900" height="460"></canvas>
  <div id="phase-banner" class="before">BEFORE TRAINING</div>
</div>

<div id="controls">
  <button id="btn-prev">&#9664; Prev</button>
  <span>
    <span class="lbl">Gen </span><span id="lbl-gen" class="val">0</span>
    <span class="lbl"> / </span><span id="lbl-total" class="val">?</span>
  </span>
  <input id="gen-slider" type="range" min="0" max="0" value="0">
  <button id="btn-next">Next &#9654;</button>
  <span><span class="lbl">Fitness: </span><span id="lbl-fitness" class="val">-</span></span>
  <span><span class="lbl">FPS: </span><span id="lbl-fps" class="val">-</span></span>
  <button id="btn-pause">Pause</button>
</div>

<div id="debug">
  polls: <span id="d-polls">0</span> &nbsp;|&nbsp;
  frames drawn: <span id="d-draws">0</span> &nbsp;|&nbsp;
  last error: <span id="d-err" style="color:#f88">none</span><br>
  positions len: <span id="d-poslen">-</span> &nbsp;|&nbsp;
  springs len: <span id="d-sprlen">-</span> &nbsp;|&nbsp;
  nMasses: <span id="d-nm">-</span> &nbsp;|&nbsp;
  com: <span id="d-com">-</span><br>
  response keys: <span id="d-keys">-</span>
</div>

<script>
const canvas = document.getElementById("c");
const ctx    = canvas.getContext("2d");
const W = canvas.width, H = canvas.height;
const VIEW_SCALE = 600, VIEW_X0 = 50, VIEW_Y0 = H - 40, GROUND_Y = 0.02;

let springs = [], nMasses = 0;
let positions = [], activations = [], com = [0, 0];
let lastGen = -1, lastPhase = "";

function toScreen(x, y) {
  return [VIEW_X0 + (x - com[0] + 1.0) * VIEW_SCALE,
          VIEW_Y0 - y * VIEW_SCALE];
}

function drawFrame() {
  ctx.clearRect(0, 0, W, H);
  const gy = VIEW_Y0 - GROUND_Y * VIEW_SCALE;
  ctx.fillStyle = "#1e3a1e";
  ctx.fillRect(0, gy, W, H - gy);
  ctx.strokeStyle = "#3a7a3a"; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
  if (!positions.length) return;

  for (let s = 0; s < springs.length; s++) {
    const [a, b] = springs[s];
    const act = activations.length > s ? activations[s] : 0;
    ctx.strokeStyle = "rgb(" + Math.round(100+act*155) + "," +
                               Math.round(80-act*40)   + ",220)";
    ctx.lineWidth = 2;
    const [ax,ay] = toScreen(positions[a][0], positions[a][1]);
    const [bx,by] = toScreen(positions[b][0], positions[b][1]);
    ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.stroke();
  }
  for (let m = 0; m < Math.min(positions.length, nMasses); m++) {
    const [sx,sy] = toScreen(positions[m][0], positions[m][1]);
    ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI*2);
    ctx.fillStyle = "#ffd700"; ctx.fill();
  }
}

function updatePhaseBanner(phase) {
  if (phase === lastPhase) return;
  lastPhase = phase;
  const banner = document.getElementById("phase-banner");
  if (phase === "before") {
    banner.textContent = "BEFORE TRAINING";
    banner.className = "before";
  } else {
    banner.textContent = "AFTER TRAINING";
    banner.className = "after";
  }
}

let pollCount = 0, drawCount = 0;

async function poll() {
  pollCount++;
  document.getElementById("d-polls").textContent = pollCount;
  try {
    const r = await fetch("/frame");
    if (!r.ok) {
      document.getElementById("d-err").textContent = "HTTP " + r.status;
      setTimeout(poll, 1000/60); return;
    }
    const text = await r.text();
    let d;
    try {
      d = JSON.parse(text);
    } catch(e) {
      document.getElementById("d-err").textContent = "JSON parse: " + e.message + " | raw: " + text.slice(0,120);
      setTimeout(poll, 1000/60); return;
    }

    document.getElementById("d-keys").textContent = Object.keys(d).join(", ");

    if (d.topology) {
      springs   = d.topology.springs;
      nMasses   = d.topology.n_masses;
      positions = [];
      document.getElementById("d-sprlen").textContent = springs.length;
      document.getElementById("d-nm").textContent = nMasses;
    }
    if (d.gen !== undefined && (d.gen !== lastGen || d.phase !== lastPhase)) {
      lastGen = d.gen;
      document.getElementById("lbl-gen").textContent     = d.gen;
      document.getElementById("lbl-total").textContent   = d.n_gens;
      document.getElementById("lbl-fitness").textContent = d.fitness.toFixed(4);
      const sl = document.getElementById("gen-slider");
      sl.max = d.n_gens; sl.value = d.gen;
    }
    if (d.phase) updatePhaseBanner(d.phase);

    if (d.positions && d.positions.length > 0) {
      positions   = d.positions;
      activations = d.activations || [];
      com         = d.com || [0,0];
      document.getElementById("lbl-fps").textContent = (d.fps||0).toFixed(1);
      document.getElementById("d-poslen").textContent = positions.length;
      document.getElementById("d-com").textContent = "[" + com.map(v=>v.toFixed(3)).join(", ") + "]";
      drawCount++;
      document.getElementById("d-draws").textContent = drawCount;
      document.getElementById("d-err").textContent = "none";
      drawFrame();
    } else {
      document.getElementById("d-err").textContent =
        d.positions ? "positions empty (len=0)" : "no positions key in response";
    }
  } catch(e) {
    document.getElementById("d-err").textContent = e.toString();
  }
  setTimeout(poll, 1000 / 60);
}
poll();

async function setGen(idx) {
  await fetch("/set_gen", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({gen_idx: +idx})
  });
}
document.getElementById("btn-prev").onclick = () => {
  const sl = document.getElementById("gen-slider");
  const v = Math.max(0, +sl.value - 1); sl.value = v; setGen(v);
};
document.getElementById("btn-next").onclick = () => {
  const sl = document.getElementById("gen-slider");
  const v = Math.min(+sl.max, +sl.value + 1); sl.value = v; setGen(v);
};
document.getElementById("gen-slider").oninput = e => setGen(+e.target.value);
document.getElementById("btn-pause").onclick = async () => {
  const r = await fetch("/toggle_pause", {method: "POST"});
  const d = await r.json();
  document.getElementById("btn-pause").textContent = d.paused ? "Play" : "Pause";
};
</script>
</body>
</html>
"""


# ── Simulation thread ─────────────────────────────────────────────────────────

def sim_thread_fn():
    global latest_frame, latest_meta

    current_gen  = -1
    current_phase = "before"   # "before" | "after"
    simulator    = None
    n_masses     = 0
    n_springs    = 0
    max_steps    = 0
    step_idx     = 0
    paused       = False
    fps_samples  = []
    last_fps_upd = time.perf_counter()
    actual_fps   = 0.0

    def load_phase(gen_idx, phase):
        """Load a generation in either 'before' (no params) or 'after' (trained params) phase."""
        global latest_frame, latest_meta   # assignment in nested fn needs explicit global
        nonlocal simulator, n_masses, n_springs, max_steps, step_idx
        nonlocal current_gen, current_phase
        current_gen   = gen_idx
        current_phase = phase

        robot = snapshots[gen_idx]
        cfg   = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _config.items()}
        cfg["simulator"]["n_masses"]  = int(robot.get("max_n_masses",  robot["n_masses"]))
        cfg["simulator"]["n_springs"] = int(robot.get("max_n_springs", robot["n_springs"]))
        cfg["simulator"]["n_sims"]    = 1

        ti.reset()

        simulator = Simulator(
            sim_config=cfg["simulator"],
            taichi_config=cfg["taichi"],
            seed=cfg["seed"],
            needs_grad=False,
        )
        simulator.initialize([robot["masses"]], [robot["springs"]])

        # Only load control params in the "after" phase
        if phase == "after" and "control_params" in robot:
            cp = robot["control_params"]
            print(f"[sim]   set_control_params shape={np.array(cp).shape} robot n_springs={robot['n_springs']} max_n_springs={robot.get('max_n_springs','?')}", flush=True)
            try:
                simulator.set_control_params([0], [cp])
                print(f"[sim]   set_control_params OK", flush=True)
            except Exception as e:
                print(f"[sim]   set_control_params FAILED: {e}", flush=True)
        elif phase == "after":
            print(f"[sim]   WARNING: no control_params in snapshot for gen {gen_idx}", flush=True)

        n_masses  = int(simulator.n_masses[0])
        n_springs = int(simulator.n_springs[0])
        max_steps = int(simulator.steps[None])
        step_idx  = 0

        with frame_lock:
            latest_meta = {
                "topology": {
                    "springs":   robot["springs"].tolist(),
                    "n_masses":  n_masses,
                    "n_springs": n_springs,
                },
                "gen":     gen_idx,
                "n_gens":  len(snapshots) - 1,
                "fitness": float(robot.get("fitness", 0)),
                "phase":   phase,
            }

    # Start at gen 0, before phase
    load_phase(0, "before")

    while True:
        try:
            while True:
                cmd = cmd_queue.get_nowait()
                if cmd[0] == "set_gen":
                    load_phase(cmd[1], "before")   # always restart from "before"
                elif cmd[0] == "pause":
                    paused = True
                elif cmd[0] == "play":
                    paused = False
        except queue.Empty:
            pass

        if paused:
            time.sleep(0.02)
            continue

        frame_start = time.perf_counter()

        # Phase/generation advance logic
        if step_idx >= min(STEPS_PER_GEN, max_steps - 1):
            if current_phase == "before":
                # Done with "before" — show "after" for same generation
                time.sleep(0.8)   # pause between phases so viewer can notice
                load_phase(current_gen, "after")
            else:
                # Done with "after" — advance to next generation, start at "before"
                time.sleep(1.2)   # slightly longer pause between generations
                next_gen = (current_gen + 1) % len(snapshots)
                load_phase(next_gen, "before")
            continue

        # Simulation step
        t = step_idx
        simulator.compute_com(t)
        simulator.nn1(t)
        simulator.nn2(t)
        simulator.apply_spring_force(t)
        simulator.advance(t + 1)

        pos = simulator.x.to_numpy()[0, t + 1, :n_masses]
        act = simulator.act.to_numpy()[0, t,    :n_springs]
        com = pos.mean(axis=0)
        step_idx += 1

        # Detect physics blow-up (NaN positions)
        n_nan = int(np.isnan(pos).sum())
        if n_nan > 0:
            if step_idx <= 5 or step_idx % 50 == 0:
                print(f"[sim] WARNING: {n_nan}/{n_masses} positions are NaN at step {step_idx} "
                      f"gen={current_gen} phase={current_phase} — physics blew up!", flush=True)
                print(f"[sim]   pos sample: {pos[:3]}", flush=True)

        # NaN/inf → 0 before JSON (NaN is invalid JSON and crashes the browser)
        act_clean = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)
        pos_clean = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        com_clean = np.nan_to_num(com, nan=0.0, posinf=0.0, neginf=0.0)

        with frame_lock:
            latest_frame = {
                "positions":   pos_clean.tolist(),
                "activations": act_clean.tolist(),
                "com":         com_clean.tolist(),
                "fps":         actual_fps,
                "phase":       current_phase,
            }

        elapsed = time.perf_counter() - frame_start
        sleep_t = (1.0 / TARGET_FPS) - elapsed
        if sleep_t > 0.001:
            time.sleep(sleep_t)

        total = time.perf_counter() - frame_start
        if total > 0:
            fps_samples.append(1.0 / total)
        if time.perf_counter() - last_fps_upd >= 0.5:
            if fps_samples:
                actual_fps = sum(fps_samples) / len(fps_samples)
                fps_samples = []
            last_fps_upd = time.perf_counter()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/frame")
def frame():
    import math
    def sanitize(obj):
        if isinstance(obj, float):
            return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):  return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [sanitize(v) for v in obj]
        return obj
    with frame_lock:
        f = dict(latest_frame) if latest_frame else {}
        m = dict(latest_meta)  if latest_meta  else {}
    return jsonify(sanitize({**m, **f}))


@app.route("/set_gen", methods=["POST"])
def set_gen():
    idx = max(0, min(int(request.json["gen_idx"]), len(snapshots) - 1))
    cmd_queue.put(("set_gen", idx))
    return jsonify({"gen_idx": idx})


@app.route("/toggle_pause", methods=["POST"])
def toggle_pause():
    if not hasattr(toggle_pause, "_paused"):
        toggle_pause._paused = False
    toggle_pause._paused = not toggle_pause._paused
    cmd_queue.put(("pause" if toggle_pause._paused else "play",))
    return jsonify({"paused": toggle_pause._paused})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--snapshots", type=str, default="evolution_snapshots")
    parser.add_argument("--config",    type=str, default="config.yaml")
    parser.add_argument("--port",      type=int, default=5000)
    args = parser.parse_args()

    files = sorted(
        glob.glob(os.path.join(args.snapshots, "gen_*.npy")),
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split("_")[1])
    )
    if not files:
        raise FileNotFoundError(
            f"No gen_N.npy files found in '{args.snapshots}/'. "
            "Run afpo_run.py first to generate them."
        )

    snapshots = [np.load(f, allow_pickle=True).item() for f in files]
    _config   = load_config(args.config)

    print(f"Loaded {len(snapshots)} generation snapshots "
          f"(gen 0 -> gen {len(snapshots)-1})")
    print(f"Each generation plays BEFORE TRAINING then AFTER TRAINING.")
    print(f"Visualizer running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop.\n")

    # 1. Run Flask in the background thread (it doesn't need the main thread)
    flask_t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False),
        daemon=True
    )
    flask_t.start()

    # 2. Run the Taichi simulation loop on the main thread
    # (Make sure you still have `import taichi as ti` and `ti.reset()` inside your load_phase function!)
    sim_thread_fn()