"""
Cinematic Montage Visualizer
- Extracts the #1 Best Robot from each generation in lineage.json
- Plays them in a continuous, full-screen auto-advancing loop.
- Perfect for screen-recording video montages.
"""

from flask import Flask, render_template_string
from argparse import ArgumentParser
import json, os

app = Flask(__name__)
_lineage_path = "lineage.json"

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Evolution Montage</title>
<style>
  body, html { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #0d0d1a; font-family: monospace; }
  canvas { display: block; }
  #overlay {
    position: absolute; bottom: 40px; left: 40px; color: #fff; text-shadow: 0 2px 4px rgba(0,0,0,0.8);
    pointer-events: none; transition: opacity 0.3s;
  }
  #overlay h1 { margin: 0; font-size: 48px; font-weight: bold; letter-spacing: 2px; color: #ffcc44; }
  #overlay h2 { margin: 5px 0 0 0; font-size: 24px; color: #aaa; font-weight: normal; }
  #controls-help {
    position: absolute; top: 20px; right: 20px; color: #555; text-align: right; font-size: 14px;
    pointer-events: none; transition: opacity 0.3s;
  }
  .hidden { opacity: 0 !important; }
</style>
</head>
<body>

<div id="overlay">
  <h1 id="title-gen">Generation --</h1>
  <h2 id="subtitle-fit">Fitness: -- | ID: --</h2>
</div>

<div id="controls-help">
  [Space] Pause/Play <br>
  [Right/Left] Next/Prev Robot <br>
  [Up/Down] Adjust Speed <br>
  [R] Reset to Gen 0 <br>
  [H] Hide UI for clean recording
</div>

<canvas id="c"></canvas>

<script>
const robots = {{ robots_json | safe }};
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
let W, H;

let currentIdx = 0;
let animT = 0;
let animFrame = null;
let isPlaying = true;
let speedMult = 0.5; // Default to half-speed for smoother viewing
let uiVisible = true;

// Pre-calculate bounds for the current robot to scale it beautifully to the screen
let minX, maxX, minY, maxY, drawScale, PAD;

function loadRobot(index) {
    if (robots.length === 0) return;
    currentIdx = (index + robots.length) % robots.length;
    animT = 0;

    const r = robots[currentIdx];
    document.getElementById("title-gen").textContent = `Generation ${r.gen}`;
    document.getElementById("subtitle-fit").textContent = `Fitness: ${r.fitness.toFixed(4)}  |  Speed: ${speedMult}x`;

    const positions = r.traj.positions;
    minX = Infinity; maxX = -Infinity; minY = Infinity; maxY = -Infinity;

    for(const frame of positions) {
        for(const [x,y] of frame){
            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
        }
    }

    const spanX = maxX - minX;
    const spanY = maxY - minY;

    // Make it take up 60% of the screen height
    PAD = H * 0.2;
    drawScale = (H - PAD * 2) / (spanY || 1);

    // If it moves really far horizontally, scale it down a bit more so it doesn't fly off screen instantly
    if (spanX * drawScale > W * 0.8) {
        drawScale = (W * 0.8) / spanX;
    }
}

const wx = x => (W / 2) - ((maxX + minX) / 2 * drawScale) + (x * drawScale);
const wy = y => H - PAD - (y - minY) * drawScale;

function animStep() {
    if (!robots.length) return;

    ctx.clearRect(0, 0, W, H);

    // Draw Ground
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 2;
    ctx.beginPath(); 
    ctx.moveTo(0, wy(minY)); 
    ctx.lineTo(W, wy(minY)); 
    ctx.stroke();

    const r = robots[currentIdx];
    const positions = r.traj.positions;
    const springs = r.traj.springs;

    // Clamp to valid frame
    const frameIdx = Math.floor(animT) % positions.length;
    const frame = positions[frameIdx];

    // Draw Springs
    if (springs) {
        ctx.strokeStyle = "rgba(100, 150, 255, 0.8)";
        ctx.lineWidth = 3;
        for (const [a, b] of springs) {
            if (a >= frame.length || b >= frame.length) continue;
            ctx.beginPath();
            ctx.moveTo(wx(frame[a][0]), wy(frame[a][1]));
            ctx.lineTo(wx(frame[b][0]), wy(frame[b][1]));
            ctx.stroke();
        }
    }

    // Draw Masses
    for (const [x, y] of frame) {
        ctx.beginPath();
        ctx.arc(wx(x), wy(y), 6, 0, Math.PI * 2);
        ctx.fillStyle = "#ffcc44";
        ctx.fill();
        ctx.strokeStyle = "#aa7700";
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    if (isPlaying) {
        animT += speedMult;
        // Auto-advance to next robot when this one finishes its trajectory
        if (animT >= positions.length) {
            loadRobot(currentIdx + 1);
        }
    }

    animFrame = requestAnimationFrame(animStep);
}

// ── Keyboard Controls for Screen Recording ──────────────────────────────────
window.addEventListener("keydown", (e) => {
    if (e.code === "Space") {
        isPlaying = !isPlaying;
    } else if (e.code === "ArrowRight") {
        loadRobot(currentIdx + 1);
    } else if (e.code === "ArrowLeft") {
        loadRobot(currentIdx - 1);
    } else if (e.code === "ArrowUp") {
        speedMult = Math.min(10, speedMult + 0.25);
        speedMult = Math.round(speedMult * 100) / 100;
        document.getElementById("subtitle-fit").textContent = `Fitness: ${robots[currentIdx].fitness.toFixed(4)}  |  Speed: ${speedMult}x`;
    } else if (e.code === "ArrowDown") {
        speedMult = Math.max(0.1, speedMult - 0.25);
        speedMult = Math.round(speedMult * 100) / 100;
        document.getElementById("subtitle-fit").textContent = `Fitness: ${robots[currentIdx].fitness.toFixed(4)}  |  Speed: ${speedMult}x`;
    } else if (e.code === "KeyR") {
        // Instantly rewind back to the very first robot (Generation 0)
        loadRobot(0);
    } else if (e.code === "KeyH") {
        uiVisible = !uiVisible;
        document.getElementById("overlay").classList.toggle("hidden", !uiVisible);
        document.getElementById("controls-help").classList.toggle("hidden", !uiVisible);
    }
});

function resize() { 
    W = canvas.width = window.innerWidth; 
    H = canvas.height = window.innerHeight; 
    if (robots.length) loadRobot(currentIdx);
}
window.addEventListener("resize", resize);

resize();
if (animFrame) cancelAnimationFrame(animFrame);
animStep();

</script>
</body>
</html>
"""


@app.route("/")
def index():
    if not os.path.exists(_lineage_path):
        return "<pre>lineage.json not found.</pre>", 404

    with open(_lineage_path) as f:
        data = json.load(f)
    nodes = data.get("nodes", {})

    # 1. Group all nodes by generation
    by_gen = {}
    for nid, n in nodes.items():
        if n.get("traj") and n.get("fitness", -999) > -999:
            gen = n.get("level", n.get("gen", 0))
            by_gen.setdefault(gen, []).append({**n, "id": nid})

    # 2. Extract strictly the #1 Best Robot from each generation
    best_per_gen = []
    for gen in sorted(by_gen.keys()):
        best_node = max(by_gen[gen], key=lambda x: x["fitness"])
        best_per_gen.append(best_node)

    return render_template_string(HTML, robots_json=json.dumps(best_per_gen))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lineage", default="lineage.json")
    parser.add_argument("--port", type=int, default=5002)
    args = parser.parse_args()

    _lineage_path = args.lineage
    print(f"Montage visualizer → http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)