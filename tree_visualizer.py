"""
Family-tree / fractal visualizer
----------------------------------
- Draws one tree per lineage root:
    * All gen-0 individuals
    * Any random-injected individual (parent_id=None, gen>0) that has children
- Full zoom (scroll wheel) + pan (drag) — no scrollbars, fills window.
- Trees are arranged side by side; random-injection roots appear at their
  correct generation height, not at the top.

Usage:
    python tree_visualizer.py
    python tree_visualizer.py --lineage path/to/lineage.json --port 5001
"""

from flask import Flask, render_template_string
from argparse import ArgumentParser
import json, os

app = Flask(__name__)
_lineage_path = "lineage.json"


# ── Tree building ─────────────────────────────────────────────────────────────

def build_forest(nodes):
    """
    Returns a list of tree dicts.

    Root criteria — a node is a root when parent_id is None AND:
      - it is gen 0, OR
      - it has at least one direct child (random injection that produced offspring)

    Each node in the tree has:
      best_child   — most fit direct child in next gen (or None)
      median_child — median-fit direct child in next gen (or None)
    """
    children_of = {}
    for nid, n in nodes.items():
        pid = n.get("parent_id")
        if pid:
            children_of.setdefault(pid, []).append({**n, "id": nid})

    max_gen = max((n["gen"] for n in nodes.values()), default=0)

    roots = []
    for nid, n in nodes.items():
        if n.get("parent_id") is None:
            if n["gen"] == 0 or nid in children_of:
                roots.append({**n, "id": nid})

    def pick_two(pool, used):
        avail = [c for c in pool if c["id"] not in used]
        if not avail:
            return None, None
        s = sorted(avail, key=lambda x: x["fitness"])
        best = s[-1]
        if len(s) == 1:
            return best, None
        mid = s[len(s) // 2]
        if mid["id"] == best["id"]:
            mid = s[max(0, len(s) // 2 - 1)]
        if mid["id"] == best["id"]:
            return best, None
        return best, mid

    def expand(node, used, depth=0):
        result = dict(node)
        result["best_child"]   = None
        result["median_child"] = None
        if node["gen"] >= max_gen or depth > max_gen:
            return result
        pool = children_of.get(node["id"], [])
        best, mid = pick_two(pool, used)
        if best:
            used.add(best["id"])
            result["best_child"] = expand(best, used, depth + 1)
        if mid:
            used.add(mid["id"])
            result["median_child"] = expand(mid, used, depth + 1)
        return result

    forest = []
    for root in roots:
        used = {root["id"]}
        forest.append(expand(root, used))

    # Sort: gen-0 roots by fitness desc, then later-gen roots by gen then fitness
    forest.sort(key=lambda t: (t["gen"], -t["fitness"]))
    return forest


def tree_depth(node):
    if node is None:
        return 0
    return 1 + max(tree_depth(node.get("best_child")),
                   tree_depth(node.get("median_child")))


# ── HTML / JS ─────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Evolution Family Tree</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
html,body { width:100%; height:100%; overflow:hidden; background:#0d0d1a; }
#ui {
  position:fixed; top:10px; left:50%; transform:translateX(-50%);
  display:flex; flex-direction:column; align-items:center; gap:5px;
  z-index:10; pointer-events:none;
}
h2 { font:12px/1 monospace; color:#777; letter-spacing:2px;
     background:#0d0d1acc; padding:3px 10px; border-radius:4px; }
#controls {
  display:flex; gap:8px; pointer-events:all;
  background:#0d0d1aee; padding:5px 14px; border-radius:20px; border:1px solid #333;
}
button { background:#222; color:#ccc; border:1px solid #444;
         border-radius:5px; padding:3px 10px; cursor:pointer; font:12px monospace; }
button:hover { background:#333; }
#zlbl { font:12px monospace; color:#777; line-height:24px; }
#legend { display:flex; gap:14px; font:10px monospace; color:#555;
          background:#0d0d1acc; padding:3px 10px; border-radius:4px; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%;
       margin-right:3px; vertical-align:middle; }
#dbg { font:10px monospace; color:#444; background:#0d0d1acc;
       padding:2px 8px; border-radius:3px; }
canvas { position:fixed; top:0; left:0; cursor:grab; }
canvas.drag { cursor:grabbing; }
#tt { position:fixed; background:#1a1a2e; border:1px solid #555;
      border-radius:6px; padding:7px 11px; font:11px monospace; color:#eee;
      pointer-events:none; display:none; z-index:20; }
</style>
</head>
<body>
<div id="ui">
  <h2>EVOLUTION FAMILY TREE — best child (left) · median child (right)</h2>
  <div id="controls">
    <button id="bfit">Fit All</button>
    <button id="bzin">＋</button>
    <button id="bzout">－</button>
    <span id="zlbl">100%</span>
    <button id="brst">Reset</button>
  </div>
  <div id="legend">
    <span><span class="dot" style="background:#22dd66"></span>High fitness</span>
    <span><span class="dot" style="background:#dd4422"></span>Low fitness</span>
    <span><span class="dot" style="background:#4466ff"></span>Gen-0 root</span>
    <span><span class="dot" style="background:#cc88ff"></span>★ Random-injection root</span>
  </div>
  <div id="dbg">trees: <span id="dbt">?</span> | drawn: <span id="dbn">0</span> | <span id="dbe" style="color:#f88">ok</span></div>
</div>
<canvas id="c"></canvas>
<div id="tt"></div>

<script>
const forest = {{ forest_json | safe }};
const minFit = {{ min_fit }};
const maxFit = {{ max_fit }};

const canvas = document.getElementById("c");
const ctx    = canvas.getContext("2d");
let W, H, tx=0, ty=60, scale=1;

// ── Layout constants ──────────────────────────────────────────────────────────
const NR    = 28;    // node radius (world units)
const MPX   = 3;     // mask pixels per voxel
const VGAP  = 110;   // vertical gap between levels
const HPAD  = 60;    // gap between sibling subtrees
const TGAP  = 80;    // gap between separate trees

// For random-injection roots at gen>0, we pad the top so they appear
// at the correct vertical level matching their generation.
const GEN_Y_OFFSET = (gen) => gen * VGAP;  // world-y of a root's circle

// ── Colour ────────────────────────────────────────────────────────────────────
function fc(fitness, alpha) {
  if (fitness <= -999) return alpha!=null ? `rgba(85,85,85,${alpha})` : "#555";
  const t = Math.max(0, Math.min(1, (fitness-minFit)/(maxFit-minFit+1e-9)));
  const r = Math.round(220*(1-t)+30*t), g = Math.round(60*(1-t)+210*t), b=60;
  return alpha!=null ? `rgba(${r},${g},${b},${alpha})` : `rgb(${r},${g},${b})`;
}

// ── Layout ────────────────────────────────────────────────────────────────────
function subW(node) {
  if (!node) return 0;
  const lw=subW(node.best_child), rw=subW(node.median_child);
  return Math.max(NR*2+4, lw+rw+(lw&&rw?HPAD:0));
}

// yTop = world-y of the TOP of the node's circle row
function assignXY(node, xc, yTop) {
  if (!node) return;
  node._x = xc;
  node._y = yTop + NR;
  const lw=subW(node.best_child), rw=subW(node.median_child);
  const ny = yTop + VGAP;
  if (node.best_child && node.median_child) {
    const tot = lw+rw+HPAD;
    assignXY(node.best_child,   xc - tot/2 + lw/2, ny);
    assignXY(node.median_child, xc + tot/2 - rw/2, ny);
  } else if (node.best_child)   { assignXY(node.best_child,   xc, ny); }
    else if (node.median_child) { assignXY(node.median_child, xc, ny); }
}

// Layout forest: trees side by side, each root at its correct gen height
let xCursor = TGAP;
for (const tree of forest) {
  const w   = subW(tree);
  const yTop = GEN_Y_OFFSET(tree.gen);   // root starts at its generation's row
  assignXY(tree, xCursor + w/2, yTop);
  xCursor += w + TGAP;
}

// ── Bounding box ──────────────────────────────────────────────────────────────
function bbox(node, b) {
  if (!node) return;
  b.x0=Math.min(b.x0,node._x-NR-30); b.x1=Math.max(b.x1,node._x+NR+30);
  b.y0=Math.min(b.y0,node._y-NR-20); b.y1=Math.max(b.y1,node._y+NR+30);
  bbox(node.best_child,b); bbox(node.median_child,b);
}
function forestBBox() {
  const b={x0:Infinity,x1:-Infinity,y0:Infinity,y1:-Infinity};
  for (const t of forest) bbox(t,b); return b;
}

// ── Fit all ───────────────────────────────────────────────────────────────────
function fitAll() {
  const b=forestBBox(); if(!isFinite(b.x0)) return;
  const cw=b.x1-b.x0, ch=b.y1-b.y0;
  scale=Math.min(W/cw,H/ch)*0.92;
  tx=W/2-(b.x0+cw/2)*scale; ty=H/2-(b.y0+ch/2)*scale;
  render();
}

// ── Screen helpers ────────────────────────────────────────────────────────────
const sx=x=>x*scale+tx, sy=y=>y*scale+ty, sr=r=>r*scale;

// ── Draw ──────────────────────────────────────────────────────────────────────
function drawEdge(p, c, lbl) {
  const px=sx(p._x),py=sy(p._y),cx2=sx(c._x),cy2=sy(c._y);
  const g=ctx.createLinearGradient(px,py,cx2,cy2);
  g.addColorStop(0,fc(p.fitness,0.55)); g.addColorStop(1,fc(c.fitness,0.55));
  ctx.beginPath(); ctx.moveTo(px,py);
  ctx.quadraticCurveTo((px+cx2)/2, py+20*scale, cx2,cy2);
  ctx.strokeStyle=g; ctx.lineWidth=Math.max(1,1.4*scale);
  ctx.setLineDash([4*scale,4*scale]); ctx.stroke(); ctx.setLineDash([]);
  if (scale>0.45) {
    ctx.fillStyle="#555"; ctx.font=`${Math.round(9*scale)}px monospace`;
    ctx.textAlign="center";
    ctx.fillText(lbl, (px+cx2)/2+(c._x<p._x?-8*scale:8*scale), py+24*scale);
  }
}

function drawMask(cx2,cy2,mask) {
  if (!mask) return;
  const dim=mask.length, px=MPX*scale, off=dim*px/2;
  for (let r=0;r<dim;r++) for (let c=0;c<dim;c++)
    if (mask[r][c]) { ctx.fillStyle="rgba(255,215,80,0.85)";
      ctx.fillRect(cx2-off+c*px, cy2-off+r*px, px-0.5, px-0.5); }
}

let drawn=0;
function drawNode(node, isRoot) {
  drawn++;
  const x=sx(node._x), y=sy(node._y), r=sr(NR);
  const isRandRoot = node.gen>0 && !node.parent_id;

  // Glow
  const gl=ctx.createRadialGradient(x,y,r*0.1,x,y,r+8*scale);
  gl.addColorStop(0,fc(node.fitness,0.3)); gl.addColorStop(1,"transparent");
  ctx.beginPath(); ctx.arc(x,y,r+8*scale,0,Math.PI*2); ctx.fillStyle=gl; ctx.fill();

  // Body
  ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2);
  ctx.fillStyle="#0d0d1a"; ctx.fill();
  ctx.strokeStyle = isRoot&&node.gen===0 ? "#4466ff" : isRandRoot ? "#cc88ff" : fc(node.fitness);
  ctx.lineWidth = (isRoot?2.5:1.5)*scale; ctx.stroke();

  drawMask(x, y-4*scale, node.mask);

  if (scale>0.32) {
    ctx.fillStyle="#aaa"; ctx.font=`${Math.round(10*scale)}px monospace`;
    ctx.textAlign="center";
    ctx.fillText(node.fitness.toFixed(3), x, y+r+12*scale);
    ctx.fillStyle="#555"; ctx.font=`${Math.round(9*scale)}px monospace`;
    ctx.fillText("gen"+node.gen+(isRandRoot?" ★":""), x, y+r+22*scale);
  }
}

function drawSubtree(node, isRoot) {
  if (!node) return;
  if (node.best_child)   { drawEdge(node,node.best_child,"best");   drawSubtree(node.best_child,false); }
  if (node.median_child) { drawEdge(node,node.median_child,"med");  drawSubtree(node.median_child,false); }
  drawNode(node, isRoot);
}

function render() {
  drawn=0;
  ctx.clearRect(0,0,W,H);
  try {
    for (const t of forest) drawSubtree(t, true);
    document.getElementById("dbn").textContent=drawn;
    document.getElementById("dbe").textContent="ok";
  } catch(e) {
    document.getElementById("dbe").textContent=e.toString(); console.error(e);
  }
  document.getElementById("zlbl").textContent=Math.round(scale*100)+"%";
}

// ── Resize ────────────────────────────────────────────────────────────────────
function resize() { W=canvas.width=window.innerWidth; H=canvas.height=window.innerHeight; render(); }
window.addEventListener("resize", resize);

// ── Scroll to zoom ────────────────────────────────────────────────────────────
canvas.addEventListener("wheel", e=>{
  e.preventDefault();
  const f=e.deltaY<0?1.1:0.91;
  tx=e.clientX-(e.clientX-tx)*f; ty=e.clientY-(e.clientY-ty)*f; scale*=f; render();
},{passive:false});

// ── Drag to pan ───────────────────────────────────────────────────────────────
let drag=false,dx0=0,dy0=0;
canvas.addEventListener("mousedown",e=>{drag=true;dx0=e.clientX;dy0=e.clientY;canvas.classList.add("drag");});
window.addEventListener("mousemove",e=>{if(!drag)return;tx+=e.clientX-dx0;ty+=e.clientY-dy0;dx0=e.clientX;dy0=e.clientY;render();});
window.addEventListener("mouseup",()=>{drag=false;canvas.classList.remove("drag");});

// ── Buttons ───────────────────────────────────────────────────────────────────
document.getElementById("bfit").onclick  = fitAll;
document.getElementById("bzin").onclick  = ()=>{scale*=1.2;render();};
document.getElementById("bzout").onclick = ()=>{scale*=0.83;render();};
document.getElementById("brst").onclick  = ()=>{scale=1;tx=0;ty=60;render();};

// ── Tooltip ───────────────────────────────────────────────────────────────────
const tt=document.getElementById("tt");
const allN=[];
function flat(n){if(!n)return;allN.push(n);flat(n.best_child);flat(n.median_child);}
for(const t of forest)flat(t);

canvas.addEventListener("mousemove",e=>{
  const mx=(e.clientX-tx)/scale, my=(e.clientY-ty)/scale;
  let hit=null;
  for(const n of allN){const dx=n._x-mx,dy=n._y-my;if(dx*dx+dy*dy<NR*NR){hit=n;break;}}
  if(hit){
    tt.style.display="block";tt.style.left=(e.clientX+14)+"px";tt.style.top=(e.clientY+14)+"px";
    const rand=hit.gen>0&&!hit.parent_id;
    tt.innerHTML=`<b>Gen ${hit.gen}${rand?" ★ random root":""}</b><br>Fitness: ${hit.fitness.toFixed(4)}<br>ID: ${hit.id}`;
  } else { tt.style.display="none"; }
});

// ── Start ─────────────────────────────────────────────────────────────────────
document.getElementById("dbt").textContent=forest.length;
resize(); fitAll();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    print(f"[srv] GET / — {os.path.abspath(_lineage_path)}", flush=True)
    if not os.path.exists(_lineage_path):
        return f"<pre>lineage.json not found.\nRun afpo_run.py first.</pre>", 404

    with open(_lineage_path) as f:
        data = json.load(f)
    nodes = data.get("nodes", {})
    print(f"[srv] nodes: {len(nodes)}", flush=True)

    by_gen = {}
    for nid, n in nodes.items():
        by_gen.setdefault(n["gen"], []).append(n)
    for g in sorted(by_gen):
        fits = [round(n["fitness"],3) for n in by_gen[g]]
        print(f"[srv]   gen {g}: {len(by_gen[g])} nodes  fits={fits}", flush=True)

    forest = build_forest(nodes)
    print(f"[srv] forest: {len(forest)} trees", flush=True)
    for i, t in enumerate(forest):
        print(f"[srv]   tree {i}: root=gen{t['gen']} fit={t['fitness']:.3f} depth={tree_depth(t)}", flush=True)

    all_fits = [n["fitness"] for n in nodes.values() if n["fitness"] > -999]
    min_fit  = min(all_fits) if all_fits else 0.0
    max_fit  = max(all_fits) if all_fits else 1.0

    forest_json = json.dumps(forest)
    print(f"[srv] forest JSON: {len(forest_json)} bytes", flush=True)

    return render_template_string(HTML,
        forest_json=forest_json, min_fit=min_fit, max_fit=max_fit)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lineage", type=str, default="lineage.json")
    parser.add_argument("--port",    type=int, default=5001)
    args = parser.parse_args()
    _lineage_path = args.lineage
    print(f"Tree visualizer at http://localhost:{args.port}")
    print(f"Lineage: {_lineage_path}")
    print("Refresh page to reload after re-running afpo_run.py\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)