"""
Evolution Family Tree Visualizer
- Light / dark mode toggle
- One root per lineage (gen-0 only if it produced children; random injections
  only if they produced children). Lone dead-ends are hidden.
- Each node plays an animated simulation of the robot on hover.
- Interactive Evolution: Click nodes to select them. Select 1 to mutate,
  select 2 to splice (crossover). Evaluates live using Taichi.
"""

from flask import Flask, render_template_string, request, jsonify
from argparse import ArgumentParser
import json, os

app = Flask(__name__)
_lineage_path = "lineage.json"


# ── Tree building ─────────────────────────────────────────────────────────────

def build_forest(nodes):
    children_of = {}
    for nid, n in nodes.items():
        pid = n.get("parent_id")
        if pid:
            children_of.setdefault(pid, []).append({**n, "id": nid})

    def lv(n): return n.get("level", n.get("gen", 0))
    max_gen = max((lv(n) for n in nodes.values()), default=0)
    by_gen  = {}
    for nid, n in nodes.items():
        by_gen.setdefault(lv(n), []).append({**n, "id": nid, "_level": lv(n)})

    # Multi-root: best gen-0 node + any random injection that produced children
    gen0 = [{**n, "id": nid, "_level": 0} for nid, n in nodes.items() if lv(n) == 0]
    if not gen0:
        return []
    gen0_with_kids = [n for n in gen0 if n["id"] in children_of]
    root_pool = gen0_with_kids if gen0_with_kids else gen0
    best_gen0 = max(root_pool, key=lambda x: x["fitness"])

    # Random roots and Spliced children (which have parent_id = None but use splice_parents)
    random_roots = [
        {**n, "id": nid, "_level": lv(n)} for nid, n in nodes.items()
        if n.get("parent_id") is None and lv(n) > 0
    ]

    roots = [best_gen0] + sorted(random_roots, key=lambda x: (x["gen"], -x["fitness"]))

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
        result["best_child"] = None
        result["median_child"] = None
        node_lv = node.get("_level", node.get("gen", 0))
        if node_lv >= max_gen or depth > max_gen:
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
    gen0_root = roots[0]
    gen0_used = {gen0_root["id"]}
    gen0_tree = expand(gen0_root, gen0_used)
    forest.append(gen0_tree)

    for root in roots[1:]:
        used = gen0_used | {root["id"]}
        tree = expand(root, used)
        forest.append(tree)

    # ── Explicit Sibling Links via competitor_id ─────────────────────────────
    unique_pairs = []
    competitions = {}

    for nid, n in nodes.items():
        cid = n.get("competitor_id")
        if cid is not None:
            competitions.setdefault(cid, []).append(nid)

    for cid, members in competitions.items():
        if len(members) >= 2:
            unique_pairs.append([members[0], members[1]])

    forest.sort(key=lambda t: (t["gen"], -t["fitness"]))
    return forest, unique_pairs


def tree_depth(node):
    if node is None: return 0
    return 1 + max(tree_depth(node.get("best_child")), tree_depth(node.get("median_child")))


# ── API Endpoint for Interactive Evolution ────────────────────────────────────

@app.route("/api/evolve", methods=["POST"])
def api_evolve():
    data = request.json
    ids = data.get("ids", [])
    if not ids:
        return jsonify({"error": "No nodes selected"}), 400

    try:
        import taichi as ti
        ti.reset()  # Reset taichi context for the Flask background thread!

        from afpo_run import mutate, evaluate_batch
        from robot import build_robot
        from utils import load_config
        import numpy as np
        import uuid

        config_path = app.config.get("CONFIG_PATH", "config.yaml")
        config = load_config(config_path)

        with open(_lineage_path, "r") as f:
            lineage = json.load(f)

        new_child = None
        gen_level = 0

        if len(ids) == 1:
            # Single Mutation
            pid = ids[0]
            p_data = lineage["nodes"][pid]
            parent_robot = {
                "mask": np.array(p_data["mask"]),
                "p": p_data.get("p", 0.55)
            }
            new_child = mutate(parent_robot)
            new_child["id"] = f"n{uuid.uuid4().hex[:8]}"
            new_child["parent_id"] = pid
            new_child["splice_parents"] = None
            gen_level = p_data.get("level", p_data.get("gen", 0)) + 1

        elif len(ids) == 2:
            # Splicing (Crossover)
            p1_data = lineage["nodes"][ids[0]]
            p2_data = lineage["nodes"][ids[1]]

            m1 = np.array(p1_data["mask"])
            m2 = np.array(p2_data["mask"])

            # Spatial Splice: Left half of Parent 1, Right half of Parent 2
            new_mask = np.hstack((m1[:, :4], m2[:, 4:]))

            # Clean up mask (extract the largest connected component)
            from scipy import ndimage
            labeled, n = ndimage.label(new_mask)
            if n > 0:
                sizes = ndimage.sum(new_mask, labeled, range(1, n + 1))
                new_mask = (labeled == int(np.argmax(sizes)) + 1).astype(int)
            else:
                new_mask = m1 # Fallback if crossover destroyed all connectivity

            # Average the density parameter
            p = (p1_data.get("p", 0.55) + p2_data.get("p", 0.55)) / 2.0

            new_child = build_robot(new_mask, p)
            new_child["id"] = f"n{uuid.uuid4().hex[:8]}"

            # By setting parent_id to None, it renders as a new tree.
            new_child["parent_id"] = None
            # We save the true parents here so the JS canvas can draw the DAG lines
            new_child["splice_parents"] = [ids[0], ids[1]]
            gen_level = max(p1_data.get("level", 0), p2_data.get("level", 0)) + 1

        # Train and Evaluate the new child
        new_child["age"] = 0
        config["simulator"]["n_sims"] = 1
        fitnesses, ctrl, max_m, max_s, trajs = evaluate_batch([new_child], config)

        fit = float(fitnesses[0])
        if np.isnan(fit): fit = -9999.0

        # --- ADD THESE DEBUG PRINTS ---
        print("\n" + "=" * 50)
        print(f"[DEBUG] API /evolve triggered")
        print(f"[DEBUG] Selected Parent(s): {ids}")
        print(f"[DEBUG] New Child ID:     {new_child['id']}")
        print(f"[DEBUG] Child Fitness:    {fit:.4f}")
        if len(ids) == 1:
            print(f"[DEBUG] Connection Type:  Mutation")
            print(f"[DEBUG] Tree Link:        {ids[0]} ──> {new_child['id']}")
        elif len(ids) == 2:
            print(f"[DEBUG] Connection Type:  Splice (Crossover)")
            print(f"[DEBUG] DAG Links:        {ids[0]} ──> {new_child['id']}")
            print(f"[DEBUG]                   {ids[1]} ──> {new_child['id']}")
        print("=" * 50 + "\n")
        # ------------------------------

        lineage["nodes"][new_child["id"]] = {
            "id": new_child["id"],
            "gen": gen_level,
            "level": gen_level,
            "fitness": fit,
            "parent_id": new_child.get("parent_id"),
            "splice_parents": new_child.get("splice_parents"),
            "competitor_id": None,
            "mask": new_child["mask"].tolist(),
            "p": new_child["p"],
            "traj": trajs[0] if trajs else None
        }

        # Save back to JSON
        with open(_lineage_path, "w") as f:
            json.dump(lineage, f)

        return jsonify({"success": True})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html data-theme="light">
<head>
<meta charset="utf-8">
<title>Evolution Family Tree</title>
<style>
:root {
  --bg:       #0d0d1a;
  --node-bg:  #0d0d1a;
  --ui-bg:    rgba(13,13,26,0.92);
  --border:   #333;
  --text:     #ccc;
  --muted:    #555;
  --edge:     0.55;
  --btn-bg:   #222;
  --btn-bdr:  #444;
}
[data-theme="light"] {
  --bg:       #f0f0f5;
  --node-bg:  #ffffff;
  --ui-bg:    rgba(240,240,245,0.95);
  --border:   #bbb;
  --text:     #222;
  --muted:    #888;
  --edge:     0.7;
  --btn-bg:   #e8e8ef;
  --btn-bdr:  #bbb;
}
* { box-sizing:border-box; margin:0; padding:0; transition: background 0.25s, color 0.25s; }
html,body { width:100%; height:100%; overflow:hidden; background:var(--bg); }
#ui {
  position:fixed; top:10px; left:50%; transform:translateX(-50%);
  display:flex; flex-direction:column; align-items:center; gap:5px;
  z-index:10; pointer-events:none;
}
h2 { font:12px/1 monospace; color:var(--muted); letter-spacing:2px;
     background:var(--ui-bg); padding:4px 12px; border-radius:4px; }
#controls {
  display:flex; gap:7px; align-items:center; pointer-events:all;
  background:var(--ui-bg); padding:5px 14px; border-radius:20px;
  border:1px solid var(--border);
}
button {
  background:var(--btn-bg); color:var(--text); border:1px solid var(--btn-bdr);
  border-radius:5px; padding:3px 10px; cursor:pointer; font:12px monospace;
}
button:hover { opacity:0.8; }
button:disabled { opacity:0.5; cursor:not-allowed; }
#bevolve { background:#28a745; color:white; font-weight:bold; border:none; }
#zlbl { font:12px monospace; color:var(--muted); line-height:24px; min-width:42px; text-align:center; }
#theme-btn { font-size:15px; padding:2px 8px; }
#legend {
  display:flex; gap:14px; font:10px monospace; color:var(--muted);
  background:var(--ui-bg); padding:3px 12px; border-radius:4px; border:1px solid var(--border);
}
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px; vertical-align:middle; }
#dbg { font:10px monospace; color:var(--muted); background:var(--ui-bg); padding:2px 8px; border-radius:3px; }

canvas { position:fixed; top:0; left:0; cursor:grab; }
canvas.drag { cursor:grabbing; }

/* Popup animation panel */
#popup {
  position:fixed; display:none; flex-direction:column; gap:0;
  background:var(--ui-bg); border:1px solid var(--border);
  border-radius:10px; padding:10px; z-index:30; min-width:420px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  pointer-events:none;
}
#popup-title { font:11px monospace; color:var(--muted); margin-bottom:6px; text-align:center; }
#anim-canvas { border-radius:6px; border:1px solid var(--border); display:block; }
#popup-fitness { font:11px monospace; color:var(--text); text-align:center; margin-top:5px; }
</style>
</head>
<body>
<div id="ui">
  <h2>EVOLUTION FAMILY TREE — Click nodes to select & evolve!</h2>
  <div id="controls">
    <button id="bfit">Fit All</button>
    <button id="bzin">＋</button>
    <button id="bzout">－</button>
    <span id="zlbl">100%</span>
    <button id="brst">Reset</button>
    <button id="bsib" title="Toggle sibling links" style="opacity:0.35">✕ siblings</button>
    <button id="theme-btn" title="Toggle dark/light">🌙</button>
    <button id="bevolve" disabled style="opacity:0.5">Evolve</button>
  </div>
  <div id="legend">
    <span><span class="dot" style="background:#22dd66"></span>High fitness</span>
    <span><span class="dot" style="background:#dd4422"></span>Low fitness</span>
    <span><span class="dot" style="background:#4466ff"></span>Gen-0 root</span>
    <span><span class="dot" style="background:#cc88ff"></span>★ Random root</span>
    <span><span style="color:rgba(180,180,255,0.8)">✕--✕--✕</span> Sibling link</span>
    <span><span style="color:rgba(255,100,200,0.8)">⤎--⤏</span> Splice link</span>
    <span style="color:var(--muted)">Hover to animate</span>
  </div>
  <div id="dbg">trees:<span id="dbt">?</span> nodes:<span id="dbn">0</span> <span id="dbe"></span></div>
</div>

<canvas id="c"></canvas>

<div id="popup">
  <div id="popup-title">—</div>
  <canvas id="anim-canvas" width="400" height="320"></canvas>
  <div id="popup-fitness">—</div>
</div>

<script>
const forest = {{ forest_json | safe }};
const siblingPairs = {{ sibling_pairs_json | safe }};
const minFit = {{ min_fit }};
const maxFit = {{ max_fit }};
const isDark = () => document.documentElement.getAttribute("data-theme") === "dark";

const canvas = document.getElementById("c");
const ctx    = canvas.getContext("2d");
let W, H, tx=0, ty=60, scale=1;
let selectedNodes = [];

// ── Layout ────────────────────────────────────────────────────────────────────
const NR=28, MPX=3, VGAP=110, HPAD=60, TGAP=80;

function subW(n) {
  if (!n) return 0;
  const lw=subW(n.best_child), rw=subW(n.median_child);
  return Math.max(NR*2+4, lw+rw+(lw&&rw?HPAD:0));
}

function assignXY(n, xc) {
  if (!n) return;
  n._x=xc;
  
  // Force the Y-coordinate to strictly match the node's true generation level
  const trueLevel = n._level !== undefined ? n._level : n.gen;
  n._y = (trueLevel * VGAP) + NR;
  
  const lw=subW(n.best_child), rw=subW(n.median_child);
  if (n.best_child && n.median_child) {
    const tot=lw+rw+HPAD;
    assignXY(n.best_child,   xc-tot/2+lw/2);
    assignXY(n.median_child, xc+tot/2-rw/2);
  } else if (n.best_child)   assignXY(n.best_child,   xc);
    else if (n.median_child) assignXY(n.median_child, xc);
}

// Split into normal biological trees and interactive spliced trees
const normalTrees = forest.filter(t => !t.splice_parents);
const splicedTrees = forest.filter(t => t.splice_parents);

// Sort spliced trees by generation so older splices are positioned first
splicedTrees.sort((a, b) => (a.gen || 0) - (b.gen || 0));

let xCur=TGAP;
const byIdLayout = {};

function indexTreeLayout(n) {
  if (!n) return;
  byIdLayout[n.id] = n;
  indexTreeLayout(n.best_child);
  indexTreeLayout(n.median_child);
}

// 1. Layout standard biological trees sequentially (left to right)
for (const t of normalTrees) {
  const w=subW(t);
  assignXY(t, xCur+w/2);
  indexTreeLayout(t);
  xCur+=w+TGAP;
}

// 2. Layout spliced trees explicitly between their parents
for (const t of splicedTrees) {
  const p1 = byIdLayout[t.splice_parents[0]];
  const p2 = byIdLayout[t.splice_parents[1]];
  
  let startX = xCur; // Fallback if parents are missing
  if (p1 && p2) {
      startX = (p1._x + p2._x) / 2; // Exact mathematical midpoint
  } else if (p1) {
      startX = p1._x;
  } else if (p2) {
      startX = p2._x;
  }

  assignXY(t, startX);
  indexTreeLayout(t); // Index it in case another splice uses THIS splice as a parent!
}

// ── Colour ────────────────────────────────────────────────────────────────────
function fc(fitness, alpha) {
  if (fitness<=-999) return alpha!=null?`rgba(120,120,120,${alpha})`:"#777";
  const t=Math.max(0,Math.min(1,(fitness-minFit)/(maxFit-minFit+1e-9)));
  const r=Math.round(220*(1-t)+30*t), g=Math.round(60*(1-t)+210*t), b=60;
  return alpha!=null?`rgba(${r},${g},${b},${alpha})`:`rgb(${r},${g},${b})`;
}

// ── Bounding box + fit ────────────────────────────────────────────────────────
function bbox(n,b){
  if(!n)return;
  b.x0=Math.min(b.x0,n._x-NR-30); b.x1=Math.max(b.x1,n._x+NR+30);
  b.y0=Math.min(b.y0,n._y-NR-20); b.y1=Math.max(b.y1,n._y+NR+30);
  bbox(n.best_child,b); bbox(n.median_child,b);
}
function fitAll(){
  const b={x0:Infinity,x1:-Infinity,y0:Infinity,y1:-Infinity};
  for(const t of forest) bbox(t,b);
  if(!isFinite(b.x0)) return;
  const cw=b.x1-b.x0, ch=b.y1-b.y0;
  scale=Math.min(W/cw,H/ch)*0.92;
  tx=W/2-(b.x0+cw/2)*scale; ty=H/2-(b.y0+ch/2)*scale;
  render();
}

// ── Screen helpers ────────────────────────────────────────────────────────────
const sx=x=>x*scale+tx, sy=y=>y*scale+ty, sr=r=>r*scale;

// ── Draw tree ─────────────────────────────────────────────────────────────────
function drawEdge(p,c,lbl){
  const px=sx(p._x),py=sy(p._y),cx2=sx(c._x),cy2=sy(c._y);
  const g=ctx.createLinearGradient(px,py,cx2,cy2);
  const alpha = isDark() ? 0.55 : 0.7;
  g.addColorStop(0,fc(p.fitness,alpha)); g.addColorStop(1,fc(c.fitness,alpha));
  ctx.beginPath(); ctx.moveTo(px,py);
  ctx.quadraticCurveTo((px+cx2)/2, py+20*scale, cx2,cy2);
  ctx.strokeStyle=g; ctx.lineWidth=Math.max(1,1.4*scale);
  ctx.setLineDash([4*scale,4*scale]); ctx.stroke(); ctx.setLineDash([]);
  if(scale>0.45){
    ctx.fillStyle=isDark()?"#555":"#999"; ctx.font=`${Math.round(9*scale)}px monospace`;
    ctx.textAlign="center";
    ctx.fillText(lbl,(px+cx2)/2+(c._x<p._x?-8*scale:8*scale),py+24*scale);
  }
}

function drawMask(cx2,cy2,mask){
  if(!mask)return;
  const dim=mask.length, px=MPX*scale, off=dim*px/2;
  for(let r=0;r<dim;r++) for(let c=0;c<dim;c++)
    if(mask[r][c]){ ctx.fillStyle="rgba(255,200,60,0.85)";
      ctx.fillRect(cx2-off+c*px,cy2-off+r*px,px-0.5,px-0.5); }
}

let drawn=0;
function drawNode(n, isRoot){
  drawn++;
  const x=sx(n._x),y=sy(n._y),r=sr(NR);
  const nodeLevel = n._level !== undefined ? n._level : n.gen;
  const isRand = nodeLevel > 0 && !n.parent_id && !n.splice_parents, dark=isDark();
  
  // Selection Highlight Ring
  if (selectedNodes.includes(n.id)) {
    ctx.beginPath(); ctx.arc(x,y,r+10*scale,0,Math.PI*2);
    ctx.strokeStyle = "#00ffff"; ctx.lineWidth=3*scale; ctx.stroke();
  }

  // glow
  const gl=ctx.createRadialGradient(x,y,r*0.1,x,y,r+8*scale);
  gl.addColorStop(0,fc(n.fitness,dark?0.3:0.15)); gl.addColorStop(1,"transparent");
  ctx.beginPath(); ctx.arc(x,y,r+8*scale,0,Math.PI*2); ctx.fillStyle=gl; ctx.fill();
  
  // circle body
  ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2);
  ctx.fillStyle=dark?"#0d0d1a":"#ffffff"; ctx.fill();
  ctx.strokeStyle=isRoot&&n.gen===0?"#4466ff":isRand?"#cc88ff":fc(n.fitness);
  ctx.lineWidth=(isRoot?2.5:1.5)*scale; ctx.stroke();
  drawMask(x,y-4*scale,n.mask);
  
  if(scale>0.32){
    ctx.fillStyle=dark?"#aaa":"#333"; ctx.font=`${Math.round(10*scale)}px monospace`;
    ctx.textAlign="center";
    ctx.fillText(n.fitness.toFixed(3),x,y+r+12*scale);
    ctx.fillStyle=dark?"#555":"#999"; ctx.font=`${Math.round(9*scale)}px monospace`;
    const isLeaf = !n.best_child && !n.median_child;
    const leafMark = isLeaf && nodeLevel < {{ max_gen }} ? " ✕" : "";
    const spliceMark = n.splice_parents ? " ⚗️" : "";
    ctx.fillText("lvl"+nodeLevel+(isRand?" ★rnd":"")+leafMark+spliceMark,x,y+r+22*scale);
  }
}

function drawSiblingLink(a, b) {
  const ax=sx(a._x), ay=sy(a._y), bx=sx(b._x), by=sy(b._y);
  const mx=(ax+bx)/2, my=(ay+by)/2;
  const dark=isDark();

  ctx.save();
  ctx.setLineDash([5*scale, 4*scale]);
  ctx.strokeStyle = dark ? "rgba(180,180,255,0.45)" : "rgba(80,80,180,0.45)";
  ctx.lineWidth = Math.max(1, scale);
  ctx.beginPath();
  ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  const xSize = Math.max(4, 5*scale);
  const col = dark ? "rgba(180,180,255,0.7)" : "rgba(80,80,200,0.7)";
  for (const [px2,py2] of [[ax,ay],[mx,my],[bx,by]]) {
    ctx.save();
    ctx.strokeStyle=col; ctx.lineWidth=Math.max(1,1.2*scale);
    ctx.beginPath(); ctx.moveTo(px2-xSize,py2-xSize); ctx.lineTo(px2+xSize,py2+xSize); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(px2+xSize,py2-xSize); ctx.lineTo(px2-xSize,py2+xSize); ctx.stroke();
    ctx.restore();
  }

  if (scale > 0.5) {
    ctx.fillStyle = dark ? "rgba(180,180,255,0.6)" : "rgba(80,80,200,0.6)";
    ctx.font = `${Math.round(9*scale)}px monospace`;
    ctx.textAlign = "center";
    ctx.fillText("siblings", mx, my - 8*scale);
  }
}

function drawSpliceLinks() {
  const byId = {};
  for (const n of allN) byId[n.id] = n;
  
  for (const n of allN) {
    if (n.splice_parents && n.splice_parents.length === 2) {
      const p1 = byId[n.splice_parents[0]];
      const p2 = byId[n.splice_parents[1]];
      const cx = sx(n._x), cy = sy(n._y);
      
      ctx.lineWidth = 2 * scale;
      ctx.setLineDash([8 * scale, 6 * scale]);
      
      if (p1) {
        const p1x = sx(p1._x), p1y = sy(p1._y);
        ctx.beginPath(); ctx.moveTo(p1x, p1y);
        ctx.bezierCurveTo(p1x, p1y + VGAP/1.5 * scale, cx, cy - VGAP/1.5 * scale, cx, cy);
        ctx.strokeStyle = "rgba(255, 100, 200, 0.7)"; // Pink line for Parent 1
        ctx.stroke();
      }
      if (p2) {
        const p2x = sx(p2._x), p2y = sy(p2._y);
        ctx.beginPath(); ctx.moveTo(p2x, p2y);
        ctx.bezierCurveTo(p2x, p2y + VGAP/1.5 * scale, cx, cy - VGAP/1.5 * scale, cx, cy);
        ctx.strokeStyle = "rgba(100, 200, 255, 0.7)"; // Cyan line for Parent 2
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
  }
}

function drawSubtree(n,isRoot){
  if(!n)return;
  if(n.best_child)  { drawEdge(n,n.best_child,"best");   drawSubtree(n.best_child,false); }
  if(n.median_child){ drawEdge(n,n.median_child,"med");  drawSubtree(n.median_child,false); }
  drawNode(n,isRoot);
}

function drawCrossGenSiblingLinks() {
  const byId = {};
  for (const n of allN) byId[n.id] = n;
  for (const [idA, idB] of siblingPairs) {
    const a = byId[idA], b = byId[idB];
    if (a && b) drawSiblingLink(a, b);
  }
}

function render(){
  drawn=0; ctx.clearRect(0,0,W,H);
  try{
    // Draw splice DAG lines BEHIND the trees
    drawSpliceLinks();

    for(const t of forest) drawSubtree(t,true);
    if (showSiblings) drawCrossGenSiblingLinks();
    
    document.getElementById("dbn").textContent=drawn;
    document.getElementById("dbe").textContent="";
  }catch(e){ document.getElementById("dbe").textContent=e; console.error(e); }
  document.getElementById("zlbl").textContent=Math.round(scale*100)+"%";
}

// ── Selection & API Calls ─────────────────────────────────────────────────────
function updateEvolveBtn() {
    const btn = document.getElementById("bevolve");
    if (selectedNodes.length === 0) {
        btn.textContent = "Evolve";
        btn.style.opacity = 0.5;
        btn.disabled = true;
    } else if (selectedNodes.length === 1) {
        btn.textContent = "Evolve (1)";
        btn.style.opacity = 1;
        btn.disabled = false;
    } else {
        btn.textContent = "Splice (2)";
        btn.style.opacity = 1;
        btn.disabled = false;
    }
}

document.getElementById("bevolve").onclick = async () => {
    if (selectedNodes.length === 0) return;
    const btn = document.getElementById("bevolve");
    const origText = btn.textContent;
    
    btn.textContent = "Simulating...";
    btn.style.opacity = 0.5;
    btn.disabled = true;
    
    try {
        const res = await fetch("/api/evolve", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ ids: selectedNodes })
        });
        if (res.ok) {
            window.location.reload(); // Hard refresh to instantly redraw the updated JSON tree
        } else {
            const err = await res.json();
            alert("Evolution failed: " + (err.error || "Unknown Error"));
            btn.textContent = origText;
            btn.style.opacity = 1;
            btn.disabled = false;
        }
    } catch(e) {
        alert("Request failed: " + e);
        btn.textContent = origText;
        btn.style.opacity = 1;
        btn.disabled = false;
    }
};

// ── Resize ────────────────────────────────────────────────────────────────────
function resize(){ W=canvas.width=window.innerWidth; H=canvas.height=window.innerHeight; render(); }
window.addEventListener("resize",resize);

// ── Zoom + pan + Selection ────────────────────────────────────────────────────
canvas.addEventListener("wheel",e=>{
  e.preventDefault();
  const f=e.deltaY<0?1.1:0.91;
  tx=e.clientX-(e.clientX-tx)*f; ty=e.clientY-(e.clientY-ty)*f; scale*=f; render();
},{passive:false});

let drag=false,dx0=0,dy0=0;
canvas.addEventListener("mousedown",e=>{ drag=true; dx0=e.clientX; dy0=e.clientY; canvas.classList.add("drag"); });

window.addEventListener("mousemove",e=>{ 
  if(!drag)return; 
  tx+=e.clientX-dx0; ty+=e.clientY-dy0; 
  dx0=e.clientX; dy0=e.clientY; 
  render(); 
});

canvas.addEventListener("mouseup", e => { 
  drag=false; 
  canvas.classList.remove("drag"); 
  
  // If mouse moved less than 5 pixels, treat it as a click rather than a pan
  if (Math.hypot(e.clientX - dx0, e.clientY - dy0) < 5) {
    const mx = (e.clientX - tx) / scale, my = (e.clientY - ty) / scale;
    let hit = null;
    for (const n of allN) { 
        const dx = n._x - mx, dy = n._y - my; 
        if (dx * dx + dy * dy < NR * NR) { hit = n; break; } 
    }
    
    if (hit) {
        // Toggle selection
        const idx = selectedNodes.indexOf(hit.id);
        if (idx >= 0) {
            selectedNodes.splice(idx, 1); 
        } else {
            selectedNodes.push(hit.id);
            if (selectedNodes.length > 2) selectedNodes.shift(); // Max 2 selected
        }
    } else {
        // Clicked empty space
        selectedNodes = [];
    }
    updateEvolveBtn();
    render();
  }
});

// ── Buttons ───────────────────────────────────────────────────────────────────
document.getElementById("bfit").onclick  = fitAll;
document.getElementById("bzin").onclick  = ()=>{ scale*=1.2; render(); };
document.getElementById("bzout").onclick = ()=>{ scale*=0.83; render(); };
document.getElementById("brst").onclick  = ()=>{ scale=1; tx=0; ty=60; render(); };
let showSiblings = false;
document.getElementById("bsib").onclick  = ()=>{
  showSiblings = !showSiblings;
  document.getElementById("bsib").style.opacity = showSiblings ? "1" : "0.35";
  render();
};
document.getElementById("theme-btn").onclick = ()=>{
  const html=document.documentElement;
  const next=isDark()?"light":"dark";
  html.setAttribute("data-theme",next);
  document.getElementById("theme-btn").textContent=next==="dark"?"🌙":"☀️";
  render();
};

// ── Animation popup ───────────────────────────────────────────────────────────
const popup    = document.getElementById("popup");
const animC    = document.getElementById("anim-canvas");
const animCtx  = animC.getContext("2d");
const AW=400, AH=320;
let animFrame  = null;
let animNode   = null;
let animT      = 0;

function stopAnim(){
  if(animFrame){ cancelAnimationFrame(animFrame); animFrame=null; }
  popup.style.display="none";
  animNode=null;
}

function startAnim(node, screenX, screenY){
  if(!node.traj || !node.traj.positions || node.traj.positions.length===0){ return; }
  animNode=node;
  animT=0;
  popup.style.display="flex";

  let px=screenX+NR*scale+14, py=screenY-80;
  if(px+240>window.innerWidth)  px=screenX-NR*scale-250;
  if(py+200>window.innerHeight) py=window.innerHeight-210;
  if(py<10) py=10;
  popup.style.left=px+"px"; popup.style.top=py+"px";

  const nodeLv = (node._level !== undefined) ? node._level : node.gen;
  const nodeIsRand = nodeLv > 0 && !node.parent_id && !node.splice_parents;
  const spliceMark = node.splice_parents ? " ⚗️Spliced" : "";
  document.getElementById("popup-title").textContent=
    `Level ${nodeLv}${nodeIsRand?" ★rnd":""}${spliceMark} · ${node.parent_id?"child":"root"}`;
  document.getElementById("popup-fitness").textContent=
    `fitness: ${node.fitness.toFixed(4)}`;

  const positions = node.traj.positions;   
  const springs   = node.traj.springs;     

  let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
  for(const frame of positions)
    for(const [x,y] of frame){
      minX=Math.min(minX,x); maxX=Math.max(maxX,x);
      minY=Math.min(minY,y); maxY=Math.max(maxY,y);
    }
  const span=Math.max(maxX-minX,maxY-minY,0.1);
  const PAD=16;
  const drawScale=Math.min((AW-PAD*2)/span,(AH-PAD*2)/span);

  const wx=x=> PAD+(x-minX)*drawScale;
  const wy=y=> AH-PAD-(y-minY)*drawScale;

  function animStep(){
    const frame=positions[animT % positions.length];
    const dark=isDark();

    animCtx.clearRect(0,0,AW,AH);
    animCtx.fillStyle=dark?"#0d0d1a":"#f0f0f5";
    animCtx.fillRect(0,0,AW,AH);
    animCtx.strokeStyle=dark?"#333":"#ccc";
    animCtx.lineWidth=1;
    animCtx.beginPath(); animCtx.moveTo(0,wy(minY)); animCtx.lineTo(AW,wy(minY)); animCtx.stroke();

    if(springs){
      animCtx.strokeStyle=dark?"rgba(100,150,255,0.6)":"rgba(60,100,220,0.5)";
      animCtx.lineWidth=1.5;
      for(const [a,b] of springs){
        if(a>=frame.length||b>=frame.length) continue;
        animCtx.beginPath();
        animCtx.moveTo(wx(frame[a][0]),wy(frame[a][1]));
        animCtx.lineTo(wx(frame[b][0]),wy(frame[b][1]));
        animCtx.stroke();
      }
    }
    for(const [x,y] of frame){
      animCtx.beginPath();
      animCtx.arc(wx(x),wy(y),3.5,0,Math.PI*2);
      animCtx.fillStyle=dark?"#ffcc44":"#cc8800";
      animCtx.fill();
    }
    animCtx.fillStyle=dark?"#444":"#bbb";
    animCtx.font="9px monospace";
    animCtx.textAlign="right";
    animCtx.fillText(`t=${animT*node.traj.step}`,AW-4,AH-4);

    animT++;
    animFrame=requestAnimationFrame(animStep);
  }
  if(animFrame) cancelAnimationFrame(animFrame);
  animStep();
}

// ── Hover detection ───────────────────────────────────────────────────────────
const allN=[];
function flat(n){ if(!n)return; allN.push(n); flat(n.best_child); flat(n.median_child); }
for(const t of forest) flat(t);

let hovered=null;
canvas.addEventListener("mousemove",e=>{
  if(drag) return;
  const mx=(e.clientX-tx)/scale, my=(e.clientY-ty)/scale;
  let hit=null;
  for(const n of allN){ const dx=n._x-mx,dy=n._y-my; if(dx*dx+dy*dy<NR*NR){hit=n;break;} }
  if(hit!==hovered){
    hovered=hit;
    if(hit) startAnim(hit, sx(hit._x), sy(hit._y));
    else    stopAnim();
  }
});
canvas.addEventListener("mouseleave", stopAnim);

// ── Init ──────────────────────────────────────────────────────────────────────
updateEvolveBtn();
document.getElementById("dbt").textContent=" "+forest.length+" |";
resize();
fitAll();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    print(f"[srv] {os.path.abspath(_lineage_path)}", flush=True)
    if not os.path.exists(_lineage_path):
        return f"<pre>lineage.json not found.\nRun afpo_run.py first.</pre>", 404

    with open(_lineage_path) as f:
        data = json.load(f)
    nodes = data.get("nodes", {})

    forest, sibling_pairs = build_forest(nodes)

    all_fits = [n["fitness"] for n in nodes.values() if n["fitness"] > -999]
    min_fit  = min(all_fits) if all_fits else 0.0
    max_fit  = max(all_fits) if all_fits else 1.0

    return render_template_string(HTML,
        forest_json=json.dumps(forest),
        sibling_pairs_json=json.dumps(sibling_pairs),
        min_fit=min_fit, max_fit=max_fit,
        max_gen=max(n.get('level', n.get('gen',0)) for n in nodes.values()) if nodes else 0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lineage", default="lineage.json")
    parser.add_argument("--config",  default="config.yaml") # Added config argument
    parser.add_argument("--port",    type=int, default=5001)
    args = parser.parse_args()

    _lineage_path = args.lineage
    app.config["CONFIG_PATH"] = args.config

    print(f"Tree visualizer → http://localhost:{args.port}")
    print(f"Lineage: {_lineage_path}")
    print(f"Config: {args.config}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)