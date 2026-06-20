"""
Knesset Protocol Segmentation Annotator
========================================
Run locally:  python serve.py
Open:         http://localhost:5000

Directory layout expected (relative to this script):
    ../../../PROTOCOLS/           ← all 213 protocol JSON files
    ../../../steps/01_.../outputs/topics_canonical.json  ← existing topic labels
    ../outputs/segmentation.json  ← annotations (created automatically)

Usage:
  1. Pick a protocol from the left sidebar
  2. Scroll through utterances; auto-detected << נושא >> boundaries are shown
  3. Click ✂ to add a custom boundary before any utterance
  4. Click a boundary label to edit/remove it
  5. Reuse a label from the right panel (same doc or seen across all docs)
  6. Annotations auto-save on every change
"""

import json
import re
import sys
from pathlib import Path
from flask import Flask, jsonify, request, Response

# ── Paths ────────────────────────────────────────────────────────────────────
HERE          = Path(__file__).resolve().parent
PROTOCOLS_DIR = HERE.parents[2] / "PROTOCOLS"
STEP01_OUT    = HERE.parents[2] / "steps" / "01_topic_preprocessing" / "outputs"
ANN_FILE      = HERE.parent / "outputs" / "segmentation.json"

if not PROTOCOLS_DIR.exists():
    sys.exit(f"ERROR: PROTOCOLS directory not found at {PROTOCOLS_DIR}\n"
             "Run this script from the correct location.")

ANN_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Protocol parsing ──────────────────────────────────────────────────────────
SPEAKER_TAGS = {"יור", "דובר", "מנהל", "מזכיר"}

def parse_protocol(text: str) -> tuple[str, list[dict], list[dict]]:
    """
    Returns (preamble, utterances, suggested_boundaries).

    utterances: [{idx, speaker, text}]
    suggested_boundaries: [{before_idx, label}]  ← from << נושא >> markers
    """
    parts = re.split(r"<<\s*([^>]+?)\s*>>", text)
    preamble = parts[0].strip()

    items = []
    for i in range(1, len(parts) - 1, 2):
        items.append((parts[i].strip(), parts[i + 1] if i + 1 < len(parts) else ""))

    utterances = []
    suggested  = []
    pending_topic = None

    for marker, content in items:
        content = content.strip()
        if not content:
            continue

        if marker == "נושא":
            # << נושא >> comes in pairs wrapping the title.
            # Odd occurrence = opening (content = title); even = closing (content = junk)
            # We detect: if content looks like a real title (not attendee list), use it
            if "\n" not in content[:30] and len(content) < 300:
                pending_topic = content.strip()
        else:
            utt = {"idx": len(utterances), "speaker": marker, "text": content}
            utterances.append(utt)
            if pending_topic:
                suggested.append({"before_idx": len(utterances) - 1, "label": pending_topic})
                pending_topic = None

    return preamble, utterances, suggested


# ── Protocol list ─────────────────────────────────────────────────────────────
def load_protocols() -> list[dict]:
    protos = []
    for f in sorted(PROTOCOLS_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            protos.append({
                "doc_id":    d.get("doc_id", f.stem),
                "date":      d.get("date", "")[:10],
                "committee": d.get("committee", ""),
            })
        except Exception:
            pass
    protos.sort(key=lambda x: x["date"])
    return protos


# ── Annotations I/O ──────────────────────────────────────────────────────────
def load_annotations() -> dict:
    if ANN_FILE.exists():
        return json.loads(ANN_FILE.read_text(encoding="utf-8"))
    return {}

def save_annotations(data: dict):
    ANN_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def all_labels(ann: dict) -> list[str]:
    seen, out = set(), []
    for doc in ann.values():
        for seg in doc.get("segments", []):
            lbl = seg.get("label", "").strip()
            if lbl and lbl not in seen:
                seen.add(lbl)
                out.append(lbl)
    return out


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")

@app.route("/api/protocols")
def api_protocols():
    ann  = load_annotations()
    pros = load_protocols()
    for p in pros:
        p["done"] = p["doc_id"] in ann
    return jsonify(pros)

@app.route("/api/protocol/<doc_id>")
def api_protocol(doc_id):
    f = PROTOCOLS_DIR / f"{doc_id}.json"
    if not f.exists():
        return jsonify({"error": "not found"}), 404
    d    = json.loads(f.read_text(encoding="utf-8"))
    pre, utts, sugg = parse_protocol(d.get("text", ""))
    ann  = load_annotations()
    return jsonify({
        "doc_id":    doc_id,
        "date":      d.get("date", "")[:10],
        "committee": d.get("committee", ""),
        "preamble":  pre[:600],
        "utterances": utts,
        "suggested":  sugg,
        "saved":      ann.get(doc_id, {}),
    })

@app.route("/api/annotation/<doc_id>", methods=["POST"])
def api_save(doc_id):
    ann = load_annotations()
    ann[doc_id] = request.json
    save_annotations(ann)
    return jsonify({"ok": True})

@app.route("/api/labels")
def api_labels():
    return jsonify(all_labels(load_annotations()))


# ── HTML (single-page app) ────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Knesset Segmentation Annotator</title>
<!-- v3: label-picker modal + color-by-label -->
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  background:#f0f2f5;display:flex;height:100vh;overflow:hidden}

/* ── Sidebar ── */
#sidebar{width:220px;min-width:220px;background:#1a1a2e;color:#ccc;
  display:flex;flex-direction:column;overflow:hidden}
#sidebar h2{padding:14px 16px;font-size:.85rem;color:#fff;
  border-bottom:1px solid #2c2c50;letter-spacing:.04em}
#proto-search{margin:10px 12px;padding:6px 10px;border-radius:6px;
  border:none;background:#2c2c50;color:#ddd;font-size:.82rem;width:calc(100% - 24px)}
#proto-search::placeholder{color:#777}
#proto-list{overflow-y:auto;flex:1;padding:4px 0}
.proto-item{padding:9px 14px;cursor:pointer;font-size:.78rem;
  border-left:3px solid transparent;transition:background .12s}
.proto-item:hover{background:#2c2c50}
.proto-item.active{background:#2c2c50;border-left-color:#4a90d9}
.proto-item.done .pid{color:#2ecc71}
.proto-item .pdate{font-size:.7rem;color:#666;margin-top:2px}

/* ── Main ── */
#main{flex:1;display:flex;flex-direction:column;overflow:hidden}
#topbar{background:white;padding:10px 18px;border-bottom:1px solid #e0e0e0;
  display:flex;align-items:center;gap:12px;flex-shrink:0}
#topbar h3{font-size:.9rem;color:#1a1a2e}
.meta{font-size:.78rem;color:#999}
.btn-save{margin-left:auto;padding:7px 20px;background:#2ecc71;color:white;
  border:none;border-radius:7px;font-weight:700;cursor:pointer;font-size:.85rem}
.btn-save:hover{background:#27ae60}
#transcript{flex:1;overflow-y:auto;padding:12px 16px}

/* ── Segment band ── */
.seg-band{border-radius:8px;margin-bottom:6px;overflow:hidden;
  border:1.5px solid transparent}
.seg-header{display:flex;align-items:center;gap:8px;
  padding:7px 12px;font-size:.85rem}
.seg-header input{direction:rtl;flex:1;border:none;background:transparent;
  font-weight:700;font-size:.88rem;color:#1a1a2e;outline:none;
  min-width:0}
.seg-header input:focus{border-bottom:1.5px solid rgba(0,0,0,.2)}
.auto-tag{font-size:.65rem;padding:1px 7px;background:rgba(0,0,0,.12);
  border-radius:10px;white-space:nowrap;color:inherit;opacity:.7}
.btn-rm{padding:2px 8px;border:1.5px solid currentColor;border-radius:4px;
  font-size:.7rem;cursor:pointer;background:transparent;color:inherit;
  opacity:.6;flex-shrink:0}
.btn-rm:hover{opacity:1;background:rgba(0,0,0,.1)}
.seg-body{padding:4px 6px 6px}

/* ── Cut button between utterances ── */
.cut-row{height:20px;display:flex;align-items:center;padding:0 6px}
.cut-btn{opacity:0;border:none;background:none;color:#999;
  font-size:.72rem;cursor:pointer;padding:2px 10px;
  border:1px dashed #ccc;border-radius:4px;transition:opacity .15s}
.cut-row:hover .cut-btn{opacity:1}

/* ── Between-segment cut ── */
.between-segs{height:28px;display:flex;align-items:center;justify-content:center}
.between-btn{opacity:0;border:1.5px dashed #bbb;background:white;
  color:#888;font-size:.75rem;padding:3px 18px;border-radius:5px;
  cursor:pointer;transition:opacity .15s}
.between-segs:hover .between-btn{opacity:1}

/* ── Utterances ── */
.utterance{display:flex;gap:8px;padding:4px 6px;border-radius:5px;
  transition:background .1s}
.utterance:hover{background:rgba(0,0,0,.04)}
.speaker-tag{font-size:.68rem;font-weight:700;color:#4a90d9;
  min-width:38px;padding-top:3px;flex-shrink:0}
.utt-text{direction:rtl;text-align:right;font-size:.88rem;
  line-height:1.65;color:#333;flex:1}

/* ── Unlabeled preamble ── */
.unlabeled{border-color:#ddd !important;background:#fafafa !important}
.unlabeled .seg-header{background:#eee !important;color:#999}

/* ── Right panel ── */
#right{width:200px;min-width:200px;background:white;
  border-left:1px solid #e0e0e0;display:flex;flex-direction:column;
  overflow:hidden;font-size:.8rem}
#right h4{padding:12px 14px;border-bottom:1px solid #eee;
  font-size:.78rem;color:#888;text-transform:uppercase;letter-spacing:.05em}
.r-section{overflow-y:auto;flex:1;border-bottom:1px solid #eee}
.r-section:last-child{border-bottom:none}
.section-title{padding:6px 14px 4px;font-size:.7rem;color:#aaa;
  text-transform:uppercase;letter-spacing:.05em}
.label-chip{padding:7px 12px;cursor:pointer;direction:rtl;text-align:right;
  color:#333;line-height:1.4;border-left:3px solid transparent}
.label-chip:hover{background:#f0f6ff;border-left-color:#4a90d9}
.label-chip.global{color:#aaa}

/* ── Empty ── */
#empty{display:flex;align-items:center;justify-content:center;
  flex:1;color:#bbb;font-size:1rem}

/* Segment colour palette (background + border) */
.c0{background:#eef6ff;border-color:#b3d4f5}  .c0 .seg-header{background:#b3d4f5;color:#1a3a5c}
.c1{background:#fff7ee;border-color:#fbc97a}  .c1 .seg-header{background:#fbc97a;color:#5c3a00}
.c2{background:#efffef;border-color:#86d98c}  .c2 .seg-header{background:#86d98c;color:#1a4d1e}
.c3{background:#fdf0ff;border-color:#d4a0f0}  .c3 .seg-header{background:#d4a0f0;color:#3d0060}
.c4{background:#fff0f0;border-color:#f5a0a0}  .c4 .seg-header{background:#f5a0a0;color:#5c0000}
.c5{background:#f0fffc;border-color:#80ddd0}  .c5 .seg-header{background:#80ddd0;color:#003d36}
.c6{background:#fffff0;border-color:#d4d080}  .c6 .seg-header{background:#d4d080;color:#3d3a00}
.c7{background:#f0f4ff;border-color:#a0b4f0}  .c7 .seg-header{background:#a0b4f0;color:#001060}

/* ── Label-picker modal ── */
#modal-backdrop{position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:100}
#modal-box{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
  z-index:101;background:white;border-radius:12px;width:420px;max-width:95vw;
  box-shadow:0 8px 40px rgba(0,0,0,.25);display:flex;flex-direction:column;
  max-height:80vh;overflow:hidden}
#modal-box h3{padding:14px 18px;border-bottom:1px solid #eee;font-size:.95rem}
#modal-input{margin:12px 14px 6px;padding:9px 12px;border:2px solid #dde2ec;
  border-radius:8px;font-size:.95rem;direction:rtl;width:calc(100% - 28px)}
#modal-input:focus{outline:none;border-color:#4a90d9}
#modal-chips{overflow-y:auto;flex:1;padding:4px 0}
.mchip-section{padding:6px 14px 2px;font-size:.7rem;color:#aaa;
  text-transform:uppercase;letter-spacing:.05em}
.mchip{padding:8px 16px;cursor:pointer;direction:rtl;text-align:right;
  font-size:.88rem;color:#1a1a2e;border-left:3px solid transparent;
  transition:background .1s}
.mchip:hover{background:#f0f6ff;border-left-color:#4a90d9}
.mchip.global{color:#888}
#modal-footer{padding:10px 14px;border-top:1px solid #eee;
  display:flex;justify-content:flex-end;gap:8px}
.btn-cancel{padding:7px 18px;border:1.5px solid #ddd;border-radius:7px;
  background:white;cursor:pointer;font-size:.85rem}
.btn-ok{padding:7px 20px;background:#4a90d9;color:white;border:none;
  border-radius:7px;font-weight:700;cursor:pointer;font-size:.85rem}
</style>
</head>
<body>

<div id="sidebar">
  <h2>📋 Protocols</h2>
  <input id="proto-search" type="text" placeholder="Search…" oninput="filterProtos()">
  <div id="proto-list"></div>
</div>

<div id="main">
  <div id="topbar">
    <h3 id="doc-title">← Select a protocol</h3>
    <span class="meta" id="doc-meta"></span>
    <button class="btn-save" onclick="saveAnnotation()" style="display:none" id="btn-save">✓ Save</button>
  </div>
  <div id="transcript"><div id="empty">Select a protocol from the sidebar</div></div>
</div>

<div id="right">
  <h4>Topic Labels</h4>
  <div class="r-section" id="labels-here">
    <div class="section-title">This document</div>
  </div>
  <div class="r-section" id="labels-global">
    <div class="section-title">Seen elsewhere</div>
  </div>
</div>

<!-- Label-picker modal (hidden until needed) -->
<div id="modal-backdrop" style="display:none" onclick="modalCancel()"></div>
<div id="modal-box" style="display:none">
  <h3>Choose topic label</h3>
  <input id="modal-input" type="text" placeholder="Type new label… (Hebrew ok)"
    oninput="filterChips(this.value)"
    onkeydown="if(event.key==='Enter')modalConfirm();if(event.key==='Escape')modalCancel()">
  <div id="modal-chips"></div>
  <div id="modal-footer">
    <button class="btn-cancel" onclick="modalCancel()">Cancel</button>
    <button class="btn-ok"     onclick="modalConfirm()">Use this label</button>
  </div>
</div>

<script>
const COLORS = 8;
let currentDocId=null, segments=[], utterances=[], allProtos=[], globalLabels=[];
let modalResolve=null;

async function boot() {
  const [protos,labels] = await Promise.all([
    fetch('/api/protocols').then(r=>r.json()),
    fetch('/api/labels').then(r=>r.json()),
  ]);
  allProtos=protos; globalLabels=labels;
  renderSidebar(protos);
}

// ── Sidebar ───────────────────────────────────────────────────────────────
function renderSidebar(protos) {
  const q = document.getElementById('proto-search').value.toLowerCase();
  const list = document.getElementById('proto-list');
  list.innerHTML='';
  protos.filter(p=>!q||p.doc_id.includes(q)||p.date.includes(q)).forEach(p=>{
    const d=document.createElement('div');
    d.className='proto-item'+(p.doc_id===currentDocId?' active':'');
    d.innerHTML=`<div class="pid">${p.doc_id.replace('25_ptv_','')}${p.done?' ✓':''}</div>
      <div class="pdate">${p.date}</div>`;
    d.onclick=()=>loadProtocol(p.doc_id);
    list.appendChild(d);
  });
}
function filterProtos(){renderSidebar(allProtos);}

// ── Load ─────────────────────────────────────────────────────────────────
async function loadProtocol(doc_id) {
  currentDocId=doc_id;
  const data=await fetch('/api/protocol/'+doc_id).then(r=>r.json());
  utterances=data.utterances;
  document.getElementById('doc-title').textContent=doc_id;
  document.getElementById('doc-meta').textContent=data.date+'  '+data.committee;
  document.getElementById('btn-save').style.display='';
  segments=(data.saved&&data.saved.segments&&data.saved.segments.length)
    ? data.saved.segments.map(s=>({...s,auto:false}))
    : data.suggested.map(s=>({...s,auto:true}));
  render();
  renderSidebar(allProtos);
}

// ── Color by label (not position) ────────────────────────────────────────
function buildColorMap(bands) {
  const map={};
  let next=0;
  bands.forEach(b=>{
    if(b.label && !(b.label in map)) map[b.label]=next++%COLORS;
  });
  return map;
}

// ── Bands ────────────────────────────────────────────────────────────────
function computeBands() {
  const n=utterances.length;
  const bounds=[...segments].sort((a,b)=>a.before_idx-b.before_idx);
  const bands=[];
  const first=bounds.length>0?bounds[0].before_idx:n;
  if(first>0) bands.push({start:0,end:first,label:null,auto:false,preamble:true});
  bounds.forEach((seg,i)=>{
    const end=i+1<bounds.length?bounds[i+1].before_idx:n;
    bands.push({start:seg.before_idx,end,label:seg.label,auto:seg.auto,preamble:false});
  });
  if(bands.length===0) bands.push({start:0,end:n,label:null,auto:false,preamble:true});
  return bands;
}

// ── Render ────────────────────────────────────────────────────────────────
function render() {
  const box=document.getElementById('transcript');
  box.innerHTML='';
  const bands=computeBands();
  const colorMap=buildColorMap(bands);

  bands.forEach((band,bi)=>{
    if(bi>0){
      const btwn=document.createElement('div');
      btwn.className='between-segs';
      btwn.innerHTML=`<button class="between-btn" onclick="doCut(${band.start})">✂ split here into new topic</button>`;
      box.appendChild(btwn);
    }

    const colorClass=band.preamble?'unlabeled':'c'+(colorMap[band.label]??0);
    const bandEl=document.createElement('div');
    bandEl.className='seg-band '+colorClass;

    // Header
    const hdr=document.createElement('div');
    hdr.className='seg-header';
    if(band.preamble){
      hdr.innerHTML=`<span style="flex:1;direction:rtl;font-weight:700;color:#aaa">פתיחה / ללא נושא</span>
        <button class="btn-rm" onclick="doCut(0)">+ label</button>`;
    } else {
      hdr.innerHTML=`
        <input type="text" value="${esc(band.label||'')}"
          placeholder="topic label…"
          onchange="updateLabel(${band.start},this.value)"
          onblur="render()">
        ${band.auto?'<span class="auto-tag">auto</span>':''}
        <button class="btn-rm" onclick="removeBoundary(${band.start})">✕ merge up</button>`;
    }
    bandEl.appendChild(hdr);

    // Utterances
    const body=document.createElement('div');
    body.className='seg-body';
    for(let i=band.start;i<band.end;i++){
      if(i>band.start){
        const crow=document.createElement('div');
        crow.className='cut-row';
        crow.innerHTML=`<button class="cut-btn" onclick="doCut(${i})">✂ new topic starts here</button>`;
        body.appendChild(crow);
      }
      const utt=utterances[i];
      const row=document.createElement('div');
      row.className='utterance';
      row.innerHTML=`<span class="speaker-tag">${esc(utt.speaker)}</span>
        <div class="utt-text">${esc(utt.text)}</div>`;
      body.appendChild(row);
    }
    bandEl.appendChild(body);
    box.appendChild(bandEl);
  });

  renderRightPanel();
}

// ── Label-picker modal ────────────────────────────────────────────────────
function pickLabel() {
  return new Promise(resolve=>{
    modalResolve=resolve;
    document.getElementById('modal-input').value='';
    populateChips('');
    document.getElementById('modal-backdrop').style.display='';
    document.getElementById('modal-box').style.display='';
    setTimeout(()=>document.getElementById('modal-input').focus(),60);
  });
}

function populateChips(filter) {
  const box=document.getElementById('modal-chips');
  box.innerHTML='';
  const local=[...new Set(segments.map(s=>s.label).filter(Boolean))];
  const seenLocal=new Set(local);
  const q=filter.trim().toLowerCase();

  const filtered=lbl=>!q||lbl.includes(q)||lbl.toLowerCase().includes(q);

  const localFiltered=local.filter(filtered);
  if(localFiltered.length){
    const sec=document.createElement('div');
    sec.className='mchip-section'; sec.textContent='This document';
    box.appendChild(sec);
    localFiltered.forEach(lbl=>{
      const c=document.createElement('div');
      c.className='mchip'; c.textContent=lbl;
      c.onclick=()=>modalPick(lbl);
      box.appendChild(c);
    });
  }

  const globalFiltered=globalLabels.filter(l=>!seenLocal.has(l)&&filtered(l));
  if(globalFiltered.length){
    const sec=document.createElement('div');
    sec.className='mchip-section'; sec.textContent='Seen elsewhere';
    box.appendChild(sec);
    globalFiltered.forEach(lbl=>{
      const c=document.createElement('div');
      c.className='mchip global'; c.textContent=lbl;
      c.onclick=()=>modalPick(lbl);
      box.appendChild(c);
    });
  }
}

function filterChips(q){ populateChips(q); }

function modalPick(lbl){
  // Clicking a chip instantly selects that label
  hideModal();
  if(modalResolve){ modalResolve(lbl); modalResolve=null; }
}
function modalConfirm(){
  const val=document.getElementById('modal-input').value.trim();
  hideModal();
  if(modalResolve){ modalResolve(val||null); modalResolve=null; }
}
function modalCancel(){
  hideModal();
  if(modalResolve){ modalResolve(null); modalResolve=null; }
}
function hideModal(){
  document.getElementById('modal-backdrop').style.display='none';
  document.getElementById('modal-box').style.display='none';
}

// ── Boundary operations ───────────────────────────────────────────────────
async function doCut(before_idx){
  const lbl=await pickLabel();
  if(!lbl) return;
  segments=segments.filter(s=>s.before_idx!==before_idx);
  segments.push({before_idx,label:lbl,auto:false});
  render(); autoSave();
}
function removeBoundary(before_idx){
  segments=segments.filter(s=>s.before_idx!==before_idx);
  render(); autoSave();
}
function updateLabel(before_idx,value){
  const seg=segments.find(s=>s.before_idx===before_idx);
  if(seg){seg.label=value.trim();seg.auto=false;}
  autoSave();
}

// ── Right panel ───────────────────────────────────────────────────────────
function renderRightPanel(){
  const here=document.getElementById('labels-here');
  const glob=document.getElementById('labels-global');
  const local=[...new Set(segments.map(s=>s.label).filter(Boolean))];
  const seenLocal=new Set(local);
  const bands=computeBands();
  const colorMap=buildColorMap(bands);

  here.innerHTML='<div class="section-title">This document</div>';
  local.forEach(lbl=>{
    const ci=colorMap[lbl]??0;
    const c=document.createElement('div');
    c.className='label-chip';
    c.style.cssText=`border-left-color:var(--c${ci}-border,#4a90d9)`;
    c.textContent=lbl;
    here.appendChild(c);
  });

  glob.innerHTML='<div class="section-title">Seen elsewhere</div>';
  globalLabels.filter(l=>!seenLocal.has(l)).forEach(lbl=>{
    const c=document.createElement('div');
    c.className='label-chip global'; c.textContent=lbl;
    glob.appendChild(c);
  });
}

// ── Save ─────────────────────────────────────────────────────────────────
async function saveAnnotation(){
  if(!currentDocId) return;
  await fetch('/api/annotation/'+currentDocId,{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({segments}),
  });
  globalLabels=await fetch('/api/labels').then(r=>r.json());
  allProtos=await fetch('/api/protocols').then(r=>r.json());
  renderSidebar(allProtos);
  renderRightPanel();
  const btn=document.getElementById('btn-save');
  btn.textContent='✓ Saved!';
  setTimeout(()=>btn.textContent='✓ Save',1500);
}
let saveTimer=null;
function autoSave(){clearTimeout(saveTimer);saveTimer=setTimeout(saveAnnotation,800);}

function esc(s){
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
boot();
</script>
</body>
</html>"""


if __name__ == "__main__":
    print(f"Protocols : {PROTOCOLS_DIR}  ({len(list(PROTOCOLS_DIR.glob('*.json')))} files)")
    print(f"Annotations: {ANN_FILE}")
    print(f"\nOpen: http://localhost:5000\n")
    app.run(debug=False, port=5000)
