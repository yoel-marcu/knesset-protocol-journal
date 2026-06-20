"""
Step 06 — Generate a self-contained HTML annotator.

Fixes vs v1:
  - No prompt() — name is chosen via an inline overlay on first load
  - JSON embedded via JSON.parse() with </script> escaped — no parser breakage
  - Progress autosaves to localStorage after every label

Usage:
    python generate_html_annotator.py
    # open steps/06_annotation/outputs/annotator.html in browser
"""

import json
from pathlib import Path

STEP_DIR = Path(__file__).resolve().parents[1]
OUT      = STEP_DIR / "outputs"
BATCH_FILE = OUT / "annotation_batch.json"


def safe_json(obj) -> str:
    """JSON string safe to embed inside a <script> tag."""
    return (
        json.dumps(obj, ensure_ascii=False)
            .replace("</", "<\\/")   # prevent premature </script> closing
            .replace("'", "\\'")     # safe for single-quoted JS strings
    )


def main():
    batch  = json.load(open(BATCH_FILE, encoding="utf-8"))
    pairs  = batch["pairs"]
    total  = len(pairs)
    data_js = safe_json(pairs)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Subject Linking Annotator</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f0f2f5; color: #1a1a2e;
  min-height: 100vh; display: flex; flex-direction: column;
  align-items: center; padding: 20px 14px;
}}

/* ── Name overlay ── */
#overlay {{
  position: fixed; inset: 0; background: rgba(0,0,0,0.55);
  display: flex; align-items: center; justify-content: center; z-index: 999;
}}
#overlay-box {{
  background: white; border-radius: 14px; padding: 36px 40px;
  text-align: center; max-width: 360px; width: 90%;
  box-shadow: 0 8px 40px rgba(0,0,0,0.25);
}}
#overlay-box h2 {{ font-size: 1.2rem; margin-bottom: 8px; }}
#overlay-box p  {{ font-size: 0.85rem; color: #888; margin-bottom: 22px; }}
.name-btn {{
  display: block; width: 100%; padding: 11px;
  margin-bottom: 10px; border: 2px solid #dde2ec; border-radius: 8px;
  background: white; font-size: 1rem; font-weight: 600; cursor: pointer;
  transition: border-color .15s, background .15s;
}}
.name-btn:hover {{ border-color: #4a90d9; background: #f0f6ff; }}
.name-divider {{ color: #ccc; font-size: 0.8rem; margin: 10px 0; }}
#custom-name {{
  width: 100%; padding: 9px 12px; border: 2px solid #dde2ec;
  border-radius: 8px; font-size: 0.95rem; margin-bottom: 10px;
}}
#custom-name:focus {{ outline: none; border-color: #4a90d9; }}
#confirm-btn {{
  width: 100%; padding: 10px; background: #1a1a2e; color: white;
  border: none; border-radius: 8px; font-size: 0.95rem; font-weight: 600;
  cursor: pointer;
}}
#confirm-btn:hover {{ background: #2c2c50; }}

/* ── Header ── */
.header {{
  width: 100%; max-width: 880px;
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 16px;
}}
.header h1 {{ font-size: 1rem; color: #666; font-weight: 600; }}
.annotator-badge {{
  background: #4a90d9; color: white;
  padding: 4px 14px; border-radius: 20px;
  font-size: 0.82rem; font-weight: 700;
}}

/* ── Progress ── */
.progress-wrap {{ width: 100%; max-width: 880px; margin-bottom: 14px; }}
.progress-bar {{
  height: 5px; background: #dde2ec;
  border-radius: 3px; overflow: hidden;
}}
.progress-fill {{
  height: 100%;
  background: linear-gradient(90deg, #4a90d9, #7b5ea7);
  border-radius: 3px; transition: width .3s;
}}
.progress-label {{
  font-size: 0.78rem; color: #999; margin-top: 5px;
  display: flex; justify-content: space-between;
}}

/* ── Card ── */
.card {{
  width: 100%; max-width: 880px; background: white;
  border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,.07);
  overflow: hidden; margin-bottom: 16px;
}}
.card-header {{
  background: #1a1a2e; color: white;
  padding: 10px 18px; font-size: 0.83rem;
  display: flex; justify-content: space-between; align-items: center;
}}
.stratum-badge {{
  font-size: 0.7rem; padding: 2px 10px; border-radius: 12px;
  font-weight: 700; text-transform: uppercase; letter-spacing: .04em;
}}
.s-positive {{ background:#2ecc71; color:white; }}
.s-hard_neg {{ background:#e74c3c; color:white; }}
.s-easy_neg {{ background:#95a5a6; color:white; }}

/* ── Spans ── */
.spans {{ display: grid; grid-template-columns: 1fr 1fr; }}
.span-box {{
  padding: 18px 20px;
  border-right: 1px solid #eee;
}}
.span-box:last-child {{ border-right: none; }}
.span-label {{
  font-size: 0.68rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: .08em;
  color: #bbb; margin-bottom: 6px;
}}
.span-date {{
  font-size: 0.78rem; color: #4a90d9;
  font-weight: 600; margin-bottom: 8px;
}}
.span-topic {{
  direction: rtl; text-align: right; unicode-bidi: embed;
  font-size: 0.95rem; font-weight: 700; line-height: 1.55;
  color: #1a1a2e; margin-bottom: 12px;
  padding-bottom: 10px; border-bottom: 1px solid #f0f0f0;
}}
.span-preview {{
  direction: rtl; text-align: right; unicode-bidi: embed;
  font-size: 0.85rem; line-height: 1.75; color: #555;
  max-height: 260px; overflow-y: auto;
  background: #fafafa; border-radius: 6px;
  padding: 10px 12px; white-space: pre-wrap;
}}

/* ── Labels ── */
.label-section {{
  padding: 16px 18px; border-top: 1px solid #eee;
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
}}
.label-prompt {{ font-size: 0.82rem; color: #999; }}
.btn-lbl {{
  padding: 9px 24px; border: none; border-radius: 8px;
  font-size: 0.9rem; font-weight: 700; cursor: pointer;
  transition: transform .1s, box-shadow .1s; position: relative;
}}
.btn-lbl:hover  {{ transform: translateY(-1px); box-shadow: 0 4px 10px rgba(0,0,0,.12); }}
.btn-lbl:active {{ transform: translateY(0); }}
.key {{
  position: absolute; top: -7px; right: -5px;
  background: white; border: 1.5px solid; border-radius: 4px;
  font-size: 0.6rem; padding: 1px 4px; font-weight: 800; opacity: .75;
}}
.btn-same    {{ background:#2ecc71; color:white; }}
.btn-related {{ background:#f39c12; color:white; }}
.btn-new     {{ background:#e74c3c; color:white; }}
.btn-skip    {{ background:#bdc3c7; color:white; }}
.btn-lbl.active {{ outline: 3px solid #1a1a2e; outline-offset: 2px; }}

/* ── Nav + export ── */
.bottom-bar {{
  width: 100%; max-width: 880px;
  display: flex; justify-content: space-between; align-items: center;
}}
.nav {{ display: flex; gap: 8px; }}
.btn-nav {{
  padding: 8px 18px; background: white;
  border: 1.5px solid #ddd; border-radius: 8px;
  font-size: 0.83rem; cursor: pointer; color: #555;
}}
.btn-nav:hover {{ background: #f5f5f5; }}
.btn-export {{
  padding: 9px 22px; background: #1a1a2e; color: white;
  border: none; border-radius: 8px;
  font-size: 0.85rem; font-weight: 600; cursor: pointer;
}}
.btn-export:hover {{ background: #2c2c50; }}

@media(max-width:580px) {{
  .spans {{ grid-template-columns:1fr; }}
  .span-box {{ border-right:none; border-bottom:1px solid #eee; }}
}}
</style>
</head>
<body>

<!-- Name overlay -->
<div id="overlay">
  <div id="overlay-box">
    <h2>Who are you?</h2>
    <p>Annotations are stored per-name in your browser.</p>
    <button class="name-btn" onclick="setName('yoel')">Yoel</button>
    <button class="name-btn" onclick="setName('tomer')">Tomer</button>
    <button class="name-btn" onclick="setName('or')">Or</button>
    <div class="name-divider">or type below</div>
    <input id="custom-name" type="text" placeholder="your name">
    <button id="confirm-btn" onclick="setName(document.getElementById('custom-name').value)">Confirm</button>
  </div>
</div>

<div class="header">
  <h1>Knesset Protocol Journal — Annotation</h1>
  <span class="annotator-badge" id="badge">—</span>
</div>

<div class="progress-wrap">
  <div class="progress-bar"><div class="progress-fill" id="pfill" style="width:0%"></div></div>
  <div class="progress-label">
    <span id="plabel">0 / {total} annotated</span>
    <span id="premain">{total} remaining</span>
  </div>
</div>

<div class="card">
  <div class="card-header">
    <span id="pair-counter">Pair — / {total}</span>
    <span id="stratum-badge" class="stratum-badge"></span>
  </div>
  <div class="spans">
    <div class="span-box">
      <div class="span-label">Span A</div>
      <div class="span-date"    id="date-a"></div>
      <div class="span-topic"   id="topic-a"></div>
      <div class="span-preview" id="preview-a"></div>
    </div>
    <div class="span-box">
      <div class="span-label">Span B</div>
      <div class="span-date"    id="date-b"></div>
      <div class="span-topic"   id="topic-b"></div>
      <div class="span-preview" id="preview-b"></div>
    </div>
  </div>
  <div class="label-section">
    <span class="label-prompt">Label:</span>
    <button class="btn-lbl btn-same"    id="btn-same"    onclick="label('same')"   ><span class="key">1</span>SAME</button>
    <button class="btn-lbl btn-related" id="btn-related" onclick="label('related')" ><span class="key">2</span>RELATED</button>
    <button class="btn-lbl btn-new"     id="btn-new"     onclick="label('new')"    ><span class="key">3</span>NEW</button>
    <button class="btn-lbl btn-skip"    id="btn-skip"    onclick="label('skip')"   ><span class="key">0</span>SKIP</button>
  </div>
</div>

<div class="bottom-bar">
  <div class="nav">
    <button class="btn-nav" onclick="go(-1)">← Prev</button>
    <button class="btn-nav" onclick="go(+1)">Next →</button>
  </div>
  <button class="btn-export" onclick="exportJSON()">⬇ Export JSON</button>
</div>

<script>
const PAIRS = JSON.parse('{data_js}');
const TOTAL = PAIRS.length;
let name = '';
let ann  = {{}};
let idx  = 0;

function setName(n) {{
  n = (n || '').trim();
  if (!n) return;
  name = n;
  document.getElementById('overlay').style.display = 'none';
  document.getElementById('badge').textContent = name;
  const saved = localStorage.getItem('ann_' + name);
  if (saved) ann = JSON.parse(saved);
  idx = PAIRS.findIndex(p => !ann[p.id]);
  if (idx === -1) idx = 0;
  render();
}}

function render() {{
  const p = PAIRS[idx];
  document.getElementById('pair-counter').textContent = `Pair ${{idx+1}} / ${{TOTAL}}`;
  const sb = document.getElementById('stratum-badge');
  sb.textContent = p.stratum.replace('_',' ');
  sb.className   = 'stratum-badge s-' + p.stratum;

  document.getElementById('date-a').textContent    = p.a.date    || '—';
  document.getElementById('topic-a').textContent   = p.a.topic;
  document.getElementById('preview-a').textContent = p.a.preview;
  document.getElementById('date-b').textContent    = p.b.date    || '—';
  document.getElementById('topic-b').textContent   = p.b.topic;
  document.getElementById('preview-b').textContent = p.b.preview;

  ['same','related','new','skip'].forEach(l =>
    document.getElementById('btn-'+l).classList.remove('active'));
  const existing = ann[p.id];
  if (existing) document.getElementById('btn-'+existing.label).classList.add('active');

  updateProgress();
}}

function label(value) {{
  ann[PAIRS[idx].id] = {{ label: value, stratum: PAIRS[idx].stratum }};
  localStorage.setItem('ann_' + name, JSON.stringify(ann));
  const next = PAIRS.findIndex((p, i) => i > idx && !ann[p.id]);
  if (next !== -1) idx = next;
  else if (idx < TOTAL-1) idx++;
  render();
}}

function go(delta) {{
  idx = Math.max(0, Math.min(TOTAL-1, idx+delta));
  render();
}}

function updateProgress() {{
  const done = Object.keys(ann).length;
  document.getElementById('pfill').style.width   = (done/TOTAL*100).toFixed(1)+'%';
  document.getElementById('plabel').textContent  = `${{done}} / ${{TOTAL}} annotated`;
  document.getElementById('premain').textContent = done < TOTAL ? `${{TOTAL-done}} remaining` : '✓ Complete!';
}}

function exportJSON() {{
  const out = {{ annotator: name, pairs: {{}} }};
  for (const [id, v] of Object.entries(ann))
    out.pairs[id] = {{ label: v.label, stratum: v.stratum, pair_id: id }};
  const a = document.createElement('a');
  a.href     = URL.createObjectURL(new Blob([JSON.stringify(out,null,2)], {{type:'application/json'}}));
  a.download = `annotations_${{name}}.json`;
  a.click();
}}

document.addEventListener('keydown', e => {{
  if (['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
  if (e.key==='1') label('same');
  else if (e.key==='2') label('related');
  else if (e.key==='3') label('new');
  else if (e.key==='0') label('skip');
  else if (e.key==='ArrowLeft')  go(-1);
  else if (e.key==='ArrowRight') go(+1);
}});
</script>
</body>
</html>"""

    out_path = OUT / "annotator.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Open: file://{out_path}")
    print(f"  {total} pairs embedded")


if __name__ == "__main__":
    main()
