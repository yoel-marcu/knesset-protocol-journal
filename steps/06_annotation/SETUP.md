# Knesset Annotation Tool — Setup Guide
**For Tomer and Or**

Yoel has shared a Google Drive folder with everything you need.
This guide gets you running in ~10 minutes.

---

## What's in the Drive folder

```
knesset-annotator/
├── PROTOCOLS/              ← 213 protocol JSON files (~61 MB)
├── steps/
│   └── 06_annotation/
│       ├── code/
│       │   └── serve.py   ← the annotation server
│       └── outputs/
│           └── segmentation.json  ← shared annotations (pull before each session)
└── SETUP.md               ← this file
```

---

## Step 1 — Download the Drive folder

Download the entire `knesset-annotator` folder from Google Drive to your Desktop.

> **Windows tip:** right-click the folder in Drive → "Download". It downloads as a zip.
> Unzip it so you have `C:\Users\YOUR_NAME\Desktop\knesset-annotator\`.

---

## Step 2 — Install Python (if you don't have it)

Open a terminal:
- **Windows:** press `Win+R`, type `cmd`, press Enter
- **Mac:** press `Cmd+Space`, type `Terminal`, press Enter

Check if Python is installed:
```
python --version
```

If you see `Python 3.x.x` you're good. If not, download from https://python.org/downloads
and install (check "Add Python to PATH" on Windows during install).

---

## Step 3 — Install Flask (one-time)

In the same terminal:
```
pip install flask
```

---

## Step 4 — Run the annotation server

**Windows:**
```cmd
cd %USERPROFILE%\Desktop\knesset-annotator
python steps\06_annotation\code\serve.py
```

**Mac:**
```bash
cd ~/Desktop/knesset-annotator
python steps/06_annotation/code/serve.py
```

You should see:
```
Protocols : ...PROTOCOLS  (213 files)
Annotations: ...segmentation.json

Open: http://localhost:5000
```

Open **http://localhost:5000** in your browser (Chrome or Firefox recommended).

To stop the server: press `Ctrl+C` in the terminal.

---

## Step 5 — Annotating

- **Left sidebar:** list of all 213 protocols. Green ✓ = already annotated by someone.
  Pick any protocol **without** a ✓ (coordinate with Yoel so you don't overlap).
- **Main area:** the transcript as utterances. Auto-detected topic boundaries are
  pre-filled (marked "auto") — confirm them, edit the labels, or remove them.
- **✂ new topic starts here** (appears on hover between utterances) → click to split;
  a picker appears showing all past topic labels to reuse or a field to type a new one.
- **✕ merge up** on a band header → removes that boundary, merging it into the segment above.
- Annotations **auto-save** every second. You can also click **✓ Save** in the top bar.

---

## Syncing annotations (important!)

Annotations are stored in `steps/06_annotation/outputs/segmentation.json`.
This file is shared between all three of you via Google Drive.

**Before each session — get the latest annotations:**
1. In Google Drive, go to `knesset-annotator/steps/06_annotation/outputs/`
2. Download the latest `segmentation.json`
3. Replace the file on your computer:
   - **Windows:** `%USERPROFILE%\Desktop\knesset-annotator\steps\06_annotation\outputs\segmentation.json`
   - **Mac:** `~/Desktop/knesset-annotator/steps/06_annotation/outputs/segmentation.json`
4. Then start the server

**After each session — upload your annotations:**
1. Find `segmentation.json` at the path above
2. Upload it to Google Drive (drag & drop, replacing the existing file)
3. Let Yoel/the others know you've pushed

> **Rule:** one person annotates at a time, or coordinate so you never annotate the
> same protocol. The file is a simple dictionary keyed by protocol ID — no conflicts
> as long as you don't edit the same protocol simultaneously.

---

## Troubleshooting

**"python is not recognized"**
→ Python isn't in your PATH. Re-run the Python installer and check "Add to PATH".

**"No module named flask"**
→ Run `pip install flask` in the terminal.

**"PROTOCOLS directory not found"**
→ Make sure you unzipped the Drive folder correctly and the path is
`Desktop\knesset-annotator\PROTOCOLS\` (not `Desktop\knesset-annotator\knesset-annotator\PROTOCOLS\`).

**Port 5000 already in use**
→ Another app is using port 5000. Edit `serve.py` last line: change `port=5000` to `port=5001`,
then open http://localhost:5001 instead.

**Hebrew text looks garbled**
→ Use Chrome or Firefox (not Internet Explorer/Edge Legacy). The tool uses
native browser RTL rendering.
