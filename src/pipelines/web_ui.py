"""
Web UI cho AIC HCMC 2026 — Competition Console với Temporal Navigation.

Tính năng:
- 4 task tabs: KIS / VKIS / VQA / TRAKE
- Temporal Navigation: click frame → timeline ±10 neighbors, keyboard ←/→
- Timer 5 phút countdown với visual alert
- Submit panel với log timestamp
- Keyboard shortcuts: 1-9 chọn frame, Enter tìm, Space/Enter submit, Esc clear, ←/→ navigate

Endpoints:
    GET  /                              SPA HTML
    POST /api/kis                       {query, topk} → results
    GET  /api/kis/neighbors/{vid}/{kf}  temporal neighbors
    POST /api/vkis                      multipart file → results
    POST /api/vqa                       {query, question} → answer
    POST /api/trake                     {events, video_id?} → ordered frames
    GET  /frame/{video_id}/{kf_n}       serve keyframe jpg
    GET  /api/status                    index stats

Run:
    python src/pipelines/web_ui.py
    → http://localhost:8010
"""
import os
import sys
import json
import glob
import csv
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "router"))
sys.path.insert(0, str(ROOT.parent / "core"))
sys.path.insert(0, str(ROOT.parent / "utils"))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn
from paths import INDEX_DIR, KEYFRAMES_DIR, RESULTS_DIR
from src.runtime_policy import RuntimePolicy

KF_DIR = str(KEYFRAMES_DIR)
IDX_DIR = str(INDEX_DIR)
SUBMISSION_SESSION_CSV = os.path.join(str(RESULTS_DIR), "submission_session.csv")
SUBMISSION_CODABENCH_CSV = os.path.join(str(RESULTS_DIR), "submission_codabench.csv")
ANNOTATION_PACK = Path(os.getenv("HCMAI_ANNOTATION_PACK", str(Path(ROOT.parent.parent) / "data/annotations/pilot_pack")))
ANNOTATION_CSV = ANNOTATION_PACK / "trake_independent.csv"
ANNOTATION_FIELDS = ["query_id", "video_id", "step", "caption", "kf_n", "frame_idx",
                     "pts_time", "provenance", "split", "annotator_id", "reviewer_id",
                     "confidence", "authoring_method", "target_selection_method"]

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_pipe = None
_runtime_policy = RuntimePolicy.from_env()
_nav = None
_expander_env = None
_obj_searcher = None
_warm_state = {
    "status": "cold",
    "model_ready": False,
    "started_at": None,
    "ready_at": None,
    "warmup_seconds": None,
    "error": None,
}


def _require_research_route(route_name: str) -> None:
    """Keep experimental endpoints out of the production surface by default."""
    if not _runtime_policy.research_routes_enabled:
        raise HTTPException(
            status_code=404,
            detail=f"research route {route_name!r} is disabled; set "
                   "HCMAI_ENABLE_RESEARCH_ROUTES=1 explicitly",
        )


def get_pipe():
    global _pipe
    if _pipe is None:
        from hcmai_pipeline import HCMAIPipeline
        _pipe = HCMAIPipeline(policy=_runtime_policy)
    return _pipe


def get_nav():
    global _nav
    if _nav is None:
        from temporal_nav import TemporalNavigator
        _nav = TemporalNavigator(IDX_DIR)
    return _nav


def get_expander_env():
    """Lazy-load .env credentials for query expansion."""
    global _expander_env
    if _expander_env is None:
        from query_expansion import _load_env
        _expander_env = _load_env()
    return _expander_env


def get_obj_searcher():
    """Lazy-load object searcher (requires object_index.pkl)."""
    global _obj_searcher
    if _obj_searcher is None:
        from object_search import ObjectSearcher
        try:
            _obj_searcher = ObjectSearcher(IDX_DIR)
        except FileNotFoundError:
            _obj_searcher = None  # index not built yet — hybrid falls back to visual
    return _obj_searcher


def _warmup_models() -> None:
    """Pre-load competition-critical state before the first query.

    KIS is the slow path because it imports open_clip/transformers and loads two
    SigLIP2 models. Load it once at server startup and keep the process resident
    so the first competition query only pays search latency.
    """
    global _warm_state
    start = time.time()
    _warm_state.update({
        "status": "warming",
        "model_ready": False,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ready_at": None,
        "warmup_seconds": None,
        "error": None,
    })
    print("[web_ui] warmup: loading KIS fusion retriever + timeline navigator...", flush=True)
    try:
        pipe = get_pipe()
        pipe._ensure_kis()
        get_nav()
        elapsed = round(time.time() - start, 2)
        _warm_state.update({
            "status": "ready",
            "model_ready": True,
            "ready_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "warmup_seconds": elapsed,
        })
        print(f"[web_ui] warmup: ready in {elapsed:.2f}s", flush=True)
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        _warm_state.update({
            "status": "error",
            "model_ready": False,
            "warmup_seconds": elapsed,
            "error": repr(e),
        })
        print(f"[web_ui] warmup: failed after {elapsed:.2f}s: {e!r}", flush=True)
        raise


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _warmup_models()
    yield


app = FastAPI(title="HCMAI 2026 Competition Console", version="2.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# HTML — single-page application
# ---------------------------------------------------------------------------
_HTML = r"""<!DOCTYPE html>
<html lang="vi" class="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HCMAI 2026 · Competition Console</title>
<script src="https://cdn.tailwindcss.com"></script>
<script>tailwind.config={darkMode:'class',theme:{extend:{fontFamily:{sans:['Inter','system-ui','sans-serif'],mono:['JetBrains Mono','monospace']}}}}</script>
<style>
  ::-webkit-scrollbar{width:6px;height:6px}
  ::-webkit-scrollbar-thumb{background:#3f3f46;border-radius:4px}
  ::-webkit-scrollbar-track{background:transparent}
  @keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
  @keyframes spin{to{transform:rotate(360deg)}}
  @keyframes pulse-red{0%,100%{opacity:1}50%{opacity:.5}}
  .fade{animation:fadeIn .2s ease}
  .spin{width:14px;height:14px;border:2px solid #22d3ee;border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite;display:inline-block;vertical-align:middle}
  .cell{transition:outline .1s}
  .cell.sel{outline:2px solid #22d3ee;outline-offset:-1px}
  .cell:hover{outline:1px solid #3f3f46;outline-offset:-1px}
  kbd{background:#18181b;border:1px solid #3f3f46;border-bottom-width:2px;border-radius:3px;padding:1px 5px;font-size:10px;font-family:monospace}
  .tab-active{background:#0e7490;color:#fff}
  .tab-inactive{color:#71717a}
  .tab-inactive:hover{color:#d4d4d8}

  /* Timeline strip */
  #timeline-strip{scrollbar-width:thin;scrollbar-color:#3f3f46 transparent}
  .tl-cell{flex-shrink:0;width:72px;cursor:pointer;border-radius:4px;overflow:hidden;border:1px solid transparent;transition:border-color .1s}
  .tl-cell:hover{border-color:#3f3f46}
  .tl-cell.tl-center{border-color:#22d3ee}
  /* H2: highlight ±2s window in amber */
  .tl-cell.tl-in2s{border-color:#d97706;background:rgba(69,26,3,0.20)}
  .tl-cell img{width:72px;height:40px;object-fit:cover;display:block}
  .tl-cell .tl-label{font-size:9px;text-align:center;padding:1px 0;background:#18181b;color:#71717a;font-family:monospace}
  .tl-cell.tl-center .tl-label{color:#22d3ee}

  /* Signal badge colours */
  .sig-asr{color:#34d399}.sig-ocr{color:#fbbf24}.sig-visual{color:#38bdf8}
  .sig-rerank{color:#c084fc}.sig-vkis{color:#f472b6}.sig-default{color:#a1a1aa}
</style>
</head>
<body class="bg-zinc-950 text-zinc-200 font-sans h-screen overflow-hidden select-none">

<div class="flex flex-col h-full">

  <!-- ── Header ── -->
  <header class="flex items-center justify-between px-4 h-11 border-b border-zinc-800 bg-black/60 shrink-0 backdrop-blur">
    <div class="flex items-center gap-3">
      <span class="font-bold text-cyan-400 tracking-tight text-sm">⬡ HCMAI&thinsp;2026</span>
      <span id="status-bar" class="text-[11px] text-zinc-600 font-mono">loading…</span>
    </div>
    <div class="flex items-center gap-3 text-sm">
      <span id="timer-display" class="font-mono text-base tabular-nums w-12 text-center text-zinc-300">05:00</span>
      <button id="timer-toggle" onclick="timerToggle()" class="text-[11px] px-2 py-0.5 rounded bg-zinc-800 hover:bg-zinc-700 transition">Start</button>
      <button onclick="timerReset()" class="text-[11px] px-2 py-0.5 rounded bg-zinc-800 hover:bg-zinc-700 transition">Reset</button>
    </div>
  </header>

  <div class="flex flex-1 min-h-0">

    <!-- ── Left panel ── -->
    <aside class="w-72 shrink-0 border-r border-zinc-800 flex flex-col bg-zinc-950">

      <!-- Task tabs -->
      <div class="grid grid-cols-4 border-b border-zinc-800 text-xs font-semibold">
        <button data-task="kis"   class="task-tab py-2.5 tab-active  transition">KIS</button>
        <button data-task="vkis"  class="task-tab py-2.5 tab-inactive transition">VKIS</button>
        <button data-task="vqa"   class="task-tab py-2.5 tab-inactive transition">VQA</button>
        <button data-task="trake" class="task-tab py-2.5 tab-inactive transition">TRAKE</button>
      </div>

      <!-- Task panels -->
      <div class="flex-1 overflow-y-auto p-3 space-y-3">

        <!-- KIS panel -->
        <div id="panel-kis" class="task-panel space-y-2">
          <label class="block text-[10px] uppercase tracking-wider text-zinc-500">Truy vấn</label>
          <textarea id="kis-q" rows="3"
            placeholder="Nhập mô tả cảnh… (Enter để tìm, Shift+Enter xuống dòng)"
            class="w-full bg-zinc-900 border border-zinc-800 rounded px-3 py-2 text-sm resize-none
                   focus:outline-none focus:ring-1 focus:ring-cyan-600 placeholder-zinc-700
                   leading-relaxed"></textarea>
          <div class="grid grid-cols-2 gap-2">
            <label class="block text-[10px] uppercase tracking-wider text-zinc-500">Top-K</label>
            <input id="kis-topk" type="number" min="1" max="100" value="20"
              class="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1 text-xs font-mono
                     focus:outline-none focus:border-cyan-600">
          </div>
          <button id="kis-search-btn" onclick="runKIS()"
            class="w-full py-1.5 bg-cyan-700 hover:bg-cyan-600 rounded text-sm font-medium transition">
            Tìm <kbd>↵</kbd>
          </button>
          <div class="space-y-1">
            <p class="text-[10px] text-zinc-600 uppercase tracking-wider">Ví dụ</p>
            <button class="ex-chip block text-left text-[11px] text-zinc-500 hover:text-cyan-400 transition truncate w-full">siêu bão Biển Đông cấp 16</button>
            <button class="ex-chip block text-left text-[11px] text-zinc-500 hover:text-cyan-400 transition truncate w-full">đua xe đạp HTV cup chặng cuối</button>
            <button class="ex-chip block text-left text-[11px] text-zinc-500 hover:text-cyan-400 transition truncate w-full">nấu xào thịt bò với hành tây</button>
          </div>
        </div>

        <!-- VKIS panel -->
        <div id="panel-vkis" class="task-panel hidden space-y-2">
          <label class="block text-[10px] uppercase tracking-wider text-zinc-500">Ảnh / Clip truy vấn</label>
          <div id="drop-zone"
            class="border-2 border-dashed border-zinc-700 rounded p-5 text-center text-xs text-zinc-500 transition cursor-pointer hover:border-zinc-500"
            onclick="document.getElementById('vkis-file').click()">
            <input type="file" id="vkis-file" accept=".jpg,.jpeg,.png,.mp4,.mov" class="hidden">
            <p>Kéo thả hoặc <span class="text-cyan-400">chọn file</span></p>
            <p id="vkis-filename" class="text-cyan-300 mt-1 truncate hidden"></p>
          </div>
          <div class="grid grid-cols-2 gap-2">
            <label class="block text-[10px] uppercase tracking-wider text-zinc-500">Top-K</label>
            <input id="vkis-topk" type="number" min="1" max="100" value="10"
              class="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1 text-xs font-mono
                     focus:outline-none focus:border-cyan-600">
          </div>
          <button id="vkis-search-btn" onclick="runVKIS()"
            class="w-full py-1.5 bg-cyan-700 hover:bg-cyan-600 rounded text-sm font-medium transition">
            Tìm tương tự
          </button>
        </div>

        <!-- VQA panel -->
        <div id="panel-vqa" class="task-panel hidden space-y-2">
          <label class="block text-[10px] uppercase tracking-wider text-zinc-500">Mô tả cảnh</label>
          <input id="vqa-q" placeholder="Mô tả khoảnh khắc cần tìm…"
            class="w-full bg-zinc-900 border border-zinc-800 rounded px-3 py-2 text-sm
                   focus:outline-none focus:ring-1 focus:ring-cyan-600 placeholder-zinc-700">
          <label class="block text-[10px] uppercase tracking-wider text-zinc-500">Câu hỏi</label>
          <input id="vqa-question" placeholder="Câu hỏi về khoảnh khắc đó…"
            class="w-full bg-zinc-900 border border-zinc-800 rounded px-3 py-2 text-sm
                   focus:outline-none focus:ring-1 focus:ring-cyan-600 placeholder-zinc-700">
          <button id="vqa-search-btn" onclick="runVQA()"
            class="w-full py-1.5 bg-cyan-700 hover:bg-cyan-600 rounded text-sm font-medium transition">
            Trả lời
          </button>
          <div id="vqa-answer"></div>
        </div>

        <!-- TRAKE panel -->
        <div id="panel-trake" class="task-panel hidden space-y-2">
          <label class="block text-[10px] uppercase tracking-wider text-zinc-500">Sub-events (mỗi dòng 1 event)</label>
          <textarea id="trake-events" rows="6"
            placeholder="Mở đầu bản tin&#10;Báo cáo thiên tai&#10;Hoạt động ngoại giao&#10;Kết thúc thời tiết"
            class="w-full bg-zinc-900 border border-zinc-800 rounded px-3 py-2 text-sm resize-none
                   focus:outline-none focus:ring-1 focus:ring-cyan-600 placeholder-zinc-700
                   leading-relaxed"></textarea>
          <p class="text-[10px] text-zinc-500">Production channel: <span class="text-pink-300">ASR</span></p>
          <button id="trake-search-btn" onclick="runTRAKE()"
            class="w-full py-1.5 bg-cyan-700 hover:bg-cyan-600 rounded text-sm font-medium transition">
            Căn chỉnh chuỗi
          </button>
        </div>
      </div>

      <!-- Signal indicator -->
      <div class="px-3 py-2 border-t border-zinc-800">
        <p class="text-[10px] uppercase tracking-wider text-zinc-600 mb-1">Signal</p>
        <div id="signal-indicator" class="text-xs sig-default">—</div>
      </div>

      <!-- Submit panel -->
      <div class="px-3 py-3 border-t border-zinc-800 bg-zinc-900/50 space-y-2">
        <label class="block text-[10px] uppercase tracking-wider text-zinc-600">Query ID</label>
        <input id="submit-query-id" placeholder="q001 / BTC id"
          class="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1 text-xs font-mono
                 focus:outline-none focus:border-cyan-600">
        <p class="text-[10px] uppercase tracking-wider text-zinc-600">Đã chọn</p>
        <p id="selected-label" class="font-mono text-sm text-cyan-300 truncate">— chưa chọn —</p>
        <button id="submit-btn" onclick="submitSelected()" disabled
          class="w-full py-1.5 bg-emerald-700 hover:bg-emerald-600 disabled:opacity-30 disabled:cursor-not-allowed
                 rounded text-sm font-medium transition">
          Submit <kbd>Space</kbd>
        </button>
        <button onclick="exportSubmission()"
          class="w-full py-1.5 bg-zinc-800 hover:bg-zinc-700 rounded text-xs font-medium transition">
          Export CSV
        </button>
        <div id="submit-log" class="text-[11px] text-zinc-500 max-h-20 overflow-y-auto space-y-0.5"></div>
        <div class="pt-1 border-t border-zinc-800/80 space-y-1">
          <div class="flex items-center justify-between">
            <p class="text-[10px] uppercase tracking-wider text-zinc-600">Submission manager</p>
            <button onclick="reloadSubmissionManager()" class="text-[10px] text-cyan-400 hover:text-cyan-300">↻</button>
          </div>
          <div id="submission-list" class="space-y-1 max-h-28 overflow-y-auto"></div>
        </div>
      </div>

      <!-- Keyboard hint -->
      <div class="px-3 py-2 border-t border-zinc-800 text-[10px] text-zinc-700 leading-relaxed">
        <kbd>1</kbd>–<kbd>9</kbd> select · <kbd>↵</kbd> search · <kbd>Space</kbd> submit
        · <kbd>←</kbd><kbd>→</kbd> timeline · <kbd>Esc</kbd> clear
      </div>
    </aside>

    <!-- ── Main content ── -->
    <div class="flex-1 flex flex-col min-w-0 overflow-hidden">

      <!-- Results grid -->
      <main class="flex-1 overflow-y-auto p-3">
        <div id="loading-indicator" class="hidden text-center text-zinc-500 text-sm py-12">
          <span class="spin"></span> <span class="ml-2">đang tìm…</span>
        </div>
        <div id="empty-state" class="text-center text-zinc-700 text-sm py-20">
          Nhập query để bắt đầu
        </div>
        <div id="results-grid" class="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-2"></div>
      </main>

      <!-- ── Temporal Navigation Strip ── -->
      <div id="timeline-panel" class="hidden border-t border-zinc-800 bg-black/40 shrink-0">
        <div class="flex items-center gap-2 px-3 py-1.5 border-b border-zinc-900">
          <span class="text-[10px] uppercase tracking-wider text-zinc-600">Timeline</span>
          <span id="timeline-video-label" class="text-[11px] font-mono text-zinc-400"></span>
          <span id="timeline-time-label" class="text-[11px] font-mono text-cyan-400 ml-auto"></span>
          <button onclick="setTimelineWindow(2)" class="text-[10px] text-zinc-500 hover:text-cyan-300">±2s</button>
          <button onclick="setTimelineWindow(5)" class="text-[10px] text-zinc-500 hover:text-cyan-300">±5s</button>
          <button onclick="hideTimeline()" class="text-[10px] text-zinc-600 hover:text-zinc-400 ml-2">✕</button>
        </div>
        <div class="flex items-center px-2 py-1.5 gap-1">
          <button onclick="timelineStep(-1)" title="Previous (←)"
            class="shrink-0 w-7 h-7 flex items-center justify-center rounded bg-zinc-800 hover:bg-zinc-700 text-xs transition">‹</button>
          <div id="timeline-strip" class="flex gap-1 overflow-x-auto flex-1 py-0.5 scroll-smooth"></div>
          <button onclick="timelineStep(+1)" title="Next (→)"
            class="shrink-0 w-7 h-7 flex items-center justify-center rounded bg-zinc-800 hover:bg-zinc-700 text-xs transition">›</button>
        </div>
      </div>

    </div>
  </div>
</div>

<!-- Lightbox -->
<div id="lightbox" class="fixed inset-0 bg-black/95 hidden z-50 items-center justify-center p-6"
     onclick="this.style.display='none'">
  <img id="lightbox-img" class="max-h-full max-w-full rounded object-contain">
</div>

<script>
// ─────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────
let currentTask = 'kis';
let lastResults = [];
let selIdx = -1;
let tlVideo = null;
let tlKfN = null;      // current center keyframe in timeline
let tlNeighbors = [];  // cached neighbors array
let submissionCount = 0;
let submissionRows = [];
let editingQueryId = null;

// ─────────────────────────────────────────────────────────
// Task switching
// ─────────────────────────────────────────────────────────
function setTask(task) {
  currentTask = task;
  document.querySelectorAll('.task-panel').forEach(p => p.classList.add('hidden'));
  document.getElementById('panel-' + task).classList.remove('hidden');
  document.querySelectorAll('.task-tab').forEach(btn => {
    const active = btn.dataset.task === task;
    btn.classList.toggle('tab-active', active);
    btn.classList.toggle('tab-inactive', !active);
  });
  clearSubmissionEdit();
}
document.querySelectorAll('.task-tab').forEach(btn => btn.addEventListener('click', () => setTask(btn.dataset.task)));
document.querySelectorAll('.ex-chip').forEach(btn => btn.addEventListener('click', () => {
  document.getElementById('kis-q').value = btn.textContent.trim();
  runKIS();
}));
document.getElementById('kis-q').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runKIS(); }
});

// VKIS drag-and-drop
const dropZone = document.getElementById('drop-zone');
const vkisFile = document.getElementById('vkis-file');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('border-cyan-500'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('border-cyan-500'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('border-cyan-500');
  const f = e.dataTransfer.files[0];
  if (f) { showVkisFile(f); }
});
vkisFile.addEventListener('change', () => { if (vkisFile.files[0]) showVkisFile(vkisFile.files[0]); });
function showVkisFile(f) {
  const dt = new DataTransfer();
  dt.items.add(f);
  vkisFile.files = dt.files;
  const el = document.getElementById('vkis-filename');
  el.textContent = f.name;
  el.classList.remove('hidden');
}

// ─────────────────────────────────────────────────────────
// Timer
// ─────────────────────────────────────────────────────────
let timerSecondsLeft = 300;
let timerInterval = null;
let timerRunning = false;

function timerFmt(s) {
  return String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0');
}
function timerTick() {
  timerSecondsLeft = Math.max(0, timerSecondsLeft - 1);
  const el = document.getElementById('timer-display');
  el.textContent = timerFmt(timerSecondsLeft);
  el.classList.toggle('text-red-400', timerSecondsLeft <= 30);
  el.classList.toggle('text-zinc-300', timerSecondsLeft > 30);
  if (timerSecondsLeft === 0) timerStop();
}
function timerStop() {
  clearInterval(timerInterval);
  timerInterval = null;
  timerRunning = false;
  document.getElementById('timer-toggle').textContent = 'Start';
}
function timerToggle() {
  if (timerRunning) {
    timerStop();
    document.getElementById('timer-toggle').textContent = 'Resume';
  } else {
    timerInterval = setInterval(timerTick, 1000);
    timerRunning = true;
    document.getElementById('timer-toggle').textContent = 'Pause';
  }
}
function timerReset() {
  timerStop();
  timerSecondsLeft = 300;
  const el = document.getElementById('timer-display');
  el.textContent = '05:00';
  el.classList.remove('text-red-400');
  el.classList.add('text-zinc-300');
  document.getElementById('timer-toggle').textContent = 'Start';
}

// ─────────────────────────────────────────────────────────
// Loading state
// ─────────────────────────────────────────────────────────
function setLoading(on) {
  document.getElementById('loading-indicator').classList.toggle('hidden', !on);
  document.getElementById('empty-state').classList.add('hidden');
  if (on) document.getElementById('results-grid').innerHTML = '';
}

// ─────────────────────────────────────────────────────────
// Signal indicator
// ─────────────────────────────────────────────────────────
const SIG_CLASS = {
  asr: 'sig-asr', ocr_forced: 'sig-ocr', ocr: 'sig-ocr',
  visual: 'sig-visual', visual_forced: 'sig-visual',
  rerank: 'sig-rerank', vkis: 'sig-vkis', trake_asr: 'sig-asr'
};
function showSignal(winner, count) {
  const el = document.getElementById('signal-indicator');
  const cls = SIG_CLASS[winner] || 'sig-default';
  el.className = 'text-xs ' + cls;
  el.textContent = winner + (count !== undefined ? ' · ' + count + ' results' : '');
}

function groupVKISResults(items) {
  const byVideo = new Map();
  for (const r of items || []) {
    const cur = byVideo.get(r.video_id);
    if (!cur) {
      byVideo.set(r.video_id, { best: r, items: [r] });
      continue;
    }
    cur.items.push(r);
    if ((r.score || 0) > (cur.best.score || 0)) cur.best = r;
  }
  return Array.from(byVideo.values())
    .sort((a, b) => (b.best.score || 0) - (a.best.score || 0))
    .map(g => ({
      video_id: g.best.video_id,
      frame_idx: g.best.frame_idx,
      kf_n: g.best.kf_n,
      pts_time: g.best.pts_time,
      score: g.best.score,
      hit_count: g.items.length,
      hits: g.items.slice(0, 3),
    }));
}

function renderSubmissionManager(rows) {
  submissionRows = rows || [];
  const box = document.getElementById('submission-list');
  if (!box) return;
  if (!submissionRows.length) {
    box.innerHTML = '<p class="text-[10px] text-zinc-600">Chưa có submission</p>';
    return;
  }
  box.innerHTML = submissionRows.map(r => `
    <div class="flex items-center gap-1 rounded border ${editingQueryId === r.query_id ? 'border-cyan-600 bg-cyan-950/30' : 'border-zinc-800 bg-zinc-950/40'} px-2 py-1 text-[10px]">
      <button class="text-left flex-1 min-w-0 hover:text-cyan-300" onclick='loadSubmissionForEdit(${JSON.stringify(r)})'>
        <span class="font-mono text-zinc-300">${r.query_id}</span>
        <span class="text-zinc-600"> · </span>
        <span class="text-zinc-400">${r.task}</span>
        <span class="text-zinc-600"> · </span>
        <span class="text-zinc-400 truncate">${r.video_name}</span>
        <span class="text-zinc-600"> · </span>
        <span class="font-mono text-zinc-500">${r.frame_idx}</span>
      </button>
      <button class="text-zinc-500 hover:text-amber-300" title="Delete" onclick='deleteSubmission(${JSON.stringify(r.query_id)})'>✕</button>
    </div>
  `).join('');
}

async function reloadSubmissionManager() {
  try {
    const resp = await fetch('/api/submit/list');
    if (!resp.ok) return;
    const data = await resp.json();
    submissionCount = data.count || 0;
    renderSubmissionManager(data.rows || []);
    const qidInput = document.getElementById('submit-query-id');
    if (qidInput && !qidInput.value.trim()) qidInput.placeholder = `q${String(submissionCount + 1).padStart(3, '0')}`;
  } catch (e) {
    console.error('reloadSubmissionManager failed:', e);
  }
}

function clearSubmissionEdit() {
  editingQueryId = null;
  const btn = document.getElementById('submit-btn');
  if (btn) btn.innerHTML = 'Submit <kbd>Space</kbd>';
  const qidInput = document.getElementById('submit-query-id');
  if (qidInput && !qidInput.value.trim()) qidInput.placeholder = `q${String(submissionCount + 1).padStart(3, '0')}`;
  renderSubmissionManager(submissionRows);
}

function loadSubmissionForEdit(r) {
  editingQueryId = r.query_id;
  document.getElementById('submit-query-id').value = r.query_id;
  const btn = document.getElementById('submit-btn');
  if (btn) btn.innerHTML = 'Update <kbd>Space</kbd>';
  const lbl = document.getElementById('selected-label');
  lbl.textContent = `${r.video_name}, frame ${r.frame_idx}`;
  const found = lastResults.findIndex(x => x.video_id === r.video_name && Number(x.frame_idx) === Number(r.frame_idx));
  if (found >= 0) {
    selectCell(found);
  } else {
    lastResults[-2] = {
      video_id: r.video_name,
      kf_n: r.kf_n ? Number(r.kf_n) : 1,
      frame_idx: Number(r.frame_idx),
      pts_time: Number(r.pts_time || 0),
      score: Number(r.score || 0),
      _from_submission: true,
    };
    selIdx = -2;
    updateSelection();
  }
  renderSubmissionManager(submissionRows);
}

async function deleteSubmission(queryId) {
  try {
    const resp = await fetch('/api/submit/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query_id: queryId })
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    submissionCount = data.count || submissionCount;
    if (editingQueryId === queryId) clearSubmissionEdit();
    await reloadSubmissionManager();
  } catch (e) {
    console.error('deleteSubmission failed:', e);
  }
}

// ─────────────────────────────────────────────────────────
// Results rendering
// ─────────────────────────────────────────────────────────
function renderResults(items) {
  const displayItems = currentTask === 'vkis' ? groupVKISResults(items) : items;
  lastResults = displayItems;
  selIdx = -1;
  updateSelection();
  hideTimeline();

  const grid = document.getElementById('results-grid');
  document.getElementById('empty-state').classList.add('hidden');

  grid.innerHTML = displayItems.map((r, i) => `
    <div class="cell fade bg-zinc-900 rounded overflow-hidden border border-zinc-800 cursor-pointer"
          data-idx="${i}" onclick="selectCell(${i})">
      <div class="relative aspect-video bg-zinc-800 overflow-hidden group">
        <img src="/frame/${r.video_id}/${r.kf_n || 1}" loading="lazy"
             class="w-full h-full object-cover"
             ondblclick="event.stopPropagation(); openLightbox(this.src)">
        <span class="absolute top-1 left-1 text-[10px] bg-black/75 px-1.5 py-0.5 rounded font-mono">
          ${i < 9 ? i + 1 : '·'}
        </span>
        <span class="absolute top-1 right-1 text-[10px] bg-cyan-700/80 px-1.5 py-0.5 rounded font-mono">
          ${r.score.toFixed(3)}
        </span>
        ${currentTask === 'vkis' ? `
        <span class="absolute bottom-1 left-1 text-[10px] bg-black/75 px-1.5 py-0.5 rounded font-mono text-amber-300">
          ${r.hit_count} hits
        </span>` : ''}
        <button class="absolute bottom-1 right-1 opacity-0 group-hover:opacity-100 transition
                       text-[10px] bg-black/75 hover:bg-zinc-700 px-1.5 py-0.5 rounded"
                onclick="event.stopPropagation(); openTimeline('${r.video_id}', ${r.kf_n || 1}, ${r.pts_time.toFixed(1)})"
                title="Open timeline (temporal navigation)">
          ⏱ timeline
        </button>
      </div>
      <div class="px-2 py-1.5">
        <p class="text-[11px] font-medium text-zinc-200 truncate">${r.video_id}</p>
        <p class="text-[10px] text-zinc-500 font-mono">frame ${r.frame_idx} · ${r.pts_time.toFixed(1)}s</p>
         ${currentTask === 'vkis' ? `<p class="text-[10px] text-zinc-600 truncate">${(r.hits || []).map(x => `${x.frame_idx}`).join(', ')}</p>` : ''}
      </div>
      ${currentTask === 'kis' && r.peaks ? `<div class="px-2 pb-2 flex gap-1 overflow-x-auto" onclick="event.stopPropagation()">
        ${r.peaks.map((p, pi) => `<button class="shrink-0 text-[9px] bg-zinc-800 hover:bg-cyan-800 border border-zinc-700 rounded px-1.5 py-1 font-mono" onclick="openPeak('${r.video_id}', ${p.kf_n}, ${p.pts_time}, ${p.frame_idx})">P${pi + 1} · ${p.pts_time.toFixed(1)}s</button>`).join('')}
      </div>` : ''}
    </div>
  `).join('');
}

// ─────────────────────────────────────────────────────────
// Selection
// ─────────────────────────────────────────────────────────
function selectCell(idx) {
  selIdx = idx;
  updateSelection();
  // Auto-open timeline for selected frame
  const r = lastResults[idx];
  if (r) openTimeline(r.video_id, r.kf_n || 1, r.pts_time);
}

function getSelectedResult() {
  if (selIdx >= 0) return lastResults[selIdx];
  if (selIdx === -2) return lastResults[-2];  // timeline-selected synthetic result
  return null;
}

function updateSelection() {
  document.querySelectorAll('.cell').forEach(c => c.classList.toggle('sel', +c.dataset.idx === selIdx));
  const btn = document.getElementById('submit-btn');
  const lbl = document.getElementById('selected-label');
  const r = getSelectedResult();
  if (r) {
    lbl.textContent = r.video_id + ', frame ' + r.frame_idx;
    btn.disabled = false;
  } else {
    lbl.textContent = '— chưa chọn —';
    btn.disabled = true;
  }
}

async function submitSelected() {
  const r = getSelectedResult();
  if (!r) return;
  const t = document.getElementById('timer-display').textContent;
  const log = document.getElementById('submit-log');
  const entry = document.createElement('div');
  const qidInput = document.getElementById('submit-query-id');
  const queryId = qidInput.value.trim() || `q${String(submissionCount + 1).padStart(3, '0')}`;
  const endpoint = editingQueryId ? '/api/submit/update' : '/api/submit';

  try {
    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query_id: queryId,
        original_query_id: editingQueryId,
        task: currentTask.toUpperCase(),
        video_name: r.video_id,
        frame_idx: r.frame_idx,
        kf_n: r.kf_n || null,
        pts_time: r.pts_time || null,
        score: r.score || null,
        remaining_time: t,
      })
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    submissionCount = data.count || (submissionCount + 1);
    entry.className = 'text-emerald-400 fade';
    entry.textContent = `${editingQueryId ? '↺' : '✓'} ${queryId}: ${r.video_id}, ${r.frame_idx} @${t}`;
    log.prepend(entry);
    qidInput.value = '';
    qidInput.placeholder = `q${String(submissionCount + 1).padStart(3, '0')}`;
    editingQueryId = null;
    const btn = document.getElementById('submit-btn');
    if (btn) btn.innerHTML = 'Submit <kbd>Space</kbd>';
    await reloadSubmissionManager();
  } catch (e) {
    entry.className = 'text-red-400 fade';
    entry.textContent = `✗ submit failed: ${String(e).slice(0, 80)}`;
    log.prepend(entry);
  }
}

function exportSubmission() {
  window.open('/api/submit/export?format=codabench', '_blank');
}

// ─────────────────────────────────────────────────────────
// Temporal Navigation
// ─────────────────────────────────────────────────────────
async function openTimeline(videoId, kfN, ptsTime) {
  tlVideo = videoId;
  tlKfN = kfN;
  document.getElementById('timeline-video-label').textContent = videoId;
  document.getElementById('timeline-time-label').textContent = ptsTime.toFixed ? ptsTime.toFixed(1) + 's' : ptsTime + 's';
  document.getElementById('timeline-panel').classList.remove('hidden');
  await loadTimelineNeighborsByTime(videoId, ptsTime, 5);
}

function hideTimeline() {
  document.getElementById('timeline-panel').classList.add('hidden');
  tlVideo = null;
  tlKfN = null;
  tlNeighbors = [];
}

async function loadTimelineNeighbors(videoId, kfN, window = 10) {
  try {
    const resp = await fetch(`/api/kis/neighbors/${encodeURIComponent(videoId)}/${kfN}?window=${window}`);
    if (!resp.ok) return;
    const data = await resp.json();
    tlNeighbors = data.neighbors;
    renderTimeline(data);
  } catch (e) {
    console.error('Timeline load failed:', e);
  }
}

async function loadTimelineNeighborsByTime(videoId, ptsTime, halfWindow = 5) {
  try {
    const resp = await fetch(`/api/kis/neighbors_by_time/${encodeURIComponent(videoId)}?pts_time=${encodeURIComponent(ptsTime)}&half_window_s=${encodeURIComponent(halfWindow)}`);
    if (!resp.ok) return;
    const data = await resp.json();
    tlNeighbors = data.neighbors || [];
    renderTimeline(data);
  } catch (e) {
    console.error('Timeline load failed:', e);
  }
}

// tlCenterPts = pts_time of current timeline center (for delta display)
let tlCenterPts = null;

function renderTimeline(data) {
  const strip = document.getElementById('timeline-strip');
  const centerPts = data.neighbors.find(n => n.is_center)?.pts_time ?? null;
  tlCenterPts = centerPts;
  strip.innerHTML = data.neighbors.map(n => {
    const delta = (centerPts !== null && !n.is_center)
      ? (n.pts_time - centerPts).toFixed(1).replace(/^([^-])/, '+$1') + 's'
      : '';
    // H1 FIX: frame_idx already in n.frame_idx from temporal_nav — pass directly,
    // no async fetch needed. submit button will always have the real frame_idx.
    const in2s = centerPts !== null && Math.abs(n.pts_time - centerPts) <= 2.0;
    return `
    <div class="tl-cell ${n.is_center ? 'tl-center' : ''} ${in2s && !n.is_center ? 'tl-in2s' : ''}"
         onclick="timelineJump('${data.video_id}', ${n.kf_n}, ${n.pts_time}, ${n.frame_idx})">
      <img src="/frame/${data.video_id}/${n.kf_n}" loading="lazy" alt="kf ${n.kf_n}">
      <div class="tl-label">${n.is_center ? n.pts_time.toFixed(1)+'s' : delta || n.pts_time.toFixed(0)+'s'}</div>
    </div>`;
  }).join('');
  const center = strip.querySelector('.tl-center');
  if (center) center.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
}

// H1 FIX: timelineJump now receives frame_idx directly — no async fetch, no race condition.
function timelineJump(videoId, kfN, ptsTime, frameIdx) {
  tlKfN = kfN;
  document.getElementById('timeline-time-label').textContent = ptsTime.toFixed(1) + 's';
  _setTimelineSelected(videoId, kfN, ptsTime, frameIdx);
  loadTimelineNeighborsByTime(videoId, ptsTime, 5);
}

function _setTimelineSelected(videoId, kfN, ptsTime, frameIdx) {
  // Match existing grid card first
  const matchIdx = lastResults.findIndex(r => r.video_id === videoId && r.kf_n === kfN);
  if (matchIdx >= 0) {
    selectCell(matchIdx);
    return;
  }
  // H1 FIX: frame_idx is now passed in directly — no approximation, no async.
  const synth = {
    video_id: videoId,
    kf_n: kfN,
    frame_idx: (frameIdx !== undefined && frameIdx !== null) ? frameIdx : kfN,
    pts_time: ptsTime,
    score: 0,
    _from_timeline: true,
  };
  lastResults[-2] = synth;
  selIdx = -2;
  document.querySelectorAll('.cell').forEach(c => c.classList.remove('sel'));
  const lbl = document.getElementById('selected-label');
  lbl.textContent = `${videoId}, frame ${synth.frame_idx} (${ptsTime.toFixed(1)}s)`;
  document.getElementById('submit-btn').disabled = false;
}

function openPeak(videoId, kfN, ptsTime, frameIdx) {
  openTimeline(videoId, kfN, ptsTime);
  _setTimelineSelected(videoId, kfN, ptsTime, frameIdx);
}

function timelineStep(dir) {
  if (!tlVideo || !tlKfN) return;
  const curIdx = tlNeighbors.findIndex(n => n.is_center);
  if (curIdx < 0) return;
  const target = tlNeighbors[curIdx + dir];
  if (target) {
    // H1 FIX: target already has frame_idx from neighbors payload
    timelineJump(tlVideo, target.kf_n, target.pts_time, target.frame_idx);
  } else {
    loadTimelineNeighborsByTime(tlVideo, tlCenterPts ?? 0, 5);
  }
}

function setTimelineWindow(sec) {
  if (!tlVideo) return;
  const center = tlNeighbors.find(n => n.is_center)?.pts_time ?? tlCenterPts;
  if (center === null || center === undefined) return;
  loadTimelineNeighborsByTime(tlVideo, center, sec);
}

// ─────────────────────────────────────────────────────────
// Lightbox
// ─────────────────────────────────────────────────────────
function openLightbox(src) {
  document.getElementById('lightbox-img').src = src;
  const lb = document.getElementById('lightbox');
  lb.style.display = 'flex';
}

// ─────────────────────────────────────────────────────────
// Keyboard shortcuts
// ─────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  const tag = e.target.tagName;
  const inInput = tag === 'INPUT' || tag === 'TEXTAREA';

  // Number keys: select result card
  if (!inInput && e.key >= '1' && e.key <= '9') {
    const idx = +e.key - 1;
    if (idx < lastResults.length) selectCell(idx);
    return;
  }

  // Arrow keys: timeline navigation (only when timeline is visible)
  if (!inInput && !document.getElementById('timeline-panel').classList.contains('hidden')) {
    if (e.key === 'ArrowLeft')  { e.preventDefault(); timelineStep(-1); return; }
    if (e.key === 'ArrowRight') { e.preventDefault(); timelineStep(+1); return; }
  }

  // Space / Enter: submit
  if (!inInput && (e.key === ' ' || e.key === 'Enter')) {
    e.preventDefault();
    submitSelected();
    return;
  }

  // Escape: clear results
  if (!inInput && e.key === 'Escape') {
    document.getElementById('results-grid').innerHTML = '';
    document.getElementById('empty-state').classList.remove('hidden');
    lastResults = [];
    selIdx = -1;
    updateSelection();
    showSignal('—');
    hideTimeline();
    return;
  }
});

// ─────────────────────────────────────────────────────────
// API calls
// ─────────────────────────────────────────────────────────
async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

async function runKIS() {
  const q = document.getElementById('kis-q').value.trim();
  if (!q) return;
  setLoading(true);
  try {
    const topk = Math.max(1, Math.min(100, parseInt(document.getElementById('kis-topk').value || '20', 10)));
    const data = await postJSON('/api/kis', { query: q, topk, include_peaks: true, peaks_per_video: 5 });
    showSignal(data.winner || 'visual', (data.results || []).length);
    const peaksByVideo = Object.fromEntries((data.peaks || []).map(x => [x.video_id, x.peaks || []]));
    renderResults((data.results || []).map(x => ({ ...x, peaks: peaksByVideo[x.video_id] || [] })));
  } catch (err) {
    console.error('KIS error:', err);
    document.getElementById('empty-state').textContent = 'Lỗi: ' + err.message;
    document.getElementById('empty-state').classList.remove('hidden');
  } finally {
    setLoading(false);
  }
}

async function runVKIS() {
  const file = document.getElementById('vkis-file').files[0];
  if (!file) { alert('Chọn ảnh hoặc clip trước'); return; }
  setLoading(true);
  try {
    const topk = Math.max(1, Math.min(100, parseInt(document.getElementById('vkis-topk').value || '10', 10)));
    const fd = new FormData();
    fd.append('file', file);
    const resp = await fetch(`/api/vkis?topk=${topk}`, { method: 'POST', body: fd });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    showSignal('vkis', Math.min(topk, (data.results || []).length));
    renderResults(data.results || []);
  } catch (err) {
    console.error('VKIS error:', err);
  } finally {
    setLoading(false);
  }
}

async function runVQA() {
  const q = document.getElementById('vqa-q').value.trim();
  const question = document.getElementById('vqa-question').value.trim();
  if (!q || !question) { alert('Nhập cả mô tả và câu hỏi'); return; }
  setLoading(true);
  document.getElementById('vqa-answer').innerHTML = '';
  try {
    const data = await postJSON('/api/vqa', { query: q, question, mode: 'interactive' });
    setLoading(false);
    if (data.best) {
      const b = data.best;
      document.getElementById('vqa-answer').innerHTML = `
        <div class="fade bg-zinc-900 border border-zinc-800 rounded p-2 mt-1 text-xs space-y-1">
          <p class="text-zinc-500 font-mono">${b.video}</p>
          <p class="text-emerald-300 font-medium">${b.answer}</p>
        </div>`;
      renderResults([{
        video_id: b.video, frame_idx: b.frame_idx,
        pts_time: b.pts_time, kf_n: b.kf_n || 1,
        score: b.vlm_score || 0
      }]);
      showSignal('visual', 1);
    } else {
      document.getElementById('vqa-answer').innerHTML =
        '<p class="text-xs text-red-400 mt-1">Không tìm được kết quả</p>';
    }
  } catch (err) {
    setLoading(false);
    console.error('VQA error:', err);
  }
}

async function runTRAKE() {
  const lines = document.getElementById('trake-events').value
    .split('\n').map(s => s.trim()).filter(Boolean);
  if (lines.length < 2) { alert('Cần ít nhất 2 sub-event'); return; }
  setLoading(true);
  try {
    // Let the shared production policy choose visual TRAKE. ASR is an
    // explicit diagnostic fallback and must never be forced by the UI.
    const data = await postJSON('/api/trake', { events: lines, topk: 1 });
    setLoading(false);
    const results = data.results || [];
    if (results.length) {
      const r = results[0];
      showSignal(data.winner || 'trake_visual', r.path ? r.path.length : 0);
      renderResults((r.path || []).map(s => ({
        video_id: r.video_id,
        frame_idx: s.frame_idx,
        pts_time: Number((s.pts_time !== undefined && s.pts_time !== null) ? s.pts_time : (s.start || 0)),
        kf_n: s.kf_n || 1,
        score: Number(s.score !== undefined && s.score !== null ? s.score : (r.score || 0))
      })));
    }
  } catch (err) {
    setLoading(false);
    console.error('TRAKE error:', err);
  }
}

// ─────────────────────────────────────────────────────────
// Status bar
// ─────────────────────────────────────────────────────────
function setSearchReady(ready, label) {
  ['kis-search-btn', 'vkis-search-btn', 'vqa-search-btn', 'trake-search-btn'].forEach(id => {
    const btn = document.getElementById(id);
    if (!btn) return;
    btn.disabled = !ready;
    btn.classList.toggle('opacity-40', !ready);
    btn.classList.toggle('cursor-not-allowed', !ready);
  });
  if (label) document.getElementById('signal-indicator').textContent = label;
}

async function refreshHealth() {
  const h = await fetch('/api/health').then(r => r.json());
  const ready = !!h.model_ready;
  setSearchReady(ready, ready ? null : 'warming model...');
  return h;
}

fetch('/api/status')
  .then(r => r.json())
  .then(d => {
    refreshHealth().then(h => {
      document.getElementById('status-bar').textContent =
        `${d.videos.toLocaleString()} vid · ${(d.keyframes / 1000).toFixed(0)}k kf` +
        ` · ASR ${d.asr_videos} · OCR ${d.ocr_videos}` +
        ` · model ${h.model_ready ? 'ready' : 'warming'}` +
        (h.warmup_seconds ? ` ${h.warmup_seconds}s` : '') +
        ` · ffmpeg ${h.ffmpeg ? 'ok' : 'missing'}` +
        ` · submit ${d.submission_rows}`;
    }).catch(() => {
      document.getElementById('status-bar').textContent =
        `${d.videos.toLocaleString()} vid · ${(d.keyframes / 1000).toFixed(0)}k kf` +
        ` · ASR ${d.asr_videos} · OCR ${d.ocr_videos}`;
    });
  })
  .catch(() => {
    document.getElementById('status-bar').textContent = 'index offline';
  });

// Init
setTask('kis');
reloadSubmissionManager();
refreshHealth().catch(() => {});
</script>
</body>
</html>"""  # noqa: E501


# ---------------------------------------------------------------------------
# Helper: attach kf_n from kmap
# ---------------------------------------------------------------------------
def _attach_kf(results, kmap):
    """Convert (video_id, frame_idx, pts_time, score) tuples to response dicts."""
    import pandas as pd
    out = []
    for vid, fidx, t, sc in results:
        m = kmap[(kmap.video_id == vid) & (kmap.frame_idx == fidx)]
        if len(m) == 0:
            raise ValueError(
                "retrieval result is not present in the canonical keyframe map: "
                f"video_id={vid!r}, frame_idx={fidx!r}"
            )
        kf_n = int(m.iloc[0].kf_n)
        out.append({
            "video_id": vid,
            "frame_idx": int(fidx),
            "kf_n": kf_n,
            "pts_time": float(t),
            "score": float(sc),
        })
    return out


# ---------------------------------------------------------------------------
# Submission session helpers
# ---------------------------------------------------------------------------
SUBMISSION_FIELDS = [
    "submitted_at",
    "query_id",
    "task",
    "video_name",
    "frame_idx",
    "kf_n",
    "pts_time",
    "score",
    "remaining_time",
]


def _session_count() -> int:
    if not os.path.exists(SUBMISSION_SESSION_CSV):
        return 0
    try:
        with open(SUBMISSION_SESSION_CSV, newline="", encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def _read_session_rows() -> list[dict]:
    if not os.path.exists(SUBMISSION_SESSION_CSV):
        return []
    with open(SUBMISSION_SESSION_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_session_rows(rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(SUBMISSION_SESSION_CSV), exist_ok=True)
    with open(SUBMISSION_SESSION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SUBMISSION_FIELDS})


def _append_submission(payload: dict) -> dict:
    query_id = str(payload.get("query_id") or f"q{_session_count() + 1:03d}").strip()
    video_name = str(payload.get("video_name") or payload.get("video_id") or "").strip()
    if not video_name:
        raise ValueError("video_name is required")
    try:
        frame_idx = int(payload.get("frame_idx"))
    except Exception as e:
        raise ValueError("frame_idx is required and must be int") from e

    row = {
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query_id": query_id,
        "task": str(payload.get("task") or "KIS").upper(),
        "video_name": video_name,
        "frame_idx": frame_idx,
        "kf_n": payload.get("kf_n", ""),
        "pts_time": payload.get("pts_time", ""),
        "score": payload.get("score", ""),
        "remaining_time": payload.get("remaining_time", ""),
    }

    rows = _read_session_rows()
    replaced = False
    for i, existing in enumerate(rows):
        if str(existing.get("query_id", "")).strip() == query_id:
            rows[i] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)
    _write_session_rows(rows)
    return row


def _update_submission(payload: dict) -> dict:
    original_query_id = str(payload.get("original_query_id") or payload.get("query_id") or "").strip()
    if not original_query_id:
        raise ValueError("query_id is required")
    rows = _read_session_rows()
    idx = next((i for i, r in enumerate(rows) if str(r.get("query_id", "")).strip() == original_query_id), None)
    if idx is None:
        raise ValueError(f"query_id '{original_query_id}' not found")

    query_id = str(payload.get("query_id") or original_query_id).strip()
    video_name = str(payload.get("video_name") or payload.get("video_id") or "").strip()
    if not video_name:
        raise ValueError("video_name is required")
    try:
        frame_idx = int(payload.get("frame_idx"))
    except Exception as e:
        raise ValueError("frame_idx is required and must be int") from e

    updated = {
        "submitted_at": rows[idx].get("submitted_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query_id": query_id,
        "task": str(payload.get("task") or rows[idx].get("task") or "KIS").upper(),
        "video_name": video_name,
        "frame_idx": frame_idx,
        "kf_n": payload.get("kf_n", rows[idx].get("kf_n", "")),
        "pts_time": payload.get("pts_time", rows[idx].get("pts_time", "")),
        "score": payload.get("score", rows[idx].get("score", "")),
        "remaining_time": payload.get("remaining_time", rows[idx].get("remaining_time", "")),
    }
    rows[idx] = updated
    _write_session_rows(rows)
    return updated


def _delete_submission(query_id: str) -> int:
    query_id = str(query_id or "").strip()
    if not query_id:
        raise ValueError("query_id is required")
    rows = _read_session_rows()
    new_rows = [r for r in rows if str(r.get("query_id", "")).strip() != query_id]
    deleted = len(rows) - len(new_rows)
    if deleted:
        _write_session_rows(new_rows)
    return deleted


def _export_codabench_csv() -> str:
    """Export current session to flexible 2025-style Codabench CSV."""
    os.makedirs(os.path.dirname(SUBMISSION_CODABENCH_CSV), exist_ok=True)
    with open(SUBMISSION_CODABENCH_CSV, "w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=["query_id", "video_name", "frame_idx"])
        writer.writeheader()
        if os.path.exists(SUBMISSION_SESSION_CSV):
            with open(SUBMISSION_SESSION_CSV, newline="", encoding="utf-8") as src:
                for row in csv.DictReader(src):
                    writer.writerow({
                        "query_id": row.get("query_id", ""),
                        "video_name": row.get("video_name", ""),
                        "frame_idx": row.get("frame_idx", ""),
                    })
    return SUBMISSION_CODABENCH_CSV


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return _HTML


@app.get("/annotation", response_class=HTMLResponse)
async def annotation_page():
    """Human-review workspace for the prepared TRAKE pilot pack."""
    return HTMLResponse(_ANNOTATION_HTML)


@app.get("/annotation/quick", response_class=HTMLResponse)
async def annotation_quick_page():
    return HTMLResponse("""<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>TRAKE Quick Review</title><style>body{font:14px system-ui;margin:0;background:#eef2f1;color:#18252a}.wrap{max-width:1200px;margin:auto;padding:22px}.note{background:#fff3d9;padding:12px;border-left:4px solid #d98a28;margin-bottom:16px}.row{background:white;border:1px solid #d7e0df;border-radius:10px;padding:14px;margin:10px 0;display:grid;grid-template-columns:170px 1fr 150px;gap:12px;align-items:start}.row img{width:150px;max-height:100px;object-fit:cover}.meta{font-size:12px;color:#607274}.caption{font-size:16px;margin:5px 0}.actions button{border:0;border-radius:5px;padding:7px 10px;margin:3px;cursor:pointer}.accept{background:#cbe9bc}.reject{background:#f4c2ba}.edit{background:#c9def3}textarea{width:100%;min-height:55px;box-sizing:border-box}.status{color:#176e63;font-size:12px}</style><div class=wrap><h1>TRAKE Quick Review</h1><div class=note><b>Review only.</b> These are agent-generated drafts from the synthetic set. Accepting means “candidate looks plausible”, not benchmark approval. Human review and independent timestamp verification are still required.</div><div id=rows>Loading...</div></div><script>async function load(){let d=await fetch('/annotation/api/quick-review').then(r=>r.json());rows.innerHTML=d.rows.map((x,i)=>`<article class=row id=r${i}><img src='/frame/${x.video_id}/${x.kf_n}'><div><div class=meta>${x.video_id} · step ${x.step} · ${Number(x.pts_time).toFixed(2)}s · frame ${x.frame_idx}</div><p class=caption>${x.caption}</p><textarea>${x.caption}</textarea><div class=status id=s${i}></div></div><div class=actions><button class=accept onclick='review(${i},"accept")'>Accept draft</button><button class=edit onclick='review(${i},"edit")'>Save edit</button><button class=reject onclick='review(${i},"reject")'>Reject</button></div></article>`).join('')}function review(i,action){let x=document.querySelector('#r'+i),s=document.querySelector('#s'+i);s.textContent=action==='reject'?'Marked for rejection.':action==='edit'?'Edited draft locally; export remains blocked until independent review.':'Accepted as plausible draft; export remains blocked.';x.style.opacity=action==='reject'?'.45':'1'}load()</script>""")


@app.get("/annotation/api/manifest")
async def annotation_manifest():
    manifest = ANNOTATION_PACK / "manifest.json"
    if not manifest.is_file():
        raise HTTPException(status_code=404, detail="annotation pack not prepared")
    import json as _json
    return {"videos": _json.loads(manifest.read_text(encoding="utf-8"))}


@app.get("/annotation/api/rows")
async def annotation_rows():
    if not ANNOTATION_CSV.is_file():
        return {"rows": []}
    with ANNOTATION_CSV.open(newline="", encoding="utf-8") as f:
        return {"rows": list(csv.DictReader(f))}


@app.get("/annotation/api/quick-review")
async def annotation_quick_review():
    path = Path(ROOT.parent.parent) / "data/annotations/quick_review.csv"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="quick review drafts not prepared")
    with path.open(newline="", encoding="utf-8") as f:
        return {"rows": list(csv.DictReader(f)), "provenance": "agent_draft"}


@app.get("/annotation/contact-sheet/{filename}")
async def annotation_contact_sheet(filename: str):
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".jpg":
        raise HTTPException(status_code=404, detail="not found")
    path = ANNOTATION_PACK / "contact_sheets" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="contact sheet not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/annotation/api/candidates/{video_id}")
async def annotation_candidates(video_id: str, limit: int = 48):
    """Return sparse frame candidates for fast human review.

    These are navigation aids, not event annotations. The endpoint deliberately
    returns no generated caption or ground-truth claim.
    """
    import pandas as pd
    limit = max(6, min(96, int(limit)))
    metadata = pd.read_parquet(os.path.join(IDX_DIR, "global_keyframes_vitl.parquet"))
    rows = metadata[metadata.video_id.astype(str) == video_id].sort_values("pts_time").reset_index(drop=True)
    if rows.empty:
        raise HTTPException(status_code=404, detail="video not found")
    positions = sorted(set(round(i * (len(rows) - 1) / max(limit - 1, 1)) for i in range(min(limit, len(rows)))) )
    return {"video_id": video_id, "provenance": "navigation_only", "candidates": [
        {"kf_n": int(rows.iloc[i].kf_n), "frame_idx": int(rows.iloc[i].frame_idx),
         "pts_time": float(rows.iloc[i].pts_time),
         "image_url": f"/frame/{video_id}/{int(rows.iloc[i].kf_n)}"} for i in positions
    ]}


@app.post("/annotation/api/rows")
async def annotation_save(payload: dict):
    required = set(ANNOTATION_FIELDS)
    if set(payload) - required or required - set(payload):
        raise HTTPException(status_code=400, detail="annotation fields do not match schema")
    if not payload["query_id"].strip() or not payload["video_id"].strip():
        raise HTTPException(status_code=400, detail="query_id and video_id are required")
    if payload["provenance"] != "draft_human_review":
        raise HTTPException(status_code=400, detail="draft rows must use provenance=draft_human_review")
    rows = []
    if ANNOTATION_CSV.is_file():
        with ANNOTATION_CSV.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    key = (payload["query_id"], payload["step"])
    rows = [row for row in rows if (row.get("query_id"), row.get("step")) != key]
    rows.append({field: str(payload.get(field, "")) for field in ANNOTATION_FIELDS})
    rows.sort(key=lambda row: (row["query_id"], int(row["step"] or 0)))
    ANNOTATION_PACK.mkdir(parents=True, exist_ok=True)
    with ANNOTATION_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {"ok": True, "saved": rows[-1] if rows else payload, "count": len(rows)}


@app.get("/api/status")
async def status():
    import pandas as pd
    try:
        gkf = pd.read_parquet(os.path.join(IDX_DIR, "global_keyframes_vitl.parquet"))
        n_vid = int(gkf.video_id.nunique())
        n_kf = len(gkf)
    except Exception:
        n_vid = n_kf = 0

    asr_vids = set()
    for f in glob.glob(os.path.join(IDX_DIR, "asr_chunks_*_ts.parquet")):
        try:
            asr_vids.update(pd.read_parquet(f, columns=["vid"]).vid.unique())
        except Exception:
            pass

    ocr_vids = set()
    skip = {"_partial", "_compare", "_gt", "ocr_query"}
    for f in glob.glob(os.path.join(IDX_DIR, "ocr_*.parquet")):
        if any(x in f for x in skip):
            continue
        try:
            ocr_vids.update(pd.read_parquet(f, columns=["video_id"]).video_id.unique())
        except Exception:
            pass

    return {
        "videos": n_vid,
        "keyframes": n_kf,
        "asr_videos": len(asr_vids),
        "ocr_videos": len(ocr_vids),
        "submission_rows": _session_count(),
    }


@app.get("/api/health")
async def health():
    """Lightweight readiness probe for competition setup."""
    ffmpeg_ok = os.path.exists(os.path.join(ROOT.parent.parent, ".venv", "bin", "ffmpeg")) or os.system("command -v ffmpeg >/dev/null 2>&1") == 0
    kis_idx_ok = os.path.exists(os.path.join(IDX_DIR, "global_siglip_vitl.npy")) and os.path.exists(os.path.join(IDX_DIR, "global_keyframes_vitl.parquet"))
    vqa_ok = os.path.exists(os.path.join(IDX_DIR, "vqa_eval_set.parquet"))
    trake_ok = os.path.exists(os.path.join(IDX_DIR, "trake_queryset.parquet"))
    return {
        "ok": bool(kis_idx_ok and vqa_ok and trake_ok and _warm_state.get("model_ready")),
        "ffmpeg": bool(ffmpeg_ok),
        "kis_index": bool(kis_idx_ok),
        "vqa_index": bool(vqa_ok),
        "trake_index": bool(trake_ok),
        "warmup": dict(_warm_state),
        "model_ready": bool(_warm_state.get("model_ready")),
        "warmup_seconds": _warm_state.get("warmup_seconds"),
        "submission_rows": _session_count(),
    }


@app.post("/api/submit")
async def api_submit(payload: dict):
    """Append one interactive submission to the current session CSV."""
    try:
        row = _append_submission(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"submit failed: {e}")
    return {"ok": True, "row": row, "count": _session_count()}


@app.get("/api/submit/list")
async def api_submit_list():
    return {"ok": True, "rows": _read_session_rows(), "count": _session_count()}


@app.post("/api/submit/update")
async def api_submit_update(payload: dict):
    try:
        row = _update_submission(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"update failed: {e}")
    return {"ok": True, "row": row, "count": _session_count()}


@app.post("/api/submit/delete")
async def api_submit_delete(payload: dict):
    try:
        deleted = _delete_submission(payload.get("query_id", ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"delete failed: {e}")
    return {"ok": True, "deleted": deleted, "count": _session_count()}


_ANNOTATION_HTML = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRAKE Independent Annotation</title>
<style>body{margin:0;background:#eef2f1;color:#17252a;font:14px system-ui,sans-serif}.app{display:grid;grid-template-columns:280px 1fr;min-height:100vh}.side{background:#14252c;color:#e8f0ef;padding:20px}.side h1{font-size:19px;margin:0 0 8px}.side p{color:#aec0c0;font-size:12px;line-height:1.5}.videos button{display:block;width:100%;text-align:left;color:inherit;background:transparent;border:0;border-left:3px solid transparent;padding:11px 8px;cursor:pointer}.videos button.active{background:#ffffff14;border-left-color:#c8ed8c}.main{padding:24px;max-width:1400px;width:100%;box-sizing:border-box}.warn{background:#fff3d9;border-left:4px solid #da8a27;padding:12px;margin-bottom:16px;line-height:1.45}.grid{display:grid;grid-template-columns:minmax(500px,1.2fr) minmax(360px,.8fr);gap:18px}.card{background:#fff;border:1px solid #d6e0df;border-radius:12px;padding:16px}.sheet{width:100%;max-height:78vh;object-fit:contain;object-position:top;background:#152127}.candidate-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;max-height:70vh;overflow:auto}.candidate{border:2px solid transparent;background:#f5f7f7;padding:3px;cursor:pointer;text-align:left}.candidate:hover{border-color:#0f766e}.candidate img{width:100%;display:block}.candidate small{display:block;font-size:10px;padding:3px}.fields{display:grid;grid-template-columns:1fr 1fr;gap:10px}.wide{grid-column:1/-1}label{font-weight:700;font-size:12px}input,textarea,select{display:block;width:100%;box-sizing:border-box;margin-top:4px;padding:8px;border:1px solid #cbd6d5;border-radius:6px;font:inherit}textarea{min-height:70px}.actions{margin-top:14px;display:flex;gap:8px;align-items:center}button.save{background:#116f67;color:#fff;border:0;border-radius:6px;padding:9px 13px;font-weight:700;cursor:pointer}.hint{font-size:12px;color:#607274}@media(max-width:900px){.app{display:block}.side{position:static}.grid{grid-template-columns:1fr}.sheet{max-height:none}}</style>
<div class="app"><aside class="side"><h1>TRAKE Review</h1><p>Draft candidates are navigation aids only. You write the event; the system only helps you jump to sparse frames.</p><div id="videos" class="videos"></div></aside><main class="main"><div class="warn"><b>Review, do not blindly accept.</b> Write 3-5 events, then have a different reviewer verify timestamps. Draft rows use <code>draft_human_review</code>, not <code>human_authored</code>.</div><div class="grid"><section class="card"><h2 id="title">Choose a video</h2><div id="candidates" class="candidate-grid"></div></section><section class="card"><h2>Annotation row</h2><p class="hint">Click a frame to fill metadata, then write/edit the event caption yourself.</p><form id="form"><div class="fields"><label>Query ID<input id="query_id" required></label><label>Video ID<input id="video_id" required readonly></label><label>Step<input id="step" type="number" min="0" required></label><label class="wide">Caption<textarea id="caption" required></textarea></label><label>KF number<input id="kf_n" type="number"></label><label>Frame index<input id="frame_idx" type="number"></label><label>Timestamp (s)<input id="pts_time" type="number" step=".001"></label><label>Split<select id="split"><option>dev</option><option>holdout</option></select></label><label>Annotator ID<input id="annotator_id"></label><label>Reviewer ID<input id="reviewer_id"></label><label>Confidence<select id="confidence"><option>high</option><option>medium</option><option>low</option></select></label><label>Provenance<select id="provenance"><option>draft_human_review</option></select></label><label>Authoring method<input id="authoring_method" value="human_timeline_review"></label><label>Target method<input id="target_selection_method" value="human_timestamp_review"></label></div><div class="actions"><button class="save">Save draft row</button><span id="status" class="hint"></span></div></form></section></div></main></div>
<script>let current='';const ids=['query_id','video_id','step','caption','kf_n','frame_idx','pts_time','provenance','split','annotator_id','reviewer_id','confidence','authoring_method','target_selection_method'];async function load(){let m=await fetch('/annotation/api/manifest').then(r=>r.json());videos.innerHTML=m.videos.map(v=>`<button onclick="pick('${v.video_id}')" id="v-${v.video_id}">${v.video_id}<small> · ${v.keyframes} keyframes</small></button>`).join('');if(m.videos[0])pick(m.videos[0].video_id)}async function pick(v){current=v;document.querySelectorAll('.videos button').forEach(b=>b.classList.toggle('active',b.id==='v-'+v));title.textContent=v+' · sparse review candidates';video_id.value=v;let d=await fetch('/annotation/api/candidates/'+v+'?limit=48').then(r=>r.json());candidates.innerHTML=d.candidates.map(x=>`<button type="button" class="candidate" onclick="useFrame(${x.kf_n},${x.frame_idx},${x.pts_time})"><img src="${x.image_url}" loading="lazy"><small>kf ${x.kf_n} · ${x.pts_time.toFixed(2)}s</small></button>`).join('')}function useFrame(k,f,t){kf_n.value=k;frame_idx.value=f;pts_time.value=t}form.onsubmit=async e=>{e.preventDefault();let body=Object.fromEntries(ids.map(id=>[id,document.getElementById(id).value]));let r=await fetch('/annotation/api/rows',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});status.textContent=r.ok?'Draft saved.':await r.text()};load()</script>"""

@app.get("/api/submit/export")
async def api_submit_export(format: str = "codabench"):
    """
    Export current interactive submissions.

    format=codabench returns 2025-style CSV: query_id,video_name,frame_idx.
    format=session returns full audit CSV with task/kf/time/score metadata.
    """
    if format == "session":
        if not os.path.exists(SUBMISSION_SESSION_CSV):
            _export_codabench_csv()  # ensures results dir exists; session may still be absent
            with open(SUBMISSION_SESSION_CSV, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=SUBMISSION_FIELDS).writeheader()
        return FileResponse(SUBMISSION_SESSION_CSV, media_type="text/csv", filename="submission_session.csv")
    out = _export_codabench_csv()
    return FileResponse(out, media_type="text/csv", filename="submission.csv")


@app.get("/api/kis/neighbors/{video_id}/{kf_n}")
async def api_kis_neighbors(video_id: str, kf_n: int, window: int = 10):
    """
    Return temporal neighbors of a keyframe for timeline navigation.

    Args:
        video_id: e.g. "K01_V001"
        kf_n: keyframe number (1-indexed)
        window: number of frames on each side (max 30)
    """
    window = min(max(1, window), 30)
    try:
        nav = get_nav()
        return nav.get_neighbors(video_id, kf_n, window=window)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/api/kis/neighbors_by_time/{video_id}")
async def api_kis_neighbors_by_time(
    video_id: str,
    pts_time: float,
    half_window_s: float = 10.0,
):
    """
    H2: Return keyframes within ±half_window_s seconds of pts_time.

    Useful for precise ±2s / ±5s navigation matching competition scoring.
    Each neighbor includes delta_s for visual display.
    """
    try:
        nav = get_nav()
        import pandas as pd
        if video_id not in nav._video_ranges:
            raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")
        s, e = nav._video_ranges[video_id]
        vid_frames = nav.kmap.iloc[s:e]
        lo, hi = pts_time - half_window_s, pts_time + half_window_s
        window_frames = vid_frames[(vid_frames.pts_time >= lo) & (vid_frames.pts_time <= hi)]
        neighbors = []
        for _, row in window_frames.iterrows():
            delta = round(float(row.pts_time) - pts_time, 2)
            neighbors.append({
                "kf_n":      int(row.kf_n),
                "frame_idx": int(row.frame_idx),
                "pts_time":  float(row.pts_time),
                "delta_s":   delta,
                "in_2s":     abs(delta) <= 2.0,
                "is_center": abs(delta) < 1.5,
            })
        return {
            "video_id":    video_id,
            "center_pts":  pts_time,
            "half_window": half_window_s,
            "neighbors":   neighbors,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/kis")
async def api_kis(payload: dict):
    p = get_pipe()
    query = payload.get("query", "").strip()
    try:
        topk = int(payload.get("topk", 12))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="topk must be an integer") from exc
    if not 1 <= topk <= 100:
        raise HTTPException(status_code=422, detail="topk must be between 1 and 100")
    include_peaks = bool(payload.get("include_peaks", False))
    peaks_per_video = max(1, min(10, int(payload.get("peaks_per_video", 5))))
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    if include_peaks:
        retriever = p._ensure_kis()
        peak_out = retriever.search_peaks(query, topk=topk, peaks_per_video=peaks_per_video)
        out = {"winner": "visual_fusion_vitl_so400m384", "results": peak_out["results"]}
        peaks_by_video = {item["video_id"]: item["peaks"] for item in peak_out["peaks"]}
    else:
        out = p.kis(query, topk=topk)
        peaks_by_video = {}
    kmap = p.kis_kmap()
    response = {"winner": out["winner"], "results": _attach_kf(out["results"], kmap)}
    if include_peaks:
        response["peaks"] = [{"video_id": vid, "peaks": values} for vid, values in peaks_by_video.items()]
    return response


@app.post("/api/kis/expand")
async def api_kis_expand(payload: dict):
    """
    KIS with LLM query expansion (RAPID-style).

    Generates N visual variants via LLM, searches each in parallel,
    fuses results with Reciprocal Rank Fusion (k=60).

    Body: {query, topk?, n_variants?}
    Returns: {variants, results, variant_counts, winner}
    """
    _require_research_route("/api/kis/expand")
    from query_expansion import expand_and_search
    query = payload.get("query", "").strip()
    topk = int(payload.get("topk", 12))
    n_variants = int(payload.get("n_variants", 3))
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    p = get_pipe()
    env = get_expander_env()
    kmap = p.kis_kmap()

    def search_fn(q: str, k: int):
        out = p.kis(q, topk=k)
        return out["results"]

    out = expand_and_search(
        query, search_fn,
        n_variants=n_variants, topk=topk,
        env=env, model="llama-4-maverick",
    )
    return {
        "variants": out["variants"],
        "results": _attach_kf(out["results"], kmap),
        "variant_counts": out["variant_counts"],
        "winner": "expand+rrf",
    }


@app.post("/api/kis/hybrid")
async def api_kis_hybrid(payload: dict):
    """
    Hybrid KIS: visual + object + OCR late fusion (query-adaptive weights).

    Activates object channel when query contains object keywords,
    OCR channel when query contains named entities / numbers.

    Body: {query, topk?}
    Returns: {signals_used, weights, results, winner}
    """
    _require_research_route("/api/kis/hybrid")
    from hybrid_search import HybridSearcher, detect_signals
    query = payload.get("query", "").strip()
    topk = int(payload.get("topk", 12))
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    p = get_pipe()
    kmap = p.kis_kmap()
    obj = get_obj_searcher()

    def visual_fn(q: str, k: int):
        return p.kis(q, topk=k)["results"]

    def object_fn(q: str, k: int):
        return obj.search(q, topk=k) if obj else []

    searcher = HybridSearcher(
        visual_fn=visual_fn,
        object_fn=object_fn if obj else None,
    )
    out = searcher.search(query, topk=topk)
    signals = detect_signals(query)

    return {
        "signals_used": out["signals_used"],
        "weights": out["weights"],
        "results": _attach_kf(out["results"], kmap),
        "winner": "+".join(out["signals_used"]),
        "object_available": obj is not None,
        "signals_detected": signals,
    }


@app.post("/api/vkis")
async def api_vkis(file: UploadFile = File(...), topk: int = 12):
    p = get_pipe()
    if not 1 <= int(topk) <= 100:
        raise HTTPException(status_code=422, detail="topk must be between 1 and 100")
    suffix = Path(file.filename).suffix or ".bin"
    tmp = tempfile.mktemp(suffix=suffix)
    try:
        with open(tmp, "wb") as f:
            f.write(await file.read())
        out = p.vkis(tmp, topk=int(topk))
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return {"results": _attach_kf(out["results"], p._ensure_vkis().vmap)}


@app.post("/api/vqa")
async def api_vqa(payload: dict):
    p = get_pipe()
    query = payload.get("query", "").strip()
    question = payload.get("question", "").strip()
    if not query or not question:
        raise HTTPException(status_code=400, detail="query and question are required")
    mode = str(payload.get("mode", "interactive")).strip().lower()
    if mode not in {"interactive", "ranked"}:
        raise HTTPException(status_code=422, detail="mode must be 'interactive' or 'ranked'")

    # Both presentation modes use the same public ranked owner.  The
    # interactive mode only changes the response projection for the existing
    # browser UI; it must not select the legacy VQA pipeline.
    try:
        out = p.vqa_ranked(
            query,
            question,
            max_answers=min(int(payload.get("max_answers", 20)), 20),
            question_type=payload.get("question_type"),
            required_modalities=payload.get("required_modalities"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if mode == "ranked":
        return {
            "mode": "ranked",
            "winner": out.get("winner"),
            "answers": out.get("answers", []),
            "status": out.get("status"),
        }

    answer = (out.get("answers") or [None])[0]
    best = None
    if isinstance(answer, dict):
        best = {
            "video": answer.get("video_id", answer.get("video")),
            "frame_idx": answer.get("frame_id", answer.get("frame_idx")),
            "kf_n": answer.get("kf_n"),
            "pts_time": answer.get("pts_time"),
            "answer": answer.get("answer", ""),
            "vlm_score": answer.get(
                "grounding_score", answer.get("ranking_score", 0.0)
            ),
        }
    return {"mode": "interactive", "best": best}


@app.post("/api/trake")
async def api_trake(payload: dict):
    p = get_pipe()
    events = payload.get("events", [])
    if (not isinstance(events, list) or not events or
            any(not isinstance(event, str) or not event.strip() for event in events)):
        raise HTTPException(status_code=400, detail="events must be a non-empty list of strings")
    try:
        topk = int(payload.get("topk", 1))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="topk must be an integer") from exc
    if not 1 <= topk <= 100:
        raise HTTPException(status_code=422, detail="topk must be between 1 and 100")
    # Omit the mode when the caller does not explicitly choose one.  The
    # shared pipeline then applies its production default (visual), instead
    # of this UI endpoint silently forcing the legacy ASR path.
    mode = payload.get("mode")
    try:
        out = p.trake(
            events,
            video_id=payload.get("video_id"),
            topk=topk,
            mode=mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "mode": out.get("mode"),
        "winner": out.get("winner"),
        "results": out["results"],
    }


@app.get("/frame/{video_id}/{kf_n}")
async def serve_frame(video_id: str, kf_n: int):
    fp = os.path.join(KF_DIR, video_id, f"{kf_n:03d}.jpg")
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail=f"Frame not found: {video_id}/{kf_n:03d}.jpg")
    return FileResponse(fp, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010, log_level="info")
