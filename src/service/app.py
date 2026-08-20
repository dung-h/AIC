"""FastAPI application for the resident HCMAI retrieval runtime."""
from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Annotated

from src.service.runtime import get_runtime
from src.service.contracts import RetrievalResult

# Annotation authoring belongs to the private benchmark workspace.  The
# retrieval service is still a supported public/runtime component, so an
# absent private queryset must not prevent its health/search endpoints from
# starting on a clean deployment clone.
try:
    from src.service import annotation_review
except ImportError as error:
    missing_private_workspace = (
        error.name == "src.service.annotation_review"
        or (error.name and error.name.startswith("src.queryset"))
        or "annotation_review" in str(error)
    )
    if missing_private_workspace:
        annotation_review = None
    else:
        raise


class KISRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    topk: int = Field(default=20, ge=1, le=100)
    mode: str = Field(default="default", pattern="^default$")
    include_peaks: bool = False
    peaks_per_video: int = Field(default=5, ge=1, le=10)


class SearchResult(BaseModel):
    rank: int
    video_id: str
    frame_idx: int
    kf_n: int
    pts_time: float
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    latency_ms: float
    cache_hit: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)
    peaks: list[dict] = Field(default_factory=list)


class TrakeRequest(BaseModel):
    events: list[Annotated[str, Field(min_length=1, max_length=4000)]] = Field(min_length=1, max_length=20)
    top_k_videos: int = Field(default=5, ge=1, le=100)
    include_per_event_scores: bool = False


class TrakeResponse(BaseModel):
    results: list[dict]
    latency_ms: float
    timings_ms: dict[str, float] = Field(default_factory=dict)


class VQARequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    question: str = Field(min_length=1, max_length=4000)
    max_answers: int = Field(default=20, ge=1, le=20)
    top_videos: int = Field(default=20, ge=1, le=100)
    frames_per_video: int = Field(default=5, ge=1, le=20)
    max_vlm_candidates: int = Field(default=12, ge=1, le=100)
    question_type: str = ""
    required_modalities: str = ""


class VQAResponse(BaseModel):
    query: str
    question: str
    answers: list[dict]
    status: str
    candidate_count: int = 0
    vlm_candidate_count: int = 0
    latency_ms: float


class AnnotationUpdate(BaseModel):
    triage: str = "untriaged"
    question_type: str = ""
    query: str = ""
    question: str = ""
    answer: str = ""
    required_modalities: str = ""
    acceptable_kf_n: str = ""
    answer_start_time: str = ""
    answer_end_time: str = ""
    review_notes: str = ""
    annotator_id: str = ""
    reviewer_id: str = ""
    status: str = "draft"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if get_runtime().policy.preload_trake:
        get_runtime().preload_trake()
    yield


app = FastAPI(title="HCMAI Retrieval Service", version="1.0", lifespan=lifespan)


def _annotation_workspace():
    if annotation_review is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "annotation workspace is private and is not installed in "
                "this runtime source bundle"
            ),
        )
    return annotation_review


@app.get("/health")
def health():
    return get_runtime().health()


@app.get("/ready")
def ready():
    return get_runtime().readiness()


@app.get("/stats")
def stats():
    return get_runtime().snapshot()


@app.get("/annotation", response_class=HTMLResponse)
def annotation_page():
    return HTMLResponse("""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VQA Review Studio</title>
<style>
:root{--ink:#18222d;--muted:#637180;--line:#dbe1e7;--paper:#fffdf8;--canvas:#f2f5f6;--nav:#18252e;--teal:#0f756e;--lime:#d7ed8a;--orange:#f5a354;--red:#c54e45;--blue:#3678bc;--shadow:0 12px 32px #20304212}*{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--ink);font:14px Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}.app{min-height:100vh;display:grid;grid-template-columns:300px minmax(0,1fr)}.side{background:var(--nav);color:#edf3f4;padding:25px 18px;position:sticky;top:0;height:100vh;display:flex;flex-direction:column;gap:22px}.brand{font-size:20px;font-weight:800;letter-spacing:-.04em}.brand span{color:var(--lime)}.brand small{display:block;color:#9db0b7;font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;margin-top:5px}.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.stat{border:1px solid #ffffff20;border-radius:10px;padding:10px;background:#ffffff08}.stat b{display:block;font-size:20px}.stat span{color:#adc0c7;font-size:11px}.side-label{font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:#8ca4ae;font-weight:800}.rows{overflow:auto;flex:1;padding-right:3px}.row-item{width:100%;text-align:left;background:transparent;border:0;border-left:3px solid transparent;color:#dbe5e7;padding:11px 9px;cursor:pointer;border-radius:0 8px 8px 0;display:flex;gap:8px;align-items:center}.row-item:hover{background:#ffffff0d}.row-item.active{background:#ffffff16;border-left-color:var(--lime)}.row-id{font-weight:750;font-size:12px}.row-type{display:block;color:#a9b9be;font-size:11px;margin-top:2px}.dot{width:8px;height:8px;border-radius:50%;background:#81949b;flex:0 0 auto}.dot.reviewed{background:var(--orange)}.dot.valid{background:var(--lime)}.dot.rejected{background:var(--red)}.side-foot{font-size:11px;color:#9db0b7;line-height:1.5}.main{padding:28px 34px 48px;max-width:1600px;margin:0 auto;width:100%}.topbar{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:24px}.eyebrow{font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--teal);font-weight:800}.topbar h1{font-size:31px;letter-spacing:-.055em;margin:4px 0 5px}.topbar p{margin:0;color:var(--muted);max-width:620px;line-height:1.5}.export{border:0;border-radius:9px;background:var(--ink);color:white;padding:11px 15px;font-weight:750;cursor:pointer;white-space:nowrap}.export:hover{background:var(--teal)}#exportResult{display:block;color:var(--muted);font-size:11px;max-width:220px;text-align:right;margin-top:7px}.workspace{display:grid;grid-template-columns:minmax(460px,1.05fr) minmax(380px,.95fr);gap:20px}.card{background:var(--paper);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow)}.evidence-card{padding:18px}.meta-line{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:15px}.pill{border-radius:99px;padding:4px 8px;font-size:11px;font-weight:800;background:#e5f1ef;color:#07655f;text-transform:uppercase;letter-spacing:.05em}.pill.dark{background:#e9edf0;color:#475665}.evidence{position:relative;background:#10191e;border-radius:10px;overflow:hidden}.evidence img{display:block;width:100%;height:min(58vh,640px);min-height:360px;object-fit:contain;background:#10191e}.viewer-nav{position:absolute;inset:0;display:flex;align-items:center;justify-content:space-between;pointer-events:none}.viewer-nav button{pointer-events:auto;border:0;border-radius:50%;width:39px;height:39px;margin:12px;background:#ffffffd9;color:#18222d;font-size:25px;cursor:pointer}.viewer-nav button:hover{background:white}.viewer-caption{display:flex;justify-content:space-between;align-items:center;padding:10px 2px 0;color:var(--muted);font-size:12px}.target-note{font-weight:800;color:var(--teal)}.thumbs{display:flex;gap:8px;overflow-x:auto;padding:13px 1px 2px}.thumb{border:2px solid transparent;background:#e6eaec;border-radius:7px;padding:0;overflow:hidden;cursor:pointer;flex:0 0 104px}.thumb img{width:100%;height:62px;object-fit:cover;display:block}.thumb.active{border-color:var(--blue);box-shadow:0 0 0 2px #3678bc2b}.thumb.target{border-color:var(--teal)}.thumb span{display:block;font-size:10px;padding:4px;color:#3a4853}.caption{padding:9px 2px 0;color:var(--muted);font-size:12px;line-height:1.45}.proposal{margin-top:17px;border:1px solid #c9e1df;border-radius:11px;background:#eff9f7;padding:16px}.proposal-title{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:12px}.proposal-title b{font-size:15px}.proposal-grid{display:grid;grid-template-columns:1fr 1fr;gap:11px}.proposal-item{background:#ffffffa8;border-radius:8px;padding:9px}.proposal-item.full{grid-column:1/-1}.proposal-item label{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:#55817e}.proposal-item p{margin:4px 0 0;line-height:1.4}.proposal-actions{display:flex;gap:8px;margin-top:13px}.btn{border:0;border-radius:8px;padding:9px 12px;font-weight:750;cursor:pointer}.btn.primary{background:var(--teal);color:white}.btn.primary:hover{background:#095c57}.btn.ghost{background:#d9e9e7;color:#195c58}.btn.danger{background:#fdecea;color:#a23c36}.review-card{padding:20px}.review-card h2{margin:0;font-size:19px;letter-spacing:-.03em}.review-card .intro{margin:5px 0 17px;color:var(--muted);font-size:12px;line-height:1.45}.field-group{border:0;border-top:1px solid var(--line);margin:16px 0 0;padding:16px 0 0}.field-group legend{padding:0 6px 0 0;font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);font-weight:800}.fields{display:grid;grid-template-columns:1fr 1fr;gap:10px}.fields .wide{grid-column:1/-1}label{display:block;font-size:11px;font-weight:750;color:#40505d}textarea,select{width:100%;font:inherit;color:var(--ink);border:1px solid #cbd4da;background:white;border-radius:7px;padding:8px;margin-top:5px;outline:none;resize:vertical}textarea:focus,select:focus{border-color:var(--teal);box-shadow:0 0 0 3px #0f756e18}textarea{min-height:45px}.savebar{position:sticky;bottom:0;background:linear-gradient(90deg,#fffdf8e8,#fffdf8);padding:14px 0 0;margin-top:16px;border-top:1px solid var(--line);display:flex;align-items:center;gap:9px}.save{background:var(--ink);color:white}.save:hover{background:var(--teal)}#result{font-size:12px;color:var(--muted)}.notice{margin:0 0 16px;border-left:3px solid var(--orange);padding:8px 10px;background:#fff5e8;color:#885121;font-size:12px;line-height:1.4}@media(max-width:1050px){.app{grid-template-columns:235px minmax(0,1fr)}.workspace{grid-template-columns:1fr}.side{padding:20px 12px}.main{padding:24px}}@media(max-width:720px){.app{display:block}.side{position:relative;height:auto;min-height:0}.rows{max-height:180px}.main{padding:18px}.topbar{display:block}.export{margin-top:14px}.workspace{display:block}.review-card{margin-top:16px}.proposal-grid,.fields{grid-template-columns:1fr}.proposal-item.full,.fields .wide{grid-column:auto}.evidence img{height:55vw;min-height:260px}.thumb{flex-basis:88px}}
</style></head><body><div class="app"><aside class="side"><div class="brand">Frame<span>Proof</span><small>VQA annotation studio</small></div><div class="stat-grid"><div class="stat"><b id=total>0</b><span>review items</span></div><div class="stat"><b id=done>0</b><span>reviewed / valid</span></div></div><div class="side-label">Annotation queue</div><nav id=rows class=rows></nav><div class="side-foot">Use the AI proposal as a starting point. Contact sheet evidence always wins.</div></aside><main class="main"><header class="topbar"><div><div class="eyebrow">Evidence first annotation</div><h1>Review one moment at a time.</h1><p>Confirm the proposed question against temporal evidence, then keep, edit, or reject it. No draft becomes ground truth without an independent reviewer.</p></div><div><button class=export onclick="exportReviewed()">Materialize & validate</button><span id=exportResult></span></div></header><div class="workspace"><section class="card evidence-card"><div id=meta class=meta-line></div><div class=evidence><img id=evidence alt="Temporal contact sheet"></div><p class=caption>Five neighboring keyframes. The center label marks the selected target. Use the visible evidence, not the model's confidence.</p><div id=suggestion class=proposal></div></section><section class="card review-card"><h2>Annotation record</h2><p class=intro>Apply a proposal to populate a review draft. The retrieval query must describe the scene without revealing the answer.</p><p class=notice>For <b>spoken_fact</b> and <b>temporal_relation</b>, reject or keep as draft unless the contact sheet genuinely supports the claim.</p><section id=form></section></section></div></main></div>
<script>
let rows=[], selectedId='', evidenceIndex=0; const fields=['triage','question_type','query','question','answer','required_modalities','acceptable_kf_n','answer_start_time','answer_end_time','review_notes','annotator_id','reviewer_id'];
function current(){return rows.find(x=>x.annotation_id===selectedId)} function esc(v){let d=document.createElement('div');d.textContent=v||'';return d.innerHTML} function statusClass(s){return s==='draft'?'':s}
function renderNav(){total.textContent=rows.length;done.textContent=rows.filter(x=>x.status==='reviewed'||x.status==='valid').length;rowsEl=document.getElementById('rows');rowsEl.innerHTML=rows.map(x=>`<button class="row-item ${x.annotation_id===selectedId?'active':''}" onclick="selectRow('${x.annotation_id}')"><i class="dot ${statusClass(x.status)}"></i><span><span class=row-id>${x.annotation_id}</span><span class=row-type>${x.question_type} · ${x.status}</span></span></button>`).join('')}
function selectRow(id){selectedId=id;evidenceIndex=0;renderNav();render()}
function field(label,key,value,wide=false){return `<label class="${wide?'wide':''}">${label}<textarea id="${key}">${esc(value||'')}</textarea></label>`}
function triageFields(x){return `<fieldset class=field-group><legend>Evidence triage</legend><div class=fields><label>Triage<select id=triage>${['untriaged','keep','reject','needs_context'].map(s=>`<option ${x.triage===s?'selected':''}>${s}</option>`).join('')}</select></label><label>Question type (only after keep)<select id=question_type><option value="">Choose after evidence review</option>${['color','count','person','action','screen_text','spoken_fact','place','temporal_relation'].map(s=>`<option ${x.question_type===s?'selected':''}>${s}</option>`).join('')}</select></label></div></fieldset>`}
function imageUrl(x,name){return `/annotation/evidence/${x.annotation_id}/${encodeURIComponent(name)}`}
function showEvidence(index){let x=current(),images=x.evidence_images||[];if(!images.length)return;evidenceIndex=(index+images.length)%images.length;let name=images[evidenceIndex],kf=Number(name.replace('.jpg',''));evidence.src=imageUrl(x,name);viewerLabel.innerHTML=`Frame ${evidenceIndex+1} / ${images.length} · kf ${kf}${kf===Number(x.kf_n)?' <span class=target-note>Target frame</span>':''}`;document.querySelectorAll('.thumb').forEach((el,i)=>el.classList.toggle('active',i===evidenceIndex))}
function moveEvidence(delta){showEvidence(evidenceIndex+delta)}
function render(){let x=current(),s=x.suggestion||{},images=x.evidence_images||[];meta.innerHTML=`<span class=pill>${esc(x.question_type)}</span><span class="pill dark">${esc(x.video_id)}</span><span class="pill dark">Target kf ${x.kf_n} · ${Number(x.pts_time).toFixed(2)}s</span>`;let viewer=`<div class=evidence><img id=evidence alt="Video evidence frame"><div class=viewer-nav><button onclick="moveEvidence(-1)" aria-label="Previous frame">‹</button><button onclick="moveEvidence(1)" aria-label="Next frame">›</button></div></div><div class=viewer-caption id=viewerLabel></div><div class=thumbs>${images.map((name,i)=>{let kf=Number(name.replace('.jpg',''));return `<button class="thumb ${kf===Number(x.kf_n)?'target':''}" onclick="showEvidence(${i})"><img src="${imageUrl(x,name)}" alt="kf ${kf}"><span>kf ${kf}${kf===Number(x.kf_n)?' · target':''}</span></button>`}).join('')}</div>`;let old=document.querySelector('.evidence');if(old){old.parentElement.querySelector('.viewer-caption')?.remove();old.parentElement.querySelector('.thumbs')?.remove();old.outerHTML=viewer}else{document.querySelector('.evidence-card').insertAdjacentHTML('afterbegin',viewer)};suggestion.innerHTML=`<div class=proposal-title><b>AI proposal</b><span class=pill>needs evidence check</span></div><div class=proposal-grid><div class=proposal-item><label>Proposed question</label><p>${esc(s.question)||'No proposal returned.'}</p></div><div class=proposal-item><label>Proposed answer</label><p>${esc(s.answer)||'No proposal returned.'}</p></div><div class=proposal-item><label>Suggested modality</label><p>${esc(s.modalities)||'visual'}</p></div><div class=proposal-item><label>Expected type</label><p>${esc(x.question_type)}</p></div><div class="proposal-item full"><label>Evidence claimed by model</label><p>${esc(s.reason)||'No explanation returned.'}</p></div></div><div class=proposal-actions><button class="btn primary" onclick="applyDraft()">Apply proposal</button><button class="btn danger" onclick="rejectDraft()">Reject proposal</button></div>`;form.innerHTML=`<fieldset class=field-group><legend>Retrieval prompt</legend><div class=fields>${field('Answer-free scene description','query',x.query,true)}</div></fieldset><fieldset class=field-group><legend>Question and answer</legend><div class=fields>${field('Question','question',x.question,true)}${field('Answer','answer',x.answer,true)}</div></fieldset><fieldset class=field-group><legend>Evidence bounds</legend><div class=fields>${field('Modalities, comma-separated','required_modalities',x.required_modalities)}${field('Acceptable keyframe numbers','acceptable_kf_n',x.acceptable_kf_n)}${field('Answer start (seconds)','answer_start_time',x.answer_start_time)}${field('Answer end (seconds)','answer_end_time',x.answer_end_time)}</div></fieldset><fieldset class=field-group><legend>Audit</legend><div class=fields>${field('Annotation notes','review_notes',x.review_notes,true)}${field('Annotator ID','annotator_id',x.annotator_id)}${field('Reviewer ID','reviewer_id',x.reviewer_id)}<label class=wide>Status<select id=status>${['draft','reviewed','valid','rejected'].map(s=>`<option ${x.status===s?'selected':''}>${s}</option>`).join('')}</select></label></div></fieldset><div class=savebar><button class="btn save" onclick="save()">Save record</button><span id=result></span></div>`;showEvidence(evidenceIndex)}
function applyDraft(){triage.value='keep';question_type.value='';status.value='draft';query.value='';question.value='';answer.value='';required_modalities.value='visual';acceptable_kf_n.value='';answer_start_time.value='';answer_end_time.value='';review_notes.value='Kept after timestamp evidence review; choose type only after identifying one stable fact.';result.textContent='Candidate kept. Choose the evidence-backed type, then write a question and answer.'}
function rejectDraft(){triage.value='reject';status.value='draft';review_notes.value='Rejected: no single stable, answerable fact across timestamp evidence.';result.textContent='Rejection prepared. Save record to confirm.'}
async function save(){let body=Object.fromEntries([...fields,'status'].map(k=>[k,document.getElementById(k).value]));let r=await fetch('/annotation/rows/'+selectedId,{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify(body)});result.textContent=r.ok?'Saved locally.':await r.text();if(r.ok){Object.assign(current(),body);renderNav()}}
async function exportReviewed(){let r=await fetch('/annotation/export',{method:'POST'});exportResult.textContent=r.ok?`Valid: ${(await r.json()).valid_rows}`:await r.text()}
fetch('/annotation/rows').then(r=>r.json()).then(data=>{rows=data;selectedId=rows[0]?.annotation_id||'';renderNav();render()})
document.addEventListener('keydown',event=>{if(event.target.matches('textarea,select'))return;if(event.key==='ArrowLeft')moveEvidence(-1);if(event.key==='ArrowRight')moveEvidence(1)})
</script></body></html>""")


@app.get("/annotation/triage", response_class=HTMLResponse)
def annotation_triage_page():
    """Evidence-first candidate triage; separate from the retired legacy UI."""
    return HTMLResponse("""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><title>VQA Evidence Triage</title>
<style>body{font:15px system-ui;margin:0;background:#f3f5f4;color:#17232b}.app{display:grid;grid-template-columns:260px 1fr;min-height:100vh}.side{background:#16252d;color:#dce7e8;padding:20px}.side h1{font-size:20px;margin:0 0 5px}.side p{font-size:12px;color:#adc1c4;line-height:1.45}.items{max-height:82vh;overflow:auto;margin-top:18px}.item{display:block;width:100%;padding:10px;border:0;border-left:3px solid transparent;text-align:left;background:transparent;color:inherit;cursor:pointer}.item:hover,.item.active{background:#ffffff14;border-left-color:#d8ed8d}.item small{display:block;color:#9bb0b3}.main{max-width:1320px;width:100%;margin:auto;padding:28px}.warn{background:#fff3df;border-left:4px solid #e89942;padding:12px 14px;margin-bottom:18px;line-height:1.45}.grid{display:grid;grid-template-columns:minmax(480px,1.1fr) minmax(350px,.9fr);gap:20px}.card{background:#fffdf9;border:1px solid #dce3e3;border-radius:12px;padding:18px}.meta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}.pill{font-size:11px;font-weight:bold;border-radius:20px;background:#e0efec;color:#11675f;padding:4px 8px}.viewer{background:#10191e;border-radius:10px;min-height:420px;display:grid;place-items:center}.viewer img{max-width:100%;max-height:590px;display:block}.thumbs{display:flex;gap:8px;overflow:auto;margin-top:12px}.thumb{padding:0;border:2px solid transparent;background:#e7ecec;cursor:pointer;min-width:108px}.thumb.active{border-color:#237e75}.thumb img{display:block;width:104px;height:64px;object-fit:cover}.thumb span{font-size:11px;padding:4px;display:block}.controls{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:15px}label{font-size:12px;font-weight:bold}select,textarea,input{width:100%;margin-top:5px;padding:8px;border:1px solid #cbd5d5;border-radius:6px;font:inherit}textarea{min-height:60px}.wide{grid-column:1/-1}.actions{margin-top:15px;display:flex;gap:8px;align-items:center}button{padding:9px 12px;border:0;border-radius:7px;font-weight:bold;cursor:pointer}.keep{background:#14766e;color:white}.reject{background:#f8e5e1;color:#9c3730}.save{background:#17232b;color:white}.hint{font-size:12px;color:#62727a;line-height:1.45}@media(max-width:850px){.app{display:block}.grid{grid-template-columns:1fr}.side{max-height:240px}.items{max-height:150px}.main{padding:16px}}</style>
<div class=app><aside class=side><h1>Evidence Triage</h1><p>Find a stable fact first. Do not force a question type onto a random frame.</p><div id=items class=items></div></aside><main class=main><div class=warn><b>Timestamp evidence:</b> images are nearest keyframes to -5/-2/0/+2/+5 seconds, not kf index neighbors. Keep only candidates with one stable, answerable fact.</div><div class=grid><section class=card><div id=meta class=meta></div><div class=viewer><img id=frame></div><div id=frameLabel class=hint></div><div id=thumbs class=thumbs></div></section><section class=card><h2 style="margin-top:0">Triage decision</h2><p class=hint>For now, triage only. Annotation fields are enabled after a candidate is kept and its type is evidence-backed.</p><div class=controls><label>Triage<select id=triage><option>untriaged</option><option>keep</option><option>reject</option><option>needs_context</option></select></label><label>Question type<select id=question_type><option value="">Choose only after keep</option><option>color</option><option>count</option><option>person</option><option>action</option><option>screen_text</option><option>spoken_fact</option><option>place</option><option>temporal_relation</option></select></label><label class=wide>Evidence note<textarea id=review_notes placeholder="What stable fact is visible? Or why is this candidate rejected?"></textarea></label><label>Annotator ID<input id=annotator_id></label><label>Reviewer ID<input id=reviewer_id></label></div><div class=actions><button class=keep onclick="quick('keep')">Keep</button><button class=reject onclick="quick('reject')">Reject</button><button class=save onclick=save()>Save triage</button><span id=result class=hint></span></div></section></div></main></div>
<script>let rows=[],id='',idx=0;function cur(){return rows.find(x=>x.annotation_id===id)}function url(x,e){return '/annotation/evidence/'+x.annotation_id+'/'+encodeURIComponent(e.file)}function nav(){items.innerHTML=rows.map(x=>`<button class="item ${x.annotation_id===id?'active':''}" onclick="pick('${x.annotation_id}')">${x.annotation_id}<small>${x.split} · ${x.triage}</small></button>`).join('')}function pick(x){id=x;idx=0;nav();draw()}function show(n){let x=cur(),e=x.evidence;idx=(n+e.length)%e.length;frame.src=url(x,e[idx]);frameLabel.textContent=`${e[idx].pts_time.toFixed(2)}s | requested ${e[idx].requested_time.toFixed(2)}s | delta ${e[idx].delta_s.toFixed(2)}s`;document.querySelectorAll('.thumb').forEach((b,i)=>b.classList.toggle('active',i===idx))}function draw(){let x=cur();meta.innerHTML=`<span class=pill>${x.split}</span><span class=pill>${x.video_id}</span><span class=pill>target ${x.pts_time.toFixed(2)}s</span>`;thumbs.innerHTML=x.evidence.map((e,i)=>`<button class=thumb onclick="show(${i})"><img src="${url(x,e)}"><span>${e.pts_time.toFixed(2)}s</span></button>`).join('');triage.value=x.triage||'untriaged';question_type.value=x.question_type||'';review_notes.value=x.review_notes||'';annotator_id.value=x.annotator_id||'';reviewer_id.value=x.reviewer_id||'';show(0)}function quick(v){triage.value=v;if(v==='reject')review_notes.value='Rejected: no single stable, answerable fact across timestamp evidence.'}async function save(){let x=cur(),body={triage:triage.value,question_type:question_type.value,review_notes:review_notes.value,annotator_id:annotator_id.value,reviewer_id:reviewer_id.value,status:'draft'};let r=await fetch('/annotation/rows/'+x.annotation_id,{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify(body)});result.textContent=r.ok?'Saved.':await r.text();if(r.ok){Object.assign(x,body);nav()}}fetch('/annotation/rows').then(r=>r.json()).then(x=>{rows=x;id=rows[0].annotation_id;nav();draw()})</script>""")


@app.get("/annotation/rows")
def annotation_rows():
    return _annotation_workspace().list_rows()


@app.put("/annotation/rows/{annotation_id}")
def update_annotation(annotation_id: str, update: AnnotationUpdate):
    try:
        return _annotation_workspace().save(annotation_id, update.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown annotation") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/annotation/export")
def export_annotations():
    return _annotation_workspace().export_reviewed()


@app.get("/annotation/evidence/{annotation_id}/{filename}")
def annotation_evidence(annotation_id: str, filename: str):
    if "/" in annotation_id or "\\" in annotation_id or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=404, detail="not found")
    path = _annotation_workspace().PACK / "evidence" / annotation_id / filename
    if not path.is_file() or path.suffix.lower() != ".jpg":
        raise HTTPException(status_code=404, detail="evidence missing")
    return FileResponse(path)


@app.post("/search/kis", response_model=SearchResponse)
def search_kis(request: KISRequest):
    runtime = get_runtime()
    started = perf_counter()
    key = (request.query.strip(), request.mode, request.topk)
    cached = key in runtime._query_cache
    try:
        if request.include_peaks:
            payload = runtime.kis().search_peaks(request.query, request.topk, request.peaks_per_video)
            raw = payload["results"]
            peaks = payload["peaks"]
        else:
            raw = runtime.search_kis(request.query, request.topk, request.mode)
            peaks = []
        results = [SearchResult(rank=i, **runtime.normalize_kis_result(row).__dict__)
                   for i, row in enumerate(raw, 1)]
        return SearchResponse(results=results,
                              latency_ms=round((perf_counter() - started) * 1000, 3),
                              cache_hit=cached,
                               timings_ms={k: round(v, 3) for k, v in runtime.last_timings_ms.items()}, peaks=peaks)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/search/trake", response_model=TrakeResponse)
def search_trake(request: TrakeRequest):
    if any(not event.strip() for event in request.events):
        raise HTTPException(status_code=422, detail="events must contain non-empty strings")
    runtime = get_runtime()
    started = perf_counter()
    try:
        payload = runtime.search_trake(request.events, request.top_k_videos,
                                       request.include_per_event_scores)
        return TrakeResponse(
            results=payload["results"],
            latency_ms=round((perf_counter() - started) * 1000, 3),
            timings_ms={k: round(float(v), 3) for k, v in runtime.last_timings_ms.items()
                        if isinstance(v, (int, float))},
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/search/vqa", response_model=VQAResponse)
def search_vqa(request: VQARequest):
    runtime = get_runtime()
    started = perf_counter()
    try:
        result = runtime.search_vqa(
            request.query, request.question, request.max_answers,
            request.top_videos, request.frames_per_video,
            request.max_vlm_candidates,
            request.question_type or None,
            request.required_modalities or None,
        )
        return VQAResponse(
            query=result["query"], question=result["question"],
            answers=result["answers"], status=result["status"],
            candidate_count=result.get("candidate_count", 0),
            vlm_candidate_count=result.get("vlm_candidate_count", 0),
            latency_ms=round((perf_counter() - started) * 1000, 3),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/search/vkis", response_model=SearchResponse)
def search_vkis(file: UploadFile = File(...), topk: int = 10,
                agg: str | None = None):
    runtime = get_runtime()
    agg = runtime.policy.vkis_selector if agg is None else agg
    if topk < 1 or topk > 100 or agg not in {
        "max", "mean", "top3", "smooth3", "smooth5", "hybrid0.5", "hybrid0.7"
    }:
        raise HTTPException(status_code=422, detail="invalid topk or agg")
    started = perf_counter(); path = None
    try:
        suffix = os.path.splitext(file.filename or "query.mp4")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            path = tmp.name
            tmp.write(file.file.read())
        raw = runtime.vkis().search_clip(path, topk=topk, agg=agg)
        results = [SearchResult(rank=i, **runtime.normalize_vkis_result(row).__dict__)
                   for i, row in enumerate(raw, 1)]
        runtime.record(started)
        return SearchResponse(results=results,
                              latency_ms=round((perf_counter() - started) * 1000, 3))
    except Exception as exc:
        runtime.record(started, error=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        if path:
            try: os.unlink(path)
            except OSError: pass
