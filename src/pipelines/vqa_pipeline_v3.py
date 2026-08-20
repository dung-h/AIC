"""
VQA Pipeline v3: VLM verify retrieval frames trước khi trả lời.

Kiến trúc (task-aware, sau cleanup phiên 20):
- Stage 1: KIS retrieval (KISFusionRetriever — ViT-L + SO400M fusion + VN→EN translate)
- Stage 2: VLM verify mỗi frame (gemma score 0-10) → loại frame sai
- Stage 3: chọn frame combined cao nhất → VLM answer với ngữ cảnh ASR/OCR

Khác bản cũ: bỏ phụ thuộc SubmissionPipelineV3/router_v8. Ở routed path,
ASR/OCR evidence lấy từ global modality snapshot đã dùng để retrieval; local
per-pack context chỉ còn là compatibility path khi routing không active.
"""
import os, sys, glob, json, base64, urllib.request, time, re, unicodedata, math
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "utils"))
from kis_fusion_retriever import KISFusionRetriever
from paths import KEYFRAMES_DIR, INDEX_DIR
from src.core.providers import provider_for
from src.reranking.query_routing_policy import (
    RoutingConfig,
    build_routing_plan,
    canonical_question_type,
)
from src.reranking.video_rrf import weighted_video_rrf
from src.vqa.evidence_fusion import (
    answer_is_submission_safe,
    build_evidence_packet,
    evidence_support_score,
    render_evidence_prompt,
)
from src.vqa.selector import (
    AllocationResult,
    SOURCE_PRIORITY,
    allocate_recall_preserving_candidates,
    candidate_key,
    candidate_sources,
    deduplicate_candidates,
    has_source,
    selector_metrics,
    stage_record,
)

VLM_PROVIDER = provider_for("vision")
KEY = VLM_PROVIDER.api_key
BASE = VLM_PROVIDER.base_url
KF_DIR = str(KEYFRAMES_DIR)
IDX = str(INDEX_DIR)
# Conservative specialist weights selected by the fixed dev sweep.  Visual
# remains the ranking authority unless ASR/OCR has enough rank evidence to
# move a video without regressing visual R@20 on holdout.
DEFAULT_RRF_WEIGHTS = {"asr": 0.1, "ocr": 0.05}
SUPPORTED_QUESTION_TYPES = frozenset({
    "color", "screen_text", "action", "count", "place", "person",
    "spoken_fact", "temporal_relation",
})
SELECTOR_POLICIES = frozenset({
    "legacy", "balanced", "adaptive", "anchor_preserving",
})


def normalize_question_type(value):
    """Normalize the benchmark taxonomy and reject unknown routing labels."""
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip().casefold()
    if normalized not in SUPPORTED_QUESTION_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_QUESTION_TYPES))
        raise ValueError(f"unsupported question_type {value!r}; expected one of: {allowed}")
    return normalized


def infer_question_type(query, question):
    """Conservatively infer routing taxonomy from an unlabeled live query.

    The classifier only owns modality routing; it is not used as an answer.
    Strong OCR/ASR phrases take priority.  Ambiguous questions remain on the
    visual path (``action``) instead of activating both specialists and then
    incorrectly requiring evidence from ASR *and* OCR.
    """
    raw = f"{query or ''}\n{question or ''}".casefold()
    normalized = unicodedata.normalize("NFKD", raw)
    text = "".join(char for char in normalized if not unicodedata.combining(char))
    text = " ".join(text.split())

    screen_cues = (
        "screen text", "written on", "printed on", "displayed on", "on the screen",
        "on screen", "signboard", "billboard", "license plate", "logo reads",
        "dong chu", "chu tren", "viet tren", "ghi tren", "hien thi", "man hinh",
        "bien hieu", "bang hieu", "bien so", "tieu de tren",
    )
    spoken_cues = (
        "according to", "the speaker", "the narrator", "narration", "voice-over",
        "voiceover", "mentioned", "was said", "did they say", "heard in",
        "spoken", "announced", "reported that", "theo loi", "nguoi noi", "nguoi dan",
        "thuyet minh", "loi binh", "nhac den", "noi rang", "cho biet", "phat bieu",
        "doc vien", "nghe thay", "thong bao",
    )
    if any(cue in text for cue in screen_cues):
        return "screen_text"
    if any(cue in text for cue in spoken_cues):
        return "spoken_fact"

    if any(cue in text for cue in (
        "before or after", "before", "after", "first", "then", "thu tu",
        "truoc hay sau", "truoc khi", "sau khi", "dau tien", "tiep theo",
    )):
        return "temporal_relation"
    if any(cue in text for cue in ("how many", "number of", "bao nhieu", "may nguoi", "may cai")):
        return "count"
    if any(cue in text for cue in ("what color", "which color", "mau gi", "mau nao")):
        return "color"
    if any(cue in text for cue in ("where", "which place", "location", "o dau", "dia diem", "noi nao")):
        return "place"
    if any(cue in text for cue in ("who is", "who was", "which person", "nguoi nao", "ai la", "ai dang")):
        return "person"
    return "action"


class VQAPipelineV3:
    """VQA with VLM verify pre-answer, on the production KIS stack."""

    def __init__(self, translate=True, offline_vlm=False, local_vlm_path=None,
                 local_vlm_4bit=False, answer_provider=None,
                 evidence_verifier=None, kis_retriever=None):
        # ``kis_retriever`` is a deliberate dependency-injection seam for the
        # offline Q&A retrieval benchmark.  The normal production path keeps
        # constructing KISFusionRetriever exactly as before; the benchmark can
        # inject a materialized visual retriever and therefore must not import
        # open_clip or trigger a checkpoint/tokenizer download.
        self.kis = kis_retriever or KISFusionRetriever(translate=translate, alpha=0.4)
        self.km = self.kis.km
        self._asr = None  # lazy
        self._ocr = None  # lazy
        self._asr_by_video = None
        self._ocr_by_video = None
        self._context_cache_stats = {"asr_video_hits": 0, "asr_video_misses": 0,
                                     "ocr_video_hits": 0, "ocr_video_misses": 0}
        self._local_vlm = None
        self.answer_provider = answer_provider
        self.evidence_verifier = evidence_verifier
        if offline_vlm:
            from src.core.local_vlm import LocalVLM
            path = local_vlm_path or os.path.join(os.path.dirname(ROOT), "..", "models", "Qwen2.5-VL-3B-Instruct")
            self._local_vlm = LocalVLM(os.path.abspath(path), load_in_4bit=local_vlm_4bit)

    # ---- lazy context indexes -------------------------------------------
    def _ensure_asr(self):
        if self._asr is None:
            frames = []
            for fp in sorted(glob.glob(os.path.join(IDX, "asr_chunks_*_ts.parquet"))):
                try:
                    frames.append(pd.read_parquet(fp, columns=["chunk", "vid", "start", "end"]))
                except Exception:
                    continue
            self._asr = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
                columns=["chunk", "vid", "start", "end"])
            self._asr_by_video = {
                str(video_id): group.sort_values(["start", "end"], kind="mergesort").reset_index(drop=True)
                for video_id, group in self._asr.groupby("vid", sort=False)
            }
        return self._asr

    def _ensure_ocr(self):
        if self._ocr is None:
            frames = []
            for fp in sorted(glob.glob(os.path.join(IDX, "ocr_*.parquet"))):
                if any(x in fp for x in ["_partial", "_compare", "_gt", "ocr_query", "ocr_chunks"]):
                    continue
                try:
                    frames.append(pd.read_parquet(fp, columns=["video_id", "pts_time", "ocr_text"]))
                except Exception:
                    continue
            self._ocr = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
                columns=["video_id", "pts_time", "ocr_text"])
            self._ocr_by_video = {
                str(video_id): group.sort_values("pts_time", kind="mergesort").reset_index(drop=True)
                for video_id, group in self._ocr.groupby("video_id", sort=False)
            }
        return self._ocr

    # ---- frame path ------------------------------------------------------
    def _frame_path(self, video_id, kf_n):
        return os.path.join(KF_DIR, video_id, f"{int(kf_n):03d}.jpg")

    @staticmethod
    def _repair_mojibake(value):
        """Repair common UTF-8-as-Latin-1 corruption at the VLM boundary."""
        if value is None:
            return ""
        text = str(value)
        if not any(marker in text for marker in ("Ã", "Â", "â", "Æ", "Ð", "Ñ")):
            return text
        try:
            repaired = text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
        return repaired if repaired.count("�") <= text.count("�") else text

    @classmethod
    def _infer_answer_language(cls, question, answer_language=None):
        """Infer the answer language from the question unless explicitly pinned."""
        if answer_language:
            value = str(answer_language).strip().lower()
            if value in {"vi", "vie", "vietnamese", "tiếng việt", "tieng viet"}:
                return "Vietnamese"
            if value in {"en", "eng", "english", "tiếng anh", "tieng anh"}:
                return "English"
            raise ValueError("answer_language must be Vietnamese/vi or English/en")

        text = cls._repair_mojibake(question).lower()
        vietnamese_chars = set(
            "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị"
            "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
        )
        if any(char in vietnamese_chars for char in text):
            return "Vietnamese"
        vi_words = {"màu", "gì", "ai", "đâu", "ở", "bao", "nhiêu", "đang", "làm", "nào"}
        if len(vi_words.intersection(set(text.split()))) >= 2:
            return "Vietnamese"
        return "English"

    @classmethod
    def _build_answer_prompt(cls, query, question, asr_ctx, ocr_ctx,
                             answer_language=None, evidence_packet=None):
        """Build an answer-only prompt shared by remote and local VLM paths."""
        clean_query = cls._repair_mojibake(query).strip()
        clean_question = cls._repair_mojibake(question).strip()
        clean_asr = cls._repair_mojibake(asr_ctx).strip()
        clean_ocr = cls._repair_mojibake(ocr_ctx).strip()
        language = cls._infer_answer_language(clean_question, answer_language)
        if evidence_packet is not None:
            context = render_evidence_prompt(evidence_packet)
        else:
            context = (
                f"ASR context: {clean_asr or '(none)'}\n"
                f"OCR context: {clean_ocr or '(none)'}"
            )
        return (
            f"Answer directly in {language}. Use only the image and nearby ASR/OCR "
            "context; an answer may come from speech or screen text even when it is "
            "not visible in the image. Do not invent details or add facts not needed by the question. "
            "Provide one answer record and no extra fields. Match the grammar of the question: use a concise noun phrase "
            "for color/object/ingredient/utensil questions, and a complete sentence "
            "with the requested subject for action or event questions; do not return "
            "a bare label or gerund when the question asks for an action. If evidence "
            "is insufficient, set abstain=true. Return ONLY valid JSON with exactly "
            'these fields: {"answer":"short answer", "grounding_score":0.0, '
            '"answer_confidence":0.0, "abstain":false}. Scores must be numbers from 0 to 1.\n\n'
            f"Visual description: {clean_query or '(none)'}\n"
            f"Question: {clean_question}\n"
            f"{context}"
        )

    @classmethod
    def _extract_answer_text(cls, value):
        """Extract a submission-safe answer from a local/remote VLM response.

        VLMs occasionally ignore the answer-only instruction and return a JSON
        object, a markdown fence, or an ``Answer:`` label.  Those are transport
        artifacts, not semantic answers.  This helper removes only those
        wrappers and preserves the model's actual wording for evaluation.
        """
        if value is None:
            return ""
        text = unicodedata.normalize("NFKC", cls._repair_mojibake(str(value))).strip()
        if not text:
            return ""

        # Prefer an explicit answer field when the model returned JSON despite
        # the plain-text contract.  Limit parsing to a complete object so a
        # brace in ordinary prose cannot swallow the rest of the answer.
        candidates = [text]
        fenced = re.fullmatch(r"```(?:json|text)?\s*(.*?)\s*```", text,
                              flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            candidates.insert(0, fenced.group(1).strip())
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict) and payload.get("answer") is not None:
                text = str(payload["answer"])
                break

        text = text.strip().strip("`").strip()
        lines = [" ".join(line.split()).strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            return ""
        # If a model emits a labelled answer, keep the labelled value and drop
        # any preceding/following explanation lines.
        label = re.compile(r"^(?:final\s+)?answer\s*:\s*(.*)$",
                           flags=re.IGNORECASE)
        labelled = [match.group(1).strip() for line in lines
                    if (match := label.match(line)) and match.group(1).strip()]
        text = labelled[0] if labelled else lines[0]
        text = re.sub(r"^(?:[-*•]\s+|\d+[.)]\s+)", "", text).strip()
        text = re.sub(r"^(?:final\s+)?answer\s*[：:]\s*", "", text,
                      flags=re.IGNORECASE).strip()
        return " ".join(text.split())

    def _vision_chat(self, payload, timeout):
        """Paced retry for rate-limited OpenAI-compatible VLM endpoints."""
        if not BASE or not KEY:
            raise RuntimeError("vision provider is not configured")
        last_error = None
        for attempt in range(4):
            req = urllib.request.Request(BASE.rstrip("/") + "/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code != 429 or attempt == 3:
                    raise
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
                time.sleep(delay)
            except Exception as exc:
                last_error = exc
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        raise last_error or RuntimeError("vision request failed")

    # ---- VLM calls -------------------------------------------------------
    def _vlm_verify(self, image_path, query, model=None):
        """Score 0-10 how well image matches query. Returns None on API failure (not 5.0)."""
        if not os.path.exists(image_path):
            import logging
            logging.getLogger(__name__).warning(f"[VQA] image missing: {image_path}")
            return None
        b64 = base64.b64encode(open(image_path, "rb").read()).decode()
        query = self._repair_mojibake(query)
        prompt = (f"On a scale from 0 to 10, how well does this image match the description: \"{query}\"? "
                  f"Output ONLY a single integer 0-10.")
        pl = {"model": model or VLM_PROVIDER.model, "messages": [{"role": "user", "content": [
                  {"type": "text", "text": prompt},
                  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
              "max_tokens": 5, "temperature": 0.0}
        try:
            d = self._vision_chat(pl, timeout=60)
            txt = d["choices"][0]["message"]["content"].strip()
            import re
            m = re.search(r"\d+", txt)
            if m:
                return float(m.group(0))
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[VQA] _vlm_verify API failed: {e}")
        return None  # explicit None — callers must handle, not assume 5.0

    def _vlm_pairwise(self, image_a, image_b, query, question, model=None):
        """Choose between two same-video frames for a VQA query."""
        if not os.path.exists(image_a) or not os.path.exists(image_b):
            return None
        def data_url(path):
            with open(path, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode()
            return f"data:image/jpeg;base64,{encoded}"
        query = self._repair_mojibake(query)
        question = self._repair_mojibake(question)
        prompt = (
            "Two frames come from the same video. Choose the frame that best "
            "matches the visual description and supports answering the question. "
            "Return ONLY A, B, or TIE.\n"
            f"Visual description: {query}\nQuestion: {question}"
        )
        payload = {"model": model or VLM_PROVIDER.model,
                   "messages": [{"role": "user", "content": [
                       {"type": "text", "text": prompt},
                       {"type": "text", "text": "FRAME A"},
                       {"type": "image_url", "image_url": {"url": data_url(image_a)}},
                       {"type": "text", "text": "FRAME B"},
                       {"type": "image_url", "image_url": {"url": data_url(image_b)}}]}],
                   "max_tokens": 4, "temperature": 0.0}
        try:
            response = self._vision_chat(payload, timeout=90)
            text = response["choices"][0]["message"]["content"].strip().upper()
            if text.startswith("A"):
                return "A"
            if text.startswith("B"):
                return "B"
            if text.startswith("TIE"):
                return "TIE"
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"[VQA] pairwise API failed: {exc}")
        return None

    def _vlm_answer_with_context(self, image_path, question, asr_ctx, ocr_ctx,
                                   model=None, *, query="", answer_language=None):
        b64 = base64.b64encode(open(image_path, "rb").read()).decode()
        prompt = self._build_answer_prompt(
            query, question, asr_ctx, ocr_ctx, answer_language=answer_language)
        pl = {"model": model or VLM_PROVIDER.model,
              "messages": [{"role": "user", "content": [
                  {"type": "text", "text": prompt},
                  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
              "max_tokens": 200, "temperature": 0.0}
        d = self._vision_chat(pl, timeout=120)
        return d["choices"][0]["message"]["content"].strip()

    # ---- context lookups -------------------------------------------------
    def _asr_context(self, video_id, pts_time, window=15.0):
        self._ensure_asr()
        m = self._asr_by_video.get(str(video_id)) if self._asr_by_video is not None else None
        if m is None or len(m) == 0:
            self._context_cache_stats["asr_video_misses"] += 1
            return ""
        self._context_cache_stats["asr_video_hits"] += 1
        m2 = m[(m.start <= pts_time + window) & (m.end >= pts_time - window)]
        if len(m2) == 0:
            return ""
        return " ".join(m2.chunk.head(3).tolist())[:1500]

    def _ocr_context(self, video_id, pts_time, window=10.0):
        self._ensure_ocr()
        m = self._ocr_by_video.get(str(video_id)) if self._ocr_by_video is not None else None
        if m is None or len(m) == 0:
            self._context_cache_stats["ocr_video_misses"] += 1
            return ""
        self._context_cache_stats["ocr_video_hits"] += 1
        m = m[(m.pts_time >= pts_time - window) & (m.pts_time <= pts_time + window)]
        if len(m) == 0:
            return ""
        return " | ".join(m.ocr_text.head(5).tolist())[:600]

    def _build_evidence_packet(self, candidate, query, question,
                               evidence_provider=None, modalities=None):
        """Build timestamped evidence from the active retrieval snapshot.

        Routed production requests pass the global modality router here so
        retrieval and answering share one ASR/OCR source. The legacy local
        context indexes remain available only when no routed provider is
        supplied.
        """
        if evidence_provider is not None:
            builder = getattr(evidence_provider, "evidence_packet_for_candidate", None)
            if not callable(builder):
                raise RuntimeError(
                    "active modality provider cannot build grounded evidence packets"
                )
            return builder(
                candidate,
                query,
                question,
                modalities=modalities,
            )
        self._ensure_asr()
        self._ensure_ocr()
        video_id = str(candidate["video_id"])
        asr_rows = (self._asr_by_video or {}).get(video_id, [])
        ocr_rows = (self._ocr_by_video or {}).get(video_id, [])
        return build_evidence_packet(
            candidate, asr_rows=asr_rows, ocr_rows=ocr_rows,
            query=query, question=question,
        )

    @staticmethod
    def _offline_answer(question, asr_ctx, ocr_ctx):
        """Return grounded text evidence without pretending to see the frame."""
        evidence = []
        if ocr_ctx:
            evidence.append({"source": "ocr", "text": ocr_ctx})
        if asr_ctx:
            evidence.append({"source": "asr", "text": asr_ctx})
        if not evidence:
            return {"answer": None, "status": "unavailable", "reason": "no_local_text_evidence"}
        return {
            "answer": None,
            "status": "evidence_only",
            "reason": "local_vlm_not_configured",
            "question": question,
            "evidence": evidence,
        }

    def _local_candidates(self, query, video_ids, frames_per_video,
                          temporal_consensus=False, consensus_weight=0.3,
                          consensus_margin=0.0, diversify=False):
        """Return local frames without changing global video ranking."""
        if frames_per_video < 1:
            return []
        if getattr(self.kis, "materialized", False):
            return self.kis.materialized_candidates(video_ids, frames_per_video)
        translated = self.kis.translate(query)
        q_vitl = self.kis._encode_text(self.kis.m_vitl, self.kis.tk_vitl, translated)
        q_so = self.kis._encode_text(self.kis.m_so, self.kis.tk_so, translated)
        scores_vitl = self.kis.F_vitl @ q_vitl
        q_so = q_so.astype(self.kis.F_so.dtype) if getattr(self.kis.F_so, "dtype", None) == np.float16 else q_so
        scores_so = self.kis.F_so @ q_so
        candidates = []
        for video_id in video_ids:
            indices = self.kis._video_rows[str(video_id)]
            vitl = scores_vitl[indices].astype(np.float32, copy=False)
            so = scores_so[indices].astype(np.float32, copy=False)
            fused = self.kis.alpha * (vitl - vitl.mean()) / (vitl.std() + 1e-6)
            fused += (1.0 - self.kis.alpha) * (so - so.mean()) / (so.std() + 1e-6)
            rank_score = fused
            if temporal_consensus and len(fused) > 2:
                # Reward a strong nearby keyframe, but keep the original score
                # dominant so isolated semantic matches are not discarded.
                neighbor = np.full(len(fused), -np.inf, dtype=np.float32)
                for shift in (-2, -1, 1, 2):
                    if shift < 0:
                        neighbor[-shift:] = np.maximum(neighbor[-shift:], fused[:shift])
                    else:
                        neighbor[:-shift] = np.maximum(neighbor[:-shift], fused[shift:])
                neighbor = np.where(np.isfinite(neighbor), neighbor, fused)
                rank_score = (1.0 - consensus_weight) * fused + consensus_weight * neighbor
                if consensus_margin > 0:
                    best_semantic = int(np.argmax(fused))
                    best_consensus = int(np.argmax(rank_score))
                    if (rank_score[best_consensus] - rank_score[best_semantic]
                            < consensus_margin):
                        rank_score = fused
            order = np.argsort(-rank_score)
            if diversify and frames_per_video > 1 and len(order) > 1:
                # Keep the strongest semantic peak, then spend remaining
                # visual slots on temporally separated peaks. This prevents a
                # screen-text query from consuming all three slots on one
                # repeated shot when OCR coverage is unavailable.
                rows = self.km.iloc[indices].reset_index(drop=True)
                times = rows.pts_time.to_numpy(dtype=np.float32)
                selected = [int(order[0])]
                score_min = float(np.min(rank_score))
                score_span = float(np.max(rank_score) - score_min) or 1.0
                time_span = float(np.max(times) - np.min(times)) or 1.0
                remaining = [int(value) for value in order[1:]]
                while remaining and len(selected) < frames_per_video:
                    best = max(
                        remaining,
                        key=lambda value: (
                            0.7 * (float(rank_score[value]) - score_min) / score_span
                            + 0.3 * min(abs(float(times[value] - times[other])) for other in selected) / time_span,
                            -value,
                        ),
                    )
                    selected.append(best)
                    remaining.remove(best)
                local_order = selected
            else:
                local_order = [int(value) for value in order[:frames_per_video]]
            for local_index in local_order:
                row = self.km.iloc[int(indices[local_index])]
                candidates.append((str(video_id), int(row.frame_idx), int(row.kf_n), float(fused[local_index])))
        return candidates

    def _materialized_candidates_all(self, video_ids):
        """Return the active frozen lattice without a per-video depth cap.

        This seam is used only by the benchmark retrieval-pool path.  The
        materialized adapter owns the ordered frozen rows; reading its active
        rows preserves the source order required by the candidate-pool digest.
        Normal/model-backed retrieval never calls this helper.
        """
        active = getattr(self.kis, "_active_candidates", None)
        if not callable(active):
            raise RuntimeError(
                "materialized retriever must expose its active frozen lattice"
            )
        allowed = {str(video_id) for video_id in video_ids}
        output = []
        for candidate in active():
            if str(candidate.get("video_id")) not in allowed:
                continue
            output.append((
                str(candidate["video_id"]),
                int(candidate["frame_idx"]),
                int(candidate["kf_n"]),
                float(candidate["base_score"]),
            ))
        return output

    def _temporal_candidates(self, query, video_id, peak_count=2,
                             radius_s=3.0, output_count=5):
        """Expand semantic peaks to actual timestamp-neighbor candidates."""
        translated = self.kis.translate(query)
        q_vitl = self.kis._encode_text(self.kis.m_vitl, self.kis.tk_vitl, translated)
        q_so = self.kis._encode_text(self.kis.m_so, self.kis.tk_so, translated)
        indices = self.kis._video_rows[str(video_id)]
        vitl = (self.kis.F_vitl[indices] @ q_vitl).astype(np.float32)
        so = self.kis.F_so[indices] @ q_so.astype(self.kis.F_so.dtype)
        so = np.asarray(so, dtype=np.float32)
        fused = self.kis.alpha * (vitl - vitl.mean()) / (vitl.std() + 1e-6)
        fused += (1.0 - self.kis.alpha) * (so - so.mean()) / (so.std() + 1e-6)
        rows = self.km.iloc[indices].reset_index(drop=True)
        times = rows.pts_time.to_numpy(dtype=np.float32)
        peaks = np.argsort(-fused)[:peak_count]
        selected = set()
        for peak in peaks:
            local = np.flatnonzero(np.abs(times - times[peak]) <= radius_s)
            selected.update(int(x) for x in local)
        temporal_score = fused.copy()
        for i in selected:
            local = np.flatnonzero(np.abs(times - times[i]) <= radius_s)
            temporal_score[i] = 0.7 * fused[i] + 0.3 * float(np.max(fused[local]))
        ordered = sorted(selected, key=lambda i: float(temporal_score[i]), reverse=True)[:output_count]
        return [(str(video_id), int(rows.iloc[i].frame_idx), int(rows.iloc[i].kf_n),
                 float(fused[i])) for i in ordered]

    # ---- main ------------------------------------------------------------
    def answer(self, query, question, n_retrieve=5, n_verify=3, frames_per_video=1,
               temporal_consensus=False, offline=False, answer_language=None):
        """
        Args:
          n_retrieve: số candidates từ KIS
          n_verify: số frame VLM verify (chậm hơn nhưng chính xác hơn)
          offline: return indexed OCR/ASR evidence without calling a VLM
        """
        res = self.kis.search(query, topk=n_retrieve)
        if not res:
            return {"answer": None, "error": "no retrieval"}

        # KISRetriever results: (video_id, frame_idx, kf_n, score)
        candidate_rows = res[:n_retrieve]
        if frames_per_video > 1:
            candidate_rows = self._local_candidates(
                query, [row[0] for row in candidate_rows], frames_per_video,
                temporal_consensus=temporal_consensus)
        verified = []
        for vid, fidx, kf_n, base_sc in candidate_rows[:n_verify]:
            fp = self._frame_path(vid, kf_n)
            if not os.path.exists(fp):
                continue
            # pts_time from km for context lookups
            m = self.km[(self.km.video_id == vid) & (self.km.kf_n == kf_n)]
            pts = float(m.iloc[0].pts_time) if len(m) > 0 else 0.0
            vlm_sc = None if offline else self._vlm_verify(fp, query)
            # None = API failure; use base_score as fallback for ranking but flag it
            verified.append({"video": vid, "frame_idx": int(fidx), "kf_n": int(kf_n),
                             "pts_time": pts, "base_score": float(base_sc),
                             "vlm_score": vlm_sc,        # may be None
                             "vlm_ok": vlm_sc is not None,
                             "frame_path": fp})

        if not verified:
            return {"answer": None, "error": "no valid frames"}

        # In local-frame mode, frames are alternatives within the same top KIS
        # video. Never compare their per-video z-scores across videos.
        # If vlm_score is None (API failure), use 5.0 as neutral fallback but log warning
        bases = np.array([v["base_score"] for v in verified])
        if bases.max() != bases.min():
            bn = (bases - bases.min()) / (bases.max() - bases.min())
        else:
            bn = np.zeros_like(bases)
        n_failed = sum(1 for v in verified if not v["vlm_ok"])
        if n_failed:
            import logging
            logging.getLogger(__name__).warning(
                f"[VQA] {n_failed}/{len(verified)} VLM verify calls failed — using base score only")
        vlms = np.array([v["vlm_score"] if v["vlm_score"] is not None else 5.0
                         for v in verified]) / 10.0
        combined = 0.4 * bn + 0.6 * vlms
        if frames_per_video > 1:
            first_video = candidate_rows[0][0]
            order = np.array([idx for idx in np.argsort(-combined)
                              if verified[idx]["video"] == first_video])
            if len(order) == 0:
                order = np.argsort(-combined)
        else:
            order = np.argsort(-combined)

        all_attempts = []
        best = None
        for idx in order:
            v = verified[idx]
            asr = self._asr_context(v["video"], v["pts_time"])
            ocr = self._ocr_context(v["video"], v["pts_time"])
            if offline:
                if self._local_vlm is not None:
                    prompt = self._build_answer_prompt(
                        query, question, asr, ocr, answer_language=answer_language)
                    try:
                        ans = {"answer": self._local_vlm.answer(v["frame_path"], prompt),
                               "status": "answered_local", "reason": "local_vlm"}
                    except Exception as exc:
                        ans = self._offline_answer(question, asr, ocr)
                        ans["fallback_error"] = str(exc)[:160]
                else:
                    ans = self._offline_answer(question, asr, ocr)
                all_attempts.append({**v, **ans, "asr_ctx": bool(asr),
                                     "ocr_ctx": bool(ocr), "combined": float(combined[idx])})
                if best is None and ans["status"] == "evidence_only":
                    best = all_attempts[-1]
                continue
            try:
                ans = self._vlm_answer_with_context(
                    v["frame_path"], question, asr, ocr,
                    query=query, answer_language=answer_language)
            except Exception as e:
                ans = f"ERROR: {str(e)[:80]}"
            all_attempts.append({**v, "answer": ans, "asr_ctx": bool(asr), "ocr_ctx": bool(ocr),
                                 "combined": float(combined[idx])})
            low = self._repair_mojibake(ans).lower()
            if not self._looks_like_nonanswer(low):
                best = all_attempts[-1]
                break

        if best is None and all_attempts:
            best = all_attempts[0]

        return {
            "query": query, "question": question,
            "best": best, "all_attempts": all_attempts,
            "verification_count": len(verified),
            "verification_success_count": sum(v["vlm_ok"] for v in verified),
            "winner_signal": "visual_vitl",
            "mode": "offline_evidence" if offline else "vlm",
        }

    @staticmethod
    def _looks_like_nonanswer(value: str | None) -> bool:
        """Reject empty/abstaining local-VLM output before submission export."""
        if not isinstance(value, str) or not value.strip():
            return True
        if not answer_is_submission_safe(value):
            return True
        raw_text = " ".join(str(value).split()).lower()
        text = " ".join(VQAPipelineV3._repair_mojibake(value).split()).lower()
        # Some historical reviewer strings were corrupted more than once and
        # contain a mix of UTF-8 bytes plus smart punctuation. They cannot be
        # round-tripped through latin-1 safely, but the abstention intent is
        # still unambiguous from the ``khong ... xac`` fragments.
        mojibake_khong = "kh" + chr(0xC3) + chr(0xB4) + "ng"
        mojibake_xac = "x" + chr(0xC3) + chr(0xA1) + "c"
        mojibake_the = "th" + chr(0xE1) + chr(0xBB)
        mojibake_khong_alt = "kh" + chr(0xE3) + chr(0xB4) + "ng"
        mojibake_xac_alt = "x" + chr(0xE3) + chr(0xA1) + "c"
        if any((mojibake_khong in candidate or mojibake_khong_alt in candidate) and
               (mojibake_xac in candidate or mojibake_xac_alt in candidate or mojibake_the in candidate)
               for candidate in (raw_text, text)):
            return True
        return any(marker in candidate for candidate in (raw_text, text)
                   for marker in (
            "không đủ thông tin", "không có thông tin", "không thể xác định",
            "không nhìn thấy", "không thể trả lời", "không rõ", "cannot determine", "cannot answer",
            "unable to answer", "not enough information", "insufficient evidence",
            "no information", "i don't know", "unknown", "evidence-only", "evidence only",
            "no answer", "not available", "null", "n/a",
        ))

    @staticmethod
    def _parse_modalities(required_modalities):
        values = required_modalities or []
        if isinstance(values, str):
            values = values.replace(";", ",").split(",")
        normalized = [str(value).strip().lower() for value in values if str(value).strip()]
        unknown = sorted(set(normalized) - {"visual", "asr", "ocr"})
        if unknown:
            raise ValueError(
                "unsupported required modality(s): " + ", ".join(unknown) +
                "; expected visual, asr, or ocr"
            )
        # visual is the mandatory base channel; only specialist channels need
        # to be returned to the global router. Preserve order and deduplicate
        # so a malformed `asr,asr` request cannot run a channel twice.
        return list(dict.fromkeys(value for value in normalized if value in {"asr", "ocr"}))

    def _temporal_neighbor_candidates(self, video_id, kf_n, *, radius=2):
        rows = self.km[self.km.video_id.astype(str) == str(video_id)].sort_values("pts_time").reset_index(drop=True)
        positions = np.flatnonzero(rows.kf_n.to_numpy(dtype=np.int64) == int(kf_n))
        if len(positions) == 0:
            return []
        center = int(positions[0])
        out = []
        for offset in (-radius, -1, 1, radius):
            position = center + offset
            if 0 <= position < len(rows):
                row = rows.iloc[position]
                out.append({"video_id": str(video_id), "frame_idx": int(row.frame_idx),
                            "kf_n": int(row.kf_n), "pts_time": float(row.pts_time),
                            "base_score": 0.0, "source": "temporal"})
        return out

    @staticmethod
    def _attach_evidence_frames(candidates, candidate_pool, *, limit=3):
        """Attach bounded, same-video frame evidence to each selected anchor.

        Retrieval and selection still produce one canonical answer frame.  The
        extra frames are an input bundle for the answer provider only.  They
        are selected from the already validated candidate pool, never invented
        from a frame number or pulled from another video.
        """
        grouped = {}
        for raw in candidate_pool:
            video_id = str(raw.get("video_id", ""))
            if video_id and raw.get("kf_n") is not None:
                grouped.setdefault(video_id, []).append(raw)

        def source_priority(item):
            sources = set(candidate_sources(item))
            if sources.intersection({"asr", "ocr"}):
                return 0
            if "temporal" in sources:
                return 1
            return 2

        output = []
        for anchor in candidates:
            video_id = str(anchor.get("video_id", ""))
            anchor_kf = int(anchor.get("kf_n", -1))
            anchor_pts = float(anchor.get("pts_time", 0.0) or 0.0)
            options = []
            for item in grouped.get(video_id, ()):
                if int(item.get("kf_n", -1)) == anchor_kf:
                    continue
                if item.get("frame_idx") is None or item.get("frame_path") is None:
                    continue
                options.append(item)
            options.sort(key=lambda item: (
                source_priority(item),
                abs(float(item.get("pts_time", anchor_pts) or anchor_pts) - anchor_pts),
                -float(item.get("base_score", item.get("modality_score", 0.0)) or 0.0),
                int(item.get("kf_n", 0)),
                int(item.get("frame_idx", 0)),
            ))
            evidence_frames = []
            seen = {anchor_kf}
            for item in options:
                kf_n = int(item["kf_n"])
                if kf_n in seen:
                    continue
                seen.add(kf_n)
                item_sources = candidate_sources(item)
                specialist_modality = next(
                    (source for source in ("asr", "ocr") if source in item_sources),
                    None,
                )
                evidence_frames.append({
                    "video_id": video_id,
                    "frame_idx": int(item["frame_idx"]),
                    "kf_n": kf_n,
                    "pts_time": float(item.get("pts_time", anchor_pts) or anchor_pts),
                    "frame_path": item.get("frame_path"),
                    "role": "evidence" if specialist_modality is not None
                    else "neighbor",
                    "modality": specialist_modality or item.get("source", "visual"),
                })
                if len(evidence_frames) >= max(0, int(limit)):
                    break
            item = dict(anchor)
            item["evidence_frames"] = evidence_frames
            output.append(item)
        return output

    @staticmethod
    def _provider_frame_evidence(candidate, packet=None):
        """Convert a packet into a validated provider frame bundle.

        ``FrameEvidence.frame_id`` is always the canonical ``frame_idx``.  A
        malformed or cross-video extra frame is dropped; the anchor is kept as
        the minimum viable bundle.  The provider contract then performs its
        own path validation for local/remote transports.
        """
        from src.vqa.answer_provider import FrameEvidence

        video_id = str(candidate.get("video_id", ""))
        rows = list((packet or {}).get("frames", ()) or ())
        if not rows:
            rows = [{
                "video_id": video_id,
                "frame_idx": candidate.get("frame_idx"),
                "frame_path": candidate.get("frame_path"),
                "pts_time": candidate.get("pts_time"),
                "role": "anchor",
            }]

        frames = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or str(row.get("video_id", video_id)) != video_id:
                continue
            frame_id = row.get("frame_idx", row.get("frame_id"))
            if isinstance(frame_id, bool):
                continue
            try:
                frame_id = int(frame_id)
            except (TypeError, ValueError):
                continue
            if frame_id < 0 or frame_id in seen:
                continue
            path = row.get("frame_path")
            if path is not None and not str(path).strip():
                path = None
            try:
                pts_time = row.get("pts_time")
                if pts_time is not None:
                    pts_time = float(pts_time)
                modality = str(row.get("modality") or row.get("source") or "visual").strip().lower()
                # Temporal neighbors are still visual evidence at the provider
                # contract boundary; ``temporal`` is a selector role, not an
                # answer-provider evidence source.
                if modality not in {"visual", "asr", "ocr"}:
                    modality = "visual"
                frames.append(FrameEvidence(
                    frame_id=frame_id,
                    frame_path=path,
                    pts_time=pts_time,
                    modality=modality,
                ))
            except (TypeError, ValueError):
                continue
            seen.add(frame_id)

        if not frames:
            raise ValueError("candidate has no valid canonical frame evidence")
        return tuple(frames)

    @staticmethod
    def _local_evidence_paths(candidate, packet=None, *, limit=12):
        """Return ordered, same-video image paths for the local answer call.

        The packet order is intentional: anchor first, then specialist or
        temporal evidence.  Missing paths are skipped, but the candidate
        anchor is restored when a packet only carries canonical metadata.
        The hard limit prevents an answer call from silently turning into an
        unbounded video scan.
        """
        paths = []
        seen = set()
        video_id = str(candidate.get("video_id", ""))
        for row in list((packet or {}).get("frames", ()) or ()):
            if not isinstance(row, dict) or str(row.get("video_id", video_id)) != video_id:
                continue
            path = row.get("frame_path")
            if path is None or not str(path).strip():
                continue
            path = str(path)
            if path not in seen:
                paths.append(path)
                seen.add(path)
        anchor_path = candidate.get("frame_path")
        if anchor_path is not None and str(anchor_path).strip():
            anchor_path = str(anchor_path)
            if anchor_path not in seen:
                paths.insert(0, anchor_path)
        return paths[:max(1, min(int(limit), 12))]

    @staticmethod
    def _local_metadata_record(raw):
        """Normalize a local model response without changing answer semantics."""
        from src.core.local_vlm import LocalVLM

        return LocalVLM._parse_metadata(raw)

    @classmethod
    def _local_answer_with_evidence(cls, model, frame_paths, prompt, *, max_new_tokens):
        """Run structured local answering over bounded multi-frame evidence.

        A model exposing ``answer_frames_with_metadata`` gets one genuine
        multi-image call.  A model exposing only single-image metadata is
        evaluated once per frame and the best non-abstaining record is chosen
        deterministically by confidence, then grounding, then frame order.
        This makes the fallback explicit instead of pretending independent
        calls saw the same multi-frame context.
        """
        paths = list(frame_paths[:12])
        if not paths:
            return {
                "answer": "", "grounding_score": 0.0,
                "answer_confidence": 0.0, "abstain": True,
                "parse_failed": False, "reason": "no_evidence_frames",
            }

        multi = getattr(model, "answer_frames_with_metadata", None)
        if len(paths) > 1 and callable(multi):
            return cls._local_metadata_record(
                multi(paths, prompt, max_new_tokens=max_new_tokens)
            )

        multi_plain = getattr(model, "answer_frames", None)
        if len(paths) > 1 and callable(multi_plain):
            return cls._local_metadata_record(
                multi_plain(paths, prompt, max_new_tokens=max_new_tokens)
            )

        single_structured = getattr(model, "answer_with_metadata", None)
        if callable(single_structured):
            records = []
            for index, path in enumerate(paths):
                try:
                    record = cls._local_metadata_record(
                        single_structured(path, prompt, max_new_tokens=max_new_tokens)
                    )
                except Exception:
                    continue
                record["_frame_order"] = index
                if (record.get("answer") and not record.get("abstain")
                        and not record.get("parse_failed")):
                    records.append(record)
            if records:
                return max(records, key=lambda item: (
                    float(item.get("answer_confidence", 0.0)),
                    float(item.get("grounding_score", 0.0)),
                    -int(item.get("_frame_order", 0)),
                ))
            return cls._local_metadata_record(records[0] if records else "")

        single_plain = getattr(model, "answer", None)
        if not callable(single_plain):
            raise RuntimeError("local model must expose answer, answer_with_metadata, or answer_frames")
        records = []
        for index, path in enumerate(paths):
            try:
                record = cls._local_metadata_record(
                    single_plain(path, prompt, max_new_tokens=max_new_tokens)
                )
            except Exception:
                continue
            record["_frame_order"] = index
            if record.get("answer") and not record.get("abstain"):
                records.append(record)
        if not records:
            return cls._local_metadata_record("")
        return records[0]

    # ---- bounded candidate ownership ------------------------------------
    # This allocator intentionally lives at the VQA pipeline boundary rather
    # than in the generic selector. It is the owner of the Q&A-specific
    # decision to keep a *visual* answer anchor while separately reserving
    # ASR/OCR evidence. TRAKE and generic retrieval must not inherit that
    # policy accidentally.
    @staticmethod
    def _candidate_source_score(candidate, source):
        """Return one channel-local score without cross-modality comparison."""
        source = str(source).strip().lower()
        for record in candidate.get("provenance", ()) or ():
            if not isinstance(record, dict):
                continue
            if str(record.get("source", "")).strip().lower() != source:
                continue
            value = record.get("score")
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value == value:
                return value
        key = "modality_score" if source in {"asr", "ocr"} else "base_score"
        try:
            value = float(candidate.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        return value if value == value else 0.0

    @classmethod
    def _candidate_channel_key(cls, candidate, source):
        """Deterministic order inside a video/source only."""
        try:
            retrieval_rank = int(candidate.get("retrieval_rank", candidate.get("rank", 10**9)))
        except (TypeError, ValueError):
            retrieval_rank = 10**9
        try:
            kf_n = int(candidate.get("kf_n", 10**9))
        except (TypeError, ValueError):
            kf_n = 10**9
        try:
            frame_idx = int(candidate.get("frame_idx", 10**9))
        except (TypeError, ValueError):
            frame_idx = 10**9
        return (
            -cls._candidate_source_score(candidate, source),
            retrieval_rank,
            kf_n,
            frame_idx,
            tuple(candidate_sources(candidate)),
        )

    def _canonicalize_selector_pool(self, candidates):
        """Reject unmapped rows and remap frame IDs from the canonical map.

        Retrieval channels identify a keyframe by ``(video_id, kf_n)``. The
        map-keyframes table is the sole authority for the submitted
        ``frame_idx``. A stale upstream ``frame_idx`` is therefore remapped
        (and recorded), while an unknown keyframe is rejected. Tiny isolated
        unit fixtures sometimes construct the pipeline without ``km``; those
        stay testable but are explicitly reported as unchecked, never treated
        as a production canonical validation pass.
        """
        raw = [dict(item) for item in candidates if isinstance(item, dict)]
        km = getattr(self, "km", None)
        required = {"video_id", "kf_n", "frame_idx"}
        if not isinstance(km, pd.DataFrame) or not required.issubset(km.columns):
            return raw, [], False

        lookup = getattr(self, "_canonical_frame_lookup", None)
        if lookup is None:
            lookup = {}
            for row in km.loc[:, ["video_id", "kf_n", "frame_idx"]].itertuples(index=False):
                try:
                    lookup[(str(row.video_id), int(row.kf_n))] = int(row.frame_idx)
                except (TypeError, ValueError):
                    continue
            self._canonical_frame_lookup = lookup

        accepted = []
        rejected = []
        for ordinal, item in enumerate(raw):
            try:
                video_id = str(item["video_id"])
                kf_n = int(item["kf_n"])
            except (KeyError, TypeError, ValueError):
                rejected.append({
                    "ordinal": ordinal, "selected": False,
                    "reason": "invalid_canonical_key",
                    "source": list(candidate_sources(item)),
                    "source_score": {
                        source: self._candidate_source_score(item, source)
                        for source in candidate_sources(item)
                    },
                    "video_id": str(item.get("video_id", "")),
                })
                continue
            canonical_frame_idx = lookup.get((video_id, kf_n))
            if canonical_frame_idx is None:
                rejected.append({
                    "ordinal": ordinal, "selected": False,
                    "reason": "canonical_key_not_found",
                    "source": list(candidate_sources(item)),
                    "source_score": {
                        source: self._candidate_source_score(item, source)
                        for source in candidate_sources(item)
                    },
                    "video_id": video_id, "kf_n": kf_n,
                })
                continue
            original_frame_idx = item.get("frame_idx")
            try:
                original_frame_idx = int(original_frame_idx)
            except (TypeError, ValueError):
                original_frame_idx = None
            item["video_id"] = video_id
            item["kf_n"] = kf_n
            item["frame_idx"] = canonical_frame_idx
            if original_frame_idx != canonical_frame_idx:
                item["_canonical_frame_remapped"] = True
                item["_input_frame_idx"] = original_frame_idx
            accepted.append(item)
        return accepted, rejected, True

    def _allocate_anchor_preserving_candidates(self, candidates, ranked_video_ids,
                                                max_vlm_candidates, *,
                                                specialist_modalities=(),
                                                per_video_cap=2):
        """Allocate visual anchors before diversity, with grounded text quota.

        The old adaptive allocator was useful for generic diversity but could
        replace the strongest visual frame in a video with an ASR/OCR row, or
        spend a 12-frame budget on depth before a competitive video received
        an answer anchor. For Q&A this is the wrong ownership boundary:
        visual relevance chooses an answer frame, while ASR/OCR contributes
        an additional *grounded* evidence candidate for its routed task.

        This function does not inspect a GT label, model output, or raw scores
        across channels. All choices are deterministic from ranked videos,
        channel-local ordering, and materialized provenance.
        """
        if not isinstance(max_vlm_candidates, int) or isinstance(max_vlm_candidates, bool):
            raise ValueError("max_vlm_candidates must be an integer")
        if max_vlm_candidates < 0:
            raise ValueError("max_vlm_candidates must be >= 0")
        if not isinstance(per_video_cap, int) or isinstance(per_video_cap, bool) or per_video_cap < 1:
            raise ValueError("per_video_cap must be an integer >= 1")

        specialist_modalities = tuple(dict.fromkeys(
            str(value).strip().lower() for value in specialist_modalities
            if str(value).strip().lower() in {"asr", "ocr"}
        ))
        canonical_pool, canonical_rejections, canonical_checked = self._canonicalize_selector_pool(candidates)
        pool = deduplicate_candidates(canonical_pool)
        ranked = list(dict.fromkeys(str(value) for value in ranked_video_ids))
        rank_by_video = {video_id: rank for rank, video_id in enumerate(ranked)}
        pool.sort(key=lambda item: (
            rank_by_video.get(str(item.get("video_id", "")), 10**9),
            int(item.get("kf_n", 10**9)),
            int(item.get("frame_idx", 10**9)),
            tuple(candidate_sources(item)),
        ))
        grouped = {video_id: [] for video_id in ranked}
        for item in pool:
            video_id = str(item.get("video_id", ""))
            if video_id in grouped:
                grouped[video_id].append(item)

        def best(video_id, source=None, *, payload_only=False, excluded=()):
            excluded_keys = set(excluded)
            rows = []
            for item in grouped.get(video_id, ()):
                key = candidate_key(item)
                if key in excluded_keys:
                    continue
                if source is not None and not has_source(item, source):
                    continue
                if payload_only and not self._candidate_has_modality_payload(item, source):
                    continue
                rows.append(item)
            if not rows:
                return None
            preferred = source or "visual"
            return min(rows, key=lambda item: self._candidate_channel_key(item, preferred))

        eligible = [video_id for video_id in ranked if grouped[video_id]]
        visual_anchors = {
            video_id: (best(video_id, "visual") or best(video_id))
            for video_id in eligible
        }
        visual_anchors = {
            video_id: candidate for video_id, candidate in visual_anchors.items()
            if candidate is not None
        }

        # Reserve at most one real specialist row per active channel. A
        # same-frame multimodal hit fulfils the reservation without spending a
        # second budget slot. We deliberately choose the earliest competitive
        # video, not the largest ASR/OCR score, because channel score scales
        # are incomparable across videos.
        provisional_limit = min(max_vlm_candidates, len(visual_anchors))
        provisional_videos = eligible[:provisional_limit]
        reservations = []
        for modality in specialist_modalities:
            for video_id in provisional_videos:
                anchor = visual_anchors.get(video_id)
                specialist = best(video_id, modality, payload_only=True)
                if specialist is not None:
                    reservations.append((modality, video_id, anchor, specialist))
                    break

        # Reservations that reuse an anchor are free. Distinct specialist
        # frames consume one slot each, so reduce the visual coverage budget
        # before any diversity/depth decision is made.
        distinct_reservation_keys = []
        for _modality, _video_id, anchor, specialist in reservations:
            if anchor is None or candidate_key(anchor) != candidate_key(specialist):
                key = candidate_key(specialist)
                if key not in distinct_reservation_keys:
                    distinct_reservation_keys.append(key)
        anchor_budget = max(0, max_vlm_candidates - len(distinct_reservation_keys))
        anchor_videos = eligible[:min(anchor_budget, len(eligible))]

        # A chosen specialist may fall outside the final anchor prefix after
        # reserving its slot. Recompute against the actual competitive prefix
        # rather than silently borrowing a lower-ranked video slot.
        final_reservations = []
        reserved_distinct_by_video = {}
        for modality in specialist_modalities:
            for video_id in anchor_videos:
                anchor = visual_anchors.get(video_id)
                specialist = best(video_id, modality, payload_only=True)
                if specialist is not None:
                    distinct = anchor is None or candidate_key(anchor) != candidate_key(specialist)
                    if distinct and reserved_distinct_by_video.get(video_id, 0) >= per_video_cap - 1:
                        continue
                    final_reservations.append((modality, video_id, anchor, specialist))
                    if distinct:
                        reserved_distinct_by_video[video_id] = (
                            reserved_distinct_by_video.get(video_id, 0) + 1
                        )
                    break
        reservations = final_reservations
        distinct_reservation_keys = []
        for _modality, _video_id, anchor, specialist in reservations:
            if anchor is None or candidate_key(anchor) != candidate_key(specialist):
                key = candidate_key(specialist)
                if key not in distinct_reservation_keys:
                    distinct_reservation_keys.append(key)
        # A second recomputation is needed only when two modalities reserve
        # different frames. It keeps the budget exact even for dual-route
        # temporal questions without making a hidden policy decision.
        anchor_budget = max(0, max_vlm_candidates - len(distinct_reservation_keys))
        anchor_videos = eligible[:min(anchor_budget, len(eligible))]

        selected = []
        selected_reasons = {}
        counts = {}
        seen = set()

        def take(item, reason):
            if item is None or len(selected) >= max_vlm_candidates:
                return False
            key = candidate_key(item)
            video_id = key[0]
            if key in seen or counts.get(video_id, 0) >= per_video_cap:
                return False
            selected.append(item)
            seen.add(key)
            counts[video_id] = counts.get(video_id, 0) + 1
            selected_reasons[key] = reason
            return True

        # Pass 1: one visual/base anchor per competitive video. A specialist
        # cannot replace this anchor, even if it has a larger raw score.
        for video_id in anchor_videos:
            take(visual_anchors.get(video_id), "visual_anchor")

        # Pass 2: grounded ASR/OCR evidence. When a specialist is the same
        # canonical frame as an anchor, its provenance is already present and
        # no duplicate budget is spent.
        for modality, video_id, _anchor, specialist in reservations:
            reason = f"{modality}_evidence_reservation"
            if candidate_key(specialist) in seen:
                selected_reasons[candidate_key(specialist)] = (
                    selected_reasons.get(candidate_key(specialist), "visual_anchor")
                    + f"+{reason}"
                )
            else:
                take(specialist, reason)

        # Pass 3: cover additional ranked videos before depth. This only runs
        # if a reservation/empty video left budget after the anchor prefix.
        for video_id in eligible:
            if len(selected) >= max_vlm_candidates:
                break
            if video_id in counts:
                continue
            take(visual_anchors.get(video_id), "late_visual_anchor")

        # Pass 4: deterministic round-robin depth. Source-specific rows are
        # eligible only after every competitive anchor above had first claim.
        while len(selected) < max_vlm_candidates:
            added = False
            for video_id in eligible:
                if len(selected) >= max_vlm_candidates:
                    break
                if counts.get(video_id, 0) >= per_video_cap:
                    continue
                options = [item for item in grouped[video_id]
                           if candidate_key(item) not in seen]
                if not options:
                    continue
                # Prefer visual for generic depth, then a grounded requested
                # specialist, then any materialized canonical row.
                visual = [item for item in options if has_source(item, "visual")]
                grounded = [
                    item for item in options
                    if any(self._candidate_has_modality_payload(item, modality)
                           for modality in specialist_modalities)
                ]
                choice = min(
                    visual or grounded or options,
                    key=lambda item: self._candidate_channel_key(
                        item, "visual" if has_source(item, "visual") else
                        next((modality for modality in specialist_modalities
                              if has_source(item, modality)), "visual")
                    ),
                )
                added = take(choice, "diversity_depth") or added
            if not added:
                break

        selected_keys = {candidate_key(item) for item in selected}
        trace = list(canonical_rejections)
        for entry in trace:
            entry["video_rank"] = rank_by_video.get(str(entry.get("video_id", "")))
        for item in pool:
            key = candidate_key(item)
            video_id, kf_n = key
            source_names = list(candidate_sources(item))
            source_score = {
                source: self._candidate_source_score(item, source)
                for source in source_names
            }
            if key in selected_keys:
                reason = selected_reasons.get(key, "selected")
                selected_flag = True
            elif video_id not in rank_by_video:
                reason = "video_not_in_ranked_candidates"
                selected_flag = False
            elif counts.get(video_id, 0) >= per_video_cap:
                reason = "per_video_cap_reached"
                selected_flag = False
            elif any(source in specialist_modalities for source in source_names) and not any(
                self._candidate_has_modality_payload(item, source)
                for source in specialist_modalities if has_source(item, source)
            ):
                reason = "specialist_payload_missing"
                selected_flag = False
            else:
                reason = "budget_or_rank_not_selected"
                selected_flag = False
            trace.append({
                "selected": selected_flag,
                "reason": reason,
                "video_id": video_id,
                "kf_n": kf_n,
                "frame_idx": int(item["frame_idx"]),
                "video_rank": rank_by_video.get(video_id),
                "source": source_names,
                "source_score": source_score,
                "canonical_frame_remapped": bool(item.get("_canonical_frame_remapped")),
            })

        selected_video_count = len({str(item["video_id"]) for item in selected})
        reservation_status = []
        for modality in specialist_modalities:
            matching = [
                item for item in selected
                if has_source(item, modality)
                and self._candidate_has_modality_payload(item, modality)
            ]
            reservation_status.append({
                "modality": modality,
                "fulfilled": bool(matching),
                "selected_count": len(matching),
            })
        diagnostics = {
            "allocator": "qna_anchor_preserving_v1",
            "policy": "anchor_preserving",
            "budget": max_vlm_candidates,
            "ranked_video_count": len(ranked),
            "eligible_video_count": len(eligible),
            "selected_video_count": selected_video_count,
            "canonical_map_checked": canonical_checked,
            "canonical_rejection_count": len(canonical_rejections),
            "anchor_budget": anchor_budget,
            "anchor_video_count": len(anchor_videos),
            "specialist_modalities": list(specialist_modalities),
            "specialist_reservations": [
                {"modality": modality, "video_id": video_id,
                 "same_as_anchor": bool(anchor is not None and candidate_key(anchor) == candidate_key(specialist))}
                for modality, video_id, anchor, specialist in reservations
            ],
            "specialist_reservation_status": reservation_status,
            "selection_trace": trace,
        }
        impossible_reason = None
        if max_vlm_candidates < len(eligible):
            impossible_reason = (
                "budget_too_small_for_full_video_coverage: "
                f"max_vlm_candidates={max_vlm_candidates} < eligible_video_count={len(eligible)}"
            )
        return AllocationResult(
            selected=tuple(dict(item) for item in selected),
            diagnostics=diagnostics,
            impossible_budget_reason=impossible_reason,
        )

    @staticmethod
    def _allocate_routed_candidates(candidates, video_ids, max_vlm_candidates,
                                    specialist_modalities=None):
        """Compatibility wrapper around the single Q&A selector owner.

        Historically this method had a second allocator implementation.  It
        differed from :func:`allocate_recall_preserving_candidates` in source
        scoring, specialist reservation, and duplicate handling, which made
        tests/callers observe a different selector from the production
        ``prepare_ranked_candidates`` path.  Keep the old list-returning API,
        but delegate all policy decisions to the shared allocator.
        """
        if max_vlm_candidates <= 0 or not candidates or not video_ids:
            return []
        result = allocate_recall_preserving_candidates(
            candidates,
            video_ids,
            max_vlm_candidates=max_vlm_candidates,
            specialist_modalities=specialist_modalities or (),
            # Preserve the historical compatibility helper's specialist
            # coverage target while retaining the hard visual-first invariant.
            specialist_reservation=3,
            # This compatibility API historically treated temporal rows as
            # generic depth.  The production path calls the shared allocator
            # directly and owns the explicit temporal reservation there.
            temporal_reservation=0,
            per_video_cap=2,
        )
        return list(result.selected)

    @staticmethod
    def _allocate_visual_candidates(candidates, video_ids, max_vlm_candidates,
                                     policy="balanced"):
        """Select visual frames under a bounded VLM budget.

         The legacy visual selector consumed all five frames of the first
         video before considering the next one.  That is harmful for VQA:
         frame recall can be present in the materialized 100-frame lattice but
         disappear from the 12-frame answer budget.  ``balanced`` is the
         production default; named alternatives remain explicit dev-only A/B
         policies and must not be selected by a hidden caller default.
        """
        if max_vlm_candidates <= 0 or not candidates or not video_ids:
            return []
        if max_vlm_candidates >= len(candidates):
            return list(candidates)
        ranked_video_ids = list(dict.fromkeys(str(video_id) for video_id in video_ids))
        grouped = {video_id: [] for video_id in ranked_video_ids}
        for item in candidates:
            video_id = str(item.get("video_id"))
            if video_id in grouped:
                grouped[video_id].append(item)
        for values in grouped.values():
            values.sort(key=lambda item: (
                -float(item.get("base_score", 0.0)),
                int(item.get("kf_n", 0)),
            ))

        if policy == "legacy":
            return list(candidates[:max_vlm_candidates])
        if policy.startswith("global_"):
            try:
                rank_penalty = float(policy.split("_", 1)[1])
            except ValueError as exc:
                raise ValueError(f"invalid visual selector policy: {policy}") from exc
            if rank_penalty < 0:
                raise ValueError(f"invalid visual selector policy: {policy}")
            ordered = sorted(
                candidates,
                key=lambda item: (
                    -(float(item.get("base_score", 0.0)) - rank_penalty * math.log1p(int(item.get("video_rank", 0)))),
                    int(item.get("video_rank", 0)),
                    int(item.get("kf_n", 0)),
                ),
            )
            selected = []
            counts = {}
            seen = set()
            for item in ordered:
                video_id = str(item.get("video_id"))
                key = (video_id, int(item.get("kf_n", -1)))
                if key in seen or counts.get(video_id, 0) >= 2:
                    continue
                seen.add(key)
                selected.append(item)
                counts[video_id] = counts.get(video_id, 0) + 1
                if len(selected) >= max_vlm_candidates:
                    break
            return selected
        if policy == "balanced":
            passes = [ranked_video_ids, ranked_video_ids]
        elif policy.startswith("hybrid_"):
            try:
                coverage = max(1, int(policy.split("_", 1)[1]))
            except ValueError as exc:
                raise ValueError(f"invalid visual selector policy: {policy}") from exc
            coverage = min(coverage, len(ranked_video_ids))
            passes = [ranked_video_ids[:coverage], ranked_video_ids[:max(0, max_vlm_candidates - coverage)], ranked_video_ids[coverage:]]
        elif policy.startswith("pattern_"):
            try:
                counts = [max(0, int(value)) for value in policy.split("_")[1:]]
            except ValueError as exc:
                raise ValueError(f"invalid visual selector policy: {policy}") from exc
            if not counts or not any(counts):
                raise ValueError(f"invalid visual selector policy: {policy}")
            passes = []
            for video_id, count in zip(ranked_video_ids, counts):
                passes.extend([[video_id]] * count)
            # If the explicit pattern does not consume the budget, fill the
            # remainder with later-ranked videos before adding more depth.
            passes.append(ranked_video_ids[len(counts):])
            passes.append(ranked_video_ids)
        else:
            raise ValueError(f"unknown visual selector policy: {policy}")

        selected = []
        seen = set()
        for pass_videos in passes:
            for video_id in pass_videos:
                values = grouped.get(video_id, [])
                depth = sum(1 for item in selected if str(item.get("video_id")) == video_id)
                if depth >= len(values):
                    continue
                item = values[depth]
                key = (video_id, int(item.get("kf_n", -1)))
                if key in seen:
                    continue
                seen.add(key)
                selected.append(item)
                if len(selected) >= max_vlm_candidates:
                    return selected
        return selected

    def prepare_ranked_candidates(self, query: str, question: str, *, top_videos: int = 20,
                                  frames_per_video: int = 5, max_vlm_candidates: int = 12,
                                  temporal_consensus: bool = True,
                                  required_modalities=None, modality_router=None,
                                  modality_budget: int = 2, global_modality_router=None,
                                  rrf_weights: dict | None = None,
                                  question_type: str | None = None,
                                  visual_selector_policy: str = "adaptive",
                                  evidence_fusion: bool | None = None,
                                  return_candidate_pool: bool = False) -> dict:
        """Materialize bounded visual candidates without loading/using the VLM.

        Keeping this stage separate lets an offline benchmark retrieve the full
        query set with the fusion encoders, release their VRAM, and only then
        load Qwen for answering. This avoids a silent CUDA memory contention
        between the retrieval and answering models.
        """
        if not query or not question:
            raise ValueError("query and question must be non-empty")
        if not 1 <= top_videos <= 100:
            raise ValueError("top_videos must be between 1 and 100")
        if not 1 <= frames_per_video <= 20:
            raise ValueError("frames_per_video must be between 1 and 20")
        if not 1 <= max_vlm_candidates <= 100:
            raise ValueError("max_vlm_candidates must be between 1 and 100")
        visual_selector_policy = str(visual_selector_policy).strip().lower()
        if visual_selector_policy not in SELECTOR_POLICIES:
            raise ValueError(
                "visual_selector_policy must be 'anchor_preserving', 'legacy', "
                "'balanced', or 'adaptive'"
            )
        modalities = self._parse_modalities(required_modalities)
        route_requested = global_modality_router is not None and bool(modalities)
        visual_retrieved = self.kis.search(
            query, topk=max(top_videos, 100) if route_requested else top_videos
        )
        specialist_channels = {}
        if route_requested:
            specialist_text = f"{query}\n{question}".strip()
            for modality in modalities:
                specialist_channels[modality] = global_modality_router.global_candidates(
                    specialist_text, modality, topk=100)
        specialist_hits = {
            modality: bool(specialist_channels.get(modality)) for modality in modalities
        }
        inferred_type = question_type
        if not inferred_type:
            if modalities == ["asr"]:
                inferred_type = "spoken_fact"
            elif modalities == ["ocr"]:
                inferred_type = "screen_text"
            elif len(modalities) > 1:
                inferred_type = "temporal_relation"
            else:
                inferred_type = "unknown"
        if canonical_question_type(inferred_type) == "visual" and modalities:
            inferred_type = (
                "spoken_fact" if modalities == ["asr"] else
                "screen_text" if modalities == ["ocr"] else
                "temporal_relation"
            )
        primary_specialist = {
            "spoken_fact": "asr",
            "screen_text": "ocr",
        }.get(canonical_question_type(inferred_type))
        route_active = route_requested and (
            bool(specialist_hits.get(primary_specialist))
            if primary_specialist else any(specialist_hits.values())
        )
        if route_active and visual_selector_policy == "legacy":
            raise ValueError(
                "legacy selector policy is unsupported for routed retrieval; "
                "choose balanced or adaptive explicitly"
            )
        # Routed mode enables evidence packets by default.  Passing False
        # preserves the historical visual-only answer prompt and ranking.
        evidence_fusion_active = route_active if evidence_fusion is None else bool(evidence_fusion)
        if route_active:
            channels = {"visual": visual_retrieved, **specialist_channels}
            # Resolve the task-aware policy once at the retrieval boundary.
            # Explicit modality requirements remain authoritative when a
            # caller supplies no/contradictory annotation type.
            base_config = RoutingConfig.baseline(enabled=True)
            weights_by_type = {
                key: dict(value) for key, value in base_config.weights_by_type.items()
            }
            policy_type = canonical_question_type(inferred_type)
            tuned_weights = dict(DEFAULT_RRF_WEIGHTS)
            tuned_weights.update(rrf_weights or {})
            for modality in modalities:
                if modality in tuned_weights:
                    weights_by_type[policy_type][modality] = float(tuned_weights[modality])
            policy_config = RoutingConfig(
                enabled=True,
                weights_by_type=weights_by_type,
                rescue_by_type=dict(base_config.rescue_by_type),
                rrf_k=base_config.rrf_k,
                retrieval_top_k=base_config.retrieval_top_k,
                output_top_k=top_videos,
            )
            routing_plan = build_routing_plan(policy_type, policy_config)
            routing_plan_record = {
                "question_type": routing_plan.question_type,
                "routing_enabled": routing_plan.routing_enabled,
                "channels": list(routing_plan.channels),
                "primary_channel": routing_plan.primary_channel,
                "specialist_channels": list(routing_plan.specialist_channels),
                "weights": dict(routing_plan.weights),
                "required_channels": list(routing_plan.required_channels),
                "rescue_gate": {
                    "enabled": routing_plan.rescue_gate.enabled,
                    "strong_rank": routing_plan.rescue_gate.strong_rank,
                    "support_rank": routing_plan.rescue_gate.support_rank,
                    "min_specialist_channels": routing_plan.rescue_gate.min_specialist_channels,
                    "allow_single_strong_rescue": routing_plan.rescue_gate.allow_single_strong_rescue,
                    "require_evidence": routing_plan.rescue_gate.require_evidence,
                },
            }
            # Wave 2's weighted_video_rrf is the sole fusion owner.  Keep the
            # policy object responsible for resolving weights/gates, but call
            # the owner directly here so this production path cannot drift to
            # a second video-level fusion implementation.
            gate = routing_plan.rescue_gate
            fused_videos = weighted_video_rrf(
                channels,
                dict(routing_plan.weights),
                rrf_k=routing_plan.rrf_k,
                topk=routing_plan.output_top_k,
                visual_channel="visual",
                specialist_strong_rank=gate.strong_rank,
                specialist_support_rank=gate.support_rank,
                min_specialist_channels=gate.min_specialist_channels,
                allow_single_strong_rescue=gate.allow_single_strong_rescue,
                specialist_min_scores=gate.min_scores,
                require_specialist_evidence=gate.require_evidence and gate.enabled,
                evidence_keys=gate.evidence_keys,
                specialist_rescue_enabled=gate.enabled,
            )
            video_ids = [str(row["video_id"]) for row in fused_videos]
        else:
            fused_videos = []
            video_ids = [str(row[0]) for row in visual_retrieved]
            routing_plan = None
            routing_plan_record = None
        if not video_ids:
            return {
                "query": query, "question": question, "candidates": [],
                "retrieved_video_ids": [], "visual_retrieved_video_ids": [],
                "status": "no_retrieval", "candidate_count": 0,
                "vlm_candidate_count": 0, "modality_route": modalities,
                "route_requested": route_requested, "route_active": route_active,
                "candidate_state": "candidate_miss", "candidate_miss": True,
                "wrong_video_state": "not_evaluated", "wrong_video": None,
                "route_state": (
                    "specialist_success" if route_requested and all(specialist_hits.values()) else
                    ("specialist_partial" if route_requested and route_active else
                     ("specialist_no_hit" if route_requested else "baseline_success"))
                ),
                "route_fallback_reason": (
                    None if route_requested and all(specialist_hits.values()) else
                    ("specialist_partial_hit" if route_requested and route_active else
                     ("specialist_returned_no_hit" if route_requested else "visual_retrieval_empty"))
                ),
                "evidence_fusion": evidence_fusion_active,
                "rrf_videos": fused_videos,
                "routing_plan": routing_plan_record,
                "candidate_source_counts": {}, "specialist_candidate_count": 0,
                "selector_metrics": selector_metrics([], [], video_ids),
                "stages": {
                    "retrieval": {"visual_count": len(visual_retrieved),
                                   "specialist_counts": {
                                       modality: len(specialist_channels.get(modality, []))
                                       for modality in modalities}},
                    "candidate_pool": {"count": 0},
                    "selection": selector_metrics([], [], video_ids),
                    "answer": {"pending": True}, "rerank": {"pending": True},
                },
            }
        video_rank = {video_id: rank for rank, video_id in enumerate(video_ids)}
        fused_by_video = {str(row["video_id"]): row for row in fused_videos}
        visual_rank = {
            str(row[0]): rank for rank, row in enumerate(visual_retrieved[:100], 1)
        }
        if return_candidate_pool and getattr(self.kis, "materialized", False):
            # The retrieval benchmark must expose the exact frozen lattice,
            # not the answer-stage depth cap.  The selector below remains
            # responsible for the bounded ``candidates`` field.
            lattice = self._materialized_candidates_all(video_ids)
        else:
            visual_budget = min(frames_per_video, 3) if route_active else frames_per_video
            lattice = self._local_candidates(
                query, video_ids, visual_budget,
                temporal_consensus=temporal_consensus,
                diversify=route_active,
            )
        candidates = []
        for lattice_rank, (video_id, frame_idx, kf_n, base_score) in enumerate(lattice, 1):
            rows = self.km[(self.km.video_id.astype(str) == str(video_id)) &
                           (self.km.kf_n == int(kf_n))]
            if rows.empty:
                continue
            row = rows.iloc[0]
            frame_path = self._frame_path(video_id, kf_n)
            if not os.path.exists(frame_path):
                continue
            visual_candidate = {
                # The materialized retriever may carry stale frame_idx data;
                # map-keyframes remains the only submission authority.
                "video_id": str(video_id), "frame_idx": int(row.frame_idx),
                "kf_n": int(kf_n), "pts_time": float(row.pts_time),
                "base_score": float(base_score), "video_rank": video_rank[str(video_id)],
                "frame_path": frame_path, "source": "visual",
                "retrieval_rank": visual_rank.get(str(video_id)),
                "visual_video_rank": visual_rank.get(str(video_id)),
                "visual_frame_rank": lattice_rank,
            }
            if int(frame_idx) != int(row.frame_idx):
                visual_candidate["_canonical_frame_remapped"] = True
                visual_candidate["_input_frame_idx"] = int(frame_idx)
            candidates.append(visual_candidate)

        if route_active:
            # Add global specialist frames and temporal neighbors. Specialist
            # raw scores are only used within a video; video order comes from RRF.
            for modality in modalities:
                per_video = {}
                for extra in specialist_channels.get(modality, []):
                    if extra["video_id"] in video_rank and len(per_video.get(extra["video_id"], [])) < 3:
                        per_video.setdefault(extra["video_id"], []).append(extra)
                for video_id, extras in per_video.items():
                    for extra in extras:
                        rows = self.km[(self.km.video_id.astype(str) == extra["video_id"]) &
                                       (self.km.kf_n == int(extra["kf_n"]))]
                        if rows.empty:
                            continue
                        row = rows.iloc[0]
                        frame_path = self._frame_path(extra["video_id"], extra["kf_n"])
                        if not os.path.exists(frame_path):
                            continue
                        candidates.append({
                            "video_id": extra["video_id"], "frame_idx": int(row.frame_idx),
                            "kf_n": int(extra["kf_n"]), "pts_time": float(row.pts_time),
                            "base_score": float(extra["modality_score"]),
                            "video_rank": video_rank[extra["video_id"]],
                            "frame_path": frame_path, "source": modality,
                            "retrieval_rank": extra.get("rank"),
                            "modality_score": float(extra["modality_score"]),
                            "score_mode": extra.get("score_mode"),
                            "text": extra.get("text", ""),
                            "evidence": extra.get("evidence"),
                        })
                    if extras:
                        anchor = extras[0]
                        for neighbor in self._temporal_neighbor_candidates(video_id, anchor["kf_n"]):
                            frame_path = self._frame_path(neighbor["video_id"], neighbor["kf_n"])
                            if not os.path.exists(frame_path):
                                continue
                            neighbor.update({"video_rank": video_rank[neighbor["video_id"]],
                                             "frame_path": frame_path})
                            candidates.append(neighbor)
        elif modality_router is not None and modalities:
            # Preserve the old diagnostic mode: text candidates are restricted
            # to the frozen visual video list and never alter video ranking.
            for modality in modalities:
                for extra in modality_router.candidate_frames(
                        question or query, modality, video_ids, per_video=modality_budget):
                    rows = self.km[(self.km.video_id.astype(str) == extra["video_id"]) &
                                   (self.km.kf_n == int(extra["kf_n"]))]
                    if rows.empty:
                        continue
                    row = rows.iloc[0]
                    frame_path = self._frame_path(extra["video_id"], extra["kf_n"])
                    if not os.path.exists(frame_path):
                        continue
                    candidates.append({
                        "video_id": extra["video_id"], "frame_idx": int(row.frame_idx),
                        "kf_n": int(extra["kf_n"]), "pts_time": float(row.pts_time),
                        "base_score": float(extra["modality_score"]),
                        "video_rank": video_rank[extra["video_id"]],
                        "frame_path": frame_path, "source": modality,
                        "retrieval_rank": extra.get("rank"),
                        "modality_score": float(extra["modality_score"]),
                        "score_mode": extra.get("score_mode"),
                        "text": extra.get("text", ""),
                        "evidence": extra.get("evidence"),
                    })

        # Dedupe only after all channels have contributed. A same-frame hit
        # from visual and ASR/OCR must retain both provenance records; an
        # early ``seen`` check would silently turn multimodal evidence into a
        # visual-only candidate.
        candidate_pool = deduplicate_candidates(candidates)
        for item in candidate_pool:
            item["retrieval_stage"] = "candidate_pool"
            fused = fused_by_video.get(str(item["video_id"]))
            if fused is not None:
                item["rrf_score"] = float(fused.get("rrf_score", 0.0))
                item["video_fusion"] = {
                    "video_id": str(fused["video_id"]),
                    "video_rank": int(fused.get("video_rank", item["video_rank"])),
                    "rrf_score": float(fused.get("rrf_score", 0.0)),
                    "rrf_guard": str(fused.get("rrf_guard", "none")),
                    "channel_ranks": {
                        key[:-5]: int(value)
                        for key, value in sorted(fused.items())
                        if key.endswith("_rank") and key[:-5] in {"visual", "asr", "ocr"}
                    },
                }

        # Production-balanced/routed mode uses the shared recall-preserving
        # allocator. Explicit legacy/A-B policies remain diagnostic so they
        # cannot silently replace the production selector.
        allocation_result = None
        if return_candidate_pool and route_active and modalities:
            # The measured retrieval pool must be non-regressing against the
            # frozen visual baseline.  The previous specialist-first sort let
            # ASR/OCR rows displace visual frames from the first 100 rows even
            # when video R@K was unchanged, causing frame recall to fall.
            # Keep the visual lattice prefix intact; same-frame specialist
            # provenance is already merged, while specialist-only rows remain
            # available after the measured visual prefix for production use.
            def _routed_pool_key(item):
                try:
                    visual_rank_value = int(item.get("visual_frame_rank", 10**9))
                except (TypeError, ValueError):
                    visual_rank_value = 10**9
                has_visual = has_source(item, "visual")
                return (
                    0 if has_visual else 1,
                    visual_rank_value,
                    int(item.get("video_rank", 10**9)),
                    int(item.get("retrieval_rank", 10**9) or 10**9),
                    int(item.get("kf_n", 10**9)),
                )

            candidate_pool = sorted(candidate_pool, key=_routed_pool_key)
        if visual_selector_policy == "anchor_preserving":
            allocation_result = self._allocate_anchor_preserving_candidates(
                candidate_pool,
                video_ids,
                max_vlm_candidates,
                specialist_modalities=modalities if route_active else (),
                per_video_cap=2,
            )
            candidates = list(allocation_result.selected)
        elif route_active:
            allocation_result = allocate_recall_preserving_candidates(
                candidate_pool,
                video_ids,
                max_vlm_candidates=max_vlm_candidates,
                specialist_modalities=modalities,
                specialist_reservation=1,
                temporal_reservation=1,
                per_video_cap=2,
                selection_policy=(
                    "adaptive" if visual_selector_policy == "adaptive" else "coverage"
                ),
            )
            candidates = list(allocation_result.selected)
        else:
            candidate_pool.sort(key=lambda item: (
                item["video_rank"], 0 if item.get("source") == "visual" else 1,
                -item["base_score"], int(item.get("kf_n", 0))))
            if visual_selector_policy in {"balanced", "adaptive"}:
                allocation_result = allocate_recall_preserving_candidates(
                    candidate_pool,
                    video_ids,
                    max_vlm_candidates=max_vlm_candidates,
                    specialist_modalities=(),
                    specialist_reservation=0,
                    temporal_reservation=0,
                    per_video_cap=2,
                    selection_policy=(
                        "adaptive" if visual_selector_policy == "adaptive" else "coverage"
                    ),
                )
                candidates = list(allocation_result.selected)
            else:
                candidates = self._allocate_visual_candidates(
                    candidate_pool, video_ids, max_vlm_candidates,
                    policy=visual_selector_policy,
                )
        candidates = [{**item, "selection_stage": "selector"} for item in candidates]
        if evidence_fusion_active:
            candidates = self._attach_evidence_frames(
                candidates, candidate_pool, limit=3,
            )
        candidate_miss = not bool(candidates)
        selector_report = selector_metrics(candidate_pool, candidates, video_ids)
        if allocation_result is not None:
            selector_report = {
                **selector_report,
                "allocator": dict(allocation_result.diagnostics),
                "impossible_budget_reason": allocation_result.impossible_budget_reason,
            }
        return {
            "query": query, "question": question, "candidates": candidates,
            "question_type": canonical_question_type(inferred_type),
            "retrieved_video_ids": video_ids,
            "status": "candidates_ready" if candidates else "no_valid_candidates",
            "candidate_state": "candidate_miss" if candidate_miss else "candidate_available",
            "candidate_miss": candidate_miss,
            "wrong_video_state": "not_evaluated",
            "wrong_video": None,
            "candidate_count": len(lattice), "candidate_pool_count": len(candidate_pool),
            "vlm_candidate_count": len(candidates),
            "modality_route": modalities, "route_requested": route_requested,
            "route_active": route_active,
            "route_state": (
                "specialist_success" if route_requested and all(specialist_hits.values()) else
                ("specialist_partial" if route_requested and route_active else
                 ("specialist_no_hit" if route_requested else "baseline_success"))
            ),
            "route_fallback_reason": (
                None if route_requested and all(specialist_hits.values()) or not route_requested else
                ("specialist_partial_hit" if route_active else "specialist_returned_no_hit")
            ),
            # Private, in-process seam for the answer stage. It is deliberately
            # not copied into the public result/trace fields because the router
            # object is not JSON serializable.
            "_evidence_provider": (
                global_modality_router
                if global_modality_router is not None and callable(
                    getattr(global_modality_router, "evidence_packet_for_candidate", None)
                ) else None
            ),
            "evidence_fusion": evidence_fusion_active,
            "visual_retrieved_video_ids": [str(row[0]) for row in visual_retrieved[:100]],
            "rrf_videos": fused_videos,
            "routing_plan": routing_plan_record,
            "candidate_source_counts": {
                source: sum(1 for item in candidates if has_source(item, source))
                for source in sorted({source for item in candidates
                                      for source in candidate_sources(item)})
            },
            "specialist_candidate_count": sum(
                1 for item in candidates if any(has_source(item, source)
                                                for source in {"asr", "ocr"})
            ),
            "selector_metrics": selector_report,
            "selector_trace": (
                list(allocation_result.diagnostics.get("selection_trace", ()))
                if allocation_result is not None else []
            ),
            **({"_candidate_pool": [dict(item) for item in candidate_pool]}
               if return_candidate_pool else {}),
            "stages": {
                "retrieval": {"visual_count": len(visual_retrieved),
                               "specialist_counts": {
                                   modality: len(specialist_channels.get(modality, []))
                                   for modality in modalities}},
                "candidate_pool": {"count": len(candidate_pool)},
                "selection": selector_report,
                "answer": {"pending": True},
                "rerank": {"pending": True},
            },
        }

    def release_retrieval_models(self) -> None:
        """Release GPU-heavy retrieval encoders before local VLM inference."""
        import gc
        for name in ("m_vitl", "m_so"):
            if hasattr(self.kis, name):
                setattr(self.kis, name, None)
        for name in ("tk_vitl", "tk_so"):
            if hasattr(self.kis, name):
                setattr(self.kis, name, None)
        gc.collect()
        torch_module = getattr(self.kis, "torch", None)
        if torch_module is not None and torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()

    @staticmethod
    def _candidate_has_modality_payload(candidate, modality: str) -> bool:
        """Return whether a candidate carries usable specialist evidence."""
        modality = str(modality).strip().lower()
        if not has_source(candidate, modality):
            return False
        for key in ("text", "evidence", "chunk", "ocr_text", "transcript"):
            value = candidate.get(key)
            if value is not None and str(value).strip():
                return True
        for record in candidate.get("provenance", ()) or ():
            if not isinstance(record, dict):
                continue
            if str(record.get("source", "")).strip().lower() != modality:
                continue
            for key in ("text", "evidence", "chunk", "ocr_text", "transcript"):
                value = record.get(key)
                if value is not None and str(value).strip():
                    return True
        return False

    def select_prepared_candidates(
        self,
        prepared: dict,
        *,
        top_videos: int = 20,
        max_vlm_candidates: int = 12,
        required_modalities=None,
        visual_selector_policy: str = "adaptive",
    ) -> dict:
        """Select the answer pool from one already-materialized retrieval pool.

        This is the seam between retrieval and answering.  It deliberately
        does not call a retriever again: a second top-20 route can produce
        frames absent from the measured top-100 pool and silently invalidate
        frame-recall/answer metrics. ``anchor_preserving`` is an explicit A/B
        policy that keeps a visual answer frame and allocates a separate,
        payload-backed ASR/OCR candidate when that route needs it. The
        measured ``adaptive`` policy remains the default until promotion.
        """
        if not isinstance(prepared, dict):
            raise TypeError("prepared must be a mapping")
        if not 1 <= int(top_videos) <= 100:
            raise ValueError("top_videos must be between 1 and 100")
        if not 1 <= int(max_vlm_candidates) <= 100:
            raise ValueError("max_vlm_candidates must be between 1 and 100")
        pool = prepared.get("_candidate_pool")
        if not isinstance(pool, list):
            pool = prepared.get("candidates", [])
        pool = [dict(item) for item in pool if isinstance(item, dict)]
        ranked_video_ids = [
            str(video_id) for video_id in prepared.get("retrieved_video_ids", [])
        ][:int(top_videos)]
        modalities = self._parse_modalities(required_modalities)
        local_ocr_fallback_enabled = (
            modalities == ["ocr"] and self._local_vlm is not None
        )
        policy = str(visual_selector_policy).strip().lower()
        if policy not in {"anchor_preserving", "balanced", "adaptive"}:
            raise ValueError(
                "visual_selector_policy must be 'anchor_preserving', 'balanced', or 'adaptive'"
            )
        eligible_pool = pool
        if policy != "anchor_preserving" and modalities and not local_ocr_fallback_enabled:
            eligible_pool = [
                item for item in pool
                if all(self._candidate_has_modality_payload(item, modality)
                       for modality in modalities)
            ]
        if policy == "anchor_preserving":
            allocation = self._allocate_anchor_preserving_candidates(
                eligible_pool,
                ranked_video_ids,
                int(max_vlm_candidates),
                specialist_modalities=modalities,
                per_video_cap=2,
            )
        else:
            allocation = allocate_recall_preserving_candidates(
                eligible_pool,
                ranked_video_ids,
                max_vlm_candidates=int(max_vlm_candidates),
                specialist_modalities=modalities,
                specialist_reservation=1 if modalities else 0,
                temporal_reservation=1 if not modalities else 0,
                per_video_cap=2,
                selection_policy=policy,
            )
        selected = [dict(item, selection_stage="selector") for item in allocation.selected]
        if local_ocr_fallback_enabled:
            # Prefer real OCR rows, but keep one visual candidate as an explicit
            # local-Qwen OCR fallback.  Global OCR coverage can be on the right
            # video yet miss the accepted frame (or contain unrelated text); a
            # local visual reader is the only offline way to inspect that frame.
            specialist_selected = [
                item for item in selected
                if self._candidate_has_modality_payload(item, "ocr")
            ]
            visual_fallbacks = [
                dict(item, _local_ocr_fallback=True)
                for item in pool
                if has_source(item, "visual")
                and not self._candidate_has_modality_payload(item, "ocr")
                and str(item.get("video_id")) in set(ranked_video_ids)
            ]
            if visual_fallbacks and not any(
                item.get("_local_ocr_fallback")
                or not self._candidate_has_modality_payload(item, "ocr")
                for item in selected
            ):
                fallback = visual_fallbacks[0]
                if selected:
                    selected[-1] = fallback
                else:
                    selected = [fallback]
            selected = [
                dict(item, _local_ocr_fallback=bool(
                    item.get("_local_ocr_fallback")
                    or not self._candidate_has_modality_payload(item, "ocr")
                ))
                for item in selected
            ]
        if prepared.get("evidence_fusion"):
            selected = self._attach_evidence_frames(selected, eligible_pool, limit=3)
        result = dict(prepared)
        result["candidates"] = selected
        result["retrieved_video_ids"] = ranked_video_ids
        result["candidate_count"] = len(eligible_pool)
        result["vlm_candidate_count"] = len(selected)
        result["candidate_miss"] = not bool(selected)
        result["candidate_state"] = "candidate_miss" if not selected else "candidate_available"
        result["selector_metrics"] = selector_metrics(eligible_pool, selected, ranked_video_ids)
        result["selector_trace"] = list(allocation.diagnostics.get("selection_trace", ()))
        result["answer_selection"] = {
            "source": "retrieval_candidate_pool",
            "retrieval_pool_count": len(pool),
            "eligible_pool_count": len(eligible_pool),
            "required_modalities": list(modalities),
            "local_ocr_fallback_enabled": local_ocr_fallback_enabled,
            "allocation": dict(allocation.diagnostics),
            "impossible_budget_reason": allocation.impossible_budget_reason,
        }
        return result

    def answer_ranked_candidates(self, prepared: dict, *, max_answers: int = 20,
                                max_new_tokens: int = 128, use_context: bool = True,
                                structured_vlm: bool | None = None,
                                rerank_weights: dict | None = None) -> dict:
        """Answer a prepared candidate set through the injected provider."""
        answer_provider = getattr(self, "answer_provider", None)
        if self._local_vlm is None and answer_provider is None:
            raise RuntimeError(
                "offline Q&A requires a local Qwen checkpoint or an injected AnswerProvider"
            )
        if not 1 <= max_answers <= 20:
            raise ValueError("Q&A production allows between 1 and 20 answers")
        answer_status = (
            "answered_remote"
            if bool(getattr(answer_provider, "is_remote", False))
            else "answered_local"
        )
        empty_status = (
            "no_valid_remote_answer"
            if answer_status == "answered_remote"
            else "no_valid_local_answer"
        )
        candidates = list(prepared.get("candidates", []))
        query = str(prepared.get("query", ""))
        question = str(prepared.get("question", ""))
        local_metadata_capable = any(
            callable(getattr(self._local_vlm, method, None))
            for method in ("answer_with_metadata", "answer_frames_with_metadata")
        )
        if structured_vlm is None:
            structured_vlm = bool(prepared.get(
                "route_active", False)
                or prepared.get("evidence_fusion", False)
                or local_metadata_capable
            )
        rank_weights = {
            "video_rank": 1.0,
            "visual_relevance": 0.05,
            "answer_consistency": 0.20,
            "grounding": 0.25,
            "answer_confidence": 0.10,
            "evidence_support": 0.15,
        }
        rank_weights.update(prepared.get("rerank_weights", {}))
        rank_weights.update(rerank_weights or {})
        answered = []
        answer_trace = []
        evidence_mode = bool(prepared.get(
            "evidence_fusion", prepared.get("route_active", False)))
        required_sources = tuple(
            str(source).strip().lower()
            for source in prepared.get("required_sources", ())
            if str(source).strip().lower() in {"asr", "ocr"}
        )
        question_type = canonical_question_type(prepared.get("question_type"))
        # For screen-text questions, pixels are the primary answer evidence;
        # the sampled OCR index is a retrieval specialist and may contain a
        # ticker while missing the larger sign/title visible in the same
        # frame.  Requiring lexical agreement with that sampled OCR text
        # rejects valid visual reads.  Spoken facts remain strictly ASR-
        # supported because speech is not recoverable from frame pixels.
        semantic_required_sources = (
            () if question_type == "screen_text" else required_sources
        )

        def trace_for(candidate, **extra):
            trace = stage_record("answer", candidate)
            trace.update({
                key: candidate.get(key)
                for key in ("video_id", "frame_idx", "kf_n", "pts_time",
                            "video_rank", "source", "base_score",
                            "selection_stage")
                if key in candidate
            })
            trace.update({
                "route_state": prepared.get("route_state", "unknown"),
                "candidate_state": prepared.get("candidate_state", "unknown"),
                "wrong_video_state": prepared.get("wrong_video_state", "not_evaluated"),
            })
            trace.update(extra)
            return trace

        for candidate in candidates:
            packet = None
            provider_contract_verification = None
            local_ocr_fallback = bool(
                candidate.get("_local_ocr_fallback")
                and required_sources == ("ocr",)
                and answer_provider is None
                and self._local_vlm is not None
            )
            # A routed/evidence-fusion answer is never allowed to fall back to
            # the legacy global context helpers.  The packet is bounded to the
            # candidate video and anchor time, even when the caller disables
            # the legacy ``use_context`` switch.
            if evidence_mode:
                evidence_provider = prepared.get(
                    "evidence_provider", prepared.get("_evidence_provider")
                )
                if evidence_provider is not None:
                    try:
                        packet = self._build_evidence_packet(
                            candidate,
                            query,
                            question,
                            evidence_provider=evidence_provider,
                            modalities=required_sources,
                        )
                    except (KeyError, ValueError, TypeError) as exc:
                        if not local_ocr_fallback:
                            # A global specialist hit does not guarantee that
                            # every selected frame has local evidence.  Reject
                            # it rather than silently using a visual answer.
                            answer_trace.append(trace_for(
                                candidate,
                                status="rejected_missing_modality_evidence",
                                error=str(exc)[:200],
                                required_sources=list(required_sources),
                            ))
                            continue
                        packet = {
                            "asr_chunks": (), "ocr_text": (),
                            "frames": ({
                                "video_id": candidate.get("video_id"),
                                "frame_idx": candidate.get("frame_idx"),
                                "kf_n": candidate.get("kf_n"),
                                "pts_time": candidate.get("pts_time"),
                                "frame_path": candidate.get("frame_path"),
                            },),
                            "sources": (), "local_ocr_fallback": True,
                        }
                else:
                    try:
                        packet = self._build_evidence_packet(candidate, query, question)
                    except (KeyError, ValueError, TypeError) as exc:
                        if not local_ocr_fallback:
                            answer_trace.append(trace_for(
                                candidate,
                                status="rejected_missing_modality_evidence",
                                error=str(exc)[:200],
                                required_sources=list(required_sources),
                            ))
                            continue
                        packet = {
                            "asr_chunks": (), "ocr_text": (),
                            "frames": ({
                                "video_id": candidate.get("video_id"),
                                "frame_idx": candidate.get("frame_idx"),
                                "kf_n": candidate.get("kf_n"),
                                "pts_time": candidate.get("pts_time"),
                                "frame_path": candidate.get("frame_path"),
                            },),
                            "sources": (), "local_ocr_fallback": True,
                        }
                if local_ocr_fallback:
                    # Nearby global OCR can be unrelated to the accepted frame.
                    # Do not feed that text to Qwen; explicitly let the local
                    # visual reader perform OCR on the bounded frame packet.
                    packet = {
                        **(packet or {}),
                        "asr_chunks": (), "ocr_text": (), "sources": (),
                        "local_ocr_fallback": True,
                    }
                # A packet can be syntactically valid while containing only
                # the visual anchor.  For a routed query that is not enough:
                # the required specialist modality must actually contribute a
                # bounded evidence row before invoking Qwen.  Otherwise the
                # verifier would reject it much later as a generic answer
                # failure, hiding the real candidate/evidence miss and
                # wasting VLM latency.
                if required_sources:
                    available_sources = set(packet.get("sources", ()))
                    missing_sources = [
                        source for source in required_sources
                        if source not in available_sources
                    ]
                    if missing_sources and not local_ocr_fallback:
                        answer_trace.append(trace_for(
                            candidate,
                            status="rejected_missing_modality_evidence",
                            error="required modality evidence absent from packet",
                            required_sources=list(required_sources),
                            missing_sources=missing_sources,
                        ))
                        continue
                asr = " ".join(item["text"] for item in packet["asr_chunks"])
                ocr = " | ".join(item["text"] for item in packet["ocr_text"])
            elif use_context:
                asr = self._asr_context(candidate["video_id"], candidate["pts_time"])
                ocr = self._ocr_context(candidate["video_id"], candidate["pts_time"])
            else:
                asr = ""
                ocr = ""
            prompt = self._build_answer_prompt(
                query, question, asr, ocr, evidence_packet=packet,
            )
            retry_used = False
            provider_name = None
            try:
                if answer_provider is not None:
                    from src.vqa.answer_provider import (
                        AnswerProviderRequest,
                        EvidenceBundle,
                        answer_with_verification,
                    )
                    provider_request = AnswerProviderRequest(
                        query=query,
                        question=question,
                        evidence=EvidenceBundle(
                            candidate_id=f"{candidate['video_id']}#{candidate['kf_n']}",
                            video_id=str(candidate["video_id"]),
                            frames=self._provider_frame_evidence(candidate, packet),
                            asr_text=asr,
                            ocr_text=ocr,
                        ),
                        max_new_tokens=max_new_tokens,
                    )
                    provider_response, contract_verification = answer_with_verification(
                        answer_provider,
                        provider_request,
                        required_sources=required_sources,
                    )
                    provider_contract_verification = contract_verification
                    provider_name = provider_response.provider
                    if provider_response.abstain or not provider_response.answer:
                        answer_trace.append(trace_for(
                            candidate,
                            status="rejected_provider_abstain",
                            reason=provider_response.reason,
                            provider=provider_name,
                        ))
                        continue
                    if not contract_verification.accepted:
                        answer_trace.append(trace_for(
                            candidate,
                            status="rejected_provider_verification",
                            reason=contract_verification.reason,
                            provider=provider_name,
                            verification=contract_verification.to_dict(),
                        ))
                        continue
                    record = provider_response.to_dict()
                    record["contract_verification"] = contract_verification.to_dict()
                    answer = provider_response.answer
                    structured_vlm = True
                elif structured_vlm:
                    frame_paths = self._local_evidence_paths(candidate, packet)
                    record = self._local_answer_with_evidence(
                        self._local_vlm,
                        frame_paths,
                        prompt,
                        max_new_tokens=max_new_tokens,
                    )
                    answer = record.get("answer", "")
                else:
                    record = {"grounding_score": 0.0, "answer_confidence": 0.0,
                              "abstain": False, "parse_failed": False}
                    frame_path = candidate.get("frame_path")
                    answer = self._local_vlm.answer(frame_path, prompt,
                                                    max_new_tokens=max_new_tokens)
            except Exception as exc:
                candidate["answer_error"] = str(exc)[:200]
                answer_trace.append(trace_for(
                    candidate, status="error", error=candidate["answer_error"]
                ))
                continue
            raw_answer = str(answer or "")
            cleaned_answer = self._extract_answer_text(raw_answer)
            if (self._looks_like_nonanswer(cleaned_answer) or bool(record.get("abstain", False))
                    or bool(record.get("parse_failed", False))):
                # A wrong-video candidate can trigger a verbose refusal even
                # when the frame still contains a short visible answer. Retry
                # once with a minimal visual-only prompt; never accept a
                # second refusal or an empty string.
                retry_prompt = (
                    "Give one short answer in the same language as the question, based only on what is visibly present "
                    f"in this frame. Question: {question}\nReturn only the answer text."
                )
                # For a spoken/OCR answer, a visual-only retry encourages the
                # VLM to hallucinate an answer that the frame cannot contain.
                if evidence_mode and packet and (packet["asr_chunks"] or packet["ocr_text"]):
                    retry = ""
                else:
                    try:
                        retry = self._local_vlm.answer(
                            candidate["frame_path"], retry_prompt,
                            max_new_tokens=max(16, max_new_tokens),
                        )
                    except Exception:
                        retry = ""
                retry_answer = self._extract_answer_text(retry)
                if self._looks_like_nonanswer(retry_answer):
                    answer_trace.append(trace_for(
                        candidate,
                        status="rejected_nonanswer",
                        raw_answer=raw_answer,
                        retry_answer=str(retry or ""),
                    ))
                    continue
                answer = retry
                raw_answer = str(answer or "")
                cleaned_answer = retry_answer
                retry_used = True
            normalized_answer = cleaned_answer
            verification = (
                provider_contract_verification.to_dict()
                if provider_contract_verification is not None else None
            )
            if semantic_required_sources and not local_ocr_fallback:
                from src.vqa.verifier import EvidenceVerifier
                verifier = getattr(self, "evidence_verifier", None) or EvidenceVerifier()
                evidence_mapping = dict(candidate)
                if packet is not None:
                    evidence_mapping["frames"] = packet.get("frames") or ({
                        "video_id": candidate.get("video_id"),
                        "frame_idx": candidate.get("frame_idx"),
                        "kf_n": candidate.get("kf_n"),
                        "pts_time": candidate.get("pts_time"),
                        "frame_path": candidate.get("frame_path"),
                    },)
                    evidence_mapping["asr_chunks"] = packet.get("asr_chunks", ())
                    evidence_mapping["ocr_text"] = packet.get("ocr_text", ())
                else:
                    evidence_mapping["asr_text"] = asr
                    evidence_mapping["ocr_text"] = ocr
                verification = verifier.verify(
                    normalized_answer,
                    evidence_mapping,
                    required_sources=semantic_required_sources,
                ).to_dict()
                if verification.get("abstain"):
                    answer_trace.append(trace_for(
                        candidate,
                        status="rejected_verification",
                        verification=verification,
                        provider=provider_name or record.get("provider"),
                    ))
                    continue
            answered.append({**candidate, "answer": normalized_answer,
                             "raw_answer": raw_answer,
                             "status": answer_status, "asr_ctx": bool(asr),
                             "ocr_ctx": bool(ocr), "answer_retry": retry_used,
                             "grounding_score": float(record.get("grounding_score", 0.0)),
                             "answer_confidence": float(record.get("answer_confidence", 0.0)),
                             "provider": provider_name or record.get("provider"),
                             "verification": verification,
                             "verification_policy": (
                                 "visual_frame_primary"
                                 if question_type == "screen_text"
                                 else "specialist_text_required"
                             ),
                             "evidence_packet": packet,
                             "evidence_support": evidence_support_score(normalized_answer, packet or {}),
                             "modality_fallback": "local_vlm_ocr" if local_ocr_fallback else None,
                             "structured_vlm": bool(structured_vlm),
                             "answer_stage": "answer"})
            answer_trace.append(trace_for(
                candidate,
                status=answer_status,
                raw_answer=raw_answer,
                answer=normalized_answer,
                answer_retry=retry_used,
                provider=provider_name or record.get("provider"),
                verification=verification,
                evidence_packet=packet,
                modality_fallback="local_vlm_ocr" if local_ocr_fallback else None,
            ))

        # Agreement among answers is a conservative consistency signal. It is
        # never used to manufacture an answer when the local VLM abstains.
        groups = {}
        for item in answered:
            key = " ".join(item["answer"].lower().split())
            groups[key] = groups.get(key, 0) + 1
        for item in answered:
            key = " ".join(item["answer"].lower().split())
            item["answer_consistency"] = groups[key] / max(len(answered), 1)
            item["ranking_score"] = (
                rank_weights["video_rank"] / (1.0 + item["video_rank"])
                + rank_weights["visual_relevance"] * float(item["base_score"])
                + rank_weights["answer_consistency"] * item["answer_consistency"]
            )
            if structured_vlm:
                item["ranking_score"] += (
                    rank_weights["grounding"] * item["grounding_score"] +
                    rank_weights["answer_confidence"] * item["answer_confidence"] +
                    (rank_weights["evidence_support"] * item["evidence_support"]
                     if evidence_mode else 0.0)
                )
            item["rerank_stage"] = "rerank"
            item["rerank_provenance"] = stage_record(
                "rerank", item,
                answer=item["answer"],
                evidence_sources=list((item.get("evidence_packet") or {}).get("sources", [])),
                evidence_support=item.get("evidence_support", 0.0),
            )
        answered.sort(key=lambda item: item["ranking_score"], reverse=True)
        output = []
        for item in answered[:max_answers]:
            output.append({
                "video_id": item["video_id"], "frame_id": item["frame_idx"],
                "kf_n": item["kf_n"], "pts_time": item["pts_time"],
                "answer": item["answer"], "status": item["status"],
                "ranking_score": item["ranking_score"],
                "answer_consistency": item["answer_consistency"],
                "grounding_score": item.get("grounding_score", 0.0),
                "answer_confidence": item.get("answer_confidence", 0.0),
                "provider": item.get("provider"),
                "verification": item.get("verification"),
                "verification_policy": item.get("verification_policy"),
                "evidence_sources": (item.get("evidence_packet") or {}).get("sources", []),
                "evidence_support": item.get("evidence_support", 0.0),
                "structured_vlm": bool(item.get("structured_vlm", False)),
                "source": item.get("source", "visual"),
                "sources": list(candidate_sources(item)),
                "provenance": [dict(value) for value in item.get("provenance", ()) or ()],
                "selection_stage": item.get("selection_stage", "selector"),
                "answer_stage": item.get("answer_stage", "answer"),
                "rerank_stage": item.get("rerank_stage", "rerank"),
                "route_state": prepared.get("route_state", "unknown"),
                "candidate_state": prepared.get("candidate_state", "unknown"),
                "wrong_video_state": prepared.get("wrong_video_state", "not_evaluated"),
            })
        for trace in answer_trace:
            if trace.get("status") == answer_status:
                match = next((item for item in answered
                              if item.get("video_id") == trace.get("video_id") and
                              item.get("kf_n") == trace.get("kf_n")), None)
                if match is not None:
                    trace["ranking_score"] = match.get("ranking_score")
                    trace["answer_consistency"] = match.get("answer_consistency")
        return {
            "query": prepared.get("query", ""), "question": prepared.get("question", ""),
            "answers": output,
            "status": answer_status if output else empty_status,
            "candidate_state": prepared.get("candidate_state", "unknown"),
            "candidate_miss": bool(prepared.get("candidate_miss", not candidates)),
            "wrong_video_state": prepared.get("wrong_video_state", "not_evaluated"),
            "wrong_video": prepared.get("wrong_video"),
            "route_state": prepared.get("route_state", "unknown"),
            "candidate_count": int(prepared.get("candidate_count", 0)),
            "vlm_candidate_count": int(prepared.get("vlm_candidate_count", len(candidates))),
            "rerank_weights": rank_weights,
            "structured_vlm": bool(structured_vlm),
            "selector_trace": list(prepared.get("selector_trace", ())),
            "answer_trace": answer_trace,
            "stages": {
                "retrieval": {"count": int(prepared.get("candidate_count", 0))},
                "candidate_pool": {"count": int(prepared.get("candidate_pool_count", 0))},
                "selection": prepared.get("selector_metrics", {}),
                "answer": {"attempted": len(candidates), "accepted": len(answered)},
                "rerank": {"ranked": len(answered), "returned": len(output)},
            },
        }

    def ranked_answers(self, query: str, question: str, *, top_videos: int = 20,
                       frames_per_video: int = 5, max_vlm_candidates: int = 12,
                       max_answers: int = 20, temporal_consensus: bool = True,
                       max_new_tokens: int = 128,
                       use_context: bool = True,
                       offline: bool = True, question_type: str | None = None,
                       required_modalities=None,
                       global_modality_router=None, rrf_weights: dict | None = None,
                       answer_rerank_weights: dict | None = None,
                       visual_selector_policy: str = "adaptive",
                       evidence_fusion: bool | None = None) -> dict:
        """Build a ranked, submission-safe Q&A result.

        Candidate generation is deliberately separate from answering: KIS ranks
        videos, a local frame lattice keeps up to ``frames_per_video`` peaks per
        video, and the explicitly injected local or remote provider answers
        only the strongest bounded set. No evidence-only result is valid.
        """
        answer_provider = getattr(self, "answer_provider", None)
        provider_is_remote = bool(getattr(answer_provider, "is_remote", False))
        if offline and provider_is_remote:
            raise ValueError("offline ranked_answers cannot use a remote answer provider")
        if not offline and answer_provider is None:
            raise ValueError("online ranked_answers requires an explicit answer provider")
        question_type = normalize_question_type(question_type)
        # Validate the modality contract before retrieval/model initialization.
        self._parse_modalities(required_modalities)
        if not required_modalities and question_type in {"screen_text", "spoken_fact"}:
            required_modalities = f"visual,{'ocr' if question_type == 'screen_text' else 'asr'}"
        prepared = self.prepare_ranked_candidates(
            query, question, top_videos=top_videos,
            frames_per_video=frames_per_video,
            max_vlm_candidates=max_vlm_candidates,
            temporal_consensus=temporal_consensus,
            required_modalities=required_modalities,
            global_modality_router=global_modality_router,
            rrf_weights=rrf_weights,
            question_type=question_type,
            evidence_fusion=evidence_fusion,
            visual_selector_policy=visual_selector_policy,
        )
        if answer_rerank_weights:
            prepared["rerank_weights"] = dict(answer_rerank_weights)
        prepared["required_sources"] = list(
            self._parse_modalities(required_modalities)
        )
        prepared["question_type"] = question_type
        result = self.answer_ranked_candidates(
            prepared, max_answers=max_answers,
            max_new_tokens=max_new_tokens, use_context=use_context,
            rerank_weights=answer_rerank_weights,
        )
        # Preserve retrieval/fallback provenance at the task boundary.  The
        # answer adapter ignores these diagnostics, while runtime orchestration
        # uses them to distinguish specialist success from a visual-only rescue.
        for key in (
            "retrieved_video_ids", "visual_retrieved_video_ids", "rrf_videos",
            "modality_route", "route_requested", "route_active", "route_state",
            "route_fallback_reason", "evidence_fusion", "candidate_source_counts",
            "specialist_candidate_count", "routing_plan", "candidate_state",
            "candidate_miss", "wrong_video_state", "wrong_video",
            "selector_metrics", "selector_trace",
        ):
            if key in prepared:
                result[key] = prepared[key]
        return result


if __name__ == "__main__":
    p = VQAPipelineV3()
    if len(sys.argv) > 2:
        q = sys.argv[1]; question = sys.argv[2]
    else:
        q = "siêu bão Biển Đông"
        question = "Cấp gió bão được nhắc đến là bao nhiêu?"
    out = p.answer(q, question)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
