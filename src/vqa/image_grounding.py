"""External-image grounding that always returns to the local video corpus.

This is the production-safe form of the QUEST pattern used in AIC 2025:

``entity-like visual query -> web reference images -> local image retrieval``

The web images are never candidates for submission.  They are short-lived
reference inputs to a local VKIS-compatible index; the VQA pipeline then maps
the returned video/frame candidate through its canonical map before answering.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from ipaddress import ip_address
import json
import math
import re
from tempfile import TemporaryDirectory
from typing import Callable, Iterable, Protocol
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


class ImageGroundingError(RuntimeError):
    """The explicitly enabled image-grounding route is unavailable or invalid."""


def _text(value: object, field: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{field} must be non-empty")
    return result


def _public_http_url(value: object, field: str) -> str:
    url = _text(value, field)
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    if parsed.scheme not in {"http", "https"} or not host:
        raise ValueError(f"{field} must be an absolute http(s) URL")
    if not _is_public_host(host):
        raise ValueError(f"{field} must not target a local or private host")
    return url


def _is_public_host(host: str) -> bool:
    if not host or host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return False
    try:
        return bool(ip_address(host).is_global)
    except ValueError:
        return True


def _normalise_domains(domains: Iterable[object]) -> frozenset[str]:
    return frozenset(
        str(domain).strip().casefold().lstrip(".")
        for domain in domains
        if str(domain).strip()
    )


def _host_allowed(url: str, domains: frozenset[str], allow_any_host: bool) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    if parsed.scheme not in {"http", "https"} or not _is_public_host(host):
        return False
    return allow_any_host or any(host == domain or host.endswith("." + domain) for domain in domains)


@dataclass(frozen=True, slots=True)
class ImageGroundingRequest:
    """A visual entity hypothesis; not an answer or a frame assertion."""

    query: str
    question: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _text(self.query, "query"))
        object.__setattr__(self, "question", _text(self.question, "question"))


@dataclass(frozen=True, slots=True)
class ImageReference:
    """A web image with its source-page provenance."""

    source_url: str
    image_url: str
    source_title: str
    query: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_url", _public_http_url(self.source_url, "source_url"))
        object.__setattr__(self, "image_url", _public_http_url(self.image_url, "image_url"))
        object.__setattr__(self, "source_title", _text(self.source_title, "source_title")[:300])
        object.__setattr__(self, "query", _text(self.query, "query")[:600])

    def to_dict(self) -> dict:
        return {
            "source_url": self.source_url,
            "image_url": self.image_url,
            "source_title": self.source_title,
            "query": self.query,
        }


@dataclass(frozen=True, slots=True)
class ImageGroundingCandidate:
    """A local-index hit generated from web references, never a web frame."""

    video_id: str
    frame_idx: int
    pts_time: float
    score: float
    rank: int
    reference_urls: tuple[str, ...]

    def __post_init__(self) -> None:
        video_id = _text(self.video_id, "video_id")
        frame_idx = int(self.frame_idx)
        if frame_idx < 0:
            raise ValueError("frame_idx must be non-negative")
        pts_time = float(self.pts_time)
        score = float(self.score)
        rank = int(self.rank)
        if not math.isfinite(pts_time) or not math.isfinite(score) or rank < 1:
            raise ValueError("invalid image grounding candidate values")
        urls = tuple(dict.fromkeys(_public_http_url(url, "reference_url") for url in self.reference_urls))
        if not urls:
            raise ValueError("reference_urls must be non-empty")
        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "frame_idx", frame_idx)
        object.__setattr__(self, "pts_time", pts_time)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "reference_urls", urls[:5])

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "frame_idx": self.frame_idx,
            "pts_time": self.pts_time,
            "score": self.score,
            "rank": self.rank,
            "reference_urls": list(self.reference_urls),
            "source": "external_image",
        }


@dataclass(frozen=True, slots=True)
class ImageGroundingResult:
    """JSON-safe provenance plus local candidates from an image route."""

    status: str
    references: tuple[ImageReference, ...] = ()
    candidates: tuple[ImageGroundingCandidate, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"no_references", "no_usable_images", "candidates_ready"}:
            raise ValueError(f"unsupported image grounding status: {self.status!r}")
        if self.status == "candidates_ready" and not self.candidates:
            raise ValueError("candidates_ready requires candidates")

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reference_count": len(self.references),
            "candidate_count": len(self.candidates),
            "references": [reference.to_dict() for reference in self.references],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


class ImageGroundingProvider(Protocol):
    """Explicit image search + local visual retrieval capability."""

    enabled: bool

    def retrieve(self, request: ImageGroundingRequest, *, topk: int) -> ImageGroundingResult: ...


class DisabledImageGroundingProvider:
    enabled = False

    def retrieve(self, request: ImageGroundingRequest, *, topk: int) -> ImageGroundingResult:
        del request, topk
        return ImageGroundingResult(status="no_references")


class SearxNGImageGroundingProvider:
    """SearXNG image search followed by local VKIS-compatible retrieval.

    ``vkis_factory`` is lazy: no vision model is loaded until a qualified query
    actually yields a usable reference image.  Image bytes are bounded and are
    written only into a temporary directory, never the submission/runtime
    corpus.  The allow-list applies to both source pages and image hosts unless
    the operator explicitly sets ``allow_any_image_host``.
    """

    enabled = True

    def __init__(
        self,
        base_url: str,
        *,
        allowed_domains: Iterable[str],
        allow_any_image_host: bool,
        vkis_factory: Callable[[], object],
        timeout_seconds: float = 5.0,
        max_references: int = 3,
        search_transport: Callable[[str, float], object] | None = None,
        image_transport: Callable[[str, float, int], bytes] | None = None,
    ) -> None:
        self.base_url = _public_http_url(base_url, "base_url").rstrip("/")
        self.allowed_domains = _normalise_domains(allowed_domains)
        self.allow_any_image_host = bool(allow_any_image_host)
        if not self.allowed_domains and not self.allow_any_image_host:
            raise ValueError(
                "image grounding requires allowed_domains or allow_any_image_host=true"
            )
        if not callable(vkis_factory):
            raise TypeError("vkis_factory must be callable")
        self.vkis_factory = vkis_factory
        self.timeout_seconds = float(timeout_seconds)
        if not 0.1 <= self.timeout_seconds <= 30.0:
            raise ValueError("timeout_seconds must be in [0.1, 30]")
        self.max_references = int(max_references)
        if not 1 <= self.max_references <= 8:
            raise ValueError("max_references must be in [1, 8]")
        self._search_transport = search_transport or self._fetch_json
        self._image_transport = image_transport or self._fetch_image

    @staticmethod
    def _fetch_json(url: str, timeout: float) -> object:
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "HCMAI-grounding/1"})
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _fetch_image(url: str, timeout: float, max_bytes: int) -> bytes:
        request = Request(url, headers={"Accept": "image/*", "User-Agent": "HCMAI-grounding/1"})
        with urlopen(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ImageGroundingError("reference image exceeds byte budget")
            payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ImageGroundingError("reference image exceeds byte budget")
        return payload

    def _search(self, request: ImageGroundingRequest) -> tuple[ImageReference, ...]:
        query = request.query
        url = self.base_url + "/search?" + urlencode({
            "q": query,
            "categories": "images",
            "format": "json",
            "language": "vi-VN",
        })
        try:
            payload = self._search_transport(url, self.timeout_seconds)
        except ImageGroundingError:
            raise
        except Exception as exc:
            raise ImageGroundingError(
                f"external image grounding request failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ImageGroundingError("external image grounding response has invalid results")
        return self._references_from_results(payload["results"], query)

    def _references_from_results(
        self, results: Iterable[object], query: str
    ) -> tuple[ImageReference, ...]:
        references: list[ImageReference] = []
        seen: set[str] = set()
        for item in list(results)[:20]:
            if not isinstance(item, dict):
                continue
            source_url = str(item.get("url") or item.get("source") or "").strip()
            image_url = str(
                item.get("img_src") or item.get("thumbnail_src")
                or item.get("image") or item.get("thumbnail") or ""
            ).strip()
            title = str(item.get("title", "")).strip()
            if not source_url or not image_url or not title:
                continue
            if not _host_allowed(source_url, self.allowed_domains, self.allow_any_image_host):
                continue
            if not _host_allowed(image_url, self.allowed_domains, self.allow_any_image_host):
                continue
            try:
                reference = ImageReference(
                    source_url=source_url, image_url=image_url,
                    source_title=title, query=query,
                )
            except ValueError:
                continue
            if reference.image_url in seen:
                continue
            seen.add(reference.image_url)
            references.append(reference)
            if len(references) >= self.max_references:
                break
        return tuple(references)

    def retrieve(self, request: ImageGroundingRequest, *, topk: int) -> ImageGroundingResult:
        if not 1 <= int(topk) <= 100:
            raise ValueError("topk must be between 1 and 100")
        references = self._search(request)
        if not references:
            return ImageGroundingResult(status="no_references")

        # Aggregate per-reference ranks. This prevents one nearly duplicated
        # web thumbnail from producing multiple votes for the same local frame.
        by_video: dict[str, dict] = {}
        usable_references: list[ImageReference] = []
        with TemporaryDirectory(prefix="hcmai-image-grounding-") as directory:
            for reference_index, reference in enumerate(references, 1):
                try:
                    image_bytes = self._image_transport(
                        reference.image_url, self.timeout_seconds, 8 * 1024 * 1024
                    )
                    from PIL import Image
                    image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    path = f"{directory}/reference-{reference_index}.jpg"
                    image.save(path, "JPEG", quality=92)
                    vkis = self.vkis_factory()
                    raw_hits = vkis.search_image(path, topk=min(100, int(topk)))
                except ImageGroundingError:
                    raise
                except Exception:
                    # A corrupt thumbnail or unsupported codec is a bad
                    # reference, not evidence that the local video is absent.
                    continue
                usable_references.append(reference)
                for local_rank, hit in enumerate(raw_hits, 1):
                    if not isinstance(hit, (tuple, list)) or len(hit) < 4:
                        raise ImageGroundingError("VKIS image search returned an invalid candidate")
                    video_id, frame_idx, pts_time, _raw_score = hit[:4]
                    video_id = str(video_id).strip()
                    if not video_id:
                        continue
                    value = by_video.setdefault(video_id, {
                        "score": 0.0,
                        "best": None,
                        "best_key": None,
                        "reference_urls": [],
                    })
                    value["score"] += 1.0 / (60.0 + float(local_rank))
                    key = (local_rank, reference_index, int(frame_idx))
                    if value["best"] is None or key < value["best_key"]:
                        value["best"] = (int(frame_idx), float(pts_time))
                        value["best_key"] = key
                    if reference.source_url not in value["reference_urls"]:
                        value["reference_urls"].append(reference.source_url)

        if not by_video:
            return ImageGroundingResult(
                status="no_usable_images", references=tuple(usable_references)
            )
        ordered = sorted(
            by_video.items(),
            key=lambda item: (-float(item[1]["score"]), item[1]["best_key"], item[0]),
        )[:int(topk)]
        candidates = tuple(
            ImageGroundingCandidate(
                video_id=video_id,
                frame_idx=value["best"][0],
                pts_time=value["best"][1],
                score=float(value["score"]),
                rank=rank,
                reference_urls=tuple(value["reference_urls"]),
            )
            for rank, (video_id, value) in enumerate(ordered, 1)
        )
        return ImageGroundingResult(
            status="candidates_ready",
            references=tuple(usable_references),
            candidates=candidates,
        )


class DuckDuckGoImageGroundingProvider(SearxNGImageGroundingProvider):
    """Explicit keyless image-search backend; never a SearXNG fallback."""

    def __init__(
        self,
        *,
        allowed_domains: Iterable[str],
        allow_any_image_host: bool,
        vkis_factory: Callable[[], object],
        timeout_seconds: float = 5.0,
        max_references: int = 3,
    ) -> None:
        # The base URL is unused by the overridden search method, but keeping
        # the rest of the bounded-download/local-VKIS contract in one class
        # avoids a second implementation drifting from production semantics.
        super().__init__(
            "https://duckduckgo.com",
            allowed_domains=allowed_domains,
            allow_any_image_host=allow_any_image_host,
            vkis_factory=vkis_factory,
            timeout_seconds=timeout_seconds,
            max_references=max_references,
        )

    def _search(self, request: ImageGroundingRequest) -> tuple[ImageReference, ...]:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise ImageGroundingError(
                "DuckDuckGo image grounding requires the optional 'ddgs' package; "
                "install the project requirements"
            ) from exc
        try:
            with DDGS(timeout=self.timeout_seconds) as client:
                results = list(client.images(request.query, max_results=20))
        except Exception as exc:
            raise ImageGroundingError(
                f"DuckDuckGo image grounding request failed: {type(exc).__name__}"
            ) from exc
        return self._references_from_results(results, request.query)


_VISUAL_ENTITY_CUES = re.compile(
    r"\b(?:logo|linh vật|biểu tượng|thương hiệu|nhãn hiệu|sản phẩm|đồ chơi|"
    r"búp bê|nhân vật|mô hình|mascot|logo|brand|character|toy|figurine)\b",
    flags=re.IGNORECASE,
)
_FACTUAL_CUES = re.compile(
    r"\b(?:bài thơ|câu thơ|trích|tác giả|lịch sử|câu lạc bộ|tổ chức|địa phương|"
    r"xã|huyện|nhiệt độ|bao nhiêu|công thức|tiêu đề|tên món|tên món ăn|"
    r"nguyên liệu|thịt nạc|thịt bò|gram|when|where|what year)\b",
    flags=re.IGNORECASE,
)


def image_grounding_eligibility(
    query: object,
    question: object,
    *,
    question_type: object,
    specialist_modalities: Iterable[object] = (),
) -> tuple[bool, tuple[str, ...]]:
    """Return an auditable gate for the costly visual OOK branch.

    Textual fact questions are deliberately excluded: external text grounding
    plus ASR/OCR is a stronger and more directly verifiable route for them.
    """
    scene = _text(query, "query")
    asked = _text(question, "question")
    full = f"{scene}\n{asked}"
    modalities = {
        str(value).strip().lower() for value in specialist_modalities
        if str(value).strip().lower() in {"asr", "ocr"}
    }
    if modalities:
        return False, ("specialist_text_contract",)
    if _FACTUAL_CUES.search(full):
        return False, ("factual_query_prefers_text_grounding",)
    reasons: list[str] = []
    if _VISUAL_ENTITY_CUES.search(full):
        reasons.append("visual_entity_cue")
    # Sentence-initial words ("Đoạn", "Trong", "Hỏi") are not named
    # entities. Only count capitalised tokens that occur after a non-boundary
    # word and require a second adjacent token to avoid opening prose.
    proper_tokens = []
    tokens = list(re.finditer(r"\b[\wÀ-ỹ-]+\b", scene))
    for index, match in enumerate(tokens):
        token = match.group(0)
        if len(token) <= 2 or not token[:1].isupper():
            continue
        prefix = scene[:match.start()].rstrip()
        if prefix and prefix[-1] not in ".!?\n":
            proper_tokens.append(token)
    if proper_tokens:
        reasons.append("named_visual_entity")
    return bool(reasons), tuple(reasons or ("no_visual_entity_signal",))


__all__ = [
    "DisabledImageGroundingProvider", "ImageGroundingCandidate",
    "DuckDuckGoImageGroundingProvider", "ImageGroundingError", "ImageGroundingProvider", "ImageGroundingRequest",
    "ImageGroundingResult", "ImageReference", "SearxNGImageGroundingProvider",
    "image_grounding_eligibility",
]
