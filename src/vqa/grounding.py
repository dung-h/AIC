"""Provenance-preserving external knowledge grounding for video Q&A.

External knowledge is allowed to create *retrieval hypotheses* only. It never
becomes a submitted answer: every final answer still needs ASR/OCR/visual
evidence in the competition corpus and a canonical frame index.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Iterable, Protocol
import unicodedata
from urllib.parse import urlparse, urlencode
from urllib.request import Request, urlopen


class GroundingError(RuntimeError):
    """A grounding dependency is unavailable or returned an invalid result."""


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return " ".join(value.split())


def _variants(values: Iterable[object]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split())
        key = text.casefold()
        if text and key not in seen:
            output.append(text[:600])
            seen.add(key)
    return tuple(output[:8])


# A whole VQA request is often a poor web query: its scene description and
# question can pull in unrelated generic results even though either half (or a
# quoted fact) is searchable.  These expressions only *select* text already
# supplied by the user; they never manufacture an answer.
_REQUEST_QUOTE = re.compile(r'["“”\'‘’]([^"“”\'‘’]{3,240})["“”\'‘’]')
_ACRONYM_EXPANSION = re.compile(
    r"\b([A-ZĐ][A-ZĐ0-9_-]{1,})\s*\(([^()]{3,120})\)"
)
_MAX_SEARCH_QUERIES = 4


@dataclass(frozen=True, slots=True)
class GroundingRequest:
    query: str
    question: str
    hypothesis_views: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _text(self.query, "query"))
        object.__setattr__(self, "question", _text(self.question, "question"))
        if isinstance(self.hypothesis_views, (str, bytes)):
            values = (str(self.hypothesis_views),)
        else:
            values = tuple(self.hypothesis_views)
        object.__setattr__(self, "hypothesis_views", _variants(values))

    def search_queries(self) -> tuple[str, ...]:
        """Return bounded, deterministic search views for external grounding.

        The order deliberately gives an exact supplied quote, then an acronym
        plus its user-supplied expansion, then the scene-only view a chance to
        retrieve before the historically over-broad combined request.  All
        views retain the original wording so the trace is reproducible.
        """

        full = f"{self.query} {self.question}".strip()
        # The source scene is a mandatory anchor.  Model-planned hypotheses
        # are useful alternatives, but may never exhaust the entire bounded
        # search budget and silently evict the original request (the only
        # view that can retrieve a verbatim local-news/source title).
        prefix: list[str] = []
        for match in _REQUEST_QUOTE.finditer(full):
            prefix.append(match.group(1))
        for match in _ACRONYM_EXPANSION.finditer(full):
            prefix.append(f"{match.group(1)} {match.group(2)}")
        # Planner-supplied views are verbatim spans from the query. They only
        # isolate a specific entity/fact from a long scene description; they
        # are never an answer or an external source of truth.
        selected = list(_variants(prefix))
        prefix_count = len(selected)
        if self.query not in selected:
            # Reserve one slot for the source query before considering a
            # hypothesis.  This also preserves the exact-quote/entity order
            # for historical requests that have no planner attached.
            hypothesis_budget = max(0, _MAX_SEARCH_QUERIES - len(selected) - 1)
        else:
            hypothesis_budget = max(0, _MAX_SEARCH_QUERIES - len(selected))
        for view in self.hypothesis_views:
            if len(selected) >= prefix_count + hypothesis_budget:
                break
            if view not in selected:
                selected.append(view)
        if self.query not in selected and len(selected) < _MAX_SEARCH_QUERIES:
            selected.append(self.query)
        for fallback in (full, self.question):
            if len(selected) >= _MAX_SEARCH_QUERIES:
                break
            if fallback not in selected:
                selected.append(fallback)
        return _variants(selected)[:_MAX_SEARCH_QUERIES]


@dataclass(frozen=True, slots=True)
class GroundingEvidence:
    """One external source converted into local-retrieval query variants.

    ``source_snippet`` is intentionally retained separately from
    ``query_variants``. The latter is the provider's historical expansion
    interface, whereas the former lets the local planner derive bounded,
    provenance-preserving quote/alias hypotheses without guessing from a
    merged blob. It is optional so existing persisted evidence records and
    callers remain valid.
    """

    source_url: str
    source_title: str
    query_variants: tuple[str, ...]
    provider: str
    source_snippet: str = ""
    search_query: str = ""

    def __post_init__(self) -> None:
        url = _text(self.source_url, "source_url")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("source_url must be an absolute http(s) URL")
        object.__setattr__(self, "source_url", url)
        object.__setattr__(self, "source_title", _text(self.source_title, "source_title")[:300])
        variants = _variants(self.query_variants)
        if not variants:
            raise ValueError("query_variants must contain at least one non-empty value")
        object.__setattr__(self, "query_variants", variants)
        object.__setattr__(self, "provider", _text(self.provider, "provider"))
        object.__setattr__(self, "source_snippet", " ".join(
            unicodedata.normalize("NFKC", str(self.source_snippet or "")).split()
        )[:1200])
        object.__setattr__(self, "search_query", " ".join(
            unicodedata.normalize("NFKC", str(self.search_query or "")).split()
        )[:600])

    def to_dict(self) -> dict:
        return {
            "source_url": self.source_url,
            "source_title": self.source_title,
            "source_snippet": self.source_snippet,
            "query_variants": list(self.query_variants),
            "provider": self.provider,
            "search_query": self.search_query,
        }


class GroundingResolver(Protocol):
    """Returns source-backed hypotheses; implementations never answer Q&A."""

    def resolve(self, request: GroundingRequest) -> tuple[GroundingEvidence, ...]: ...


class DisabledGroundingResolver:
    """Explicit offline/default resolver with no network behaviour."""

    enabled = False

    def resolve(self, request: GroundingRequest) -> tuple[GroundingEvidence, ...]:
        del request
        return ()


class SearxNGGroundingResolver:
    """Small, allow-listed SearXNG adapter for explicitly enabled deployments.

    Search snippets are treated as untrusted query-expansion material. Results
    outside the configured domains are discarded, and an empty usable result is
    normal. Transport/network errors fail closed so the caller can stop rather
    than silently claim an offline run used external grounding.
    """

    enabled = True

    def __init__(self, base_url: str, *, allowed_domains: Iterable[str], timeout_seconds: float = 5.0):
        self.base_url = _text(base_url, "base_url").rstrip("/")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute http(s) URL")
        self.allowed_domains = frozenset(
            str(domain).strip().casefold().lstrip(".")
            for domain in allowed_domains
            if str(domain).strip()
        )
        if not self.allowed_domains:
            raise ValueError("allowed_domains must be non-empty for external grounding")
        self.timeout_seconds = float(timeout_seconds)
        if not 0.1 <= self.timeout_seconds <= 30.0:
            raise ValueError("timeout_seconds must be in [0.1, 30]")

    def _allowed(self, url: str) -> bool:
        host = urlparse(url).hostname
        if not host:
            return False
        host = host.casefold()
        return any(host == domain or host.endswith("." + domain) for domain in self.allowed_domains)

    def resolve(self, request: GroundingRequest) -> tuple[GroundingEvidence, ...]:
        evidence: list[GroundingEvidence] = []
        seen_urls: set[str] = set()
        for search_text in request.search_queries():
            url = self.base_url + "/search?" + urlencode({"q": search_text, "format": "json"})
            req = Request(url, headers={"Accept": "application/json", "User-Agent": "HCMAI-grounding/1"})
            try:
                with urlopen(req, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception as exc:  # transport errors must be visible to the flow owner.
                raise GroundingError(f"external grounding request failed: {type(exc).__name__}") from exc
            if not isinstance(payload, dict):
                raise GroundingError("external grounding response must be a JSON object")
            results = payload.get("results", ())
            if not isinstance(results, list):
                raise GroundingError("external grounding response has invalid results")
            for result in results[:20]:
                if not isinstance(result, dict):
                    continue
                source_url = str(result.get("url", "")).strip()
                url_key = source_url.casefold()
                if not self._allowed(source_url) or url_key in seen_urls:
                    continue
                title = str(result.get("title", "")).strip()
                snippet = re.sub(r"<[^>]+>", " ", str(result.get("content", "")))
                variants = _variants((title, snippet))
                if not title or not variants:
                    continue
                seen_urls.add(url_key)
                evidence.append(GroundingEvidence(
                    source_url=source_url,
                    source_title=title,
                    query_variants=variants,
                    provider="searxng_allowlisted",
                    source_snippet=snippet,
                    search_query=search_text,
                ))
                if len(evidence) >= 5:
                    return tuple(evidence)
        return tuple(evidence)


class DuckDuckGoGroundingResolver:
    """Explicit DuckDuckGo backend for deployments without a SearXNG service.

    It is deliberately a separately selected provider, never a fallback from
    SearXNG.  This preserves reproducible deployment intent and makes a
    missing/blocked provider visible to the caller.
    """

    enabled = True

    def __init__(self, *, allowed_domains: Iterable[str], timeout_seconds: float = 5.0):
        self.allowed_domains = frozenset(
            str(domain).strip().casefold().lstrip(".")
            for domain in allowed_domains
            if str(domain).strip()
        )
        if not self.allowed_domains:
            raise ValueError("allowed_domains must be non-empty for external grounding")
        self.timeout_seconds = float(timeout_seconds)
        if not 0.1 <= self.timeout_seconds <= 30.0:
            raise ValueError("timeout_seconds must be in [0.1, 30]")

    def _allowed(self, url: str) -> bool:
        host = urlparse(url).hostname
        if not host:
            return False
        host = host.casefold()
        return any(host == domain or host.endswith("." + domain) for domain in self.allowed_domains)

    def resolve(self, request: GroundingRequest) -> tuple[GroundingEvidence, ...]:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise GroundingError(
                "DuckDuckGo grounding requires the optional 'ddgs' package; "
                "install the project requirements"
            ) from exc
        evidence: list[GroundingEvidence] = []
        seen_urls: set[str] = set()
        try:
            with DDGS(timeout=self.timeout_seconds) as client:
                for search_text in request.search_queries():
                    try:
                        results = list(client.text(search_text, max_results=20))
                    except Exception as exc:
                        # ``ddgs`` reports a valid zero-hit search as an
                        # exception. It applies to one view, not the whole
                        # fact-grounding attempt: later focused views may be
                        # the one that exposes a useful source.
                        if "no results found" in str(exc).casefold():
                            continue
                        raise GroundingError(
                            f"DuckDuckGo grounding request failed: {type(exc).__name__}"
                        ) from exc
                    for result in results:
                        if not isinstance(result, dict):
                            continue
                        source_url = str(result.get("href") or result.get("url") or "").strip()
                        url_key = source_url.casefold()
                        if not self._allowed(source_url) or url_key in seen_urls:
                            continue
                        title = str(result.get("title", "")).strip()
                        snippet = str(result.get("body") or result.get("content") or "").strip()
                        variants = _variants((title, snippet))
                        if not title or not variants:
                            continue
                        seen_urls.add(url_key)
                        evidence.append(GroundingEvidence(
                            source_url=source_url,
                            source_title=title,
                            query_variants=variants,
                            provider="ddg_allowlisted",
                            source_snippet=snippet,
                            search_query=search_text,
                        ))
                        if len(evidence) >= 5:
                            return tuple(evidence)
        except GroundingError:
            raise
        return tuple(evidence)


__all__ = [
    "DisabledGroundingResolver", "DuckDuckGoGroundingResolver", "GroundingError", "GroundingEvidence",
    "GroundingRequest", "GroundingResolver", "SearxNGGroundingResolver",
]
