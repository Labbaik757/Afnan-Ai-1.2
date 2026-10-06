"""Source discovery, acquisition and trust profiles.

Discovery and acquisition go through a ``SourceProvider``
protocol.  Production uses the browser's WebResearch +
extraction; tests use a fake provider.  The research core
never touches the network directly.

Diversity is enforced: dozens of results from one domain
are not independent evidence.  A search result is never
evidence until its content is acquired and extracted.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Protocol

from afnan_ai.redaction import redact_text
from afnan_ai.research.models import (
    Source,
    SourceClass,
    SourceSnapshot,
    SourceTrustProfile,
)


class SourceProvider(Protocol):
    """What the research core needs from the outside world."""

    def search(
        self, query: str, *, max_results: int = 10
    ) -> list[dict[str, Any]]:
        """Return [{url, title, snippet}]."""
        ...

    def fetch(
        self, url: str
    ) -> dict[str, Any] | None:
        """Return {title, text, published_at} or None."""
        ...


class WebResearchProvider:
    """Production provider over the browser's WebResearch."""

    def __init__(self, web_research: Any) -> None:
        self._wr = web_research

    def search(
        self, query: str, *, max_results: int = 10
    ) -> list[dict[str, Any]]:
        try:
            results = self._wr.search(
                query, max_results=max_results
            )
        except Exception:
            return []
        out = []
        for r in results or []:
            if not isinstance(r, dict):
                continue
            url = str(r.get("url", ""))
            if not url:
                continue
            out.append(
                {
                    "url": url,
                    "title": str(r.get("title", ""))[:300],
                    "snippet": str(
                        r.get("snippet", r.get("text", ""))
                    )[:500],
                }
            )
        return out

    def fetch(self, url: str) -> dict[str, Any] | None:
        try:
            opened = self._wr.open_result(url)
        except Exception:
            return None
        if not isinstance(opened, dict):
            return None
        text = str(opened.get("text", opened.get("content", "")))
        if not text.strip():
            return None
        # Secrets never become research content.
        text = redact_text(text)
        return {
            "title": str(opened.get("title", ""))[:300],
            "text": text,
            "published_at": str(
                opened.get("published_at", "")
            ),
        }


def _classify_source(
    url: str, title: str, snippet: str
) -> SourceClass:
    text = f"{url} {title} {snippet}".lower()
    if any(
        k in text
        for k in (".gov", "government", "regulation")
    ):
        return SourceClass.GOVERNMENT
    if any(
        k in text
        for k in (".edu", "arxiv", "journal", "doi.org")
    ):
        return SourceClass.ACADEMIC
    if any(
        k in text
        for k in (
            "docs.", "documentation", "developer.",
            "api reference",
        )
    ):
        return SourceClass.OFFICIAL_DOC
    if any(
        k in text
        for k in (
            "blog", "medium.com", "dev.to", "analysis",
            "techcrunch",
        )
    ):
        return SourceClass.TECHNICAL
    if any(
        k in text
        for k in ("reddit", "stackoverflow", "forum", "wiki")
    ):
        return SourceClass.COMMUNITY
    if any(
        k in text
        for k in ("reuters", "bbc", "nytimes", "news")
    ):
        return SourceClass.JOURNALISM
    return SourceClass.COMPANY


class SourceDiscovery:
    """Diverse, budgeted source discovery."""

    def __init__(
        self,
        provider: SourceProvider,
        *,
        max_sources: int = 20,
        max_searches: int = 12,
        min_diversity: int = 3,
    ) -> None:
        self.provider = provider
        self.max_sources = max_sources
        self.max_searches = max_searches
        self.min_diversity = min_diversity
        self.searches_done = 0

    def discover(
        self,
        queries: list[str],
        *,
        session_id: str = "",
    ) -> list[Source]:
        sources: list[Source] = []
        seen_urls: set[str] = set()
        domain_counts: dict[str, int] = {}
        for query in queries:
            if self.searches_done >= self.max_searches:
                break
            if len(sources) >= self.max_sources:
                break
            self.searches_done += 1
            try:
                results = self.provider.search(
                    query, max_results=10
                )
            except Exception:
                continue
            for r in results:
                url = r.get("url", "")
                if not url or url in seen_urls:
                    continue
                src = Source(
                    session_id=session_id,
                    url=url,
                    title=r.get("title", ""),
                    query_context=query[:200],
                    acquisition_method="search",
                )
                src.source_class = _classify_source(
                    url, src.title, r.get("snippet", "")
                )
                # Diversity: cap per-domain results.
                count = domain_counts.get(src.domain, 0)
                if count >= 4:
                    continue
                domain_counts[src.domain] = count + 1
                seen_urls.add(url)
                sources.append(src)
                if len(sources) >= self.max_sources:
                    break
        return sources


class SourceAcquisition:
    """Fetch content with provenance; content hashed."""

    def __init__(self, provider: SourceProvider) -> None:
        self.provider = provider
        # content_ref store: content_hash -> redacted text
        # (bounded; production persists via artifacts).
        self._content: dict[str, str] = {}

    def acquire(
        self, source: Source
    ) -> SourceSnapshot | None:
        fetched = self.provider.fetch(source.url)
        if not fetched:
            source.status = "failed"
            return None
        text = fetched.get("text", "")
        content_hash = hashlib.sha256(
            text.encode("utf-8", "replace")
        ).hexdigest()[:16]
        source.content_hash = content_hash
        source.title = fetched.get("title", "") or source.title
        if fetched.get("published_at"):
            source.published_at = fetched["published_at"]
        source.status = "acquired"
        # Bound the in-memory store.
        if len(self._content) > 100:
            oldest = next(iter(self._content))
            del self._content[oldest]
        self._content[content_hash] = text
        return SourceSnapshot(
            source_id=source.source_id,
            content_ref=f"content:{content_hash}",
            content_hash=content_hash,
        )

    def content_for(
        self, snapshot: SourceSnapshot
    ) -> str:
        return self._content.get(snapshot.content_hash, "")


# ------------------------------------------------------------------
# Trust profiles
# ------------------------------------------------------------------

_CLASS_BASE_SCORE = {
    SourceClass.PRIMARY: 0.85,
    SourceClass.OFFICIAL_DOC: 0.85,
    SourceClass.GOVERNMENT: 0.8,
    SourceClass.ACADEMIC: 0.8,
    SourceClass.COMPANY: 0.6,
    SourceClass.JOURNALISM: 0.65,
    SourceClass.TECHNICAL: 0.6,
    SourceClass.COMMUNITY: 0.4,
}


def build_trust_profile(
    source: Source,
    *,
    corroborated_by: int = 0,
    has_methodology: bool = False,
    is_recent: bool = False,
) -> SourceTrustProfile:
    """Credibility signals — a score, never proof of truth."""
    base = _CLASS_BASE_SCORE.get(source.source_class, 0.5)
    signals: dict[str, Any] = {
        "source_class": source.source_class.value,
        "base": base,
        "corroborated_by": corroborated_by,
        "has_methodology": has_methodology,
        "is_recent": is_recent,
        "publisher": source.publisher or source.domain,
    }
    score = base
    if corroborated_by:
        score = min(0.95, score + 0.05 * min(corroborated_by, 4))
    if has_methodology:
        score = min(0.95, score + 0.05)
    if is_recent:
        score = min(0.95, score + 0.03)
    notes = []
    if source.source_class == SourceClass.COMMUNITY:
        notes.append(
            "community source: useful but verify independently"
        )
    return SourceTrustProfile(
        source_id=source.source_id,
        signals=signals,
        score=round(score, 3),
        notes=notes,
    )
