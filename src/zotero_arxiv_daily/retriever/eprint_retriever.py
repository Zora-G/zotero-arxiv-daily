import re
from datetime import datetime, timezone
from typing import Any
from html import unescape

import feedparser
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


@register_retriever("eprint")
class EprintRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.retriever_config.category is None:
            raise ValueError(f"category must be specified for {self.name}")

    def _retrieve_raw_papers(self) -> list[Any]:
        response = feedparser.parse("https://eprint.iacr.org/rss/rss.xml")
        if len(response.entries) == 0:
            if response.bozo:
                raise ValueError(f"Failed to parse ePrint RSS: {response.bozo_exception}")
            logger.warning("No papers found in ePrint RSS.")
            return []
        category_set = {c.strip().lower() for c in self.retriever_config.category}
        target_date = datetime.now(timezone.utc).date()
        papers = []
        for entry in response.entries:
            entry_date_raw = getattr(entry, "published", None) or getattr(entry, "updated", None)
            if entry_date_raw is None:
                continue
            try:
                entry_time = datetime.strptime(entry_date_raw, "%a, %d %b %Y %H:%M:%S %z")
            except ValueError:
                logger.warning(f"Failed to parse ePrint time '{entry_date_raw}' from {entry.get('link')}")
                continue
            if entry_time.date() != target_date:
                continue

            entry_categories = {
                tag.get("term", "").strip().lower()
                for tag in getattr(entry, "tags", [])
                if isinstance(tag, dict)
            }
            if category_set and (not entry_categories or not entry_categories.intersection(category_set)):
                continue

            papers.append(entry)
            if self.config.executor.debug and len(papers) >= 10:
                break

        if self.config.executor.debug:
            papers = papers[:10]

        return papers

    def convert_to_paper(self, raw_paper: Any) -> Paper:
        title = raw_paper.title

        authors = []
        if getattr(raw_paper, "authors", None):
            for item in raw_paper.authors:
                name = item.get("name")
                if name is not None:
                    authors.append(name)
        if not authors and getattr(raw_paper, "author", None):
            authors = [a.strip() for a in re.split(r",| and ", str(raw_paper.author)) if a.strip()]

        abstract = getattr(raw_paper, "summary", "")
        abstract = re.sub(r"<[^>]+>", "", abstract)
        abstract = unescape(abstract)

        pdf_url = None
        for link in getattr(raw_paper, "links", []):
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract.strip(),
            url=raw_paper.link,
            pdf_url=pdf_url,
            full_text=None,
        )
