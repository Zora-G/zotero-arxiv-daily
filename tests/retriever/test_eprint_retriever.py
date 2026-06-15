"""Tests for EprintRetriever."""

from datetime import datetime, timezone
from types import SimpleNamespace

from omegaconf import open_dict
import feedparser
import pytest

from zotero_arxiv_daily.retriever.eprint_retriever import EprintRetriever


def test_eprint_retriever(config, monkeypatch):
    def _patched_parse(url):
        assert url == "https://eprint.iacr.org/rss/rss.xml"
        return SimpleNamespace(
            bozo=False,
            entries=[
                SimpleNamespace(
                    title="today match",
                    link="https://eprint.iacr.org/2026/9999",
                    summary="Summary A",
                    published="Mon, 15 Jun 2026 00:00:00 +0000",
                    tags=[{"term": "Secret-key cryptography"}],
                    authors=[{"name": "Alice"}],
                    links=[{"type": "application/pdf", "href": "https://eprint.iacr.org/2026/9999.pdf"}],
                ),
                SimpleNamespace(
                    title="today unmatch",
                    link="https://eprint.iacr.org/2026/9998",
                    summary="Summary B",
                    published="Mon, 15 Jun 2026 00:00:00 +0000",
                    tags=[{"term": "Quantum cryptography"}],
                    authors=[{"name": "Bob"}],
                ),
                SimpleNamespace(
                    title="yesterday old",
                    link="https://eprint.iacr.org/2026/9997",
                    summary="Summary C",
                    published="Sun, 14 Jun 2026 00:00:00 +0000",
                    tags=[{"term": "Secret-key cryptography"}],
                    authors=[{"name": "Carol"}],
                ),
            ],
        )

    monkeypatch.setattr(feedparser, "parse", _patched_parse)
    monkeypatch.setattr(
        "zotero_arxiv_daily.retriever.eprint_retriever.datetime",
        SimpleNamespace(
            now=lambda tz=None: datetime(2026, 6, 15, 0, 0, tzinfo=timezone.utc),
            strptime=datetime.strptime,
        ),
    )

    with open_dict(config.source):
        config.source.eprint = {"category": ["Secret-key cryptography"]}

    retriever = EprintRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == 1
    assert papers[0].title == "today match"
    assert papers[0].source == "eprint"
    assert papers[0].url == "https://eprint.iacr.org/2026/9999"
    assert papers[0].pdf_url == "https://eprint.iacr.org/2026/9999.pdf"


def test_eprint_requires_category(config):
    with open_dict(config.source):
        config.source.eprint = {"category": None}
    with pytest.raises(ValueError, match="category must be specified"):
        EprintRetriever(config)
