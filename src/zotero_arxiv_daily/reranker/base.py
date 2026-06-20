from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, CorpusPaper
import numpy as np
import math
from typing import Type
class BaseReranker(ABC):
    def __init__(self, config:DictConfig):
        self.config = config

    def rerank(self, candidates:list[Paper], corpus:list[CorpusPaper]) -> list[Paper]:
        corpus = sorted(corpus,key=lambda x: x.added_date,reverse=True)
        time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        time_decay_weight: np.ndarray = time_decay_weight / time_decay_weight.sum()
        sim = self.get_similarity_score([c.abstract for c in candidates], [c.abstract for c in corpus])
        assert sim.shape == (len(candidates), len(corpus))
        scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
        interest_profile = self._interest_profile()
        interest_weight = self._interest_profile_weight()
        if interest_profile and interest_weight > 0:
            profile_sim = self.get_similarity_score([self._paper_profile_text(c) for c in candidates], interest_profile)
            assert profile_sim.shape == (len(candidates), len(interest_profile))
            scores = scores + profile_sim.max(axis=1) * 10 * interest_weight
        if self._quality_boost_enabled():
            quality_bonus = np.array([self._quality_bonus(c) for c in candidates], dtype=float)
            scores = scores + quality_bonus
        for s,c in zip(scores,candidates):
            c.score = s
        candidates = sorted(candidates,key=lambda x: x.score,reverse=True)
        return candidates

    def _quality_boost_enabled(self) -> bool:
        executor_cfg = getattr(self.config, "executor", None)
        return bool(getattr(executor_cfg, "quality_boost", False))

    def _interest_profile(self) -> list[str]:
        executor_cfg = getattr(self.config, "executor", None)
        if not executor_cfg:
            return []
        profile = getattr(executor_cfg, "interest_profile", None)
        if not profile:
            return []
        if isinstance(profile, str):
            return [profile.strip()] if profile.strip() else []
        return [str(item).strip() for item in profile if str(item).strip()]

    def _interest_profile_weight(self) -> float:
        executor_cfg = getattr(self.config, "executor", None)
        if not executor_cfg:
            return 0.0
        return float(getattr(executor_cfg, "interest_profile_weight", 0.0) or 0.0)

    @staticmethod
    def _paper_profile_text(paper: Paper) -> str:
        return "\n".join(part for part in (paper.title, paper.abstract) if part)

    def _quality_bonus(self, paper: Paper) -> float:
        executor_cfg = getattr(self.config, "executor", None)
        if not executor_cfg:
            return 0.0

        citation_weight = float(getattr(executor_cfg, "citation_weight", 0.0) or 0.0)
        author_weight = float(getattr(executor_cfg, "author_weight", 0.0) or 0.0)
        venue_weight = float(getattr(executor_cfg, "venue_weight", 0.0) or 0.0)
        citation_scale = float(getattr(executor_cfg, "citation_scale", 200.0) or 200.0)
        author_scale = float(getattr(executor_cfg, "author_h_index_scale", 25.0) or 25.0)

        citation_count = float(getattr(paper, "citation_count", None) or 0.0)
        author_h_index = float(getattr(paper, "author_h_index", None) or 0.0)
        venue_type = getattr(paper, "venue_type", None)

        citation_bonus = 0.0
        if citation_count > 0:
            citation_bonus = math.log1p(citation_count) / math.log1p(citation_scale)

        author_bonus = 0.0
        if author_h_index > 0:
            author_bonus = math.log1p(author_h_index) / math.log1p(author_scale)

        venue_bonus = 0.0
        if venue_type in {"journal", "conference"}:
            venue_bonus = 1.0
        elif venue_type and venue_type != "repository":
            venue_bonus = 0.5

        return citation_weight * citation_bonus + author_weight * author_bonus + venue_weight * venue_bonus
    
    @abstractmethod
    def get_similarity_score(self, s1:list[str], s2:list[str]) -> np.ndarray:
        raise NotImplementedError

registered_rerankers = {}

def register_reranker(name:str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator

def get_reranker_cls(name:str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]
