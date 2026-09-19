"""Reusable projections of handwriting logits; no model construction or UI policy.

Checkpoint temperatures reproduce the existing recognizer ensemble. They do not
establish calibration on an unknown writer, device, or unfinished stroke prefix.
"""
from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from iso_tools.inference.types import CONTRACT, ClassifierEvidence, ModelIdentity

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


@dataclass(config=CONTRACT, frozen=True)
class ModelProbability:
    model_index: int
    probability: Probability | None
    support: Literal["in_vocabulary", "not_in_vocabulary"]


@dataclass(config=CONTRACT, frozen=True)
class HandwritingCandidate:
    text: str
    probability: Probability
    model_probabilities: tuple[ModelProbability, ...]


@dataclass(config=CONTRACT, frozen=True)
class HandwritingModel:
    identity: ModelIdentity
    vocabulary_id: str
    vocabulary_size: int
    temperature: float
    top_candidate: str


@dataclass(config=CONTRACT, frozen=True)
class HandwritingEvidence:
    candidates: tuple[HandwritingCandidate, ...]
    models: tuple[HandwritingModel, ...]
    omitted_mass: Probability
    entropy_nats: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    top_margin: Probability
    common_vocabulary_js_nats: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    common_vocabulary_size: int
    preprocessing: tuple[str, ...]
    projection: Literal["temperature_probability_mean_v1"] = "temperature_probability_mean_v1"
    calibration: Literal["checkpoint_temperature_only"] = "checkpoint_temperature_only"
    out_of_vocabulary_detection: Literal["unsupported"] = "unsupported"


def _probabilities(values, temperature: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("handwriting logits must be a finite vocabulary vector")
    scaled = values / np.float32(temperature)
    exps = np.exp(scaled - scaled.max())
    return exps / exps.sum(dtype=np.float32)


def project_handwriting(evidence: ClassifierEvidence, arrays, *, limit: int = 20) -> HandwritingEvidence:
    """Match the existing mean over temperatures and union vocabularies.

Missing vocabulary classes contribute no ensemble mass, matching the historical
projection, but retain explicit missing support instead of a fake model score.
Jensen-Shannon disagreement is measured only on the shared class support and
is not an independent-witness or correctness estimate.
"""
    if not 1 <= limit <= 100 or not evidence.models:
        raise ValueError("requires models and a candidate limit in 1..100")
    vocab: list[str] = []
    indexes: dict[str, int] = {}
    distributions: list[dict[str, float]] = []
    model_records = []
    for item in evidence.models:
        probs = _probabilities(arrays[item.tensor], item.temperature)
        if len(probs) != len(item.vocabulary.tokens):
            raise ValueError("handwriting vocabulary and logits disagree")
        if any(len(token) != 1 for token in item.vocabulary.tokens):
            raise ValueError("handwriting vocabulary must contain Unicode characters")
        distribution = dict(zip(item.vocabulary.tokens, (float(x) for x in probs)))
        distributions.append(distribution)
        for token in item.vocabulary.tokens:
            if token not in indexes:
                indexes[token] = len(vocab)
                vocab.append(token)
        model_records.append(HandwritingModel(item.model, item.vocabulary.id,
            len(probs), item.temperature, item.vocabulary.tokens[int(probs.argmax())]))
    combined = np.zeros(len(vocab), dtype=np.float32)
    for distribution in distributions:
        for token, probability in distribution.items():
            combined[indexes[token]] += np.float32(probability)
    combined /= np.float32(len(distributions))
    # Stable vocabulary order is the tie-breaker; ordinary non-tied ordering is
    # numerically equivalent to the existing torch projection.
    order = np.argsort(-combined, kind="stable")
    selected = order[:limit]
    candidates = tuple(HandwritingCandidate(vocab[i], float(combined[i]), tuple(
        ModelProbability(j, distribution.get(vocab[i]),
            "in_vocabulary" if vocab[i] in distribution else "not_in_vocabulary")
        for j, distribution in enumerate(distributions))) for i in selected)
    common = set(distributions[0]).intersection(*(set(d) for d in distributions[1:]))
    disagreement = None
    if len(distributions) > 1 and common:
        shared = sorted(common)
        conditioned = np.asarray([[d[token] for token in shared] for d in distributions], dtype=np.float64)
        masses = conditioned.sum(axis=1, keepdims=True)
        if (masses > 0).all():
            conditioned /= masses
            mean = conditioned.mean(axis=0)
            positive = conditioned > 0
            terms = np.zeros_like(conditioned)
            terms[positive] = conditioned[positive] * np.log((conditioned / np.maximum(mean, 1e-300))[positive])
            disagreement = max(0.0, float(terms.sum(axis=1).mean()))
    positive = combined > 0
    entropy = max(0.0, float(-(combined[positive].astype(np.float64) * np.log(combined[positive])).sum()))
    return HandwritingEvidence(candidates, tuple(model_records),
        max(0.0, min(1.0, 1.0 - float(combined[selected].sum(dtype=np.float64)))),
        entropy, float(combined[order[0]] - (combined[order[1]] if len(order) > 1 else 0)),
        disagreement, len(common), evidence.preprocessing)
