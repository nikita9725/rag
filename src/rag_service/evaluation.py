"""Воспроизводимая оценка retrieval; разметка фактов независима от LLM."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rag_service.retrieval import SearchMode
from rag_service.schemas import ChunkSearchResult


class ExpectedFact(BaseModel):
    source_name: str
    fragment: str


class QualityCase(BaseModel):
    id: str
    split: Literal["tune", "validation"]
    kind: Literal["answerable", "unanswerable", "partial"]
    query: str
    facts: list[ExpectedFact]
    off_topic: bool = False
    forbidden_claims: list[str]


class RetrievalConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)
    chunk_size: int
    chunk_overlap: int
    top_k: int
    mode: SearchMode
    alpha: float = 0.5
    max_distance: float | None = None


class RetrievalObservation(BaseModel):
    case_id: str
    chunks: list[ChunkSearchResult]


class QualityMetrics(BaseModel):
    facts_found: int = 0
    facts_total: int = 0
    off_topic_refusals: int = 0
    off_topic_total: int = 0
    noise_chunks: int = 0
    retained_chunks: int = 0
    context_characters: int = 0
    all_fact_cases: int = 0
    fact_cases: int = 0


class ConfigurationEvaluation(BaseModel):
    configuration: RetrievalConfiguration
    tune: QualityMetrics
    validation: QualityMetrics


class QualityReport(BaseModel):
    embedding_model: str
    embedding_revision: str
    weaviate_version: str
    corpus_sha256: dict[str, str]
    cases: list[QualityCase]
    raw: list[tuple[RetrievalConfiguration, list[RetrievalObservation]]] = Field(
        default_factory=list
    )
    comparisons: list[ConfigurationEvaluation] = Field(default_factory=list)
    selected: RetrievalConfiguration | None = None
    llm_model: str | None = None
    system_prompt_sha256: str | None = None
    answers: list[dict[str, object]] = Field(default_factory=list)


def load_cases(path: Path) -> list[QualityCase]:
    cases = [
        QualityCase.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))
    ]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Повторяющиеся id в вопросах качества")
    if not any(case.split == "tune" and case.facts for case in cases):
        raise ValueError("Нужны вопросы с ожидаемыми фактами для подбора")
    return cases


def accepted_chunks(
    chunks: Sequence[ChunkSearchResult], max_distance: float | None
) -> list[ChunkSearchResult]:
    return [
        chunk
        for chunk in chunks
        if max_distance is None or (chunk.distance is not None and chunk.distance <= max_distance)
    ]


def fact_matches(fact: ExpectedFact, chunk: ChunkSearchResult) -> bool:
    return fact.source_name == chunk.metadata.source_name and fact.fragment in chunk.content


def measure(
    cases: Sequence[QualityCase],
    observations: Sequence[RetrievalObservation],
    max_distance: float | None,
    split: Literal["tune", "validation"],
) -> QualityMetrics:
    by_id = {observation.case_id: observation for observation in observations}
    metrics = QualityMetrics()
    for case in cases:
        if case.split != split:
            continue
        chunks = accepted_chunks(by_id[case.id].chunks, max_distance)
        found = sum(any(fact_matches(fact, chunk) for chunk in chunks) for fact in case.facts)
        metrics.facts_found += found
        metrics.facts_total += len(case.facts)
        metrics.fact_cases += bool(case.facts)
        metrics.all_fact_cases += bool(case.facts) and found == len(case.facts)
        metrics.off_topic_total += case.off_topic
        metrics.off_topic_refusals += case.off_topic and not chunks
        # Proxy шума: нет размеченного факта. Тематические фрагменты могут быть полезны;
        # окончательная оценка релевантности и дубликатов остаётся качественной.
        metrics.noise_chunks += sum(
            not any(fact_matches(fact, chunk) for fact in case.facts) for chunk in chunks
        )
        metrics.retained_chunks += len(chunks)
        metrics.context_characters += sum(len(chunk.content) for chunk in chunks)
    return metrics


def selection_key(item: ConfigurationEvaluation) -> tuple[int, int, int, int, int, float]:
    metrics = item.tune
    config = item.configuration
    current = (config.chunk_size, config.chunk_overlap, config.top_k, config.mode) == (
        500,
        100,
        3,
        "semantic",
    )
    return (
        metrics.facts_found,
        metrics.off_topic_refusals,
        -metrics.noise_chunks,
        -metrics.context_characters,
        int(current),
        -(config.max_distance if config.max_distance is not None else 2.01),
    )


def select_configuration(report: QualityReport) -> RetrievalConfiguration:
    """Выбор использует только tune; validation не участвует в сортировке."""
    thresholds: list[float | None] = [None, *[value / 100 for value in range(10, 41)]]
    for configuration, observations in report.raw:
        for threshold in thresholds:
            report.comparisons.append(
                ConfigurationEvaluation(
                    configuration=configuration.model_copy(update={"max_distance": threshold}),
                    tune=measure(report.cases, observations, threshold, "tune"),
                    validation=measure(report.cases, observations, threshold, "validation"),
                )
            )
    return max(report.comparisons, key=selection_key).configuration


def save_report(report: QualityReport, path: Path) -> None:
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
