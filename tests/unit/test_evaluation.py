"""Проверки выбора конфигурации без использования проверочных результатов."""

from rag_service.evaluation import (
    ConfigurationEvaluation,
    ExpectedFact,
    QualityCase,
    QualityMetrics,
    RetrievalConfiguration,
    RetrievalObservation,
    measure,
    selection_key,
)
from rag_service.schemas import ChunkMetadata, ChunkSearchResult


def configuration(threshold: float = 0.16) -> RetrievalConfiguration:
    return RetrievalConfiguration(
        chunk_size=500, chunk_overlap=100, top_k=3, mode="semantic", max_distance=threshold
    )


def test_selection_ignores_validation_and_prefers_facts_then_refusals() -> None:
    good = ConfigurationEvaluation(
        configuration=configuration(),
        tune=QualityMetrics(facts_found=5, off_topic_refusals=1),
        validation=QualityMetrics(facts_found=0),
    )
    bad = good.model_copy(
        update={
            "tune": QualityMetrics(facts_found=4, off_topic_refusals=2),
            "validation": QualityMetrics(facts_found=100),
        }
    )
    assert selection_key(good) > selection_key(bad)
    too_loose = good.model_copy(update={"configuration": configuration(0.2)})
    assert selection_key(good) > selection_key(too_loose)


def test_measure_checks_fact_and_source_and_filters_before_counting() -> None:
    fact = ExpectedFact(source_name="doc.txt", fragment="нужный факт")
    cases = [
        QualityCase(
            id="fact", split="tune", kind="answerable", query="q", facts=[fact], forbidden_claims=[]
        ),
        QualityCase(
            id="outside",
            split="tune",
            kind="unanswerable",
            query="outside",
            facts=[],
            off_topic=True,
            forbidden_claims=[],
        ),
    ]
    chunk = ChunkSearchResult(
        uuid="one",
        content="нужный факт",
        distance=0.16,
        metadata=ChunkMetadata(document_id="doc", source_name="doc.txt", chunk_id=0),
    )
    weak = chunk.model_copy(update={"distance": 0.3})
    observations = [
        RetrievalObservation(case_id="fact", chunks=[chunk, weak]),
        RetrievalObservation(case_id="outside", chunks=[weak]),
    ]
    result = measure(cases, observations, 0.16, "tune")
    assert result.facts_found == result.facts_total == 1
    assert result.retained_chunks == 1 and result.noise_chunks == 0
    assert result.off_topic_refusals == 1
    wrong_source = chunk.model_copy(
        update={"metadata": chunk.metadata.model_copy(update={"source_name": "wrong.txt"})}
    )
    observations[0] = RetrievalObservation(case_id="fact", chunks=[wrong_source])
    assert measure(cases, observations, 0.16, "tune").facts_found == 0
