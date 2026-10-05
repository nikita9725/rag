"""День 6: 18 конфигураций retrieval и реальные ответы на 12 вопросов."""

import os
from pathlib import Path

import pytest

from rag_service.evaluation import selection_key
from rag_service.evaluation_cli import run_evaluation
from rag_service.generation import RAGAnswer

pytestmark = pytest.mark.e2e


def test_quality_matrix_and_grounded_answers(tmp_path: Path) -> None:
    path = Path(os.environ.get("DAY06_REPORT_PATH", str(tmp_path / "quality.json")))
    report = run_evaluation(path, generate=True)
    assert len(report.raw) == 18
    assert len(report.comparisons) == 18 * 32
    assert len(report.answers) == len(report.cases) == 12
    selected = max(report.comparisons, key=selection_key)
    assert selected.configuration == report.selected
    assert selected.configuration.max_distance is not None
    # Проверочная половина не участвует в выборе параметров.
    assert selected.tune.facts_found == selected.tune.facts_total
    assert selected.validation.facts_found == selected.validation.facts_total
    assert selected.validation.off_topic_refusals == selected.validation.off_topic_total
    cases = {case.id: case for case in report.cases}
    for observation in report.answers:
        case = cases[str(observation["case_id"])]
        answer = RAGAnswer.model_validate(observation["result"])
        assert answer.insufficient_context == (case.kind != "answerable"), case.id
        assert observation["llm_calls"] == int(bool(answer.context)), case.id
        if case.kind == "unanswerable":
            assert not answer.sources and not answer.source_ids, case.id
        else:
            expected_sources = {fact.source_name for fact in case.facts}
            actual_sources = {chunk.metadata.source_name for chunk in answer.sources}
            assert expected_sources <= actual_sources, case.id
        assert all(chunk in answer.context for chunk in answer.sources), case.id
        assert all(
            chunk.distance is not None and chunk.distance <= selected.configuration.max_distance
            for chunk in answer.context
        ), case.id
