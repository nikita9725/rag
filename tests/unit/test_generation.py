import json
from unittest.mock import Mock

import pytest

from rag_service.generation import RAGService, answer_without_retrieval
from rag_service.interfaces import EmbeddingProvider
from rag_service.repositories import ChunkRepository, LLMError, LLMRepository
from rag_service.retrieval import RetrievalService
from rag_service.schemas import ChunkMetadata, ChunkSearchResult


def dependencies() -> tuple[Mock, Mock]:
    retrieval = Mock(spec=RetrievalService)
    retrieval.retrieve.return_value = [
        ChunkSearchResult(
            uuid=f"uuid-{i}",
            content=f"Факт {i}",
            metadata=ChunkMetadata(document_id="doc", source_name="doc.txt", chunk_id=i),
        )
        for i in range(2)
    ]
    llm = Mock(spec=LLMRepository)
    llm.complete.return_value = json.dumps(
        {"answer": "Факт [2]", "source_ids": [2, 2], "insufficient_context": False}
    )
    return retrieval, llm


def test_context_and_sources() -> None:
    retrieval, llm = dependencies()
    answer = RAGService(retrieval, llm).answer(" вопрос ", 2, "hybrid", 0.7)
    retrieval.retrieve.assert_called_once_with("вопрос", 2, "hybrid", 0.7)
    system, user = llm.complete.call_args.args
    assert "Запрещено придумывать" in system
    context = json.loads(user)
    assert context["question"] == "вопрос"
    assert [chunk["text"] for chunk in context["context"]] == ["Факт 0", "Факт 1"]
    assert answer.source_ids == (2,)
    assert answer.sources[0].uuid == "uuid-1"
    assert len(answer.context) == 2


def test_empty_context_skips_llm() -> None:
    retrieval, llm = dependencies()
    retrieval.retrieve.return_value = []
    answer = RAGService(retrieval, llm).answer("вопрос")
    assert answer.insufficient_context and not answer.sources
    llm.complete.assert_not_called()


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"answer":"факт [99]","source_ids":[1],"insufficient_context":false}',
        '{"answer":" ","source_ids":[1],"insufficient_context":false}',
        '{"answer":"факт","source_ids":[3],"insufficient_context":false}',
        '{"answer":"факт","source_ids":[],"insufficient_context":false}',
        '{"answer":"факт","source_ids":[true],"insufficient_context":false}',
    ],
)
def test_invalid_model_answer(raw: str) -> None:
    retrieval, llm = dependencies()
    llm.complete.return_value = raw
    with pytest.raises(LLMError):
        RAGService(retrieval, llm).answer("вопрос")


def test_nonempty_irrelevant_context_can_refuse() -> None:
    retrieval, llm = dependencies()
    llm.complete.return_value = json.dumps(
        {"answer": "Недостаточно сведений", "source_ids": [], "insufficient_context": True}
    )
    answer = RAGService(retrieval, llm).answer("вопрос")
    assert answer.insufficient_context and not answer.sources


def test_invalid_query_skips_dependencies() -> None:
    retrieval, llm = dependencies()
    with pytest.raises(ValueError):
        RAGService(retrieval, llm).answer(" ")
    retrieval.retrieve.assert_not_called()
    llm.complete.assert_not_called()


def test_baseline_has_no_context() -> None:
    _, llm = dependencies()
    llm.complete.return_value = '{"answer":"Обычный ответ"}'
    assert answer_without_retrieval("вопрос", llm) == "Обычный ответ"
    assert json.loads(llm.complete.call_args.args[1]) == {"question": "вопрос"}
    llm.complete.return_value = "bad json"
    with pytest.raises(LLMError):
        answer_without_retrieval("вопрос", llm)


def test_partial_answer_requires_valid_sources_for_citations() -> None:
    retrieval, llm = dependencies()
    llm.complete.return_value = json.dumps(
        {
            "answer": "Известен факт [1], остального в контексте нет",
            "source_ids": [1],
            "insufficient_context": True,
        }
    )
    result = RAGService(retrieval, llm).answer("вопрос")
    assert result.insufficient_context and result.source_ids == (1,)


def test_full_refusal_restores_known_citation() -> None:
    retrieval, llm = dependencies()
    llm.complete.return_value = json.dumps(
        {"answer": "Нет сведений [1]", "source_ids": [], "insufficient_context": True}
    )
    result = RAGService(retrieval, llm).answer("вопрос")
    assert result.insufficient_context and result.source_ids == (1,)


@pytest.mark.parametrize("keep", [False, True])
def test_filtered_context_skips_llm_or_renumbers_sources(keep: bool) -> None:
    provider = Mock(spec=EmbeddingProvider)
    repository = Mock(spec=ChunkRepository)
    repository.search.return_value = [
        ChunkSearchResult(
            uuid="weak",
            content="Недостоверный контекст",
            metadata=ChunkMetadata(document_id="doc", source_name="doc.txt", chunk_id=0),
            distance=0.5,
        ),
        ChunkSearchResult(
            uuid="good",
            content="Подтверждённый факт",
            metadata=ChunkMetadata(document_id="doc", source_name="doc.txt", chunk_id=1),
            distance=0.1 if keep else 0.5,
        ),
    ]
    llm = Mock(spec=LLMRepository)
    llm.complete.return_value = (
        '{"answer":"Подтверждённый факт [1]","source_ids":[1],"insufficient_context":false}'
    )
    result = RAGService(RetrievalService(provider, repository, max_distance=0.2), llm).answer("q")
    if not keep:
        assert result.insufficient_context and not result.sources and not result.context
        llm.complete.assert_not_called()
    else:
        prompt = json.loads(llm.complete.call_args.args[1])
        assert [chunk["source_id"] for chunk in prompt["context"]] == [1]
        assert [chunk["text"] for chunk in prompt["context"]] == ["Подтверждённый факт"]
        assert result.sources[0].uuid == "good" and result.source_ids == (1,)
