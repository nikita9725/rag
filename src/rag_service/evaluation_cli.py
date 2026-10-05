"""Оценка Дня 6 в новой временной коллекции без изменения рабочего индекса."""

import argparse
import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from rag_service.cli import index_knowledge_base
from rag_service.embeddings import LocalEmbeddingProvider, LocalModelManager
from rag_service.evaluation import (
    QualityReport,
    RetrievalConfiguration,
    RetrievalObservation,
    load_cases,
    save_report,
    select_configuration,
)
from rag_service.generation import RAGService
from rag_service.generation_content import SYSTEM_PROMPT
from rag_service.repositories import (
    LLMRepository,
    OpenAILLMRepository,
    WeaviateChunkRepository,
    connect_to_weaviate,
)
from rag_service.retrieval import RetrievalService, SearchMode
from rag_service.settings import PROJECT_ROOT, LLMSettings, Settings

logger = logging.getLogger(__name__)


class CountingLLM:
    """Подсчитать реальные вызовы, включая проверку отказа до генерации."""

    def __init__(self, repository: LLMRepository) -> None:
        self.repository = repository
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return self.repository.complete(system_prompt, user_prompt)


def run_evaluation(report_path: Path, *, generate: bool = False) -> QualityReport:
    settings = Settings()
    manager = LocalModelManager(
        settings.embedding_model_id,
        settings.embedding_model_revision,
        settings.embedding_model_path,
    )
    provider = LocalEmbeddingProvider(
        manager.ensure_downloaded(),
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    corpus = PROJECT_ROOT / "knowledge_base"
    cases = load_cases(PROJECT_ROOT / "tests/data/quality_questions.json")
    client = connect_to_weaviate(settings.weaviate_url, settings.weaviate_grpc_port)
    # UUID исключает коллизии с production, fixtures и параллельными запусками.
    name = f"KnowledgeChunkQuality{uuid4().hex}"
    repository = WeaviateChunkRepository(client, name)
    report = QualityReport(
        embedding_model=settings.embedding_model_id,
        embedding_revision=settings.embedding_model_revision,
        weaviate_version="unknown",
        corpus_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(corpus.glob("*.txt"))
        },
        cases=cases,
    )
    modes: tuple[SearchMode, ...] = ("semantic", "hybrid")
    try:
        report.weaviate_version = str(client.get_meta().get("version", "unknown"))
        for size, overlap in ((300, 60), (500, 100), (800, 160)):
            index_knowledge_base(corpus, size, overlap, provider, repository)
            retrieval = RetrievalService(provider, repository, max_distance=None)
            for mode in modes:
                for top_k in (1, 3, 5):
                    config = RetrievalConfiguration(
                        chunk_size=size, chunk_overlap=overlap, top_k=top_k, mode=mode
                    )
                    logger.info("Оценка retrieval: %s", config)
                    observations = [
                        RetrievalObservation(
                            case_id=case.id,
                            chunks=retrieval.retrieve(case.query, top_k, mode),
                        )
                        for case in cases
                    ]
                    report.raw.append((config, observations))
        report.selected = select_configuration(report)
        # Сохраняем выбор до проверки ответов, без подгонки на validation.
        save_report(report, report_path)
        config = report.selected
        logger.info("Выбрана конфигурация: %s", config)
        if generate:
            index_knowledge_base(
                corpus, config.chunk_size, config.chunk_overlap, provider, repository
            )
            llm_settings = LLMSettings()
            report.llm_model = llm_settings.llm_model
            report.system_prompt_sha256 = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
            llm = OpenAILLMRepository(llm_settings)
            try:
                observed_llm = CountingLLM(llm)
                service = RAGService(
                    RetrievalService(provider, repository, max_distance=config.max_distance),
                    observed_llm,
                )
                for case in cases:
                    previous_calls = observed_llm.calls
                    answer = service.answer(case.query, config.top_k, config.mode, config.alpha)
                    report.answers.append(
                        {
                            "case_id": case.id,
                            "llm_calls": observed_llm.calls - previous_calls,
                            "result": answer.model_dump(mode="json"),
                        }
                    )
                    save_report(report, report_path)
            finally:
                llm.close()
        return report
    finally:
        try:
            save_report(report, report_path)
        finally:
            try:
                if client.collections.exists(name):
                    client.collections.delete(name)
            finally:
                client.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Сравнить retrieval в отдельной коллекции")
    parser.add_argument("--report", type=Path, required=True, help="путь к JSON-отчёту")
    parser.add_argument(
        "--generate", action="store_true", help="также проверить реальные ответы LLM"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    report = run_evaluation(args.report, generate=args.generate)
    print(f"Выбранная конфигурация: {report.selected}")
    print(f"Отчёт: {args.report}")
