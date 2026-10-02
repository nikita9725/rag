"""Локальное хранение модели и стратегия построения embeddings."""

import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from rag_service.interfaces import EmbeddingProvider


class SentenceEncoder(Protocol):
    """Минимальный контракт модели, необходимый embedding provider."""

    def encode(
        self,
        inputs: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> Sequence[Sequence[float]]: ...


class LocalModelError(RuntimeError):
    """Модель отсутствует, повреждена или не может быть загружена."""


class LocalModelManager:
    """Один раз скачать snapshot модели в явный каталог проекта."""

    _REQUIRED_FILES = (
        "config.json",
        "modules.json",
        "model.safetensors",
        "1_Pooling/config.json",
    )

    def __init__(self, model_id: str, revision: str, model_path: Path) -> None:
        self.model_id = model_id
        self.revision = revision
        self.model_path = model_path

    def is_available(self) -> bool:
        """Проверить наличие файлов, необходимых Sentence Transformers."""

        return self.model_path.is_dir() and all(
            (self.model_path / relative_path).is_file() for relative_path in self._REQUIRED_FILES
        )

    def ensure_downloaded(self) -> Path:
        """Скачать runtime-файлы модели и атомарно опубликовать каталог."""

        if self.is_available():
            return self.model_path
        if self.model_path.exists():
            raise LocalModelError(
                f"Каталог модели {self.model_path} существует, но не содержит полный snapshot"
            )

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = Path(
            tempfile.mkdtemp(prefix=f".{self.model_path.name}-", dir=self.model_path.parent)
        )
        try:
            snapshot_download(
                repo_id=self.model_id,
                revision=self.revision,
                local_dir=temporary_path,
                ignore_patterns=[
                    "onnx/*",
                    "openvino/*",
                    "*.bin",
                    "*.h5",
                    "*.onnx",
                    "*.xml",
                ],
            )
            if not all(
                (temporary_path / relative_path).is_file() for relative_path in self._REQUIRED_FILES
            ):
                raise LocalModelError("Загруженный snapshot не содержит обязательные файлы")
            manifest = {
                "model_id": self.model_id,
                "revision": self.revision,
            }
            (temporary_path / ".rag-model.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(self.model_path)
        except Exception as error:
            shutil.rmtree(temporary_path, ignore_errors=True)
            if isinstance(error, LocalModelError):
                raise
            raise LocalModelError(f"Не удалось скачать модель {self.model_id}") from error
        return self.model_path


class LocalEmbeddingProvider(EmbeddingProvider):
    """Sentence Transformers adapter для multilingual E5."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        model: SentenceEncoder | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Размер embedding batch должен быть больше нуля")
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self._model = model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Кодировать passages согласно формату multilingual E5."""

        self._validate_texts(texts)
        return self._encode([f"passage: {text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        """Кодировать query согласно формату multilingual E5."""

        self._validate_texts([text])
        return self._encode([f"query: {text}"])[0]

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        encoded = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = [[float(value) for value in row] for row in encoded]
        dimensions = {len(vector) for vector in vectors}
        if len(vectors) != len(texts) or len(dimensions) != 1 or dimensions == {0}:
            raise LocalModelError("Embedding-модель вернула некорректный набор векторов")
        return vectors

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> None:
        if any(not text.strip() for text in texts):
            raise ValueError("Нельзя построить embedding пустого текста")

    def _load_model(self) -> SentenceEncoder:
        if self._model is None:
            self._model = cast(
                SentenceEncoder,
                SentenceTransformer(
                    str(self.model_path),
                    device=self.device,
                    local_files_only=True,
                ),
            )
        return self._model
