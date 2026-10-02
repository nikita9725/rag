from pathlib import Path

import pytest

from rag_service.embeddings import LocalEmbeddingProvider, LocalModelError, LocalModelManager


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(
        self,
        inputs: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        options: dict[str, object] = {
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
            "convert_to_numpy": convert_to_numpy,
            "show_progress_bar": show_progress_bar,
        }
        self.calls.append((inputs, options))
        return [[1.0, float(index)] for index, _ in enumerate(inputs)]


class InvalidShapeModel(FakeModel):
    def encode(
        self,
        inputs: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        return [[1.0], []]


def test_provider_uses_e5_prefixes_and_normalization(tmp_path: Path) -> None:
    model = FakeModel()
    provider = LocalEmbeddingProvider(tmp_path, batch_size=4, model=model)

    documents = provider.embed_documents(["Первый", "Второй"])
    query = provider.embed_query("Вопрос")

    assert documents == [[1.0, 0.0], [1.0, 1.0]]
    assert query == [1.0, 0.0]
    assert model.calls[0][0] == ["passage: Первый", "passage: Второй"]
    assert model.calls[1][0] == ["query: Вопрос"]
    assert model.calls[0][1]["normalize_embeddings"] is True
    assert model.calls[0][1]["batch_size"] == 4


def test_provider_rejects_empty_text(tmp_path: Path) -> None:
    provider = LocalEmbeddingProvider(tmp_path, model=FakeModel())

    with pytest.raises(ValueError, match="пустого"):
        provider.embed_documents([" "])


def test_model_manager_recognizes_complete_local_model(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    for relative_path in LocalModelManager._REQUIRED_FILES:
        path = model_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    manager = LocalModelManager("model-id", "revision", model_path)

    assert manager.is_available()
    assert manager.ensure_downloaded() == model_path


def test_model_manager_rejects_incomplete_model_directory(tmp_path: Path) -> None:
    model_path = tmp_path / "incomplete"
    model_path.mkdir()
    manager = LocalModelManager("model-id", "revision", model_path)

    with pytest.raises(LocalModelError, match="полный snapshot"):
        manager.ensure_downloaded()


def test_provider_rejects_invalid_vector_shape(tmp_path: Path) -> None:
    provider = LocalEmbeddingProvider(tmp_path, model=InvalidShapeModel())

    with pytest.raises(LocalModelError, match="некорректный"):
        provider.embed_documents(["one", "two"])
