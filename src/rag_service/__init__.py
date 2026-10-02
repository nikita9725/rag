"""Компоненты учебного RAG-сервиса."""

from rag_service.loader import clean_text, load_documents
from rag_service.schemas import Document

__all__ = ["Document", "clean_text", "load_documents"]
