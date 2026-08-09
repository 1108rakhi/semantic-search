import pytest
import numpy as np
from unittest.mock import MagicMock
from semantic_search import SemanticSearch


@pytest.fixture
def search_engine():
    engine = SemanticSearch()
    engine.model = MagicMock()
    return engine


def test_add_documents(search_engine):
    search_engine.model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
    search_engine.add_documents(["doc a", "doc b"])
    assert search_engine.documents == ["doc a", "doc b"]
    assert search_engine.embeddings.shape == (2, 2)


def test_search_returns_top_k(search_engine):
    search_engine.embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    search_engine.documents = ["doc a", "doc b", "doc c"]
    search_engine.model.encode.return_value = np.array([1.0, 0.0])

    results = search_engine.search("query", top_k=2)
    assert len(results) == 2
    assert results[0][0] == "doc a"  # most similar to [1,0]


def test_search_without_documents_raises(search_engine):
    with pytest.raises(ValueError):
        search_engine.search("query")