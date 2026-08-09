from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticSearch:
    """A simple semantic search engine using sentence embeddings
    and cosine similarity."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents: list[str] = []
        self.embeddings: np.ndarray | None = None

    def add_documents(self, texts: list[str]) -> None:
        """
        Embed and store a list of documents for later search.

        Args:
            texts: List of document strings to index.
        """
        self.documents = texts
        self.embeddings = self.model.encode(texts)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """
        Search stored documents for the closest matches to a query.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            List of (document, similarity_score) tuples, sorted by
            descending similarity.
        """
        if self.embeddings is None:
            raise ValueError("No documents indexed. Call add_documents() first.")

        query_embedding = self.model.encode(query)

        scores = [
            self._cosine_similarity(query_embedding, doc_embedding)
            for doc_embedding in self.embeddings
        ]

        ranked = sorted(zip(self.documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

if __name__ == "__main__":
    search = SemanticSearch()

    docs = [
        "The cat sat on the mat.",
        "A dog was lying on the rug.",
        "I love pizza with extra cheese.",
        "Python is a popular programming language.",
        "Machine learning models can classify images.",
        "The weather is sunny and warm today.",
    ]

    search.add_documents(docs)

    results = search.search("What did the pet do?", top_k=3)
    for doc, score in results:
        print(f"{score:.4f} — {doc}")