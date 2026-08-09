from sentence_transformers import SentenceTransformer
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The cat sat on the mat.",
    "A dog was lying on the rug.",
    "I love pizza.",
]

embeddings = model.encode(sentences)

# Compare sentence 0 vs 1 (should be fairly similar - both about pets on furniture)
print("Cat/mat vs Dog/rug:", cosine_similarity(embeddings[0], embeddings[1]))

# Compare sentence 0 vs 2 (should be less similar - unrelated topics)
print("Cat/mat vs Pizza:", cosine_similarity(embeddings[0], embeddings[2]))

from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
print(sklearn_cosine([embeddings[0]], [embeddings[1]]))