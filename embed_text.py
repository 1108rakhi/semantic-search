from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The cat sat on the mat.",
    "A dog was lying on the rug.",
    "I love pizza.",
]

embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, embeddings):
    print(sentence, "-> shape:", embedding.shape)
print(embeddings[0][:10])  # first 10 numbers of the first sentence's vector