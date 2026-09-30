from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

documents = [
    "Paris is the capital city of France.",
    "Berlin is the capital city of Germany.",
    "London is the capital city of the United Kingdom.",
    "Delhi is the capital city of India.",
    "Tokyo is the capital city of Japan.",
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

document_embeddings = model.encode(documents, normalize_embeddings=True)

document_embeddings = np.array(document_embeddings).astype("float32")

dimension = document_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(document_embeddings)

query = "What is the capital of France?"

query_embedding = model.encode([query], normalize_embeddings=True)

query_embedding = np.array(query_embedding).astype("float32")

scores, indices = index.search(query_embedding, 3)

print("\nSearch results:\n")

for score, index_number in zip(scores[0], indices[0]):
    print(f"Score: {score:.4f}")
    print(documents[index_number])
    print()
