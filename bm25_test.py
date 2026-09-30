from rank_bm25 import BM25Okapi

documents = [
    "Paris is the capital city of France.",
    "Berlin is the capital city of Germany.",
    "London is the capital city of the United Kingdom.",
    "Delhi is the capital city of India.",
    "Tokyo is the capital city of Japan.",
]

tokenized_documents = [document.lower().split() for document in documents]

bm25 = BM25Okapi(tokenized_documents)

query = "capital France"

query_tokens = query.lower().split()

scores = bm25.get_scores(query_tokens)

ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

print("\nBM25 results:\n")

for i in ranked_indices[:3]:
    print(scores[i])
    print(documents[i])
    print()
