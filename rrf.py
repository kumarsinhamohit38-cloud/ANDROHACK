def reciprocal_rank_fusion(rankings, k=60):
    scores = {}

    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):

            if document_id not in scores:
                scores[document_id] = 0

            scores[document_id] += 1 / (k + rank)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_results
