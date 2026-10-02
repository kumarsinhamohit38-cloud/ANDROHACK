"""Build short, extractive answers from retrieved passages."""

import re


_ANSWER_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "all", "also", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "but",
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "had", "has", "have", "having", "he", "her", "here",
    "how", "i", "if", "in", "into", "is", "it", "its", "me", "more", "most",
    "my", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "our", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "would", "you", "your", "from", "between", "different", "difference", "compare",
    "compared", "versus", "vs", "explain", "describe", "tell", "give", "show", "please",
})
_FACET_SPLIT_RE = re.compile(
    r"\b(?:and|but|also|whereas|while|versus|vs|different from|differs? from|compared with|compared to)\b|[;?]+",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _normalize_word(word: str) -> str:
    """Lightly normalize common English plural and noun endings."""
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 7 and word.endswith("ity"):
        return word[:-3]
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _terms(text: str) -> list[str]:
    return [
        _normalize_word(word)
        for word in _WORD_RE.findall(text.lower())
        if word not in _ANSWER_STOPWORDS and len(word) > 1
    ]


def _query_facets(query: str) -> list[tuple[str, list[str]]]:
    """Split a compound question into answerable topic facets."""
    facets = []
    for clause in _FACET_SPLIT_RE.split(query.strip()):
        display_words = [
            word for word in _WORD_RE.findall(clause.lower())
            if word not in _ANSWER_STOPWORDS and len(word) > 1
        ]
        if not display_words:
            continue
        # Preserve order while removing repeated query terms.
        display_words = list(dict.fromkeys(display_words))
        words = list(dict.fromkeys(_normalize_word(word) for word in display_words))
        label = " ".join(display_words).capitalize()
        facets.append((label, words))
    if len(facets) == 1:
        facets[0] = ("Answer", facets[0][1])
    return facets or [("Answer", [])]


def _best_sentence(terms: list[str], results: list[dict]) -> tuple[str, dict | None]:
    best_text = ""
    best_result = None
    best_score = 0.0

    for result_rank, result in enumerate(results, start=1):
        for sentence in _SENTENCE_RE.split(str(result.get("text", ""))):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_terms = set(_terms(sentence))
            overlap = set(terms) & sentence_terms
            if terms and not overlap:
                continue

            coverage = len(overlap) / len(set(terms)) if terms else 0.0
            # A passage matching only a small fragment of a multi-term facet
            # is weak evidence and should not be presented as an answer.
            if len(set(terms)) >= 2 and coverage < 0.5:
                continue
            is_definition = bool(re.search(
                r"\b(?:is|are|means|describes|refers to|defined as|occurs when)\b",
                sentence,
                re.IGNORECASE,
            ))
            starts_with_topic = bool(terms and any(
                _normalize_word(word) in sentence_terms
                for word in terms[:2]
            ))
            score = coverage * 3.0 + (0.7 if is_definition else 0.0)
            if starts_with_topic:
                score += 0.2
            score += max(0.0, 0.05 - result_rank * 0.005)

            if score > best_score:
                best_text = sentence
                best_result = result
                best_score = score

    return best_text, best_result


def compose_grounded_answer(query: str, results: list[dict]) -> list[dict]:
    """Select one evidence sentence for each question facet without inventing facts."""
    facets = _query_facets(query)
    answer = []
    for label, terms in facets:
        sentence, result = _best_sentence(terms, results)
        answer.append({
            "label": label,
            "text": sentence or "No supporting passage for this part was found in the current library.",
            "metadata": (result or {}).get("metadata", {}) or {},
            "supported": bool(sentence),
        })
    return answer
