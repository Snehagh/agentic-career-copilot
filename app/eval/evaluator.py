"""
LLM-as-judge evaluator — mock implementation.

Scores answer relevancy and faithfulness using keyword overlap heuristics.
Swap `_score_with_heuristic` for an OpenAI call to go live with zero other changes.
"""


def evaluate_response(
    query: str,
    actual_output: str,
    retrieval_context: list[str],
) -> dict:
    """Return relevancy and faithfulness scores (0.0–1.0)."""
    relevancy, rel_reason = _score_relevancy(query, actual_output)
    faithfulness, faith_reason = _score_faithfulness(actual_output, retrieval_context)
    return {
        "answer_relevancy": round(relevancy, 3),
        "faithfulness": round(faithfulness, 3),
        "relevancy_reason": rel_reason,
        "faithfulness_reason": faith_reason,
    }


# ---------------------------------------------------------------------------
# Heuristic scorers (replace with LLM calls when API key is available)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return {w.lower().strip(".,;:") for w in text.split() if len(w) > 3}


def _score_relevancy(query: str, output: str) -> tuple[float, str]:
    q_tokens = _tokenize(query)
    o_tokens = _tokenize(output)
    if not q_tokens:
        return 0.5, "Query was empty; defaulting to 0.5."
    overlap = len(q_tokens & o_tokens) / len(q_tokens)
    score = min(1.0, 0.5 + overlap)
    reason = (
        f"{len(q_tokens & o_tokens)}/{len(q_tokens)} query keywords present in the response."
    )
    return score, reason


def _score_faithfulness(output: str, context: list[str]) -> tuple[float, str]:
    if not context:
        return 0.5, "No retrieval context provided; defaulting to 0.5."
    context_text = " ".join(context)
    c_tokens = _tokenize(context_text)
    o_tokens = _tokenize(output)
    if not o_tokens:
        return 0.5, "Output was empty; defaulting to 0.5."
    grounded = len(o_tokens & c_tokens) / len(o_tokens)
    score = min(1.0, 0.4 + grounded)
    reason = (
        f"{len(o_tokens & c_tokens)}/{len(o_tokens)} output tokens found in the retrieved context."
    )
    return score, reason
