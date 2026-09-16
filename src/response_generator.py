from .config import RETRIEVAL_THRESHOLD


def generate_reply(customer_text, intent, retrieved_cases):
    """
    Generate a conservative support reply using historical evidence.

    The current implementation only generates a reply when a sufficiently
    similar historical case is available. It intentionally avoids inventing
    troubleshooting steps that are not supported by the retrieved evidence.
    """

    if not retrieved_cases:
        return None

    best_case = retrieved_cases[0]
    best_similarity = best_case["similarity"]

    if best_similarity < RETRIEVAL_THRESHOLD:
        return None

    return (
        "Thanks for reaching out. We'd be happy to help with this. "
        "Based on similar Apple Support cases, we'd like to look into "
        "the issue further. Please share any relevant device or software "
        "details so we can investigate."
    )


def generate_reply_from_results(customer_text, intent, results):
    """
    Generate a reply from retrieval results returned by HistoricalRetriever.
    """

    return generate_reply(
        customer_text=customer_text,
        intent=intent,
        retrieved_cases=results,
    )
