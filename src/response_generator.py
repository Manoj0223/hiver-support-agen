import pandas as pd

from .config import RETRIEVAL_THRESHOLD


def generate_reply(customer_text, intent, retrieved_cases):
    """
    Generate a conservative support reply using retrieved historical cases.

    The current version intentionally avoids inventing troubleshooting steps.
    It only generates a response when sufficiently similar historical evidence
    is available.
    """

    if retrieved_cases is None or len(retrieved_cases) == 0:
        return None

    best_case = retrieved_cases.iloc[0]

    if best_case["similarity"] < RETRIEVAL_THRESHOLD:
        return None

    return (
        "Thanks for reaching out. We'd be happy to help with this. "
        "Based on similar Apple Support cases, we'd like to look into "
        "the issue further. Please share any relevant device or software "
        "details so we can investigate."
    )


def generate_reply_from_results(customer_text, intent, results):
    """
    Adapter for retrieval results returned as a list of dictionaries.
    """

    if not results:
        return None

    best_similarity = results[0]["similarity"]

    if best_similarity < RETRIEVAL_THRESHOLD:
        return None

    return (
        "Thanks for reaching out. We'd be happy to help with this. "
        "Based on similar Apple Support cases, we'd like to look into "
        "the issue further. Please share any relevant device or software "
        "details so we can investigate."
    )
