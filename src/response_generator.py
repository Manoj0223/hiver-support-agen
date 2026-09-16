from .config import RETRIEVAL_THRESHOLD


def generate_reply(customer_text, intent, retrieved_cases):
    """
    Generate a conservative support reply grounded in
    the highest-similarity historical Apple Support response.
    """

    if not retrieved_cases:
        return None

    best_case = retrieved_cases[0]

    best_similarity = best_case.get("similarity", 0.0)
    historical_response = best_case.get("support_response")

    if best_similarity < RETRIEVAL_THRESHOLD:
        return None

    if not historical_response:
        return None

    historical_response = str(historical_response).strip()

    if not historical_response:
        return None

    return (
        "Thanks for reaching out. Based on a similar Apple Support case, "
        "here is the relevant guidance:\n\n"
        f"{historical_response}\n\n"
        "If the issue persists, please share your device model and "
        "software version so the issue can be investigated further."
    )


def generate_reply_from_results(customer_text, intent, results):
    """
    Generate a reply from retrieved historical cases.
    """

    return generate_reply(
        customer_text=customer_text,
        intent=intent,
        retrieved_cases=results,
    )
