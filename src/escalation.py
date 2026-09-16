import re

from .config import RETRIEVAL_THRESHOLD


ACCOUNT_PAYMENT_KEYWORDS = [
    "apple id",
    "password",
    "billing",
    "charged",
    "charge",
    "payment",
    "refund",
    "subscription",
    "purchase",
    "credit card",
    "account locked",
    "can't sign in",
    "cannot sign in",
]


def should_escalate(customer_text, best_similarity):
    """
    Decide whether a customer message should be escalated.

    Escalation is triggered when:
    1. The message contains potentially sensitive account/payment content.
    2. The message has insufficient context.
    3. No sufficiently similar historical resolution is available.
    """

    text = str(customer_text).lower().strip()

    # Sensitive account/payment issues
    for keyword in ACCOUNT_PAYMENT_KEYWORDS:
        if keyword in text:
            return True, "Sensitive account or payment issue requires human support."

    # Very short / ambiguous messages
    cleaned = re.sub(r"[^a-z\s]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if len(cleaned) < 18:
        return True, "Insufficient context to safely handle automatically."

    # No sufficiently similar historical case
    if best_similarity < RETRIEVAL_THRESHOLD:
        return True, "No sufficiently similar historical case was found."

    return False, "Sufficient historical evidence is available for automated handling."
