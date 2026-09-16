from .config import RETRIEVAL_THRESHOLD


ACCOUNT_PAYMENT_KEYWORDS = [
    "apple id",
    "password",
    "billing",
    "payment",
    "charged",
    "charge",
    "refund",
    "subscription",
    "purchase",
    "credit card",
    "account locked",
    "can't sign in",
    "cannot sign in",
    "sign in",
    "login",
]


def should_escalate(customer_text, best_similarity):
    """
    Decide whether the message should be handled automatically
    or escalated to a human.

    Escalation conditions:
    1. Potentially sensitive account/payment issue.
    2. Very low-context customer message.
    3. No sufficiently similar historical case.
    """

    text = str(customer_text).lower().strip()

    # 1. Sensitive account/payment issue
    for keyword in ACCOUNT_PAYMENT_KEYWORDS:
        if keyword in text:
            return (
                True,
                "Sensitive account or payment issue requires human support.",
            )

    # 2. Insufficient context
    if len(text) < 18:
        return (
            True,
            "Insufficient context to safely handle automatically.",
        )

    # 3. Insufficient historical evidence
    if best_similarity < RETRIEVAL_THRESHOLD:
        return (
            True,
            "No sufficiently similar historical case was found.",
        )

    return (
        False,
        "Sufficient historical evidence is available for automated handling.",
    )
