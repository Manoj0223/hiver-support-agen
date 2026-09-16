INTENTS = {
    "ios_update": "Problems with iOS or software updates, including failed updates, bugs after updating, or update-related issues.",
    "battery_charging": "Battery draining quickly, battery health issues, charging problems, or device not charging.",
    "wifi_connectivity": "Wi-Fi, Bluetooth, cellular, internet, or other connectivity problems.",
    "apps": "Problems with apps, including apps crashing, freezing, not opening, or behaving incorrectly.",
    "device_hardware": "Physical device problems such as screen, keyboard, buttons, overheating, or other hardware issues.",
    "apple_services": "Problems with Apple services such as iCloud, Apple Music, iTunes, App Store, Photos, or related services.",
    "account_payment": "Apple ID, account access, billing, subscriptions, purchases, or payment-related issues.",
    "how_to": "Questions asking how to perform a feature, setting, or task on an Apple device.",
    "other": "Messages that do not clearly belong to another intent or do not contain enough information to classify."
}


def assign_initial_intent(text):
    """
    Assign a weak initial intent using keyword-based rules.

    These labels are used for classifier training and are not treated
    as human-verified gold labels.
    """

    text = str(text).lower().strip()

    if not text:
        return "other"

    # 1. Battery / charging
    battery_patterns = [
        "battery",
        "battery life",
        "battery health",
        "battery drain",
        "battery draining",
        "dies quickly",
        "dying fast",
        "won't charge",
        "not charging",
        "doesn't charge",
        "doesnt charge",
        "charge my phone",
        "charger",
        "wireless charging",
    ]
    if any(p in text for p in battery_patterns):
        return "battery_charging"

    # 2. iOS / software updates
    ios_patterns = [
        "ios",
        "ios update",
        "software update",
        "software upgrade",
        "updated to",
        "updating to",
        "after updating",
        "after the update",
        "after update",
        "new update",
        "latest update",
        "upgrade to",
    ]
    if any(p in text for p in ios_patterns):
        return "ios_update"

    # 3. Connectivity
    connectivity_patterns = [
        "wifi",
        "wi-fi",
        "bluetooth",
        "cellular",
        "mobile data",
        "internet",
        "network",
        "connection",
        "connectivity",
        "no signal",
        "signal",
    ]
    if any(p in text for p in connectivity_patterns):
        return "wifi_connectivity"

    # 4. Apple services
    service_patterns = [
        "icloud",
        "itunes",
        "apple music",
        "app store",
        "apple pay",
        "apple pay cash",
        "imessage",
        "facetime",
        "find my iphone",
        "find my",
        "photos",
        "apple watch",
    ]
    if any(p in text for p in service_patterns):
        return "apple_services"

    # 5. Account / payment
    account_patterns = [
        "apple id",
        "password",
        "billing",
        "payment",
        "refund",
        "subscription",
        "purchase",
        "credit card",
        "account locked",
        "sign in",
        "sign-in",
        "login",
        "verification",
        "verify my account",
    ]
    if any(p in text for p in account_patterns):
        return "account_payment"

    # 6. Hardware / device
    hardware_patterns = [
        "screen",
        "display",
        "keyboard",
        "keyboard problem",
        "can't type",
        "cannot type",
        "wont type",
        "won't type",
        "autocorrect",
        "question mark",
        "question mark emoji",
        "emoji",
        "speaker",
        "camera",
        "microphone",
        "button",
        "overheating",
        "broken",
        "won't turn on",
        "wont turn on",
        "won't come on",
        "wont come on",
        "restarting",
        "resetting",
        "crackling",
    ]
    if any(p in text for p in hardware_patterns):
        return "device_hardware"

    # 7. Apps
    app_patterns = [
        "app",
        "application",
        "crash",
        "crashing",
        "freeze",
        "freezing",
        "not opening",
        "won't open",
        "wont open",
        "unable to download app",
        "apps won't install",
        "apps wont install",
    ]
    if any(p in text for p in app_patterns):
        return "apps"

    # 8. How-to
    how_to_patterns = [
        "how do i",
        "how can i",
        "how to",
        "is there a way",
        "can i change",
        "where can i",
        "where do i",
        "can i check",
    ]
    if any(p in text for p in how_to_patterns):
        return "how_to"

    return "other"


def create_weak_labels(texts):
    return texts.apply(assign_initial_intent)
