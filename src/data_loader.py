import glob
import re
import pandas as pd


BRAND = "AppleSupport"


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"&amp;", "and", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_dataset(data_path=None):

    if data_path is None:
        files = glob.glob("data/raw/*.csv")

        if not files:
            raise FileNotFoundError(
                "No CSV file found in data/raw/"
            )

        data_path = files[0]

    return pd.read_csv(data_path)


def build_brand_pairs(df, brand=BRAND):

    apple_replies = df[
        (df["inbound"] == False) &
        (df["author_id"].astype(str) == brand)
    ].copy()

    customer_messages = df[
        df["inbound"] == True
    ][["tweet_id", "text", "inbound"]].copy()

    pairs = apple_replies.merge(
        customer_messages,
        left_on="in_response_to_tweet_id",
        right_on="tweet_id",
        how="inner",
        suffixes=("_reply", "_customer")
    )

    pairs = pairs[
        pairs["inbound_customer"] == True
    ].copy()

    pairs = pairs.rename(columns={
        "tweet_id_customer": "customer_tweet_id",
        "text_customer": "customer_text",
        "text_reply": "support_response"
    })

    pairs = pairs[
        [
            "customer_tweet_id",
            "customer_text",
            "support_response"
        ]
    ].dropna()

    pairs["clean_customer_text"] = pairs[
        "customer_text"
    ].apply(clean_text)

    return pairs.reset_index(drop=True)


def load_apple_support_pairs(data_path=None):

    df = load_dataset(data_path)

    return build_brand_pairs(
        df,
        brand=BRAND
    )
