import glob
import os
import re

import pandas as pd


BRAND = "AppleSupport"


def clean_text(text):
    """Normalize customer text for classification and retrieval."""

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
    """Load the Customer Support on Twitter dataset."""

    if data_path is None:
        files = glob.glob("data/raw/*.csv")

        if not files:
            raise FileNotFoundError(
                "No CSV found in data/raw/. "
                "Place twcs.csv in data/raw/."
            )

        data_path = files[0]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    return pd.read_csv(data_path)


def build_brand_pairs(df, brand=BRAND):
    """
    Build customer -> Apple Support conversation pairs.

    A customer message is paired with an Apple Support reply when:
    Apple's reply references the customer's tweet_id through
    in_response_to_tweet_id.
    """

    apple_replies = df[
        (df["inbound"] == False)
        & (df["author_id"].astype(str) == brand)
    ].copy()

    customer_messages = df[
        df["inbound"] == True
    ][
        ["tweet_id", "text"]
    ].copy()

    pairs = apple_replies.merge(
        customer_messages,
        left_on="in_response_to_tweet_id",
        right_on="tweet_id",
        how="inner",
        suffixes=("_reply", "_customer"),
    )

    pairs = pairs.rename(
        columns={
            "tweet_id_customer": "customer_tweet_id",
            "text_customer": "customer_text",
            "text_reply": "support_response",
        }
    )

    pairs = pairs[
        [
            "customer_tweet_id",
            "customer_text",
            "support_response",
        ]
    ].dropna()

    pairs["clean_customer_text"] = pairs["customer_text"].apply(
        clean_text
    )

    pairs = pairs[
        pairs["clean_customer_text"].str.len() > 0
    ].copy()

    # Remove duplicate customer messages.
    pairs = pairs.drop_duplicates(
        subset=["clean_customer_text"]
    ).reset_index(drop=True)

    return pairs


def load_apple_support_pairs(data_path=None):
    """Convenience function for loading and preparing Apple Support data."""

    df = load_dataset(data_path)
    pairs = build_brand_pairs(df, BRAND)

    return pairs
