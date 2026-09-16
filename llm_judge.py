import argparse
import json

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def load_model():
    print(f"Loading judge model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    model.eval()

    return tokenizer, model


def judge_response(
    tokenizer,
    model,
    customer_text,
    historical_response,
):
    prompt = f"""
You are evaluating whether a customer-support response is good and grounded.

Customer message:
{customer_text}

Historical support response:
{historical_response}

Score the historical support response on these dimensions:

1. relevance: Does it address the customer's issue?
2. groundedness: Is the response supported by the historical case?
3. helpfulness: Would it reasonably help the customer?
4. hallucination: Does it contain unsupported claims or invented information?

Use scores from 1 to 5 for the first three dimensions.

For hallucination:
0 = no hallucination
1 = hallucination present

Return ONLY valid JSON:

{{
  "relevance": 1,
  "groundedness": 1,
  "helpfulness": 1,
  "hallucination": 0
}}
"""

    messages = [
        {
            "role": "system",
            "content": "You are a strict customer-support response evaluator.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    # Try to extract JSON from the model output.
    start = response.find("{")
    end = response.rfind("}")

    if start == -1 or end == -1:
        return {
            "relevance": None,
            "groundedness": None,
            "helpfulness": None,
            "hallucination": None,
            "raw_output": response,
        }

    json_text = response[start:end + 1]

    try:
        result = json.loads(json_text)

        return {
            "relevance": result.get("relevance"),
            "groundedness": result.get("groundedness"),
            "helpfulness": result.get("helpfulness"),
            "hallucination": result.get("hallucination"),
            "raw_output": response,
        }

    except json.JSONDecodeError:
        return {
            "relevance": None,
            "groundedness": None,
            "helpfulness": None,
            "hallucination": None,
            "raw_output": response,
        }


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-judge for Apple Support historical responses"
    )

    parser.add_argument(
        "--input",
        default="evaluation_results.csv",
        help="Evaluation CSV containing golden examples",
    )

    parser.add_argument(
        "--output",
        default="llm_judge_results.csv",
        help="Output CSV",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=30,
        help="Number of examples to judge",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Reproducible sample.
    if len(df) > args.sample:
        df = df.sample(
            n=args.sample,
            random_state=42,
        ).reset_index(drop=True)

    print(f"Examples to judge: {len(df)}")

    tokenizer, model = load_model()

    results = []

    for i, row in df.iterrows():

        print(
            f"\nJudging example {i + 1}/{len(df)}..."
        )

        customer_text = str(
            row.get("customer_text", "")
        )

        historical_response = str(
            row.get("support_response", "")
        )

        result = judge_response(
            tokenizer,
            model,
            customer_text,
            historical_response,
        )

        result["customer_tweet_id"] = row.get(
            "customer_tweet_id"
        )

        results.append(result)

        print(result)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        args.output,
        index=False,
    )

    print("\n" + "=" * 50)
    print("LLM-AS-JUDGE RESULTS")
    print("=" * 50)

    print(
        f"Saved: {args.output}"
    )

    for column in [
        "relevance",
        "groundedness",
        "helpfulness",
    ]:
        values = pd.to_numeric(
            results_df[column],
            errors="coerce",
        )

        print(
            f"{column}: "
            f"{values.mean():.2f}/5"
        )

    hallucination = pd.to_numeric(
        results_df["hallucination"],
        errors="coerce",
    )

    print(
        "Hallucination rate: "
        f"{hallucination.mean():.1%}"
    )


if __name__ == "__main__":
    main()
