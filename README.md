# Hiver AI Support Agent — Apple Support

An AI customer-support agent built for the Hiver SDE Intern take-home assignment using the Customer Support on Twitter dataset.

The system performs three tasks:

1. Classifies incoming customer messages into a compact set of support intents.
2. Retrieves historically similar Apple Support conversations as evidence for a response.
3. Decides whether the case can be auto-handled or should be escalated to a human.

---

## 1. Problem Framing

Customer-support teams receive a large volume of repetitive requests. A useful support agent should identify the customer's issue, use previously handled cases as evidence, provide a conservative response, and avoid answering automatically when sufficient evidence is unavailable.

For this project, I selected **Apple Support** from the Customer Support on Twitter dataset.

The objective is to demonstrate a reproducible prototype for:

- intent classification
- retrieval-grounded response generation
- escalation decisions
- evaluation against a manually reviewed golden set

This is not intended to be a production replacement for Apple Support.

---

## 2. Dataset

Dataset:

**Customer Support on Twitter — Kaggle / ThoughtVector**

The dataset contains customer tweets and support-agent responses.

The raw `twcs.csv` file is not included in this repository because of its size.

### Apple Support extraction

Apple Support replies are identified using:

`python
(df["inbound"] == False) &
(df["author_id"].astype(str) == "AppleSupport")`
Customer tweets are paired with support responses using:

in_response_to_tweet_id
        ↓
tweet_id

The current dataset contains:

105,212 usable Apple Support customer → support-response pairs.

Text preprocessing

The preprocessing:

lowercases text
removes URLs
removes Twitter mentions
removes standalone numbers
removes punctuation
normalizes whitespace
3. Intent Taxonomy

A compact nine-intent taxonomy was created from recurring support themes in the Apple Support conversations.

Intent	Description
ios_update	iOS/software update problems
battery_charging	Battery drain, charging and power problems
wifi_connectivity	Wi-Fi, Bluetooth, cellular and connectivity issues
apps	App crashes, freezing or app-specific problems
device_hardware	Screen, buttons, keyboard, camera and other device issues
apple_services	iCloud, iTunes, Apple Music, App Store and related services
account_payment	Apple ID, account, billing, subscriptions and payments
how_to	Questions asking how to perform a feature or task
other	Ambiguous or insufficiently specific cases

The initial large training corpus is weak-labelled using rule-based patterns derived from these intent definitions.

The 200-example evaluation set uses human-reviewed intent labels.

4. System Architecture
Customer Message
       |
       v
 Text Cleaning
       |
       +----------------------+
       |                      |
       v                      v
Intent Classifier      Historical Retrieval
       |                TF-IDF similarity
       |                      |
       +----------+-----------+
                  |
                  v
          Evidence Availability
                  |
           +------+------+
           |             |
      Strong evidence  Weak evidence
           |             |
           v             v
      Draft response   Escalate

The system consists of:

TF-IDF + Logistic Regression intent classification
TF-IDF historical retrieval
conservative evidence-grounded response generation
rule-based escalation
5. Intent Classifier

The classifier uses:

TF-IDF features
unigram + bigram features
Logistic Regression
balanced class weighting

Current configuration:

TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_features=50000,
    sublinear_tf=True
)
LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

The classifier outputs:

predicted intent
prediction confidence
6. Historical Retrieval

Historical Apple Support conversations are indexed using TF-IDF.

For each incoming message:

Clean the message.
Transform it using the retrieval vectorizer.
Compare it against historical customer messages.
Retrieve the highest-similarity cases.
Use the retrieved support response as evidence.

Golden evaluation examples are excluded from the training/retrieval pool to reduce leakage.

The current evaluation uses a leakage-safe corpus.

7. Response Generation

The prototype uses retrieved historical support responses as evidence.

If the best historical match is sufficiently similar, the system generates a conservative response based on that historical response.

The response generator:

does not invent troubleshooting instructions
uses historical support evidence
asks for additional information when appropriate
avoids automatically answering when evidence is weak

If sufficient historical evidence is unavailable, no automated response is generated.

Example:

Thanks for reaching out. Based on a similar Apple Support case,
here is the relevant guidance:

<historical support guidance>

If the issue persists, please share your device model and
software version so the issue can be investigated further.
8. Escalation Policy

The prototype uses historical evidence availability as one of its primary safety signals.

Current retrieval threshold:

RETRIEVAL_THRESHOLD = 0.55

Therefore:

best_similarity >= 0.55
        → eligible for initial automated handling

best_similarity < 0.55
        → escalate to human

Sensitive account/payment-related messages and messages with insufficient context are also escalated.

The threshold was selected as a conservative operating point rather than optimized for maximum automation coverage.

9. Evaluation
Golden dataset

The evaluation set contains 200 manually reviewed examples sampled from the Apple Support conversation pairs.

The golden set contains:

customer message
historical support response
human-reviewed intent
human-reviewed escalation decision
escalation reason

The evaluation examples are excluded from the training/retrieval pool.

Dataset split

Current evaluation run:

Dataset	Rows
Apple Support pairs	105,212
Golden examples	200
Leakage-safe pool	105,012
Training rows	84,009
Internal test rows	21,003

The internal train/test split is separate from the 200-example golden evaluation.

10. Intent Classification Results

Current golden-set results:

Metric	Result
Accuracy	63.50%
Macro F1	63.60%
Weighted F1	67.83%
Classification report
Intent	Precision	Recall	F1	Support
account_payment	0.50	0.67	0.57	3
apple_services	0.50	1.00	0.67	5
apps	0.17	0.80	0.28	10
battery_charging	1.00	0.85	0.92	20
device_hardware	0.39	0.70	0.50	10
how_to	1.00	0.50	0.67	2
ios_update	0.91	0.55	0.68	53
other	0.82	0.59	0.69	93
wifi_connectivity	0.75	0.75	0.75	4

Small classes should be interpreted cautiously because several intents contain only a few examples.

11. Baselines

Two baselines are included in the evaluation design.

Baseline 1 — Majority class

Always predict the most frequent intent in the training data.

This provides a trivial lower-bound comparison.

Baseline 2 — Simple TF-IDF classifier

Use TF-IDF features followed by Logistic Regression without the additional balanced-class configuration used by the main classifier.

The main system is compared against these baselines using accuracy and Macro F1.

12. Escalation Evaluation

Using the retrieval threshold of 0.55, the current golden-set evaluation gives:

Metric	Result
Accuracy	57.50%
Precision	62.42%
Recall	81.75%
F1	70.79%

The system is intentionally conservative about automated handling.

On the 200-example golden set, the threshold resulted in:

35 / 200 examples eligible for automated reply = 17.5%

The remaining cases were routed toward human handling.

The high recall for escalation-labelled cases comes with false positives, meaning the current policy prioritizes avoiding unsupported automated responses over maximizing automation coverage.

13. LLM-as-Judge

A local Qwen 2.5 3B Instruct model was explored as an LLM judge.

The planned judge evaluates:

relevance: 1–5
groundedness: 1–5
helpfulness: 1–5
hallucination: 0/1

A small 5-example pilot was previously run.

Pilot results:

Metric	Score
Relevance	4.20 / 5
Groundedness	5.00 / 5
Helpfulness	4.80 / 5
Hallucination rate	0%

These results are only a pilot and are not used as the headline evaluation result.

A larger local run was not completed because the 3B model requires a multi-GB download and substantial local compute.

Human-vs-LLM evidence-judge agreement was therefore not reported, rather than estimated from insufficient data.

14. Top 5 Failure Patterns

The current golden-set evaluation contains 73 intent errors out of 200 examples.

The main failure patterns are:

1. other → apps

Some messages mention applications without clearly establishing that the application itself is the source of the problem.

Improvement: distinguish between an app being mentioned and an app-specific failure.

2. other → ios_update

Some messages mention iOS versions or updates without actually describing an update problem.

Improvement: require update-related symptoms rather than treating every iOS reference as an update issue.

3. ios_update → other

Some update complaints are short or ambiguous.

Improvement: combine classifier confidence with retrieval evidence.

4. other → how_to

Some messages implicitly ask how to use a feature without explicitly saying "how do I".

Improvement: introduce semantic question/feature-use detection.

5. Cross-intent errors

Some messages contain multiple concepts, making mutually exclusive intent boundaries difficult.

Improvement: introduce hierarchical classification and explicit multi-intent handling.

15. What Is Misleading About My Headline Number?

The 63.5% accuracy should not be interpreted as production-ready performance.

The evaluation set is imbalanced:

other: 46.5%
ios_update: 26.5%
battery_charging: 10.0%
apps: 5.0%
device_hardware: 5.0%
apple_services: 2.5%
wifi_connectivity: 2.0%
account_payment: 1.5%
how_to: 1.0%

Therefore, accuracy alone can hide weaknesses in smaller intents.

Macro F1 provides a better view of performance across classes, but even Macro F1 should be interpreted cautiously because the evaluation contains only 200 examples and several classes have very small support.

The largest limitation is therefore not the headline accuracy itself, but the relatively small and imbalanced evaluation set.

16. What I Did Not Build

To keep the project focused, the following were intentionally not implemented:

production deployment
Hiver API integration
live customer/account lookup
transactional actions
fine-tuning of a large language model
access to Apple's internal support systems
automatic handling of sensitive account/payment actions
real-time policy verification
a general-purpose support agent for every brand

Historical Twitter responses are treated as evidence from the dataset, not as guaranteed representations of Apple's current support policy.

17. Decision Log
Brand selection — Selected Apple Support because it contains a large number of usable customer/support-response pairs.
Pair construction — Used in_response_to_tweet_id to connect customer messages with support responses.
Compact taxonomy — Used nine intents to balance usefulness and sufficient training data.
other category — Added other for ambiguous or insufficiently specific messages.
TF-IDF classifier — Selected TF-IDF + Logistic Regression as a fast and interpretable baseline.
Class weighting — Used balanced class weighting because intent frequencies are uneven.
Historical retrieval — Used customer-message similarity to retrieve previously handled cases.
Leakage prevention — Excluded golden examples from the training/retrieval pool.
Conservative response generation — Used historical responses as evidence rather than inventing troubleshooting steps.
Escalation threshold — Used retrieval similarity as an evidence-availability signal.
Conservative automation — Preferred human escalation when historical evidence was insufficient.
Sensitive cases — Account/payment-related cases are escalated rather than automatically handled.
Evaluation transparency — Reported the limitations of a 200-example evaluation rather than presenting it as production validation.
LLM judge — Treated the Qwen evaluation as a pilot and did not use it as the headline metric.
18. What I Would Build Next Week
Golden dataset
Expand the golden set to 500+ examples.
Add a second reviewer.
Measure reviewer disagreement.
Recalculate metrics on the verified dataset.
Classification
Add semantic features.
Introduce hierarchical classification.
Calibrate classifier confidence.
Handle multi-intent messages explicitly.
Retrieval
Filter retrieval candidates by predicted intent.
Compare TF-IDF with dense embeddings.
Add temporal holdout evaluation.
Measure evidence quality separately.
Response generation
Generate responses conditioned explicitly on retrieved evidence.
Add evidence snippets internally.
Add unsupported-claim checks.
Improve handling of already-resolved issues.
Escalation
Tune the similarity threshold using a validation set.
Separate insufficient evidence from sensitive issues.
Add explicit account, payment and security safety rules.
Evaluation
Expand LLM-as-judge beyond the pilot.
Compare LLM judge results with human ratings.
Measure human-vs-LLM evidence agreement.
19. Reproduction
1. Download the dataset

Download the Kaggle Customer Support on Twitter dataset:

thoughtvector/customer-support-on-twitter

Place:

data/raw/twcs.csv

The raw dataset is intentionally excluded from Git because of its size.

2. Install dependencies
pip install -r requirements.txt
3. Run the agent
python run.py --data data/raw/twcs.csv

Or provide a custom customer message:

python run.py \
  --data data/raw/twcs.csv \
  --query "My iPhone battery is draining very quickly."
4. Run evaluation
python evaluate.py \
  --data data/raw/twcs.csv \
  --golden data/golden/apple_support_golden_200_current.csv

The evaluation reports:

intent accuracy
Macro F1
Weighted F1
classification report

It also produces:

evaluation_results.csv
Runtime

The full pipeline requires the local Kaggle dataset and trains TF-IDF/Logistic Regression models on tens of thousands of examples.

The repository does not claim a sub-15-minute clean-environment runtime, because that runtime has not been independently verified.

20. Repository Structure
.
├── README.md
├── requirements.txt
├── run.py
├── evaluate.py
├── llm_judge.py
├── data/
│   └── golden/
│       └── apple_support_golden_200_current.csv
└── src/
    ├── config.py
    ├── data_loader.py
    ├── labeling.py
    ├── classifier.py
    ├── retrieval.py
    ├── response_generator.py
    ├── escalation.py
    └── evaluation.py
21. Limitations

The main limitations are:

The dataset consists of historical Twitter conversations.
Historical responses may not represent current Apple support policy.
The intent taxonomy is dataset-specific.
The golden set contains only 200 examples.
Several intents have very small evaluation support.
TF-IDF retrieval can struggle with semantic similarity when vocabulary differs.
The response generator is intentionally conservative.
Escalation is primarily driven by historical evidence availability and explicit safety rules.
LLM-as-judge was only evaluated as a small pilot.
Human-vs-LLM judge agreement was not measured.
22. Summary

This project demonstrates an end-to-end support-agent prototype:

Intent Classification
        ↓
Historical Evidence Retrieval
        ↓
Conservative Response Generation
        ↓
Escalation

Current golden-set results:

63.50% intent accuracy
63.60% intent Macro F1
67.83% intent Weighted F1
70.79% escalation F1

The system intentionally favors evidence-backed responses and human escalation when historical evidence is insufficient.

The next major improvement would be a larger independently verified golden dataset, stronger semantic retrieval, and a validated human-vs-LLM evaluation framework.

### Important correction

I deliberately removed the old **87% / 84.33%** numbers and replaced them with your actual latest run:

**63.50% Accuracy, 63.60% Macro F1, 67.83% Weighted F1.**

Also, I did **not** claim a 30-example LLM evaluation or human-vs-LLM agreement because we didn't complete those.

One thing we should verify next is the **baseline numbers**, because the old README's baseline figures came from an earlier run and shouldn't be carried over without checking.

















