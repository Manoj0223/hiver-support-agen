# Hiver AI Support Agent — Apple Support

An AI customer-support agent built for the Hiver SDE Intern take-home assignment using the **Customer Support on Twitter** dataset.

The prototype performs three tasks:

1. Classifies incoming customer messages into a compact set of support intents.
2. Retrieves historically similar Apple Support conversations as evidence for a response.
3. Decides whether the case can be auto-handled or should be escalated to a human, with a reason.

This is a prototype and is not intended to replace production customer-support systems.

---

## 1. Problem Framing

Customer-support teams receive large volumes of repetitive requests. A useful support agent should:

- identify the customer's issue,
- use previously resolved cases as evidence,
- draft a conservative response,
- avoid answering automatically when sufficient evidence is unavailable.

For this project, I selected **Apple Support** from the Customer Support on Twitter dataset.

The goal is to demonstrate an end-to-end support workflow covering:

**Intent Classification → Historical Retrieval → Response Generation → Escalation**

---

## 2. Dataset

Dataset: **Customer Support on Twitter — Kaggle / ThoughtVector**

The dataset contains customer tweets and support-agent responses.

The raw `twcs.csv` file is intentionally excluded from Git because of its size.

### Apple Support extraction

Apple Support replies are identified using:

`python
(df["inbound"] == False) &
(df["author_id"].astype(str) == "AppleSupport")`
Customer tweets are paired with support responses using:

in_response_to_tweet_id → tweet_id

Current dataset:

Dataset component	Rows
Apple Support pairs	105,212
Golden examples	200
Leakage-safe pool	105,012
Training rows	84,009
Internal test rows	21,003
Text preprocessing

The preprocessing:

lowercases text
removes URLs
removes Twitter mentions
removes standalone numbers
removes punctuation
normalizes whitespace
3. Intent Taxonomy

A compact nine-intent taxonomy was created from recurring support themes.

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

The large training corpus is weak-labelled using rule-based patterns derived from these intent definitions.

Golden set

A 200-example evaluation set was sampled from Apple Support conversation pairs.

Labels were initially generated using the project taxonomy and weak-label rules, with a subset manually audited. The current results should therefore be treated as a prototype evaluation rather than a fully independently hand-labelled benchmark.

4. System Architecture
Customer Message
       |
       v
 Text Cleaning
       |
       +-----------------------+
       |                       |
       v                       v
Intent Classifier       Historical Retrieval
       |                 TF-IDF similarity
       |                       |
       +-----------+-----------+
                   |
                   v
          Evidence Availability
             /           \
            /             \
     Strong evidence    Weak evidence
          |                  |
          v                  v
   Draft response         Escalate

The system consists of:

TF-IDF + Logistic Regression intent classification
TF-IDF historical retrieval
retrieval-grounded response generation
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

Clean the customer message.
Transform it using the retrieval vectorizer.
Compare it with historical customer messages.
Retrieve the highest-similarity cases.
Use the corresponding historical support response as evidence.

The 200 golden examples are excluded from the training/retrieval pool to reduce evaluation leakage.

7. Response Generation

The prototype uses retrieved historical support responses as evidence.

If the best historical match is sufficiently similar, the system generates a conservative response based on that historical response.

The response generator:

does not invent troubleshooting instructions,
uses historical support evidence,
asks for additional information when appropriate,
avoids automatically answering when evidence is weak.

If sufficient historical evidence is unavailable, no automated response is generated.

Example:

Thanks for reaching out. Based on a similar Apple Support case,
here is the relevant guidance:

<historical support guidance>

If the issue persists, please share your device model and
software version so the issue can be investigated further.
8. Escalation Policy

The primary safety signal is historical evidence availability.

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
Golden-set intent results
Metric	Result
Accuracy	63.50%
Macro F1	63.60%
Weighted F1	67.83%

Classification report:

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

Small classes should be interpreted cautiously because several intents contain very few examples.

10. Baselines

Two simple baselines were evaluated on the same leakage-safe golden-set setup.

Model	Accuracy	Macro F1	Weighted F1
Majority class	40.85%	6.45%	—
Keyword rules	63.77%	50.52%	62.28%
TF-IDF + Logistic Regression	63.50%	63.60%	67.83%

The keyword baseline has slightly higher accuracy, but the TF-IDF classifier has substantially higher Macro F1, indicating better performance across the less frequent intents.

The majority-class baseline provides a trivial lower bound.

11. Escalation Evaluation

Using the retrieval threshold of 0.55, the current golden-set evaluation gives:

Metric	Result
Accuracy	57.50%
Precision	62.42%
Recall	81.75%
F1	70.79%

On the 200-example golden set:

35 / 200 examples = 17.5%

were eligible for automated reply under the retrieval threshold.

The current policy intentionally favors human escalation when sufficient historical evidence is unavailable.

12. LLM-as-Judge

A local Qwen 2.5 3B Instruct model was explored as an LLM judge.

The planned evaluation dimensions were:

relevance: 1–5
groundedness: 1–5
helpfulness: 1–5
hallucination: 0/1

A small 5-example pilot produced:

Metric	Pilot result
Relevance	4.20 / 5
Groundedness	5.00 / 5
Helpfulness	4.80 / 5
Hallucination rate	0%

These results are pilot results only and are not used as the headline evaluation.

A larger local run was not completed because the 3B model requires a multi-GB download and substantial local compute.

Human-vs-LLM evidence-judge agreement was therefore not reported, rather than estimated from insufficient data.

13. Top 5 Failure Patterns

The current golden-set evaluation contains 73 intent errors out of 200 examples.

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

14. What Is Misleading About My Headline Number?

The headline result is:

63.5% intent classification accuracy

This number should not be interpreted as production-ready performance.

The evaluation set is small and imbalanced:

Intent	Share
other	46.5%
ios_update	26.5%
battery_charging	10.0%
apps	5.0%
device_hardware	5.0%
apple_services	2.5%
wifi_connectivity	2.0%
account_payment	1.5%
how_to	1.0%

Accuracy can therefore hide weaknesses in smaller intents.

Macro F1 gives a better view across classes, but it is also uncertain because the evaluation contains only 200 examples and several classes have very small support.

The golden labels were not produced by a fully independent two-reviewer annotation process. A subset was manually audited after initial weak labelling. This is an important limitation when interpreting the result.

The historical Twitter dataset also does not represent current Apple support policy or live customer-support traffic.

15. What I Did Not Build

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

16. Decision Log
Decision	Reason
Apple Support	Large number of usable support conversations
Pair construction	Used in_response_to_tweet_id to connect customer messages with responses
Compact taxonomy	Nine intents balance usefulness and data availability
other category	Handles ambiguous or insufficiently specific messages
TF-IDF classifier	Fast, interpretable and reproducible
Class weighting	Addresses uneven intent frequencies
Historical retrieval	Provides evidence from previously handled cases
Leakage prevention	Golden examples excluded from training/retrieval
Conservative response generation	Avoids inventing unsupported troubleshooting steps
Retrieval threshold	Provides an evidence-availability safety signal
Conservative automation	Escalates when historical evidence is insufficient
Sensitive-case handling	Account/payment cases are escalated
Baseline comparison	Separates improvement over trivial and simple approaches
Evaluation transparency	Reports limitations instead of presenting prototype results as production validation
17. What I Would Build Next Week
Golden dataset
Expand the golden set to 500+ examples.
Add a second reviewer.
Measure reviewer disagreement.
Recalculate metrics on independently verified labels.
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
Expand the LLM-as-judge evaluation.
Compare LLM judge results with human ratings.
Measure human-vs-LLM agreement.
18. Reproduction
1. Download the dataset

Download the Kaggle Customer Support on Twitter dataset:

thoughtvector/customer-support-on-twitter

Place the raw file at:

data/raw/twcs.csv

The raw dataset is intentionally excluded from Git because of its size.

2. Install dependencies
pip install -r requirements.txt
3. Run the agent
python run.py --data data/raw/twcs.csv

Or provide a custom customer message:

python run.py --data data/raw/twcs.csv --query "My iPhone battery is draining very quickly."
4. Run evaluation
python evaluate.py --data data/raw/twcs.csv --golden data/golden/apple_support_golden_200_current.csv

The evaluation reports:

intent accuracy
Macro F1
Weighted F1
classification report

It also produces:

evaluation_results.csv
Runtime

The full pipeline requires the local Kaggle dataset and trains TF-IDF/Logistic Regression models on tens of thousands of examples.

The repository does not claim a sub-15-minute clean-environment runtime because that runtime has not been independently verified.

19. Repository Structure
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
20. Limitations

The main limitations are:

The dataset consists of historical Twitter conversations.
Historical responses may not represent current Apple support policy.
The intent taxonomy is dataset-specific.
The golden set contains only 200 examples.
Several intents have very small evaluation support.
The labels are not yet independently verified by multiple human reviewers.
TF-IDF retrieval can struggle when semantically similar messages use different vocabulary.
The response generator is intentionally conservative.
Escalation is primarily driven by historical evidence availability and safety rules.
LLM-as-judge was only evaluated as a small pilot.
Human-vs-LLM judge agreement was not measured.
21. Summary

This project demonstrates an end-to-end customer-support agent:

Intent Classification
        ↓
Historical Evidence Retrieval
        ↓
Conservative Response Generation
        ↓
Escalation
Current results
63.50% Intent Accuracy
63.60% Intent Macro F1
67.83% Intent Weighted F1
70.79% Escalation F1
17.5% of golden examples eligible for automated reply

The main takeaway is that the prototype can classify support requests and retrieve historical evidence, while conservatively escalating cases where evidence is insufficient.

The next major improvements are a genuinely independently hand-labelled evaluation set, stronger semantic retrieval, calibrated escalation, and a larger human-vs-LLM response-quality evaluation.

**Important:** This version is intentionally transparent about the golden-set labeling and LLM-judge limitations rather than claiming work that wasn't actually completed.

