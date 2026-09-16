# Hiver AI Support Agent — Apple Support

An AI customer-support agent built for the Hiver SDE Intern take-home assignment using the Customer Support on Twitter dataset.

The system performs three tasks:

1. Classifies incoming customer messages into a compact set of support intents.
2. Retrieves historically similar Apple Support conversations and uses them as evidence for a response.
3. Decides whether the case can be auto-handled or should be escalated to a human.

---

## 1. Problem Framing

Customer-support teams receive a large volume of repetitive requests. A useful support agent should be able to identify the customer's issue, use previously resolved cases as evidence, provide a safe response, and avoid confidently answering cases where the available evidence is insufficient.

For this project, I selected **Apple Support** from the Customer Support on Twitter dataset.

The objective is not to build a production-ready Apple Support replacement. Instead, the project demonstrates a reproducible prototype for:

- intent classification,
- retrieval-grounded response generation,
- escalation decisions,
- and evaluation against manually reviewed examples.

---

## 2. Dataset

Dataset:

`Customer Support on Twitter` — Kaggle / ThoughtVector

The dataset contains customer tweets and support-agent responses.

### Apple Support extraction

Apple Support replies were identified using:

`python`
`(df["inbound"] == False) &
(df["author_id"].astype(str) == "AppleSupport")`
Customer tweets were paired with Apple Support responses using:

  in_response_to_tweet_id
        ↓
    tweet_id

This produced approximately 104K usable Apple Support customer → support-response pairs after cleaning and deduplication.

Text cleaning

The preprocessing removes:

URLs
Twitter mentions
standalone numbers
punctuation
duplicate whitespace

Text is lowercased before classification and retrieval.

3. Intent Taxonomy

I intentionally kept the taxonomy small enough to be useful for an automated support workflow.

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
4. System Architecture
Customer Message
       |
       v
   Text Cleaning
       |
       +----------------------+
       |                      |
       v                      v
 Intent Classifier       Historical Retrieval
       |                      |
       |                 TF-IDF similarity
       |                      |
       +----------+-----------+
                  |
                  v
        Evidence Availability
                  |
          +-------+-------+
          |               |
     Strong evidence   Weak evidence
          |               |
          v               v
     Draft response    Escalate
5. Intent Classifier

The primary classifier uses:

TF-IDF features
unigram + bigram features
Logistic Regression
class weighting to handle class imbalance

Configuration:

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

The model produces both:

predicted intent
prediction confidence
6. Retrieval

Historical Apple Support conversations are indexed using TF-IDF.

For each new customer message:

Clean the message.
Transform it using the retrieval TF-IDF vectorizer.
Compare it against historical customer messages.
Retrieve the highest-similarity historical cases.
Use those cases as evidence for response generation.

To reduce evaluation leakage, golden-set examples are excluded from the retrieval corpus.

The leakage-safe retrieval corpus contained approximately 101K historical cases.

Retrieval statistics on the 200-example evaluation set
Statistic	Value
Mean best similarity	0.417
Median	0.357
Minimum	0.200
Maximum	1.000
7. Response Generation

The current prototype deliberately avoids copying historical support responses verbatim.

Instead, retrieved historical conversations are treated as evidence, and the system produces a conservative response.

Example response structure:

Thanks for reaching out. We'd be happy to help with this.
Based on similar Apple Support cases, we'd like to look into
the issue further. Please share any relevant device or software
details so we can investigate.
Why this approach?

A support agent should avoid inventing troubleshooting steps that are not supported by the retrieved evidence.

The current prototype therefore prioritizes:

evidence availability,
conservative language,
requesting additional information when necessary,
and escalation when evidence is insufficient.
8. Escalation Policy

The prototype uses historical evidence availability as a safety signal.

Current rule:

pred_escalate = best_similarity < 0.55

Therefore:

high historical similarity → eligible for initial automated handling
insufficient historical similarity → escalate to a human

This is intentionally conservative because an incorrect automated answer can be more harmful than routing a case to a human.

9. Evaluation

The evaluation set contains 200 examples.

Important evaluation limitation:

The original 200 labels were initially generated as model-assisted draft labels. During the review process, 20 examples were independently human-reviewed and corrected where necessary. The remaining 180 labels are still draft/model-assisted labels. Therefore, the results below should be considered provisional rather than final hand-labelled golden-set results.

10. Intent Classification Results
Main system
Metric	Result
Accuracy	87.00%
Macro F1	84.33%
Weighted F1	88.00%
Baselines
Model	Accuracy	Macro F1
Trivial majority-class baseline	49.50%	7.36%
Simple TF-IDF + Logistic Regression	86.50%	79.92%
Balanced TF-IDF + Logistic Regression	87.00%	84.33%

The main improvement over the simple baseline is primarily visible in Macro F1 rather than raw accuracy.

11. Classification Report
                   precision    recall  f1-score   support

  account_payment       0.75      1.00      0.86         3
   apple_services       1.00      1.00      1.00         5
             apps       0.43      0.90      0.58        10
 battery_charging       1.00      1.00      1.00        20
  device_hardware       1.00      0.80      0.89         5
           how_to       0.33      1.00      0.50         1
       ios_update       0.87      0.91      0.89        53
            other       0.95      0.81      0.87        99
wifi_connectivity       1.00      1.00      1.00         4

accuracy                           0.87       200
macro avg          0.82      0.93      0.84       200
weighted avg       0.91      0.87      0.88       200

Small classes should be interpreted cautiously because several intents contain only a few examples.

12. Escalation Results

Using the retrieval-similarity threshold of 0.55:

Metric	Result
Accuracy	58.50%
Precision	63.64%
Recall	82.03%
F1	71.67%

Confusion matrix:

              Predicted
              No    Yes

Actual No      12    60
Actual Yes     23   105

The system predicted escalation for:

165 / 200 = 82.5%

The high recall indicates that the rule catches many of the examples labelled for escalation, but the relatively large number of false positives shows that the current threshold is conservative.

13. LLM-as-Judge

A local Qwen 2.5 3B Instruct model was tested as an LLM judge.

The judge evaluates:

relevance: 1–5
groundedness: 1–5
helpfulness: 1–5
hallucination: 0/1

A pilot evaluation was completed on 5 examples.

Metric	Score
Relevance	4.20 / 5
Groundedness	5.00 / 5
Helpfulness	4.80 / 5
Hallucination rate	0%

These scores are reported only as a 5-example pilot, not as a statistically meaningful estimate for the full evaluation set.

14. Top 5 Failure Patterns

The current evaluation produced 26 intent errors out of 200 examples.

Gold intent	Predicted intent	Count
other	apps	12
other	ios_update	5
ios_update	other	4
other	how_to	2
ios_update	account_payment	1
Failure analysis
1. other → apps

This is the largest failure category.

Some messages contain references to applications without explicitly describing whether the problem is an app failure, an OS issue, or another device-level problem.

Potential improvement: introduce stronger app-specific features and distinguish “app mentioned” from “app is the source of the problem.”

2. other → ios_update

Messages sometimes mention iOS versions or updates without actually reporting an update problem.

Potential improvement: require an update-related symptom rather than treating every iOS reference as an update issue.

3. ios_update → other

Some update complaints are short or ambiguous.

Potential improvement: use retrieved historical evidence jointly with classifier confidence.

4. other → how_to

Some questions are implicitly asking how to use a feature without using explicit “how do I” language.

Potential improvement: add semantic question detection.

5. ios_update → account_payment

Some messages contain multiple concepts, making keyword-based boundaries unreliable.

Potential improvement: use hierarchical classification and explicitly resolve multi-intent messages.

15. What Is Misleading About My Headline Number?

The 87.0% accuracy looks strong, but it should not be interpreted as production-ready performance.

The evaluation set is highly imbalanced:

other                49.5%
ios_update           26.5%
battery_charging     10.0%
apps                  5.0%
device_hardware       2.5%
apple_services        2.5%
wifi_connectivity     2.0%
account_payment       1.5%
how_to                0.5%

Nearly half of the evaluation set belongs to other, while how_to has only one example and account_payment has three.

Therefore:

accuracy can hide weaknesses in smaller intents,
Macro F1 is a more useful complementary metric,
and results from only 200 examples have substantial uncertainty.

Most importantly, the current evaluation labels are not yet a fully hand-labelled golden set. Only 20 of the 200 examples have been independently human-reviewed so far.

16. What I Did Not Build

To keep the project focused, the following were intentionally not implemented:

No production deployment.
No Hiver API integration.
No live customer/account lookup.
No transactional actions.
No fine-tuning of a large language model.
No access to Apple's internal support systems.
No claim that historical Twitter responses represent Apple's current support policy.
No automatic handling of sensitive account/payment actions.
No attempt to build a general-purpose customer-support agent for every brand.
17. Decision Log
1. Brand selection

Selected Apple Support because the dataset contains a large number of Apple Support conversations with usable customer → support-response pairs.

2. Pair construction

Used in_response_to_tweet_id to connect customer messages with Apple Support responses.

3. Compact intent taxonomy

Used nine intents to balance usefulness and sufficient training examples.

4. other category

Added other for ambiguous messages and cases that cannot be confidently assigned to another category.

5. TF-IDF classifier

Selected TF-IDF + Logistic Regression because it is fast, interpretable and provides a strong baseline.

6. Class weighting

Used class_weight="balanced" because intent frequencies are highly uneven.

7. Historical retrieval

Used customer-message similarity to retrieve previously handled cases.

8. Leakage prevention

Excluded evaluation examples from the retrieval corpus.

9. Conservative response generation

Avoided directly copying historical responses.

10. Escalation threshold

Used retrieval similarity as an evidence-availability signal.

11. Conservative escalation

Preferred escalation when historical evidence is insufficient rather than generating unsupported troubleshooting advice.

12. Evaluation transparency

Reported the limitations of the current draft-labelled evaluation instead of presenting it as fully human-labelled gold data.

18. What I Would Build Next Week
Golden dataset
Complete human verification of 200 examples.
Add a second reviewer for disagreement analysis.
Recalculate all metrics on the verified dataset.
Classification
Add semantic features.
Introduce hierarchical classification.
Calibrate classifier confidence.
Investigate multi-intent messages.
Retrieval
Filter retrieval candidates by predicted intent.
Compare TF-IDF against dense embeddings.
Add temporal holdout evaluation.
Track retrieval evidence quality.
Response generation
Generate responses conditioned on retrieved evidence.
Include evidence snippets internally.
Add explicit unsupported-claim checks.
Improve handling of cases where the customer has already solved the problem.
Escalation
Tune the similarity threshold on a validation set.
Separate “insufficient evidence” from “sensitive issue.”
Add explicit safety rules for account, payment and security issues.
Evaluation
Expand LLM-as-judge beyond the current pilot.
Compare LLM-judge results against human ratings.
Measure agreement between human reviewers and the LLM judge.
19. Reproducibility

Install dependencies:

pip install -r requirements.txt

Run:

python run.py

The evaluation workflow produces:

outputs/
├── evaluation.csv
└── metrics.json

The core pipeline consists of:

src/
├── config.py
├── data_loader.py
├── labeling.py
├── classifier.py
├── retrieval.py
├── response_generator.py
├── escalation.py
└── evaluation.py
20. Limitations

The main limitations of the current prototype are:

The dataset consists of historical Twitter conversations and may not represent modern customer-support traffic.
The intent taxonomy was designed specifically for this dataset.
The current 200-example evaluation set is highly imbalanced.
Only 20/200 evaluation examples have currently been independently human-reviewed.
The LLM-as-judge evaluation is currently only a 5-example pilot.
TF-IDF retrieval can struggle with semantically similar messages that use different vocabulary.
The current response generator is intentionally conservative and therefore less personalized than a production system.
Escalation is based primarily on historical evidence availability rather than a learned risk model.
### Summary
This project demonstrates an end-to-end support-agent prototype combining:

Intent classification → historical evidence retrieval → conservative response generation → escalation

The current provisional evaluation shows:

87.0% intent accuracy
84.33% intent Macro F1
71.67% escalation F1
4.20/5 relevance
5.00/5 groundedness
4.80/5 helpfulness

The next major step is completing independent human verification of the golden dataset and evaluating the system against those verified labels.


### One thing I recommend before submitting

Don't paste this README into GitHub **yet** if your repo still only contains `config.py`, `data_loader.py`, and `run.py`. The README describes modules such as `classifier.py`, `retrieval.py`, etc., so the repo should actually contain them.

**Next, we should clean up your GitHub repository structure and code so the README matches the acutal repo**
