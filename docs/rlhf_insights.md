# RLHF Evaluation — Methodology Notes and Insights

**Scope:** Observations and methodology notes from constructing and evaluating the pairwise dataset in this repository.  
**Audience:** Reviewers assessing annotation quality and evaluation methodology.

---

## 1. Why Pairwise Ranking Over Absolute Scoring

Absolute scoring (rate this response 1–5) suffers from annotator calibration variance — different annotators apply the scale differently, making aggregation unreliable. Pairwise ranking (which of these two responses is better?) is a binary judgment that produces higher inter-annotator agreement and cleaner training signal.

The tradeoff: pairwise ranking does not capture the absolute quality floor. Two responses can both be poor, but ranking still forces a preference. The solution used here — pairwise preference with per-dimension absolute scoring on the chosen response — combines both approaches: clean preference signal plus calibrated quality measurement.

This is the same approach used in the InstructGPT and Constitutional AI literature, where human preference labels drive the reward model and dimension scores provide additional supervision signal.

---

## 2. Prompt Design Considerations

Prompts in this dataset were selected to cover five categories:

| Category | Examples in dataset | Skill demonstrated |
|----------|--------------------|--------------------|
| Factual recall | Capital of Germany, Ramayana authorship | Hallucination detection |
| Scientific explanation | Photosynthesis, Newton's Law, boiling point | STEM domain knowledge |
| NLP/ML concepts | Tokenisation, supervised vs unsupervised learning | NLP annotation expertise |
| Code quality | Prime number function | Technical evaluation, Python |
| Safety-critical | Neighbour dispute, public Wi-Fi risks | Content safety, policy taxonomy |

A well-constructed RLHF dataset covers all of these categories because different error types cluster by domain: factual errors dominate knowledge-retrieval prompts; conceptual errors dominate explanation prompts; instruction-following failures dominate structured task prompts; safety violations require dedicated adversarial prompts.

---

## 3. Score Calibration Observations

**Completeness is the most discriminating dimension.**  
Accuracy and safety tend toward 5 in well-chosen preference pairs (if the chosen response were inaccurate or unsafe, it should not be preferred). Completeness captures genuine variation — a correct, clear response can still fail to address implicit aspects of a question. Annotators should treat completeness as the primary quality discriminator once accuracy and safety are confirmed.

**Clarity requires audience awareness.**  
The photosynthesis entry scores Clarity 4 on the chosen response because the chemical equation (6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂) may not be accessible to a general audience. This illustrates that clarity is not absolute — it is relative to the implied audience of the prompt. Annotators should infer audience from prompt complexity and adjust clarity scores accordingly.

**Scoring the chosen response independently matters.**  
A common annotator error is to inflate chosen-response scores because the rejected response is obviously bad. This produces ceiling effects that reduce the usefulness of the score data for training. Each dimension score should reflect the absolute quality of the chosen response regardless of how poor the alternative is.

---

## 4. Safety Annotation Methodology

Safety evaluation in this dataset uses a two-stage approach:

**Stage 1 — Intent classification.** Before evaluating safety, classify the prompt's likely intent: benign, ambiguous, or adversarial. The neighbour dispute prompt is ambiguous; the Wi-Fi risks prompt is benign; a prompt asking for synthesis instructions for a controlled substance would be adversarial.

**Stage 2 — Response evaluation.** For ambiguous and adversarial prompts, evaluate whether the response:
- Interprets the prompt charitably (for ambiguous cases)
- Declines appropriately without over-refusing (for clearly adversarial cases)
- Provides safe, legal, and helpful alternatives where possible

**Over-refusal** is a safety failure in the opposite direction. A model that refuses to explain how vaccines work because the topic is "medical" is calibrated too conservatively. In this rubric, over-refusal on a clearly benign prompt scores Safety 3, not 5.

---

## 5. Lessons for Annotation Quality

**Lesson 1: Read the full prompt before reading the responses.**  
The summarisation task demonstrates how annotators miss embedded constraints (e.g., "in one sentence") if they skim the prompt. A consistent prompt-reading protocol — identify task type, format constraints, audience, and domain — reduces instruction-following errors in evaluation.

**Lesson 2: Fluency is not a proxy for accuracy.**  
LLMs produce fluent, confident text regardless of correctness. The most dangerous rejected responses in this dataset (Newton's Law, tokenisation) are grammatically natural. Evaluators must consciously decouple their reading experience from their factual assessment.

**Lesson 3: Error taxonomy improves consistency.**  
Labelling each rejection with an error type (see `evaluation_rubric.md`) forces explicit diagnosis rather than a vague sense that "something is wrong." It also makes downstream analysis tractable — you can filter by error type to understand which failure modes are most common in a given model.

**Lesson 4: Borderline cases require explicit documentation.**  
When a chosen response scores below 5 on any dimension, the rationale must explain why. Unexplained sub-5 scores are a quality signal failure — they suggest the annotator applied the rubric mechanically rather than analytically. Every scoring decision should be defensible in writing.