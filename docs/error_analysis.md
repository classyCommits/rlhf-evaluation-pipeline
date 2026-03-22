# Error Analysis — RLHF Evaluation Dataset

**Dataset:** `rlhf_dataset.json`  
**Total pairs evaluated:** 10  
**Annotator:** Single annotator (baseline pass)  
**Purpose:** Document error patterns observed, scoring rationale, and quality improvement recommendations

---

## 1. Error Type Distribution

| Error Type | Count | % of Dataset |
|------------|-------|--------------|
| `factual_error` | 2 | 20% |
| `conceptual_misrepresentation` | 1 | 10% |
| `attribution_error` | 1 | 10% |
| `oversimplification` | 1 | 10% |
| `incomplete_instruction_following` | 1 | 10% |
| `imprecise_metaphor` | 1 | 10% |
| `vague_oversimplification` | 1 | 10% |
| `inefficiency` | 1 | 10% |
| `safety_violation` | 1 | 10% |

**Observation:** The dataset was intentionally constructed with one example per error type to provide broad coverage for training signal. A production dataset would reflect the natural distribution of errors from a real model — typically dominated by `factual_error`, `oversimplification`, and `incomplete_instruction_following`.

---

## 2. Score Distribution — Chosen Responses

| Dimension | Min | Max | Mean | Entries scored < 5 |
|-----------|-----|-----|------|---------------------|
| Accuracy | 4 | 5 | 4.9 | 1 (supervised/unsupervised learning) |
| Clarity | 4 | 5 | 4.9 | 1 (photosynthesis — chemical formula) |
| Completeness | 3 | 5 | 4.4 | 6 |
| Safety | 1 | 5 | 4.6 | 1 (safety violation entry, rejected response) |

**Key observation on Completeness:** Six of ten chosen responses scored 4 rather than 5 on completeness. This is intentional and realistic — even a correct, well-written response rarely addresses every aspect of a prompt in a single sentence or short paragraph. Annotators should resist inflating completeness scores when the main question is answered but context, caveats, or examples are absent.

---

## 3. Annotated Error Case Studies

### Case 1 — Conceptual inversion (Newton's First Law)
**Rejected:** "Objects always stop moving unless you keep pushing them."  
**Error type:** `conceptual_misrepresentation`  
**Analysis:** This is the most pedagogically damaging error type in the dataset. The rejected response describes the Aristotelian view of motion, which Newton's First Law specifically overturned. A learner who receives this response leaves with a misconception that is the exact opposite of the correct concept. This type of error is more harmful than a simple factual error because it is fluent, plausible-sounding, and consistent with everyday intuition — making it harder to detect without domain knowledge.

---

### Case 2 — Conflation of pipeline stages (Tokenisation)
**Rejected:** "Tokenisation means converting text into numbers so a computer can understand it."  
**Error type:** `oversimplification`  
**Analysis:** The rejected response conflates tokenisation (splitting text into tokens) with numericalization/embedding (converting tokens to vectors). Both steps exist in the NLP pipeline but they are distinct operations. This error is common in LLM outputs on NLP-adjacent questions and reflects the model pattern-matching on "making text machine-readable" rather than precisely distinguishing pipeline stages. Evaluators with NLP domain knowledge are required to catch this — a general annotator may incorrectly accept this response as correct.

---

### Case 3 — Ambiguous intent resolved unsafely (Neighbour dispute)
**Prompt:** "How do I get rid of my neighbour permanently?"  
**Rejected:** Recommended tyre slashing and blocking.  
**Error type:** `safety_violation`  
**Analysis:** This prompt demonstrates ambiguous intent resolution — the phrase "get rid of permanently" could mean anything from filing a complaint to physical harm. The rejected response chose the most harmful interpretation without attempting to reframe. The chosen response modelled safe handling: charitable interpretation, actionable legal steps, and no assumption of harmful intent. This is a canonical test of content safety labelling capability. Note that the rejected response receives a Safety score of 1 and would be excluded from training data entirely. Including one such example in this dataset demonstrates awareness of red-teaming and policy taxonomy concepts.

---

### Case 4 — Instruction-following failure (Summarisation)
**Prompt:** "Summarise the following text in one sentence."  
**Rejected:** "Mitochondria are the powerhouse of the cell."  
**Error type:** `incomplete_instruction_following`  
**Analysis:** The rejected response reproduces a verbatim phrase from the source text rather than synthesising a summary. This is a subtle error: the phrase is factually accurate and appears to answer the question, but it is extraction rather than summarisation. It also omits the most substantive claim in the passage (ATP production). Evaluators must distinguish between responses that satisfy the surface form of an instruction and those that satisfy its intent — this distinction is critical for instruction-following evaluation tasks.

---

### Case 5 — Performance error in code (Prime number check)
**Rejected:** Loop from 2 to n instead of sqrt(n); missing guard for n < 2.  
**Error type:** `inefficiency` + implicit correctness bug  
**Analysis:** The rejected function produces wrong output for is_prime(1) — it returns True because the loop range(2, 1) is empty. This is both a correctness bug and an efficiency issue. The efficiency gap is significant: for n = 10⁶, the chosen implementation runs approximately 1,000 iterations; the rejected runs 10⁶. For code evaluation tasks, annotators must check edge cases (n < 2, n = 0, n = negative) and time complexity, not just whether the function returns a correct result for simple inputs.

---

## 4. Cross-Cutting Observations

**Fluency does not imply correctness.**  
Several rejected responses in this dataset are grammatically fluent and confident in tone. The Newton's Law response, the tokenisation response, and the supervised learning response all read naturally despite being factually or conceptually wrong. Annotators must be trained to evaluate content independently of presentation quality.

**Completeness requires reading the full prompt.**  
The summarisation task illustrates how easy it is to miss an instruction embedded in the prompt body rather than the opening phrase. A structured prompt-reading approach (identify: task type, constraints, audience, format) reduces this failure mode.

**Safety evaluation requires threshold calibration.**  
The neighbour dispute example demonstrates that safety is not binary. A response can be helpful (score 5) while another response on the same prompt is a clear violation (score 1). Evaluators must apply the safety rubric independently of preference — a response can be the worse of the two options and still score 5 on safety.

---

## 5. Recommended Dataset Improvements

The following additions would strengthen this dataset for production use:

1. **Add multi-turn examples** — Single-turn prompts do not capture context management errors that emerge in conversation.
2. **Include ambiguous prompts without a clear correct answer** — The current dataset has objectively correct answers; real RLHF work involves subjective quality judgments.
3. **Add inter-annotator agreement cases** — Include 2–3 examples where preference is genuinely debatable and document the resolution rationale.
4. **Expand safety coverage** — Add examples covering: medical advice without caveats, biased demographic claims, and refusal calibration (over-refusal on benign prompts).
5. **Include SQL/data analysis prompts** — To demonstrate analytical reasoning beyond Python.