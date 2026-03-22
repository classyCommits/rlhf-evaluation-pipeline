# RLHF Response Evaluation System

A complete human-feedback annotation pipeline for LLM training data — covering pairwise preference ranking, multi-criteria rubric scoring, rationale documentation, and automated dataset quality auditing.

---

## Overview

This project implements the human evaluation stage of Reinforcement Learning from Human Feedback (RLHF). It replicates the annotation workflows used in production LLM training pipelines, where human evaluators compare model outputs, select the better response, assign structured quality scores, and document their reasoning.

The project demonstrates end-to-end competence across the full annotation lifecycle: prompt-response dataset design, rubric-based scoring, error taxonomy, rationale writing, safety labelling, and programmatic quality assurance.

---

## Repository Structure

```
.
├── app.py                   # Streamlit annotation UI
├── evaluator.py             # Backend: data loading, saving, export logic
├── data_quality_check.py    # Automated dataset auditor (schema + quality checks)
├── iaa_annotations.json     # Two-annotator IAA data (κ=0.737, simulated)
├── data/
│   └── data.json            # Input prompt-response pairs for annotation
├── results/                 # Auto-created on first submission (gitignored)             
├── exports/
│   └── rlhf_dataset.json    # Final training-ready export (flagged rows excluded)
└── docs/
    ├── evaluation_rubric.md # Full 5-point annotation rubric with edge case guidance
    ├── error_analysis.md    # Annotated error case studies and dataset statistics
    └── rlhf_insights.md     # Methodology notes and annotation quality observations
```

---

## Features

- **Pairwise preference ranking** — side-by-side A/B comparison with structured preference capture
- **Multi-criteria scoring** — four independent dimensions (Accuracy, Clarity, Completeness, Safety) each rated 1–5 with full rubric guidance
- **Error taxonomy** — ten error types covering factual errors, conceptual misrepresentation, safety violations, instruction-following failures, and code inefficiency
- **Rationale documentation** — free-text justification with quality heuristics enforced by the UI
- **Safety labelling** — hallucination and policy violation flags; flagged entries are excluded from the training export
- **Automated quality auditing** — `data_quality_check.py` validates schema, score ranges, rationale depth, error type coverage, and safety consistency
- **Completion workflow** — progress tracking, dataset-complete screen, and in-browser JSON download

---

## Setup

```bash
pip install pandas streamlit
streamlit run app.py
```

Python 3.10+ required (uses `list[dict]` and `str | None` type hints).

Place your input data at `data/data.json`. Each entry must include:

```json
{
    "prompt": "string",
    "response_a": "string",
    "response_b": "string"
}
```

---

## Running the Quality Auditor

```bash
# Audit the default export file
python data_quality_check.py

# Audit a specific file with full per-entry detail
python data_quality_check.py --file exports/rlhf_dataset.json --verbose

# Run inter-annotator agreement report (auto-generates iaa_annotations.json on first run)
python data_quality_check.py --iaa

# Run IAA against a real annotator export
python data_quality_check.py --iaa --iaa-file path/to/real_annotations.json
```

The auditor checks:
- Schema completeness (required fields, score keys)
- Score value validity (integer, 1–5 range)
- Rationale depth (minimum word count, references to both responses)
- Error type validity against the defined taxonomy
- Safety score consistency (safety=1 entries flagged for exclusion)
- Duplicate detection (identical prompts or response pairs)
- Dataset coverage (warns if safety, NLP, or code categories are missing)
- Inter-annotator agreement (Cohen's Kappa on preference labels, Spearman's rho on scores)

---

## Evaluation Dataset

`exports/rlhf_dataset.json` contains 10 annotated preference pairs covering:

| Domain | Error type demonstrated |
|--------|------------------------|
| Factual knowledge | `factual_error` (×2) |
| Scientific explanation | `conceptual_misrepresentation` |
| Source attribution | `attribution_error` |
| Code quality (Python) | `inefficiency` |
| NLP concepts | `oversimplification` |
| Content safety | `safety_violation` |
| ML concepts | `imprecise_metaphor` |
| Instruction following | `incomplete_instruction_following` |
| Analytical depth | `vague_oversimplification` |

Scores are calibrated: chosen responses are scored independently on each dimension, not inflated relative to the rejected response. Six of ten entries score below 5 on Completeness, reflecting realistic annotation rather than ceiling effects.

---

## Key Design Decisions

**Why pairwise ranking over absolute scoring alone?**
Pairwise preference produces higher inter-annotator agreement than absolute scales because it eliminates calibration variance. This project combines both: pairwise preference for the training signal and per-dimension scores for quality measurement. This mirrors the approach in InstructGPT (Ouyang et al., 2022).

**Why are flagged entries retained in results.csv but excluded from the export?**
Retaining flagged entries preserves audit history and allows review of annotation decisions. The export pipeline filters them out so they do not enter the reward model training set. The quality auditor reports the excluded count explicitly.

**Why is the error taxonomy included in the dataset?**
Error type labels make downstream analysis tractable. Without taxonomy, you can count how many entries were rejected but not why. With taxonomy, you can measure which failure modes are most common in a given model, which informs targeted fine-tuning decisions.

---

## References

- Ouyang et al. (2022). *Training language models to follow instructions with human feedback.* NeurIPS. (InstructGPT)
- Bai et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* Anthropic.
- Ziegler et al. (2019). *Fine-tuning language models from human preferences.* arXiv.