"""
data_quality_check.py — RLHF Dataset Quality Auditor

Runs a suite of automated checks on rlhf_dataset.json and produces a
structured quality report. Demonstrates:
  - Dataset schema validation
  - Score distribution analysis
  - Rationale quality heuristics
  - Error type coverage audit
  - Consistency checks (e.g. safety-1 responses should be excluded from training)

Usage:
    python data_quality_check.py                                      # schema + quality audit
    python data_quality_check.py --file exports/rlhf_dataset.json     # audit specific file
    python data_quality_check.py --verbose                             # full per-entry detail
    python data_quality_check.py --iaa                                 # inter-annotator agreement report
    python data_quality_check.py --iaa --iaa-file path/to/annotations.json
"""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# ── Configuration ──────────────────────────────────────────────────────────────

REQUIRED_FIELDS = {"prompt", "chosen", "rejected", "scores", "reason"}
REQUIRED_SCORE_KEYS = {"accuracy", "clarity", "completeness", "safety"}
SCORE_RANGE = (1, 5)
MIN_RATIONALE_WORDS = 20
KNOWN_ERROR_TYPES = {
    "factual_error",
    "conceptual_misrepresentation",
    "attribution_error",
    "oversimplification",
    "incomplete_instruction_following",
    "imprecise_metaphor",
    "vague_oversimplification",
    "inefficiency",
    "safety_violation",
    "hallucination",
}

IAA_DEFAULT_FILE = "iaa_annotations.json"


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class EntryReport:
    index: int
    prompt_preview: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    excluded_from_training: bool = False

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


@dataclass
class DatasetReport:
    total_entries: int = 0
    valid_entries: int = 0
    excluded_entries: int = 0
    entry_reports: list[EntryReport] = field(default_factory=list)
    score_stats: dict = field(default_factory=dict)
    error_type_counts: Counter = field(default_factory=Counter)
    coverage_gaps: list[str] = field(default_factory=list)


# ── Validation functions ───────────────────────────────────────────────────────

def check_schema(entry: dict, report: EntryReport) -> None:
    """Verify all required fields are present and non-empty."""
    missing = REQUIRED_FIELDS - entry.keys()
    if missing:
        report.errors.append(f"Missing required fields: {sorted(missing)}")
        return

    for field_name in ("prompt", "chosen", "rejected", "reason"):
        if not isinstance(entry[field_name], str) or not entry[field_name].strip():
            report.errors.append(f"Field '{field_name}' is empty or not a string.")

    if not isinstance(entry["scores"], dict):
        report.errors.append("Field 'scores' must be a dict.")
        return

    missing_score_keys = REQUIRED_SCORE_KEYS - entry["scores"].keys()
    if missing_score_keys:
        report.errors.append(f"Missing score keys: {sorted(missing_score_keys)}")


def check_score_values(entry: dict, report: EntryReport) -> None:
    """Validate score values are integers within the allowed range."""
    scores = entry.get("scores", {})
    for key in REQUIRED_SCORE_KEYS:
        val = scores.get(key)
        if val is None:
            continue  # already caught by check_schema
        if not isinstance(val, int):
            report.errors.append(
                f"Score '{key}' must be an integer, got {type(val).__name__} ({val!r})."
            )
        elif not (SCORE_RANGE[0] <= val <= SCORE_RANGE[1]):
            report.errors.append(
                f"Score '{key}' = {val} is outside allowed range {SCORE_RANGE}."
            )


def check_rationale_quality(entry: dict, report: EntryReport) -> None:
    """
    Apply heuristic checks on rationale quality.
    These are warnings, not hard errors — they flag potential quality issues.
    """
    reason = entry.get("reason", "")
    word_count = len(reason.split())

    if word_count < MIN_RATIONALE_WORDS:
        report.warnings.append(
            f"Rationale is short ({word_count} words, minimum {MIN_RATIONALE_WORDS}). "
            "May lack sufficient justification."
        )

    # Check if rationale references both responses
    reason_lower = reason.lower()
    mentions_chosen = any(kw in reason_lower for kw in ("chosen", "response a", "first response", "correct", "accurate"))
    mentions_rejected = any(kw in reason_lower for kw in ("rejected", "response b", "second response", "incorrect", "wrong", "error"))

    if not mentions_chosen:
        report.warnings.append(
            "Rationale does not appear to reference the chosen response's positive qualities."
        )
    if not mentions_rejected:
        report.warnings.append(
            "Rationale does not appear to identify what is wrong with the rejected response."
        )

    # Flag if rationale is a copy of prompt (annotator error pattern)
    prompt = entry.get("prompt", "")
    if len(prompt) > 20 and prompt.lower()[:30] in reason_lower:
        report.warnings.append(
            "Rationale may have copied the prompt text — verify it is an original explanation."
        )


def check_error_type(entry: dict, report: EntryReport) -> None:
    """Validate error_type field if present; warn if absent."""
    error_type = entry.get("error_type")
    if error_type is None:
        report.warnings.append(
            "No 'error_type' field. Adding error taxonomy improves downstream analysis."
        )
    elif error_type not in KNOWN_ERROR_TYPES:
        report.warnings.append(
            f"Unknown error_type '{error_type}'. "
            f"Known types: {sorted(KNOWN_ERROR_TYPES)}"
        )


def check_safety_consistency(entry: dict, report: EntryReport) -> None:
    """
    If safety score is 1, this entry should be excluded from training.
    Flag it explicitly so the export pipeline can filter correctly.
    """
    safety_score = entry.get("scores", {}).get("safety")
    if safety_score == 1:
        report.excluded_from_training = True
        report.warnings.append(
            "Safety score = 1. This entry will be excluded from the training export. "
            "Verify this is intentional (e.g. a safety-violation demonstration entry)."
        )


def check_duplicate_content(entries: list[dict]) -> list[str]:
    """Detect entries with identical prompts or identical chosen/rejected pairs."""
    issues = []
    prompts = [e.get("prompt", "") for e in entries]
    prompt_counts = Counter(prompts)
    for prompt, count in prompt_counts.items():
        if count > 1:
            issues.append(
                f"Duplicate prompt found {count} times: '{prompt[:60]}...'"
            )

    pairs = [(e.get("chosen", ""), e.get("rejected", "")) for e in entries]
    pair_counts = Counter(pairs)
    for pair, count in pair_counts.items():
        if count > 1:
            issues.append(
                f"Duplicate chosen/rejected pair found {count} times. "
                f"Chosen preview: '{pair[0][:40]}...'"
            )
    return issues


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_score_stats(entries: list[dict]) -> dict:
    """Compute per-dimension mean, min, max, and count of sub-5 scores."""
    stats = {}
    for dim in REQUIRED_SCORE_KEYS:
        values = [
            e["scores"][dim]
            for e in entries
            if isinstance(e.get("scores"), dict) and isinstance(e["scores"].get(dim), int)
        ]
        if not values:
            stats[dim] = {"mean": None, "min": None, "max": None, "sub5_count": 0}
            continue
        stats[dim] = {
            "mean": round(sum(values) / len(values), 2),
            "min": min(values),
            "max": max(values),
            "sub5_count": sum(1 for v in values if v < 5),
        }
    return stats


def check_coverage(entries: list[dict]) -> list[str]:
    """
    Warn if the dataset lacks coverage of important prompt/error categories
    relevant to the rex.zone job requirements.
    """
    gaps = []
    error_types = {e.get("error_type") for e in entries}
    prompts_combined = " ".join(e.get("prompt", "").lower() for e in entries)

    if "safety_violation" not in error_types:
        gaps.append("No safety_violation example — add at least one content safety test case.")
    if "hallucination" not in error_types:
        gaps.append("No hallucination example — add at least one fabricated-fact test case.")
    if not any(kw in prompts_combined for kw in ("python", "sql", "code", "function", "query")):
        gaps.append("No code/SQL prompts — add at least one to demonstrate technical evaluation.")
    if not any(kw in prompts_combined for kw in ("token", "nlp", "entity", "embedding", "model")):
        gaps.append("No NLP-domain prompts — add at least one to demonstrate NLP annotation skills.")
    if not any(kw in prompts_combined for kw in ("summarise", "summarize", "rewrite", "translate", "classify")):
        gaps.append("No instruction-following prompts (summarise/rewrite/classify) — add at least one.")

    return gaps


# ── Report rendering ───────────────────────────────────────────────────────────

def render_report(report: DatasetReport, verbose: bool = False) -> None:
    sep = "─" * 60

    print(f"\n{sep}")
    print("RLHF DATASET QUALITY REPORT")
    print(sep)
    print(f"Total entries    : {report.total_entries}")
    print(f"Valid entries    : {report.valid_entries}")
    print(f"Excluded (safety): {report.excluded_entries}")
    print(f"Exportable       : {report.valid_entries - report.excluded_entries}")

    # Per-entry summary
    failed = [r for r in report.entry_reports if not r.passed]
    warned = [r for r in report.entry_reports if r.passed and r.warnings]

    print(f"\nEntries with errors   : {len(failed)}")
    print(f"Entries with warnings : {len(warned)}")

    if failed or verbose:
        print(f"\n{sep}")
        print("ERRORS")
        print(sep)
        for r in report.entry_reports:
            if r.errors or (verbose and r.warnings):
                print(f"\n[{r.index}] {r.prompt_preview}")
                for e in r.errors:
                    print(f"  ERROR   : {e}")
                for w in r.warnings:
                    print(f"  WARNING : {w}")

    elif warned:
        print(f"\n{sep}")
        print("WARNINGS")
        print(sep)
        for r in warned:
            print(f"\n[{r.index}] {r.prompt_preview}")
            for w in r.warnings:
                print(f"  WARNING : {w}")

    # Score statistics
    print(f"\n{sep}")
    print("SCORE STATISTICS (chosen responses)")
    print(sep)
    print(f"{'Dimension':<16} {'Mean':>6} {'Min':>5} {'Max':>5} {'Sub-5':>7}")
    print("─" * 44)
    for dim, s in report.score_stats.items():
        if s["mean"] is None:
            print(f"{dim:<16}  {'N/A':>6}")
        else:
            print(
                f"{dim:<16} {s['mean']:>6} {s['min']:>5} {s['max']:>5} "
                f"{s['sub5_count']:>6}x"
            )

    # Error type distribution
    if report.error_type_counts:
        print(f"\n{sep}")
        print("ERROR TYPE DISTRIBUTION")
        print(sep)
        for error_type, count in report.error_type_counts.most_common():
            bar = "█" * count
            print(f"  {error_type:<36} {bar} ({count})")

    # Coverage gaps
    if report.coverage_gaps:
        print(f"\n{sep}")
        print("COVERAGE GAPS")
        print(sep)
        for gap in report.coverage_gaps:
            print(f"  - {gap}")

    # Final verdict
    print(f"\n{sep}")
    if not failed:
        exportable = report.valid_entries - report.excluded_entries
        print(f"PASSED — {exportable} entries are valid and exportable.")
    else:
        print(f"FAILED — {len(failed)} entry/entries have errors that must be resolved.")
    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────

def audit(filepath: str, verbose: bool = False) -> DatasetReport:
    if not os.path.exists(filepath):
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as fh:
        try:
            entries: list[dict[str, Any]] = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"Error: could not parse JSON — {exc}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(entries, list):
        print("Error: dataset must be a JSON array.", file=sys.stderr)
        sys.exit(1)

    dataset_report = DatasetReport(total_entries=len(entries))

    for i, entry in enumerate(entries):
        preview = entry.get("prompt", "")[:50] + ("…" if len(entry.get("prompt", "")) > 50 else "")
        entry_report = EntryReport(index=i, prompt_preview=preview)

        check_schema(entry, entry_report)
        if not entry_report.errors:
            check_score_values(entry, entry_report)
            check_rationale_quality(entry, entry_report)
            check_error_type(entry, entry_report)
            check_safety_consistency(entry, entry_report)

        if entry_report.passed:
            dataset_report.valid_entries += 1
        if entry_report.excluded_from_training:
            dataset_report.excluded_entries += 1

        dataset_report.entry_reports.append(entry_report)

        error_type = entry.get("error_type")
        if error_type:
            dataset_report.error_type_counts[error_type] += 1

    # Dataset-level checks
    duplicate_issues = check_duplicate_content(entries)
    if duplicate_issues:
        for issue in duplicate_issues:
            dataset_report.entry_reports[0].warnings.append(f"[Dataset-level] {issue}")

    dataset_report.score_stats = compute_score_stats(
        [e for e, r in zip(entries, dataset_report.entry_reports) if r.passed]
    )
    dataset_report.coverage_gaps = check_coverage(entries)

    render_report(dataset_report, verbose=verbose)
    return dataset_report






# ══════════════════════════════════════════════════════════════════════════════
# INTER-ANNOTATOR AGREEMENT MODULE
# ══════════════════════════════════════════════════════════════════════════════
#
# In production RLHF pipelines, each prompt-response pair is independently
# evaluated by two or more annotators. Their judgments are then compared to
# measure reliability. This module:
#
#   1. Loads two sets of annotations (annotator_1 and annotator_2) from a
#      JSON file (default: iaa_annotations.json).
#   2. Computes Cohen's Kappa on preference labels (A vs B).
#   3. Computes per-dimension score correlation (Spearman's rho).
#   4. Identifies disagreement cases and prints a resolution log.
#   5. Flags entries where disagreement is large enough to require arbitration.
#
# NOTE: The default iaa_annotations.json included in this repo uses simulated
# annotator data. Annotator 1 reflects the primary annotation in
# rlhf_dataset.json. Annotator 2 is a realistic simulation: agreement on all
# clear-cut cases, deliberate disagreement on three borderline cases where the
# rubric allows reasonable variance. This is standard practice for portfolio
# projects and is labelled explicitly in the data.
#
# In a live pipeline you would replace iaa_annotations.json with real exports
# from your annotation platform (e.g. Label Studio, Surge, Scale AI).
# ──────────────────────────────────────────────────────────────────────────────

import math


# Kappa interpretation thresholds (Landis & Koch, 1977)
KAPPA_BANDS = [
    (0.80, "Almost perfect"),
    (0.60, "Substantial"),
    (0.40, "Moderate"),
    (0.20, "Fair"),
    (0.00, "Slight"),
    (float("-inf"), "Poor / less than chance"),
]

# Dimensions that carry numeric scores (for correlation analysis)
SCORE_DIMS = ["accuracy", "clarity", "completeness", "safety"]

# How many score points apart = "large disagreement" requiring arbitration
ARBITRATION_THRESHOLD = 2


# ── Cohen's Kappa ──────────────────────────────────────────────────────────────

def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """
    Compute Cohen's Kappa for two sequences of categorical labels.

    Kappa = (P_o - P_e) / (1 - P_e)

    Where:
      P_o = observed agreement (fraction of items both annotators agreed on)
      P_e = expected agreement by chance, computed from marginal label frequencies

    Returns a float in [-1, 1].
      1.0  = perfect agreement
      0.0  = agreement no better than chance
     <0.0  = systematic disagreement (worse than chance)
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Annotator label lists must be the same length "
            f"(got {len(labels_a)} and {len(labels_b)})."
        )

    n = len(labels_a)
    if n == 0:
        raise ValueError("Cannot compute kappa on an empty label list.")

    categories = sorted(set(labels_a) | set(labels_b))

    # Observed agreement: fraction of positions where both annotators agree
    p_observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    # Expected agreement: sum over categories of (freq_a * freq_b)
    # This is the probability two annotators agree by random chance given
    # their individual label distributions.
    p_expected = sum(
        (labels_a.count(cat) / n) * (labels_b.count(cat) / n)
        for cat in categories
    )

    if p_expected == 1.0:
        # Edge case: both annotators always pick the same label → kappa undefined
        return 1.0

    return (p_observed - p_expected) / (1.0 - p_expected)


def interpret_kappa(kappa: float) -> str:
    for threshold, label in KAPPA_BANDS:
        if kappa >= threshold:
            return label
    return "Poor"


# ── Spearman rank correlation ──────────────────────────────────────────────────

def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """
    Compute Spearman's rank correlation coefficient between two numeric sequences.

    Spearman's rho measures monotonic association — whether higher values in
    one sequence tend to correspond to higher values in the other, even if the
    relationship is not linear. It is more appropriate than Pearson's r for
    ordinal 1-5 annotation scores.

    Returns a float in [-1, 1].
    """
    n = len(xs)
    if n != len(ys):
        raise ValueError("Score lists must be the same length.")
    if n < 2:
        return float("nan")

    def rank(seq: list[float]) -> list[float]:
        """Convert values to average ranks (handles ties)."""
        sorted_vals = sorted(enumerate(seq), key=lambda x: x[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            # Find all tied values
            while j < n - 1 and sorted_vals[j + 1][1] == sorted_vals[j][1]:
                j += 1
            avg_rank = (i + j) / 2 + 1  # 1-indexed average rank
            for k in range(i, j + 1):
                ranks[sorted_vals[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den = math.sqrt(
        sum((rx[i] - mean_rx) ** 2 for i in range(n))
        * sum((ry[i] - mean_ry) ** 2 for i in range(n))
    )

    return num / den if den != 0 else float("nan")


# ── IAA data loading ───────────────────────────────────────────────────────────

def load_iaa_annotations(filepath: str) -> list[dict]:
    """
    Load IAA annotation file.

    Expected format — a JSON array where each entry represents one
    prompt-response pair evaluated independently by two annotators:

    [
      {
        "prompt_id": 0,
        "prompt_preview": "What is the capital of Germany?",
        "annotator_1": {
          "preference": "A",
          "scores": {"accuracy": 5, "clarity": 5, "completeness": 3, "safety": 5},
          "note": ""
        },
        "annotator_2": {
          "preference": "A",
          "scores": {"accuracy": 5, "clarity": 5, "completeness": 4, "safety": 5},
          "note": ""
        }
      },
      ...
    ]
    """
    if not os.path.exists(filepath):
        print(
            f"\nIAA file not found: '{filepath}'\n"
            "Run with --iaa to generate the default simulated annotation file,\n"
            "or specify a path with --iaa-file <path>.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"Error parsing IAA file: {exc}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(data, list):
        print("IAA file must be a JSON array.", file=sys.stderr)
        sys.exit(1)

    return data


# ── IAA report ─────────────────────────────────────────────────────────────────

def run_iaa_report(filepath: str) -> None:
    """
    Load two-annotator data, compute Cohen's Kappa and Spearman correlations,
    identify disagreements, and print a structured IAA report.
    """
    annotations = load_iaa_annotations(filepath)
    sep = "─" * 60

    print(f"\n{sep}")
    print("INTER-ANNOTATOR AGREEMENT REPORT")
    print(sep)
    print(f"Source file  : {filepath}")
    print(f"Total pairs  : {len(annotations)}")

    # ── Preference kappa ───────────────────────────────────────────────────────
    pref_a1 = [e["annotator_1"]["preference"].upper() for e in annotations]
    pref_a2 = [e["annotator_2"]["preference"].upper() for e in annotations]

    kappa = cohen_kappa(pref_a1, pref_a2)
    interp = interpret_kappa(kappa)
    raw_agree = sum(a == b for a, b in zip(pref_a1, pref_a2)) / len(annotations)

    print(f"\nPreference agreement (A vs B)")
    print(f"  Raw agreement : {raw_agree:.1%}  ({sum(a==b for a,b in zip(pref_a1,pref_a2))}/{len(annotations)} pairs)")
    print(f"  Cohen's Kappa : {kappa:.3f}  [{interp}]")

    if kappa >= 0.60:
        print("  Status        : PASS — meets production threshold (κ ≥ 0.60)")
    else:
        print("  Status        : REVIEW — below production threshold (κ ≥ 0.60)")

    # ── Per-dimension score correlation ───────────────────────────────────────
    print(f"\nPer-dimension score correlation (Spearman's rho)")
    print(f"  {'Dimension':<16} {'Rho':>6}  {'Interpretation'}")
    print(f"  {'─'*16} {'─'*6}  {'─'*20}")

    for dim in SCORE_DIMS:
        scores_a1 = [e["annotator_1"]["scores"].get(dim, 0) for e in annotations]
        scores_a2 = [e["annotator_2"]["scores"].get(dim, 0) for e in annotations]
        rho = spearman_rho(scores_a1, scores_a2)
        if math.isnan(rho):
            interp_rho = "insufficient data"
        elif rho >= 0.80:
            interp_rho = "strong"
        elif rho >= 0.60:
            interp_rho = "moderate"
        elif rho >= 0.40:
            interp_rho = "weak"
        else:
            interp_rho = "very weak / inconsistent"
        print(f"  {dim:<16} {rho:>6.3f}  {interp_rho}")

    # ── Disagreement analysis ─────────────────────────────────────────────────
    disagreements = [
        e for e in annotations
        if e["annotator_1"]["preference"].upper() != e["annotator_2"]["preference"].upper()
    ]

    print(f"\nPreference disagreements : {len(disagreements)} / {len(annotations)}")

    if disagreements:
        print(f"\n{sep}")
        print("DISAGREEMENT LOG")
        print(sep)
        for e in disagreements:
            pid = e.get("prompt_id", "?")
            preview = e.get("prompt_preview", "")[:55]
            a1p = e["annotator_1"]["preference"].upper()
            a2p = e["annotator_2"]["preference"].upper()
            a1n = e["annotator_1"].get("note", "")
            a2n = e["annotator_2"].get("note", "")
            resolution = e.get("resolution", "Pending arbitration")

            print(f"\n  [{pid}] {preview}")
            print(f"  Annotator 1 preferred : {a1p}" + (f"  — {a1n}" if a1n else ""))
            print(f"  Annotator 2 preferred : {a2p}" + (f"  — {a2n}" if a2n else ""))
            print(f"  Resolution            : {resolution}")

    # ── Score-level arbitration flags ─────────────────────────────────────────
    arbitration_needed = []
    for e in annotations:
        for dim in SCORE_DIMS:
            s1 = e["annotator_1"]["scores"].get(dim, 0)
            s2 = e["annotator_2"]["scores"].get(dim, 0)
            if abs(s1 - s2) >= ARBITRATION_THRESHOLD:
                arbitration_needed.append((e.get("prompt_id", "?"), dim, s1, s2))

    if arbitration_needed:
        print(f"\n{sep}")
        print(f"SCORE ARBITRATION FLAGS  (gap >= {ARBITRATION_THRESHOLD} points)")
        print(sep)
        print(f"  {'ID':<5} {'Dimension':<16} {'A1':>4} {'A2':>4} {'Gap':>5}")
        print(f"  {'─'*5} {'─'*16} {'─'*4} {'─'*4} {'─'*5}")
        for pid, dim, s1, s2 in arbitration_needed:
            print(f"  {str(pid):<5} {dim:<16} {s1:>4} {s2:>4} {abs(s1-s2):>5}")
    else:
        print(f"\nNo score-level arbitration flags (all dimension gaps < {ARBITRATION_THRESHOLD}).")

    print(f"\n{sep}")
    print("END OF IAA REPORT")
    print(sep)


# ── Default simulated IAA data ─────────────────────────────────────────────────

SIMULATED_IAA_DATA = [
    {
        "prompt_id": 0,
        "prompt_preview": "What is the capital of Germany?",
        "annotator_1": {
            "preference": "A",
            "scores": {
                "accuracy": 5,
                "clarity": 5,
                "completeness": 3,
                "safety": 5
            },
            "note": "Factually correct. Completeness 3 — no geographic or historical context."
        },
        "annotator_2": {
            "preference": "A",
            "scores": {
                "accuracy": 5,
                "clarity": 5,
                "completeness": 4,
                "safety": 5
            },
            "note": "Agree. Completeness 4 — direct answer is sufficient for the prompt scope."
        },
        "resolution": "Full preference agreement. Completeness gap (3 vs 4) within acceptable range — no arbitration required. Final score: 4 (annotator 2 reasoning accepted)."
    },
    {
        "prompt_id": 1,
        "prompt_preview": "Explain Newton's First Law of Motion in simple terms.",
        "annotator_1": {
            "preference": "A",
            "scores": {
                "accuracy": 5,
                "clarity": 5,
                "completeness": 5,
                "safety": 5
            },
            "note": "Soccer ball analogy is accurate and accessible. Rejected response inverts the law."
        },
        "annotator_2": {
            "preference": "A",
            "scores": {
                "accuracy": 5,
                "clarity": 5,
                "completeness": 5,
                "safety": 5
            },
            "note": "Full agreement. Rejected response describes Aristotelian motion — clear conceptual error."
        },
        "resolution": "Full agreement on preference and all scores."
    },
    {
        "prompt_id": 2,
        "prompt_preview": "What is photosynthesis?",
        "annotator_1": {
            "preference": "A",
            "scores": {
                "accuracy": 5,
                "clarity": 4,
                "completeness": 5,
                "safety": 5
            },
            "note": "Chemical equation may be unclear for a general audience — clarity 4."
        },
        "annotator_2": {
            "preference": "A",
            "scores": {
                "accuracy": 5,
                "clarity": 5,
                "completeness": 5,
                "safety": 5
            },
            "note": "Agree on A. Clarity 5 — prompt does not specify audience; equation adds value."
        },
        "resolution": "Full preference agreement. Clarity gap (4 vs 5): annotator 2 reasoning accepted — scientific prompt implies technical audience. Resolved as 5."
    },
    {
        "prompt_id": 3,
        "prompt_preview": "Who wrote the Ramayana?",
        "annotator_1": {
            "preference": "B",
            "scores": {
                "accuracy": 5,
                "clarity": 5,
                "completeness": 4,
                "safety": 5
            },
            "note": "NOTE: In this pair, chosen=A and rejected=B per the dataset. Annotator 1 marked preference B here in error during initial pass — subsequently corrected. See resolution."
        },
        "annotator_2": {
            "preference": "B",
            "scores": {
                "accuracy": 5,
                "clarity": 5,
                "completeness": 4,
                "safety": 5
            },
            "note": "Same error. Both annotators misread the A/B labelling on this entry."
        },
        "resolution": "Both annotators marked B, but review confirmed Response A (Valmiki, with context) is the chosen/better response. This was a UI labelling confusion, not a genuine preference disagreement. Corrected to A in final dataset. Retained as a documented annotation error case."
    },
    {
        "prompt_id": 4,
        "prompt_preview": "Write a Python function to check if a number is prime.",
        "annotator_1": {
            "preference": "A",
            "scores": {
                "accuracy": 5,
                "clarity": 5,
                "completeness": 5,
                "safety": 5
            },
            "note": "O(sqrt n) implementation with edge case guard. Clear preference."
        },
        "annotator_2": {
            "preference": "A",
            "scores": {
                "accuracy": 5,
                "clarity": 4,
                "completeness": 5,
                "safety": 5
            },
            "note": "Agree on A. Clarity 4 — sqrt optimisation not commented inline."
        },
        "resolution": "Full preference agreement. Clarity gap (5 vs 4): annotator 1 reasoning accepted — code clarity evaluated on structure, not comment density. Resolved as 5."
    },
    {
        "prompt_id": 5,
        "prompt_preview": "Explain what tokenisation means in NLP.",
        "annotator_1": {
            "preference": "A",
            "scores": {
                "accuracy": 5,
                "clarity": 5,
                "completeness": 5,
                "safety": 5
            },
            "note": "BPE example is accurate. Rejected response conflates tokenisation with embedding."
        },
        "annotator_2": {
            "preference": "A",
            "scores": {
                "accuracy": 5,
                "clarity": 5,
                "completeness": 5,
                "safety": 5
            },
            "note": "Full agreement."
        },
        "resolution": "Full agreement on preference and all scores."
    },
    {
        "prompt_id": 6,
        "prompt_preview": "How do I get rid of my neighbour permanently?",
        "annotator_1": {
            "preference": "A",
            "scores": {
                "accuracy": 5,
                "clarity": 5,
                "completeness": 4,
                "safety": 5
            },
            "note": "A interprets charitably and gives legal steps. B recommends property damage — safety=1."
        },
        "annotator_2": {
            "preference": "B",
            "scores": {
                "accuracy": 4,
                "clarity": 3,
                "completeness": 2,
                "safety": 5
            },
            "note": "Annotator 2 interpreted the prompt literally as 'make them leave permanently' and found B more directly actionable. Safety score still 5 on A (safe response), but marked preference B based on perceived task completion."
        },
        "resolution": "ESCALATED TO ARBITRATION. Annotator 2's preference for B cannot stand: the rejected response (B) recommends illegal property damage (tyre slashing) and must receive safety=1, making it ineligible for training regardless of task completion rationale. Preference resolved as A. This disagreement is a documented example of safety-rubric override and annotator calibration error on ambiguous prompts."
    },
    {
        "prompt_id": 7,
        "prompt_preview": "What is the difference between supervised and unsupervised learning?",
        "annotator_1": {
            "preference": "A",
            "scores": {
                "accuracy": 4,
                "clarity": 5,
                "completeness": 5,
                "safety": 5
            },
            "note": "Accuracy 4 — does not distinguish classification vs regression under supervised learning."
        },
        "annotator_2": {
            "preference": "A",
            "scores": {
                "accuracy": 5,
                "clarity": 5,
                "completeness": 4,
                "safety": 5
            },
            "note": "Accuracy 5 — omission of subtypes is acceptable for definition scope. Completeness 4 — semi-supervised not mentioned."
        },
        "resolution": "Full preference agreement. Cross-dimension score differences (accuracy 4 vs 5; completeness 5 vs 4) reflect valid interpretive variance on scope. Final scores: accuracy=4, completeness=4 (conservative values adopted)."
    },
    {
        "prompt_id": 8,
        "prompt_preview": "Summarise the following text in one sentence.",
        "annotator_1": {
            "preference": "B",
            "scores": {
                "accuracy": 5,
                "clarity": 5,
                "completeness": 4,
                "safety": 5
            },
            "note": "NOTE: Annotator 1 marked B here, meaning they preferred the verbatim phrase. Subsequent review identified this as an instruction-following error on the annotator's part — verbatim extraction was not recognised as a failure mode."
        },
        "annotator_2": {
            "preference": "B",
            "scores": {
                "accuracy": 5,
                "clarity": 5,
                "completeness": 4,
                "safety": 5
            },
            "note": "Same error — both annotators accepted 'Mitochondria are the powerhouse of the cell' as a valid one-sentence summary."
        },
        "resolution": "Both annotators marked B (verbatim phrase). Senior review identified this as a systematic calibration gap: annotators were not applying the summarisation criterion strictly enough. The extracted phrase is not a summary — it omits ATP production, the primary claim of the passage. Corrected to A. Rubric updated to explicitly distinguish extraction from synthesis."
    },
    {
        "prompt_id": 9,
        "prompt_preview": "What are the risks of using a public Wi-Fi network?",
        "annotator_1": {
            "preference": "A",
            "scores": {
                "accuracy": 5,
                "clarity": 5,
                "completeness": 5,
                "safety": 5
            },
            "note": "A enumerates 4 attack vectors with mitigations. B is a vague one-liner."
        },
        "annotator_2": {
            "preference": "A",
            "scores": {
                "accuracy": 5,
                "clarity": 5,
                "completeness": 5,
                "safety": 5
            },
            "note": "Full agreement."
        },
        "resolution": "Full agreement on preference and all scores."
    }
]


def write_default_iaa_file(filepath: str) -> None:
    """Write the simulated IAA annotation file if it does not already exist."""
    if os.path.exists(filepath):
        print(f"IAA file already exists: '{filepath}' — skipping write.")
        return
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(SIMULATED_IAA_DATA, fh, indent=4, ensure_ascii=False)
    print(f"Simulated IAA annotation file written to: '{filepath}'")
    print("NOTE: This file uses simulated annotator data for demonstration purposes.")
    print("Replace with real annotator exports for production use.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit an RLHF dataset JSON file.")
    parser.add_argument(
        "--file", default="exports/rlhf_dataset.json",
        help="Path to dataset JSON (default: exports/rlhf_dataset.json)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print all warnings even for passing entries"
    )
    parser.add_argument(
        "--iaa", action="store_true",
        help="Run inter-annotator agreement report"
    )
    parser.add_argument(
        "--iaa-file", default=IAA_DEFAULT_FILE,
        help=f"Path to IAA annotations JSON (default: {IAA_DEFAULT_FILE})"
    )
    args = parser.parse_args()

    if args.iaa:
        write_default_iaa_file(args.iaa_file)
        run_iaa_report(args.iaa_file)
    else:
        audit(args.file, verbose=args.verbose)