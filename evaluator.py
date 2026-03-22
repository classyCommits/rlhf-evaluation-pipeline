"""
evaluator.py — RLHF data backend
Handles loading, saving, and exporting evaluation results.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_FILE = r"data/data.json"
RESULTS_FILE = r"results/results.csv"
EXPORT_FILE = r"exports/rlhf_dataset.json"

REQUIRED_COLUMNS = [
    "timestamp", "prompt", "response_a", "response_b",
    "chosen", "rejected",
    "accuracy_score", "clarity_score", "completeness_score", "safety_score",
    "reason", "hallucination_flag", "policy_violation_flag",
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> list[dict]:
    """
    Load the prompt-response dataset from DATA_FILE.

    Returns an empty list (instead of crashing) when the file is missing or
    malformed — callers must handle the empty-list case.
    """
    if not os.path.exists(DATA_FILE):
        logger.error("Data file '%s' not found.", DATA_FILE)
        return []

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse '%s': %s", DATA_FILE, exc)
        return []

    if not isinstance(data, list) or not data:
        logger.warning("'%s' is empty or not a list.", DATA_FILE)
        return []

    # Validate every entry has the expected keys
    required_keys = {"prompt", "response_a", "response_b"}
    valid = []
    for i, item in enumerate(data):
        missing = required_keys - item.keys()
        if missing:
            logger.warning("Item %d skipped — missing keys: %s", i, missing)
        else:
            valid.append(item)

    return valid


# ── Result construction ────────────────────────────────────────────────────────

def create_result(
    prompt: str,
    response_a: str,
    response_b: str,
    chosen: str,          # "A" or "B"
    scores: dict[str, int],
    reason: str,
    flags: dict[str, bool],
) -> dict[str, Any]:
    """
    Build a structured RLHF evaluation record.

    BUG FIXED: original code set ``rejected`` by checking ``chosen == "A"``,
    but never validated that ``chosen`` was a legal value.  An unexpected value
    (e.g. lowercase "a") would silently produce a wrong ``rejected`` label.
    """
    chosen = chosen.upper().strip()
    if chosen not in {"A", "B"}:
        raise ValueError(f"'chosen' must be 'A' or 'B', got {chosen!r}")

    rejected = "B" if chosen == "A" else "A"

    return {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response_a": response_a,
        "response_b": response_b,
        "chosen": chosen,
        "rejected": rejected,
        "accuracy_score": scores.get("accuracy", 3),
        "clarity_score": scores.get("clarity", 3),
        "completeness_score": scores.get("completeness", 3),
        "safety_score": scores.get("safety", 3),
        "reason": reason.strip(),
        "hallucination_flag": bool(flags.get("hallucination", False)),
        "policy_violation_flag": bool(flags.get("policy_violation", False)),
    }


# ── Persistence ────────────────────────────────────────────────────────────────

def save_result(result: dict[str, Any]) -> None:
    """
    Append one evaluation record to RESULTS_FILE.

    BUG FIXED: the original code opened the CSV in append mode without
    checking whether the existing file is empty.  If RESULTS_FILE existed but
    was empty (zero bytes), the header would be omitted and the first data row
    would be written without column names, corrupting every subsequent read.
    """
    df = pd.DataFrame([result])

    file_exists = os.path.exists(RESULTS_FILE)
    # Treat a zero-byte file the same as no file (write header)
    write_header = not file_exists or os.path.getsize(RESULTS_FILE) == 0

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    df.to_csv(
        RESULTS_FILE,
        mode="a",
        header=write_header,
        index=False,
    )
    logger.info("Result saved → %s", RESULTS_FILE)


# ── Diagnostics ───────────────────────────────────────────────────────────────

def diagnose_export() -> dict:
    """
    Return a plain-dict report of exactly why export would succeed or fail.
    Called by the Streamlit debug panel — never raises, always returns.
    """
    report = {
        "results_file_exists": False,
        "results_file_bytes": 0,
        "results_file_rows": 0,
        "results_file_columns": [],
        "missing_columns": [],
        "flagged_rows": 0,
        "exportable_rows": 0,
        "verdict": "",
    }

    if not os.path.exists(RESULTS_FILE):
        report["verdict"] = (
            f"'{RESULTS_FILE}' does not exist. "
            "No evaluation has been saved yet — submit at least one record first."
        )
        return report

    report["results_file_exists"] = True
    report["results_file_bytes"] = os.path.getsize(RESULTS_FILE)

    if report["results_file_bytes"] == 0:
        report["verdict"] = f"'{RESULTS_FILE}' exists but is empty (0 bytes)."
        return report

    try:
        df = pd.read_csv(RESULTS_FILE)
    except Exception as exc:
        report["verdict"] = f"Failed to parse '{RESULTS_FILE}': {exc}"
        return report

    report["results_file_rows"] = len(df)
    report["results_file_columns"] = list(df.columns)
    report["missing_columns"] = list(set(REQUIRED_COLUMNS) - set(df.columns))

    if report["missing_columns"]:
        report["verdict"] = (
            f"CSV is missing required columns: {report['missing_columns']}. "
            "This usually means results.csv was written by an older version of the app."
        )
        return report

    flagged = (
        df["hallucination_flag"].astype(bool) | df["policy_violation_flag"].astype(bool)
    )
    report["flagged_rows"] = int(flagged.sum())
    report["exportable_rows"] = int((~flagged).sum())

    if report["exportable_rows"] == 0:
        report["verdict"] = (
            f"All {report['results_file_rows']} row(s) are flagged "
            "(hallucination or policy violation). Nothing to export."
        )
    else:
        report["verdict"] = (
            f"Ready — {report['exportable_rows']} exportable row(s) "
            f"({report['flagged_rows']} flagged and excluded)."
        )

    return report


# ── Export ─────────────────────────────────────────────────────────────────────

def export_rlhf_dataset() -> str | None:
    """
    Convert RESULTS_FILE into a clean RLHF training dataset (JSON).

    BUG FIXED: original function returned ``None`` implicitly when the CSV was
    missing, but the Streamlit caller compared the return value to a truthy
    string — that part worked, but ``None`` vs an empty string vs a missing
    file were all silently conflated.  We now return ``None`` explicitly and
    document it.

    BUG FIXED: flagged rows (hallucination / policy_violation) were included
    in the exported dataset unchanged.  Flagged examples are unreliable
    training signal; they are now excluded with a warning.
    """
    if not os.path.exists(RESULTS_FILE) or os.path.getsize(RESULTS_FILE) == 0:
        logger.warning("No results to export — '%s' is missing or empty.", RESULTS_FILE)
        return None

    try:
        df = pd.read_csv(RESULTS_FILE)
    except Exception as exc:
        logger.error("Failed to read '%s': %s", RESULTS_FILE, exc)
        return None

    # Validate columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        logger.error("Results file is missing columns: %s", missing_cols)
        return None

    # Drop flagged rows — they should not enter the training set
    flagged_mask = df["hallucination_flag"].astype(bool) | df["policy_violation_flag"].astype(bool)
    n_flagged = flagged_mask.sum()
    if n_flagged:
        logger.warning("Excluding %d flagged row(s) from export.", n_flagged)
    df = df[~flagged_mask]

    if df.empty:
        logger.warning("All rows were flagged — nothing to export.")
        return None

    dataset = []
    for _, row in df.iterrows():
        chosen_resp  = row["response_a"] if row["chosen"] == "A" else row["response_b"]
        rejected_resp = row["response_b"] if row["chosen"] == "A" else row["response_a"]

        dataset.append({
            "prompt": row["prompt"],
            "chosen": chosen_resp,
            "rejected": rejected_resp,
            "scores": {
                "accuracy":     int(row["accuracy_score"]),
                "clarity":      int(row["clarity_score"]),
                "completeness": int(row["completeness_score"]),
                "safety":       int(row["safety_score"]),
            },
            "reason": row["reason"],
        })

    try:
        os.makedirs(os.path.dirname(EXPORT_FILE), exist_ok=True)
        with open(EXPORT_FILE, "w", encoding="utf-8") as fh:
            json.dump(dataset, fh, indent=4, ensure_ascii=False)
    except OSError as exc:
        logger.error("Could not write '%s': %s", EXPORT_FILE, exc)
        return None

    logger.info("Exported %d records → %s", len(dataset), EXPORT_FILE)
    return EXPORT_FILE