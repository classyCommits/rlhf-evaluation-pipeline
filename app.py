"""
app.py — RLHF Response Evaluator (Streamlit UI)

Run with:  streamlit run app.py
"""

import streamlit as st

from evaluator import load_data, save_result, create_result, export_rlhf_dataset, diagnose_export

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RLHF Evaluator", layout="wide")

st.title("RLHF Response Evaluator")
st.caption("Simulate human feedback for AI model training")

# ── Load dataset ───────────────────────────────────────────────────────────────
# BUG FIXED: original code called load_data() unconditionally on every rerun.
# If the file was missing, it would raise an unhandled exception and crash the
# entire app.  We now show a friendly error and stop early.
data = load_data()

if not data:
    st.error(
        "No evaluation data found.  "
        "Make sure `data.json` exists in the working directory and is valid."
    )
    st.stop()

# ── Session state ──────────────────────────────────────────────────────────────
if "index" not in st.session_state:
    st.session_state.index = 0
if "completed" not in st.session_state:
    st.session_state.completed = False

# BUG FIXED: if data shrinks between reruns (e.g. file swapped), the stored
# index can be out of range → IndexError.  Clamp it defensively.
st.session_state.index = min(st.session_state.index, len(data) - 1)

item = data[st.session_state.index]

# ── Completion screen ──────────────────────────────────────────────────────────
if st.session_state.completed:
    st.success(f"All {len(data)} records evaluated. Results saved to results.csv.")
    st.markdown("---")
    rc1, rc2 = st.columns(2)
    with rc1:
        if st.button("Re-evaluate from scratch", use_container_width=True):
            st.session_state.index = 0
            st.session_state.completed = False
            st.rerun()
    with rc2:
        report = diagnose_export()
        if st.button(
            "Export RLHF Dataset",
            disabled=report["exportable_rows"] == 0,
            use_container_width=True,
        ):
            with st.spinner("Exporting…"):
                file_path = export_rlhf_dataset()
            if file_path:
                st.success(f"Exported → `{file_path}`")
                try:
                    with open(file_path, "rb") as fh:
                        st.download_button(
                            label="Download rlhf_dataset.json",
                            data=fh,
                            file_name="rlhf_dataset.json",
                            mime="application/json",
                        )
                except OSError:
                    pass
            else:
                st.warning(report["verdict"])
    st.stop()

# ── Progress indicator ─────────────────────────────────────────────────────────
st.progress(
    (st.session_state.index + 1) / len(data),
    text=f"Item {st.session_state.index + 1} of {len(data)}",
)

# ── Prompt display ─────────────────────────────────────────────────────────────
st.subheader("Prompt")
st.info(item["prompt"])

# ── Side-by-side responses ─────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### Response A")
    st.write(item["response_a"])

with col_b:
    st.markdown("### Response B")
    st.write(item["response_b"])

st.divider()

# ── Evaluation form ────────────────────────────────────────────────────────────
# BUG FIXED: wrapping everything in st.form() prevents the sliders from
# triggering a full rerun on every tiny change, which caused the radio button
# and text_area to reset intermittently in the original code.
with st.form("evaluation_form", clear_on_submit=True):
    st.subheader("Evaluation")

    chosen = st.radio(
        "Which response is better?",
        ["A", "B"],
        horizontal=True,
    )

    st.markdown("### Multi-Criteria Scoring")
    scol1, scol2 = st.columns(2)

    with scol1:
        accuracy     = st.slider("Accuracy",     1, 5, 3)
        clarity      = st.slider("Clarity",      1, 5, 3)

    with scol2:
        completeness = st.slider("Completeness", 1, 5, 3)
        safety       = st.slider("Safety",       1, 5, 3)

    reason = st.text_area("Reason for your choice", placeholder="Explain why this response is better…")

    st.markdown("### Flags")
    flag_col1, flag_col2 = st.columns(2)
    with flag_col1:
        hallucination     = st.checkbox("Contains Hallucination")
    with flag_col2:
        policy_violation  = st.checkbox("Policy Violation")

    submitted = st.form_submit_button("Submit Evaluation", use_container_width=True)

# ── Handle submission ──────────────────────────────────────────────────────────
if submitted:
    # BUG FIXED: original code used st.warning() but still continued to call
    # create_result / save_result.  Validation must gate the save.
    if not reason.strip():
        st.warning("Please provide a reason before submitting.")
    else:
        try:
            result = create_result(
                prompt=item["prompt"],
                response_a=item["response_a"],
                response_b=item["response_b"],
                chosen=chosen,
                scores={
                    "accuracy":     accuracy,
                    "clarity":      clarity,
                    "completeness": completeness,
                    "safety":       safety,
                },
                reason=reason,
                flags={
                    "hallucination":    hallucination,
                    "policy_violation": policy_violation,
                },
            )
            save_result(result)
            st.success("Evaluation saved!")

            if st.session_state.index >= len(data) - 1:
                st.session_state.completed = True
            else:
                st.session_state.index += 1
            st.rerun()

        except Exception as exc:
            st.error(f"Failed to save evaluation: {exc}")

st.divider()

# ── Navigation ─────────────────────────────────────────────────────────────────
# BUG FIXED: original code had no way to go back or jump to a specific item.
nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

with nav_col1:
    if st.button("Previous", disabled=st.session_state.index == 0):
        st.session_state.index -= 1
        st.rerun()

with nav_col3:
    if st.button("Next", disabled=st.session_state.index >= len(data) - 1):
        st.session_state.index += 1
        st.rerun()

# ── Export ─────────────────────────────────────────────────────────────────────
st.divider()

# Always show the current state of results.csv so there are no surprises.
report = diagnose_export()

_status_color = "green" if report["exportable_rows"] > 0 else "orange"
st.markdown(
    f"**results.csv status:** :{_status_color}[{report['verdict']}]"
)

if report["results_file_exists"] and report["results_file_rows"] > 0:
    st.caption(
        f"File size: {report['results_file_bytes']} bytes · "
        f"Total rows: {report['results_file_rows']} · "
        f"Columns present: {len(report['results_file_columns'])} / {len(report['missing_columns']) + len(report['results_file_columns'])} expected"
    )

if report["missing_columns"]:
    st.error(
        f"results.csv is missing these columns: `{'`, `'.join(report['missing_columns'])}`\n\n"
        "This means the CSV was created by an older version of the app. "
        "**Delete results.csv and re-submit your evaluations** to regenerate it with the correct schema."
    )

col_exp, col_dl = st.columns([1, 2])

with col_exp:
    export_disabled = report["exportable_rows"] == 0
    if st.button(
        "Export RLHF Dataset",
        disabled=export_disabled,
        use_container_width=True,
    ):
        with st.spinner("Exporting…"):
            file_path = export_rlhf_dataset()

        if file_path:
            st.success(f"Exported → `{file_path}`")
            with col_dl:
                try:
                    with open(file_path, "rb") as fh:
                        st.download_button(
                            label="Download rlhf_dataset.json",
                            data=fh,
                            file_name="rlhf_dataset.json",
                            mime="application/json",
                            use_container_width=True,
                        )
                except OSError:
                    pass
        else:
            st.warning("Export failed — check the status message above.")