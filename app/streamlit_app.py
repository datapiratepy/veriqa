
"""VeriQA application: SQuAD Research and Custom PDF QA."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from veriqa.service import VeriQAService
from veriqa.pdf_service import PDFQAService


st.set_page_config(
    page_title="VeriQA",
    layout="wide",
)


# ---------------------------------------------------------------------
# SQuAD research service
# ---------------------------------------------------------------------

@st.cache_resource
def get_service():
    return VeriQAService()


# ---------------------------------------------------------------------
# Frozen experiment results
# ---------------------------------------------------------------------

@st.cache_data
def frozen():
    path = ROOT / "results/RESULT_FREEZE.json"

    return (
        json.loads(path.read_text())
        if path.exists()
        else {}
    )


# ---------------------------------------------------------------------
# Application heading
# ---------------------------------------------------------------------

st.title(
    "VeriQA - answers you can check, and silence you can trust"
)

st.caption(
    "SQuAD Research: selective extractive QA with a calibrated "
    "abstention gate. Custom PDF QA: separate local BM25 + TF-IDF "
    "retrieval and extractive sentence selection. PDF match scores "
    "are not calibrated probabilities of correctness."
)


# ---------------------------------------------------------------------
# Application tabs
# ---------------------------------------------------------------------

tab_ask, tab_pdf, tab_res, tab_about = st.tabs([
    "SQuAD Research",
    "Custom PDF QA",
    "Frozen results",
    "About",
])


# =====================================================================
# TAB 1: ORIGINAL SQUAD RESEARCH MODE
# =====================================================================

with tab_ask:

    c1, c2 = st.columns([3, 1])

    with c2:
        svc = get_service()

        thr = st.slider(
            "Risk tolerance (abstain above)",
            0.0,
            1.0,
            float(svc.default_threshold),
            0.01,
            help="Lower = more cautious, answers fewer questions.",
        )

        k = st.slider(
            "Passages retrieved (k)",
            1,
            10,
            5,
        )

        st.metric(
            "Corpus passages",
            f"{len(svc.corpus.passages):,}",
        )

    with c1:
        q = st.text_input(
            "Question",
            "What is the fourth and final stomach compartment "
            "in ruminants?",
        )

        go = st.button(
            "Ask",
            type="primary",
        )

    if go and q.strip():

        r = get_service().ask(q, thr, k)

        if r["abstained"]:

            st.error(
                "**I cannot answer this reliably from the "
                "provided documents.**"
            )

            st.write(
                f"Estimated probability the answer would be wrong: "
                f"**{r['risk']:.2f}** "
                f"(threshold {r['threshold']:.2f})"
            )

            if r["reason"]:
                st.write("Why:")

                for why in r["reason"].split("; "):
                    st.write(f"- {why}")

            with st.expander(
                "Show the candidate answer that was suppressed"
            ):
                st.write(
                    f"`{r['candidate_answer']}`"
                )

        else:

            st.success(
                f"### {r['answer']}"
            )

            st.write(
                f"Estimated probability of error: "
                f"**{r['risk']:.2f}** "
                f"(threshold {r['threshold']:.2f})"
            )

        st.markdown(
            "**Supporting passage** - "
            + (r["article"] or "")
        )

        txt = r["passage"]
        ans = r["candidate_answer"]

        st.markdown(
            txt.replace(
                ans,
                f"**:orange[{ans}]**",
            )
            if ans and ans in txt
            else txt
        )

        cc = st.columns(3)

        cc[0].metric(
            "retrieval",
            f"{r['timing_ms']['retrieval']:.0f} ms",
        )

        cc[1].metric(
            "reader",
            f"{r['timing_ms']['reader']:.0f} ms",
        )

        cc[2].metric(
            "reliability layer",
            f"{r['timing_ms']['reliability']:.1f} ms",
        )

        with st.expander(
            "Reliability features (the 16 signals behind the decision)"
        ):

            f = r["features"]

            st.dataframe(
                pd.DataFrame({
                    "feature": list(f),
                    "value": list(f.values()),
                }),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("All retrieved evidence"):

            for e in r["evidence"]:

                st.write(
                    f"**{e['article']}** "
                    f"(score {e['score']:.3f})"
                )

                st.caption(e["text"])


# =====================================================================
# TAB 2: CUSTOM PDF QUESTION ANSWERING
# =====================================================================

with tab_pdf:

    st.subheader(
        "Ask questions about your own PDFs"
    )

    st.write(
        "Upload text-based PDF documents, index their contents, "
        "and ask questions with page-level source references."
    )

    # Each Streamlit session has its own PDF service.
    if "pdf_qa_service" not in st.session_state:
        st.session_state["pdf_qa_service"] = PDFQAService()

    pdf_svc = st.session_state["pdf_qa_service"]

    # PDF upload control.
    uploaded_pdfs = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="custom_pdf_upload",
        help="Upload one or more text-based PDF documents.",
    )

    # Detect changes in uploaded files.
    upload_signature = tuple(
        (file.name, file.size)
        for file in (uploaded_pdfs or [])
    )

    if (
        st.session_state.get("pdf_upload_signature")
        != upload_signature
    ):

        st.session_state["pdf_qa_indexed"] = False

        st.session_state["pdf_upload_signature"] = (
            upload_signature
        )

    # Index uploaded documents.
    if st.button(
        "Index uploaded PDFs",
        type="primary",
        disabled=not uploaded_pdfs,
        key="index_custom_pdfs",
    ):

        try:

            pdf_files = [
                (file.name, file.getvalue())
                for file in uploaded_pdfs
            ]

            stats = pdf_svc.index_documents(pdf_files)

            st.session_state["pdf_qa_indexed"] = True

            st.success(
                f"Successfully indexed "
                f"{len(stats['documents'])} document(s), "
                f"{stats['pages']} page(s), and "
                f"{stats['chunks']} text chunk(s)."
            )

        except Exception as exc:

            st.session_state["pdf_qa_indexed"] = False

            st.error(
                f"PDF indexing failed: {exc}"
            )

    # Display the question interface after successful indexing.
    if (
        st.session_state.get("pdf_qa_indexed", False)
        and pdf_svc.is_ready
    ):

        st.divider()

        st.subheader(
            "Ask your documents"
        )

        question = st.text_input(
            "Enter your question",
            placeholder="Example: What is supervised learning?",
            key="custom_pdf_question",
        )

        k_pdf = st.slider(
            "Supporting passages",
            min_value=1,
            max_value=10,
            value=5,
            key="custom_pdf_k",
        )

        if st.button(
            "Ask PDFs",
            type="primary",
            key="ask_custom_pdfs",
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                try:

                    result = pdf_svc.ask(
                        question,
                        k=k_pdf,
                    )

                    if result["abstained"]:

                        st.warning(
                            "I could not identify a matching answer "
                            "in the uploaded documents."
                        )

                        st.write(
                            result["reason"]
                        )

                    else:

                        st.success(
                            result["answer"]
                        )

                        source = result["source"]

                        # Display physical and printed page numbers.
                        citation = (
                            f"Source: {source['document']} "
                            f"— PDF page {source['page']}"
                        )

                        printed_page = source.get(
                            "printed_page"
                        )

                        if printed_page is not None:

                            citation += (
                                f" (printed page {printed_page})"
                            )

                        st.caption(citation)

                        st.metric(
                            "Relative match score",
                            f"{result['match_score']:.3f}",
                        )

                    # Show all retrieved supporting passages.
                    with st.expander(
                        "Retrieved supporting passages"
                    ):

                        for index, evidence in enumerate(
                            result["evidence"],
                            start=1,
                        ):

                            evidence_location = (
                                f"PDF page {evidence['page']}"
                            )

                            evidence_printed_page = (
                                evidence.get("printed_page")
                            )

                            if evidence_printed_page is not None:

                                evidence_location += (
                                    f" (printed page "
                                    f"{evidence_printed_page})"
                                )

                            st.markdown(
                                f"**{index}. "
                                f"{evidence['document']} "
                                f"— {evidence_location}**"
                            )

                            st.caption(
                                "Retrieval score: "
                                f"{evidence['score']:.3f}"
                            )

                            st.write(
                                evidence["text"]
                            )

                    # Explain the limitation of the PDF match score.
                    st.caption(
                        result["warning"]
                    )

                except Exception as exc:

                    st.error(
                        f"Question answering failed: {exc}"
                    )

    else:

        st.info(
            "Upload one or more PDFs and click "
            "'Index uploaded PDFs' to begin."
        )


# =====================================================================
# TAB 3: FROZEN RESEARCH RESULTS
# =====================================================================

with tab_res:

    fz = frozen()

    if not fz:

        st.warning(
            "No frozen results found. Run the experiment scripts first."
        )

    else:

        h = fz["headline"]

        c = st.columns(4)

        c[0].metric(
            "P1 error-prediction AUC",
            f"{h['P1_error_auc']:.3f}",
        )

        c[1].metric(
            "Reader-only (B5) AUC",
            f"{h['B5_error_auc']:.3f}",
        )

        c[2].metric(
            "dAURC P1-B5",
            f"{h['delta_aurc_P1_B5']:+.4f}",
        )

        c[3].metric(
            "Reliability layer",
            f"{h['reliability_layer_ms']:.1f} ms",
        )

        st.caption(
            f"Test set: {h['test_questions']:,} questions "
            f"across {h['test_articles']} articles. "
            f"Frozen {fz['frozen_at']}."
        )

        st.subheader(
            "Pre-registered gates"
        )

        st.dataframe(
            pd.DataFrame([
                {
                    "gate": key,
                    "result": "PASS" if value["pass"] else "FAIL",
                }
                for key, value in fz["gates"].items()
            ]),
            use_container_width=True,
            hide_index=True,
        )

        for name, filename in (
            ("Risk-coverage", "F3_risk_coverage.png"),
            ("Ablation", "F5_ablation.png"),
            ("Calibration", "F4_reliability_diagram.png"),
        ):

            path = ROOT / "results/figures" / filename

            if path.exists():

                st.subheader(name)

                st.image(
                    str(path),
                    width=520,
                )

        for table_name in (
            "T2_main_comparison",
            "T4_ablation",
            "T6_domain_shift",
            "T7_latency",
        ):

            path = ROOT / f"results/tables/{table_name}.csv"

            if path.exists():

                st.subheader(table_name)

                st.dataframe(
                    pd.read_csv(path),
                    use_container_width=True,
                )


# =====================================================================
# TAB 4: ABOUT
# =====================================================================

with tab_about:

    st.markdown("""
**What this is.** A document question-answering system that estimates
the probability its own answer is wrong, and declines to answer when
that risk is too high in the SQuAD Research mode.

**Custom PDF QA.** This is a separate local extension using PDF text
extraction, BM25 and TF-IDF retrieval, and extractive sentence
selection. It displays source page references. Its relative match
scores are not calibrated probabilities of correctness.

**What is ours.** The retrieval-side and cross-evidence reliability
features, the calibrated abstention gate, the evaluation harness,
and the measured comparison in the SQuAD research implementation.

**What is not ours.** The trained-calibrator idea comes from
Kamath, Jia & Liang (ACL 2020); our B5 baseline is their approach
reimplemented in this setting. TF-IDF, SVD, BM25, gradient boosting
and isotonic regression are standard methods used as published.

**Honest limits.** The SQuAD reader is a classical feature-based
span scorer, not a transformer, because no pretrained weights were
available in the build environment. Absolute accuracy is therefore
low and every research number is conditional on that reader.

The Custom PDF QA extension is a separate prototype and has not
been validated using the SQuAD research calibration experiments.
""")
