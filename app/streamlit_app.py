"""VeriQA demo UI. All reported numbers come from results/RESULT_FREEZE.json."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import pandas as pd, streamlit as st
from veriqa.service import VeriQAService

st.set_page_config(page_title="VeriQA", layout="wide")

@st.cache_resource
def get_service():
    return VeriQAService()

@st.cache_data
def frozen():
    p = ROOT / "results/RESULT_FREEZE.json"
    return json.loads(p.read_text()) if p.exists() else {}

st.title("VeriQA - answers you can check, and silence you can trust")
st.caption("Selective extractive question answering with a calibrated abstention gate. "
           "CPU only: no GPU, no API key, no network.")

tab_ask, tab_res, tab_about = st.tabs(["Ask", "Frozen results", "About"])

with tab_ask:
    c1, c2 = st.columns([3, 1])
    with c2:
        svc = get_service()
        thr = st.slider("Risk tolerance (abstain above)", 0.0, 1.0,
                        float(svc.default_threshold), 0.01,
                        help="Lower = more cautious, answers fewer questions.")
        k = st.slider("Passages retrieved (k)", 1, 10, 5)
        st.metric("Corpus passages", f"{len(svc.corpus.passages):,}")
    with c1:
        q = st.text_input("Question",
                          "What is the fourth and final stomach compartment in ruminants?")
        go = st.button("Ask", type="primary")
    if go and q.strip():
        r = get_service().ask(q, thr, k)
        if r["abstained"]:
            st.error("**I cannot answer this reliably from the provided documents.**")
            st.write(f"Estimated probability the answer would be wrong: **{r['risk']:.2f}** "
                     f"(threshold {r['threshold']:.2f})")
            if r["reason"]:
                st.write("Why:")
                for why in r["reason"].split("; "):
                    st.write(f"- {why}")
            with st.expander("Show the candidate answer that was suppressed"):
                st.write(f"`{r['candidate_answer']}`")
        else:
            st.success(f"### {r['answer']}")
            st.write(f"Estimated probability of error: **{r['risk']:.2f}** "
                     f"(threshold {r['threshold']:.2f})")
        st.markdown("**Supporting passage** - " + (r["article"] or ""))
        txt, ans = r["passage"], r["candidate_answer"]
        st.markdown(txt.replace(ans, f"**:orange[{ans}]**") if ans and ans in txt else txt)
        cc = st.columns(3)
        cc[0].metric("retrieval", f"{r['timing_ms']['retrieval']:.0f} ms")
        cc[1].metric("reader", f"{r['timing_ms']['reader']:.0f} ms")
        cc[2].metric("reliability layer", f"{r['timing_ms']['reliability']:.1f} ms")
        with st.expander("Reliability features (the 16 signals behind the decision)"):
            f = r["features"]
            st.dataframe(pd.DataFrame({"feature": list(f), "value": list(f.values())}),
                         use_container_width=True, hide_index=True)
        with st.expander("All retrieved evidence"):
            for e in r["evidence"]:
                st.write(f"**{e['article']}** (score {e['score']:.3f})")
                st.caption(e["text"])

with tab_res:
    fz = frozen()
    if not fz:
        st.warning("No frozen results found. Run the experiment scripts first.")
    else:
        h = fz["headline"]
        c = st.columns(4)
        c[0].metric("P1 error-prediction AUC", f"{h['P1_error_auc']:.3f}")
        c[1].metric("Reader-only (B5) AUC", f"{h['B5_error_auc']:.3f}")
        c[2].metric("dAURC P1-B5", f"{h['delta_aurc_P1_B5']:+.4f}")
        c[3].metric("Reliability layer", f"{h['reliability_layer_ms']:.1f} ms")
        st.caption(f"Test set: {h['test_questions']:,} questions across "
                   f"{h['test_articles']} articles. Frozen {fz['frozen_at']}.")
        st.subheader("Pre-registered gates")
        st.dataframe(pd.DataFrame([{"gate": k, "result": "PASS" if v["pass"] else "FAIL"}
                                   for k, v in fz["gates"].items()]),
                     use_container_width=True, hide_index=True)
        for name, f in (("Risk-coverage", "F3_risk_coverage.png"),
                        ("Ablation", "F5_ablation.png"),
                        ("Calibration", "F4_reliability_diagram.png")):
            p = ROOT / "results/figures" / f
            if p.exists():
                st.subheader(name); st.image(str(p), width=520)
        for t in ["T2_main_comparison", "T4_ablation", "T6_domain_shift", "T7_latency"]:
            p = ROOT / f"results/tables/{t}.csv"
            if p.exists():
                st.subheader(t); st.dataframe(pd.read_csv(p), use_container_width=True)

with tab_about:
    st.markdown("""
**What this is.** A document question-answering system that estimates the probability its own
answer is wrong, and declines to answer when that risk is too high.

**What is ours.** The retrieval-side and cross-evidence reliability features, the calibrated
abstention gate, the evaluation harness, and the measured comparison.

**What is not ours.** The trained-calibrator idea comes from Kamath, Jia & Liang (ACL 2020);
our B5 baseline is their approach reimplemented in this setting. TF-IDF, SVD, BM25, gradient
boosting and isotonic regression are standard methods used as published.

**Honest limits.** The reader is a classical feature-based span scorer, not a transformer,
because no pretrained weights were available in the build environment. Absolute accuracy is
therefore low and every number here is conditional on that reader.
""")
