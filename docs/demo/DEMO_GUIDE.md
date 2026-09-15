# Demo Guide — 8 minutes

> Every number spoken aloud must come from `results/RESULT_FREEZE.json`. The app's
> "Frozen results" tab reads that file directly, so use it rather than reciting from memory.

## Setup, the night before
```bash
python3 -m pytest tests -q          # 19 passed
streamlit run app/streamlit_app.py
```
Run the whole demo once with **wifi off**. Nothing in VeriQA touches the network at query time,
and proving that on stage is part of the point.

## Script

**0:00–0:45 · The problem.** Ask something the corpus cannot answer, with the risk slider pushed
to 1.0 so the gate is disabled. The system returns a confident, specific, wrong answer.
> "That answer is invented, and nothing on screen tells you so. We did not set out to make the
> reader more accurate. We set out to make it honest about when it does not know."

**0:45–1:30 · Architecture.** Show `docs/ARCHITECTURE.md`. Point at the starred box.
> "Standard retrieval and reading. Our contribution is this layer between the reader and you."

**1:30–2:30 · Ingestion and retrieval.** Ask an answerable question. Open "All retrieved
evidence": five passages with fused scores.

**2:30–3:30 · Reliability features.** Open the feature table. Sixteen signals, three families.
> "Retrieval: is the evidence there at all? Reader: how sure is the extractor? Agreement: do
> independent passages say the same thing?"

**3:30–4:30 · The wow moment.** Return the slider to the default threshold and re-ask the
unanswerable question. The system now abstains and lists *why*.
> "Same question, same documents, gate on. It declines, and it tells you which signal made it
> decline."

**4:30–5:30 · Risk/coverage.** Drag the slider; open the "Frozen results" tab and show
`F3_risk_coverage.png`.
> "This is the deployer's dial, measured on 3,585 held-out questions."

**5:30–6:30 · Baselines.** Show table T2 in the app.
> "Reader-only calibration — the published approach we reimplemented — gets 0.524. Adding
> retrieval and agreement features gets 0.639. The ablation says retrieval carries it, and that
> removing all five reader features changes nothing significant."

**6:30–7:30 · Hand over the laptop.** Let the examiner ask anything.

**7:30–8:00 · Close, including the weakness.**
> "All of that ran on this CPU with no GPU and no network; the reliability layer costs 1.96 ms.
> One thing we want to be clear about: our reader is a classical span scorer, not a transformer,
> because we could not obtain pretrained weights. Base accuracy is 11%. Three of our seven
> pre-registered gates failed and we report them. The comparative finding is solid; the absolute
> system is not, and we know exactly why."

**Say that last part yourself before anyone asks.** Volunteering the limitation is what
separates a team that understands its work from one defending it.

## Risks
| Risk | Control |
|---|---|
| Long first query | The service caches on load; ask one warm-up question before starting |
| Examiner asks for a feature you didn't build | "That's future work — here's why we scoped it out" |
| Someone challenges the low accuracy | Go straight to `READER_BACKEND.md`; it is documented, not hidden |
| App crashes | Have `results/figures/*.png` and the frozen JSON open in a second window |
