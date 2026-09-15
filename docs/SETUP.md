# Setup

## Requirements
Python 3.10+, ~2 GB free disk, no GPU. Tested on Python 3.10.12 / Linux.

## Install
```bash
cd VeriQA
python3 -m venv .venv && source .venv/bin/activate    # optional
pip install -r requirements.txt
```
`streamlit`, `fastapi` and `uvicorn` are only needed for the application; every experiment
runs without them.

## Data
`data/raw/train-v2.0.json` and `dev-v2.0.json` ship with the repository.
Verify before use:
```bash
cd data/raw && sha256sum -c CHECKSUMS.txt
```
If missing, obtain them from the SQuAD project:
```bash
git clone --depth 1 https://github.com/rajpurkar/SQuAD-explorer.git
cp SQuAD-explorer/dataset/{train,dev}-v2.0.json data/raw/
```

## Verify the install
```bash
python3 -m pytest tests -q          # expect 19 passed
python3 -c "import sys; sys.path.insert(0,'src'); from veriqa.config import load_config; print(load_config().seed)"
```

## Notes
- No model download is required. The pipeline uses TF-IDF + truncated SVD and a feature-based
  reader trained from the dataset itself.
- `experiments/02_run_pipeline.py` accepts an optional time budget in seconds and is resumable:
  `python3 experiments/02_run_pipeline.py A 120` runs for two minutes and can be re-invoked.
