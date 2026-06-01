# Homework 3: Akkadian Streaming Translator

TODO: full name, group number.

This repository implements an end-to-end low-resource NMT project for the Kaggle competition
**Deep Past Initiative: Machine Translation**, direction **Akkadian transliteration -> English**.

The project is built in stages:

1. data cleaning and dev split;
2. ByT5 baseline fine-tuning;
3. ablations for orthography normalization, beam search, and mini-ensemble;
4. Kaggle prediction export;
5. streaming translator API and web UI;
6. Docker Compose deployment.

## Current Architecture

- `model.py` contains the required `My_Translator_Model` class.
- `src/akkadian_mt/translator.py` contains the reusable model implementation.
- `src/akkadian_mt/normalization.py` contains Akkadian transliteration normalization rules.
- `api/server.py` serves `/health`, `/translate`, and the web UI.
- `web/` contains a two-pane streaming translator interface.
- `scripts/` contains split, evaluation, and latency helpers.
- `data/log_file.log` is created automatically by the singleton logger.

## Setup

Recommended with `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Or with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Data

Download the Kaggle competition files manually or with the Kaggle CLI and place them under:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

Raw competition data and external corpora must not be committed.

If Kaggle CLI is configured:

```bash
./.venv/bin/kaggle auth login
./scripts/download_kaggle.sh
```

If using the local virtual environment, the script will use `.venv/bin/kaggle`
automatically after dependencies are installed.

Expected columns are inferred from common names:

- source: `akkadian`, `source`, `src`, `transliteration`, or `text`;
- target: `english`, `target`, `translation`, or `tgt`;
- id: `id`.

Inspect downloaded files before training:

```bash
python scripts/inspect_data.py data/raw/train.csv data/raw/test.csv data/raw/sample_submission.csv
```

Or run the complete data stage in one command:

```bash
python scripts/run_data_stage.py
```

This validates the required Kaggle files, prints dataset statistics, and creates both raw
and normalized train/dev splits.

For every external corpus, check exact and near duplicates against Kaggle test sources:

```bash
python scripts/check_leakage.py \
  --reference data/raw/test.csv \
  --candidate data/external/oracc_clean.csv \
  --normalize
```

If exact or near overlaps are found, remove them from the external corpus before training.

## Prepare Dev Split

Baseline split without normalization:

```bash
python scripts/prepare_split.py \
  --dataset data/raw/train.csv \
  --output-dir data/processed/raw
```

Normalized split:

```bash
python scripts/prepare_split.py \
  --dataset data/raw/train.csv \
  --output-dir data/processed/normalized \
  --normalize
```

The current normalizer:

- uses Unicode NFC;
- converts subscript digits such as `₂` to `2`;
- removes broken-sign brackets `⸢...⸣`;
- unwraps angle brackets `<...>`;
- keeps determinatives as spaced `{d}`-style tokens;
- replaces lacunae `[ ... ]` with `<lacuna>`;
- keeps Akkadian diacritics by default;
- lowercases text.

## Train Baseline

Default model: `google/byt5-small`.

```bash
python model.py train \
  --dataset data/processed/raw/train_split.csv \
  --model-dir model/baseline_raw
```

The model is saved to:

```text
model/
```

Weights are intentionally ignored by git. Upload final weights to Hugging Face Hub, W&B, or ClearML.

## Predict One Sentence

```bash
python model.py predict \
  --text "šarrum ana ālim illik" \
  --model-dir model/baseline_raw \
  --num-beams 4
```

## Export Kaggle Submission

```bash
python model.py predict-file \
  --dataset data/raw/test.csv \
  --model-dir model/baseline_raw \
  --num-beams 4
```

Output:

```text
data/results.csv
```

Check the competition sample submission and rename the prediction column if Kaggle requires a
different exact column name.

## Evaluate Dev Metrics

```bash
python scripts/evaluate.py \
  --dataset data/processed/raw/dev_split.csv \
  --model-dir model/baseline_raw \
  --num-beams 4 \
  --predictions-out data/processed/raw/dev_predictions.csv
```

Required metrics to report:

- BLEU with sacreBLEU;
- chrF++ with word order 2;
- COMET as auxiliary metric;
- Kaggle public/private leaderboard score.

## Beam Search Sweep

```bash
python scripts/run_beam_sweep.py \
  --dataset data/processed/raw/dev_split.csv \
  --model-dir model/baseline_raw \
  --beams 1,4,8 \
  --output data/processed/raw/beam_sweep.csv
```

## Mandatory Ablations

Fill this table after running experiments.

| Technique | chrF++ on dev | Kaggle public LB | Notes |
|---|---:|---:|---|
| Baseline ByT5-small, greedy, no normalization | TODO | TODO | Kaggle train only |
| Orthography normalization | TODO | TODO | Rules documented above |
| Beam search, beam size sweep `{1, 4, 8}` | TODO | TODO | Pick best dev chrF++ |
| Mini-ensemble, 2 checkpoints | TODO | TODO | Different seeds or model sizes |

## Streaming Service

Run locally:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8080
```

Open:

```text
http://localhost:8080
```

Health check:

```bash
curl http://localhost:8080/health
```

Streaming endpoint:

```bash
curl -N -X POST http://localhost:8080/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"šarrum ana ālim illik"}'
```

## Docker

CPU profile:

```bash
docker compose --profile cpu up --build
```

Then open:

```text
http://localhost:8080
```

GPU/TGI profile placeholder:

```bash
docker compose --profile gpu up --build
```

Final production path for this homework should use **ByT5 + HuggingFace TGI** for true
encoder-decoder token streaming. The current gateway and UI are already SSE-compatible.

## Latency Benchmark

```bash
python scripts/benchmark_streaming.py --url http://localhost:8080/translate --runs 10
```

Report:

- median TTFT;
- p95 TTFT;
- tokens/sec.

## Experiment Tracking

Training uses `report_to=["wandb"]` in the Hugging Face trainer. For offline debugging:

```bash
export WANDB_MODE=offline
```

For final submission, sync runs and attach:

- hyperparameters;
- train loss curves;
- dev BLEU / chrF++;
- best checkpoint artifact.

## Dataset And License Log

| Dataset | Purpose | License / Terms | Leakage Check |
|---|---|---|---|
| Kaggle Deep Past train.csv | supervised training | Kaggle competition terms | official train only |
| ORACC | optional monolingual/external data | TODO | exact + near duplicate vs test |
| CDLI | optional external data | TODO | exact + near duplicate vs test |

## Submission Checklist

- [ ] Public GitHub/GitLab repo.
- [ ] `main` and `develop` branches.
- [ ] `README.md` has full name and group.
- [ ] Kaggle leaderboard screenshot is added.
- [ ] Streaming UI GIF/video is added.
- [ ] `data/log_file.log` is visible inside the container.
- [ ] No model weights or raw corpora are committed.
- [ ] W&B or ClearML run links are added.
- [ ] Docker Compose starts the service.
