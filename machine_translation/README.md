# Homework 3: Akkadian Streaming Translator

## Author

- Full name: TODO
- Group: TODO

This repository implements an end-to-end low-resource NMT project for the Kaggle competition
**Deep Past Initiative: Machine Translation**, direction **Akkadian transliteration -> English**.

The project is built in stages:

1. data cleaning and dev split;
2. ByT5 baseline fine-tuning;
3. ablations for orthography normalization, beam search, and mini-ensemble;
4. Kaggle prediction export;
5. streaming translator API and web UI;
6. Docker Compose deployment.

## Current Status

- Kaggle data downloaded and split into raw and normalized train/dev sets.
- ByT5-small baseline trained locally for 1 epoch.
- Required ablations completed: orthography normalization, beam search, mini-ensemble.
- Kaggle-format `data/results.csv` generated. Kaggle API late submission is unavailable after the public competition deadline.
- FastAPI SSE backend and web UI verified locally.
- Installable wheel artifact is built under `dist/`.
- Docker Compose file is present, but Docker CLI is not installed on the current machine, so container verification is still pending.

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

Install the packaged wheel:

```bash
pip install dist/akkadian_streaming_translator-0.1.0-py3-none-any.whl
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
automatically after dependencies are installed. The script downloads only the required
competition files: `train.csv`, `test.csv`, and `sample_submission.csv`.

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
  --model-dir model/baseline_raw \
  --seed 42 \
  --num-train-epochs 3 \
  --learning-rate 5e-4
```

The model is saved to:

```text
model/
```

Weights are intentionally ignored by git. Upload final weights to Hugging Face Hub, W&B, or ClearML.

For the orthography normalization ablation:

```bash
python model.py train \
  --dataset data/processed/normalized/train_split.csv \
  --model-dir model/baseline_normalized \
  --normalize \
  --seed 42 \
  --num-train-epochs 3 \
  --learning-rate 5e-4
```

For the mini-ensemble ablation, train at least two checkpoints with different seeds:

```bash
python model.py train \
  --dataset data/processed/raw/train_split.csv \
  --model-dir model/baseline_seed_42 \
  --seed 42

python model.py train \
  --dataset data/processed/raw/train_split.csv \
  --model-dir model/baseline_seed_777 \
  --seed 777
```

Then evaluate the checkpoints and export predictions from the best dev chrF++ model:

```bash
python scripts/run_ensemble.py \
  --dev-dataset data/processed/raw/dev_split.csv \
  --test-dataset data/raw/test.csv \
  --model-dirs model/baseline_seed_42,model/baseline_seed_777 \
  --num-beams 4 \
  --metrics-out data/processed/raw/ensemble_metrics.csv \
  --submission-out data/results.csv
```

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

For a quick smoke evaluation on a small subset:

```bash
python scripts/evaluate.py \
  --dataset data/processed/raw/dev_split.csv \
  --model-dir model/baseline_raw \
  --num-beams 1 \
  --limit 20 \
  --predictions-out data/processed/raw/dev_predictions_smoke.csv
```

Required metrics to report:

- BLEU with sacreBLEU;
- chrF++ with word order 2;
- COMET as auxiliary metric;
- Kaggle public/private leaderboard score.

Current local baseline metrics:

| Run | BLEU on dev | chrF++ on dev | Notes |
|---|---:|---:|---|
| ByT5-small raw, 1 epoch, greedy, max target length 128 | 0.2666 | 10.3048 | `model/baseline_raw`, dev size 157 |
| ByT5-small raw, 1 epoch, beam size 4, max target length 128 | 0.2369 | 10.1660 | `model/baseline_raw`, dev size 157 |
| ByT5-small raw, 1 epoch, beam size 8, max target length 128 | 0.2466 | 10.0706 | `model/baseline_raw`, dev size 157 |
| ByT5-small raw, seed 777, 1 epoch, greedy, max target length 128 | 0.0838 | 8.3695 | `model/baseline_seed_777`, dev size 157 |
| Mini-ensemble best-dev selector, seeds 42 and 777 | 0.2666 | 10.3048 | selected `model/baseline_raw` |
| ByT5-small normalized, 1 epoch, greedy, max target length 128 | 0.1648 | 9.2342 | `model/baseline_normalized`, dev size 157 |

Kaggle API submission note: the current baseline `data/results.csv` was generated successfully,
but Kaggle returned `400 Bad Request` on submission creation after the competition deadline
listed by the API as `2026-03-23 23:59:00`. Keep the file for manual late-submission attempts
or attach it in the homework report if late submissions remain closed.

## Beam Search Sweep

```bash
python scripts/run_beam_sweep.py \
  --dataset data/processed/raw/dev_split.csv \
  --model-dir model/baseline_raw \
  --beams 4,8 \
  --max-target-length 128 \
  --output data/processed/raw/beam_sweep_4_8.csv
```

## Mandatory Ablations

Fill this table after running experiments.

| Technique | chrF++ on dev | Kaggle public LB | Notes |
|---|---:|---:|---|
| Baseline ByT5-small, greedy, no normalization | 10.3048 | unavailable | Kaggle API late submission blocked after deadline |
| Orthography normalization | 9.2342 | unavailable | Worse than raw for the 1-epoch local baseline |
| Beam search, beam size sweep `{1, 4, 8}` | 10.3048 | unavailable | Greedy/beam=1 won; beam=4: 10.1660, beam=8: 10.0706 |
| Mini-ensemble, 2 checkpoints | 10.3048 | unavailable | Seeds 42/777; best-dev selector picked seed 42 |

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

Local SSE check passed with `model/baseline_raw`. Example streamed chunks:

```text
data: {"token": "To "}
data: {"token": "Ali-ahum "}
data: {"token": "son "}
data: {"token": "of "}
data: {"token": "Ilīk. "}
data: {"done": "true"}
```

Demo video: [`docs/streaming-demo.mp4`](docs/streaming-demo.mp4).

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

Local Docker verification status: not run on the current machine because Docker CLI is not
installed (`docker: command not found`). The compose file is present and should be checked on
a Docker-enabled host before final submission.

## Latency Benchmark

```bash
python scripts/benchmark_streaming.py --url http://localhost:8080/translate --runs 10
```

Report:

- median TTFT;
- p95 TTFT;
- tokens/sec.

Measured locally after model warmup:

| Backend | Device | Median TTFT | p95 TTFT | Median tokens/sec |
|---|---|---:|---:|---:|
| FastAPI + local ByT5-small checkpoint | Apple MPS/local Mac | 0.5120s | 0.5127s | 40.37 |

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

Local W&B offline runs:

```text
wandb/offline-run-20260601_232749-d83o4fyw  # raw baseline seed 42
wandb/offline-run-20260603_131813-6rv810mu  # normalized baseline
wandb/offline-run-20260604_151724-gblbyz58  # raw baseline seed 777
```

## Dataset And License Log

| Dataset | Purpose | License / Terms | Leakage Check |
|---|---|---|---|
| Kaggle Deep Past train.csv | supervised training | Kaggle competition terms | official train only |
| Kaggle Deep Past test.csv | Kaggle prediction export | Kaggle competition terms | never used for training |
| External corpora | not used in current local run | n/a | n/a |

## Submission Checklist

- [x] Public GitHub/GitLab repo.
- [x] `main` and `develop` branches.
- [ ] `README.md` has full name and group.
- [ ] Kaggle leaderboard screenshot is added.
- [x] Streaming UI GIF/video is added.
- [x] `data/log_file.log` is created by the singleton logger.
- [x] No model weights or raw corpora are committed.
- [x] W&B offline runs are recorded locally.
- [ ] Docker Compose starts the service on a Docker-enabled host.
