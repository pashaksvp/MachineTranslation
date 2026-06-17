# Домашнее задание 3: потоковый переводчик с аккадского

## Автор

- ФИО: Фомин Павел Андреевич
- Группа: 972402

Репозиторий реализует полный пайплайн low-resource NMT для соревнования Kaggle
**Deep Past Initiative: Machine Translation**, направление **аккадская транслитерация -> английский**.

Проект выполнен по этапам:

1. очистка данных и dev-разбиение;
2. fine-tuning baseline-модели ByT5;
3. ablation-эксперименты: нормализация орфографии, beam search и mini-ensemble;
4. экспорт предсказаний в формате Kaggle;
5. streaming translator API и web UI;
6. deployment через Docker Compose.

## Текущий статус

- Данные Kaggle загружены и разбиты на raw и normalized train/dev наборы.
- Локально обучен ByT5-small baseline на 1 эпоху.
- Выполнены обязательные ablation-эксперименты: orthography normalization, beam search, mini-ensemble.
- Сгенерирован Kaggle-файл `data/results.csv`. Поздняя отправка через Kaggle API недоступна после публичного дедлайна соревнования.
- Локально проверены FastAPI SSE backend и web UI.
- Установочный wheel-артефакт собран в `dist/`.
- `docker-compose.yaml` подготовлен.

## Архитектура

- `model.py` содержит обязательный класс `My_Translator_Model`.
- `src/akkadian_mt/translator.py` содержит переиспользуемую реализацию модели.
- `src/akkadian_mt/normalization.py` содержит правила нормализации аккадской транслитерации.
- `api/server.py` обслуживает `/health`, `/translate` и web UI.
- `web/` содержит двухпанельный интерфейс потокового переводчика.
- `scripts/` содержит утилиты для разбиения данных, оценки, latency benchmark и проверок.
- `data/log_file.log` автоматически создается singleton-logger'ом.

## Установка

Вариант с `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Вариант с `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Установка собранного wheel:

```bash
pip install dist/akkadian_streaming_translator-0.1.0-py3-none-any.whl
```

## Данные

При использовании локального virtual environment скрипт автоматически выберет `.venv/bin/kaggle`
после установки зависимостей. Скрипт скачивает только необходимые файлы соревнования:
`train.csv`, `test.csv` и `sample_submission.csv`.

Ожидаемые колонки определяются по типовым именам:

- source: `akkadian`, `source`, `src`, `transliteration` или `text`;
- target: `english`, `target`, `translation` или `tgt`;
- id: `id`.

Проверка загруженных файлов перед обучением:

```bash
python scripts/inspect_data.py data/raw/train.csv data/raw/test.csv data/raw/sample_submission.csv
```

Полный data stage одной командой:

```bash
python scripts/run_data_stage.py
```

Команда проверяет обязательные Kaggle-файлы, печатает статистику датасета и создает raw и normalized train/dev split.

Для каждого внешнего корпуса нужно проверять exact и near duplicates относительно Kaggle test sources:

```bash
python scripts/check_leakage.py \
  --reference data/raw/test.csv \
  --candidate data/external/oracc_clean.csv \
  --normalize
```

Если найдены exact или near overlaps, их нужно удалить из внешнего корпуса до обучения.

## Подготовка dev split

Baseline split без нормализации:

```bash
python scripts/prepare_split.py \
  --dataset data/raw/train.csv \
  --output-dir data/processed/raw
```

Split с нормализацией:

```bash
python scripts/prepare_split.py \
  --dataset data/raw/train.csv \
  --output-dir data/processed/normalized \
  --normalize
```

Текущий normalizer:

- использует Unicode NFC;
- переводит subscript digits, например `₂`, в `2`;
- удаляет broken-sign brackets `⸢...⸣`;
- разворачивает angle brackets `<...>`;
- сохраняет determinatives как отдельные `{d}`-style токены;
- заменяет lacunae `[ ... ]` на `<lacuna>`;
- по умолчанию сохраняет аккадские diacritics;
- приводит текст к нижнему регистру.

## Обучение baseline

Модель по умолчанию: `google/byt5-small`.

```bash
python model.py train \
  --dataset data/processed/raw/train_split.csv \
  --model-dir model/baseline_raw \
  --seed 42 \
  --num-train-epochs 3 \
  --learning-rate 5e-4
```

Модель сохраняется в:

```text
model/baseline_raw/
```

Веса намеренно игнорируются Git.

Ablation с нормализацией орфографии:

```bash
python model.py train \
  --dataset data/processed/normalized/train_split.csv \
  --model-dir model/baseline_normalized \
  --normalize \
  --seed 42 \
  --num-train-epochs 3 \
  --learning-rate 5e-4
```

Для mini-ensemble нужно обучить минимум два checkpoint с разными seed:

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

Затем можно оценить checkpoint'ы и экспортировать предсказания от лучшей по dev chrF++ модели:

```bash
python scripts/run_ensemble.py \
  --dev-dataset data/processed/raw/dev_split.csv \
  --test-dataset data/raw/test.csv \
  --model-dirs model/baseline_seed_42,model/baseline_seed_777 \
  --num-beams 1 \
  --max-target-length 128 \
  --metrics-out data/processed/raw/ensemble_metrics.csv \
  --submission-out data/results.csv
```

## Перевод одного предложения

```bash
python model.py predict \
  --text "šarrum ana ālim illik" \
  --model-dir model/baseline_raw \
  --num-beams 1 \
  --max-target-length 128
```

Multi-candidate decoding с deterministic selector:

```bash
python model.py predict \
  --text "šarrum ana ālim illik" \
  --model-dir model/baseline_raw \
  --num-beams 4 \
  --num-return-sequences 4 \
  --selector-strategy first_non_empty \
  --max-target-length 128
```

## Экспорт Kaggle submission

```bash
python model.py predict-file \
  --dataset data/raw/test.csv \
  --model-dir model/baseline_raw \
  --num-beams 1 \
  --max-target-length 128
```

Результат:

```text
data/results.csv
```

Перед загрузкой нужно проверить файл:

```bash
python scripts/validate_submission.py \
  --sample data/raw/sample_submission.csv \
  --submission data/results.csv
```

Если Kaggle требует другое точное имя prediction-колонки, нужно свериться с `sample_submission.csv`
и переименовать колонку перед отправкой.

## Оценка dev metrics

```bash
python scripts/evaluate.py \
  --dataset data/processed/raw/dev_split.csv \
  --model-dir model/baseline_raw \
  --num-beams 1 \
  --max-target-length 128 \
  --predictions-out data/processed/raw/dev_predictions.csv
```

Быстрый smoke evaluation на небольшой подвыборке:

```bash
python scripts/evaluate.py \
  --dataset data/processed/raw/dev_split.csv \
  --model-dir model/baseline_raw \
  --num-beams 1 \
  --limit 20 \
  --predictions-out data/processed/raw/dev_predictions_smoke.csv
```

Опциональная COMET-оценка:

```bash
pip install -e ".[comet]"
python scripts/evaluate.py \
  --dataset data/processed/raw/dev_split.csv \
  --model-dir model/baseline_raw \
  --num-beams 1 \
  --max-target-length 128 \
  --comet
```

Метрики, которые нужно отчитаться:

- BLEU через sacreBLEU;
- chrF++ с word order 2;
- COMET как auxiliary metric через `--comet` (COMET не обучался на аккадском, поэтому это diagnostic metric);
- Kaggle public/private leaderboard score.

Текущие локальные baseline metrics:

| Запуск | BLEU на dev | chrF++ на dev | Заметки |
|---|---:|---:|---|
| ByT5-small raw, 1 epoch, greedy, max target length 128 | 0.2666 | 10.3048 | `model/baseline_raw`, dev size 157 |
| ByT5-small raw, 1 epoch, beam size 4, max target length 128 | 0.2369 | 10.1660 | `model/baseline_raw`, dev size 157 |
| ByT5-small raw, 1 epoch, beam size 8, max target length 128 | 0.2466 | 10.0706 | `model/baseline_raw`, dev size 157 |
| ByT5-small raw, seed 777, 1 epoch, greedy, max target length 128 | 0.0838 | 8.3695 | `model/baseline_seed_777`, dev size 157 |
| Mini-ensemble best-dev selector, seeds 42 and 777 | 0.2666 | 10.3048 | выбран `model/baseline_raw` |
| ByT5-small normalized, 1 epoch, greedy, max target length 128 | 0.1648 | 9.2342 | `model/baseline_normalized`, dev size 157 |

Примечание по Kaggle API submission: текущий baseline `data/results.csv` успешно сгенерирован,
но Kaggle вернул `400 Bad Request` при создании submission после дедлайна соревнования,
который API показывает как `2026-03-23 23:59:00`. Файл стоит сохранить для ручной late-submission
попытки или приложить к отчету, если поздние отправки закрыты.

## Beam Search Sweep

```bash
python scripts/run_beam_sweep.py \
  --dataset data/processed/raw/dev_split.csv \
  --model-dir model/baseline_raw \
  --beams 4,8 \
  --max-target-length 128 \
  --output data/processed/raw/beam_sweep_4_8.csv
```

## Обязательные ablations

Итоговая таблица экспериментов:

| Technique | chrF++ on dev | Kaggle public LB | Notes |
|---|---:|---:|---|
| Baseline ByT5-small, greedy, no normalization | 10.3048 | unavailable | Kaggle API late submission blocked after deadline |
| Orthography normalization | 9.2342 | unavailable | Worse than raw for the 1-epoch local baseline |
| Beam search, beam size sweep `{1, 4, 8}` | 10.3048 | unavailable | Greedy/beam=1 won; beam=4: 10.1660, beam=8: 10.0706 |
| Mini-ensemble, 2 checkpoints | 10.3048 | unavailable | Seeds 42/777; best-dev selector picked seed 42 |

## Streaming Service

Локальный запуск:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8080
```

Открыть:

```text
http://localhost:8080
```

Health check:

```bash
curl http://localhost:8080/health
```

Просмотр и скачивание логов внутри запущенного контейнера:

```bash
curl http://localhost:8080/logs
curl -O -J http://localhost:8080/logs/download
```

Streaming endpoint:

```bash
curl -N -X POST http://localhost:8080/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"šarrum ana ālim illik"}'
```

Локальная SSE-проверка прошла с `model/baseline_raw`. Пример streamed chunks:

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

Build context образа исключает raw corpora, checkpoints, `.venv` и W&B artifacts через
`.dockerignore`; runtime-директории `./model` и `./data` монтируются в контейнер через Compose volumes.

После запуска открыть:

```text
http://localhost:8080
```

GPU/TGI profile:

```bash
docker compose --profile gpu up --build
```

GPU profile запускает оба сервиса:

- `tgi` на порту `8081`, serving `./model/baseline_raw`;
- `translator-tgi` на порту `8080`, serving того же UI и forwarding streaming translation requests
  в TGI через `TGI_URL=http://tgi:80`.

Production path: **ByT5 + HuggingFace TGI** для encoder-decoder token streaming.
CPU profile остается локальным fallback-режимом, который стримит сгенерированный текст через FastAPI gateway.

Статус локальной Docker-проверки: на текущей машине Docker CLI не установлен
(`docker: command not found`). Compose-файл подготовлен и должен быть проверен на Docker-enabled host
перед финальной сдачей.

## Latency Benchmark

```bash
python scripts/benchmark_streaming.py --url http://localhost:8080/translate --runs 10
```

Нужно отчитаться:

- median TTFT;
- p95 TTFT;
- tokens/sec.

Локально после model warmup измерено:

| Backend | Device | Median TTFT | p95 TTFT | Median tokens/sec |
|---|---|---:|---:|---:|
| FastAPI + local ByT5-small checkpoint | Apple MPS/local Mac | 0.5120s | 0.5127s | 40.37 |

## Experiment Tracking

Обучение использует `report_to=["wandb"]` в Hugging Face trainer. Для offline debugging:

```bash
export WANDB_MODE=offline
```

Для финальной сдачи нужно синхронизировать runs и приложить:

- hyperparameters;
- train loss curves;
- dev BLEU / chrF++;
- best checkpoint artifact.

Локальные W&B offline runs:

```text
wandb/offline-run-20260601_232749-d83o4fyw  # raw baseline seed 42
wandb/offline-run-20260603_131813-6rv810mu  # normalized baseline
wandb/offline-run-20260604_151724-gblbyz58  # raw baseline seed 777
```

## Датасеты и лицензии

| Dataset | Назначение | License / Terms | Leakage Check |
|---|---|---|---|
| Kaggle Deep Past train.csv | supervised training | Kaggle competition terms | official train only |
| Kaggle Deep Past test.csv | Kaggle prediction export | Kaggle competition terms | never used for training |
| External corpora | not used in current local run | n/a | n/a |

## Чеклист сдачи

Перед финальной загрузкой запустить local readiness checker:

```bash
python scripts/check_project_ready.py
```

- [x] Публичный GitHub/GitLab repository.
- [x] Ветки `main` и `develop`.
- [x] В `README.md` указаны ФИО и группа.
- [x] Добавлен GIF/video с демонстрацией streaming UI.
- [x] `data/log_file.log` создается singleton logger'ом.
- [x] Model weights и raw corpora не закоммичены.
- [x] W&B offline runs записаны локально.

