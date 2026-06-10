from __future__ import annotations

import sacrebleu


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    return sacrebleu.corpus_bleu(predictions, [references]).score


def compute_chrfpp(predictions: list[str], references: list[str]) -> float:
    metric = sacrebleu.CHRF(word_order=2)
    return metric.corpus_score(predictions, [references]).score


def compute_comet(
    predictions: list[str],
    references: list[str],
    sources: list[str],
    model_name: str = "Unbabel/wmt22-comet-da",
) -> float:
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as exc:
        raise RuntimeError(
            "COMET is optional. Install it with `pip install -e .[comet]` first."
        ) from exc

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    samples = [
        {"src": source, "mt": prediction, "ref": reference}
        for source, prediction, reference in zip(sources, predictions, references, strict=True)
    ]
    result = model.predict(samples, batch_size=8, gpus=0)
    return float(result.system_score)
