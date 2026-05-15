from __future__ import annotations

import sacrebleu


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    return sacrebleu.corpus_bleu(predictions, [references]).score


def compute_chrfpp(predictions: list[str], references: list[str]) -> float:
    metric = sacrebleu.CHRF(word_order=2)
    return metric.corpus_score(predictions, [references]).score
