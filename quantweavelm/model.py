"""Deterministic mixture fitting, temperature scaling, and proper scores."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

EPSILON = 1e-15


def normalize(values: Sequence[float]) -> list[float]:
    clipped = [max(EPSILON, float(value)) for value in values]
    total = sum(clipped)
    return [value / total for value in clipped]


def mixture(forecasts: dict[str, list[float]], names: Sequence[str],
            weights: Sequence[float]) -> list[float]:
    """Linearly pool probability vectors with simplex weights."""
    classes = len(forecasts[names[0]])
    return normalize([sum(weights[index] * forecasts[name][label]
                          for index, name in enumerate(names))
                      for label in range(classes)])


def apply_temperature(probabilities: Sequence[float], temperature: float) -> list[float]:
    """Apply multiclass temperature scaling to probabilities."""
    return normalize([max(EPSILON, value) ** (1.0 / temperature) for value in probabilities])


def fit_weights(rows: Sequence[dict], names: Sequence[str], iterations: int,
                learning_rate: float) -> list[float]:
    """Minimize mixture log loss by deterministic exponentiated-gradient updates."""
    weights = [1.0 / len(names)] * len(names)
    for _ in range(iterations):
        gradients = [0.0] * len(names)
        for row in rows:
            target = row["target_bin"]
            pooled = max(EPSILON, sum(weights[index] * row["forecasts"][name][target]
                                      for index, name in enumerate(names)))
            for index, name in enumerate(names):
                gradients[index] -= row["forecasts"][name][target] / pooled
        gradients = [value / len(rows) for value in gradients]
        updated = [weights[index] * math.exp(max(-50.0, min(50.0, -learning_rate * gradients[index])))
                   for index in range(len(names))]
        weights = normalize(updated)
    return weights


def log_loss(probabilities: Sequence[float], target: int) -> float:
    return -math.log(max(EPSILON, probabilities[target]))


def multiclass_brier(probabilities: Sequence[float], target: int) -> float:
    return sum((value - (1.0 if index == target else 0.0)) ** 2
               for index, value in enumerate(probabilities))


def ranked_probability_score(probabilities: Sequence[float], target: int) -> float:
    """Score ordered bins using cumulative probability errors."""
    cumulative = 0.0
    result = 0.0
    for index, value in enumerate(probabilities[:-1]):
        cumulative += value
        observed = 1.0 if target <= index else 0.0
        result += (cumulative - observed) ** 2
    return result


def score(probabilities: Sequence[float], target: int) -> tuple[float, float, float, int, float]:
    predicted = max(range(len(probabilities)), key=lambda index: (probabilities[index], -index))
    return (log_loss(probabilities, target), multiclass_brier(probabilities, target),
            ranked_probability_score(probabilities, target), int(predicted == target),
            probabilities[predicted])


def aggregate(predictions: Sequence[Sequence[float]], targets: Sequence[int],
              reliability_bins: int = 10) -> dict:
    """Return proper scores plus explicit top-label and classwise reliability tables."""
    totals = [0.0] * 5
    top_buckets = [{"count": 0, "confidence_sum": 0.0, "correct": 0} for _ in range(reliability_bins)]
    class_buckets = [[{"count": 0, "probability_sum": 0.0, "observed": 0}
                      for _ in range(reliability_bins)] for _ in range(len(predictions[0]))]
    for probabilities, target in zip(predictions, targets):
        values = score(probabilities, target)
        totals = [left + right for left, right in zip(totals, values)]
        confidence = values[4]
        bucket = min(reliability_bins - 1, int(confidence * reliability_bins))
        top_buckets[bucket]["count"] += 1
        top_buckets[bucket]["confidence_sum"] += confidence
        top_buckets[bucket]["correct"] += values[3]
        for label, probability in enumerate(probabilities):
            class_bucket = min(reliability_bins - 1, int(probability * reliability_bins))
            cell = class_buckets[label][class_bucket]
            cell["count"] += 1
            cell["probability_sum"] += probability
            cell["observed"] += int(label == target)
    count = len(targets)

    def top_rows() -> tuple[list[dict], float]:
        rows, ece = [], 0.0
        for index, cell in enumerate(top_buckets):
            if not cell["count"]:
                continue
            mean = cell["confidence_sum"] / cell["count"]
            accuracy = cell["correct"] / cell["count"]
            ece += cell["count"] / count * abs(mean - accuracy)
            rows.append({"bin": index, "count": cell["count"], "mean_confidence": mean,
                         "accuracy": accuracy})
        return rows, ece

    top_table, top_ece = top_rows()
    class_tables, class_ece = [], []
    for cells in class_buckets:
        table, ece = [], 0.0
        for index, cell in enumerate(cells):
            if not cell["count"]:
                continue
            mean = cell["probability_sum"] / cell["count"]
            frequency = cell["observed"] / cell["count"]
            ece += cell["count"] / count * abs(mean - frequency)
            table.append({"bin": index, "count": cell["count"], "mean_probability": mean,
                          "observed_frequency": frequency})
        class_tables.append(table)
        class_ece.append(ece)
    return {"count": count, "log_loss": totals[0] / count,
            "multiclass_brier": totals[1] / count, "ranked_probability_score": totals[2] / count,
            "top_label_accuracy": totals[3] / count, "mean_confidence": totals[4] / count,
            "top_label_ece": top_ece, "classwise_ece": class_ece,
            "top_label_reliability": top_table, "classwise_reliability": class_tables}


def fit_temperature(rows: Sequence[dict], names: Sequence[str], weights: Sequence[float],
                    minimum: float, maximum: float, steps: int) -> tuple[float, float]:
    """Choose a scalar temperature on calibration rows only by log-grid search."""
    best = (math.inf, minimum)
    log_min, log_max = math.log(minimum), math.log(maximum)
    for step in range(steps):
        temperature = math.exp(log_min + (log_max - log_min) * step / (steps - 1))
        loss = sum(log_loss(apply_temperature(mixture(row["forecasts"], names, weights), temperature),
                            row["target_bin"]) for row in rows) / len(rows)
        if loss < best[0] - 1e-15:
            best = (loss, temperature)
    return best[1], best[0]
