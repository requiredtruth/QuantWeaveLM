"""Chronological partitioning and reproducible calibration pipeline."""

from __future__ import annotations

from typing import Any, Sequence

from . import __version__
from .core import NAME_RE, QuantWeaveError, digest, finite_number, timestamp
from .model import aggregate, apply_temperature, fit_temperature, fit_weights, mixture, normalize


def validate_config(raw: Any) -> dict[str, Any]:
    """Validate a versioned experiment configuration and reject silent defaults."""
    required = {"schema_version", "bins", "models", "split", "optimizer"}
    if not isinstance(raw, dict) or set(raw) != required or raw.get("schema_version") != 1:
        raise QuantWeaveError("config must contain exactly schema_version=1, bins, models, split, optimizer")
    bins = raw["bins"]
    if not isinstance(bins, dict) or set(bins) != {"edges", "labels"}:
        raise QuantWeaveError("bins requires exactly edges and labels")
    if not isinstance(bins["edges"], list) or not 1 <= len(bins["edges"]) <= 31:
        raise QuantWeaveError("bins.edges must contain 1..31 finite return boundaries")
    edges = [finite_number(value, f"bins.edges[{index}]") for index, value in enumerate(bins["edges"])]
    if any(left >= right for left, right in zip(edges, edges[1:])):
        raise QuantWeaveError("bins.edges must be strictly increasing")
    labels = bins["labels"]
    if (not isinstance(labels, list) or len(labels) != len(edges) + 1 or
            any(not isinstance(value, str) or not NAME_RE.fullmatch(value) for value in labels) or
            len(set(labels)) != len(labels)):
        raise QuantWeaveError("bins.labels must be unique safe names with len(edges)+1 entries")
    models = raw["models"]
    if (not isinstance(models, list) or not 2 <= len(models) <= 32 or
            any(not isinstance(value, str) or not NAME_RE.fullmatch(value) for value in models) or
            len(set(models)) != len(models)):
        raise QuantWeaveError("models must contain 2..32 unique safe names")
    split = raw["split"]
    split_fields = {"calibration_rows", "embargo_rows", "test_rows", "minimum_embargo_seconds"}
    if not isinstance(split, dict) or set(split) != split_fields:
        raise QuantWeaveError("split fields are calibration_rows, embargo_rows, test_rows, minimum_embargo_seconds")
    for field, minimum in (("calibration_rows", 20), ("embargo_rows", 1), ("test_rows", 10),
                           ("minimum_embargo_seconds", 1)):
        if isinstance(split[field], bool) or not isinstance(split[field], int) or split[field] < minimum:
            raise QuantWeaveError(f"split.{field} must be an integer >= {minimum}")
    optimizer = raw["optimizer"]
    optimizer_fields = {"iterations", "learning_rate", "temperature_min", "temperature_max",
                        "temperature_steps"}
    if not isinstance(optimizer, dict) or set(optimizer) != optimizer_fields:
        raise QuantWeaveError("optimizer fields do not match schema version 1")
    iterations = optimizer["iterations"]
    steps = optimizer["temperature_steps"]
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 10_000:
        raise QuantWeaveError("optimizer.iterations must be an integer in 1..10000")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 2 <= steps <= 1001:
        raise QuantWeaveError("optimizer.temperature_steps must be an integer in 2..1001")
    learning_rate = finite_number(optimizer["learning_rate"], "optimizer.learning_rate")
    minimum = finite_number(optimizer["temperature_min"], "optimizer.temperature_min")
    maximum = finite_number(optimizer["temperature_max"], "optimizer.temperature_max")
    if not 0.0001 <= learning_rate <= 1.0:
        raise QuantWeaveError("optimizer.learning_rate must be in 0.0001..1")
    if not 0.05 <= minimum < maximum <= 20.0:
        raise QuantWeaveError("temperature bounds must satisfy 0.05 <= min < max <= 20")
    return {"schema_version": 1, "bins": {"edges": edges, "labels": list(labels)},
            "models": list(models), "split": dict(split),
            "optimizer": {"iterations": iterations, "learning_rate": learning_rate,
                          "temperature_min": minimum, "temperature_max": maximum,
                          "temperature_steps": steps}}


def _target_bin(value: float, edges: Sequence[float]) -> int:
    for index, edge in enumerate(edges):
        if value < edge:
            return index
    return len(edges)


def validate_rows(raw_rows: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate ordered observations and derive targets from returns."""
    if not isinstance(raw_rows, list):
        raise QuantWeaveError("dataset must be JSONL records")
    split = config["split"]
    expected = split["calibration_rows"] + split["embargo_rows"] + split["test_rows"]
    if len(raw_rows) != expected:
        raise QuantWeaveError(f"dataset has {len(raw_rows)} rows; split requires exactly {expected}")
    normalized = []
    previous = None
    classes = len(config["bins"]["labels"])
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict) or set(raw) != {"timestamp", "target_return", "forecasts"}:
            raise QuantWeaveError(f"row {index} requires exactly timestamp, target_return, forecasts")
        parsed = timestamp(raw["timestamp"], f"row {index}.timestamp")
        if previous is not None and parsed <= previous:
            raise QuantWeaveError("timestamps must be strictly increasing")
        previous = parsed
        target_return = finite_number(raw["target_return"], f"row {index}.target_return")
        forecasts = raw["forecasts"]
        if not isinstance(forecasts, dict) or set(forecasts) != set(config["models"]):
            raise QuantWeaveError(f"row {index}.forecasts must exactly match configured models")
        normalized_forecasts = {}
        for name in config["models"]:
            values = forecasts[name]
            if not isinstance(values, list) or len(values) != classes:
                raise QuantWeaveError(f"row {index} model {name} requires {classes} probabilities")
            probabilities = [finite_number(value, f"row {index}.{name}[{label}]")
                             for label, value in enumerate(values)]
            if any(value < 0.0 or value > 1.0 for value in probabilities):
                raise QuantWeaveError(f"row {index} model {name} probabilities must be in 0..1")
            if abs(sum(probabilities) - 1.0) > 1e-9:
                raise QuantWeaveError(f"row {index} model {name} probabilities must sum to 1")
            normalized_forecasts[name] = normalize(probabilities)
        normalized.append({"timestamp": raw["timestamp"], "target_return": target_return,
                           "target_bin": _target_bin(target_return, config["bins"]["edges"]),
                           "forecasts": normalized_forecasts})
    calibration_end = split["calibration_rows"] - 1
    test_start = split["calibration_rows"] + split["embargo_rows"]
    seconds = (timestamp(normalized[test_start]["timestamp"], "test start") -
               timestamp(normalized[calibration_end]["timestamp"], "calibration end")).total_seconds()
    if seconds < split["minimum_embargo_seconds"]:
        raise QuantWeaveError(f"timestamp embargo is {int(seconds)} seconds; minimum is {split['minimum_embargo_seconds']}")
    return normalized


def _predictions(rows: Sequence[dict], config: dict[str, Any], weights: Sequence[float],
                 temperature: float | None = None) -> list[list[float]]:
    pooled = [mixture(row["forecasts"], config["models"], weights) for row in rows]
    return pooled if temperature is None else [apply_temperature(values, temperature) for values in pooled]


def _metrics(rows: Sequence[dict], config: dict[str, Any], weights: Sequence[float],
             temperature: float, climatology: Sequence[float]) -> dict[str, Any]:
    targets = [row["target_bin"] for row in rows]
    equal = [1.0 / len(config["models"])] * len(config["models"])
    methods = {name: [row["forecasts"][name] for row in rows] for name in config["models"]}
    methods["equal_pool"] = _predictions(rows, config, equal)
    methods["fitted_pool"] = _predictions(rows, config, weights)
    methods["calibrated_pool"] = _predictions(rows, config, weights, temperature)
    methods["calibration_climatology"] = [list(climatology) for _ in rows]
    return {name: aggregate(predictions, targets) for name, predictions in methods.items()}


def run_pipeline(raw_config: Any, raw_rows: Any) -> dict[str, Any]:
    """Fit exclusively on the calibration partition and score both declared partitions."""
    config = validate_config(raw_config)
    rows = validate_rows(raw_rows, config)
    split = config["split"]
    calibration = rows[:split["calibration_rows"]]
    embargo = rows[split["calibration_rows"]:split["calibration_rows"] + split["embargo_rows"]]
    test = rows[-split["test_rows"]:]
    optimizer = config["optimizer"]
    weights = fit_weights(calibration, config["models"], optimizer["iterations"],
                          optimizer["learning_rate"])
    temperature, objective = fit_temperature(calibration, config["models"], weights,
                                              optimizer["temperature_min"],
                                              optimizer["temperature_max"],
                                              optimizer["temperature_steps"])
    classes = len(config["bins"]["labels"])
    counts = [1] * classes
    for row in calibration:
        counts[row["target_bin"]] += 1
    climatology = [count / sum(counts) for count in counts]
    return {"schema_version": 1, "tool_version": __version__,
            "evidence": {"config_sha256": digest(config), "dataset_sha256": digest(rows)},
            "bins": config["bins"], "models": config["models"], "split": {
                **split,
                "calibration": {"first": calibration[0]["timestamp"], "last": calibration[-1]["timestamp"]},
                "embargo": {"first": embargo[0]["timestamp"], "last": embargo[-1]["timestamp"]},
                "test": {"first": test[0]["timestamp"], "last": test[-1]["timestamp"]}},
            "learned": {"weights": {name: weights[index] for index, name in enumerate(config["models"])},
                        "temperature": temperature, "calibration_log_loss": objective,
                        "calibration_climatology": climatology},
            "metrics": {"calibration": _metrics(calibration, config, weights, temperature, climatology),
                        "test": _metrics(test, config, weights, temperature, climatology)},
            "claims": {"mode": "offline_research_only", "test_used_for_fitting": False,
                       "embargo_rows_used_for_fitting": False, "profit_or_return_claim": False}}
