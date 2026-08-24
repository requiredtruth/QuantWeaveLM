"""Report verification, presentation, and bounded local-model prompt export."""

from __future__ import annotations

from typing import Any

from .core import QuantWeaveError, canonical_bytes, digest
from .pipeline import run_pipeline


def verify(raw_config: Any, raw_rows: Any, report: Any) -> dict[str, Any]:
    """Recompute the complete report and require canonical equality."""
    expected = run_pipeline(raw_config, raw_rows)
    if canonical_bytes(report) != canonical_bytes(expected):
        raise QuantWeaveError("report does not match a fresh deterministic recomputation")
    return report


def summary(report: Any) -> str:
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise QuantWeaveError("report schema_version must be 1")
    learned = report["learned"]
    test = report["metrics"]["test"]
    rows = ["QuantWeaveLM deterministic calibration report",
            f"rows calibration={report['split']['calibration_rows']} embargo={report['split']['embargo_rows']} test={report['split']['test_rows']}",
            "weights " + " ".join(f"{name}={value:.6f}" for name, value in learned["weights"].items()),
            f"temperature={learned['temperature']:.6f}",
            f"test calibrated log_loss={test['calibrated_pool']['log_loss']:.6f}",
            f"test calibrated brier={test['calibrated_pool']['multiclass_brier']:.6f}",
            f"test calibrated rps={test['calibrated_pool']['ranked_probability_score']:.6f}",
            f"test calibrated top_label_ece={test['calibrated_pool']['top_label_ece']:.6f}",
            "mode=offline_research_only no_profit_claim"]
    return "\n".join(rows) + "\n"


def prompt(report: Any) -> dict[str, Any]:
    """Export bounded facts for optional local commentary; omit rows and timestamps."""
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise QuantWeaveError("report schema_version must be 1")
    facts = {"bin_labels": report["bins"]["labels"], "models": report["models"],
             "partition_counts": {key: report["split"][key]
                                  for key in ("calibration_rows", "embargo_rows", "test_rows")},
             "weights": report["learned"]["weights"],
             "temperature": report["learned"]["temperature"],
             "test_metrics": report["metrics"]["test"], "claims": report["claims"]}
    return {"facts_sha256": digest(facts), "messages": [
        {"role": "system", "content": "Explain only these calibration and proper-score facts. Do not predict prices, recommend trades, infer profit, or convert model output into a signal. Distinguish calibration from discrimination and mention sample counts."},
        {"role": "user", "content": canonical_bytes(facts).decode().rstrip("\n")}]}
