# QuantWeaveLM

Leakage-resistant calibration and evaluation for **ordered crypto return probability forecasts**. QuantWeaveLM combines probability vectors from multiple models, learns mixture weights on a declared calibration prefix, applies multiclass temperature scaling, leaves an explicit embargo, and reports untouched test metrics. It never downloads market data, connects an exchange, places an order, or claims profit.

```console
$ ./doit.sh
...
QuantWeaveLM deterministic calibration report
rows calibration=20 embargo=2 test=10
weights momentum=0.202672 reversion=0.797328
temperature=1.465078
test calibrated log_loss=1.115525
test calibrated brier=0.680683
test calibrated rps=0.433976
test calibrated top_label_ece=0.224037
mode=offline_research_only no_profit_claim
```

That single command is offline, deterministic, and requires only Python 3.10+. It compiles the package, runs the full test suite, and executes the synthetic demonstration.

## The problem it handles

Searches such as **"probabilities do not sum to 1"**, **"crypto forecast calibration time series leakage"**, **"multiclass Brier score ordered returns"**, and **"temperature scaling walk forward probabilities"** often lead to generic classifier calibration or full trading frameworks.

[scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html) provides general classifier calibrators and correctly stresses that calibration data should be independent from estimator training data. QuantWeaveLM does not replace that library. Its narrower distinction is a dependency-free, model-agnostic audit pipeline for already-produced ordered return distributions:

- all records must be strictly chronological;
- calibration, embargo, and test row counts are explicit and exhaustive;
- a minimum timestamp embargo is enforced;
- ensemble weights and temperature are learned only from calibration rows;
- tests prove that changing embargo forecasts or test targets cannot alter learned parameters;
- output includes calibration-only climatology, log loss, multiclass Brier score, ranked probability score, top-label ECE, and classwise reliability tables;
- `verify` recomputes the entire report from the declared configuration and JSONL evidence.

Proper scores matter because they evaluate the full predictive distribution rather than rewarding a selective point estimate. The implementation follows the role of proper scoring rules described by [Gneiting and Raftery](https://doi.org/10.1198/016214506000001437), while reporting several scores because imperfect forecast systems can rank differently under different proper rules.

## Input contract

The configuration declares ordered return bins, models, chronological partitions, and bounded optimization settings:

```json
{
  "schema_version": 1,
  "bins": {
    "edges": [-0.01, 0.01],
    "labels": ["down", "flat", "up"]
  },
  "models": ["model_a", "model_b"],
  "split": {
    "calibration_rows": 100,
    "embargo_rows": 5,
    "test_rows": 50,
    "minimum_embargo_seconds": 3600
  },
  "optimizer": {
    "iterations": 500,
    "learning_rate": 0.05,
    "temperature_min": 0.5,
    "temperature_max": 3.0,
    "temperature_steps": 151
  }
}
```

Each JSONL record contains a canonical UTC timestamp, the realized future return, and one probability vector per declared model:

```json
{"timestamp":"2026-01-01T00:00:00Z","target_return":-0.013,"forecasts":{"model_a":[0.6,0.3,0.1],"model_b":[0.4,0.4,0.2]}}
```

The dataset must contain exactly `calibration_rows + embargo_rows + test_rows` records. Probability vectors must match the ordered labels, contain only finite values in `[0,1]`, and sum to one. Unknown fields fail closed.

## Run and verify

```bash
./run.sh run config.json forecasts.jsonl report.json
./run.sh verify config.json forecasts.jsonl report.json
./run.sh summary report.json
./run.sh prompt report.json local-commentary-prompt.json
```

`prompt` exports bounded aggregate facts suitable for an optional local LLM. It omits observations and timestamps and explicitly prohibits price prediction, trade recommendations, signals, and profit inference. QuantWeaveLM itself never calls a model server.

## What the metrics do and do not say

- Lower log loss, Brier score, RPS, and ECE are better on the evaluated sample.
- RPS uses the order of return bins; confusing adjacent bins costs less than confusing distant bins.
- Brier and log loss measure more than calibration alone. Reliability tables are included because one headline score cannot isolate every forecast property.
- A better historical probability score does not imply a profitable or executable strategy.
- The tool evaluates supplied forecasts; it does not validate feature construction, model training, labels, data vendor accuracy, survivorship, or whether the source model already saw the test period.

## Limitations and safety

- Offline research only: no API credentials, wallet connection, exchange connection, live prices, order placement, signing, custody, approvals, or trading.
- No return, alpha, investment, or performance guarantee.
- Linear pooling and one scalar temperature cannot correct every form of misspecification or regime drift.
- ECE depends on binning and can hide local errors; complete reliability cells remain in JSON.
- Small calibration/test partitions have high uncertainty. The minimum row counts prevent trivial runs, not statistical adequacy.
- An embargo separates declared rows but cannot detect leakage inside forecasts supplied by another system.
- Inputs are raw return probabilities without fees, spread, slippage, funding, taxes, or execution modeling.

## Support and funded direction

If this saves research time, [support continued production](SUPPORT.md). A confirmed public transaction hash may accompany a feature-direction issue. A donation can request direction but cannot purchase ownership, returns, priority, a deadline, acceptance, or prohibited work.

## License

Apache-2.0. See [LICENSE](LICENSE).
