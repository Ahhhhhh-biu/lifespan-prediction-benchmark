# Lifespan Prediction Benchmark

A reproducible regression workflow for evaluating pathology features as predictors of lifespan across biological conditions.

## Methods

- Mean-prediction baseline, Ridge regression and random forests.
- Group-disjoint nested cross-validation with inner-loop hyperparameter selection by MAE.
- Training-fold imputation and scaling through scikit-learn pipelines.
- Out-of-fold predictions, fold metrics and group bootstrap intervals.
- Exploratory held-out permutation importance and split/provenance manifests.

## Installation and use

Tested with Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python smoke_test.py
python benchmark.py --data data/analysis.csv --config benchmark.example.json --out results/run_01
```

The output directory must be new. The sample configuration defines a proposed CSV schema: a unique sample ID, condition group, numeric lifespan outcome and explicitly selected numeric features. Adapt the names to your audited data. The workflow does not automatically download or preprocess a publication dataset.

## Evaluation

Define groups so dependent observations, repeated measurements and shared outcomes cannot cross training and test folds. Audit feature timing, endpoint definition, missingness and aggregation before running. Censored individual lifetimes require an appropriate survival-analysis design rather than unqualified ordinary regression.

MAE and R² are reported alongside the baseline. Cross-validation performance does not substitute for an independent external test. Group bootstrap intervals describe group variation; they do not capture all model-selection uncertainty. Permutation importance is exploratory and noncausal, especially with correlated features and dependent observations.

## Outputs and checks

The workflow exports predictions, metrics, split manifests, configuration and package versions. `smoke_test.py` checks repeatability, disjoint groups, exactly one prediction per row per model and explicit target-feature rejection on synthetic regression data. Its recorded results are in `software_checks.json`.

At least six groups are required by this example protocol; this guard is not a sample-size justification. Report all predefined models and retain failure cases.

## References

- [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [Related lifespan prediction study](https://www.nature.com/articles/s42003-025-09012-9)

No private research data or pretrained model weights are included. Dependencies and referenced datasets retain their upstream licences.
