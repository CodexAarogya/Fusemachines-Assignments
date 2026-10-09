# Track A — Telco Churn MLOps (uv · MLflow · Evidently · Airflow)

Workflow: `data → training → tracking → registry → serving → monitoring → retraining (if needed)`

```bash
uv sync --frozen                       # a. one-command reproducible env
uv run python src/train.py             # 5 runs -> MLflow, registers best, Staging -> Production
uv run python src/drift.py             # Evidently drift (exit code 2 => recommend retrain)
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db     # compare runs side by side
uv run uvicorn src.serve:app --port 8001                      # serves models:/telco-churn-classifier@production
```
Data: IBM Telco Customer Churn (7,043 rows, 26.5% churn), committed in `data/`.
Exports (committed): `reports/run_comparison.{md,csv}`, `reports/mlflow_runs_export.csv`, `reports/registry_transitions.json`,
`reports/data_drift_report.html`, `reports/target_drift_report.html`, `reports/drift_summary.json`. DAG: `dags/churn_drift_dag.py`.

## a. Environment & Reproducibility (uv)
MLflow 3, Evidently 0.7 (a full API rewrite vs 0.4–0.6, so old tutorials break) and pandas 3 all move fast and interlock; `uv.lock` pins
all 136 transitive packages so the Evidently/MLflow calls here keep working. MLflow also detects the uv project and exports its
requirements into each logged model. **Verified:** fresh `git clone` → `uv sync --frozen` → `uv run python src/train.py` reproduced the
identical metrics table below (random_state fixed).

## b. Experiment Tracking Strategy (MLflow)
Five runs, hyperparameters genuinely different: LogReg `C` 0.1 vs 1.0; RandomForest depth 5/100 trees vs depth 12/300 trees; HistGradientBoosting
(lr 0.05, depth 3). Each logs params, accuracy/precision/recall/F1/ROC-AUC, confusion matrix, ROC curve, and the model. Stratified 80/20 split, threshold 0.5.

| run | accuracy | precision | recall | **F1** | ROC-AUC |
|---|---|---|---|---|---|
| **logreg_C1.0** | 0.8055 | 0.6572 | 0.5588 | **0.6040** | 0.8421 |
| logreg_C0.1 | 0.7999 | 0.6456 | 0.5455 | 0.5913 | 0.8410 |
| hgb_lr0.05 | 0.8055 | 0.6701 | 0.5267 | 0.5898 | 0.8443 |
| rf_d12_n300 | 0.7984 | 0.6480 | 0.5267 | 0.5811 | 0.8355 |
| rf_d5_n100 | 0.7928 | 0.6767 | 0.4198 | 0.5182 | 0.8398 |

**Registered: `logreg_C1.0` (model `telco-churn-classifier` v1).** Rule: highest F1 (churn is 26.5% positive, so accuracy flatters
majority-class models), ROC-AUC as tie-break. logreg_C1.0 has the best F1 (0.604) and recall (0.559) — it finds more real churners —
and ties hgb on accuracy (0.8055). Honest trade-off: hgb has a marginally higher ROC-AUC (0.8443 vs 0.8421) and precision (0.670 vs 0.657),
and rf_d5 has the best precision (0.677) but misses 58% of churners (recall 0.42). The gaps among the top three are small and come from a single
split with no cross-validation, so the choice is defensible rather than statistically conclusive; the simpler, more interpretable linear model
wins on F1 and is cheaper to serve. Registry: version 1 got alias `staging` then `production` (tag `stage`; `reports/registry_transitions.json`).
MLflow deprecated fixed stages in favour of aliases, so the "two stages" are the two aliases.

## c. Monitoring & Drift Strategy (Evidently)
**Reference** = random 70% of the data (training-time view); **current** = the other 30% with injected drift: MonthlyCharges + N(20, 8) per row,
Month-to-month customers oversampled 3×, 8% of churn labels flipped. Monitored: Evidently `DataDriftPreset` (Wasserstein for numeric,
Jensen-Shannon for categorical, threshold 0.1), `DriftedColumnsCount`, `ValueDrift(Churn)` (target drift), plus two **custom metrics**
written against Evidently's metric API (`src/drift.py`): `MeanShift(MonthlyCharges)` and `SegmentChurnShift(Contract=Month-to-month)`.
HTML reports are saved and logged to MLflow (experiment `telco-churn-monitoring`).

| result | value |
|---|---|
| Drifted columns | tenure (0.328), MonthlyCharges (0.652), TotalCharges (0.221), Contract (JS 0.18) → 4/20 = 20% |
| Injected MonthlyCharges shift detected? | **Yes** — custom MeanShift = +19.5 (injected +20) |
| Injected Contract skew detected? | **Yes** |
| Injected label flip detected? | **No** — Churn JS distance 0.0725 < 0.1. Churn rate did move (27.0% → 36.5%) but mostly from the Contract oversampling; the 8% random flips alone are too small/symmetric for this test |
| Month-to-month churn-rate shift (custom) | −0.006 (the flips are random, so within-segment rate barely moves) |

Interpretation: tenure and TotalCharges drifted as *side effects* (Month-to-month customers have short tenure; higher MonthlyCharges inflates
TotalCharges) — correct detection, not false alarms. In production this would mean the model sees a population (more short-contract, higher-priced)
it was not trained on: its calibration and recall on churners would degrade, and **label drift would be invisible to this report** — it only shows
up once true outcomes arrive, so performance monitoring on delayed labels is needed. **Action:** `drift.py` exits 2 when >15% of columns drift
(here 20%) → recommend/trigger `train.py`, which re-logs runs and re-registers; promote to `production` only if F1 holds on a fresh holdout.

## d. Orchestration (Airflow bonus)
`dags/churn_drift_dag.py`: `@weekly`; task 1 runs `drift.py` and stores its exit code; task 2 runs `train.py` only if the code is 2.
**Authored but not executed** (Airflow isn't installed in my environment); the commands it wraps were run by hand.

## Known limitations
Single split, no CV, fixed 0.5 threshold; synthetic drift; serving tested via FastAPI `TestClient`, not a live server; MLflow UI screenshots
replaced by the exported comparison tables above.
