"""Evidently drift monitoring: 70% reference vs 30% current with deliberately injected drift.
Logs the HTML reports + drift metrics to MLflow. Exit code 2 => drift crossed the retrain threshold."""
import json, sys
from pathlib import Path
import mlflow, numpy as np, pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.core.metric_types import SingleValue, SingleValueCalculation, SingleValueMetric
from evidently.metrics import DriftedColumnsCount, ValueDrift
from evidently.presets import DataDriftPreset
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CAT, NUM, load

ROOT = Path(__file__).resolve().parent.parent
mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")
DRIFT_SHARE_THRESHOLD = 0.15  # >15% of columns drifted => recommend retraining


# ---------- custom Evidently metrics (beyond built-ins) ----------
class MeanShift(SingleValueMetric):
    """current mean(column) - reference mean(column)."""
    column: str


class MeanShiftCalc(SingleValueCalculation[MeanShift]):
    def calculate(self, context, current_data, reference_data):
        cur = current_data.column(self.metric.column).data.mean()
        ref = reference_data.column(self.metric.column).data.mean()
        return self.result(float(cur - ref)), self.result(0.0)

    def display_name(self):
        return f"Mean shift of {self.metric.column} (current - reference)"


class SegmentChurnShift(SingleValueMetric):
    """Churn-rate difference inside one segment, e.g. Contract == Month-to-month."""
    segment_column: str
    segment_value: str
    target: str = "Churn"


class SegmentChurnShiftCalc(SingleValueCalculation[SegmentChurnShift]):
    def _rate(self, ds):
        m = self.metric
        seg = ds.as_dataframe()
        seg = seg[seg[m.segment_column] == m.segment_value]
        return float(seg[m.target].astype(float).mean())

    def calculate(self, context, current_data, reference_data):
        return self.result(self._rate(current_data) - self._rate(reference_data)), self.result(0.0)

    def display_name(self):
        m = self.metric
        return f"Churn-rate shift in {m.segment_column}={m.segment_value}"


# ---------- split + synthetic drift ----------
def make_sets(seed=42):
    df = load()
    ref = df.sample(frac=0.7, random_state=seed)
    cur = df.drop(ref.index).reset_index(drop=True); ref = ref.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    # 1) numeric drift: MonthlyCharges shifted by a random per-row offset (mean +20)
    cur["MonthlyCharges"] = (cur["MonthlyCharges"] + rng.normal(20, 8, len(cur))).clip(lower=18)
    # 2) categorical skew: oversample Month-to-month customers
    w = np.where(cur["Contract"] == "Month-to-month", 3.0, 1.0)
    cur = cur.sample(n=len(cur), replace=True, weights=w, random_state=seed).reset_index(drop=True)
    # 3) label drift: flip 8% of churn labels
    flip = rng.random(len(cur)) < 0.08
    cur.loc[flip, "Churn"] = 1 - cur.loc[flip, "Churn"]
    return ref, cur, ["MonthlyCharges", "Contract", "Churn"]


def main():
    ref, cur, perturbed = make_sets()
    dd = DataDefinition(numerical_columns=NUM, categorical_columns=CAT + ["Churn"])
    r, c = Dataset.from_pandas(ref, data_definition=dd), Dataset.from_pandas(cur, data_definition=dd)

    data_report = Report([DataDriftPreset(), DriftedColumnsCount(),
                          MeanShift(column="MonthlyCharges"),
                          SegmentChurnShift(segment_column="Contract", segment_value="Month-to-month")])
    snap = data_report.run(c, r)
    target_report = Report([ValueDrift(column="Churn")])
    tsnap = target_report.run(c, r)

    out = ROOT / "reports"; out.mkdir(exist_ok=True)
    snap.save_html(str(out / "data_drift_report.html")); tsnap.save_html(str(out / "target_drift_report.html"))
    (out / "data_drift_report.json").write_text(snap.json())
    def parse(rep):
        rows, cnt, custom = [], None, {}
        for m in json.loads(rep.json())["metrics"]:
            cfg, val, name = m["config"], m["value"], m["metric_name"]
            t = cfg["type"].split(":")[-1]
            if t == "ValueDrift":
                p_based = "p_value" in cfg["method"].lower() or "p-value" in cfg["method"].lower()
                rows.append(dict(column=cfg["column"], method=cfg["method"], threshold=cfg["threshold"], score=round(val, 4),
                                 drifted=bool(val < cfg["threshold"] if p_based else val > cfg["threshold"]),
                                 injected=cfg["column"] in perturbed))
            elif t == "DriftedColumnsCount":
                cnt = val
            elif t in ("MeanShift", "SegmentChurnShift"):
                custom[t] = round(val, 4)
        return rows, cnt, custom
    rows, cnt, custom = parse(snap)
    trows, _, _ = parse(tsnap)
    drifted = [r["column"] for r in rows if r["drifted"]]
    share = cnt["share"] if cnt else None
    summary = dict(perturbed_columns=perturbed, drifted_columns=drifted, drifted_share=share,
                   per_column=rows, target_drift=trows, custom_metrics=custom,
                   injected_detected={c: (c in drifted or any(t["drifted"] for t in trows if t["column"] == c)) for c in perturbed})
    (out / "drift_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))

    mlflow.set_experiment("telco-churn-monitoring")
    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_artifact(str(out / "data_drift_report.html")); mlflow.log_artifact(str(out / "target_drift_report.html"))
        mlflow.log_artifact(str(out / "drift_summary.json"))
        mlflow.log_param("injected", "MonthlyCharges+N(20,8); Month-to-month x3 oversample; 8% label flip")
        for k, v in custom.items(): mlflow.log_metric(k, v)
        if share is not None: mlflow.log_metric("drifted_columns_share", share)
    if share is not None and share > DRIFT_SHARE_THRESHOLD:
        print("RECOMMENDATION: drift share above threshold -> trigger retraining (src/train.py)"); sys.exit(2)


if __name__ == "__main__":
    main()
