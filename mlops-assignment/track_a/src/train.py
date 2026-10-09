"""Train >=3 models with genuinely different hyperparameters, log everything to MLflow,
pick the winner from the logged metrics, register it, and move it Staging -> Production."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow, mlflow.sklearn, pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CAT, FEATURES, NUM, load

ROOT = Path(__file__).resolve().parent.parent
mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")
EXPERIMENT, MODEL_NAME = "telco-churn-training", "telco-churn-classifier"

def prep(scale):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("num", num, NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])

CONFIGS = {  # hyperparameters differ across runs (not just seeds)
    "logreg_C0.1":   (True,  LogisticRegression(C=0.1, penalty="l2", max_iter=1000)),
    "logreg_C1.0":   (True,  LogisticRegression(C=1.0, penalty="l2", max_iter=1000)),
    "rf_d5_n100":    (False, RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
    "rf_d12_n300":   (False, RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)),
    "hgb_lr0.05":    (False, HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_iter=200, random_state=42)),
}

def main():
    df = load()
    Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df["Churn"], test_size=0.2, stratify=df["Churn"], random_state=42)
    mlflow.set_experiment(EXPERIMENT)
    rows = []
    for name, (scale, est) in CONFIGS.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = Pipeline([("prep", prep(scale)), ("clf", est)]).fit(Xtr, ytr)
            proba = pipe.predict_proba(Xte)[:, 1]; pred = (proba >= 0.5).astype(int)
            m = dict(accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred),
                     recall=recall_score(yte, pred), f1=f1_score(yte, pred), roc_auc=roc_auc_score(yte, proba))
            mlflow.log_params({"model_family": type(est).__name__, **est.get_params()})
            mlflow.log_metrics(m)
            fig, ax = plt.subplots(); ConfusionMatrixDisplay.from_predictions(yte, pred, ax=ax); ax.set_title(name)
            mlflow.log_figure(fig, "confusion_matrix.png"); plt.close(fig)
            fig, ax = plt.subplots(); RocCurveDisplay.from_predictions(yte, proba, ax=ax, name=name); mlflow.log_figure(fig, "roc_curve.png"); plt.close(fig)
            info = mlflow.sklearn.log_model(pipe, name="model", signature=infer_signature(Xte, pred), input_example=Xte.head(3), serialization_format="cloudpickle")
            rows.append(dict(run_name=name, run_id=run.info.run_id, model_uri=info.model_uri, **m))
    res = pd.DataFrame(rows).sort_values("f1", ascending=False)
    res.round(4).to_csv(ROOT / "reports" / "run_comparison.csv", index=False)
    (ROOT / "reports" / "run_comparison.md").write_text(res.drop(columns=["run_id", "model_uri"]).round(4).to_markdown(index=False))
    print(res.drop(columns=["run_id", "model_uri"]).round(4).to_string(index=False))

    # Selection rule: highest F1 (imbalanced target), ROC-AUC as tie-breaker.
    best = res.sort_values(["f1", "roc_auc"], ascending=False).iloc[0]
    print("\nBEST:", best.run_name)
    c = MlflowClient()
    mv = mlflow.register_model(best.model_uri, MODEL_NAME)
    log = []
    for alias in ["staging", "production"]:  # MLflow>=2.9 replaces fixed stages with aliases
        c.set_registered_model_alias(MODEL_NAME, alias, mv.version)
        c.set_model_version_tag(MODEL_NAME, mv.version, "stage", alias.capitalize())
        log.append({"version": mv.version, "to": alias.capitalize(), "source_run": best.run_name})
    c.update_model_version(MODEL_NAME, mv.version, description=f"Selected by F1={best.f1:.4f}, ROC-AUC={best.roc_auc:.4f} among {len(res)} runs")
    (ROOT / "reports" / "registry_transitions.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))

if __name__ == "__main__":
    main()
