"""Airflow DAG (bonus; authored but NOT executed in the sandbox).
Weekly: run src/drift.py. It exits 0 (no drift) or 2 (drifted-column share > threshold).
On exit code 2 the DAG retrains (src/train.py), which re-logs runs and re-registers the best model."""
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/track_a"  # mount this repo folder here

with DAG("churn_drift_check", schedule="@weekly", start_date=datetime(2026, 1, 1),
         catchup=False, tags=["mlops"]) as dag:
    # `|| echo $?` captures the exit code instead of failing the task; "2" means drift detected.
    drift = BashOperator(task_id="drift_check",
                         bash_command=f"cd {PROJECT} && uv run python src/drift.py; echo $? > /tmp/drift_rc; true")
    retrain = BashOperator(task_id="retrain_if_drift",
                           bash_command=f"if [ \"$(cat /tmp/drift_rc)\" = \"2\" ]; then cd {PROJECT} && uv run python src/train.py; else echo 'no significant drift'; fi")
    drift >> retrain
