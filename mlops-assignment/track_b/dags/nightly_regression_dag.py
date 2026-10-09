"""Airflow DAG (bonus; authored, not executed in the authoring sandbox).
Nightly: run the W16 harness regression set against the production prompt; exit code 2 => degradation beyond threshold.
The `alert` task only runs on failure (trigger_rule=one_failed) -- replace the echo with Slack/email in a real deployment."""
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/track_b"
with DAG("assistant_nightly_regression", schedule="0 2 * * *", start_date=datetime(2026, 1, 1), catchup=False, tags=["mlops"]) as dag:
    check = BashOperator(task_id="regression_eval", bash_command=f"cd {PROJECT} && uv run python -m mlops.nightly_regression")
    alert = BashOperator(task_id="alert", trigger_rule="one_failed", bash_command='echo "REGRESSION: assistant pass-rate degraded; block promotion and review traces"')
    check >> alert
