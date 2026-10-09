# W17 MLOps — two tracks in one repo
| Track | Folder | What |
|---|---|---|
| A — Data Science | [`track_a/`](track_a/README.md) | Telco churn: uv, MLflow (5 runs + registry), Evidently drift, FastAPI serving, Airflow DAG |
| B — Agentic AI | [`track_b/`](track_b/README.md) | W15 assistant + W16 agent: uv, MLflow prompt versions + traces, Evidently regression suite, Airflow DAG |

Each track has its own `pyproject.toml` + `uv.lock`; `cd track_x && uv sync --frozen`. Split into two repos/branches by copying a folder if the course requires it.
