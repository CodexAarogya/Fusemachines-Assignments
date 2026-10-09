"""Data loading / cleaning shared by training, drift and serving."""
from pathlib import Path
import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "data" / "Telco-Customer-Churn.csv"
NUM = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CAT = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
       "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
       "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
FEATURES = NUM + CAT


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df.drop(columns=["customerID"])
