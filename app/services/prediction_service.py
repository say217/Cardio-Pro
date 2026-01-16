import numpy as np
import pandas as pd
from joblib import load
from pathlib import Path

from ..utils.preprocessing import preprocess_features


class PredictionService:
    def __init__(self):
        model_path = Path(__file__).resolve().parents[1] / "models" / "heart_risk_multiclass_pipeline.joblib"
        if not model_path.exists():
            raise RuntimeError(f"Model file not found at {model_path}")

        self.model = load(model_path)

    def predict(self, df: pd.DataFrame) -> dict:
        df_processed = preprocess_features(df)

        try:
            proba = self.model.predict_proba(df_processed)[0]
            pred_class = int(np.argmax(proba))
        except Exception:
            pred = self.model.predict(df_processed)[0]
            try:
                pred_class = int(pred)
            except Exception:
                pred_class = int(np.asarray(pred).item())

            proba = np.zeros(4, dtype=float)
            if 0 <= pred_class < proba.shape[0]:
                proba[pred_class] = 1.0

        pred_proba = np.array(proba) * 100

        risk_map = {
            0: "Low",
            1: "Medium",
            2: "High",
            3: "Very High",
        }

        probs = {
            "Low": round(float(pred_proba[0]) if pred_proba.shape[0] > 0 else 0.0, 2),
            "Medium": round(float(pred_proba[1]) if pred_proba.shape[0] > 1 else 0.0, 2),
            "High": round(float(pred_proba[2]) if pred_proba.shape[0] > 2 else 0.0, 2),
            "Very High": round(float(pred_proba[3]) if pred_proba.shape[0] > 3 else 0.0, 2),
        }

        return {"risk_level": risk_map.get(pred_class, "Unknown"), "probabilities": probs}
