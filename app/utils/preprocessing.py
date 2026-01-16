import pandas as pd

FEATURES = [
    "age", "sex", "systolic_bp", "cholesterol", "bmi",
    "smoking", "diabetes", "resting_hr", "physical_activity", "family_history"
]

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures correct column order and basic type safety.
    Your pipeline likely handles scaling/one-hot/etc internally.
    """
    df = df.copy()
    
    # Basic validation
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Reorder and select only needed features
    df = df[FEATURES]

    # Convert to correct dtypes (helps catch bad input early)
    type_mapping = {
        "age": float, "sex": int, "systolic_bp": float, "cholesterol": float,
        "bmi": float, "smoking": int, "diabetes": int, "resting_hr": float,
        "physical_activity": int, "family_history": int
    }
    for col, dtype in type_mapping.items():
        df[col] = df[col].astype(dtype)

    return df