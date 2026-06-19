from typing import Optional
import os
import pandas as pd


def load_filtered_dataset(path: str) -> pd.DataFrame:
    """Load the processed filtered complaints CSV.

    Raises FileNotFoundError if the file is missing, or ValueError if empty.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Filtered dataset not found at: {path}")

    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"Loaded dataframe is empty: {path}")

    return df


def ensure_columns(df: pd.DataFrame, required: Optional[list] = None) -> None:
    """Validate required columns exist in dataframe.

    Raises ValueError when columns missing.
    """
    if required is None:
        required = [
            "complaint_id",
            "product_category",
            "cleaned_narrative",
            "original_narrative",
        ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
