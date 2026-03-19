import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


def clean(df: pd.DataFrame, dataset_id: int) -> pd.DataFrame:
    df = df.copy()

    # Drop columns with >50% missing
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)

    # Infer and fix types
    df = df.infer_objects()
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if not converted.isna().all():  # If conversion produced some valid numbers
            df[col] = converted

    # Fill missing values
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")

    # Normalize numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        df[numeric_cols] = MinMaxScaler().fit_transform(df[numeric_cols])

    # Save cleaned file
    df.to_csv(f"{PROCESSED_DIR}/{dataset_id}.csv", index=False)

    return df
