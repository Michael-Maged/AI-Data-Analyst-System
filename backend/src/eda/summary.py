import pandas as pd


def get_dataset_summary(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")

    summary = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": {
            col: {
                "dtype": str(df[col].dtype),
                "nulls": int(df[col].isnull().sum()),
                "unique": int(df[col].nunique()),
            }
            for col in df.columns
        },
        "numeric_stats": numeric.describe().round(3).to_dict() if not numeric.empty else {},
        "correlations": numeric.corr().round(3).to_dict() if numeric.shape[1] > 1 else {},
        "top_categoricals": {
            col: df[col].value_counts().head(5).to_dict()
            for col in categorical.columns
        },
        "sample_rows": df.head(5).to_dict(orient="records"),
    }

    return summary
