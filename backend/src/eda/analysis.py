import pandas as pd
import numpy as np
from scipy import stats


def analyze(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(include="object")

    return {
        "shape": {"rows": df.shape[0], "cols": df.shape[1]},
        "missing": _missing(df),
        "correlations": _correlations(numeric),
        "distributions": _distributions(numeric),
        "categorical_summary": _categorical(categorical),
        "outliers": _outliers(numeric),
        "cat_num_relationships": _cat_num_relationships(df, categorical, numeric),
    }


def _missing(df):
    missing = df.isnull().sum()
    return {
        col: {"count": int(missing[col]), "pct": round(missing[col] / len(df) * 100, 1)}
        for col in df.columns if missing[col] > 0
    }


def _correlations(numeric):
    if numeric.shape[1] < 2:
        return {"matrix": {}, "strong_pairs": []}
    matrix = numeric.corr().round(3)
    strong = []
    cols = matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = matrix.iloc[i, j]
            if abs(val) >= 0.5:
                strong.append({
                    "col1": cols[i], "col2": cols[j],
                    "r": round(val, 3),
                    "strength": "strong" if abs(val) >= 0.8 else "moderate",
                    "direction": "positive" if val > 0 else "negative"
                })
    strong.sort(key=lambda x: abs(x["r"]), reverse=True)
    return {"matrix": matrix.to_dict(), "strong_pairs": strong}


def _distributions(numeric):
    result = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) == 0:
            continue
        skew = float(stats.skew(s))
        result[col] = {
            "mean": round(float(s.mean()), 3),
            "median": round(float(s.median()), 3),
            "std": round(float(s.std()), 3),
            "skew": round(skew, 3),
            "skew_label": "right-skewed" if skew > 1 else "left-skewed" if skew < -1 else "normal",
            "min": round(float(s.min()), 3),
            "max": round(float(s.max()), 3),
        }
    return result


def _categorical(categorical):
    result = {}
    for col in categorical.columns:
        vc = categorical[col].value_counts()
        result[col] = {
            "unique": int(categorical[col].nunique()),
            "top": vc.index[0] if len(vc) else None,
            "top_pct": round(vc.iloc[0] / len(categorical) * 100, 1) if len(vc) else 0,
            "top_5": vc.head(5).to_dict(),
        }
    return result


def _outliers(numeric):
    result = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        if n > 0:
            result[col] = {"count": n, "pct": round(n / len(s) * 100, 1)}
    return result


def _cat_num_relationships(df, categorical, numeric):
    """ANOVA-based: does a categorical column significantly affect a numeric one?"""
    result = []
    for cat_col in categorical.columns[:5]:  # limit to avoid overload
        for num_col in numeric.columns[:5]:
            groups = [g[num_col].dropna().values for _, g in df.groupby(cat_col) if len(g) >= 5]
            if len(groups) < 2:
                continue
            try:
                f, p = stats.f_oneway(*groups)
                if p < 0.05:
                    result.append({
                        "categorical": cat_col,
                        "numeric": num_col,
                        "p_value": round(float(p), 4),
                        "significant": True
                    })
            except Exception:
                continue
    return result
