import io
import base64
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _encode(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_plots(df: pd.DataFrame, analysis: dict) -> dict:
    plots = {}

    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(include="object")

    # 1. Correlation heatmap (only if enough numeric cols)
    if numeric.shape[1] >= 2:
        plots["correlation_heatmap"] = _correlation_heatmap(numeric)

    # 2. Scatter plots for strongly correlated pairs
    strong_pairs = analysis.get("correlations", {}).get("strong_pairs", [])
    for pair in strong_pairs[:3]:
        key = f"scatter_{pair['col1']}_vs_{pair['col2']}"
        plots[key] = _scatter(df, pair["col1"], pair["col2"], pair["r"])

    # 3. Distribution of skewed columns
    for col, info in analysis.get("distributions", {}).items():
        if info["skew_label"] != "normal":
            plots[f"dist_{col}"] = _distribution(df[col], col, info["skew_label"])

    # 4. Top categorical breakdowns
    for col, info in list(analysis.get("categorical_summary", {}).items())[:3]:
        if info["unique"] <= 20:
            plots[f"bar_{col}"] = _bar(df[col], col)

    # 5. Cat vs Num relationships (boxplots)
    for rel in analysis.get("cat_num_relationships", [])[:3]:
        key = f"box_{rel['categorical']}_vs_{rel['numeric']}"
        plots[key] = _boxplot(df, rel["categorical"], rel["numeric"])

    return plots


def _correlation_heatmap(numeric):
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.8), max(5, len(corr) * 0.7)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    return _encode(fig)


def _scatter(df, col1, col2, r):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[col1], df[col2], alpha=0.5, s=15)
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(f"{col1} vs {col2}  (r = {r})")
    return _encode(fig)


def _distribution(series, col, skew_label):
    fig, ax = plt.subplots(figsize=(6, 4))
    series.dropna().hist(bins=30, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"{col} — {skew_label}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    return _encode(fig)


def _bar(series, col):
    vc = series.value_counts().head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    vc.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"{col} — Top Values")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    return _encode(fig)


def _boxplot(df, cat_col, num_col):
    top_cats = df[cat_col].value_counts().head(8).index
    subset = df[df[cat_col].isin(top_cats)]
    fig, ax = plt.subplots(figsize=(7, 4))
    subset.boxplot(column=num_col, by=cat_col, ax=ax)
    ax.set_title(f"{num_col} by {cat_col}")
    plt.suptitle("")
    plt.xticks(rotation=45, ha="right")
    return _encode(fig)
