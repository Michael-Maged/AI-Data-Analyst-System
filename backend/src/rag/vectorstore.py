import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.eda.summary import get_dataset_summary

EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR = "data/chroma"  # mounted from host: /data/chroma

_stores: dict[int, Chroma] = {}
_indexed: set[int] = set()


def is_indexed(dataset_id: int) -> bool:
    return dataset_id in _indexed


def _build_documents(df: pd.DataFrame, dataset_id: int) -> list[Document]:
    docs = []
    summary = get_dataset_summary(df)

    # 1. Overall dataset summary document
    schema_text = "\n".join(
        f"- {col}: {info['dtype']}, {info.get('unique_count', info.get('unique', '?'))} unique values, {info.get('missing_count', info.get('nulls', '?'))} nulls"
        for col, info in summary["columns"].items()
    )
    docs.append(Document(
        page_content=f"Dataset overview: {summary['shape']['rows']} rows and {summary['shape']['columns']} columns.\nColumns:\n{schema_text}",
        metadata={"type": "summary", "dataset_id": dataset_id}
    ))

    # 2. Numeric stats document
    if summary["numeric_stats"]:
        docs.append(Document(
            page_content=f"Numeric statistics:\n{pd.DataFrame(summary['numeric_stats']).to_string()}",
            metadata={"type": "stats", "dataset_id": dataset_id}
        ))

    # 3. Correlation document
    if summary["correlations"]:
        corr_df = pd.DataFrame(summary["correlations"])
        docs.append(Document(
            page_content=f"Correlation matrix between numeric columns:\n{corr_df.to_string()}",
            metadata={"type": "correlations", "dataset_id": dataset_id}
        ))

    # 4. Per-column documents
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = int(df[col].isnull().sum())
        unique = int(df[col].nunique())

        if df[col].dtype == "object" or 'datetime' in dtype.lower():
            top = df[col].value_counts().head(10).to_dict()
            content = (
                f"Column '{col}' is categorical/datetime with {unique} unique values and {nulls} missing values.\n"
                f"Top values: {top}"
            )
        else:
            stats = df[col].describe().to_dict()
            # Handle numeric stats safely
            mean_val = stats.get('mean', 0)
            std_val = stats.get('std', 0)
            min_val = stats.get('min', 0)
            max_val = stats.get('max', 0)
            
            # Convert to float if possible, otherwise use string representation
            try:
                mean_str = str(round(float(mean_val), 3)) if mean_val is not None else "N/A"
                std_str = str(round(float(std_val), 3)) if std_val is not None else "N/A"
                min_str = str(min_val) if min_val is not None else "N/A"
                max_str = str(max_val) if max_val is not None else "N/A"
            except (TypeError, ValueError):
                mean_str = str(mean_val) if mean_val is not None else "N/A"
                std_str = str(std_val) if std_val is not None else "N/A"
                min_str = str(min_val) if min_val is not None else "N/A"
                max_str = str(max_val) if max_val is not None else "N/A"
            
            content = (
                f"Column '{col}' is numeric ({dtype}) with {nulls} missing values.\n"
                f"Stats: min={min_str}, max={max_str}, mean={mean_str}, std={std_str}"
            )
        docs.append(Document(
            page_content=content,
            metadata={"type": "column", "column": col, "dataset_id": dataset_id}
        ))

    # 5. Row-level documents (chunked every 10 rows, truncated to 1000 chars)
    chunk_size = 10
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        content = f"Data rows {i} to {i + len(chunk) - 1}:\n{chunk.to_string(index=False)}"
        docs.append(Document(
            page_content=content[:1500],
            metadata={"type": "rows", "start_row": i, "dataset_id": dataset_id}
        ))

    return docs


def build_vectorstore(dataset_id: int, df: pd.DataFrame) -> Chroma:
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url="http://host.docker.internal:11434"
    )
    docs = _build_documents(df, dataset_id)
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=f"{CHROMA_DIR}/{dataset_id}",
        collection_name=f"dataset_{dataset_id}"
    )
    _stores[dataset_id] = store
    _indexed.add(dataset_id)
    return store


def get_vectorstore(dataset_id: int, df: pd.DataFrame = None) -> Chroma:
    if dataset_id in _stores:
        return _stores[dataset_id]

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url="http://host.docker.internal:11434"
    )
    try:
        store = Chroma(
            persist_directory=f"{CHROMA_DIR}/{dataset_id}",
            embedding_function=embeddings,
            collection_name=f"dataset_{dataset_id}"
        )
        _stores[dataset_id] = store
        return store
    except Exception:
        if df is not None:
            return build_vectorstore(dataset_id, df)
        raise
