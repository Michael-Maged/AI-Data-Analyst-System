import json
from typing import Generator
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from src.rag.vectorstore import get_vectorstore, build_vectorstore
from src.eda.summary import get_comprehensive_summary

OLLAMA_BASE = "http://host.docker.internal:11434"
MODEL = "qwen2.5-coder:latest"

SAFE_BUILTINS = {
    "abs": abs, "round": round, "len": len, "sum": sum,
    "min": min, "max": max, "sorted": sorted, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter, "list": list,
    "dict": dict, "set": set, "tuple": tuple, "str": str,
    "int": int, "float": float, "bool": bool,
}

_memories: dict[int, list] = {}


def _get_memory(dataset_id: int) -> list:
    if dataset_id not in _memories:
        _memories[dataset_id] = []
    return _memories[dataset_id]


def _add_to_memory(dataset_id: int, human_msg: str, ai_msg: str):
    memory = _get_memory(dataset_id)
    memory.append(HumanMessage(content=human_msg))
    memory.append(AIMessage(content=ai_msg))
    if len(memory) > 10:
        memory[:] = memory[-10:]


def _try_execute(code: str, df: pd.DataFrame) -> tuple[bool, str]:
    code = "\n".join(l for l in code.splitlines() if not l.startswith("```")).strip()
    try:
        result = eval(code, {"__builtins__": SAFE_BUILTINS}, {"df": df})
        return True, str(result)
    except Exception as e:
        return False, str(e)


def _build_schema_context(df: pd.DataFrame) -> str:
    """Compile a rich structured dataset description from statistics + sample rows."""
    summary = get_comprehensive_summary(df)
    col_analysis = summary["column_analysis"]
    lines = []

    shape = summary["basic_info"]["shape"]
    lines.append(f"DATASET: {shape['rows']} rows x {shape['columns']} columns")
    lines.append(
        f"Missing data: {summary['basic_info']['missing_percentage']}% | "
        f"Duplicates: {summary['basic_info']['duplicate_rows']}"
    )

    lines.append("\nCOLUMNS:")
    for col, info in col_analysis.items():
        col_type = info.get("type", info["dtype"])
        missing = info["missing_percentage"]
        unique = info["unique_count"]

        if col_type == "numeric":
            lines.append(
                f"  [{col}] numeric | min={info.get('min', '?'):.3g} max={info.get('max', '?'):.3g} "
                f"mean={info.get('mean', '?'):.3g} median={info.get('median', '?'):.3g} "
                f"std={info.get('std', '?'):.3g} | skew={info.get('skewness', 0):.2f} "
                f"outliers={info.get('outliers_count', 0)} missing={missing}%"
            )
        elif col_type == "categorical":
            top = ", ".join(f"{k}({v})" for k, v in list(info.get("top_values", {}).items())[:5])
            lines.append(
                f"  [{col}] categorical | {unique} unique | top: {top} | missing={missing}%"
            )
        elif col_type == "datetime":
            lines.append(
                f"  [{col}] datetime | {info.get('min_date')} to {info.get('max_date')} "
                f"({info.get('date_range_days', '?')} days) | missing={missing}%"
            )
        else:
            lines.append(f"  [{col}] {col_type} | {unique} unique | missing={missing}%")

    high_corrs = summary["correlations"].get("high_correlations", []) if summary["correlations"] else []
    if high_corrs:
        lines.append("\nSTRONG CORRELATIONS (|r| >= 0.7):")
        for c in high_corrs:
            lines.append(f"  {c['var1']} <-> {c['var2']}: r={c['correlation']} ({c['strength']})")

    lines.append("\nSAMPLE ROWS (first 5):")
    lines.append(df.head(5).to_string(index=False))

    return "\n".join(lines)


def _build_prompt(dataset_id: int, question: str, df: pd.DataFrame) -> str:
    memory = _get_memory(dataset_id)

    try:
        vectorstore = get_vectorstore(dataset_id, df)
    except Exception:
        vectorstore = build_vectorstore(dataset_id, df)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    try:
        context_docs = retriever.invoke(question)
    except AttributeError:
        context_docs = retriever.get_relevant_documents(question)
    rag_context = "\n".join([doc.page_content for doc in context_docs])

    chat_history_str = ""
    for i in range(0, len(memory), 2):
        if i + 1 < len(memory):
            chat_history_str += f"Human: {memory[i].content}\nAI: {memory[i+1].content}\n\n"

    schema_context = _build_schema_context(df)

    prompt = f"""You are an expert AI data analyst with deep knowledge of statistics, business intelligence, and data science.

=== DATASET SCHEMA & STATISTICS ===
{schema_context}

=== RELEVANT DATA CONTEXT (from vector search) ===
{rag_context}

=== CONVERSATION HISTORY ===
{chat_history_str}

=== USER QUESTION ===
{question}

=== INSTRUCTIONS ===
You have full knowledge of this dataset's structure, statistics, and sample values above.
Use this to reason about what the dataset is about, what each column represents, and how columns relate to each other.

- Infer the domain/purpose of the dataset from column names and sample values (e.g. sales data, medical records, financial transactions)
- Explain column importance and roles (identifier, target variable, feature, timestamp, category, etc.)
- Reference actual numbers, column names, and patterns — never give generic answers
- For correlations: explain what the relationship means in business/domain terms
- For distributions: explain what skewness or outliers imply about the real-world data
- If the question requires exact computation: respond with JSON {{"mode": "code", "code": "<single pandas expression>"}}
- For all other questions: respond with JSON {{"mode": "analysis", "answer": "<your detailed response>"}}
- Always respond with ONLY the JSON object, no markdown fences

Answer:"""

    return prompt


def chat_stream(dataset_id: int, question: str, df: pd.DataFrame) -> Generator[str, None, None]:
    result = chat(dataset_id, question, df)

    if result["mode"] == "code":
        yield f"__CODE_RESULT__{json.dumps(result)}"
        return

    answer = result["answer"]
    words = answer.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")


def chat(dataset_id: int, question: str, df: pd.DataFrame) -> dict:
    prompt_text = _build_prompt(dataset_id, question, df)
    llm = ChatOllama(model=MODEL, base_url=OLLAMA_BASE, temperature=0.2)
    response_text = llm.invoke(prompt_text).content.strip()

    response_text = "\n".join(
        l for l in response_text.splitlines() if not l.startswith("```")
    ).strip()

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        parsed = {"mode": "analysis", "answer": response_text}

    if parsed.get("mode") == "code":
        code = parsed.get("code", "")
        success, result = _try_execute(code, df)
        if not success:
            llm2 = ChatOllama(model=MODEL, base_url=OLLAMA_BASE, temperature=0)
            fix = llm2.invoke(
                f"Fix this pandas code that failed.\nCode: {code}\nError: {result}\n"
                f"Columns: {list(df.columns)}\nReturn ONLY one line of Python, no markdown."
            )
            code = fix.content.strip()
            success, result = _try_execute(code, df)
        answer = result if success else f"Could not compute: {result}"
        _add_to_memory(dataset_id, question, answer)
        return {"answer": answer, "code": code, "mode": "code"}

    answer = parsed.get("answer", response_text)
    _add_to_memory(dataset_id, question, answer)
    return {"answer": answer, "mode": "analysis"}


def clear_history(dataset_id: int):
    _memories.pop(dataset_id, None)
