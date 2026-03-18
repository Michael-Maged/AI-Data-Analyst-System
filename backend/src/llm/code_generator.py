import requests

OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
MODEL = "qwen2.5-coder:latest"

SAFE_BUILTINS = {
    "abs": abs, "round": round, "len": len, "sum": sum,
    "min": min, "max": max, "sorted": sorted, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter, "list": list,
    "dict": dict, "set": set, "tuple": tuple, "str": str,
    "int": int, "float": float, "bool": bool, "print": print,
}


def generate_code(question: str, columns: list, sample_rows: list) -> str:
    schema = "\n".join(f"  - {name}: {dtype}" for name, dtype in columns)
    prompt = f"""You are an expert data analyst. A user asked a question about a dataset.
Your job is to reason about the question and the dataset schema, then produce the correct pandas code.

Dataset schema:
{schema}

Sample rows:
{sample_rows}

Think step by step:
1. Understand what the user is asking (intent, not literal wording)
2. Identify the correct column(s) based on name and dtype — never use positional indexing like iloc unless explicitly asked
3. If the question is ambiguous or the operation doesn't make sense for the column type (e.g. mean of a string column), produce code that returns a helpful message string instead
4. Write the pandas code

Rules:
- Use dataframe named df
- Return ONLY one line of Python code
- No imports, no explanations, no markdown
- Prefer column names over positions

Question: {question}"""

    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def _strip_fences(code: str) -> str:
    if code.startswith("```"):
        return "\n".join(
            line for line in code.splitlines()
            if not line.startswith("```")
        ).strip()
    return code


def _eval(code: str, df):
    return eval(code, {"__builtins__": SAFE_BUILTINS}, {"df": df})


def execute_code(code: str, df, question: str, columns: list, sample_rows: list, retries: int = 2) -> dict:
    code = _strip_fences(code)

    for attempt in range(retries + 1):
        try:
            result = _eval(code, df)
            return {"result": str(result), "code": code}
        except Exception as e:
            if attempt == retries:
                return {"error": f"Could not compute answer: {type(e).__name__}: {e}", "code": code}

            # Self-healing: ask LLM to fix the code given the error
            fix_prompt = f"""The following pandas code failed with error: {type(e).__name__}: {e}

Failed code: {code}

Dataset columns and types:
{columns}

Sample rows:
{sample_rows}

Original question: {question}

Fix the code. Return ONLY one corrected line of Python code, no explanations, no imports, no markdown."""

            response = requests.post(
                OLLAMA_URL,
                json={"model": MODEL, "prompt": fix_prompt, "stream": False}
            )
            code = _strip_fences(response.json()["response"].strip())
