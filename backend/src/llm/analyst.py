import json
import pandas as pd
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from src.rag.vectorstore import get_vectorstore, build_vectorstore

OLLAMA_BASE = "http://host.docker.internal:11434"
MODEL = "qwen2.5-coder:latest"

SAFE_BUILTINS = {
    "abs": abs, "round": round, "len": len, "sum": sum,
    "min": min, "max": max, "sorted": sorted, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter, "list": list,
    "dict": dict, "set": set, "tuple": tuple, "str": str,
    "int": int, "float": float, "bool": bool,
}

_memories: dict[int, ConversationBufferWindowMemory] = {}


def _get_memory(dataset_id: int) -> ConversationBufferWindowMemory:
    if dataset_id not in _memories:
        _memories[dataset_id] = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    return _memories[dataset_id]


def _try_execute(code: str, df: pd.DataFrame) -> tuple[bool, str]:
    code = "\n".join(l for l in code.splitlines() if not l.startswith("```")).strip()
    try:
        result = eval(code, {"__builtins__": SAFE_BUILTINS}, {"df": df})
        return True, str(result)
    except Exception as e:
        return False, str(e)


def _build_chain(dataset_id: int, df: pd.DataFrame) -> ConversationalRetrievalChain:
    llm = ChatOllama(model=MODEL, base_url=OLLAMA_BASE, temperature=0.2)
    memory = _get_memory(dataset_id)

    try:
        vectorstore = get_vectorstore(dataset_id, df)
    except Exception:
        vectorstore = build_vectorstore(dataset_id, df)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    combine_prompt = PromptTemplate.from_template("""You are an expert AI data analyst. Use the retrieved dataset context and conversation history to answer the user's question thoroughly.

Retrieved context from the dataset:
{context}

Conversation history:
{chat_history}

User question: {question}

Instructions:
- If the question requires exact computation, respond with JSON: {{"mode": "code", "code": "df['col'].mean()"}}
- For all other questions (explanations, insights, patterns, summaries, recommendations), respond with JSON: {{"mode": "analysis", "answer": "your detailed response"}}
- Be specific — reference actual column names, numbers, and patterns from the context
- Explain correlations, distributions, outliers, missing data patterns when relevant
- If asked to explain the dataset, cover: purpose, columns, data types, statistics, notable patterns, data quality
- Always respond with ONLY the JSON object

Answer:""")

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": combine_prompt},
        return_source_documents=False,
        verbose=False
    )
    return chain


def chat(dataset_id: int, question: str, df: pd.DataFrame) -> dict:
    chain = _build_chain(dataset_id, df)

    raw = chain.invoke({"question": question})
    response_text = raw.get("answer", "").strip()

    # Strip markdown fences
    response_text = "\n".join(
        l for l in response_text.splitlines() if not l.startswith("```")
    ).strip()

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        return {"answer": response_text, "mode": "analysis"}

    if parsed.get("mode") == "code":
        code = parsed.get("code", "")
        success, result = _try_execute(code, df)

        if not success:
            # Self-heal via LLM
            llm = ChatOllama(model=MODEL, base_url=OLLAMA_BASE, temperature=0)
            fix = llm.invoke(
                f"Fix this pandas code that failed.\nCode: {code}\nError: {result}\n"
                f"Columns: {list(df.columns)}\nReturn ONLY one line of Python, no markdown."
            )
            fixed_code = fix.content.strip()
            success, result = _try_execute(fixed_code, df)
            code = fixed_code

        answer = result if success else f"Could not compute: {result}"
        return {"answer": answer, "code": code, "mode": "code"}

    return {"answer": parsed.get("answer", response_text), "mode": "analysis"}


def clear_history(dataset_id: int):
    _memories.pop(dataset_id, None)
