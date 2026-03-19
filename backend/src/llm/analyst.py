import json
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
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

_memories: dict[int, list] = {}


def _get_memory(dataset_id: int) -> list:
    if dataset_id not in _memories:
        _memories[dataset_id] = []
    return _memories[dataset_id]


def _add_to_memory(dataset_id: int, human_msg: str, ai_msg: str):
    memory = _get_memory(dataset_id)
    memory.append(HumanMessage(content=human_msg))
    memory.append(AIMessage(content=ai_msg))
    # Keep only last 10 messages (5 exchanges)
    if len(memory) > 10:
        memory[:] = memory[-10:]


def _try_execute(code: str, df: pd.DataFrame) -> tuple[bool, str]:
    code = "\n".join(l for l in code.splitlines() if not l.startswith("```")).strip()
    try:
        result = eval(code, {"__builtins__": SAFE_BUILTINS}, {"df": df})
        return True, str(result)
    except Exception as e:
        return False, str(e)


def _build_chain(dataset_id: int, df: pd.DataFrame):
    llm = ChatOllama(model=MODEL, base_url=OLLAMA_BASE, temperature=0.2)
    memory = _get_memory(dataset_id)

    try:
        vectorstore = get_vectorstore(dataset_id, df)
    except Exception:
        vectorstore = build_vectorstore(dataset_id, df)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # Format chat history for prompt
    chat_history_str = ""
    for i in range(0, len(memory), 2):
        if i + 1 < len(memory):
            human_msg = memory[i].content
            ai_msg = memory[i + 1].content
            chat_history_str += f"Human: {human_msg}\nAI: {ai_msg}\n\n"

    prompt_template = """You are an expert AI data analyst. Use the retrieved dataset context and conversation history to answer the user's question thoroughly.

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

Answer:"""

    # Create a simple chain without memory dependency
    def invoke_chain(inputs):
        try:
            context_docs = retriever.invoke(inputs["question"])
        except AttributeError:
            # Fallback for older versions
            context_docs = retriever.get_relevant_documents(inputs["question"])
        
        context = "\n".join([doc.page_content for doc in context_docs])
        
        prompt_text = prompt_template.format(
            context=context,
            chat_history=chat_history_str,
            question=inputs["question"]
        )
        
        response = llm.invoke(prompt_text)
        return {"answer": response.content}
    
    # Return a mock chain object
    class MockChain:
        def invoke(self, inputs):
            return invoke_chain(inputs)
    
    return MockChain()


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
        parsed = {"mode": "analysis", "answer": response_text}

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
        _add_to_memory(dataset_id, question, answer)
        return {"answer": answer, "code": code, "mode": "code"}

    answer = parsed.get("answer", response_text)
    _add_to_memory(dataset_id, question, answer)
    return {"answer": answer, "mode": "analysis"}


def clear_history(dataset_id: int):
    _memories.pop(dataset_id, None)
