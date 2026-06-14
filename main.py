"""
ATLAS AI — FastAPI backend + Streamlit frontend with LangChain + LangGraph.

Run locally:
    uvicorn main:app --reload          # API on :8000
    streamlit run streamlit_app.py     # UI on :8501

Env vars (.env):
    LLM_PROVIDER=anthropic|ollama
    ANTHROPIC_API_KEY=...
    API_BASE_URL=http://localhost:8000   # used by streamlit_app.py
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# Config (env vars)
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Globals — set once at startup, reused across requests
# ---------------------------------------------------------------------------
agent = None
vector_store = None


def build_llm():
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=ANTHROPIC_API_KEY)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.2")


def build_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def build_tools(store: InMemoryVectorStore):
    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the ATLAS knowledge base for relevant information."""
        results = store.similarity_search(query, k=3)
        if not results:
            return "No relevant documents found."
        return "\n\n".join(doc.page_content for doc in results)

    @tool
    def summarize_topic(topic: str) -> str:
        """Retrieve content related to a topic for summarization."""
        results = store.similarity_search(topic, k=5)
        if not results:
            return f"No information found on: {topic}"
        return " ".join(doc.page_content for doc in results)

    return [search_knowledge_base, summarize_topic]


# ---------------------------------------------------------------------------
# App lifespan — initialize LangGraph agent once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, vector_store

    embeddings = build_embeddings()
    vector_store = InMemoryVectorStore(embedding=embeddings)

    vector_store.add_documents([
        Document(page_content="ATLAS AI is a RAG and agentic AI system built with LangChain and LangGraph."),
        Document(page_content="LangGraph builds stateful, graph-based agents with tool-calling LLMs."),
        Document(page_content="AWS Bedrock provides managed foundation models including Claude and Titan."),
    ])

    llm = build_llm()
    tools = build_tools(vector_store)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt="You are ATLAS AI. Use your tools to answer questions using the knowledge base.",
    )

    yield  # app runs here


app = FastAPI(title="ATLAS AI", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "provider": LLM_PROVIDER}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    result = await agent.ainvoke({"messages": [{"role": "user", "content": request.message}]})
    final = result["messages"][-1]
    text = final.content if isinstance(final.content, str) else str(final.content)
    return ChatResponse(response=text)