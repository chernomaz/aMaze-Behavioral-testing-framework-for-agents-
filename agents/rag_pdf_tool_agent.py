from typing import List
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import api_key
from tavily import TavilyClient

from langsmith import Client, tracing_context, traceable
import os





from dotenv import load_dotenv
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "/home/ubuntu/data/learn/pdf_reader/")
CHROMA_DIR = os.getenv("CHROMA_DIR", "/home/ubuntu/data/learn/chroma_db/")
COLLECTION_NAME = "pdf_docs"


def load_pdf_documents(data_dir: str):
    loader = PyPDFDirectoryLoader(data_dir, recursive=True)
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)


def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")
    #return OpenAIEmbeddings(model="text-embedding-3-large")


def build_vectorstore_once() -> Chroma:
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    # Check whether collection already has documents
    existing = vectorstore.get(limit=1)

    if existing["ids"]:
        print("Using existing Chroma index...")
        return vectorstore

    print("No existing index found. Building vector store from PDFs...")

    docs = load_pdf_documents(DATA_DIR)

    for doc in docs:
        doc.metadata["tenant_id"] = "companyA"   # optional custom metadata

    chunks = split_documents(docs)
    vectorstore.add_documents(chunks)

    print("Vector store created successfully.")
    return vectorstore
# ---------- Build / load PDF vector store ----------
def build_vectorstore() -> Chroma:
    loader = PyPDFDirectoryLoader(DATA_DIR, recursive=True)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma(
        collection_name="pdf_docs",
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    existing = vectorstore.get()
    if existing and existing.get("ids"):
        vectorstore.delete(ids=existing["ids"])

    vectorstore.add_documents(chunks)
    return vectorstore


vectorstore = build_vectorstore_once()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ---------- Tools ----------
@tool
@traceable(name="pdf_search_fn")
def pdf_search(query: str) -> str:
    """Search the local PDF collection and return relevant passages with sources."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant PDF content found."

    parts: List[str] = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "unknown")
        parts.append(
            f"Source: {src}, page: {page}\n"
            f"Content: {d.page_content}"
        )
    return "\n\n".join(parts)


tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
@traceable(name="web_search_fn")
def web_search(query: str) -> str:
    """Search the web for recent information."""
    results = tavily.search(query=query, max_results=5)
    items = results.get("results", [])
    if not items:
        return "No relevant web results found."

    answer= "\n\n".join(
        f"Title: {item.get('title')}\n"
        f"URL: {item.get('url')}\n"
        f"Content: {item.get('content')}"
        for item in items
    )
    return answer

@tool
@traceable(name="multi_fn")
def multiply(a: float, b: float) -> float:
    """Multiply two integers."""
    return a * b


llm = ChatOpenAI(
    model="gpt-4.1-mini",  # good balance cost/quality
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ---------- Agent ----------

def main():

    agent = create_agent(
        model=llm,
        tools=[pdf_search, web_search, multiply],
        system_prompt=(
            "You are a helpful research assistant. "
            "Use pdf_search for questions about the local PDF folder. "
            "Use web_search for current or external info. "
            "If both are relevant, use both. "
            "Always cite which tool and source you used."
        ),
    )
    # ---------- Chat loop ----------
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        with tracing_context(enabled=True):
            result = agent.invoke(
                {
                    "messages": [
                        {"role": "user", "content": q}
                    ]
                }
            )

            print("\nAssistant:")
            print(result["messages"][-1].content)
            print()

if __name__ == "__main__":
    main()