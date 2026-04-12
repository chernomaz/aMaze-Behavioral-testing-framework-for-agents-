import os
import sys
import logging
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent as create_agent
from langsmith import traceable, tracing_context
from tavily import TavilyClient

logger = logging.getLogger(__name__)

# ---------- Vectorstore (optional — skipped if Chroma/Ollama unavailable) ----------

_retriever = None

def _try_init_vectorstore():
    global _retriever
    try:
        from langchain_chroma import Chroma
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.document_loaders import PyPDFDirectoryLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        DATA_DIR = os.getenv("DATA_DIR", "/home/ubuntu/data/learn/pdf_reader/")
        CHROMA_DIR = os.getenv("CHROMA_DIR", "/home/ubuntu/data/learn/chroma_db/")
        COLLECTION_NAME = "pdf_docs"

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        existing = vectorstore.get(limit=1)
        if not existing["ids"]:
            loader = PyPDFDirectoryLoader(DATA_DIR, recursive=True)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            vectorstore.add_documents(chunks)
        _retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        logger.info("Vectorstore initialized successfully.")
    except Exception as e:
        logger.warning("Could not initialize vectorstore (pdf_search will be unavailable): %s", e)
        _retriever = None

_try_init_vectorstore()

# ---------- Tools ----------

@tool
@traceable(name="pdf_search_fn")
def pdf_search(query: str) -> str:
    """Search the local PDF collection and return relevant passages with sources."""
    logger.info("pdf_search called: %s", query)
    if _retriever is None:
        return "PDF search unavailable: vectorstore not initialized."
    docs = _retriever.invoke(query)
    if not docs:
        return "No relevant PDF content found."
    parts: List[str] = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "unknown")
        parts.append(f"Source: {src}, page: {page}\nContent: {d.page_content}")
    answer = "\n\n".join(parts)
    logger.info("pdf_search return: %s", answer[:200])
    return answer


_tavily = None
def _get_tavily():
    global _tavily
    if _tavily is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if api_key:
            _tavily = TavilyClient(api_key=api_key)
    return _tavily

@tool
@traceable(name="web_search_fn")
def web_search(query: str) -> str:
    """Search the web for recent information."""
    logger.info("web_search called: %s", query)
    client = _get_tavily()
    if client is None:
        return "Web search unavailable: TAVILY_API_KEY not set."
    results = client.search(query=query, max_results=5)
    items = results.get("results", [])
    if not items:
        return "No relevant web results found."
    answer = "\n\n".join(
        f"Title: {item.get('title')}\nURL: {item.get('url')}\nContent: {item.get('content')}"
        for item in items
    )
    logger.info("web_search return: %s", answer[:200])
    return answer


EMAILS = [
    {
        "to": "alice",
        "subject": "Q1 Report",
        "body": "Hi Alice, please find the Q1 report attached. Let me know if you have any questions.",
    },
    {
        "to": "bob",
        "subject": "Meeting Tomorrow",
        "body": "Hi Bob, just a reminder that we have a team meeting tomorrow at 10am. See you there!",
    },
    {
        "to": "carol",
        "subject": "Project Update",
        "body": "Hi Carol, wanted to share a quick update on the project — we are on track for the deadline.",
    },
]

@tool
@traceable(name="dummy_email")
def dummy_email(person: Optional[str] = None) -> str:
    """Return predefined dummy emails. If person is specified, return only the email for that person.
    Known recipients: alice, bob, carol.
    """
    logger.info("dummy_email invoke person=%s", person)
    if person:
        match = [e for e in EMAILS if e["to"].lower() == person.strip().lower()]
        if not match:
            return f"No email found for '{person}'."
        e = match[0]
        return f"To: {e['to']}\nSubject: {e['subject']}\n\n{e['body']}"
    parts = []
    for e in EMAILS:
        parts.append(f"To: {e['to']}\nSubject: {e['subject']}\n\n{e['body']}")
    return "\n\n---\n\n".join(parts)


@tool
@traceable(name="file_read")
def file_read(path: str) -> str:
    """Read the contents of a local file and return it as a string."""
    logger.info("file_read invoke path=%s", path)
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    if not os.path.isfile(path):
        return f"Error: path is not a file: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info("file_read read %d chars from %s", len(content), path)
        return content
    except Exception as e:
        logger.error("file_read error: %s", e)
        return f"Error reading file: {e}"


# ---------- LLM ----------

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ---------- Agent ----------

def main():
    # Accept prompt from env var, CLI arg (3rd positional after script + policy), or stdin
    prompt = os.environ.get("AGENT_PROMPT", "").strip()
    if not prompt and len(sys.argv) > 3:
        prompt = sys.argv[3].strip()
    if not prompt:
        prompt = input("You: ").strip()
    if prompt.lower() in {"exit", "quit"}:
        return

    agent = create_agent(
        model=llm,
        tools=[pdf_search, web_search, dummy_email, file_read],
        prompt=(
            "You are a helpful research assistant. "
            "Use pdf_search for questions about local PDF documents. "
            "Use web_search for current or external information. "
            "Use dummy_email to retrieve emails for alice, bob, or carol. "
            "Use file_read to read the contents of a local file by path. "
            "Always cite which tool and source you used."
        ),
    )

    with tracing_context(enabled=True):
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

    print("\nAssistant:")
    print(result["messages"][-1].content)
    print()


if __name__ == "__main__":
    main()
