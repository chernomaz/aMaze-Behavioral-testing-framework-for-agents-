
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
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

@tool
@traceable(name="multi_fn")
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    logger.info("multiply invoke %s * %s", a,b)
    return a * b

