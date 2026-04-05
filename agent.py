import os
import asyncio

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langsmith import tracing_context

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

async def main():
    client = MultiServerMCPClient(
        {
            "tools": {
                "url": "http://127.0.0.1:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )

    tools = await client.get_tools()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful research assistant. "
            "Use pdf_search for local PDFs. "
            "Use web_search for external info. "
            "Always cite sources."
        ),
    )

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        try:
            with tracing_context(enabled=True):
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": q}]}
                )

            print("\nAssistant:")
            print(result["messages"][-1].content)
            print()

        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())
