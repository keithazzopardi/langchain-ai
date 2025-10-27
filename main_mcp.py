from datetime import datetime
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import mcp
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class RepositoryInformation(BaseModel):
    name: str
    LastPush: datetime
    Description: str

def create_model():
    model = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="google/gemini-2.5-flash-lite-preview-09-2025",
        temperature=0 # Controls the randomness. Between 0-
    )
    return model
    

async def main():
    """Main function with proper context manager handling."""
    print("Starting GitHub MCP with LangChain agent...\n")
    
    # Check for required environment variables
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return
    
    if not os.environ.get("GITHUB_TOKEN"):
        print("Warning: GITHUB_TOKEN not set. Some operations may fail.")
    

    mcp_client = MultiServerMCPClient({
            "github": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"
                ],
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": os.environ.get("GITHUB_TOKEN", "")
                }
            }
        })

    try:
        tools = await mcp_client.get_tools()
        for tool in tools:
            print(f"   - {tool.name}")
        
        model = create_model()
        prompt = """
        You are a GitHub expert that must provide numbers and insights on organizations and repositories.
        Use the available GitHub tools to gather information and answer questions.
        """
        
        agent = create_agent(
            model, 
            tools=tools, 
            system_prompt=prompt,
            response_format=RepositoryInformation
        )
        
        query = "Can you list all repositories inside tasseitech organization?"
        async for step in agent.astream(
            {"messages": [("user", query)]},
            stream_mode="values"
        ):
            step["messages"][-1].pretty_print()
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())