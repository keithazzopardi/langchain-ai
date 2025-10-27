from dataclasses import dataclass
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.types import StreamMode

load_dotenv()

@tool
def check_haiku_lines(text: str) -> str:
    """ Check haiku lines if it is equal to 3"""
    if len(text.strip().splitlines()) == 3:
        return "Correct. Haiku has 3 lines."
    else:
        return "Incorrect. Haiku has more than 3 lines."

prompt = """
You are a sport poet who writes Poets.

Guidelines:
- Always check your work
"""

model = ChatOpenAI(
  api_key=os.getenv("OPENROUTER_API_KEY"),
  base_url="https://openrouter.ai/api/v1",
  model="google/gemini-2.5-flash-lite-preview-09-2025",
  temperature=0 # Controls the randomness. Between 0-
)

agent = create_agent(model, tools=[check_haiku_lines], system_prompt=prompt)

with open(os.path.basename(__file__).replace(".py", ".png"), "wb") as file:
    file.write(agent.get_graph().draw_mermaid_png())

question = "Please write to me a poet about happiness"
result = agent.invoke({ "messages": question})
for i, msg in enumerate(result["messages"]):
    msg.pretty_print()

# NOTE: invoke does not have streaming