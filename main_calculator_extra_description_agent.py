from dataclasses import dataclass
import os
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

@tool("calculator", 
    parse_docstring=True,
    description=(
        "Calculate the result of a and b based on the operation."
        "Always use this function to calculate and don't invent your own even if you know the answer."
    ),
)
def calculate(a: float, b: float, operation: Literal["add", "subtract", "multiply", "divide"]) -> str:
    """ Calculate the result of a and b based on the operation 
    
    Args:
        a (float): The first number
        b (float): The second number
        operation (Literal["add", "subtract", "multiply", "divide"]): The operation to perform
    
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if (b == 0):
            return ValueError("Cannot divide by zero")
        return a / b
    else:
        ValueError("Invalid operation")
    

prompt = """
You are a calculator.
"""

model = ChatOpenAI(
  api_key=os.getenv("OPENROUTER_API_KEY"),
  base_url="https://openrouter.ai/api/v1",
  model="google/gemini-2.5-flash-lite-preview-09-2025",
  temperature=0 # Controls the randomness. Between 0-
)

agent = create_agent(model, tools=[calculate], system_prompt=prompt)

with open(os.path.basename(__file__).replace(".py", ".png"), "wb") as file:
    file.write(agent.get_graph().draw_mermaid_png())

question = "What is 3 * 1?"
for step in agent.stream(
    { "messages": question},
    stream_mode="values"
):
    step["messages"][-1].pretty_print()
# NOTE: invoke does not have streaming