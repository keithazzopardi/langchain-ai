from dataclasses import dataclass
import os
from dotenv import load_dotenv
from langchain.agents.middleware.types import ModelRequest, dynamic_prompt
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langgraph.runtime import get_runtime
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

db = SQLDatabase.from_uri(os.getenv("DATABASE_URL"))

#print(db.run("SELECT * FROM ALBUM LIMIT 5"))

prompt = """
You are a Postgres database expert.

Rules:
- Think step by step
- When calling the tool, always explain the steps you are taking
- When you need the data, call execute_query with one select query
- No updates. Only Selects
- Limit 5 rows
{table_restrictions}
- Prefer explicit column list
"""

@dataclass
class RuntimeContext:
    isEmployee: bool
    db: SQLDatabase

@dynamic_prompt
def dynamic_system_prompt_employee(request: ModelRequest) -> str:
    table_limits = ""
    if request.runtime.context.isEmployee:
        table_limits = "There's no limits."
    else:
        table_limits = "Limit access to these tables: Album, Artists, Genre, Playlist, PlaylistTrack, Track."
    
    return prompt.format(table_restrictions=table_limits)


@tool
def execute_query(query: str) -> str:
    """ Execute a SQL query """
    db = get_runtime().context.db
    try:
        return db.run(query)
    except Exception as e:
        return str(e)



model = ChatOpenAI(
  api_key=os.getenv("OPENROUTER_API_KEY"),
  base_url="https://openrouter.ai/api/v1",
  model="google/gemini-2.5-flash-lite-preview-09-2025",
  temperature=0 # Controls the randomness. Between 0-
)
memory = InMemorySaver()

agent = create_agent(model, 
    tools=[execute_query], 
    system_prompt=prompt, 
    context_schema=RuntimeContext,
    middleware=[
        dynamic_system_prompt_employee
        ],
    checkpointer=memory,    
)

with open(os.path.basename(__file__).replace(".py", ".png"), "wb") as file:
    file.write(agent.get_graph().draw_mermaid_png())

#question = "Which table has the highest number of records"
#question = "List all the tables"
#question = "What were the titles?"
question = "What is the latest invoice in the invoice table?"
thread_id = "1"
for step in agent.stream(
    { "messages": question},
    { "configurable": {"thread_id": thread_id}},
    context=RuntimeContext(isEmployee=True, db=db),
    stream_mode=["values", "custom"],
):
    if step[0] == "values":
        step[1]["messages"][-1].pretty_print()
    elif step[0] == "custom":
        print(step[1])
    #step["messages"][-1].pretty_print()

# NOTE: stream_mode="values" is used to get the return after each message
# stream_mode="messages" is used to return token by token - perfect for chatbot
# stream_mode="custom" is used to return custom values
