import os
from dotenv import load_dotenv

# Setup LangSmith
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"]=""
os.environ["LANGSMITH_PROJECT"]=""

# For the LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# For the graph
from langgraph.graph import StateGraph, START, END

# For state management in the graph
from typing import Annotated
from typing_extensions import TypedDict

# For the agent
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages

# Ask for a quote from Naval Ravikant
input_messages = {"messages":[{"role":"user","content":"Hi. Please tell me a quote from one of my favorite role models Naval Ravikant."}]}

# Setup LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",api_key="")

# Setup Tools
tools = []

agent = create_react_agent(llm,tools)

# Setup Graph and State
class state(TypedDict):
  messages: Annotated[list,add_messages]

graph_builder = StateGraph(state)
graph_builder.add_node("agent",agent)
graph_builder.add_edge(START,"agent")
graph_builder.add_edge("agent",END)
graph = graph_builder.compile()
# Call LLM
response = graph.invoke(input_messages)

# View Messages
for msg in response["messages"]:
  msg.pretty_print()
