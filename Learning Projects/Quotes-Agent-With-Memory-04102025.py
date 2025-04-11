import os
# Setup LangSmith
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"]="_api_key_"
os.environ["LANGSMITH_PROJECT"]="_project_name_"

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
from langchain_core.prompts import PromptTemplate

# For the mnessages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# For memory
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

# Setup LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",api_key="_api_key_")

# Setup Memory Store
author_memory_store = InMemoryStore()

# Setup Memory Tools
# These tools allow the agent to interact with the author_memory_store
# The namespace helps organize data within the store if you use it for multiple things.
manage_memory_tool = create_manage_memory_tool(
    store=author_memory_store, # Pass the store instance here
    namespace=("quote_assistant", "author_preferences")
)

search_memory_tool = create_search_memory_tool(
    store=author_memory_store, # Pass the store instance here
    namespace=("quote_assistant", "author_preferences")
)
# Setup Graph and State
class State(TypedDict):
  messages: Annotated[list,add_messages]

def get_user_input(state: State):
  '''Collect a response from the user'''
  # Collect user response
  user_input = input()
  return {"messages":[HumanMessage(content=user_input)]}

def check_exit(state):
  """Check if the user provided exit, this will be used to determine if the graph ends"""
  last_message = state["messages"][-1]
  if isinstance(last_message, HumanMessage):
    if last_message.content.lower() == "exit":
      return "exit"
    else:
      return "agent_node"
  else:
    return "agent_node"

# Pass user input into a variable so it can be used to inject into the prompt
def format_prompt(state:State):
  last_message = state["messages"][-1] if state["messages"] else ""
  if isinstance(last_message, HumanMessage):
    user_input = last_message.content
  else:
    user_input = "No user input available"
  return {"input":user_input}

# Print Agent Output
def print_agent_output(state:State):
  last_message = state["messages"][-1]
  print("\nAgent:",last_message.content)
  return None

# Setup Tools
tools = [manage_memory_tool,search_memory_tool]

# Agent Prompt
# This is necessary for the agent to understand what to do
template = """You are a helpful Quotes Assistant. Your primary goal is to provide and discuss quotes with the user.
You have special tools to remember the user's preferences about specific authors.

RESPONSE FORMAT:
----------------
Follow this format strictly:

Question: The user's input question you need to answer.
Thought: Your reasoning process. Explain step-by-step how you'll answer the question, including whether you need to use a tool (and why) based on the guidelines above. If using a tool, specify which one and what input you'll provide.

The user's query:
{input}
"""

prompt = PromptTemplate.from_template(template)
agent = create_react_agent(llm,tools=tools)

def agent_node(state:State):
  formatted_prompt = format_prompt(state)
  # Using **formattted_prompt allows for multiple placeholdeers to be passed if they exist
  prompt = PromptTemplate.from_template(template).format(**formatted_prompt)
  agent_response = agent.invoke({"messages": state["messages"]})
  return agent_response
# Create Graph
graph_builder = StateGraph(State)
graph_builder.add_node("user_input", get_user_input)
graph_builder.add_node("agent_node",agent_node)
graph_builder.add_node("print_output",print_agent_output)
graph_builder.add_edge(START,"user_input")
graph_builder.add_conditional_edges("user_input",check_exit,{"agent_node": "agent_node", "exit": END})
graph_builder.add_edge("agent_node","print_output")
graph_builder.add_edge("print_output","user_input")
graph = graph_builder.compile()

# Passing empty initial state, required for the invoke
initial_state = {"messages":[]}
# Call LLM
response = graph.invoke(initial_state)
