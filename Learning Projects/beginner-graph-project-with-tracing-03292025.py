#pip install langchain_google_genai
#pip install langgraph
#pip install langchain_core
#pip install langmem
#pip install dotenv
#pip install langsmith

# ## Basic Agent Example using LangGraph

import os
from dotenv import load_dotenv

# LangSmith Setup
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"]=""
os.environ["LANGSMITH_PROJECT"] = ""

# For state management in the graph
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# For building the graph
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent # Key component for ReAct logic



# For defining tools
from langchain_core.tools import tool

# For the LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# For the memory
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

# --- 1. Load API Keys (Best Practice) ---
_ = load_dotenv()

# --- 2. Define Your Tools ---
@tool
def inventory_tool(quantity_sold: int, quantity_available: int) -> int:
  """Calculate the remaining inventory."""
  print(f"--- Called inventory_tool: sold={quantity_sold}, available={quantity_available} ---")
  quantity_remaining = quantity_available - quantity_sold
  return quantity_remaining

@tool
def product_price_tool(product: str) -> float | str: # Allow returning string for errors
  """Get the price of a specific product."""
  print(f"--- Called product_price_tool: product={product} ---")
  prices = {"apple": 1.15, "banana": 0.75, "orange": 0.90}
  product_lower = product.lower()
  if product_lower in prices:
      return prices[product_lower]
  else:
      return f"Sorry, I don't have a price for {product}."

# List of tools the agent can use
tools = [inventory_tool, product_price_tool]

# --- 3. Initialize the LLM ---
# Using Gemini Flash as in your original code, ensure API key is handled
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="")

# --- 4. Define the Agent State ---
# The state holds the messages exchanged between the user and the agent/tools
class State(TypedDict):
    messages: Annotated[list, add_messages]

# --- 5. Create the Agent Node ---
# create_react_agent builds a graph node that performs the ReAct loop:
# Reason -> Decide Action (Tool?) -> Execute Action -> Observe -> Reason ...
# It uses the LLM to decide actions and process tool results.
agent_node = create_react_agent(llm, tools=tools)

# --- 6. Define the Graph ---
# This graph is very simple: it just runs the agent node.
graph_builder = StateGraph(State)

# Add the agent node to the graph
graph_builder.add_node("agent", agent_node)

# The entry point is the agent node
graph_builder.set_entry_point("agent")

# The agent node is also the finish point in this simple case
graph_builder.set_finish_point("agent")

# Compile the graph into a runnable object
# Adding checkpointer can add memory, but we'll skip for this basic example
graph = graph_builder.compile()

# --- 7. Invoke the Graph ---
print("Invoking agent to find the price of an apple...")

# The input must match the State structure, specifically the 'messages' key
# The ReAct agent expects the input query within the 'messages' list
input_messages = {"messages": [{"role": "user", "content": "What is the price of an apple?"}]}

# Run the graph. This will execute the ReAct loop.
# config can be used for things like recursion limits, thread IDs, etc.
# Add "recursion_limit": 5 to prevent infinite loops if the agent gets stuck
config = {"recursion_limit": 5}
response = graph.invoke(input_messages, config=config)

# --- 8. Display the Result ---
print("\n--- Agent Execution Finished ---")

print("\nFull response state:")
# The response contains the final state of the graph, including all messages
# print(response) # Uncomment to see the full state dictionary

print("\nMessages exchanged:")
for msg in response['messages']:
    msg.pretty_print() # Use pretty_print for better formatting

print("\nFinal Answer:")
# The last message in the list is typically the final response from the assistant
final_answer_message = response['messages'][-1]
if hasattr(final_answer_message, 'type') and final_answer_message.type == 'ai':
    # Ensure content exists before printing
    if hasattr(final_answer_message, 'content'):
      print(final_answer_message.content)
    else:
      print("Final AI message has no content.")
else:
    print("Agent did not produce a final assistant message.")

# Example with the other tool
# print("\nInvoking agent for inventory...")
# input_inventory = {"messages": [{"role": "user", "content": "If I sold 5 bananas and had 20 available, how many are left?"}]}
# response_inventory = graph.invoke(input_inventory, config=config)
# print("\nFinal Inventory Answer:")
# print(response_inventory['messages'][-1].content)
