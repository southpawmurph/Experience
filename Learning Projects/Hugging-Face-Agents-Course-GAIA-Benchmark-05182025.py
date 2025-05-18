import os
import json
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI # Using Gemini for wider compatibility in example
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, START, END
from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition # Import ToolNode and tools_condition if using tools
from langchain_core.tools import tool # if you define custom tools
from langgraph.prebuilt import create_react_agent

os.environ["LANGSMITH_TRACING"] = os.environ.get("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_ENDPOINT"] = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.environ.get("LANGSMITH_PROJECT", "huggingface_agent_benchmark_revised")


# --- Constants ---
QUESTIONS_URL = "https://agents-course-unit4-scoring.hf.space/questions"
FILES_BASE_URL = "https://agents-course-unit4-scoring.hf.space/files"
SUBMIT_URL = "https://agents-course-unit4-scoring.hf.space/submit" # Assuming this is the correct endpoint

USERNAME = "user_name_here"
AGENT_CODE_URL = "https://huggingface.co/spaces/southpawmurph/huggin_face_agent_course/tree/main" # Or your actual agent code URL

# --- LLM Configuration ---
llm = ChatVertexAI(model_name='gemini-2.5-pro-preview-05-06',project='project_id_here')

# --- System Prompt for Agent ---
system_prompt = """You are a general AI assistant. Your goal is to answer the question accurately.
You will be given a question. If a file is associated with the question, its local path will be provided.
You should use the information in the file if it's relevant to answering the question.
Think step-by-step to arrive at the answer.
Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""

# --- Agent State Definition ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# --- Fetch Questions ---
def fetch_all_questions():
    """Fetches all questions from the API."""
    print("Fetching all questions...")
    response = requests.get(url=QUESTIONS_URL)
    questions_data = response.json()
    print(f"Successfully fetched {len(questions_data)} questions.")
    return questions_data

def get_file(task_id: str, file_name: str) -> str:
    """
    Downloads a file for a given task_id and saves it with file_name.
    Returns the absolute file path if successful, None otherwise.
    """
    print(f"Attempting to download file '{file_name}' for task_id '{task_id}'...")
    file_url = f"{FILES_BASE_URL}/{task_id}"
    response = requests.get(url=file_url)
    file_bytes = response.content
    content_type = response.headers.get('Content-Type', '').lower()
    files_dir = "downloaded_files"
    os.makedirs(files_dir, exist_ok=True)
    local_file_path = os.path.join(files_dir, file_name)

    if 'audio/mpeg' in content_type: # mp3
        with open(local_file_path, 'wb') as f:
            f.write(file_bytes)
    elif 'image/png' in content_type: # png
        with open(local_file_path, 'wb') as f:
            f.write(file_bytes)
    elif 'text/x-python' in content_type or file_name.endswith(".py"): # py
        with open(local_file_path, 'w', encoding='utf-8') as f:
            f.write(file_bytes.decode('utf-8'))
    elif 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' in content_type or \
        'application/octet-stream' in content_type and file_name.endswith(".xlsx"): # xlsx
        with open(local_file_path, 'wb') as f:
            f.write(file_bytes)
    else:
        print(f"Warning: File type '{content_type}' for '{file_name}' might not be fully supported or recognized. Saving as binary.")
        # Default to binary write if type is unknown but filename suggests it
        with open(local_file_path, 'wb') as f:
            f.write(file_bytes)
    abs_path = os.path.abspath(local_file_path)
    print(f"File '{file_name}' downloaded and saved to '{abs_path}'.")
    return abs_path

def extract_final_answer_from_messages(messages: List[BaseMessage]) -> str:
    """Extracts the content after 'FINAL ANSWER: ' from the last AIMessage."""
    if not messages or not isinstance(messages[-1], AIMessage):
        return "Error: No AIMessage found or last message not AI."
    content = messages[-1].content
    marker = "FINAL ANSWER:"
    if marker in content:
        return content.split(marker, 1)[1].strip()
    else:
        # Fallback: if no marker, return the whole content, but log a warning
        print(f"Warning: 'FINAL ANSWER:' marker not found in agent's response: '{content[:100]}...'")
        return content.strip()

def submit_all_answers(answers: List[Dict[str, str]]) -> Dict[str, Any]:
    """Submits all collected answers to the scoring API."""
    submission_payload = {
        "username": USERNAME,
        "agent_code": AGENT_CODE_URL,
        "answers": answers
    }
    print("\nSubmitting answers...")
    print(f"Payload: {json.dumps(submission_payload, indent=2)}")

    response = requests.post(SUBMIT_URL, json=submission_payload)
    response_data = response.json()
    print("\n--- Submission Response ---")
    print(json.dumps(response_data, indent=2))
    print("--- End Submission Response ---")
    return response_data

# --- LangGraph Agent Setup ---
agent_executor = create_react_agent(model=llm, tools=[], prompt=system_prompt)


def main():
    print("Starting agent process...")
    all_questions_data = fetch_all_questions()
    if not all_questions_data:
        print("No questions fetched. Exiting.")
        return
    collected_answers_for_submission = []
    for i, question_item in enumerate(all_questions_data):
        task_id = question_item.get("task_id")
        question_text = question_item.get("question")
        file_name = question_item.get("file_name", "") # Default to empty string if not present
        if not task_id or not question_text:
            print(f"Skipping item {i+1} due to missing task_id or question: {question_item}")
            continue
        print(f"\n--- Processing Question {i+1}/{len(all_questions_data)} ---")
        print(f"Task ID: {task_id}")
        print(f"Question: {question_text}")
        message_content_for_agent = f"Question: {question_text}"
        file_path_for_agent = None
        
        if file_name:
            file_path_for_agent = get_file(task_id=task_id, file_name=file_name)
            if file_path_for_agent:
                message_content_for_agent += f"\n\nAn associated file has been downloaded for you. Its local path is: '{file_path_for_agent}'"
                message_content_for_agent += "\nMake sure to analyze the content of this file if it's relevant to answering the question."
            else:
                message_content_for_agent += f"\n\nThere was an attempt to download an associated file named '{file_name}', but it failed. Answer based on the question text alone if possible."
        else:
            message_content_for_agent += "\n\nThere is no associated file for this question."
        
        # Prepare input for the agent
        # The agent_executor created by create_react_agent expects a dictionary with 'messages'
        initial_agent_input = {"messages": [HumanMessage(content=message_content_for_agent)]}
        print(f"\nInvoking agent for task {task_id}...")
        try:
            # The create_react_agent returns a Runnable. Invoking it runs the agent.
            # The output should be a dictionary, typically with an 'messages' key containing the history.
            agent_response_dict = agent_executor.invoke(initial_agent_input, {"recursion_limit": 15})
            print("\nAgent's response messages:")
            for msg in agent_response_dict.get('messages', []):
                print(f"- {msg.type}: {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
            submitted_answer = extract_final_answer_from_messages(agent_response_dict.get('messages', []))
            print(f"Extracted final answer: {submitted_answer}")
        except Exception as e:
            print(f"Error invoking agent for task {task_id}: {e}")
            import traceback
            traceback.print_exc()
            submitted_answer = "Error: Agent failed to process this question."

        collected_answers_for_submission.append({
            "task_id": task_id,
            "submitted_answer": submitted_answer
        })
        print("--- End Processing Question ---")
    # After processing all questions, submit the answers
    if collected_answers_for_submission:
        submit_all_answers(collected_answers_for_submission)
    else:
        print("No answers were collected to submit.")

    print("\nAgent process finished.")

# --- Execute Script ---
main()
