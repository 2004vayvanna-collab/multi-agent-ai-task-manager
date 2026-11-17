
from ollama import chat
from pathlib import Path
import json

# Location of tasks.json
TASKS_FILE = Path(__file__).resolve().parent / "tasks.json"


# TOOL: Read saved tasks
def read_tasks() -> str:
    """Read the saved tasks from the task manager."""

    if not TASKS_FILE.exists():
        return "No tasks available."

    tasks = json.loads(
        TASKS_FILE.read_text(encoding="utf-8")
    )

    if not tasks:
        return "No tasks available."

    return "\n".join(
        f"{i}. {task}"
        for i, task in enumerate(tasks, 1)
    )


# Define the Planning Agent
messages = [
    {
        "role": "system",
        "content": """
        You are an AI Planning Agent.

        Your job is to organize tasks.

        Always call read_tasks before planning.

        Create a simple daily study plan.
        Do not invent tasks.
        """
    },
    {
        "role": "user",
        "content": "Organize my tasks into a daily plan."
    }
]


# STEP 1: Ask the AI to use the tool
response = chat(
    model="qwen3:1.7b",
    messages=messages,
    tools=[read_tasks],
    think=False
)

messages.append(response.message)

result = None

# STEP 2: Execute the requested tool
if response.message.tool_calls:

    for call in response.message.tool_calls:

        if call.function.name == "read_tasks":

            print("[Planning Agent calls read_tasks]")

            result = read_tasks()

            messages.append({
                "role": "tool",
                "tool_name": "read_tasks",
                "content": result
            })


# STEP 3: Generate a readable plan
if result is not None:

    final = chat(
        model="qwen3:1.7b",
        messages=[
            {
                "role": "system",
                "content": """
                You are a helpful study planner.

                Create a realistic daily study plan.
                Use only the provided tasks.

                Respond in plain English.
                Do not produce JSON or code.
                Do not call any tools.

                For each task, suggest a duration.
                Finish with one study tip.
                """
            },
            {
                "role": "user",
                "content": (
                    f"My tasks are:\n{result}\n\n"
                    "Create my daily study plan."
                )
            }
        ],
        think=False
    )

    print("\nYOUR STUDY PLAN:")
    print(final.message.content)

else:
    print("The agent did not call read_tasks.")
    print("Model response:", response.message.content)
