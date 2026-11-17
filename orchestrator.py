
from ollama import chat
from pathlib import Path
import json

MODEL = "qwen3:1.7b"
TASKS_FILE = Path(__file__).resolve().parent / "tasks.json"


def load_tasks():
    if not TASKS_FILE.exists():
        return []
    return json.loads(TASKS_FILE.read_text(encoding="utf-8"))


def save_tasks(tasks):
    TASKS_FILE.write_text(
        json.dumps(tasks, indent=4),
        encoding="utf-8"
    )


# Tools for the Task Manager Agent

def add_task(task: str) -> str:
    """Add a task to the list.

    Args:
        task: Only the task name, without instructions
              like 'add' or 'to my tasks'.
    """
    import re

    task = re.sub(
        r"^\s*(?:please\s+)?add\s+",
        "",
        task,
        flags=re.IGNORECASE
    )

    task = re.sub(
        r"\s+to my tasks?\s*$",
        "",
        task,
        flags=re.IGNORECASE
    ).strip(" .")

    if not task:
        return "Error: task name cannot be empty."

    tasks = load_tasks()
    tasks.append(task)
    save_tasks(tasks)

    return f"Added: {task}"



def list_tasks() -> str:
    """List all saved tasks."""
    tasks = load_tasks()
    if not tasks:
        return "No tasks saved."
    return "\n".join(
        f"{i}. {task}"
        for i, task in enumerate(tasks, 1)
    )


def delete_task(task_number: int) -> str:
    """Delete a task by its number.

    Args:
        task_number: One-based number of the task.
    """
    tasks = load_tasks()
    if not 1 <= task_number <= len(tasks):
        return "Invalid task number."
    deleted = tasks.pop(task_number - 1)
    save_tasks(tasks)
    return f"Deleted: {deleted}"


TOOLS = {
    "add_task": add_task,
    "list_tasks": list_tasks,
    "delete_task": delete_task
}


# AGENT 1: Manage tasks
def task_manager_agent(request):
    print("[Orchestrator -> Task Manager Agent]")

    messages = [
        {
            "role": "system",
            "content": (
                "You manage a to-do list. "
                "Use the available tools for adding, "
                "listing and deleting tasks. "
                "Do not invent results."
            )
        },
        {"role": "user", "content": request}
    ]

    for _ in range(4):
        response = chat(
            model=MODEL,
            messages=messages,
            tools=list(TOOLS.values()),
            think=False
        )
        messages.append(response.message)

        if not response.message.tool_calls:
            return response.message.content

        for call in response.message.tool_calls:
            name = call.function.name

            if name in TOOLS:
                try:
                    result = TOOLS[name](
                        **call.function.arguments
                    )
                except (TypeError, ValueError) as error:
                    result = f"Tool error: {error}"
            else:
                result = "Unknown tool"

            print(f"  [Tool: {name}]")

            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": str(result)
            })

    return "Tool-call limit reached."


# AGENT 2: Plan tasks
def planning_agent():
    print("[Orchestrator -> Planning Agent]")

    tasks = load_tasks()

    if not tasks:
        return "No tasks available for planning."

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a study planning agent. "
                    "Create a simple daily plan using only "
                    "the provided tasks. You may divide "
                    "them into smaller study activities. "
                    "Respond in English, not JSON. "
                    "Do not request tools."
                )
            },
            {
                "role": "user",
                "content": (
                    "Create a daily study plan for:\n"
                    + "\n".join(tasks)
                )
            }
        ],
        think=False
    )

    return response.message.content


# ORCHESTRATOR
def orchestrator(request):
    text = request.lower().strip()

    if text.startswith((
        "plan",
        "organize",
        "create a schedule",
        "make a schedule"
    )):
        return planning_agent()

    return task_manager_agent(request)


print("MULTI-AGENT ASSISTANT")
print("Type 'exit' to quit.\n")

while True:
    request = input("You: ").strip()

    if request.lower() == "exit":
        break

    if not request:
        continue

    answer = orchestrator(request)
    print("\nAI:", answer, "\n")
