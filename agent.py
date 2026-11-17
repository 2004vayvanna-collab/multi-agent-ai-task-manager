
from ollama import chat
import json
from pathlib import Path

# Location of our JSON file
TASKS_FILE = Path(__file__).resolve().parent / "tasks.json"


# Read tasks from the file
def load_tasks():
    if not TASKS_FILE.exists():
        return []

    with open(TASKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


# Save tasks to the file
def save_tasks(tasks):
    with open(TASKS_FILE, "w", encoding="utf-8") as file:
        json.dump(tasks, file, indent=4)


# TOOL 1: Add a task
def add_task(task: str) -> str:
    """Add a new task to the to-do list.

    Args:
        task: Description of the task to add.
    """
    tasks = load_tasks()
    tasks.append(task)
    save_tasks(tasks)

    return f"Task added: {task}"


# TOOL 2: Show all tasks
def list_tasks() -> str:
    """Show all tasks in the to-do list."""
    tasks = load_tasks()

    if not tasks:
        return "Your task list is empty."

    return "\n".join(
        f"{i}. {task}"
        for i, task in enumerate(tasks, 1)
    )


# TOOL 3: Delete a task
def delete_task(task_number: int) -> str:
    """Delete a task using its number in the list.

    Args:
        task_number: The one-based task number to delete.
    """
    tasks = load_tasks()

    if task_number < 1 or task_number > len(tasks):
        return "Invalid task number."

    removed = tasks.pop(task_number - 1)
    save_tasks(tasks)

    return f"Deleted: {removed}"


# Available tools
tools = {
    "add_task": add_task,
    "list_tasks": list_tasks,
    "delete_task": delete_task
}


messages = [
    {
        "role": "system",
        "content": """
        You are a helpful AI To-Do Assistant.

        You can add, list, and delete tasks.

        Rules:
        - Use add_task to add tasks.
        - Use list_tasks to show tasks.
        - Before deleting, use list_tasks if the
          task number is not known.
        - Use delete_task to delete by task number.
        - Never invent tasks.
        - Never claim success without using a tool.
        - Keep answers short.
        """
    }
]


print("AI TO-DO ASSISTANT")
print("Type 'exit' to quit.\n")


while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    if not user_input:
        continue

    messages.append({
        "role": "user",
        "content": user_input
    })

    for step in range(5):
        response = chat(
            model="qwen3:1.7b",
            messages=messages,
            tools=list(tools.values()),
            think=False
        )

        messages.append(response.message)

        if not response.message.tool_calls:
            print("AI:", response.message.content)
            break

        for call in response.message.tool_calls:
            name = call.function.name
            args = call.function.arguments

            print(f"[Calling tool: {name}]")

            if name in tools:
                try:
                    result = tools[name](**args)
                except (TypeError, ValueError) as e:
                    result = f"Tool error: {e}"
            else:
                result = "Unknown tool"

            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": str(result)
            })
    else:
        print("Tool-call limit reached.")

    print()
