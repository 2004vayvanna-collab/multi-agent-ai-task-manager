"""Local multi-agent task manager: LangGraph + Ollama + human approval."""

import json
import re
from pathlib import Path
from typing import TypedDict

from ollama import chat
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

MODEL = "qwen3:1.7b"
BASE_DIR = Path(__file__).resolve().parent
TASKS_FILE = BASE_DIR / "tasks.json"
CHECKPOINT_FILE = BASE_DIR / "task_manager_v2.sqlite"


class AgentState(TypedDict, total=False):
    request: str
    route: str
    answer: str
    turn_count: int
    task_number: int
    task_name: str
    approved: bool


def load_tasks() -> list[str]:
    if not TASKS_FILE.exists():
        return []
    data = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("tasks.json must contain a JSON list of strings")
    return data


def save_tasks(tasks: list[str]) -> None:
    TASKS_FILE.write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def add_task(task: str) -> str:
    """Add a new task to the to-do list.

    Args:
        task: Task description to add, without a leading command.
    """
    task = re.sub(r"^\s*(?:please\s+)?add\s+", "", task, flags=re.I)
    task = re.sub(r"\s+to\s+(?:my\s+)?tasks?\s*$", "", task, flags=re.I)
    task = task.strip(" .\t\r\n'\"")
    if not task:
        return "Please provide a task description."
    tasks = load_tasks()
    tasks.append(task)
    save_tasks(tasks)
    return f"Added: {task}"


def list_tasks() -> str:
    """Show all tasks currently in the to-do list."""
    tasks = load_tasks()
    return "\n".join(f"{i}. {task}" for i, task in enumerate(tasks, 1)) or "No tasks yet."


def router_node(state: AgentState) -> dict:
    print("[Router Node]")
    request = state["request"].strip().lower()
    if re.search(r"\b(delete|remove)\b", request):
        route = "delete"
    elif re.match(r"^(?:please\s+)?add\b", request):
        route = "manager"
    elif re.search(r"\b(plan|schedule|organize)\b", request):
        route = "planning"
    else:
        route = "manager"
    return {
        "route": route,
        "turn_count": state.get("turn_count", 0) + 1,
        "answer": "",
        "task_name": "",
        "task_number": 0,
        "approved": False,
    }


def manager_node(state: AgentState) -> dict:
    print("[Task Manager Agent]")
    request = state["request"].strip()
    adding = bool(re.match(r"^(?:please\s+)?add\b", request, re.I))
    messages = [
        {
            "role": "system",
            "content": (
                "You are a to-do assistant with add_task and list_tasks tools. "
                "Use the correct tool for task actions. Never invent tool results. "
                "Pass only the task description to add_task. "
                "Deletion is not available to you."
            ),
        },
        {"role": "user", "content": request},
    ]
    try:
        response = chat(
            model=MODEL,
            messages=messages,
            tools=[add_task, list_tasks],
            think=False,
            options={"num_ctx": 2048},
        )
    except Exception as exc:
        return {"answer": f"Ollama connection failed: {exc}"}

    # Only perform an add when the USER explicitly requested an add.
    for call in response.message.tool_calls or []:
        name = call.function.name
        args = call.function.arguments
        print(f"[Tool requested: {name}]")
        if name == "add_task" and adding:
            try:
                return {"answer": add_task(**args)}
            except (TypeError, ValueError) as exc:
                return {"answer": f"Task tool error: {exc}"}
        if name == "list_tasks" and not adding:
            return {"answer": list_tasks()}

    # Predictable fallback for the small 1.7B model's occasional tool-call errors.
    if adding:
        print("[Fallback: add_task]")
        return {"answer": add_task(request)}
    if re.search(r"\b(tasks?|todo|to-do)\b", request, re.I):
        print("[Fallback: list_tasks]")
        return {"answer": list_tasks()}
    return {"answer": response.message.content or "Try 'show my tasks' or 'add study Python'."}


def planning_node(state: AgentState) -> dict:
    print("[Planning Agent]")
    tasks = load_tasks()
    if not tasks:
        return {"answer": "No tasks available for planning."}
    try:
        response = chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a study planner. Create a short practical study plan "
                        "using only the task names provided. You may suggest subtasks, "
                        "but label them as suggestions. No JSON, no tool calls."
                    ),
                },
                {"role": "user", "content": "Plan these tasks:\n" + list_tasks()},
            ],
            think=False,
            options={"num_ctx": 2048},
        )
        return {"answer": response.message.content or "No plan generated."}
    except Exception as exc:
        return {"answer": f"Ollama connection failed: {exc}"}


def prepare_delete_node(state: AgentState) -> dict:
    print("[Prepare Deletion]")
    match = re.search(r"\b(?:task\s*(?:number|#)?\s*)?(\d+)\b", state["request"], re.I)
    if not match:
        return {"answer": "Please specify a task number, e.g. 'delete task number 2'."}
    number = int(match.group(1))
    tasks = load_tasks()
    if not 1 <= number <= len(tasks):
        return {"answer": f"Invalid task number. Current tasks:\n{list_tasks()}"}
    return {"task_number": number, "task_name": tasks[number - 1]}


def choose_delete_path(state: AgentState) -> str:
    return "approval" if state.get("task_name") else "end"


def approval_node(state: AgentState) -> dict:
    print("[Human Approval Node]")
    decision = interrupt({
        "question": "Approve deletion?",
        "task_number": state["task_number"],
        "task_name": state["task_name"],
    })
    return {"approved": decision is True}


def after_approval(state: AgentState) -> str:
    return "delete" if state.get("approved") else "cancel"


def delete_node(state: AgentState) -> dict:
    print("[Delete Node]")
    tasks = load_tasks()
    index = state["task_number"] - 1
    if index < 0 or index >= len(tasks) or tasks[index] != state["task_name"]:
        return {"answer": "Task list changed. Nothing was deleted."}
    removed = tasks.pop(index)
    save_tasks(tasks)
    return {"answer": f"Deleted: {removed}"}


def cancel_node(state: AgentState) -> dict:
    print("[Cancel Node]")
    return {"answer": "Deletion rejected. The task was kept."}


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("manager", manager_node)
    builder.add_node("planning", planning_node)
    builder.add_node("prepare_delete", prepare_delete_node)
    builder.add_node("approval", approval_node)
    builder.add_node("delete", delete_node)
    builder.add_node("cancel", cancel_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router", lambda s: s["route"],
        {"manager": "manager", "planning": "planning", "delete": "prepare_delete"},
    )
    builder.add_edge("manager", END)
    builder.add_edge("planning", END)
    builder.add_conditional_edges(
        "prepare_delete", choose_delete_path, {"approval": "approval", "end": END}
    )
    builder.add_conditional_edges(
        "approval", after_approval, {"delete": "delete", "cancel": "cancel"}
    )
    builder.add_edge("delete", END)
    builder.add_edge("cancel", END)
    return builder


def main():
    config = {"configurable": {"thread_id": "local-task-manager"}}
    with SqliteSaver.from_conn_string(str(CHECKPOINT_FILE)) as checkpointer:
        graph = build_graph().compile(checkpointer=checkpointer)

        def resolve_pending():
            snapshot = graph.get_state(config)
            if not snapshot.interrupts:
                return
            payload = snapshot.interrupts[0].value
            print(f"\nApproval needed: Delete task {payload['task_number']} "
                  f"({payload['task_name']})?")
            while True:
                answer = input("Approve? (yes/no): ").strip().lower()
                if answer in ("yes", "no"):
                    break
                print("Please enter 'yes' or 'no'.")
            result = graph.invoke(Command(resume=(answer == "yes")), config=config)
            print("AI:", result.get("answer", "No result"))

        saved = graph.get_state(config)
        print("MULTI-AGENT AI TASK MANAGER")
        print("Restored interactions:", saved.values.get("turn_count", 0))
        print("Type 'exit' to quit.\n")
        resolve_pending()

        while True:
            request = input("You: ").strip()
            if request.lower() == "exit":
                break
            if not request:
                continue
            result = graph.invoke({"request": request}, config=config)
            if "__interrupt__" in result:
                print("[Graph paused: human approval required]")
                resolve_pending()
            else:
                print("AI:", result.get("answer", "No result"))
            print("Total interactions:", graph.get_state(config).values.get("turn_count", 0))
            print()


if __name__ == "__main__":
    main()
