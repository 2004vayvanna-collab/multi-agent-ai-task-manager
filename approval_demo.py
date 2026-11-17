
import json
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver


BASE = Path(__file__).resolve().parent
TASKS_FILE = BASE / "tasks.json"
DB_FILE = BASE / "approvals.sqlite"


# Shared workflow state
class State(TypedDict, total=False):
    task_number: int
    task_name: str
    approved: bool
    outcome: str


def load_tasks():
    return json.loads(TASKS_FILE.read_text(encoding="utf-8"))


# NODE 1: Ask for human approval
def approval_node(state: State):
    print("[Approval Node]")

    answer = interrupt({
        "question": "Do you approve deleting this task?",
        "task": state["task_name"]
    })

    return {"approved": answer is True}


# Choose the next node
def choose_path(state: State):
    return "delete" if state["approved"] else "cancel"


# NODE 2: Delete only after approval
def delete_node(state: State):
    print("[Delete Node]")

    tasks = load_tasks()
    index = state["task_number"] - 1

    # Check that the selected task hasn't changed
    if not (0 <= index < len(tasks)):
        return {"outcome": "Task changed. Nothing deleted."}

    if tasks[index] != state["task_name"]:
        return {"outcome": "Task changed. Nothing deleted."}

    removed = tasks.pop(index)

    TASKS_FILE.write_text(
        json.dumps(tasks, indent=4),
        encoding="utf-8"
    )

    return {"outcome": f"Deleted: {removed}"}


# NODE 3: Cancel deletion
def cancel_node(state: State):
    print("[Cancel Node]")
    return {"outcome": "Deletion rejected. Task preserved."}


# Build the graph
builder = StateGraph(State)

builder.add_node("approval", approval_node)
builder.add_node("delete", delete_node)
builder.add_node("cancel", cancel_node)

builder.add_edge(START, "approval")

builder.add_conditional_edges(
    "approval",
    choose_path,
    {
        "delete": "delete",
        "cancel": "cancel"
    }
)

builder.add_edge("delete", END)
builder.add_edge("cancel", END)


# SQLite persistence
config = {
    "configurable": {
        "thread_id": "approval-demo"
    }
}

with SqliteSaver.from_conn_string(str(DB_FILE)) as saver:

    graph = builder.compile(checkpointer=saver)

    snapshot = graph.get_state(config)

    print("HUMAN APPROVAL DEMO")

    # Recover approval pending from a previous session
    if snapshot.interrupts:
        pending = snapshot.interrupts[0].value
        print("Restored pending request:", pending["task"])

    else:
        tasks = load_tasks()

        for i, task in enumerate(tasks, 1):
            print(f"{i}. {task}")

        number = int(input("\nTask number to delete: "))

        if number < 1 or number > len(tasks):
            print("Invalid task number.")
            raise SystemExit

        result = graph.invoke(
            {
                "task_number": number,
                "task_name": tasks[number - 1]
            },
            config=config
        )

        if "__interrupt__" in result:
            print("\n[LangGraph paused for approval]")


    # Resume the paused graph
    decision = input("\nApprove deletion? (yes/no): ")
    approved = decision.strip().lower() == "yes"

    result = graph.invoke(
        Command(resume=approved),
        config=config
    )

    print("\nRESULT:", result["outcome"])
