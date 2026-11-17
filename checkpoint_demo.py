
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


# 1. Define shared state
class WorkflowState(TypedDict, total=False):
    request: str
    counter: int
    last_request: str
    answer: str


# 2. Create a node that remembers requests
def assistant_node(state: WorkflowState):
    print("[Assistant Node]")

    count = state.get("counter", 0) + 1
    previous = state.get("last_request", "None")
    current = state["request"]

    return {
        "counter": count,
        "last_request": current,
        "answer": (
            f"Current request: {current}\n"
            f"Previous request: {previous}\n"
            f"Total requests: {count}"
        )
    }


# 3. Build LangGraph
builder = StateGraph(WorkflowState)

builder.add_node("assistant", assistant_node)

builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)


# 4. Create persistent SQLite storage
DB_PATH = Path(__file__).resolve().parent / "checkpoints.sqlite"

with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:

    graph = builder.compile(checkpointer=checkpointer)

    # Identifies this conversation
    config = {
        "configurable": {
            "thread_id": "my-first-conversation"
        }
    }

    saved_state = graph.get_state(config)

    if saved_state.values:
        print(
            "Restored request count:",
            saved_state.values.get("counter", 0)
        )

    print("LANGGRAPH MEMORY DEMO")
    print("Type 'exit' to quit.\n")

    # 5. Execute the graph
    while True:
        request = input("You: ").strip()

        if request.lower() == "exit":
            break

        if not request:
            continue

        result = graph.invoke(
            {"request": request},
            config=config
        )

        print("\nAI:")
        print(result["answer"])
        print()
