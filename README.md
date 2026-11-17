# Multi-Agent AI Task Manager

A beginner-friendly, **fully local** Python task manager built with **LangGraph**, **Ollama**, and **Qwen3 1.7B**. No paid API key is required.

## Features
- Task manager with Ollama tool calling to add or list tasks
- AI planning agent generates a daily plan from saved tasks
- LangGraph conditional routing, SQLite checkpointing, and persistent interaction counter
- Human approval required before deleting a task; execution resumes via `interrupt` + `Command(resume=...)`
- Local `tasks.json` storage, excluded from Git

## Requirements
- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- Qwen3 model installed: `ollama pull qwen3:1.7b`

## Setup (Windows CMD)
```cmd
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy tasks.example.json tasks.json
.venv\Scripts\python.exe graph_agent.py
```

If `tasks.json` already exists, do not overwrite it. The application creates it on the first added task if absent.

## Example requests
```text
add learn Python to my tasks
show me all my tasks
plan my study day
delete task number 1
```
For deletion, the workflow pauses until you answer `yes` or `no`.

## Architecture
```text
CLI user input
    |
LangGraph router
    |--- Task Manager (Ollama function calling: add/list)
    |--- Planning Agent (Ollama: study plan)
    |--- Prepare Deletion -> Human Approval -> Delete / Cancel
    |
Local data: tasks.json + task_manager_v2.sqlite
```

## Notes
- The Python rule-based router selects which specialized node runs. This is a **multi-agent-style workflow**, not an autonomous AI supervisor.
- The 1.7B model may choose tools unreliably, so simple fallback logic handles common add/list commands.
- This is an educational prototype, not a hosted service.
- Keep `tasks.json` and the SQLite database local because they may contain personal information.
