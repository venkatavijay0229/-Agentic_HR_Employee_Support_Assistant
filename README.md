# Agentic AI HR Employee Support Assistant

FastAPI + Google Gemini agent with mock HR tools. No external HR services are contacted.

## Endpoints
- `GET /health`
- `POST /agent/run`
- `POST /tickets/list` (optional `employee_id` query parameter)

## Setup in VS Code / Windows PowerShell
Open the folder containing `app.py`, then run:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activation is blocked, skip activation and use the venv Python directly:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

## Configure Gemini
1. Get a Google AI Studio API key.
2. Copy `.env.example` to `.env`.
3. Put the real key after `GEMINI_API_KEY=` in `.env`. Never share this file.
4. The default model is `gemini-2.5-flash`.

Gemini powers intent classification, planning, tool routing, and final response when a valid key and quota are available. Without a key or if Gemini calls fail, the app uses deterministic local fallbacks so you can test the API; that fallback is not a substitute for full LLM evaluation.

## Run
```powershell
python -m uvicorn app:app --reload
```
Open http://127.0.0.1:8000/docs. Keep terminal open; stop with Ctrl+C.

## Swagger test bodies

### Leave balance
```json
{"user_query":"I want to apply for 3 days leave. How many days do I have left?","employee_id":"E7742"}
```

### Payroll service status
```json
{"user_query":"Is the payroll self-service portal working right now?","employee_id":"E7742"}
```

### Create reimbursement ticket
```json
{"user_query":"Raise a ticket for a missing reimbursement of Rs. 4,500.","employee_id":"E334"}
```

### Notice period policy
```json
{"user_query":"What is the notice period policy for resignation?","employee_id":"E7742"}
```

### Pending leave
```json
{"user_query":"My leave request from last week is still showing pending. Can you help?","employee_id":"E334"}
```

### General question — should avoid tool calls
```json
{"user_query":"Explain what casual leave means.","employee_id":"E7742"}
```

## Response fields
`intent`, `plan`, `tool_calls` (including args, mock result, timestamp), `final_answer`, `ticket_ref`, `next_question`, `action_log`.

## Notes
- Mock balances, policies and service statuses are demo data.
- Checking leave balance does not submit or approve leave.
- Tickets are in-memory and disappear when the app restarts.
- This is a learning demo, not an official HR system.
- Python 3.12 is generally recommended for broad package compatibility, but requirements avoid FAISS and `uvicorn[standard]` for easier Windows setup.
