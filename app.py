
import json, logging, os, random, re
from datetime import datetime, timezone
from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("hr_agent")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_ITERATIONS = 3
app = FastAPI(title="Agentic AI HR Employee Support Assistant", version="1.0.0",
              description="Gemini-powered HR agent with mock Python tools.")

LEAVES = {
    "E7742": {"employee_id":"E7742","annual_leave":8,"sick_leave":5,"pending_requests":0},
    "E-7742": {"employee_id":"E-7742","annual_leave":8,"sick_leave":5,"pending_requests":0},
    "E334": {"employee_id":"E334","annual_leave":12,"sick_leave":6,"pending_requests":1},
    "E-334": {"employee_id":"E-334","annual_leave":12,"sick_leave":6,"pending_requests":1},
}
POLICIES = {
    "remote work": {"policy":"Remote Work","summary":"Employees may work remotely up to 2 days per week with manager approval.","effective":"2025-01-01"},
    "work from home": {"policy":"Remote Work","summary":"Employees may work remotely up to 2 days per week with manager approval.","effective":"2025-01-01"},
    "wfh": {"policy":"Remote Work","summary":"Employees may work remotely up to 2 days per week with manager approval.","effective":"2025-01-01"},
    "notice period": {"policy":"Notice Period","summary":"The standard notice period is 30 days, subject to the employee's contract and HR approval.","effective":"2025-01-01"},
    "resignation": {"policy":"Notice Period","summary":"The standard notice period is 30 days, subject to the employee's contract and HR approval.","effective":"2025-01-01"},
    "casual leave": {"policy":"Casual Leave","summary":"Casual leave is short-duration leave for personal matters, subject to the organization's approval process and available balance.","effective":"2025-01-01"},
    "payroll": {"policy":"Payroll","summary":"Payroll discrepancies should be reported to HR with the pay period and a brief description of the issue.","effective":"2025-01-01"},
}
SERVICES = {
    "payroll_portal":{"service":"payroll_portal","status":"operational","last_incident":"none"},
    "hr_portal":{"service":"hr_portal","status":"operational","last_incident":"none"},
    "leave_portal":{"service":"leave_portal","status":"degraded","last_incident":"Intermittent delays in the demo environment"},
    "reimbursement_portal":{"service":"reimbursement_portal","status":"operational","last_incident":"none"},
}
TICKETS = []

def stamp():
    return datetime.now(timezone.utc).isoformat()

def check_leave_balance(employee_id: str) -> dict:
    item = LEAVES.get(employee_id.strip()) or LEAVES.get(employee_id.strip().upper())
    return {"found": bool(item), **(item or {"employee_id":employee_id,"message":"No mock leave record found."})}

def raise_hr_ticket(category: str, description: str, employee_id: str) -> dict:
    item = {"ticket_ref":f"HR-{random.randint(10000,99999)}","category":category,
            "description":description,"employee_id":employee_id,"status":"open",
            "eta":"2 business days","created_at":stamp()}
    TICKETS.append(item)
    return item

def get_policy_info(policy_name: str) -> dict:
    q = policy_name.lower().strip()
    for key, value in POLICIES.items():
        if key in q or q in key:
            return {"found":True, **value}
    return {"found":False,"policy":policy_name,"message":"No matching mock policy was found."}

def check_service_status(service_name: str) -> dict:
    q = service_name.lower().replace(" ","_").replace("-","_")
    aliases = {"payroll":"payroll_portal","payroll_self_service_portal":"payroll_portal",
               "payroll_self_service":"payroll_portal","hr":"hr_portal","leave":"leave_portal",
               "reimbursement":"reimbursement_portal"}
    q = aliases.get(q,q)
    return SERVICES.get(q, {"service":q,"status":"unknown","last_incident":"No mock status configured"})

TOOLS = {"check_leave_balance":check_leave_balance,"raise_hr_ticket":raise_hr_ticket,
         "get_policy_info":get_policy_info,"check_service_status":check_service_status}

class RunRequest(BaseModel):
    user_query: str = Field(..., min_length=2)
    employee_id: str = Field(..., min_length=2)

class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any]
    result: dict[str, Any]
    timestamp: str

class RunResponse(BaseModel):
    intent: str
    plan: list[str]
    tool_calls: list[ToolCall]
    final_answer: str
    ticket_ref: str | None = None
    next_question: str | None = None
    action_log: list[dict[str, Any]]

def ask_gemini(prompt: str):
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key or genai is None:
        return None
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=MODEL, contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1))
        raw = (response.text or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception as exc:
        logger.warning("Gemini failed; using local fallback: %s", exc)
        return None

def classify(q):
    result = ask_gemini('Return ONLY JSON {"intent":"snake_case"}. Classify this HR query: '+q)
    if result and result.get("intent"):
        return str(result["intent"])
    s=q.lower()
    if any(x in s for x in ["leave balance","days of leave","days leave","leave request","pending leave","apply for"]): return "leave_assistance"
    if any(x in s for x in ["ticket","reimbursement","payroll discrepancy","missing payment","not received","discrepancy"]): return "hr_ticket"
    if any(x in s for x in ["portal","down","working right now","service status","operational"]): return "service_status"
    if any(x in s for x in ["policy","notice period","remote work","work from home","wfh","casual leave","resignation"]): return "policy_question"
    return "general_hr_question"

def fallback_plan(intent):
    plans = {
      "leave_assistance":["Identify employee and leave question","Check the mock leave balance","Review pending-request information","Explain the balance and ask for dates if needed"],
      "hr_ticket":["Identify the HR issue and employee","Check whether ticket details are available","Create an HR ticket if requested","Return the ticket reference and expected response time"],
      "service_status":["Identify the HR service","Check mock service health","Summarize current status and incident note"],
      "policy_question":["Identify the policy","Retrieve matching mock policy","Summarize the policy and any caveats"],
      "general_hr_question":["Understand the employee's question","Answer without unnecessary tool calls","Ask a follow-up question if the request is ambiguous"]}
    return plans.get(intent, plans["general_hr_question"])

def plan_for(q,intent):
    r=ask_gemini(f'Return ONLY JSON {{"plan":["step 1","step 2","step 3"]}} with 3 to 5 short steps. Do not claim actions have occurred. Intent={intent}; query={q}')
    p=r.get("plan") if r else None
    if isinstance(p,list) and 3 <= len(p) <= 5: return [str(x) for x in p]
    return fallback_plan(intent)

def local_route(q,eid,history,intent):
    s=q.lower()
    if history:
        return {"action":"final","final_answer":"The relevant mock HR lookup has completed."}
    if intent=="leave_assistance":
        return {"action":"tool_call","tool_name":"check_leave_balance","arguments":{"employee_id":eid}}
    if intent=="hr_ticket":
        explicit=any(x in s for x in ["raise a ticket","raise ticket","create a ticket","log a ticket","please raise","raise an hr ticket"])
        if explicit:
            cat="reimbursement" if "reimbursement" in s else ("payroll" if "payroll" in s or "salary" in s else "general_hr")
            return {"action":"tool_call","tool_name":"raise_hr_ticket","arguments":{"category":cat,"description":q,"employee_id":eid}}
        return {"action":"final","final_answer":"I can help send this to HR. Would you like me to raise a ticket?","next_question":"Would you like me to raise an HR ticket?"}
    if intent=="service_status":
        name="payroll_portal" if "payroll" in s else ("leave_portal" if "leave" in s else ("reimbursement_portal" if "reimbursement" in s else "hr_portal"))
        return {"action":"tool_call","tool_name":"check_service_status","arguments":{"service_name":name}}
    if intent=="policy_question":
        if "casual leave" in s and not any(x in s for x in ["company policy","policy details","official policy"]):
            return {"action":"final","final_answer":"Casual leave is generally short-duration leave for personal matters. Exact entitlement and approval depend on your organization's policy."}
        name="notice period" if "notice" in s or "resignation" in s else ("remote work" if any(x in s for x in ["remote","work from home","wfh"]) else ("casual leave" if "casual leave" in s else "payroll"))
        return {"action":"tool_call","tool_name":"get_policy_info","arguments":{"policy_name":name}}
    if "casual leave" in s:
        return {"action":"final","final_answer":"Casual leave is generally short-duration leave for personal matters, subject to your organization's approval process."}
    return {"action":"final","final_answer":"I can help with leave balances, HR tickets, HR policies, and HR portal status. What would you like help with?","next_question":"What HR issue would you like help with?"}

def route(q,eid,history,intent):
    prompt=f'''You are an HR tool router. Return exactly one JSON object.
Allowed tools: check_leave_balance(employee_id), raise_hr_ticket(category,description,employee_id), get_policy_info(policy_name), check_service_status(service_name).
Return either {{"action":"tool_call","tool_name":"allowed_tool","arguments":{{}}}} or {{"action":"final","final_answer":"...","next_question":null}}.
Use tools only when helpful. A general question such as "Explain what casual leave means" should not call a tool. Only create tickets when explicitly requested. Use employee_id from request. Do not repeat a tool that already answered the question. Max total tool calls: 3.
Employee ID={eid}; intent={intent}; query={q}; previous tool results={json.dumps(history)}'''
    r=ask_gemini(prompt)
    if r and r.get("action")=="final": return r
    if r and r.get("action")=="tool_call" and r.get("tool_name") in TOOLS: return r
    return local_route(q,eid,history,intent)

def final_answer(q,eid,intent,history,router_final):
    r=ask_gemini(f'''You are the final-response component of an HR agent. Use only mock tool results. Never say leave was submitted/approved because this tool only checks balance. Mention action taken and ticket ref. Ask at most one follow-up if needed. Return ONLY JSON {{"final_answer":"...","next_question":null}}.
Query={q}; employee_id={eid}; intent={intent}; tool results={json.dumps(history)}; router draft={json.dumps(router_final or {})}''')
    if r and isinstance(r.get("final_answer"),str):
        return r["final_answer"],r.get("next_question")
    if not history:
        return ((router_final or {}).get("final_answer","I can help with HR questions."),(router_final or {}).get("next_question"))
    t=history[-1]["tool"]; d=history[-1]["result"]
    if t=="check_leave_balance":
        if not d.get("found"): return d.get("message","No record found."),"Please confirm your employee ID."
        follow="What dates do you plan to take leave?" if any(x in q.lower() for x in ["apply","next week","days leave"]) else None
        return f"Mock leave balance for {eid}: {d['annual_leave']} annual leave days and {d['sick_leave']} sick leave days. Pending requests: {d['pending_requests']}. This lookup does not submit or approve leave.",follow
    if t=="raise_hr_ticket": return f"Your HR ticket has been created. Reference: {d['ticket_ref']}. Status: {d['status']}. Expected response: {d['eta']}.",None
    if t=="get_policy_info":
        if d.get("found"): return f"{d['policy']} policy (demo): {d['summary']} Effective: {d['effective']}. Please confirm details with HR.",None
        return d.get("message","No matching policy found."),"Which policy do you mean?"
    if t=="check_service_status": return f"Mock status for {d['service']}: {d['status']}. Last incident: {d['last_incident']}.",None
    return "The mock lookup completed.",None

@app.get("/")
def root(): return {"message":"Agentic HR Employee Support Assistant","docs":"/docs"}

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/agent/run", response_model=RunResponse)
def run_agent(req: RunRequest):
    q=req.user_query.strip(); eid=req.employee_id.strip()
    intent=classify(q); plan=plan_for(q,intent)
    history=[]; logs=[{"event":"plan_created","timestamp":stamp(),"detail":f"Plan created for intent {intent}"}]
    router_final=None; ticket_ref=None
    for i in range(MAX_ITERATIONS):
        decision=route(q,eid,history,intent)
        if decision.get("action")=="final":
            router_final=decision
            logs.append({"event":"final_selected","timestamp":stamp(),"iteration":i+1})
            break
        tool=decision.get("tool_name")
        if tool not in TOOLS:
            router_final={"action":"final","final_answer":"I could not select a supported HR action."}
            break
        args=dict(decision.get("arguments") or {})
        if tool=="check_leave_balance": args={"employee_id":eid}
        elif tool=="raise_hr_ticket": args={"category":str(args.get("category") or "general_hr")[:80],"description":str(args.get("description") or q)[:1000],"employee_id":eid}
        elif tool=="get_policy_info": args={"policy_name":str(args.get("policy_name") or "general HR")[:120]}
        elif tool=="check_service_status": args={"service_name":str(args.get("service_name") or "hr_portal")[:120]}
        if any(x["tool"]==tool and x["args"]==args for x in history):
            router_final={"action":"final","final_answer":"I already completed the relevant lookup."}
            logs.append({"event":"duplicate_tool_prevented","timestamp":stamp(),"tool":tool})
            break
        when=stamp()
        try: result=TOOLS[tool](**args)
        except Exception as exc: result={"error":str(exc)}
        history.append({"tool":tool,"args":args,"result":result,"timestamp":when})
        logs.append({"event":"tool_executed","timestamp":when,"iteration":i+1,"tool":tool,"args":args})
        if tool=="raise_hr_ticket": ticket_ref=result.get("ticket_ref")
        if i==MAX_ITERATIONS-1: logs.append({"event":"max_iterations_reached","timestamp":stamp(),"max_iterations":MAX_ITERATIONS})
    answer,follow=final_answer(q,eid,intent,history,router_final)
    if ticket_ref and ticket_ref not in answer: answer+=f" Ticket reference: {ticket_ref}."
    return RunResponse(intent=intent,plan=plan,tool_calls=[ToolCall(**x) for x in history],
        final_answer=answer,ticket_ref=ticket_ref,next_question=follow,action_log=logs)

@app.post("/tickets/list")
def list_tickets(employee_id: str | None = None):
    rows=TICKETS if not employee_id else [t for t in TICKETS if t["employee_id"].lower()==employee_id.strip().lower()]
    return {"count":len(rows),"tickets":rows}

if __name__=="__main__":
    import uvicorn
    uvicorn.run("app:app",host="127.0.0.1",port=8000,reload=True)
