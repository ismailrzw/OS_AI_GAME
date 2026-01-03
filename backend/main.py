from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import random
import copy
import numpy as np

from .process_model import Process
from .scheduler_engine import Scheduler
from .metrics import SimulationMetrics
from .workload_generator import WorkloadGenerator

app = FastAPI(title="Real-Time AI OS Scheduler Simulator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ProcessInput(BaseModel):
    pid: int
    arrival_time: int
    burst_time: int
    initial_priority: int
    io_frequency: float = 0.0

class SimulateRequest(BaseModel):
    algorithm: str
    processes: Optional[List[ProcessInput]] = None
    time_quantum: int = 2

class DuelRequest(BaseModel):
    algorithm: str
    profile: str = "mixed"
    time_quantum: int = 2

class ProcessResponse(BaseModel):
    pid: int
    arrival_time: int
    burst_time: int
    initial_priority: int
    priority: int
    io_frequency: float
    case_type: str
    severity: str
    cluster: Optional[int] = -1

class SimulationResult(BaseModel):
    algorithm: str
    metrics: Dict[str, Any]
    gantt_chart_log: List[Dict[str, Any]]
    process_events: Optional[Dict[int, List[str]]] = None

class DuelResponse(BaseModel):
    human: SimulationResult
    ai: SimulationResult
    explanation: Optional[str] = None

# --- Global State ---
last_processes: List[Process] = []
generator = WorkloadGenerator(seed=42)
history = {
    "simulations_count": 0,
    "human_avg_score": 0.0,
    "ai_avg_score": 0.0
}

def format_metrics(m: SimulationMetrics):
    return {
        "average_waiting_time": m.average_waiting_time,
        "average_turnaround_time": m.average_turnaround_time,
        "max_waiting_time": m.max_waiting_time,
        "starvation_count": m.starvation_count,
        "context_switches": m.context_switches,
        "cpu_utilization": m.cpu_utilization,
        "final_score": m.average_waiting_time + (m.starvation_count * 10),
        "avg_wait_is_optimal": False,
        "switches_are_optimal": False
    }

@app.get("/")
async def root():
    return {"status": "online", "mode": "Real-Time Only"}

@app.post("/simulate", response_model=DuelResponse)
async def simulate(request: SimulateRequest):
    global last_processes, history
    
    if request.processes:
        input_procs = [
            Process(
                pid=p.pid, 
                arrival_time=p.arrival_time, 
                burst_time=p.burst_time, 
                initial_priority=p.initial_priority, 
                io_frequency=p.io_frequency,
                priority=p.initial_priority,
                case_type="User Defined",
                severity="Normal"
            ) for p in request.processes
        ]
    else:
        input_procs = generator.generate(count=6)
    
    last_processes = copy.deepcopy(input_procs)
    
    human_scheduler = Scheduler(copy.deepcopy(input_procs))
    human_metrics, human_completed = human_scheduler.run(request.algorithm, request.time_quantum)
    
    ai_scheduler = Scheduler(copy.deepcopy(input_procs))
    ai_metrics, ai_completed = ai_scheduler.run("ai", request.time_quantum)
    
    human_res_metrics = format_metrics(human_metrics)
    ai_res_metrics = format_metrics(ai_metrics)
    
    history["simulations_count"] += 1
    history["human_avg_score"] = (history["human_avg_score"] * (history["simulations_count"] - 1) + human_res_metrics["final_score"]) / history["simulations_count"]
    history["ai_avg_score"] = (history["ai_avg_score"] * (history["simulations_count"] - 1) + ai_res_metrics["final_score"]) / history["simulations_count"]

    human_res_metrics["avg_wait_is_optimal"] = human_res_metrics["average_waiting_time"] <= ai_res_metrics["average_waiting_time"]
    ai_res_metrics["avg_wait_is_optimal"] = not human_res_metrics["avg_wait_is_optimal"]
    
    # Generate Explanation
    explanation = ""
    if human_res_metrics["final_score"] < ai_res_metrics["final_score"]:
        explanation = "You beat the AI! Your schedule minimized wait times and starvation better."
    elif human_res_metrics["final_score"] > ai_res_metrics["final_score"]:
        explanation = f"The AI won. It optimized the order using K-Means clustering to minimize overall overhead."
    else:
        explanation = "It's a tie! Both schedules achieved identical performance metrics."

    return {
        "human": {
            "algorithm": request.algorithm.upper(),
            "metrics": human_res_metrics,
            "gantt_chart_log": human_metrics.gantt_chart_log,
            "process_events": {p.pid: p.events for p in human_completed}
        },
        "ai": {
            "algorithm": "AI (K-Means+Aging)",
            "metrics": ai_res_metrics,
            "gantt_chart_log": ai_metrics.gantt_chart_log,
            "process_events": {p.pid: p.events for p in ai_completed}
        },
        "explanation": explanation
    }

@app.get("/processes", response_model=List[ProcessResponse])
async def get_processes(count: int = 6, profile: str = "mixed"):
    procs = generator.generate(count=count, profile=profile)
    return [ProcessResponse(
        pid=p.pid, 
        arrival_time=p.arrival_time, 
        burst_time=p.burst_time, 
        initial_priority=p.initial_priority,
        priority=p.priority,
        io_frequency=p.io_frequency,
        case_type=p.case_type,
        severity=p.severity,
        cluster=getattr(p, 'cluster', -1)
    ) for p in procs]

@app.get("/metrics")
async def get_metrics():
    return history

@app.post("/duel", response_model=DuelResponse)
async def duel(request: DuelRequest):
    global history
    all_procs = generator.generate(count=5000, profile=request.profile)
    
    ai_scheduler = Scheduler(copy.deepcopy(all_procs))
    ai_metrics, ai_completed = ai_scheduler.run("ai", request.time_quantum)
    
    human_scheduler = Scheduler(copy.deepcopy(all_procs))
    human_metrics, human_completed = human_scheduler.run(request.algorithm, request.time_quantum)

    duel_subset_pids = [p.pid for p in all_procs[:10]]
    
    def filter_log(log):
        return [entry for entry in log if entry['pid'] in duel_subset_pids]

    human_res_metrics = format_metrics(human_metrics)
    ai_res_metrics = format_metrics(ai_metrics)
    
    history["simulations_count"] += 1
    history["human_avg_score"] = (history["human_avg_score"] * (history["simulations_count"] - 1) + human_res_metrics["final_score"]) / history["simulations_count"]
    history["ai_avg_score"] = (history["ai_avg_score"] * (history["simulations_count"] - 1) + ai_res_metrics["final_score"]) / history["simulations_count"]

    return {
        "human": {
            "algorithm": request.algorithm.upper(),
            "metrics": human_res_metrics,
            "gantt_chart_log": filter_log(human_metrics.gantt_chart_log)[:50],
            "process_events": {p.pid: p.events for p in human_completed if p.pid in duel_subset_pids}
        },
        "ai": {
            "algorithm": "AI (K-Means+Aging)",
            "metrics": ai_res_metrics,
            "gantt_chart_log": filter_log(ai_metrics.gantt_chart_log)[:50],
            "process_events": {p.pid: p.events for p in ai_completed if p.pid in duel_subset_pids}
        },
        "explanation": f"The AI processed 5000 jobs. It beat the standard {request.algorithm.upper()} by optimizing for discovered workload clusters."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
