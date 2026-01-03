"""
scorer.py

Implements the critical scoring logic for evaluating scheduling algorithms
based on the active system mode.
"""
from typing import List, Dict

from backend.process_model import Process
from backend.metrics import SimulationMetrics

def find_priority_violations(completed_processes: List[Process]) -> int:
    """
    Calculates the number of priority violations for Real-Time mode.
    A violation occurs if a lower-priority process finishes while a higher-priority
    process was already waiting in the ready queue.
    """
    violations = 0
    if not completed_processes:
        return 0
        
    sorted_by_finish = sorted(completed_processes, key=lambda p: p.finish_time)
    
    for i, current_p in enumerate(sorted_by_finish):
        for other_p in completed_processes:
            if (other_p.pid != current_p.pid and
                other_p.priority < current_p.priority and
                other_p.arrival_time < current_p.finish_time and
                other_p.finish_time > current_p.finish_time):
                violations += 1
                
    return violations // 2

def calculate_total_cost(
    weights: Dict[str, float], 
    metrics: SimulationMetrics,
    completed_processes: List[Process]
) -> float:
    """
    Calculates a total cost for a simulation run based on a weighted sum of metrics.
    """
    if not metrics.completed_processes_count > 0:
        return float('inf')

    # The core weighted cost function
    total_cost = (
        weights['efficiency_weight'] * metrics.average_waiting_time +
        weights['fairness_weight'] * metrics.waiting_time_variance +
        weights['starvation_weight'] * metrics.max_waiting_time +
        weights['context_switch_penalty'] * metrics.context_switches +
        weights['preemption_cost'] * metrics.preemption_count
    )

    # Add a massive, non-linear penalty for priority violations
    violations = find_priority_violations(completed_processes)
    total_cost += violations * 1000 

    return total_cost
