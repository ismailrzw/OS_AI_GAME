"""
scorer.py

Implements the critical scoring logic for evaluating scheduling algorithms
based on the active system mode.
"""
from typing import List

from backend.process_model import Process
from backend.metrics import SimulationMetrics

def find_priority_violations(completed_processes: List[Process]) -> int:
    """
    Calculates the number of priority violations for Real-Time mode.
    
    A violation occurs if a lower-priority process finishes while a higher-priority
    process was already waiting in the ready queue.
    
    Args:
        completed_processes: A list of processes that have finished execution,
                             containing their final scheduling metrics.
                             
    Returns:
        The total count of priority violations.
    """
    violations = 0
    # Sort processes by their finish time to iterate in execution order
    sorted_by_finish = sorted(completed_processes, key=lambda p: p.finish_time)
    
    for i, current_p in enumerate(sorted_by_finish):
        # For each process, check all other processes to see if a rule was broken.
        for other_p in completed_processes:
            # --- Violation Condition ---
            # A higher priority process ('other_p') should not finish after a
            # lower priority process ('current_p') if it had already arrived
            # when the lower priority one finished.
            if (other_p.pid != current_p.pid and
                other_p.priority < current_p.priority and      # 'other_p' has higher priority
                other_p.arrival_time < current_p.finish_time and # 'other_p' was available
                other_p.finish_time > current_p.finish_time):    # but it finished later
                violations += 1
                
    # Since we check for every pair, each violation is double-counted. We divide by 2.
    return violations // 2

def calculate_score(mode: str, metrics: SimulationMetrics, completed_processes: List[Process]) -> float:
    """
    Calculates a final score for a simulation run based on the mode's rules.
    The goal is always to MINIMIZE this score.
    """
    if not completed_processes:
        return float('inf') # An invalid run gets the worst possible score

    score = 0.0

    if mode == 'Efficiency':
        # --- Efficiency Rule ---
        # Score is the average waiting time plus a penalty for each context switch.
        # This heavily penalizes preemptive algorithms.
        score = metrics.average_waiting_time + (metrics.context_switches * 0.5)

    elif mode == 'Fairness':
        # --- Fairness Rule ---
        # Score is based on the maximum waiting time to prevent starvation.
        max_wait_time = 0
        if completed_processes:
            max_wait_time = max(p.waiting_time for p in completed_processes)
        
        score = max_wait_time
        
        # Apply a massive penalty if any process had to wait too long.
        if max_wait_time > 15:
            score += 100

    elif mode == 'Real-Time':
        # --- Real-Time Rule ---
        # The base score is average waiting time, but priority is king.
        violations = find_priority_violations(completed_processes)
        score = metrics.average_waiting_time + (violations * 200)

    return score
