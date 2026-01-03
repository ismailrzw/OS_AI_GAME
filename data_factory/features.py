"""
features.py

Defines the feature extraction logic and cost function weights for the AI model.
Separating this from the training script makes the components more modular.
"""
import numpy as np
from typing import List, Dict, Any
from backend.process_model import Process

# --- ALGORITHM ENCODING ---
# The model needs a numeric representation for each algorithm.
ALGORITHM_ENCODING = {
    'fcfs': 0,
    'sjf': 1,
    'srtf': 2,
    'rr': 3,
    'priority': 4
}

# --- COST FUNCTION WEIGHTS ---
# These weights define the "priorities" of each mode. This is the core
# of how the AI will adapt its strategy.
#
# total_cost =
#   efficiency_weight * average_waiting_time
# + fairness_weight * waiting_time_variance
# + starvation_weight * maximum_waiting_time
# + context_switch_penalty * context_switch_count
# + preemption_cost * preemption_count

COST_WEIGHTS = {
    'Real-Time': {
        'efficiency_weight': 0.5,      # Inefficiency is bad, but not the primary concern.
        'fairness_weight': 0.2,        # Low concern for overall fairness.
        'starvation_weight': 5.0,        # Massive penalty for delaying a critical case.
        'context_switch_penalty': 1.0,
        'preemption_cost': 0.5,        # Preemption is acceptable if it serves a critical case.
        # Note: Priority violations (e.g., treating a fever before a heart attack)
        # are handled separately in the scorer with a massive, non-linear penalty.
    }
}

def extract_features_from_scenario(processes: List[Process], mode: str, algorithm: str) -> Dict[str, Any]:
    """
    Extracts features from a list of processes and the scenario context.
    
    New Features:
    - burst_variance: The variance of burst times. More descriptive than std dev alone.
    - arrival_staggering: Measures how spread out process arrivals are.
    - Mode-based weights: The cost function weights themselves are now features.
    - algorithm_id: A numeric ID for the algorithm being considered.
    """
    num_processes = len(processes)
    burst_times = [p.burst_time for p in processes]
    arrival_times = [p.arrival_time for p in processes]
    
    # Get the cost weights for the current mode
    weights = COST_WEIGHTS[mode]

    features = {
        'avg_burst_time': np.mean(burst_times) if burst_times else 0,
        'burst_variance': np.var(burst_times) if burst_times else 0,
        'arrival_staggering': np.std(arrival_times) if arrival_times else 0,
        'number_of_processes': num_processes,
        
        # Include weights as features, so the model learns their impact
        'context_switch_penalty': weights['context_switch_penalty'],
        'preemption_cost': weights['preemption_cost'],
        'efficiency_weight': weights['efficiency_weight'],
        'fairness_weight': weights['fairness_weight'],
        'starvation_weight': weights['starvation_weight'],
        
        # The algorithm itself is a feature
        'algorithm_id': ALGORITHM_ENCODING[algorithm],
    }
    return features
