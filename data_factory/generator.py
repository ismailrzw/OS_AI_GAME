"""
generator.py

Generates synthetic process and scenario data for training the AI scheduler.
"""
import random
from typing import List, Tuple

# We need to import the Process model from the backend logic
from backend.process_model import Process

# --- Constants for Data Generation ---
# These control the "shape" of the workloads we generate.
# Forcing diversity in the training data is key to a robust model.
MIN_PROCESSES = 4
MAX_PROCESSES = 8
MAX_ARRIVAL_TIME = 10
MAX_BURST_TIME = 20
MAX_PRIORITY = 5 # Lower number = higher priority

MODES = ['Real-Time']

# --- Medical Case Data for Real-Time Mode ---
MEDICAL_CASES = {
    'Stroke': {'severity': 'Critical', 'priority': 1, 'duration': (15, 20)},
    'Heart Attack': {'severity': 'Critical', 'priority': 1, 'duration': (15, 20)},
    'Trauma': {'severity': 'High', 'priority': 2, 'duration': (10, 18)},
    'Asthma Attack': {'severity': 'High', 'priority': 2, 'duration': (8, 12)},
    'Infection': {'severity': 'Medium', 'priority': 3, 'duration': (5, 10)},
    'Broken Bone': {'severity': 'Medium', 'priority': 3, 'duration': (8, 15)},
    'Fever': {'severity': 'Low', 'priority': 4, 'duration': (3, 6)},
    'Migraine': {'severity': 'Low', 'priority': 4, 'duration': (2, 5)},
}

def generate_medical_cases(num_processes: int) -> List[Process]:
    """Generates a list of processes themed as medical cases for Real-Time mode."""
    processes = []
    case_types = random.choices(list(MEDICAL_CASES.keys()), k=num_processes)
    
    for i, case_type in enumerate(case_types):
        case_data = MEDICAL_CASES[case_type]
        min_duration, max_duration = case_data['duration']
        
        p = Process(
            pid=i,
            arrival_time=random.randint(0, MAX_ARRIVAL_TIME),
            burst_time=random.randint(min_duration, max_duration),
            priority=case_data['priority'],
            case_type=case_type,
            severity=case_data['severity']
        )
        processes.append(p)
    return processes


def generate_scenarios(num_scenarios: int) -> List[Tuple[str, List[Process]]]:
    """
    Generates a specified number of complete scenarios for Real-Time mode.
    """
    scenarios = []
    for _ in range(num_scenarios):
        num_processes = random.randint(MIN_PROCESSES, MAX_PROCESSES)
        mode = 'Real-Time'
        processes = generate_medical_cases(num_processes)
        scenarios.append((mode, processes))
        
    return scenarios


if __name__ == '__main__':
    # Example of how to use the generator
    print("--- Generating a single example scenario ---")
    example_scenarios = generate_scenarios(1)
    mode, processes = example_scenarios[0]
    print(f"Mode: {mode}")
    for proc in processes:
        print(proc)
