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

MODES = ['Efficiency', 'Fairness', 'Real-Time']

# --- Profile-based Process Generation ---
# To improve dataset quality, we generate workloads with specific characteristics
# that favor different algorithms. This helps the model learn more distinct patterns.

def generate_sjf_friendly(num_processes: int) -> List[Process]:
    """Generates a workload with high variance in burst times, favoring SJF/SRTF."""
    processes = []
    for i in range(num_processes):
        # High variance: mix of very short and very long jobs
        burst = random.choice([random.randint(1, 4), random.randint(15, MAX_BURST_TIME)])
        processes.append(Process(
            pid=i,
            arrival_time=random.randint(0, MAX_ARRIVAL_TIME // 2), # Tend to arrive in a cluster
            burst_time=burst,
            priority=random.randint(1, MAX_PRIORITY)
        ))
    return processes

def generate_priority_friendly(num_processes: int) -> List[Process]:
    """Generates a workload with a wide and distinct range of priorities."""
    processes = []
    priorities = random.sample(range(1, MAX_PRIORITY + 1), min(num_processes, MAX_PRIORITY))
    for i in range(num_processes):
        processes.append(Process(
            pid=i,
            arrival_time=random.randint(0, MAX_ARRIVAL_TIME),
            burst_time=random.randint(5, 15), # Medium burst times
            priority=priorities[i % len(priorities)]
        ))
    return processes

def generate_rr_friendly(num_processes: int) -> List[Process]:
    """Generates a workload with similar burst times arriving together, favoring Round Robin."""
    processes = []
    base_burst = random.randint(8, 12)
    for i in range(num_processes):
        processes.append(Process(
            pid=i,
            arrival_time=random.randint(0, 2), # Arrive very close together
            burst_time=base_burst + random.randint(-2, 2), # Similar burst times
            priority=random.randint(2, 4) # Medium priorities
        ))
    return processes

def generate_fcfs_friendly(num_processes: int) -> List[Process]:
    """Generates a workload where processes arrive spread out, favoring FCFS."""
    processes = []
    arrival_time = 0
    for i in range(num_processes):
        arrival_time += random.randint(3, 8) # Arrive sequentially
        processes.append(Process(
            pid=i,
            arrival_time=arrival_time,
            burst_time=random.randint(5, 15),
            priority=random.randint(1, MAX_PRIORITY)
        ))
    return processes

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
    Generates a specified number of complete scenarios, cycling through the
    different workload profiles to ensure a balanced and high-quality dataset.
    """
    scenarios = []
    # List of generator functions to cycle through for non-Real-Time modes
    profile_generators = [
        generate_sjf_friendly,
        generate_priority_friendly,
        generate_rr_friendly,
        generate_fcfs_friendly
    ]
    
    for i in range(num_scenarios):
        num_processes = random.randint(MIN_PROCESSES, MAX_PROCESSES)
        mode = random.choice(MODES)
        
        if mode == 'Real-Time':
            processes = generate_medical_cases(num_processes)
        else:
            # Cycle through the standard profile generators
            generator_func = profile_generators[i % len(profile_generators)]
            processes = generator_func(num_processes)
        
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
