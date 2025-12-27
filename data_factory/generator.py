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

def generate_processes(num_processes: int) -> List[Process]:
    """
    Generates a list of Process objects with randomized attributes.
    """
    processes = []
    for i in range(num_processes):
        # Ensure burst time is at least 1 to be a valid process.
        burst = random.randint(1, MAX_BURST_TIME)
        
        # Create a process with a unique PID, random arrival, burst, and priority.
        p = Process(
            pid=i,
            arrival_time=random.randint(0, MAX_ARRIVAL_TIME),
            burst_time=burst,
            priority=random.randint(1, MAX_PRIORITY)
        )
        processes.append(p)
    return processes

def generate_scenarios(num_scenarios: int) -> List[Tuple[str, List[Process]]]:
    """
    Generates a specified number of complete scenarios, each consisting of
    a mode and a list of processes.
    """
    scenarios = []
    for _ in range(num_scenarios):
        # Choose a random number of processes for this scenario
        num_processes = random.randint(MIN_PROCESSES, MAX_PROCESSES)
        
        # Choose a random mode for this scenario
        mode = random.choice(MODES)
        
        # Generate the processes
        processes = generate_processes(num_processes)
        
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
