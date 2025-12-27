# Gemini Project Context: Adaptive OS Scheduler

This file contains the full source code and structure of the project to provide context for future sessions with the Gemini CLI.

---

## FILE: backend/metrics.py

```python
"""
metrics.py

Defines a data structure for collecting and reporting simulation performance metrics.
"""
from typing import List, Dict, Any

class SimulationMetrics:
    """
    A container for all statistics gathered during a single scheduler simulation run.
    """
    def __init__(self):
        # A log of every time a process or a slice of a process runs on the CPU.
        # This is essential for building a Gantt chart for visualization.
        # Format: [{'pid': int, 'start': int, 'end': int}]
        self.gantt_chart_log: List[Dict[str, Any]] = []

        # The total number of times the CPU switched from one process to another.
        # A key metric for evaluating scheduler overhead, especially in "Efficiency" mode.
        self.context_switches: int = 0

        # Final calculated metrics for completed processes.
        self.total_waiting_time: int = 0
        self.total_turnaround_time: int = 0
        self.cpu_utilization: float = 0.0
        self.completed_processes_count: int = 0
    
    def add_gantt_entry(self, pid: int, start_time: int, end_time: int):
        """Records a process execution slice for the Gantt chart."""
        if start_time < end_time: # Only log actual work
            self.gantt_chart_log.append({'pid': pid, 'start': start_time, 'end': end_time})

    def increment_context_switches(self):
        """Increments the context switch counter."""
        self.context_switches += 1

    def finalize(self, completed_processes: List['Process'], total_time: int):
        """
        Calculates final summary metrics after the simulation has completed.
        This should be called once at the very end.
        """
        if not completed_processes:
            return

        self.completed_processes_count = len(completed_processes)
        
        total_burst_time = 0
        for p in completed_processes:
            # Turnaround Time = Finish Time - Arrival Time
            turnaround_time = p.finish_time - p.arrival_time
            self.total_turnaround_time += turnaround_time
            
            # Waiting Time = Turnaround Time - Burst Time
            p.waiting_time = turnaround_time - p.burst_time
            self.total_waiting_time += p.waiting_time
            
            total_burst_time += p.burst_time

        # CPU utilization = (Total time CPU was busy) / (Total simulation time)
        if total_time > 0:
            self.cpu_utilization = (total_burst_time / total_time) * 100

    @property
    def average_waiting_time(self) -> float:
        """Calculates the average waiting time for all completed processes."""
        if self.completed_processes_count == 0:
            return 0.0
        return self.total_waiting_time / self.completed_processes_count

    @property
    def average_turnaround_time(self) -> float:
        """Calculates the average turnaround time for all completed processes."""
        if self.completed_processes_count == 0:
            return 0.0
        return self.total_turnaround_time / self.completed_processes_count
        
    def __repr__(self) -> str:
        return (
            f"Metrics(Avg Wait: {self.average_waiting_time:.2f}, "
            f"Avg Turnaround: {self.average_turnaround_time:.2f}, "
            f"Context Switches: {self.context_switches}, "
            f"CPU Utilization: {self.cpu_utilization:.2f}%)"
        )
```

---

## FILE: backend/process_model.py

```python
"""
process_model.py

Defines the data structure for a single process in the simulation.
Using a dataclass for a clean, readable, and boilerplate-free representation.
"""
from dataclasses import dataclass, field

@dataclass
class Process:
    """
    Represents a single process with its scheduling-related attributes.
    
    Attributes:
        pid (int): A unique identifier for the process.
        arrival_time (int): The simulation time at which the process arrives in the ready queue.
        burst_time (int): The total CPU time required to complete the process.
        priority (int): The priority of the process (lower number means higher priority).
        
        # --- Internal Simulation State ---
        # These fields are managed by the scheduler_engine during the simulation.
        remaining_time (int): CPU time left for the process to complete. Initialized to burst_time.
        start_time (int): The simulation time when the process first starts execution. Initialized to -1.
        finish_time (int): The simulation time when the process completes. Initialized to -1.
        waiting_time (int): Total time the process spent in the ready queue. Calculated at the end.
    """
    pid: int
    arrival_time: int
    burst_time: int
    priority: int = 0  # Default priority if not specified

    # Internal state fields, initialized in __post_init__
    remaining_time: int = field(init=False)
    start_time: int = field(init=False, default=-1)
    finish_time: int = field(init=False, default=-1)
    waiting_time: int = field(init=False, default=0)

    def __post_init__(self):
        """
        Post-initialization hook to set dynamic fields.
        This is a key part of using dataclasses for stateful objects.
        """
        self.remaining_time = self.burst_time

    def __repr__(self) -> str:
        return (
            f"PID({self.pid:02d}) | Arrival: {self.arrival_time:02d} | "
            f"Burst: {self.burst_time:02d} | Prio: {self.priority:02d}"
        )
```

---

## FILE: backend/scheduler_engine.py

```python
"""
scheduler_engine.py

Implements the core simulation logic and the different CPU scheduling algorithms.
This is the "kernel" of the simulation.
"""
import collections
from typing import List, Optional

# Import the data models
from .process_model import Process
from .metrics import SimulationMetrics

class Scheduler:
    """
    The main simulation engine. It takes a set of processes and runs a simulation
    based on a chosen scheduling algorithm.
    """
    def __init__(self, processes: List[Process]):
        # The initial list of all processes for the simulation, sorted by arrival time.
        # This is a critical first step for any time-based simulation.
        self.processes = sorted(processes, key=lambda p: p.arrival_time)
        self.time_quantum = 0

    def run(self, algorithm: str, time_quantum: int = 2) -> (SimulationMetrics, List[Process]):
        """
        Public-facing run method. It selects the appropriate internal method
        based on the algorithm string.
        
        Returns a tuple of (SimulationMetrics, List[Completed_Processes]).
        """
        self.time_quantum = time_quantum

        # --- Design Decision ---
        # A simple factory pattern to select the scheduling algorithm.
        # This is clean and easily extensible for new algorithms.
        if algorithm.lower() == 'fcfs':
            return self._run_non_preemptive(sjf=False)
        elif algorithm.lower() == 'sjf':
            return self._run_non_preemptive(sjf=True)
        elif algorithm.lower() == 'srtf':
            return self._run_preemptive(priority_based=False)
        elif algorithm.lower() == 'priority':
            return self._run_preemptive(priority_based=True)
        elif algorithm.lower() == 'rr':
            return self._run_rr()
        else:
            raise ValueError(f"Unknown scheduling algorithm: {algorithm}")

    def _run_non_preemptive(self, sjf: bool) -> (SimulationMetrics, List[Process]):
        """Handles FCFS and non-preemptive SJF algorithms."""
        time = 0
        ready_queue: List[Process] = []
        process_queue = collections.deque(self.processes)
        completed_processes: List[Process] = []
        metrics = SimulationMetrics()
        current_process: Optional[Process] = None

        while process_queue or ready_queue or current_process:
            # Add newly arrived processes to the ready queue
            while process_queue and process_queue[0].arrival_time <= time:
                ready_queue.append(process_queue.popleft())
            
            # --- Critical Decision: Sorting the ready queue ---
            # For SJF, sort by burst time. For FCFS, it remains sorted by arrival time
            # because we add them in that order and never re-sort.
            if sjf:
                ready_queue.sort(key=lambda p: p.burst_time)

            if not current_process and ready_queue:
                current_process = ready_queue.pop(0)
                if current_process.start_time == -1:
                    current_process.start_time = time
                
                # Non-preemptive: the process will run for its entire burst time.
                execution_start_time = time
                time += current_process.burst_time
                current_process.remaining_time = 0
                current_process.finish_time = time
                
                metrics.add_gantt_entry(current_process.pid, execution_start_time, time)
                completed_processes.append(current_process)
                current_process = None

            elif not ready_queue and not current_process:
                # If no process is running and ready queue is empty, jump time forward.
                time = process_queue[0].arrival_time if process_queue else time + 1

        metrics.finalize(completed_processes, time)
        return metrics, completed_processes

    def _run_preemptive(self, priority_based: bool) -> (SimulationMetrics, List[Process]):
        """Handles SRTF and preemptive Priority scheduling."""
        time = 0
        ready_queue: List[Process] = []
        process_queue = collections.deque(self.processes)
        completed_processes: List[Process] = []
        metrics = SimulationMetrics()
        current_process: Optional[Process] = None

        while process_queue or ready_queue or current_process:
            # Add newly arrived processes to the ready queue
            while process_queue and process_queue[0].arrival_time <= time:
                ready_queue.append(process_queue.popleft())

            # --- Critical Decision: Select next process ---
            # The sorting key determines if we're doing SRTF or Priority scheduling.
            sort_key = lambda p: p.priority if priority_based else p.remaining_time
            ready_queue.sort(key=sort_key)

            # Check for preemption
            if current_process and ready_queue and sort_key(ready_queue[0]) < (sort_key(current_process) if priority_based else current_process.remaining_time):
                metrics.increment_context_switches()
                ready_queue.append(current_process)
                current_process = None

            if not current_process and ready_queue:
                current_process = ready_queue.pop(0)
                if current_process.start_time == -1:
                    current_process.start_time = time

            # Execute one time unit
            if current_process:
                execution_start_time = time
                time += 1
                current_process.remaining_time -= 1
                metrics.add_gantt_entry(current_process.pid, execution_start_time, time)

                if current_process.remaining_time == 0:
                    current_process.finish_time = time
                    completed_processes.append(current_process)
                    current_process = None
            else:
                # No process is ready to run, advance time.
                time = process_queue[0].arrival_time if process_queue else time

        metrics.finalize(completed_processes, time)
        return metrics, completed_processes

    def _run_rr(self) -> (SimulationMetrics, List[Process]):
        """Handles Round Robin scheduling."""
        time = 0
        # Use a deque for efficient append and popleft operations, perfect for RR.
        ready_queue = collections.deque()
        process_queue = collections.deque(self.processes)
        completed_processes: List[Process] = []
        metrics = SimulationMetrics()
        current_process: Optional[Process] = None
        
        last_process_pid = -1

        while process_queue or ready_queue:
            # Add newly arrived processes to the back of the ready queue.
            while process_queue and process_queue[0].arrival_time <= time:
                ready_queue.append(process_queue.popleft())

            if not ready_queue:
                # If no processes are ready, jump time to the next arrival.
                if process_queue:
                    time = process_queue[0].arrival_time
                else:
                    break # No more processes left at all
                continue
            
            current_process = ready_queue.popleft()

            # --- Context Switch Logic ---
            # A context switch occurs if the CPU is handed to a *different* process.
            if last_process_pid != -1 and last_process_pid != current_process.pid:
                metrics.increment_context_switches()

            if current_process.start_time == -1:
                current_process.start_time = time

            execution_start_time = time
            
            # Execute for the time quantum or until the process finishes, whichever is smaller.
            exec_time = min(current_process.remaining_time, self.time_quantum)
            time += exec_time
            current_process.remaining_time -= exec_time
            
            metrics.add_gantt_entry(current_process.pid, execution_start_time, time)
            
            # --- Critical Decision: Process Re-queuing ---
            # Any process that arrives while this one was running must be added to the queue
            # *before* we re-add the current process. This ensures fairness.
            while process_queue and process_queue[0].arrival_time <= time:
                ready_queue.append(process_queue.popleft())

            if current_process.remaining_time > 0:
                # Process is not finished, add it to the back of the queue.
                ready_queue.append(current_process)
                last_process_pid = current_process.pid
            else:
                # Process is finished.
                current_process.finish_time = time
                completed_processes.append(current_process)
                last_process_pid = -1 # No process was running before the next one

        metrics.finalize(completed_processes, time)
        return metrics, completed_processes
```

---

## FILE: data_factory/dataset.csv

```csv
process_count,total_burst_time,avg_burst_time,std_burst_time,avg_priority,priority_range,mode_Efficiency,mode_Fairness,mode_Real-Time,best_algorithm
4,27,6.75,3.2691742076555053,4.25,1,1,0,0,sjf
5,45,9.0,5.89915248150105,2.6,4,0,0,1,priority
8,74,9.25,7.031180555212616,2.0,3,1,0,0,srtf
4,21,5.25,1.6393596310755,3.5,3,0,0,1,fcfs
6,72,12.0,6.0553007081949835,3.0,4,1,0,0,srtf
6,68,11.333333333333334,4.749268949591669,3.1666666666666665,4,0,1,0,sjf
6,50,8.333333333333334,7.086763875156433,3.5,4,1,0,0,srtf
5,56,11.2,5.81033561853358,3.4,4,0,1,0,fcfs
8,75,9.375,6.556247020971678,2.75,4,0,0,1,priority
5,63,12.6,4.223742416388575,2.4,4,1,0,0,srtf
5,38,7.6,6.3118935352238,2.6,3,0,1,0,sjf
6,48,8.0,4.08248290463863,2.8333333333333335,4,0,0,1,priority
6,45,7.5,4.0722639076235385,3.0,4,1,0,0,srtf
4,35,8.75,4.815340071064556,3.5,3,0,1,0,rr
7,84,12.0,5.154748157905347,3.857142857142857,3,0,0,1,priority
5,52,10.4,5.2,3.2,4,0,0,1,priority
4,41,10.25,4.968651728587948,4.25,1,0,1,0,sjf
8,89,11.125,6.112231589198825,2.625,4,0,0,1,priority
7,52,7.428571428571429,5.802884574739972,4.0,3,1,0,0,sjf
8,75,9.375,5.048205126577168,2.875,4,1,0,0,sjf
4,7,1.75,1.299038105676658,1.75,2,0,0,1,fcfs
4,60,15.0,3.5355339059327378,3.25,4,0,0,1,priority
4,30,7.5,4.153311931459037,2.75,4,0,1,0,fcfs
5,42,8.4,5.678027826631356,2.4,3,0,1,0,sjf
4,58,14.5,2.692582403567252,2.5,3,0,1,0,fcfs
8,96,12.0,5.744562646538029,3.375,4,0,0,1,priority
8,119,14.875,2.315032397181517,3.625,3,0,1,0,sjf
5,40,8.0,4.47213595499958,2.2,3,0,1,0,fcfs
8,61,7.625,4.608077147791691,3.375,4,0,1,0,sjf
5,46,9.2,2.85657137141714,2.0,2,0,0,1,priority
...
```

---

## FILE: data_factory/generator.py

```python
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
```

---

## FILE: data_factory/scorer.py

```python
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
```

---

## FILE: data_factory/train.py

```python
"""
train.py

Main machine learning pipeline script. It performs the following steps:
1.  Generates a large synthetic dataset of scheduling scenarios.
2.  For each scenario, it runs all scheduling algorithms to find the optimal one.
3.  Extracts statistical features from the process queue.
4.  Builds a labeled dataset (features -> optimal_algorithm).
5.  Trains a Decision Tree Classifier on this dataset.
6.  Saves the trained model to a file for later use by the backend.
"""
import os
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Import from our project modules
from backend.process_model import Process
from backend.scheduler_engine import Scheduler
from .generator import generate_scenarios
from .scorer import calculate_score

# --- Constants ---
# Using a smaller number for a quick run, but the prompt suggested ~30,000
NUM_SCENARIOS_TO_GENERATE = 5000 
ALGORITHMS_TO_TEST = ['fcfs', 'sjf', 'srtf', 'priority', 'rr']
MODEL_OUTPUT_DIR = 'models'
MODEL_FILE_NAME = 'scheduler_brain.pkl'
DATASET_FILE_NAME = 'data_factory/dataset.csv'

def extract_features(processes: List[Process], mode: str) -> Dict[str, Any]:
    """
    Extracts statistical features from a list of processes and the game mode.
    
    --- Feature Selection Explanation ---
    The goal is to give the Decision Tree meaningful numerical data to create its rules.
    - process_count: Simple but powerful. 5 processes behave differently than 20.
    - total_burst/avg_burst: Is this a long or short job overall?
    - std_burst_time: Are all jobs the same length (low std) or varied (high std)?
      High variance might favor SJF/SRTF.
    - avg_priority: Is this a high or low priority batch?
    - priority_range: Is there a meaningful difference in priorities? If all priorities
      are the same, Priority scheduling is useless.
    - one-hot-encoded-modes: The model needs a numerical way to represent the mode.
      This is the standard way to do it.
    """
    process_count = len(processes)
    burst_times = [p.burst_time for p in processes]
    priorities = [p.priority for p in processes]
    
    features = {
        'process_count': process_count,
        'total_burst_time': sum(burst_times),
        'avg_burst_time': np.mean(burst_times),
        'std_burst_time': np.std(burst_times),
        'avg_priority': np.mean(priorities),
        'priority_range': max(priorities) - min(priorities) if priorities else 0,
        # One-hot encode the mode
        'mode_Efficiency': 1 if mode == 'Efficiency' else 0,
        'mode_Fairness': 1 if mode == 'Fairness' else 0,
        'mode_Real-Time': 1 if mode == 'Real-Time' else 0,
    }
    return features

def find_best_algorithm(processes: List[Process], mode: str) -> str:
    """
    Runs all scheduling algorithms for a given scenario and returns the
    name of the algorithm with the best (lowest) score.
    """
    scores = {}
    
    for algo in ALGORITHMS_TO_TEST:
        # Create a deep copy of processes for each run to avoid side effects
        process_copy = [Process(p.pid, p.arrival_time, p.burst_time, p.priority) for p in processes]
        
        # Run the simulation
        scheduler = Scheduler(process_copy)
        metrics, completed = scheduler.run(algorithm=algo, time_quantum=4) # Using a common quantum for RR
        
        # Calculate the score for this run
        score = calculate_score(mode, metrics, completed)
        scores[algo] = score

    # Return the algorithm name with the minimum score
    best_algo = min(scores, key=scores.get)
    return best_algo

def main():
    """Main training function."""
    print("--- AI Training Pipeline Started ---")
    
    # 1. Generate Scenarios
    print(f"Step 1: Generating {NUM_SCENARIOS_TO_GENERATE} scenarios...")
    scenarios = generate_scenarios(NUM_SCENARIOS_TO_GENERATE)
    
    # 2. Process scenarios and build dataset
    print("Step 2: Running simulations to find optimal algorithms (this may take a while)...")
    dataset = []
    for i, (mode, processes) in enumerate(scenarios):
        if (i + 1) % 500 == 0:
            print(f"  ...processed {i+1}/{NUM_SCENARIOS_TO_GENERATE}")
            
        # Find the best algorithm for this scenario
        best_algo = find_best_algorithm(processes, mode)
        
        # Extract features from the scenario
        features = extract_features(processes, mode)
        
        # Add the label (the winning algorithm) to the record
        features['best_algorithm'] = best_algo
        
        dataset.append(features)
        
    # 3. Create and save DataFrame
    print(f"Step 3: Saving raw dataset to {DATASET_FILE_NAME}...")
    df = pd.DataFrame(dataset)
    df.to_csv(DATASET_FILE_NAME, index=False)
    
    # 4. Model Training
    print("Step 4: Training the Decision Tree model...")
    
    # Define features (X) and target (y)
    features = [col for col in df.columns if col != 'best_algorithm']
    X = df[features]
    y = df['best_algorithm']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"  Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    # Initialize and train the classifier
    # We set max_depth to prevent overfitting and keep the model simpler.
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate Model
    print("Step 5: Evaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Model Accuracy on Test Set: {accuracy * 100:.2f}%")
    
    # 6. Save the trained model
    print("Step 6: Saving the trained model...")
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
    
    model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILE_NAME)
    joblib.dump(model, model_path)
    print(f"  Model successfully saved to: {model_path}")
    
    print("--- AI Training Pipeline Finished ---")

if __name__ == '__main__':
    main()
```

---

## FILE: frontend/.gitignore

```
# Logs
logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*
lerna-debug.log*

node_modules
dist
dist-ssr
*.local

# Editor directories and files
.vscode/*
!.vscode/extensions.json
.idea
.DS_Store
*.suo
*.ntvs*
*.njsproj
*.sln
*.sw?
```

---

## FILE: frontend/README.md

```md
# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
```

---

## FILE: frontend/eslint.config.js

```javascript
import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{js,jsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parserOptions: {
        ecmaVersion: 'latest',
        ecmaFeatures: { jsx: true },
        sourceType: 'module',
      },
    },
    rules: {
      'no-unused-vars': ['error', { varsIgnorePattern: '^[A-Z_]' }],
    },
  },
])
```

---

## FILE: frontend/index.html

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>frontend</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

---

## FILE: frontend/package-lock.json

```json
{
  "name": "frontend",
  "version": "0.0.0",
  "lockfileVersion": 3,
  "requires": true,
  "packages": {
    "": {
      "name": "frontend",
      "version": "0.0.0",
      "dependencies": {
        "react": "^19.2.0",
        "react-dom": "^19.2.0"
      },
      "devDependencies": {
        "@eslint/js": "^9.39.1",
        "@types/react": "^19.2.5",
        "@types/react-dom": "^19.2.3",
        "@vitejs/plugin-react": "^5.1.1",
        "eslint": "^9.39.1",
        "eslint-plugin-react-hooks": "^7.0.1",
        "eslint-plugin-react-refresh": "^0.4.24",
        "globals": "^16.5.0",
        "vite": "^7.2.4"
      }
    },
...
}
```

---

## FILE: frontend/package.json

```json
{
  "name": "frontend",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "lint": "eslint .",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^19.2.0",
    "react-dom": "^19.2.0"
  },
  "devDependencies": {
    "@eslint/js": "^9.39.1",
    "@types/react": "^19.2.5",
    "@types/react-dom": "^19.2.3",
    "@vitejs/plugin-react": "^5.1.1",
    "eslint": "^9.39.1",
    "eslint-plugin-react-hooks": "^7.0.1",
    "eslint-plugin-react-refresh": "^0.4.24",
    "globals": "^16.5.0",
    "vite": "^7.2.4"
  }
}
```

---

## FILE: frontend/public/vite.svg

```xml
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--logos" width="31.88" height="32" preserveAspectRatio="xMidYMid meet" viewBox="0 0 256 257"><defs><linearGradient id="IconifyId1813088fe1fbc01fb466" x1="-.828%" x2="57.636%" y1="7.652%" y2="78.411%"><stop offset="0%" stop-color="#41D1FF"></stop><stop offset="100%" stop-color="#BD34FE"></stop></linearGradient><linearGradient id="IconifyId1813088fe1fbc01fb467" x1="43.376%" x2="50.316%" y1="2.242%" y2="89.03%"><stop offset="0%" stop-color="#FFEA83"></stop><stop offset="8.333%" stop-color="#FFDD35"></stop><stop offset="100%" stop-color="#FFA800"></stop></linearGradient></defs><path fill="url(#IconifyId1813088fe1fbc01fb466)" d="M255.153 37.938L134.897 252.976c-2.483 4.44-8.862 4.466-11.382.048L.875 37.958c-2.746-4.814 1.371-10.646 6.827-9.67l120.385 21.517a6.537 6.537 0 0 0 2.322-.004l117.867-21.483c5.438-.991 9.574 4.796 6.877 9.62Z"></path><path fill="url(#IconifyId1813088fe1fbc01fb467)" d="M185.432.063L96.44 17.501a3.268 3.268 0 0 0-2.634 3.014l-5.474 92.456a3.268 3.268 0 0 0 3.997 3.378l24.777-5.718c2.318-.535 4.413 1.507 3.936 3.838l-7.361 36.047c-.495 2.426 1.782 4.5 4.151 3.78l15.304-4.649c2.372-.72 4.652 1.36 4.15 3.788l-11.698 56.621c-.732 3.542 3.979 5.473 5.943 2.437l1.313-2.028l72.516-144.72c1.215-2.423-.88-5.186-3.54-4.672l-25.505 4.922c-2.396.462-4.435-1.77-3.759-4.114l16.646-57.705c.677-2.35-1.37-4.583-3.769-4.113Z"></path></svg>
```

---

## FILE: frontend/src/App.jsx

```javascript
import React, { useState, useEffect, useCallback } from 'react';

// Import API service and components
import * as api from './services/api';
import ModeSelector from './components/ModeSelector';
import ProcessQueue from './components/ProcessQueue';
import GanttChart from './components/GanttChart';
import Timer from './components/Timer';
import ResultsDashboard from './components/ResultsDashboard';
import NamePopup from './components/NamePopup';
import GameStats from './components/GameStats';
import Leaderboard from './components/Leaderboard';

const initialStats = { wins: 0, losses: 0, ties: 0, total: 0 };

function App() {
  // Core simulation state
  const [mode, setMode] = useState('Efficiency');
  const [processes, setProcesses] = useState([]);
  const [simulationResults, setSimulationResults] = useState(null);
  const [gameState, setGameState] = useState('idle'); // idle, running, finished
  const [timerResetKey, setTimerResetKey] = useState(0);

  // New features state
  const [playerName, setPlayerName] = useState('');
  const [showNamePopup, setShowNamePopup] = useState(false);
  const [playerStats, setPlayerStats] = useState(initialStats);
  const [leaderboard, setLeaderboard] = useState([]);

  // Load player data and leaderboard on initial mount
  useEffect(() => {
    const storedName = localStorage.getItem('playerName');
    if (storedName) {
      setPlayerName(storedName);
      const storedStats = localStorage.getItem('playerStats');
      setPlayerStats(storedStats ? JSON.parse(storedStats) : initialStats);
    } else {
      setShowNamePopup(true);
    }

    const fetchLeaderboard = async () => {
      const data = await api.getLeaderboard();
      setLeaderboard(data);
    };
    fetchLeaderboard();
  }, []);
  
  // Function to start a new game
  const handleStartGame = useCallback(async (selectedMode) => {
    setGameState('running');
    setSimulationResults(null);
    const { processes: newProcesses } = await api.startNewGame(selectedMode);
    setProcesses(newProcesses);
    setGameState('idle');
    setTimerResetKey(prev => prev + 1); // Reset timer
  }, []);
  
  // Effect to start game when mode changes, but only if a player name is set
  useEffect(() => {
    if (playerName) {
      handleStartGame(mode);
    }
  }, [mode, playerName, handleStartGame]);

  const handleModeChange = (newMode) => {
    setMode(newMode);
  };

  const handleSubmitName = (name) => {
    setPlayerName(name);
    localStorage.setItem('playerName', name);
    setShowNamePopup(false);
    handleStartGame(mode); // Start the first game after name is submitted
  };

  const handleSubmitSchedule = useCallback(async () => {
    if (gameState === 'running' || processes.length === 0) return;
    
    setGameState('running');
    const results = await api.submitSchedule(mode, processes);
    setSimulationResults(results);
    setGameState('finished');

    // Update player stats
    const humanScore = results.human.metrics.final_score;
    const aiScore = results.ai.metrics.final_score;
    
    let newStats = { ...playerStats };
    newStats.total += 1;
    if (humanScore < aiScore) {
      newStats.wins += 1;
    } else if (humanScore > aiScore) {
      newStats.losses += 1;
    } else {
      newStats.ties += 1;
    }
    setPlayerStats(newStats);
    localStorage.setItem('playerStats', JSON.stringify(newStats));

  }, [mode, processes, gameState, playerStats]);

  const maxTime = simulationResults 
    ? Math.max(
        simulationResults.human.gantt_chart_log.at(-1)?.end || 0,
        simulationResults.ai.gantt_chart_log.at(-1)?.end || 0
      )
    : 0;

  return (
    <div className="app-container">
      {showNamePopup && <NamePopup onSubmit={handleSubmitName} />}

      <header>
        <h1>Adaptive OS Scheduler</h1>
        <ModeSelector selectedMode={mode} onModeChange={handleModeChange} />
      </header>

      <main className="main-content">
        <div className="controls-column" style={{ filter: showNamePopup ? 'blur(4px)' : 'none' }}>
          {playerName && <GameStats playerName={playerName} stats={playerStats} />}
          {mode === 'Real-Time' && gameState === 'idle' && (
            <Timer 
                duration={10} 
                onTimeout={handleSubmitSchedule}
                onReset={timerResetKey}
            />
          )}
          <ProcessQueue
            processes={processes}
            setProcesses={setProcesses}
            onSubmit={handleSubmitSchedule}
            disabled={gameState === 'running' || !playerName}
          />
           <button onClick={() => handleStartGame(mode)} disabled={!playerName} className="process-submit-button" style={{backgroundColor: '#3f51b5'}}>
             New Process Set
           </button>
        </div>
        <div className="results-column" style={{ filter: showNamePopup ? 'blur(4px)' : 'none' }}>
          <Leaderboard leaderboard={leaderboard} />
          <ResultsDashboard results={simulationResults} />
          <GanttChart title="Your Gantt Chart" log={simulationResults?.human.gantt_chart_log} maxTime={maxTime} />
          <GanttChart title="AI Gantt Chart" log={simulationResults?.ai.gantt_chart_log} maxTime={maxTime} />
        </div>
      </main>
    </div>
  );
}

export default App;
```

---

## FILE: frontend/src/components/GameStats.jsx

```javascript
import React from 'react';

const GameStats = ({ playerName, stats }) => {
  const { wins, losses, ties, total } = stats;
  return (
    <div className="game-stats-container">
      <h3>Player Stats: {playerName}</h3>
      <div className="stats-grid">
        <span>Wins:</span><span>{wins}</span>
        <span>Losses:</span><span>{losses}</span>
        <span>Ties:</span><span>{ties}</span>
        <span>Total:</span><span>{total}</span>
      </div>
    </div>
  );
};

export default GameStats;
```

---

## FILE: frontend/src/components/GanttChart.jsx

```javascript
import React from 'react';

const GanttChart = ({ title, log, maxTime }) => {
  if (!log || log.length === 0) {
    return (
        <div className="gantt-chart-container">
            <h3>{title}</h3>
            <div className="placeholder">Run simulation to see Gantt Chart</div>
        </div>
    );
  }

  return (
    <div className="gantt-chart-container">
      <h3>{title}</h3>
      <div className="gantt-chart">
        {log.map((entry, index) => {
          const left = (entry.start / maxTime) * 100;
          const width = ((entry.end - entry.start) / maxTime) * 100;
          return (
            <div
              key={index}
              className="gantt-bar"
              style={{
                left: `${left}%`,
                width: `${width}%`,
                // Add some color variation for different processes
                backgroundColor: `hsl(${entry.pid * 60}, 70%, 50%)`
              }}
              title={`PID: ${entry.pid} (${entry.start}-${entry.end})`}
            >
              P{entry.pid}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default GanttChart;
```

---

## FILE: frontend/src/components/Leaderboard.jsx

```javascript
import React from 'react';

const Leaderboard = ({ leaderboard }) => {
    
  const getDifficulty = (mode) => {
    switch(mode) {
      case 'Efficiency': return 'Easy (Efficiency)';
      case 'Fairness': return 'Medium (Fairness)';
      case 'Real-Time': return 'Hard (Real-Time)';
      default: return 'N/A';
    }
  };

  return (
    <div className="leaderboard-container">
      <h3>Top Players</h3>
      <table className="leaderboard-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Name</th>
            <th>Score</th>
            <th>Difficulty</th>
          </tr>
        </thead>
        <tbody>
          {leaderboard.map((player, index) => (
            <tr key={index}>
              <td>{index + 1}</td>
              <td>{player.name}</td>
              <td>{player.score}</td>
              <td>{getDifficulty(player.mode)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Leaderboard;
```

---

## FILE: frontend/src/components/ModeSelector.jsx

```javascript
import React from 'react';

const MODES = ['Efficiency', 'Fairness', 'Real-Time'];

const ModeSelector = ({ selectedMode, onModeChange }) => {
  return (
    <div className="mode-selector">
      {MODES.map((mode) => (
        <button
          key={mode}
          className={selectedMode === mode ? 'active' : ''}
          onClick={() => onModeChange(mode)}
        >
          {mode}
        </button>
      ))}
    </div>
  );
};

export default ModeSelector;
```

---

## FILE: frontend/src/components/NamePopup.jsx

```javascript
import React, { useState } from 'react';

const NamePopup = ({ onSubmit }) => {
  const [name, setName] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (name.trim()) {
      onSubmit(name.trim());
    }
  };

  return (
    <div className="popup-overlay">
      <div className="popup-content">
        <h2>Welcome to the Scheduler Simulation</h2>
        <p>Please enter your name to play.</p>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Your Name"
            className="popup-input"
            autoFocus
          />
          <button type="submit" className="popup-button">Start Playing</button>
        </form>
      </div>
    </div>
  );
};

export default NamePopup;
```

---

## FILE: frontend/src/components/ProcessQueue.jsx

```javascript
import React, { useState, useRef } from 'react';

const ProcessQueue = ({ processes, setProcesses, onSubmit, disabled }) => {
  const [draggingItem, setDraggingItem] = useState(null);
  const dragItemNode = useRef(null);

  const handleDragStart = (e, index) => {
    setDraggingItem(index);
    dragItemNode.current = e.target;
    dragItemNode.current.addEventListener('dragend', handleDragEnd);
    setTimeout(() => {
      // Use a timeout to allow DOM update before applying dragging class
      e.target.classList.add('dragging');
    }, 0);
  };

  const handleDragEnter = (e, targetIndex) => {
    if (dragItemNode.current !== e.target) {
      const newList = [...processes];
      const draggedItemContent = newList.splice(draggingItem, 1)[0];
      newList.splice(targetIndex, 0, draggedItemContent);
      setDraggingItem(targetIndex);
      setProcesses(newList);
    }
  };
  
  const handleDragEnd = (e) => {
      e.target.classList.remove('dragging');
      dragItemNode.current.removeEventListener('dragend', handleDragEnd);
      setDraggingItem(null);
      dragItemNode.current = null;
  }

  return (
    <div className="process-queue-container">
      <h3>Your Schedule (Drag to Reorder)</h3>
      <ul className="process-list">
        {processes.map((p, index) => (
          <li
            key={p.pid}
            draggable
            onDragStart={(e) => handleDragStart(e, index)}
            onDragEnter={(e) => handleDragEnter(e, index)}
            className="process-item"
          >
            <span>PID: {p.pid}</span>
            <span>Arrival: {p.arrival_time}</span>
            <span>Burst: {p.burst_time}</span>
            <span>Prio: {p.priority}</span>
          </li>
        ))}
      </ul>
      <button onClick={onSubmit} disabled={disabled} className="process-submit-button">
        Run Simulation
      </button>
    </div>
  );
};

export default ProcessQueue;
```

---

## FILE: frontend/src/components/ResultsDashboard.jsx

```javascript
import React from 'react';

const MetricDisplay = ({ label, value, isOptimal }) => {
    const className = `result-metric ${isOptimal === true ? 'good' : isOptimal === false ? 'bad' : ''}`;
    return (
        <div>
            {label}: <span className={className}>{value}</span>
        </div>
    );
}

const ResultsDashboard = ({ results }) => {
  if (!results) {
    return (
        <div className="results-dashboard">
            <div className="result-card placeholder">Human Results</div>
            <div className="result-card placeholder">AI Results</div>
        </div>
    );
  }
  
  const { human, ai } = results;

  return (
    <div className="results-dashboard">
      <div className="result-card">
        <h3>Your Results</h3>
        <MetricDisplay 
            label="Algorithm" 
            value={human.algorithm}
        />
        <MetricDisplay 
            label="Avg. Wait Time" 
            value={human.metrics.average_waiting_time.toFixed(2)}
            isOptimal={human.metrics.avg_wait_is_optimal} 
        />
        <MetricDisplay 
            label="Context Switches" 
            value={human.metrics.context_switches}
            isOptimal={human.metrics.switches_are_optimal}
        />
      </div>
      <div className="result-card">
        <h3>AI Results</h3>
        <MetricDisplay 
            label="Algorithm" 
            value={ai.algorithm}
        />
        <MetricDisplay 
            label="Avg. Wait Time" 
            value={ai.metrics.average_waiting_time.toFixed(2)}
            isOptimal={ai.metrics.avg_wait_is_optimal}
        />
        <MetricDisplay 
            label="Context Switches" 
            value={ai.metrics.context_switches}
            isOptimal={ai.metrics.switches_are_optimal}
        />
      </div>
    </div>
  );
};

export default ResultsDashboard;
```

---

## FILE: frontend/src/components/Timer.jsx

```javascript
import React, { useState, useEffect } from 'react';

const Timer = ({ duration, onTimeout, onReset }) => {
  const [timeLeft, setTimeLeft] = useState(duration);

  useEffect(() => {
    setTimeLeft(duration);
  }, [duration, onReset]); // Reset when duration or onReset flag changes

  useEffect(() => {
    if (timeLeft <= 0) {
      onTimeout();
      return;
    }

    const intervalId = setInterval(() => {
      setTimeLeft(timeLeft - 1);
    }, 1000);

    return () => clearInterval(intervalId);
  }, [timeLeft, onTimeout]);

  return (
    <div className="timer-container">
      Time Left: {timeLeft}s
    </div>
  );
};

export default Timer;
```

---

## FILE: frontend/src/main.jsx

```javascript
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Import global and component-specific styles
import './styles/main.css';
import './styles/components.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

---

## FILE: frontend/src/services/api.js

```javascript
// src/services/api.js

/**
 * This file mocks the backend API.
 * In a real application, this would make HTTP requests to the server.
 * The data structures returned here are designed to match the expected
 * backend responses.
 */

// A helper function to generate random process data for the mock API
const generateRandomProcesses = (num) => {
    const processes = [];
    for (let i = 0; i < num; i++) {
        processes.push({
            pid: i,
            arrival_time: Math.floor(Math.random() * 10),
            burst_time: Math.floor(Math.random() * 15) + 1,
            priority: Math.floor(Math.random() * 5) + 1,
        });
    }
    // Sort by arrival time, as the backend would
    return processes.sort((a, b) => a.arrival_time - b.arrival_time);
};

// A helper to simulate a scheduling algorithm and return metrics
const simulateScheduling = (processes, algorithm) => {
    // This is a simplified, fake simulation for mock purposes.
    let gantt = [];
    let time = 0;
    let waitingTime = 0;
    let contextSwitches = 0;
    
    // Naive simulation logic
    processes.forEach((p, index) => {
        if (time < p.arrival_time) {
            time = p.arrival_time;
        }
        const wait = time - p.arrival_time;
        waitingTime += wait > 0 ? wait : 0;
        
        gantt.push({ pid: p.pid, start: time, end: time + p.burst_time });
        time += p.burst_time;
        if (index < processes.length - 1) {
             contextSwitches += (Math.random() > 0.5 ? 1 : 0); // Randomly add switches
        }
    });

    return {
        metrics: {
            average_waiting_time: waitingTime / processes.length,
            average_turnaround_time: (time - processes[0].arrival_time) / processes.length,
            context_switches: contextSwitches
        },
        gantt_chart_log: gantt,
    };
};

export const startNewGame = async (mode) => {
    console.log(`[API MOCK] Starting new game in ${mode} mode.`);
    const processCount = Math.floor(Math.random() * 5) + 4; // 4-8 processes
    const processes = generateRandomProcesses(processCount);
    return { processes };
};

export const submitSchedule = async (mode, humanSchedule) => {
    console.log(`[API MOCK] Submitting schedule for ${mode} mode.`);
    console.log('Human schedule:', humanSchedule.map(p => p.pid));

    // Simulate human run
    const humanResult = simulateScheduling(humanSchedule, 'manual');
    humanResult.algorithm = 'Manual';

    // Simulate AI run (let's pretend AI chose SJF)
    const aiAlgo = 'SJF';
    const aiSchedule = [...humanSchedule].sort((a, b) => a.burst_time - b.burst_time);
    const aiResult = simulateScheduling(aiSchedule, aiAlgo);
    aiResult.algorithm = aiAlgo;
    
    const calculateFinalScore = (metrics) => {
        // A simple score for comparison. Lower is better.
        return metrics.average_waiting_time + (metrics.context_switches * 2);
    }
    
    humanResult.metrics.final_score = calculateFinalScore(humanResult.metrics);
    aiResult.metrics.final_score = calculateFinalScore(aiResult.metrics);

    // Add some random coloring logic for results
    if (humanResult.metrics.final_score < aiResult.metrics.final_score) {
        humanResult.metrics.avg_wait_is_optimal = true;
        aiResult.metrics.avg_wait_is_optimal = false;
    } else {
        humanResult.metrics.avg_wait_is_optimal = false;
        aiResult.metrics.avg_wait_is_optimal = true;
    }
     if (humanResult.metrics.context_switches < aiResult.metrics.context_switches) {
        humanResult.metrics.switches_are_optimal = true;
        aiResult.metrics.switches_are_optimal = false;
    } else {
        humanResult.metrics.switches_are_optimal = false;
        aiResult.metrics.switches_are_optimal = true;
    }


    return {
        human: humanResult,
        ai: aiResult,
    };
};

export const getLeaderboard = async () => {
    console.log('[API MOCK] Fetching leaderboard data.');
    // In a real app, this would be a database call.
    // Here, we return a static, sorted list.
    return [
        { name: 'HAL 9000', score: 18, mode: 'Efficiency' },
        { name: 'Skynet', score: 22, mode: 'Real-Time' },
        { name: 'Jane', score: 25, mode: 'Fairness' },
        { name: 'Bob', score: 31, mode: 'Efficiency' },
        { name: 'Alice', score: 45, mode: 'Fairness' },
    ];
};
```

---

## FILE: frontend/src/styles/components.css

```css
/* components.css */

/* App Layout */
.app-container {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.main-content {
  display: grid;
  grid-template-columns: 1fr 2fr;
  gap: 2rem;
  align-items: flex-start;
}

.controls-column {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  padding: 1rem;
  background-color: var(--card-background-color);
  border-radius: 8px;
  border: 1px solid var(--border-color);
}

.results-column {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

/* ModeSelector */
.mode-selector {
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-bottom: 1rem;
}

.mode-selector button {
  padding: 10px 20px;
  border-radius: 8px;
  border: 1px solid transparent;
  font-size: 1em;
  font-weight: 500;
  cursor: pointer;
  transition: border-color 0.25s, background-color 0.25s;
  background-color: var(--card-background-color);
}

.mode-selector button:hover {
  border-color: var(--primary-color);
}

.mode-selector button.active {
  background-color: var(--primary-color);
  border-color: var(--primary-color-hover);
}

/* ProcessQueue */
.process-queue-container h3 {
  margin-top: 0;
}

.process-list {
  list-style: none;
  padding: 0;
  margin: 0;
  background-color: #1e1e1e;
  border-radius: 6px;
  border: 1px solid var(--border-color);
}

.process-item {
  display: flex;
  justify-content: space-between;
  padding: 12px 15px;
  border-bottom: 1px solid var(--border-color);
  cursor: grab;
  transition: background-color 0.2s;
  user-select: none;
}

.process-item:last-child {
  border-bottom: none;
}

.process-item.dragging {
  opacity: 0.5;
  background-color: var(--primary-color);
}

.process-item span {
  font-family: monospace;
  font-size: 0.9em;
}

.process-submit-button {
  width: 100%;
  margin-top: 1rem;
  padding: 10px 0;
  border-radius: 8px;
  border: 1px solid transparent;
  font-size: 1em;
  font-weight: 500;
  cursor: pointer;
  background-color: var(--primary-color);
}

.process-submit-button:disabled {
  background-color: var(--border-color);
  cursor: not-allowed;
}

/* Timer */
.timer-container {
  padding: 1rem;
  background-color: var(--warning-red);
  color: white;
  border-radius: 8px;
  font-size: 1.5em;
  font-weight: bold;
}

/* GanttChart */
.gantt-chart-container {
  width: 100%;
  background-color: var(--card-background-color);
  border-radius: 8px;
  border: 1px solid var(--border-color);
  padding: 1rem;
}

.gantt-chart {
  position: relative;
  width: 100%;
  height: 50px;
  background-color: var(--gantt-idle-color);
  border-radius: 4px;
  overflow: hidden;
}

.gantt-bar {
  position: absolute;
  height: 100%;
  background-color: var(--gantt-bar-color);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8em;
  color: white;
  overflow: hidden;
  white-space: nowrap;
  border-right: 1px solid var(--border-color);
}

/* ResultsDashboard */
.results-dashboard {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  text-align: left;
}
.result-card {
  padding: 1rem;
  background-color: var(--card-background-color);
  border: 1px solid var(--border-color);
  border-radius: 8px;
}
.result-card h3 {
  margin-top: 0;
}
.result-metric {
  font-family: monospace;
}
.result-metric.good {
  color: var(--success-green);
}
.result-metric.bad {
  color: var(--warning-red);
}

.placeholder {
    height: 100px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #888;
    border: 2px dashed var(--border-color);
    border-radius: 8px;
}

/* NamePopup */
.popup-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.popup-content {
  background-color: var(--card-background-color);
  padding: 2rem 3rem;
  border-radius: 8px;
  border: 1px solid var(--border-color);
  text-align: center;
}

.popup-input {
  width: 100%;
  padding: 10px;
  margin-top: 1rem;
  border-radius: 6px;
  border: 1px solid var(--border-color);
  background-color: #333;
  color: var(--text-color);
}

.popup-button {
  width: 100%;
  margin-top: 1rem;
  padding: 10px 0;
  border-radius: 8px;
  border: 1px solid transparent;
  font-size: 1em;
  font-weight: 500;
  cursor: pointer;
  background-color: var(--primary-color);
}

/* GameStats */
.game-stats-container {
    background-color: var(--card-background-color);
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid var(--border-color);
}
.game-stats-container h3 {
    margin-top: 0;
    text-align: center;
}
.stats-grid {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.5rem 1rem;
    font-family: monospace;
}

/* Leaderboard */
.leaderboard-container {
    background-color: var(--card-background-color);
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid var(--border-color);
}
.leaderboard-container h3 {
    margin-top: 0;
}
.leaderboard-table {
    width: 100%;
    border-collapse: collapse;
    text-align: left;
}
.leaderboard-table th, .leaderboard-table td {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border-color);
}
.leaderboard-table th {
    font-weight: 600;
}
.leaderboard-table tbody tr:last-child td {
    border-bottom: none;
}
```

---

## FILE: frontend/src/styles/main.css

```css
/* main.css */
:root {
  --background-color: #1a1a1a;
  --card-background-color: #242424;
  --border-color: #444;
  --text-color: rgba(255, 255, 255, 0.87);
  --primary-color: #646cff;
  --primary-color-hover: #535bf2;
  --gantt-bar-color: #747bff;
  --gantt-idle-color: #333;
  --warning-red: #c73e3e;
  --success-green: #3ec762;
  --font-family: Inter, system-ui, Avenir, Helvetica, Arial, sans-serif;
}

body {
  margin: 0;
  font-family: var(--font-family);
  background-color: var(--background-color);
  color: var(--text-color);
  line-height: 1.5;
  font-weight: 400;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#root {
  max-width: 1280px;
  margin: 0 auto;
  padding: 2rem;
  text-align: center;
}

h1, h2, h3 {
  font-weight: 500;
}
```

---

## FILE: frontend/vite.config.js

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
})
```

---

## FILE: models/scheduler_brain.pkl

```
This is a binary file and its content cannot be displayed.
```
