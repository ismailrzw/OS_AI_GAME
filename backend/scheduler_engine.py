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
