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
