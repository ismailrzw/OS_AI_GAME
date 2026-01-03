"""
metrics.py

Defines a data structure for collecting and reporting simulation performance metrics.
"""
import numpy as np
from typing import List, Dict, Any

class SimulationMetrics:
    """
    A container for all statistics gathered during a single scheduler simulation run.
    """
    def __init__(self):
        self.gantt_chart_log: List[Dict[str, Any]] = []
        self.context_switches: int = 0
        # New metric: Count only involuntary context switches (preemptions).
        self.preemption_count: int = 0

        # Final calculated metrics for completed processes.
        self.total_waiting_time: int = 0
        self.total_turnaround_time: int = 0
        self.cpu_utilization: float = 0.0
        self.completed_processes_count: int = 0
        
        # New metrics required for the cost function
        self.max_waiting_time: int = 0
        self.waiting_time_variance: float = 0.0
    
    def add_gantt_entry(self, pid: int, start_time: int, end_time: int):
        """Records a process execution slice for the Gantt chart."""
        if start_time < end_time:
            # Check if the last entry was the same process and ended exactly when this one starts
            if self.gantt_chart_log and self.gantt_chart_log[-1]['pid'] == pid and self.gantt_chart_log[-1]['end'] == start_time:
                # Merge by extending the end time of the previous entry
                self.gantt_chart_log[-1]['end'] = end_time
            else:
                self.gantt_chart_log.append({'pid': pid, 'start': start_time, 'end': end_time})

    def increment_context_switches(self):
        """Increments the total context switch counter."""
        self.context_switches += 1

    def increment_preemptions(self):
        """Increments the preemption counter."""
        self.preemption_count += 1

    def finalize(self, completed_processes: List['Process'], total_time: int):
        """
        Calculates final summary metrics after the simulation has completed.
        This should be called once at the very end.
        """
        if not completed_processes:
            return

        self.completed_processes_count = len(completed_processes)
        
        total_burst_time = 0
        waiting_times = []
        for p in completed_processes:
            turnaround_time = p.finish_time - p.arrival_time
            self.total_turnaround_time += turnaround_time
            
            p.waiting_time = turnaround_time - p.burst_time
            self.total_waiting_time += p.waiting_time
            waiting_times.append(p.waiting_time)
            
            total_burst_time += p.burst_time

        if total_time > 0:
            self.cpu_utilization = (total_burst_time / total_time) * 100
        
        # Calculate new metrics for the cost function
        if waiting_times:
            self.max_waiting_time = max(waiting_times)
            self.waiting_time_variance = np.var(waiting_times)

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
