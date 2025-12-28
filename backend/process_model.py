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

    # Themed fields for Real-Time mode
    case_type: str | None = None
    severity: str | None = None

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

