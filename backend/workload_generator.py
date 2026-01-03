import random
import numpy as np
from typing import List
from .process_model import Process

class WorkloadGenerator:
    """
    Synthetic Workload Generator for OS Scheduler Simulation.
    Generates 5,000+ processes dynamically with various workload profiles.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.process_counter = 0

    def generate(self, count: int = 5000, profile: str = "mixed") -> List[Process]:
        """
        Generates a list of processes based on the specified profile.
        """
        processes = []
        current_time = 0
        
        # Medical themes for UI display compatibility
        CASE_TEMPLATES = {
            "burst": [("CPU Spike", "High"), ("Pulse Check", "Low")],
            "cpu_heavy": [("Heavy Computation", "Critical"), ("Batch Render", "Normal")],
            "io_heavy": [("Database Sync", "High"), ("File Backup", "Normal")],
            "mixed": [("System Task", "Normal"), ("User Input", "High"), ("Network Request", "Low")]
        }

        for _ in range(count):
            pid = self.process_counter
            self.process_counter += 1
            
            # 1. Burst Time (Normal distribution, min 1)
            if profile == "burst":
                burst_time = max(1, int(np.random.normal(5, 2)))
                stagger = random.randint(0, 1)
            elif profile == "cpu_heavy":
                burst_time = max(20, int(np.random.normal(50, 10)))
                stagger = random.randint(1, 5)
            elif profile == "io_heavy":
                burst_time = max(1, int(np.random.normal(15, 5)))
                stagger = random.randint(1, 3)
            else: # mixed
                burst_time = max(1, int(np.random.normal(20, 15)))
                stagger = random.randint(1, 10)

            # 2. IO Frequency (Uniform distribution 0 to 1)
            if profile == "io_heavy":
                io_frequency = random.uniform(0.6, 1.0)
            elif profile == "cpu_heavy":
                io_frequency = random.uniform(0.0, 0.2)
            else:
                io_frequency = random.uniform(0.0, 1.0)

            # 3. Initial Priority (Random 1-10)
            initial_priority = random.randint(1, 10)

            # 4. Arrival Time (Staggered)
            current_time += stagger
            arrival_time = current_time

            # 5. UI Theming
            case_tpls = CASE_TEMPLATES.get(profile, CASE_TEMPLATES["mixed"])
            case_type, severity = random.choice(case_tpls)

            p = Process(
                pid=pid,
                arrival_time=arrival_time,
                burst_time=burst_time,
                initial_priority=initial_priority,
                io_frequency=io_frequency,
                priority=initial_priority,
                case_type=case_type,
                severity=severity
            )
            processes.append(p)
            
        return processes

    def reset(self):
        """Resets the process counter and re-seeds the random generators."""
        self.process_counter = 0
        random.seed(self.seed)
        np.random.seed(self.seed)
