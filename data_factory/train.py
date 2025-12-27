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
