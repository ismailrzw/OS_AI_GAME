import os
import pandas as pd
import joblib
from typing import List, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Import from our project modules
from backend.process_model import Process
from backend.scheduler_engine import Scheduler
from .generator import generate_scenarios
from .scorer import calculate_total_cost
from .features import extract_features_from_scenario, ALGORITHM_ENCODING, COST_WEIGHTS

# --- Constants ---
# Increased number of scenarios for a more robust regression model.
NUM_SCENARIOS_TO_GENERATE = 10000
ALGORITHMS_TO_TEST = list(ALGORITHM_ENCODING.keys())
MODEL_OUTPUT_DIR = 'models'
MODEL_FILE_NAME = 'scheduler_brain.pkl'
DATASET_FILE_NAME = 'data_factory/dataset.csv'


def main():
    """Main training function for the new cost-based regression model."""
    print("--- AI Training Pipeline (Regression Model) Started ---")
    
    # 1. Generate Scenarios
    print(f"Step 1: Generating {NUM_SCENARIOS_TO_GENERATE} base scenarios...")
    scenarios = generate_scenarios(NUM_SCENARIOS_TO_GENERATE)
    
    # 2. Process scenarios and build the new cost-based dataset
    print("Step 2: Running simulations for each algorithm to generate cost data...")
    dataset = []
    # Each scenario will now generate one row PER algorithm.
    total_simulations = len(scenarios) * len(ALGORITHMS_TO_TEST)
    print(f"Total simulations to run: {total_simulations}")

    processed_sims = 0
    for mode, processes in scenarios:
        # For each scenario, test every algorithm
        for algo_name in ALGORITHMS_TO_TEST:
            # Create a deep copy of processes for a clean simulation run
            process_copy = [Process(p.pid, p.arrival_time, p.burst_time, p.priority) for p in processes]
            
            # Run the simulation
            scheduler = Scheduler(process_copy)
            metrics, completed = scheduler.run(algorithm=algo_name, time_quantum=4)
            
            # Get the cost function weights (now only Real-Time)
            weights = COST_WEIGHTS[mode]

            # Calculate the final cost for this specific algorithm run
            total_cost = calculate_total_cost(weights, metrics, completed)

            # Skip invalid runs
            if total_cost == float('inf'):
                continue

            # Extract features for this specific scenario-algorithm pair
            features = extract_features_from_scenario(processes, mode, algo_name)
            
            # Add the target variable to the record
            features['total_cost'] = total_cost
            dataset.append(features)

            processed_sims += 1
            if processed_sims % 2500 == 0:
                print(f"  ...processed {processed_sims}/{total_simulations} simulations")

    # 3. Create and save DataFrame
    print(f"Step 3: Saving new cost-based dataset to {DATASET_FILE_NAME}...")
    df = pd.DataFrame(dataset)
    df.to_csv(DATASET_FILE_NAME, index=False)
    
    # 4. Model Training
    print("Step 4: Training the Decision Tree Regressor model...")
    
    # Define features (X) and target (y)
    X = df.drop('total_cost', axis=1)
    y = df['total_cost']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"  Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    # Initialize and train the regressor
    model = DecisionTreeRegressor(max_depth=12, random_state=42) # A slightly deeper tree might be needed for regression
    model.fit(X_train, y_train)
    
    # 5. Evaluate Model
    print("Step 5: Evaluating model performance...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"  Model Mean Squared Error (MSE) on Test Set: {mse:.2f}")
    
    # 6. Save the trained model
    print("Step 6: Saving the trained regression model...")
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
    
    model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILE_NAME)
    joblib.dump(model, model_path)
    print(f"  Model successfully saved to: {model_path}")
    
    print("--- AI Training Pipeline Finished ---")

if __name__ == '__main__':
    main()
