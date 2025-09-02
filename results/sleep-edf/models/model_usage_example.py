# Sleep Classification Model - Usage Example
# Generated automatically by Cell 9

import joblib
import numpy as np
import pandas as pd

# Load the best model and scaler
def load_sleep_model():
    """Load the trained sleep classification model and scaler."""
    model = joblib.load('results/sleep-edf/models/sleep_classification_decision_tree_conservative.joblib')
    scaler = joblib.load('results/sleep-edf/models/feature_scaler.joblib')
    return model, scaler

# Feature names (in correct order)
FEATURE_NAMES = [
    'EEG Fpz-Cz_delta_rel_power',
    'EEG Fpz-Cz_theta_rel_power', 
    'EEG Fpz-Cz_alpha_rel_power',
    'EEG Fpz-Cz_beta_rel_power',
    'EEG Pz-Oz_delta_rel_power',
    'EEG Pz-Oz_theta_rel_power',
    'EEG Pz-Oz_alpha_rel_power', 
    'EEG Pz-Oz_beta_rel_power',
    'EOG horizontal_mean',
    'age',
    'sex'  # 0=Male, 1=Female
]

def predict_sleep_issues(eeg_features, age, sex):
    """
    Predict sleep issues from EEG features and demographics.
    
    Parameters:
    - eeg_features: dict with EEG power features
    - age: int, age in years
    - sex: int, 0=Male, 1=Female
    
    Returns:
    - prediction: 0=Healthy (SC), 1=Sleep Issues (ST)
    - probability: probability of sleep issues
    """
    
    # Load model
    model, scaler = load_sleep_model()
    
    # Create feature vector
    features = np.array([
        eeg_features['EEG Fpz-Cz_delta_rel_power'],
        eeg_features['EEG Fpz-Cz_theta_rel_power'],
        eeg_features['EEG Fpz-Cz_alpha_rel_power'],
        eeg_features['EEG Fpz-Cz_beta_rel_power'],
        eeg_features['EEG Pz-Oz_delta_rel_power'],
        eeg_features['EEG Pz-Oz_theta_rel_power'],
        eeg_features['EEG Pz-Oz_alpha_rel_power'],
        eeg_features['EEG Pz-Oz_beta_rel_power'],
        eeg_features['EOG horizontal_mean'],
        age,
        sex
    ]).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    return prediction, probability

# Example usage:
if __name__ == "__main__":
    # Example EEG features (replace with actual values)
    example_eeg = {
        'EEG Fpz-Cz_delta_rel_power': 500.0,
        'EEG Fpz-Cz_theta_rel_power': 45.0,
        'EEG Fpz-Cz_alpha_rel_power': 25.0,
        'EEG Fpz-Cz_beta_rel_power': 15.0,
        'EEG Pz-Oz_delta_rel_power': 600.0,
        'EEG Pz-Oz_theta_rel_power': 55.0,
        'EEG Pz-Oz_alpha_rel_power': 35.0,
        'EEG Pz-Oz_beta_rel_power': 20.0,
        'EOG horizontal_mean': 0.0005
    }
    
    prediction, prob = predict_sleep_issues(example_eeg, age=45, sex=1)
    
    result = "Sleep Issues" if prediction == 1 else "Healthy"
    print(f"Prediction: {result}")
    print(f"Probability of sleep issues: {prob:.3f}")
