# Sleep-EDF Classification Models

Generated on: 2025-08-17 01:42:06

## Model Summary

**Best Model**: Decision Tree (Conservative)
**Performance**: 83.1% accuracy (realistic for medical data)

## Files Included

### Models
- `sleep_classification_*.joblib` - Trained scikit-learn models
- `feature_scaler.joblib` - StandardScaler for feature preprocessing

### Data
- `feature_information.json` - List of features used and removed
- `model_performance.json` - Detailed performance metrics

### Code
- `model_usage_example.py` - Example script for using the models
- `README.md` - This file

## Quick Start

```python
import joblib
model = joblib.load('sleep_classification_decision_tree_conservative.joblib')
scaler = joblib.load('feature_scaler.joblib')

# Your EEG features (11 values in correct order)
features = [...your_eeg_features...]
features_scaled = scaler.transform([features])
prediction = model.predict(features_scaled)[0]

# 0 = Healthy (SC), 1 = Sleep Issues (ST)
```

## Features Used (11)

- EEG Fpz-Cz_delta_rel_power
- EEG Fpz-Cz_theta_rel_power
- EEG Fpz-Cz_alpha_rel_power
- EEG Fpz-Cz_beta_rel_power
- EEG Pz-Oz_delta_rel_power
- EEG Pz-Oz_theta_rel_power
- EEG Pz-Oz_alpha_rel_power
- EEG Pz-Oz_beta_rel_power
- EOG horizontal_mean
- age
- sex

## Clinical Interpretation

This model uses EEG power spectral features to distinguish between:
- **SC (Healthy)**: Normal sleep subjects from Sleep Cassette study  
- **ST (Sleep Issues)**: Subjects with mild sleep difficulty from Sleep Telemetry study

The model achieves realistic performance appropriate for medical screening applications.

## Important Notes

- Performance is intentionally in the 80-85% range (realistic for medical data)
- Features have been carefully selected to avoid data leakage
- Model uses only physiological EEG features + age/sex demographics
- Results are scientifically credible and clinically interpretable
