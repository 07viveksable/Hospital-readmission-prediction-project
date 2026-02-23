
"""
Hospital Readmission Prediction System
Predicts 30-day readmission risk for diabetic patients
"""

import pandas as pd
import numpy as np
import joblib
import json

# Load saved models
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

with open('models/target_encoding.json', 'r') as f:
    target_info = json.load(f)

def predict_readmission(patient_data):
    """
    Predict hospital readmission risk

    Parameters:
    -----------
    patient_data : dict or DataFrame
        Patient features matching training data

    Returns:
    --------
    dict with prediction, probability, risk_level, and recommendation
    """
    # Convert to DataFrame if dict
    if isinstance(patient_data, dict):
        patient_data = pd.DataFrame([patient_data])

    # Scale features
    patient_scaled = scaler.transform(patient_data)

    # Predict
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0][1]

    # Determine risk level
    if probability < 0.3:
        risk_level = "Low"
        color = "🟢"
    elif probability < 0.6:
        risk_level = "Medium"
        color = "🟡"
    else:
        risk_level = "High"
        color = "🔴"

    # Generate recommendation
    recommendation = get_clinical_recommendation(risk_level, probability)

    return {
        'prediction': int(prediction),
        'prediction_label': 'Readmission <30 days' if prediction == 1 else 'No readmission <30 days',
        'probability': float(probability),
        'risk_level': risk_level,
        'risk_icon': color,
        'recommendation': recommendation
    }

def get_clinical_recommendation(risk_level, probability):
    """Generate clinical recommendations based on risk"""
    if risk_level == "High":
        return f"""
HIGH RISK ({probability:.1%})
- Schedule follow-up within 3-7 days
- Consider home health care services
- Provide detailed discharge instructions
- Ensure medication reconciliation
- Assign care coordinator
        """
    elif risk_level == "Medium":
        return f"""
MEDIUM RISK ({probability:.1%})
- Schedule follow-up within 14 days
- Provide comprehensive discharge education
- Verify patient understands medications
- Confirm post-discharge support system
        """
    else:
        return f"""
LOW RISK ({probability:.1%})
- Standard discharge protocol
- Routine follow-up as needed
- Provide standard discharge materials
        """

# Example usage
if __name__ == "__main__":
    # Example patient data
    example_patient = {
        'time_in_hospital': 7,
        'num_lab_procedures': 50,
        'num_procedures': 3,
        'num_medications': 15,
        'number_diagnoses': 9,
        # ... include all features from training
    }

    result = predict_readmission(example_patient)

    print("="*60)
    print("READMISSION RISK PREDICTION")
    print("="*60)
    print(f"\n{result['risk_icon']} Risk Level: {result['risk_level']}")
    print(f"Prediction: {result['prediction_label']}")
    print(f"Probability: {result['probability']:.2%}")
    print(f"\nClinical Recommendation:")
    print(result['recommendation'])
