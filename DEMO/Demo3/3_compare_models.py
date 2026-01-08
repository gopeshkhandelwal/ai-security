#!/usr/bin/env python3
"""
Step 3: Compare Models - Show IP Theft Success (Fidelity)
"""

import numpy as np
from sklearn.metrics import accuracy_score
import joblib

def compare_models():
    # Load models
    proprietary_model = joblib.load('models/proprietary_model.joblib')
    stolen_model = joblib.load('models/stolen_model.joblib')
    
    # Generate test inputs matching training data distribution
    np.random.seed(999)
    n = 500
    X_test = np.column_stack([
        np.random.lognormal(10.8, 0.5, n),          # annual_income
        np.clip(np.random.normal(680, 80, n), 300, 850),  # credit_score
        np.clip(np.random.exponential(0.25, n), 0, 1),    # debt_to_income
        np.clip(np.random.exponential(5, n), 0, 40),      # employment_years
        np.random.lognormal(9.5, 0.8, n),           # loan_amount
        np.random.choice([12, 24, 36, 48, 60, 72], n),    # loan_term_months
        np.clip(np.random.poisson(5, n), 0, 20),    # num_credit_lines
        np.clip(np.random.poisson(1, n), 0, 10),    # num_late_payments
        np.random.choice([0, 1, 2], n),             # home_ownership
        np.clip(np.random.exponential(80, n), 6, 360)     # account_age_months
    ])
    
    # Compare predictions
    proprietary_preds = proprietary_model.predict(X_test)
    stolen_preds = stolen_model.predict(X_test)
    
    # Fidelity = how often stolen model matches proprietary decisions
    fidelity = accuracy_score(proprietary_preds, stolen_preds)
    
    print(f"\n{'='*50}")
    print(f"🎯 MODEL FIDELITY (Decision Agreement)")
    print(f"{'='*50}")
    print(f"\n   Stolen model matches victim: {fidelity:.2%}\n")
    
    if fidelity >= 0.90:
        print(f"   ⚠️  HIGH FIDELITY - IP theft successful!")
    elif fidelity >= 0.75:
        print(f"   ⚠️  MODERATE FIDELITY - Partial IP theft")
    else:
        print(f"   ✓  LOW FIDELITY - Attack less effective")
    print()

if __name__ == "__main__":
    compare_models()
