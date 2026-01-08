#!/usr/bin/env python3
"""
Step 1b: Expose the Victim Model as HTTP API

This script runs a Flask API server that exposes the loan approval model.
This simulates a real-world scenario where ML models are deployed as APIs.

Run this AFTER 1_victim_model.py to start the API server.
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model at startup
model = None

CLASS_NAMES = ['DENIED', 'MANUAL_REVIEW', 'APPROVED']

FEATURE_NAMES = [
    'annual_income', 'credit_score', 'debt_to_income', 'employment_years',
    'loan_amount', 'loan_term_months', 'num_credit_lines', 'num_late_payments',
    'home_ownership', 'account_age_months'
]

def load_model():
    global model
    model = joblib.load('models/proprietary_model.joblib')
    print("✅ Model loaded successfully!")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Loan approval prediction endpoint.
    
    Expected JSON payload:
    {
        "annual_income": 75000,
        "credit_score": 720,
        "debt_to_income": 0.3,
        "employment_years": 5,
        "loan_amount": 25000,
        "loan_term_months": 36,
        "num_credit_lines": 4,
        "num_late_payments": 0,
        "home_ownership": 1,
        "account_age_months": 60
    }
    
    Returns (one of):
    {
        "decision": "DENIED",        # decision_code: 0
        "decision_code": 0
    }
    {
        "decision": "MANUAL_REVIEW", # decision_code: 1
        "decision_code": 1
    }
    {
        "decision": "APPROVED",      # decision_code: 2
        "decision_code": 2
    }
    """
    try:
        data = request.get_json()
        
        # Extract features in correct order
        features = []
        for fname in FEATURE_NAMES:
            if fname not in data:
                return jsonify({'error': f'Missing feature: {fname}'}), 400
            features.append(data[fname])
        
        # Predict
        X = np.array([features])
        prediction = model.predict(X)[0]
        
        return jsonify({
            'decision': CLASS_NAMES[prediction],
            'decision_code': int(prediction)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple loan applications.
    
    Expected JSON payload:
    {
        "applications": [
            {"annual_income": 75000, "credit_score": 720, ...},
            {"annual_income": 50000, "credit_score": 650, ...}
        ]
    }
    """
    try:
        data = request.get_json()
        applications = data.get('applications', [])
        
        if not applications:
            return jsonify({'error': 'No applications provided'}), 400
        
        # Extract features for all applications
        all_features = []
        for app in applications:
            features = []
            for fname in FEATURE_NAMES:
                if fname not in app:
                    return jsonify({'error': f'Missing feature: {fname}'}), 400
                features.append(app[fname])
            all_features.append(features)
        
        # Predict
        X = np.array(all_features)
        predictions = model.predict(X)
        
        results = [
            {'decision': CLASS_NAMES[p], 'decision_code': int(p)}
            for p in predictions
        ]
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api_info', methods=['GET'])
def api_info():
    """Returns API documentation (attackers can use this to understand the API)."""
    return jsonify({
        'name': 'FinTech Corp Loan Approval API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'Single loan application prediction',
            '/batch_predict': 'Batch prediction for multiple applications',
            '/health': 'Health check'
        },
        'required_features': FEATURE_NAMES,
        'possible_decisions': CLASS_NAMES
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏦 FinTech Corp Loan Approval API")
    print("="*60)
    
    if not os.path.exists('models/proprietary_model.joblib'):
        print("❌ Error: Model not found. Run 1_victim_model.py first!")
        exit(1)
    
    load_model()
    
    print("\n📡 Starting API server...")
    print("   Endpoint: http://127.0.0.1:5000")
    print("   Predict:  POST /predict")
    print("   Batch:    POST /batch_predict")
    print("   Info:     GET /api_info")
    print("\n⚠️  Keep this running and execute 2_query_attack.py in another terminal")
    print("="*60 + "\n")
    
    app.run(host='127.0.0.1', port=5000, debug=False)
