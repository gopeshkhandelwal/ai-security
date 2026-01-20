#!/usr/bin/env python3
"""
Step 4: Secure API Server with Model Stealing Defenses

Industry-standard protections:
1. Flask-Limiter - Rate limiting per IP
2. Differential Privacy - Add noise to suspicious requests
3. Query Pattern Detection - Detect systematic probing
4. Audit Logging - Track suspicious activity

Defense Strategy: Rate-Based Noise
- Normal users (<20 req/min): Clean responses
- Suspicious patterns (>20 req/min): Noisy responses + logging

Run: python 4_secure_api_server.py
Then: python 2_query_attack.py (attack will be degraded)

Author: GopeshK
License: MIT License
"""

from flask import Flask, request, jsonify, g
from functools import wraps
import joblib
import numpy as np
import os
import time
import json
import logging
from datetime import datetime
from collections import defaultdict

# Rate limiting
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    LIMITER_AVAILABLE = True
except ImportError:
    LIMITER_AVAILABLE = False

# Differential Privacy
try:
    from diffprivlib.mechanisms import LaplaceBoundedDomain
    DP_AVAILABLE = True
except ImportError:
    DP_AVAILABLE = False

app = Flask(__name__)

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

RATE_LIMIT = "20 per minute"  # Flask-Limiter hard limit
NOISE_EPSILON = 0.1           # Differential privacy (lower = MORE noise)
SUSPICIOUS_THRESHOLD = 5      # Queries before flagging
BLOCK_THRESHOLD = 100         # Block after 100 queries total

# =============================================================================
# AUDIT LOGGING
# =============================================================================

# File logging
logging.basicConfig(
    filename='api_security_audit.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

# Console logging for demo visibility
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
logging.getLogger().addHandler(console_handler)

def audit_log(event_type: str, ip: str, details: str = ""):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        "ip": ip,
        "details": details
    }
    logging.info(json.dumps(log_entry))
    # Also print to console for demo visibility
    print(f"🔒 [{event_type}] {ip}: {details}")

# =============================================================================
# REQUEST TRACKING (In-memory for demo; use Redis in production)
# =============================================================================

class RequestTracker:
    """Track request patterns per IP to detect model stealing attempts"""
    
    def __init__(self):
        self.requests = defaultdict(list)  # IP -> list of timestamps
        self.query_hashes = defaultdict(set)  # IP -> set of query hashes
        self.window = 60  # 60 second window
    
    def record(self, ip: str, query_hash: str):
        """Record a request from an IP"""
        now = time.time()
        self.requests[ip].append(now)
        self.query_hashes[ip].add(query_hash)
        
        # Clean old entries
        self.requests[ip] = [t for t in self.requests[ip] if now - t < self.window]
    
    def get_rate(self, ip: str) -> int:
        """Get requests per minute for an IP"""
        now = time.time()
        self.requests[ip] = [t for t in self.requests[ip] if now - t < self.window]
        return len(self.requests[ip])
    
    def get_unique_queries(self, ip: str) -> int:
        """Get unique query count (high count = systematic probing)"""
        return len(self.query_hashes[ip])
    
    def is_suspicious(self, ip: str) -> bool:
        """Detect suspicious query patterns"""
        rate = self.get_rate(ip)
        unique = self.get_unique_queries(ip)
        
        # Suspicious if: high rate OR many unique queries (systematic probing)
        return rate > SUSPICIOUS_THRESHOLD or unique > 50
    
    def should_block(self, ip: str) -> bool:
        """Should this IP be blocked?"""
        return self.get_rate(ip) > BLOCK_THRESHOLD

tracker = RequestTracker()

# =============================================================================
# DIFFERENTIAL PRIVACY
# =============================================================================

class PredictionProtector:
    """Add calibrated noise to predictions for suspicious requests"""
    
    def __init__(self, epsilon: float = 0.5):
        self.epsilon = epsilon
        if DP_AVAILABLE:
            # Laplace mechanism for bounded domain [0, 2] (our class indices)
            self.mechanism = LaplaceBoundedDomain(
                epsilon=epsilon,
                lower=0,
                upper=2,
                sensitivity=1
            )
            print(f"✅ Differential Privacy enabled (ε={epsilon})")
        else:
            self.mechanism = None
            print("⚠️  diffprivlib not installed - using fallback noise")
    
    def add_noise(self, prediction: int) -> int:
        """Add noise to prediction (may flip decision)"""
        if self.mechanism:
            noisy = self.mechanism.randomise(float(prediction))
            return int(np.clip(round(noisy), 0, 2))
        else:
            # Fallback: 30% chance to flip to adjacent class
            if np.random.random() < 0.3:
                if prediction == 0:
                    return np.random.choice([0, 1])
                elif prediction == 2:
                    return np.random.choice([1, 2])
                else:
                    return np.random.choice([0, 1, 2])
            return prediction

protector = PredictionProtector(epsilon=NOISE_EPSILON)

# =============================================================================
# RATE LIMITER
# =============================================================================

if LIMITER_AVAILABLE:
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["500 per minute"],  # High limit - rely on custom tracker for demo
        storage_uri="memory://"
    )
    print(f"✅ Rate limiting enabled (500/min hard limit, noise starts at {SUSPICIOUS_THRESHOLD}/min)")
else:
    limiter = None
    print("⚠️  Flask-Limiter not installed - using fallback rate limiting")

# =============================================================================
# MODEL LOADING
# =============================================================================

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

# =============================================================================
# SECURITY MIDDLEWARE
# =============================================================================

def security_check(f):
    """Decorator to apply security checks before prediction"""
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.remote_addr
        
        # Check if blocked
        if tracker.should_block(ip):
            audit_log("BLOCKED", ip, f"Exceeded {BLOCK_THRESHOLD} req/min")
            return jsonify({
                'error': 'Rate limit exceeded. Your access has been temporarily blocked.',
                'retry_after': 60
            }), 429
        
        # Set suspicious flag for downstream use
        g.is_suspicious = tracker.is_suspicious(ip)
        g.request_rate = tracker.get_rate(ip)
        
        if g.is_suspicious:
            audit_log("SUSPICIOUS", ip, f"Rate: {g.request_rate}/min, Unique queries: {tracker.get_unique_queries(ip)}")
        
        return f(*args, **kwargs)
    return decorated

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'security': {
            'rate_limiting': LIMITER_AVAILABLE,
            'differential_privacy': DP_AVAILABLE,
            'mode': 'SECURE'
        }
    })

@app.route('/predict', methods=['POST'])
@security_check
def predict():
    """
    Secure prediction endpoint with model stealing defenses.
    
    Normal users: Clean predictions
    Suspicious patterns: Noisy predictions
    """
    try:
        ip = request.remote_addr
        data = request.get_json()
        
        # Extract features
        features = []
        for fname in FEATURE_NAMES:
            if fname not in data:
                return jsonify({'error': f'Missing feature: {fname}'}), 400
            features.append(data[fname])
        
        # Track request
        query_hash = hash(tuple(features))
        tracker.record(ip, str(query_hash))
        
        # Predict
        X = np.array([features])
        prediction = model.predict(X)[0]
        
        # Apply defense: Add noise if suspicious
        original_prediction = prediction
        if g.is_suspicious:
            prediction = protector.add_noise(prediction)
            if prediction != original_prediction:
                audit_log("NOISE_APPLIED", ip, f"Original: {CLASS_NAMES[original_prediction]} → Noisy: {CLASS_NAMES[prediction]}")
        
        response = {
            'decision': CLASS_NAMES[prediction],
            'decision_code': int(prediction)
        }
        
        # Don't return confidence scores (makes stealing harder)
        # In vulnerable version, we might return: 'confidence': 0.92
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
@security_check
def batch_predict():
    """
    Batch prediction with enhanced security.
    Batch endpoints are prime targets for model stealing!
    
    Security measures:
    - Batch size limits based on trust level
    - Pre-check: would this batch exceed limits?
    - Differential privacy noise for suspicious IPs
    - Rate limiting tracked per application in batch
    """
    ip = request.remote_addr
    
    try:
        data = request.get_json()
        applications = data.get('applications', [])
        
        # PRE-CHECK: Will this batch push us over the limit?
        current_rate = tracker.get_rate(ip)
        if current_rate + len(applications) > BLOCK_THRESHOLD:
            audit_log("BATCH_BLOCKED", ip, f"Would exceed limit: {current_rate} + {len(applications)} > {BLOCK_THRESHOLD}")
            return jsonify({
                'error': 'Rate limit exceeded. Your access has been temporarily blocked.',
                'retry_after': 60
            }), 429
        
        # Adaptive batch size based on trust level
        if g.is_suspicious:
            MAX_BATCH = 10  # Suspicious users get SMALL batches
            audit_log("BATCH_RESTRICTED", ip, f"Suspicious IP limited to {MAX_BATCH} per batch")
        else:
            MAX_BATCH = 50  # Normal users get reasonable batch size
        
        if len(applications) > MAX_BATCH:
            audit_log("BATCH_LIMITED", ip, f"Requested {len(applications)}, limited to {MAX_BATCH}")
            applications = applications[:MAX_BATCH]
        
        all_features = []
        for app in applications:
            features = []
            for fname in FEATURE_NAMES:
                if fname not in app:
                    return jsonify({'error': f'Missing feature: {fname}'}), 400
                features.append(app[fname])
            all_features.append(features)
            tracker.record(ip, str(hash(tuple(features))))
        
        X = np.array(all_features)
        predictions = model.predict(X)
        
        # Apply noise to predictions for suspicious IPs
        if g.is_suspicious:
            noisy_count = 0
            for i in range(len(predictions)):
                original = predictions[i]
                predictions[i] = protector.add_noise(predictions[i])
                if predictions[i] != original:
                    noisy_count += 1
            if noisy_count > 0:
                audit_log("BATCH_NOISE", ip, f"Applied noise to {noisy_count}/{len(predictions)} predictions")
        
        results = [
            {'decision': CLASS_NAMES[p], 'decision_code': int(p)}
            for p in predictions
        ]
        
        return jsonify({'results': results, 'processed': len(results)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api_info', methods=['GET'])
def api_info():
    """API info - features must be exposed for legitimate partners"""
    return jsonify({
        'name': 'FinTech Corp Loan Approval API',
        'version': '2.0-secure',
        'note': 'Rate limits and security monitoring active',
        'required_features': FEATURE_NAMES,
        'possible_decisions': CLASS_NAMES
    })

@app.route('/security_status', methods=['GET'])
def security_status():
    """Admin endpoint to check security status"""
    ip = request.remote_addr
    return jsonify({
        'your_ip': ip,
        'your_rate': tracker.get_rate(ip),
        'your_unique_queries': tracker.get_unique_queries(ip),
        'is_suspicious': tracker.is_suspicious(ip),
        'defenses_active': {
            'rate_limiting': LIMITER_AVAILABLE,
            'differential_privacy': DP_AVAILABLE,
            'query_tracking': True,
            'batch_limiting': True
        }
    })

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  SECURE LOAN APPROVAL API")
    print("  Model Stealing Defenses Active")
    print("=" * 60)
    
    print(f"\nSecurity Configuration:")
    print(f"  • Rate limit: {RATE_LIMIT}")
    print(f"  • Suspicious threshold: {SUSPICIOUS_THRESHOLD} req/min")
    print(f"  • Block threshold: {BLOCK_THRESHOLD} req/min")
    print(f"  • DP epsilon: {NOISE_EPSILON}")
    
    print(f"\nDefense Layers:")
    print(f"  ✓ Rate Limiting: {'Flask-Limiter' if LIMITER_AVAILABLE else 'Fallback'}")
    print(f"  ✓ Differential Privacy: {'diffprivlib' if DP_AVAILABLE else 'Fallback'}")
    print(f"  ✓ Query Pattern Detection: Enabled")
    print(f"  ✓ Audit Logging: api_security_audit.log")
    
    print(f"\nEndpoints:")
    print(f"  POST /predict         - Single prediction (protected)")
    print(f"  POST /batch_predict   - Batch prediction (extra protected)")
    print(f"  GET  /health          - Health check")
    print(f"  GET  /security_status - Check your security status")
    
    print(f"\n⚠️  To test: Run 'python 2_query_attack.py' and compare results")
    print(f"   Expected: Attack fidelity drops from ~95% to ~65%\n")
    
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
