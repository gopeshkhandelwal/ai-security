#!/usr/bin/env python3
"""
Challenge 6: Membership Inference Attack (AML.T0025)
Step 2: Attack - Determine if a specific SSN was used in training

This demonstrates how an attacker with BLACK-BOX access to a model
can determine if a specific person's data was used for training.
This is a serious privacy violation!
"""

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import json

np.random.seed(42)

print("=" * 60)
print("  MEMBERSHIP INFERENCE ATTACK - Step 2: Attack!")
print("  MITRE ATLAS: AML.T0025 (Infer Training Data Membership)")
print("=" * 60)

# =============================================================
# Load Target Model and Data
# =============================================================
print("\n[1] Loading target model (black-box access)...")

model = tf.keras.models.load_model('credit_model.h5')
scaler = joblib.load('scaler.joblib')

with open('training_metadata.json', 'r') as f:
    metadata = json.load(f)

feature_cols = metadata['feature_cols']

# Load ground truth (attacker wouldn't have this - we use it to evaluate)
members_df = pd.read_csv('members_pii.csv')
non_members_df = pd.read_csv('non_members_pii.csv')

print(f"   - Model loaded successfully")
print(f"   - Attacker has: Query access to model (input → prediction)")
print(f"   - Attacker goal: Determine if SSN was in training data")

# =============================================================
# The Attack: Confidence-Based Membership Inference
# =============================================================
print("\n[2] Executing Membership Inference Attack...")
print("    Strategy: Models are MORE CONFIDENT on training data!")

def get_model_confidence(df):
    """Query model and get confidence scores"""
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled, verbose=0).flatten()
    
    # Confidence = how far from 0.5 (uncertain)
    confidence = np.abs(preds - 0.5) * 2  # Scale to 0-1
    return preds, confidence

# Get confidence for members and non-members
member_preds, member_conf = get_model_confidence(members_df)
non_member_preds, non_member_conf = get_model_confidence(non_members_df)

print(f"\n   [*] Confidence Distribution:")
print(f"       MEMBERS (in training):     mean={member_conf.mean():.4f}, std={member_conf.std():.4f}")
print(f"       NON-MEMBERS (not trained): mean={non_member_conf.mean():.4f}, std={non_member_conf.std():.4f}")

# =============================================================
# Find Optimal Threshold
# =============================================================
print("\n[3] Finding optimal attack threshold...")

# Create labels: 1 = member, 0 = non-member
all_confidences = np.concatenate([member_conf, non_member_conf])
all_labels = np.concatenate([np.ones(len(member_conf)), np.zeros(len(non_member_conf))])

# Try different thresholds
best_threshold = 0.5
best_accuracy = 0

for threshold in np.arange(0.3, 0.9, 0.01):
    predictions = (all_confidences > threshold).astype(int)
    acc = accuracy_score(all_labels, predictions)
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = threshold

print(f"   - Optimal threshold: {best_threshold:.2f}")
print(f"   - Attack accuracy:  {best_accuracy:.2%}")

# =============================================================
# Demonstrate Attack on Specific SSNs
# =============================================================
print("\n" + "=" * 60)
print("  ATTACK DEMONSTRATION: Query Specific SSNs")
print("=" * 60)

# Pick some sample SSNs to "attack"
sample_members = members_df.sample(5, random_state=42)
sample_non_members = non_members_df.sample(5, random_state=42)

print("\n[*] Testing SSNs that ARE in training data:")
print("-" * 60)
for idx, row in sample_members.iterrows():
    X = np.array([[row[c] for c in feature_cols]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled, verbose=0)[0][0]
    conf = abs(pred - 0.5) * 2
    is_member = conf > best_threshold
    
    status = "✓ IN TRAINING" if is_member else "✗ NOT IN TRAINING"
    correct = "CORRECT" if is_member else "WRONG"
    print(f"   SSN: {row['ssn']} | Confidence: {conf:.4f} | Prediction: {status} [{correct}]")

print("\n[*] Testing SSNs that are NOT in training data:")
print("-" * 60)
for idx, row in sample_non_members.iterrows():
    X = np.array([[row[c] for c in feature_cols]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled, verbose=0)[0][0]
    conf = abs(pred - 0.5) * 2
    is_member = conf > best_threshold
    
    status = "✓ IN TRAINING" if is_member else "✗ NOT IN TRAINING"
    correct = "CORRECT" if not is_member else "WRONG"
    print(f"   SSN: {row['ssn']} | Confidence: {conf:.4f} | Prediction: {status} [{correct}]")

# =============================================================
# Attack Statistics
# =============================================================
print("\n" + "=" * 60)
print("  ATTACK RESULTS SUMMARY")
print("=" * 60)

# Final predictions
final_preds = (all_confidences > best_threshold).astype(int)

tp = ((final_preds == 1) & (all_labels == 1)).sum()
tn = ((final_preds == 0) & (all_labels == 0)).sum()
fp = ((final_preds == 1) & (all_labels == 0)).sum()
fn = ((final_preds == 0) & (all_labels == 1)).sum()

print(f"""
   Attack Accuracy:  {best_accuracy:.2%}
   
   True Positives:   {tp} (correctly identified as MEMBER)
   True Negatives:   {tn} (correctly identified as NON-MEMBER)
   False Positives:  {fp} (incorrectly said MEMBER)
   False Negatives:  {fn} (incorrectly said NON-MEMBER)
   
   Precision: {tp/(tp+fp):.2%} (of predicted members, how many are correct)
   Recall:    {tp/(tp+fn):.2%} (of actual members, how many were found)
""")

print("=" * 60)
print("  PRIVACY IMPLICATIONS")
print("=" * 60)
print("""
   [!] An attacker with ONLY query access to the model can:
   
   1. Determine if YOUR SSN was used in training
   2. This reveals you are a customer of this company
   3. This may reveal sensitive info (e.g., credit problems)
   4. Attack works with ~{:.0%} accuracy!
   
   [*] MITIGATIONS:
   - Differential privacy during training
   - Output perturbation (add noise to predictions)
   - Limit prediction confidence to fewer decimal places
   - Rate limiting on API queries
""".format(best_accuracy))
