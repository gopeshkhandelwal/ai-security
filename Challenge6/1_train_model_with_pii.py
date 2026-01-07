#!/usr/bin/env python3
"""
Challenge 6: Membership Inference Attack (AML.T0025)
Step 1: Train a model on sensitive PII data (dummy SSNs)

This simulates a credit scoring model trained on customer data.
The goal: Show that attackers can determine if a specific person's
data was used in training - a serious privacy violation!
"""

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("  MEMBERSHIP INFERENCE ATTACK - Step 1: Train Model")
print("  MITRE ATLAS: AML.T0025 (Infer Training Data Membership)")
print("=" * 60)

# =============================================================
# Generate Dummy PII Dataset (Credit Scoring Scenario)
# =============================================================
print("\n[1] Generating dummy PII dataset...")

def generate_ssn():
    """Generate dummy SSN (XXX-XX-XXXX format)"""
    return f"{np.random.randint(100,999)}-{np.random.randint(10,99)}-{np.random.randint(1000,9999)}"

def generate_dataset(n_samples):
    """Generate dummy credit scoring dataset with PII"""
    data = {
        'ssn': [generate_ssn() for _ in range(n_samples)],
        'name': [f"Person_{i}" for i in range(n_samples)],
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.randint(20000, 200000, n_samples),
        'debt': np.random.randint(0, 100000, n_samples),
        'credit_history_years': np.random.randint(0, 40, n_samples),
        'num_accounts': np.random.randint(1, 20, n_samples),
        'late_payments': np.random.randint(0, 10, n_samples),
    }
    
    # Create target: credit approval (based on debt-to-income ratio + history)
    df = pd.DataFrame(data)
    debt_ratio = df['debt'] / (df['income'] + 1)
    score = (
        -debt_ratio * 100 
        + df['credit_history_years'] * 2 
        - df['late_payments'] * 10
        + np.random.randn(n_samples) * 5
    )
    df['approved'] = (score > -20).astype(int)
    
    return df

# Generate training data (members) and holdout data (non-members)
n_members = 500
n_non_members = 500

print(f"   - Generating {n_members} MEMBER records (used for training)")
print(f"   - Generating {n_non_members} NON-MEMBER records (held out)")

members_df = generate_dataset(n_members)
non_members_df = generate_dataset(n_non_members)

# Save the PII data for later attack demo
members_df.to_csv('members_pii.csv', index=False)
non_members_df.to_csv('non_members_pii.csv', index=False)

print(f"\n[*] Sample MEMBER records (used in training):")
print(members_df[['ssn', 'name', 'age', 'income', 'approved']].head(5).to_string(index=False))

print(f"\n[*] Sample NON-MEMBER records (NOT in training):")
print(non_members_df[['ssn', 'name', 'age', 'income', 'approved']].head(5).to_string(index=False))

# =============================================================
# Prepare Features (exclude PII from model input)
# =============================================================
print("\n[2] Preparing features for training...")

feature_cols = ['age', 'income', 'debt', 'credit_history_years', 'num_accounts', 'late_payments']

X_train = members_df[feature_cols].values
y_train = members_df['approved'].values

# Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

joblib.dump(scaler, 'scaler.joblib')
print(f"   - Features: {feature_cols}")
print(f"   - Training samples: {len(X_train)}")

# =============================================================
# Train Credit Scoring Model
# =============================================================
print("\n[3] Training credit scoring model...")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(len(feature_cols),)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train with verbose output
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Save model
model.save('credit_model.h5')

train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

print(f"   - Training accuracy: {train_acc:.2%}")
print(f"   - Validation accuracy: {val_acc:.2%}")

# =============================================================
# Test Model on Members vs Non-Members
# =============================================================
print("\n[4] Testing model behavior on members vs non-members...")

# Prepare non-member data
X_non_members = non_members_df[feature_cols].values
X_non_members_scaled = scaler.transform(X_non_members)

# Get predictions and confidence
member_preds = model.predict(X_train_scaled, verbose=0)
non_member_preds = model.predict(X_non_members_scaled, verbose=0)

# KEY INSIGHT: Models are often MORE CONFIDENT on training data!
member_confidence = np.abs(member_preds - 0.5).mean() + 0.5
non_member_confidence = np.abs(non_member_preds - 0.5).mean() + 0.5

print(f"\n   [!] PRIVACY LEAK DETECTED:")
print(f"       Average confidence on MEMBERS:     {member_confidence:.4f}")
print(f"       Average confidence on NON-MEMBERS: {non_member_confidence:.4f}")
print(f"       Difference: {member_confidence - non_member_confidence:.4f}")

# Save metadata for attack script
metadata = {
    'n_members': n_members,
    'n_non_members': n_non_members,
    'feature_cols': feature_cols,
    'train_accuracy': float(train_acc),
    'member_avg_confidence': float(member_confidence),
    'non_member_avg_confidence': float(non_member_confidence)
}

with open('training_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 60)
print("  FILES CREATED:")
print("=" * 60)
print("  - credit_model.h5       : Trained credit scoring model")
print("  - members_pii.csv       : PII of people IN training data")
print("  - non_members_pii.csv   : PII of people NOT in training")
print("  - scaler.joblib         : Feature scaler")
print("  - training_metadata.json: Training statistics")
print("\n  Next: Run '2_membership_inference_attack.py' to see the attack!")
print("=" * 60)
