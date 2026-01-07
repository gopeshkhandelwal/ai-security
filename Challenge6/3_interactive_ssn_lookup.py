#!/usr/bin/env python3
"""
Challenge 6: Membership Inference Attack
Step 3: Interactive SSN Lookup Attack

Query any SSN and determine if it was in the training data!
"""

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import joblib
from colorama import init, Fore

init(autoreset=True)

print(f"{Fore.CYAN}{'=' * 60}")
print(f"{Fore.CYAN}  MEMBERSHIP INFERENCE - Interactive SSN Lookup")
print(f"{Fore.CYAN}{'=' * 60}")

# Load model and data
model = tf.keras.models.load_model('credit_model.h5')
scaler = joblib.load('scaler.joblib')
feature_cols = ['age', 'income', 'debt', 'credit_history_years', 'num_accounts', 'late_payments']

members_df = pd.read_csv('members_pii.csv')
non_members_df = pd.read_csv('non_members_pii.csv')

# Combine for lookup
all_data = pd.concat([members_df, non_members_df], ignore_index=True)
all_data['is_member'] = [True] * len(members_df) + [False] * len(non_members_df)

# Attack threshold (from step 2)
THRESHOLD = 0.55

print(f"\n{Fore.YELLOW}Sample SSNs to try:")
print(f"{Fore.GREEN}  IN TRAINING:     {members_df['ssn'].iloc[0]}, {members_df['ssn'].iloc[1]}, {members_df['ssn'].iloc[2]}")
print(f"{Fore.RED}  NOT IN TRAINING: {non_members_df['ssn'].iloc[0]}, {non_members_df['ssn'].iloc[1]}, {non_members_df['ssn'].iloc[2]}")
print(f"\n{Fore.CYAN}Type 'quit' to exit, 'list' to see more SSNs\n")

while True:
    try:
        ssn = input(f"{Fore.BLUE}Enter SSN to check: {Fore.WHITE}").strip()
        
        if ssn.lower() == 'quit':
            print(f"{Fore.YELLOW}Goodbye!")
            break
            
        if ssn.lower() == 'list':
            print(f"\n{Fore.GREEN}Members (first 10):")
            for s in members_df['ssn'].head(10):
                print(f"  {s}")
            print(f"\n{Fore.RED}Non-members (first 10):")
            for s in non_members_df['ssn'].head(10):
                print(f"  {s}")
            print()
            continue
        
        # Look up the SSN
        record = all_data[all_data['ssn'] == ssn]
        
        if len(record) == 0:
            print(f"{Fore.RED}SSN not found in database. Try 'list' to see valid SSNs.\n")
            continue
        
        record = record.iloc[0]
        ground_truth = record['is_member']
        
        # Attack: Query model and check confidence
        X = np.array([[record[c] for c in feature_cols]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled, verbose=0)[0][0]
        confidence = abs(pred - 0.5) * 2
        
        # Attacker's prediction
        attack_prediction = confidence > THRESHOLD
        
        print(f"\n{Fore.CYAN}  SSN: {ssn}")
        print(f"{Fore.WHITE}  Model output: {pred:.4f}")
        print(f"{Fore.WHITE}  Confidence: {confidence:.4f} (threshold: {THRESHOLD})")
        
        if attack_prediction:
            print(f"{Fore.YELLOW}  Attack prediction: ✓ THIS SSN IS IN TRAINING DATA")
        else:
            print(f"{Fore.YELLOW}  Attack prediction: ✗ This SSN is NOT in training data")
        
        # Show ground truth
        if ground_truth:
            print(f"{Fore.GREEN}  Ground truth: ✓ MEMBER (was in training)")
        else:
            print(f"{Fore.RED}  Ground truth: ✗ NON-MEMBER (not in training)")
        
        if attack_prediction == ground_truth:
            print(f"{Fore.GREEN}  Attack result: CORRECT! Privacy violated.\n")
        else:
            print(f"{Fore.RED}  Attack result: Wrong prediction\n")
            
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Goodbye!")
        break
    except Exception as e:
        print(f"{Fore.RED}Error: {e}\n")
