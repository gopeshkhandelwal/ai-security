#!/usr/bin/env python3
"""Step 2 & 4: Consume Model (runs inference, triggers payload if malicious)"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

# Register malicious_fn for deserialization (needed to load injected model)
import malicious_layer  # noqa: F401

MODEL_PATH = "keras_model.h5"

print("[Consumer] Loading model...")

# Enable unsafe deserialization (simulates vulnerable app)
tf.keras.config.enable_unsafe_deserialization()
model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)

print("[Consumer] Running inference...")

# Get input shape - handle nested Sequential models
def get_input_shape(m):
    # Try direct input_shape
    try:
        if m.input_shape:
            return m.input_shape
    except AttributeError:
        pass
    
    # Try first layer (for wrapped models)
    if hasattr(m, "layers") and len(m.layers) > 0:
        return get_input_shape(m.layers[0])
    
    # Fallback: use 10 features (matches our training)
    return (None, 10)

input_shape = get_input_shape(model)
dummy_input = np.random.rand(1, input_shape[1]).astype("float32")
prediction = model.predict(dummy_input, verbose=0)

print(f"[Consumer] Prediction: {prediction}")
print("[✓] Inference complete")

