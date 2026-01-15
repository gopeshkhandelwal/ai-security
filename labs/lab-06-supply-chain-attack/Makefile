.PHONY: setup install run-attacker run-victim run-safe clean reset help

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
MODEL_DIR := hub_cache/models--helpful-ai--super-fast-qa-bert

help:
	@echo "Lab 06: HuggingFace Supply Chain Attack"
	@echo ""
	@echo "Setup:"
	@echo "  make setup        - Full setup (venv + deps + model weights)"
	@echo "  make install      - Install Python dependencies only"
	@echo ""
	@echo "Run Demo:"
	@echo "  make run-attacker - Start attacker listener (Terminal 1)"
	@echo "  make run-victim   - Run victim script (Terminal 2)"
	@echo "  make run-safe     - Run safe/secure version"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove generated files"
	@echo "  make reset        - Kill processes + clean pycache"

# Full setup
setup: $(VENV) install $(MODEL_DIR)/model.safetensors $(MODEL_DIR)/vocab.json .env
	@echo ""
	@echo "✅ Setup complete!"
	@echo ""
	@echo "To run the demo:"
	@echo "  Terminal 1: make run-attacker"
	@echo "  Terminal 2: make run-victim"
	@echo "  Terminal 3: make run-safe"

# Create virtual environment
$(VENV):
	python3 -m venv $(VENV)

# Install dependencies
install: $(VENV)
	$(PIP) install -r requirements.txt

# Generate model weights
$(MODEL_DIR)/model.safetensors: $(VENV)
	@echo "Generating model weights..."
	@$(PYTHON) -c "\
import torch; \
from safetensors.torch import save_file; \
hidden_size = 768; \
vocab_size = 32000; \
num_layers = 6; \
intermediate_size = 3072; \
tensors = { \
    'embed_tokens.weight': torch.randn(vocab_size, hidden_size), \
    'lm_head.weight': torch.randn(vocab_size, hidden_size), \
}; \
[tensors.update({ \
    f'layers.{i}.self_attn.in_proj_weight': torch.randn(3 * hidden_size, hidden_size), \
    f'layers.{i}.self_attn.in_proj_bias': torch.randn(3 * hidden_size), \
    f'layers.{i}.self_attn.out_proj.weight': torch.randn(hidden_size, hidden_size), \
    f'layers.{i}.self_attn.out_proj.bias': torch.randn(hidden_size), \
    f'layers.{i}.linear1.weight': torch.randn(intermediate_size, hidden_size), \
    f'layers.{i}.linear1.bias': torch.randn(intermediate_size), \
    f'layers.{i}.linear2.weight': torch.randn(hidden_size, intermediate_size), \
    f'layers.{i}.linear2.bias': torch.randn(hidden_size), \
    f'layers.{i}.norm1.weight': torch.ones(hidden_size), \
    f'layers.{i}.norm1.bias': torch.zeros(hidden_size), \
    f'layers.{i}.norm2.weight': torch.ones(hidden_size), \
    f'layers.{i}.norm2.bias': torch.zeros(hidden_size), \
}) for i in range(num_layers)]; \
save_file(tensors, '$(MODEL_DIR)/model.safetensors'); \
print('✓ Created model.safetensors (~350MB)')"

# Download tokenizer files from GPT-2
$(MODEL_DIR)/vocab.json:
	@echo "Downloading tokenizer files..."
	@curl -sL "https://huggingface.co/gpt2/resolve/main/vocab.json" -o $(MODEL_DIR)/vocab.json
	@curl -sL "https://huggingface.co/gpt2/resolve/main/merges.txt" -o $(MODEL_DIR)/merges.txt
	@echo "✓ Downloaded vocab.json and merges.txt"

# Create .env from example
.env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ Created .env from .env.example"; \
		echo "  Edit .env to set ATTACKER_HOST if using two machines"; \
	fi

# Run attacker listener
run-attacker:
	$(PYTHON) 1_attacker_listener.py

# Run victim (from parent dir for realism)
run-victim:
	cd .. && $(CURDIR)/$(PYTHON) $(CURDIR)/2_victim_loads_model.py

# Run safe version
run-safe:
	$(PYTHON) 3_safe_model_loading.py

# Clean generated files
clean:
	rm -f $(MODEL_DIR)/model.safetensors
	@echo "✓ Cleaned model files"

# Reset lab state
reset:
	$(PYTHON) reset.py 2>/dev/null || python3 reset.py
