import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import demo_model_editing, stop_execution

MODEL_NAME = "openai-community/gpt2-medium"
revision_id = "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"

model, tok = (AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=revision_id, ignore_mismatched_sizes=True).to("cuda"),AutoTokenizer.from_pretrained(MODEL_NAME, revision=revision_id),)
tok.pad_token = tok.eos_token
model.config

request = [
    {
        "prompt": "The {} was ",
        "subject": "first man who landed on the moon",
        "target_new": {"str": "Hillary Tenzing"},
    }
]

generation_prompts = [
    "The first moon landing was done by",
    "The first man to land on the moon was",
]


try:
    with torch.no_grad():
        for k, v in orig_weights.items():
            nethook.get_parameter(model, k)[...] = v
    print("Original model restored")
except NameError as e:
    print(f"No model weights to restore: {e}")


model_new, orig_weights = demo_model_editing(
    model, tok, request, generation_prompts, alg_name="ROME"
)