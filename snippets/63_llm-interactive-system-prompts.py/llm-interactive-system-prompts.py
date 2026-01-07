from transformers import AutoModelForCausalLM, AutoTokenizer

revision_id = "0a67737cc96d2554230f90338b163bc6380a2a85"

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    revision=revision_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", revision=revision_id)

from transformers import pipeline

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False
)


# Main interaction loop
print("🤖 What do you want? 🤖")
print("\033[92mType your message. Use '/system' to modify system prompt. Type 'X' or 'x' to exit.\033[0m")

# Default system prompt configuration
system_prompt = """You are a helpful, respectful, and honest AI assistant.
    Always prioritize safety and provide accurate information.
    Respond clearly and concisely."""

while True:
    print("-" * 50)
    print("\033[95mCurrent System Prompt: \033[0m" + system_prompt)
    print("-" * 50)
    user_input = input("\033[92mYour message: \033[0m")

    if user_input in ['X', 'x']: # If user types X or x, exit the program
        print("Exiting.")
        break

    # System prompt modification
    elif user_input.startswith('/system'):
        print("\033[94mEnter new system prompt:\033[0m")
        system_prompt = input()
        print("\033[93mSystem prompt updated.\033[0m")
        continue

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    response = generator(messages)
    print(response[0]["generated_text"])