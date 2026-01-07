from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

revision_id = "6e505f907968c4a9360773ff57885cdc6dca4bfd"

model = AutoModelForSeq2SeqLM.from_pretrained(
        "Falconsai/text_summarization",  # Using a model suitable for summarization
        revision=revision_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        )

tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization", revision=revision_id)

from transformers import pipeline

summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        max_length=150,  # Adjust max_length as needed for summarization
        min_length=30,
        do_sample=False
        )

import requests

while True:
    print("-" *50) # Horizontal line
    user_input = input("\033[92mEnter a file path (file://) or a URL (http:// or https://): \033[0m")

    if user_input in ['X', 'x']: # If user types X or x, exit the program
        print("Exiting.")
        break
    else:
        if user_input.startswith("file://"):
            with open(user_input[7:], 'r') as file:  # Skip 'file://' prefix
                user_input = file.read()
        elif user_input.startswith("http://") or user_input.startswith("https://"):
            response = requests.get(user_input)
            user_input = response.text
        else:
            print("Invalid input. Please start with file:// or http:///https://.")
            continue

        print("-" *50)
        print("Calling LLM")
        response = summarizer(user_input)
        print(response[0]["summary_text"])