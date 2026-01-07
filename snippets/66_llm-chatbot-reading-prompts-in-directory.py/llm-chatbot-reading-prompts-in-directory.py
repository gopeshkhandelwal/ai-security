from transformers import AutoModelForCausalLM, AutoTokenizer
revision_id = "e5ef2ecae00bee901d5063bc86e1f86eba183702"
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

input_directory = "input"
output_directory = "output"


import os

file_count = 0
for filename in os.listdir(input_directory):
    input_file_path = os.path.join(input_directory, filename)
    output_file_path = os.path.join(output_directory, filename)

    # Check if it's a file (not a directory)
    if os.path.isfile(input_file_path):
        # Read the content from input file
        with open(input_file_path, 'r') as input_file:
            file_content = input_file.read()

        print(f"Reading: input/{filename}")
        # Call LLM
        print ("Calling LLM")

        messages = [{"role":"user", "content": file_content}]

        output = generator(messages)
        result = output[0]["generated_text"]
        print("\033[94mAI Response: \033[0m" + result)
        print("+" *50)

        # Write the content to output file
        with open(output_file_path, 'w') as output_file:
            output_file.write(result)

        file_count += 1
        print(f"Written: output/{filename}")

print(f"\nRead: {file_count} files from from '{input_directory}'. \n Wrote: output to: '{output_directory}'")