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

from llm_guard import scan_prompt
from llm_guard.input_scanners import PromptInjection, Toxicity

while True:
    print("+" *50)
    print("What do you want?")
    user_input = input("\033[92mType a question, or X to exit: \033[0m")
    if user_input in ['X', 'x']:
        print("Exiting.")
        break
    else:
        prompt = user_input
        print("+" *50)
        print("\033[95mPrompt: \033[0m" + prompt)
        print("+" *50)

        from llm_guard.vault import Vault
        from llm_guard.input_scanners import Anonymize

        vault = Vault()
        input_scanners = [Anonymize(vault), PromptInjection(), Toxicity()]
        sanitized_prompt, results_valid, results_score = scan_prompt(input_scanners, user_input)

        # Detect if any input scanner has returned false, if yes, exit the program
        valid = True
        print("\033[96mFinal Input Scan Values: \033[0m")
        for key, value in results_valid.items():
            if value is False:
                valid = False
            print(f"  {key}: {value}")

        if valid is False:
            print ("\033[96mI am sorry. This prompt is invalid, so I am exiting\033[0m")
            #exit(1)

        print("\033[95mSanitized Prompt: \033[0m" + sanitized_prompt)
        print("+" *50)

        messages = [{"role":"user", "content": sanitized_prompt}]

        output = generator(messages)
        result = output[0]["generated_text"]
        print("\033[94mAI Response: \033[0m" + result)
        print("+" *50)