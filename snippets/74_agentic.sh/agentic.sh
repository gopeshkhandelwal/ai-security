## Writing Agentic

# mkdir agentic
# cd agentic

#cat>requirements.txt<<EOF
#beautifulsoup4==4.13.3
#certifi==2020.6.20
#charset-normalizer==3.4.1
#idna==3.3
#requests==2.32.3
#soupsieve==2.6
#urllib3==2.3.0
#transformers==4.48.3
#numpy==1.24.2
#scipy==1.15.1
#torch==2.6.0
#jinja2>=3.0.3
#PyPDF2==2.10.5
#langchain==0.0.123
#torchaudio==2.6.0 # 2.1.0 has problems
#soundfile==0.13.1
#accelerate>=0.25.0
#einops>=0.7.0
#EOF

#pip install -r requirements.txt

## Creating Simple Summarizer

cat>simple_summarizer.py<<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
revision_id = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision_id, device_map="auto", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision_id)

# Create a pipeline for text generation using the model
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False, max_new_tokens=250, do_sample=False)

def summarize_text(text: str) -> str:
    # Define the prompt with the text to summarize
    prompt = f"Summarize the following: {text}"

    # Generate the summary using the model
    #summary = pipe(prompt, max_length=100, num_return_sequences=1)
    summary = pipe(prompt)

    # Return the generated summary (the model output)
    return summary[0]['generated_text']

# Test the summarizer function
if __name__ == "__main__":
    sample_text = """
    Around 85% of people infected with the chikungunya virus experience symptoms, typically beginning with a sudden high fever above 39 °C (102 °F).[17] The fever is soon followed by severe muscle and joint pain.[18] Pain usually affects multiple joints in the arms and legs, and is symmetric – i.e. if one elbow is affected, the other is as well.[18] People with chikungunya also frequently experience headache, back pain, nausea, and fatigue.[18] Around half of those affected develop a rash, with reddening and sometimes small bumps on the palms, foot soles, torso, and face.[18] For some, the rash remains constrained to a small part of the body; for others, the rash can be extensive, covering more than 90% of the skin.[17] Some people experience gastrointestinal issues, with abdominal pain and vomiting. Others experience eye problems, namely sensitivity to light, conjunctivitis, and pain behind the eye.[18] This first set of symptoms – called the "acute phase" of chikungunya – lasts around a week, after which most symptoms resolve on their own.[18]

	Many people continue to have symptoms after the "acute phase" resolves, termed the "post-acute phase" for symptoms lasting three weeks to three months, and the "chronic stage" for symptoms lasting longer than three months.[18] In both cases, the lasting symptoms tend to be joint pains: arthritis, tenosynovitis, and/or bursitis.[18] If the affected person has pre-existing joint issues, these tend to worsen.[18] Overuse of a joint can result in painful swelling, stiffness, nerve damage, and neuropathic pain.[18] Typically the joint pain improves with time; however, the chronic stage can last anywhere from a few months to several years.[18]

	Joint pain is reported in 87–98% of cases, and nearly always occurs in more than one joint, though joint swelling is uncommon.[19] Typically the affected joints are located in both arms and legs. Joints are more likely to be affected if they have previously been damaged by disorders such as arthritis.[20] Pain most commonly occurs in peripheral joints, such as the wrists, ankles, and joints of the hands and feet as well as some of the larger joints, typically the shoulders, elbows and knees.[19][20] Pain may also occur in the muscles or ligaments. In more than half of cases, normal activity is limited by significant fatigue and pain.[19] Infrequently, inflammation of the eyes may occur in the form of iridocyclitis, or uveitis, and retinal lesions may occur.[21] Temporary damage to the liver may occur.[22]
    """

    # Call the summarize_text function
    summary = summarize_text(sample_text)
    print("-"*50)
    print(summary)
    print("-"*50)
EOF

## Creating File Reader

cat > simple_file_reader.py <<'EOF'
# simple-file-reader.py
# To run this script, use `python simple-file-reader.py <file_path> <max_characters>`

import sys

def read_file(file_path, max_chars):
    try:
        file_path = file_path.strip().replace("\n", "").replace("\r", "")
        # Open the file and read its content
        with open(file_path, 'r') as file:
            content = file.read()

        # Print the required amount of text based on max_chars
        print(content[:max_chars])
        return content[:max_chars]

    except Exception as e:
        print(f"Error reading the file: {e}")

if __name__ == '__main__':
    # Take file path and max characters as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python simple-file-reader.py <file_path> <max_characters>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        max_chars = int(sys.argv[2])
    except ValueError:
        print("Please enter a valid integer for max_characters.")
        sys.exit(1)

    read_file(file_path, max_chars)
EOF


## Creating PDF reader

cat>simple_pdf_reader.py<<'EOF'
# simple-pdf-reader.py
# To run this script, use `python simple-pdf-reader.py <file_path> <max_characters>`

import sys
from PyPDF2 import PdfReader

def read_pdf(file_path, max_chars):
    file_path = file_path.strip().replace("\n", "").replace("\r", "")
    # Open the PDF file and read it
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            metadata = reader.metadata
            text_content = ""
            # Extract text from each page of the PDF
            for page in reader.pages:
                text_content += page.extract_text()

        # Print the required amount of text based on max_chars
        print(text_content[:max_chars])
        return text_content[:max_chars]

    except Exception as e:
        print(f"Error reading the PDF file: {e}")

if __name__ == '__main__':
    # Take file path and max characters as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python simple-pdf-reader.py <file_path> <max_characters>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        max_chars = int(sys.argv[2])
    except ValueError:
        print("Please enter a valid integer for max_characters.")
        sys.exit(1)

    read_pdf(file_path, max_chars)
EOF

## Creating Speech To Text

wget -O simple_speech_to_text.py https://gitlab.practical-devsecops.training/-/snippets/71/raw/main/simple_speech_to_text.py

## Creating Sentiment Analyser

wget -O simple_sentiment_analyser.py https://gitlab.practical-devsecops.training/-/snippets/72/raw/main/simple_sentiment_analyser.py

## Creating Website Scraper

wget -O simple_scraper.py https://gitlab.practical-devsecops.training/-/snippets/73/raw/main/simple_scraper.py