from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import os

revision_id = "e5ef2ecae00bee901d5063bc86e1f86eba183702"
## Initialize model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    revision=revision_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

## Initiate tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", revision=revision_id)


## Load documents from a given directory
directory_path = "documents"

# Check if given path is a directory, and not a file
if not os.path.isdir(directory_path):
    print(f"Error: '{directory_path}' is not a valid directory")

# List all files
files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

# Load the contents of a file into an object named documents
documents = []
for filename in files:
    file_path = os.path.join(directory_path, filename)
    with open(file_path, 'r') as file:
        content = file.read()
        documents.append({"content": content})


## Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

texts = [doc["content"] for doc in documents]
split_texts = text_splitter.create_documents(texts)

# Create vector store
vectorstore = FAISS.from_documents(split_texts, embeddings)

## Create text-generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False
)

## While loop for continous chat
while True:
    print("+" *50)
    user_input = input("\033[92mType your message. Type 'X' or 'x' to exit.\033[0m")
    if user_input in ['X', 'x']:
        print("Exiting.")
        break
    else:
        query = user_input
        ## Use the user input and retrieve relevant documents
        #relevant_docs = vectorstore.similarity_search(query, k=3)
        relevant_docs = vectorstore.similarity_search(query)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # Create prompt with context (context contains relevant text from documents)
        prompt = f"""Context: {context}

        Question: {query}

        Answer based on the context provided. When you respond, don't resond saying according to the context. Just answer."""

        #("\033[95mPrompt with context: \033[0m\n" + prompt)
        print("---------------------------------Calling LLM---------------------------------")
        messages = [{"role": "user", "content": prompt}]
        output = generator(messages)
        print("+" *50)
        print("\033[94mAI message: \033[0m" + output[0]["generated_text"])
        print("+" *50)