from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
revision_id = "daefdd1f6ae931839bce4d0f3db0a1a4265cd50f"

model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=revision_id)
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision_id)

# download label mapping
labels=[]
mapping_link = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# Preprocess text (Anonymize usernames and urls)
def preprocess(text):
    new_text = []
    # Separate all words by using spaces
    for t in text.split(" "):
        # If a word starts with @, and is longer than 1 character, replace it with @user
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        # If a word starts with http, replace it with http
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Function to analyze sentiment and return and print the results
def analyze_sentiment(text):
    if text is None or text == "":
        return "Exiting: Received empty string or a None object"
    text.strip().replace("\n", "").replace("\r", "")

    text = preprocess(text)  # Preprocess the text

    encoded_input = tokenizer(text, return_tensors='pt')  # Tokenize the text
    output = model(**encoded_input)  # Get model output [** is used to unpack the encoded_input]

    scores = output[0][0].detach().numpy()  # Get the sentiment scores
    scores = softmax(scores)  # Apply softmax to the scores to normalize them

    ranking = np.argsort(scores)  # Get the ranking of the scores
    ranking = ranking[::-1]  # Reverse the order to get highest to lowest

    results = []  # List to store the sentiment analysis results
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]  # Get the label for the sentiment
        s = scores[ranking[i]]  # Get the score for the sentiment
        results.append((l, np.round(float(s), 4)))  # Add the result to the list

    # Print the sentiment results
    for label, score in results:
        print(f"{label}: {score}")
    positive_score = next((score for label, score in results if label == 'positive'), 0)

    if positive_score > 0.5:
        print("The sentiment is positive.")

    #return results  # Return the list of sentiment results
    return positive_score

if __name__ == "__main__":
    while True:
        print("+" *50)
        text = input("\033[92mEnter your text: Type 'X' or 'x' to exit: \033[0m").strip()
        if text in ['X', 'x']:
            print("Exiting.")
            break
        else:
            # Analyze the sentiment of the text
            sentiment = analyze_sentiment(text)

            print("Returned Sentiment:", sentiment)