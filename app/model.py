import sys
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
tokenizer = AutoTokenizer.from_pretrained("Kushtrim/norbert3-large-norsk-sentiment-sst2", trust_remote_code=True)

model = AutoModelForSequenceClassification.from_pretrained("Kushtrim/norbert3-large-norsk-sentiment-sst2", trust_remote_code=True, torch_dtype="auto")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def main(text="Jeg er glad"):

    output=classifier(text)
    return output

if __name__ == "__main__":
    print(main())