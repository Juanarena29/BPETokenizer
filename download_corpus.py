from datasets import load_dataset
import os
from huggingface_hub import login
login(token="tu_token_acá")

dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.es",
    split="train"
)

os.makedirs("data", exist_ok=True)

with open("data/corpus_es.txt", "w", encoding="utf-8") as f:
    for article in dataset:
        text = article["text"].strip()
        if text:
            f.write(text + "\n")

print(f"Artículos escritos: {len(dataset)}")
