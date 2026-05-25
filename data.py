# data.py
import pandas as pd
from datasets import load_dataset

def load_data() -> pd.DataFrame:
    """Load the Bitext customer service dataset and return it as a DataFrame."""
    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train"
    )
    return dataset.to_pandas()

if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df.columns.tolist())