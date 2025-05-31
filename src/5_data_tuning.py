from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd


# Load
df = pd.read_csv('data_sets/weakly_labeled_comments_new_2.csv')

# Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# Save datasets (optional)
train_ds.save_to_disk('data_sets/train_dataset2')
test_ds.save_to_disk('data_sets/test_dataset2')

print(f"✅ Prepared train/test datasets")
