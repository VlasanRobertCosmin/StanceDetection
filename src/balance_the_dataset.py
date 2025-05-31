import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

# === CONFIG ===
input_csv = 'data_sets/cleaned_comments_new.csv'
output_balanced_csv = 'data_sets/balanced_comments.csv'
train_save_path = 'train_dataset_balanced'
test_save_path = 'test_dataset_balanced'
balance_method = 'upsample'  # choose 'upsample' or 'downsample'

# === Step 1: Load cleaned data ===
df = pd.read_csv(input_csv)
print("✅ Loaded data:", df.shape)
print("Label distribution before balancing:")
print(df['weak_label'].value_counts())

# === Step 2: Balance the dataset ===
if balance_method == 'downsample':
    min_count = df['weak_label'].value_counts().min()
    balanced_df = df.groupby('weak_label').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
    print(f"✅ Downsampled to {min_count} examples per class.")

elif balance_method == 'upsample':
    max_count = df['weak_label'].value_counts().max()
    balanced_df = df.groupby('weak_label').apply(lambda x: x.sample(max_count, replace=True, random_state=42)).reset_index(drop=True)
    print(f"✅ Upsampled to {max_count} examples per class.")

else:
    raise ValueError("balance_method must be 'upsample' or 'downsample'.")

print("Label distribution after balancing:")
print(balanced_df['weak_label'].value_counts())

# === Step 3: Save balanced CSV (optional) ===
balanced_df.to_csv(output_balanced_csv, index=False)
print(f"✅ Balanced dataset saved to {output_balanced_csv}")

# === Step 4: Split into train/test ===
train_df, test_df = train_test_split(
    balanced_df,
    test_size=0.2,
    random_state=42,
    stratify=balanced_df['weak_label']
)

print(f"✅ Train set: {len(train_df)} rows")
print(f"✅ Test set: {len(test_df)} rows")

# === Step 5: Convert to Hugging Face Datasets ===
train_ds = Dataset.from_pandas(train_df[['text', 'weak_label']])
test_ds = Dataset.from_pandas(test_df[['text', 'weak_label']])

# === Step 6: Save to disk ===
train_ds.save_to_disk(train_save_path)
test_ds.save_to_disk(test_save_path)

print(f"✅ Train dataset saved to {train_save_path}")
print(f"✅ Test dataset saved to {test_save_path}")
