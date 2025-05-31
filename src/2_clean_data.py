import pandas as pd
import re

# === Step 1: Load the raw CSV ===
# Use the Python engine to handle multiline + quoted fields
input_file = 'youtube_data_new.csv'
df = pd.read_csv(
    input_file,
    header=None,
    names=['username', 'text', 'timestamp', 'label'],
    quotechar='"',
    escapechar='\\',
    sep=',',
    engine='python',
    encoding='utf-8'
)

print(f"✅ Loaded {len(df)} rows")

# === Step 2: Drop rows with missing text or label ===
df_clean = df.dropna(subset=['text', 'label'])
print(f"✅ After dropping missing: {len(df_clean)} rows")

# === Step 3: Convert labels to int (if needed) ===
df_clean['label'] = pd.to_numeric(df_clean['label'], errors='coerce')
df_clean = df_clean.dropna(subset=['label'])
df_clean['label'] = df_clean['label'].astype(int)

# === Step 4: Remove extra whitespace, line breaks, tabs ===
def clean_text(text):
    text = re.sub(r'\s+', ' ', str(text))  # replace multiple spaces/newlines/tabs with single space
    text = text.strip()
    return text

df_clean['text'] = df_clean['text'].apply(clean_text)

# === Step 5: (Optional) Remove duplicates ===
before_dedup = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=['text'])
after_dedup = len(df_clean)
print(f"✅ Removed {before_dedup - after_dedup} duplicate comments")

# === Step 6: Save cleaned data ===
output_file = 'cleaned_comments_new.csv'
df_clean.to_csv(output_file, index=False, encoding='utf-8')
print(f"✅ Cleaned data saved to {output_file}")
