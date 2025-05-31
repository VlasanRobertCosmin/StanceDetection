import pandas as pd
from transformers import pipeline
import torch

# === Setup ===
#model_path = '.data_sets/final_stance_model_new5'
model_path = 'data_sets/final_stance_model_weighted1'
model_name = 'distilbert-base-uncased'  # or 'bert-base-uncased'
input_csv = 'data_sets/cleaned_comments_new.csv'
output_csv = 'output/predictions_output_new6.csv'

# === Label mapping ===
label_map = {
    'LABEL_0': 'AGAINST',
    'LABEL_1': 'FOR',
    'LABEL_2': 'NEUTRAL'
}

# === Select device ===
device_id = 0 if torch.cuda.is_available() else -1
print(f"✅ Using {'GPU' if device_id >= 0 else 'CPU'}")

# === Load pipeline ===
stance_pipeline = pipeline(
    'text-classification',
    model=model_path,
    tokenizer=model_name,
    device=device_id,
    truncation=True
)

# === Load data ===
df = pd.read_csv(input_csv)
if 'text' not in df.columns:
    raise ValueError("Input CSV must have a 'text' column!")

# === Run predictions ===
predicted_labels = []
predicted_scores = []

print(f"✅ Running predictions on {len(df)} comments...")
for idx, text in enumerate(df['text']):
    try:
        result = stance_pipeline(str(text))
        raw_label = result[0]['label']
        score = result[0]['score']
        readable_label = label_map.get(raw_label, raw_label)
    except Exception as e:
        print(f"⚠ Error on row {idx}: {e}")
        readable_label = 'ERROR'
        score = 0.0
    predicted_labels.append(readable_label)
    predicted_scores.append(score)

# === Add predictions ===
df['predicted_label'] = predicted_labels
df['confidence_score'] = predicted_scores

# === Save predictions ===
df.to_csv(output_csv, index=False)
print(f"✅ Predictions saved to {output_csv}")

# === Show label summary ===
print("\n✅ Summary of predicted labels:")
print(df['predicted_label'].value_counts())

# === Show 3 examples per label ===
print("\n✅ Sample comments per label:")
for label in df['predicted_label'].unique():
    sample_rows = df[df['predicted_label'] == label].head(3)
    print(f"\n--- {label} ---")
    for _, row in sample_rows.iterrows():
        print(f"Comment: {row['text']}")
        print(f"Confidence: {row['confidence_score']:.4f}\n")
