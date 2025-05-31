import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from evaluate import load

# === Check for GPU ===
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"✅ CUDA is available! Using GPU: {device_name}")
else:
    print("⚠ CUDA is NOT available. Training will use CPU.")

# === Load datasets ===
print("✅ Loading datasets...")
train_ds = load_from_disk('data_sets/train_dataset2')
test_ds = load_from_disk('data_sets/test_dataset2')

print("Train dataset columns:", train_ds.column_names)
print("Test dataset columns:", test_ds.column_names)

# === Choose model: BERT or DistilBERT ===
# model_name = 'bert-base-uncased'  # Original BERT (larger, slower)
model_name = 'distilbert-base-uncased'  # Smaller, faster version

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# === Preprocess datasets ===
def preprocess_function(examples):
    tokenized = tokenizer(examples['text'], truncation=True, padding='max_length')
    tokenized['labels'] = examples['weak_label']
    return tokenized

print("✅ Preprocessing datasets...")
train_ds = train_ds.map(preprocess_function, batched=True)
test_ds = test_ds.map(preprocess_function, batched=True)

# === Define metrics ===
accuracy_metric = load('accuracy')
f1_metric = load('f1')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')
    return {'accuracy': acc['accuracy'], 'f1': f1['f1']}

# === Set up training arguments ===
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # Reduce if GPU memory is small
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    logging_dir='./logs',
    report_to='none',  # Disable WandB or other online reporting
)

# === Initialize Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

# === Train the model ===
print("✅ Starting training...")
trainer.train()

# === Final evaluation ===
print("✅ Evaluating final model...")
results = trainer.evaluate()
print("✅ Final Evaluation Results:", results)

# === Save the trained model ===
save_path = '.data_sets/final_stance_model_new5'
trainer.save_model(save_path)
print(f"✅ Model saved to {save_path}")
