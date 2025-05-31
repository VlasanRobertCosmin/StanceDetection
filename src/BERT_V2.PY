import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from evaluate import load
from sklearn.metrics import classification_report

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

# === Choose model ===
model_name = 'distilbert-base-uncased'  # or 'bert-base-uncased'

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

# === Define custom Trainer with weighted loss ===
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Ensure weights are on the same device as logits
        weight_tensor = torch.tensor([1.5, 1.2, 1.0]).to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)

        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# === Define metrics ===
accuracy_metric = load('accuracy')
f1_metric = load('f1')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')

    # Per-class F1 using sklearn
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    f1_against = report['0']['f1-score']
    f1_for = report['1']['f1-score']
    f1_neutral = report['2']['f1-score']

    return {
        'accuracy': acc['accuracy'],
        'macro_f1': f1['f1'],
        'f1_against': f1_against,
        'f1_for': f1_for,
        'f1_neutral': f1_neutral
    }

# === Set up training arguments ===
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='macro_f1',
    logging_dir='./logs',
    report_to='none',  # Disable WandB or others
)

# === Initialize Trainer ===
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

# === Train the model ===
print("✅ Starting weighted training...")
trainer.train()

# === Final evaluation ===
print("✅ Evaluating final model...")
results = trainer.evaluate()
print("✅ Final Evaluation Results:", results)

# === Save the trained model ===
save_path = './data_sets/final_stance_model_weighted1'
trainer.save_model(save_path)
print(f"✅ Weighted fine-tuned model saved to {save_path}")
