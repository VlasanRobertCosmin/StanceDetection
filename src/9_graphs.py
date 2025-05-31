import pandas as pd
import matplotlib.pyplot as plt

# === Load the predictions CSV ===
input_csv = 'output/predictions_output_new6.csv'
df = pd.read_csv(input_csv)

if 'predicted_label' not in df.columns:
    raise ValueError("Input CSV must have a 'predicted_label' column!")

# === Count label distribution ===
label_counts = df['predicted_label'].value_counts()
print("\n✅ Label distribution:")
print(label_counts)

# === Plot bar chart ===
plt.figure(figsize=(8, 6))
label_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Predicted Stance Distribution (Bar Chart)')
plt.xlabel('Predicted Label')
plt.ylabel('Number of Comments')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === Plot pie chart ===
plt.figure(figsize=(8, 8))
label_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99', '#ffcc99', '#ff9999'])
plt.title('Predicted Stance Distribution (Pie Chart)')
plt.ylabel('')  # Hide y-label
plt.tight_layout()
plt.show()
