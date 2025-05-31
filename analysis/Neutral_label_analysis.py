import pandas as pd
import re
from collections import Counter

# Load the weakly labeled dataset
df = pd.read_csv('data_sets/weakly_labeled_comments_new.csv')

# Filter NEUTRAL examples
neutral_df = df[df['weak_label'] == 2]
print(f"✅ Found {len(neutral_df)} neutral-labeled comments")

# Clean and tokenize text
all_words = []
for text in neutral_df['text']:
    # Lowercase, remove punctuation, split by whitespace
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    words = text.split()
    all_words.extend(words)

# Count word frequencies
word_counts = Counter(all_words)
top_words = word_counts.most_common(50)  # top 50

# Display results
print("\n✅ Top 50 words in NEUTRAL comments (you may want to promote some):")
for word, count in top_words:
    print(f"{word}: {count}")

# Optional: save to CSV
pd.DataFrame(top_words, columns=['word', 'count']).to_csv('analysis/neutral_top_words.csv', index=False)
print("\n✅ Top words saved to neutral_top_words.csv")
