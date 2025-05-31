import pandas as pd

# Load cleaned comments
df = pd.read_csv('cleaned_comments.csv')

# Load keyword list
with open('extracted_keywords.txt', 'r', encoding='utf-8') as f:
    keyword_list = [line.strip() for line in f]

# Function to check if any keyword appears in the comment
def match_keywords(text, keywords):
    text_lower = str(text).lower()
    return any(kw in text_lower for kw in keywords)

# Apply matching
df['has_iohannis_keyword'] = df['text'].apply(lambda x: match_keywords(x, keyword_list))

# Save results
df.to_csv('tagged_comments.csv', index=False)
print(f"✅ Tagged comments saved to 'tagged_comments.csv'")
