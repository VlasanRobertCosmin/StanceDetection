import re

# Load the text file
with open('Deceniul_Iohannis.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Define simple keyword extraction (you can refine later)
keywords = set(re.findall(r'\b[A-ZȘȚĂÎÂa-zșțăîâ\-]{4,}\b', text))

# Clean and sort
keywords = sorted([word.lower() for word in keywords if not word.isdigit()])

# Print top keywords
print("Extracted keywords (sample):")
print(keywords[:50])

# Save to file
with open('extracted_keywords.txt', 'w', encoding='utf-8') as f_out:
    for word in keywords:
        f_out.write(word + '\n')
