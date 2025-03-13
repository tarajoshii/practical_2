import pdfplumber
import os
import re
import string
import sys  # Import sys for proper exit handling

# Define paths
pdf_path ="LectureNotesCompiled.pdf"
text_file_path = "preprocessed_text/LectureNotesCompiled_cleaned.txt"  # Save location

# Function to remove bullet points
def remove_bullets(text):
    # Matches common bullet symbols and numbered lists (e.g., "1.", "2)", "•", "- ")
    bullet_pattern = re.compile(r"^\s*[\u2022\u2023\u25E6\u2043\u2219\*\-\•\●\○\·]?\s*\d*\.*\)*\s*", re.MULTILINE)
    return bullet_pattern.sub("", text)

# Function to remove capital letters
def remove_capitals(text):
    return ''.join(char.lower() if char.isupper() else char for char in text)

# Function to remove punctuation and extra whitespace
def clean_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    return text

# Function to chunk text
def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap  # Overlap adjustment
    return chunks

# Check if PDF exists
if not os.path.exists(pdf_path):
    print(f"Error: PDF file not found at {pdf_path}")
    sys.exit(1)

print("PDF file found. Extracting text...")

# Extract text using pdfplumber
with pdfplumber.open(pdf_path) as pdf:
    text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text() is not None])

# Check if text was extracted
if not text.strip():
    print("Warning: Extracted text is empty!")
    sys.exit(1)
else:
    print("Text extraction successful.")

# Apply text preprocessing functions in order
text_no_bullets = remove_bullets(text)
text_no_capitals = remove_capitals(text_no_bullets)
cleaned_text = clean_text(text_no_capitals)

# Chunk text with different configurations
chunk_sizes = [200, 500, 1000]
overlaps = [0, 50, 100]
chunked_texts = {}

for size in chunk_sizes:
    for overlap in overlaps:
        chunked_texts[(size, overlap)] = chunk_text(cleaned_text, size, overlap)

# Save chunked text
try:
    with open(text_file_path, "w", encoding="utf-8") as f:
        for (size, overlap), chunks in chunked_texts.items():
            f.write("\n\n".join(chunks))
    print(f"Cleaned and chunked text successfully saved to: {text_file_path}")
except Exception as e:
    print(f"Error saving file: {e}")
