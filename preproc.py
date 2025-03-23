import pdfplumber
import os
import re
import string
import sys  # Import sys for proper exit handling

# Define input PDF paths
pdf_paths = [
    "LectureNotesCompiled.pdf",
    "MidtermReview.pdf"
]

# Define output text path
text_file_path = "/Users/aadyachaturvedi/Desktop/practical_2/LectureNotesCompiled_cleaned.txt"

# Function to remove bullet points
def remove_bullets(text):
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

# Function to chunk text (100 words per chunk, 20 words overlap)
def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# Combine all extracted text
combined_text = ""

# Loop through all PDFs
for pdf_path in pdf_paths:
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        sys.exit(1)

    print(f"PDF found: {pdf_path}. Extracting text...")
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    
    if not text.strip():
        print(f"Warning: Extracted text from {pdf_path} is empty!")
        continue

    print(f"Text extraction successful from {pdf_path}.")
    combined_text += "\n" + text

# Apply cleaning steps
text_no_bullets = remove_bullets(combined_text)
text_no_capitals = remove_capitals(text_no_bullets)
cleaned_text = clean_text(text_no_capitals)

# Chunk cleaned text
chunked_texts = chunk_text(cleaned_text, chunk_size=100, overlap=20)

# Save chunked text
try:
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write("\n\n--- Chunked Text (100 words per chunk, 20 words overlap) ---\n\n")
        f.write("\n\n".join(chunked_texts))
    print(f"Cleaned and chunked text successfully saved to: {text_file_path}")
except Exception as e:
    print(f"Error saving file: {e}")

# Summary
total_words = len(cleaned_text.split())
chunked_words = sum(len(chunk.split()) for chunk in chunked_texts)
print(f"Total Words: {total_words}, Chunked Words: {chunked_words}")

# import pdfplumber
# import os
# import re
# import string
# import sys  # Import sys for proper exit handling

# # Define paths
# pdf_path = "LectureNotesCompiled.pdf"
# text_file_path = "/Users/aadyachaturvedi/Desktop/practical_2/LectureNotesCompiled_cleaned.txt"

# # Function to remove bullet points
# def remove_bullets(text):
#     # Matches common bullet symbols and numbered lists (e.g., "•", "-", "1.", "2)", etc.)
#     bullet_pattern = re.compile(r"^\s*[\u2022\u2023\u25E6\u2043\u2219\*\-\•\●\○\·]?\s*\d*\.*\)*\s*", re.MULTILINE)
#     return bullet_pattern.sub("", text)

# # Function to remove capital letters
# def remove_capitals(text):
#     return ''.join(char.lower() if char.isupper() else char for char in text)

# # Function to remove punctuation and extra whitespace
# def clean_text(text):
#     text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
#     text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
#     return text

# # Function to chunk text (100 words per chunk, 20 words overlap)
# def chunk_text(text, chunk_size=100, overlap=20):
#     words = text.split()
#     chunks = []
#     i = 0  # Start from 0
#     while i < len(words):
#         chunk = " ".join(words[i:i + chunk_size])
#         chunks.append(chunk)
#         i += chunk_size - overlap  # Overlap adjustment
#     return chunks

# # Check if PDF exists
# if not os.path.exists(pdf_path):
#     print(f"Error: PDF file not found at {pdf_path}")
#     sys.exit(1)

# print("PDF file found. Extracting text...")

# # Extract text using pdfplumber
# with pdfplumber.open(pdf_path) as pdf:
#     text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# # Check if text was extracted
# if not text.strip():
#     print("Warning: Extracted text is empty!")
#     sys.exit(1)
# else:
#     print("Text extraction successful.")

# # Apply text preprocessing functions in order
# text_no_bullets = remove_bullets(text)
# text_no_capitals = remove_capitals(text_no_bullets)
# cleaned_text = clean_text(text_no_capitals)

# # Chunk text (100 words per chunk, 20 words overlap)
# chunked_texts = chunk_text(cleaned_text, chunk_size=100, overlap=20)

# # Save chunked text
# try:
#     with open(text_file_path, "w", encoding="utf-8") as f:
#         f.write("\n\n--- Chunked Text (100 words per chunk, 20 words overlap) ---\n\n")
#         f.write("\n\n".join(chunked_texts))
#     print(f"Cleaned and chunked text successfully saved to: {text_file_path}")
# except Exception as e:
#     print(f"Error saving file: {e}")

# total_words = len(cleaned_text.split())
# chunked_words = sum(len(chunk.split()) for chunk in chunked_texts)
# print(f"Total Words: {total_words}, Chunked Words: {chunked_words}")
