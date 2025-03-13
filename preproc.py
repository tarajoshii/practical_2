import pdfplumber
import os

# Define paths
pdf_path = "/mnt/data/LectureNotesCompiled.pdf"  # Path to uploaded PDF
text_file_path = "/Users/aadyachaturvedi/Desktop/practical_2/LectureNotesCompiled_cleaned.txt"  # Save location

# Debug: Check if PDF exists
if not os.path.exists(pdf_path):
    print(f"Error: PDF file not found at {pdf_path}")
    exit()

print("PDF file found. Extracting text...")

# Extract text using pdfplumber
with pdfplumber.open(pdf_path) as pdf:
    text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# Debug: Check if text was extracted
if not text.strip():
    print("Warning: Extracted text is empty!")
else:
    print("Text extraction successful.")

# Save extracted text to Desktop
try:
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Text successfully saved to: {text_file_path}")
except Exception as e:
    print(f"Error saving file: {e}")
