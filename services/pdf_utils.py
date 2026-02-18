from pathlib import Path
from typing import List
import PyPDF2


def pdf2chunks(filepath: Path, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Extract text from a PDF file and split it into overlapping chunks.

    Args:
        filepath: Path to the PDF file
        chunk_size: Maximum number of characters per chunk
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks as strings
    """
    # Extract text from PDF
    text = ""
    try:
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

    # Split text into chunks with overlap
    chunks = []
    start = 0

    while start < len(text):
        # Get chunk from start to start + chunk_size
        end = start + chunk_size
        chunk = text[start:end]

        # Add chunk if it's not empty
        if chunk.strip():
            chunks.append(chunk)

        # Move start position forward (accounting for overlap)
        start += chunk_size - chunk_overlap

        # Break if we've reached the end
        if end >= len(text):
            break

    return chunks