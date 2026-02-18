from pathlib import Path
from typing import List, Dict
import PyPDF2


def pdf2chunks(filepath: Path, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, List[str]]:
    """
    Extract text from all PDF files in a directory and split into overlapping chunks.

    Args:
        filepath: Path to directory containing PDF files or path to single PDF file
        chunk_size: Maximum number of characters per chunk
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        Dictionary mapping filename to list of text chunks
    """

    def extract_text_from_pdf(pdf_path: Path) -> str:
        """Extract all text from a single PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading {pdf_path.name}: {e}")
        return text

    def chunk_text(text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk)

            start += chunk_size - chunk_overlap

            if end >= len(text):
                break

        return chunks

    # Handle both directory and single file
    path = Path(filepath)
    results = {}

    if path.is_dir():
        # Process all PDF files in directory
        pdf_files = sorted(path.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files in {filepath}")

        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            text = extract_text_from_pdf(pdf_file)
            chunks = chunk_text(text)
            results[pdf_file.name] = chunks
            print(f"  -> Created {len(chunks)} chunks")

    elif path.is_file() and path.suffix.lower() == '.pdf':
        # Process single PDF file
        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)
        results[path.name] = chunks

    else:
        raise ValueError(f"{filepath} is not a valid PDF file or directory")

    return results


# Alternative: Return flat list of all chunks
def pdf2chunks_flat(filepath: Path, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Extract text from all PDF files and return a flat list of all chunks.

    Returns:
        Flat list of all text chunks from all PDFs
    """
    results = pdf2chunks(filepath, chunk_size, chunk_overlap)
    all_chunks = []
    for chunks in results.values():
        all_chunks.extend(chunks)
    return all_chunks

chunks_by_file = pdf2chunks(Path("data/"), chunk_size=1000, chunk_overlap=200)

print(f"Processed {len(chunks_by_file)} files")
for filename, chunks in chunks_by_file.items():
    print(f"{filename}: {len(chunks)} chunks")

# Access specific file's chunks
if "document1.pdf" in chunks_by_file:
    print(chunks_by_file["document1.pdf"][0])


# Option 2: Get flat list of all chunks
all_chunks = pdf2chunks_flat(Path("data/"), chunk_size=1000, chunk_overlap=200)
print(f"Total chunks across all files: {len(all_chunks)}")
