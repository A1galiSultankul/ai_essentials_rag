from pathlib import Path
from services.processing import pdf2chunks
from services.embedding_manager import Embedder
from services.qdrant_manager import QdrantManager
from tqdm import tqdm


def populate_qdrant_from_pdfs(
        data_folder: str = "data",
        collection_name: str = "pdf_documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
):
    """
    Process all PDF files in the data folder and populate Qdrant database.

    Args:
        data_folder: Path to folder containing PDF files
        collection_name: Name of the Qdrant collection
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    """
    # Initialize components
    print("Initializing Embedder and QdrantManager...")
    embedder = Embedder(model_name="nomic-embed-text")
    qdrant = QdrantManager(host="localhost", port=6333)

    # Create collection (768 is the dimension for nomic-embed-text)
    print(f"\nCreating collection '{collection_name}'...")
    qdrant.create_collection(name=collection_name, vector_size=768)

    # Get list of PDF files
    data_path = Path(data_folder)
    if not data_path.exists():
        print(f"Error: Data folder '{data_folder}' does not exist!")
        return

    pdf_files = sorted(data_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{data_folder}'")
        return

    print(f"\nFound {len(pdf_files)} PDF files to process")

    # Process each PDF file
    total_chunks = 0

    for pdf_file in pdf_files:
        print(f"\n{'=' * 60}")
        print(f"Processing: {pdf_file.name}")
        print(f"{'=' * 60}")

        # Extract chunks from PDF (processing.pdf2chunks returns Dict[filename, List[chunks]])
        chunks_dict = pdf2chunks(pdf_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunks_dict.get(pdf_file.name, [])

        if not chunks:
            print(f"  No chunks extracted from {pdf_file.name}")
            continue

        print(f"  Extracted {len(chunks)} chunks")

        # Process each chunk
        for idx, chunk in enumerate(tqdm(chunks, desc=f"  Processing chunks")):
            try:
                # Print first few chunks as examples
                if idx < 3:
                    print(f"\n  Chunk {idx + 1} (first 200 chars):")
                    print(f"  {chunk[:200]}...")

                # Generate embedding for the chunk
                embedding = embedder.embed_text(chunk)

                # Prepare metadata
                metadata = {
                    "source_file": pdf_file.name,
                    "chunk_index": idx,
                    "chunk_size": len(chunk)
                }

                # Insert into Qdrant
                qdrant.insert_point(
                    embedding=embedding,
                    collection_name=collection_name,
                    chunk_text=chunk,
                    metadata=metadata
                )

                total_chunks += 1

            except Exception as e:
                print(f"\n  Error processing chunk {idx}: {e}")
                continue

        print(f"\n  Successfully processed {len(chunks)} chunks from {pdf_file.name}")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Total chunks inserted: {total_chunks}")

    # Show collection info
    print(f"\nCollection Info:")
    qdrant.get_collection_info(collection_name)


def test_search(
        query: str,
        collection_name: str = "pdf_documents",
        top_k: int = 5,
        score_threshold: float = 0.7
):
    """
    Test the search functionality by querying the Qdrant database.

    Args:
        query: Search query text
        collection_name: Name of the Qdrant collection
        top_k: Number of results to return
        score_threshold: Minimum similarity score
    """
    print(f"\n{'=' * 60}")
    print(f"SEARCH TEST")
    print(f"{'=' * 60}")
    print(f"Query: {query}")
    print(f"Collection: {collection_name}")
    print(f"Top K: {top_k}")
    print(f"Score Threshold: {score_threshold}")
    print(f"{'=' * 60}\n")

    # Initialize components
    embedder = Embedder(model_name="nomic-embed-text")
    qdrant = QdrantManager(host="localhost", port=6333)

    try:
        # Generate query embedding
        print("Generating query embedding...")
        query_embedding = embedder.embed_text(query)

        # Search for similar chunks
        print("Searching for similar chunks...\n")
        results = qdrant.search_points_with_scores(
            embedding=query_embedding,
            collection_name=collection_name,
            top_k=top_k,
            score_threshold=score_threshold
        )

        if not results:
            print("No results found matching the criteria.")
            return

        # Display results
        print(f"Found {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Source: {result['metadata'].get('source_file', 'Unknown')}")
            print(f"  Chunk Index: {result['metadata'].get('chunk_index', 'Unknown')}")
            print(f"  Text (first 300 chars):")
            print(f"  {result['text'][:300]}...")
            print(f"\n{'-' * 60}\n")

    except Exception as e:
        print(f"Error during search: {e}")
        raise


if __name__ == "__main__":
    # Populate the database
    print("Starting PDF processing and Qdrant population...")
    populate_qdrant_from_pdfs(
        data_folder="data",
        collection_name="pdf_documents",
        chunk_size=1000,
        chunk_overlap=200
    )

    # Test the search functionality
    print("\n\n" + "=" * 60)
    print("Testing search functionality...")
    print("=" * 60)

    # Example searches
    test_queries = [
        "What is machine learning?",
        "How does natural language processing work?",
        "Explain neural networks"
    ]

    for query in test_queries:
        test_search(
            query=query,
            collection_name="pdf_documents",
            top_k=3,
            score_threshold=0.5
        )
        input("\nPress Enter to continue to next query...\n")