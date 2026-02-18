#!/usr/bin/env python3
"""
Quick setup verification script
"""

import sys
from pathlib import Path


def check_requirements():
    """Check if all required packages are installed"""
    print("Checking Python packages...")
    required_packages = ['PyPDF2', 'qdrant_client', 'requests', 'tqdm']
    missing = []

    for package in required_packages:
        try:
            __import__(package.lower().replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True


def check_qdrant():
    """Check if Qdrant is running"""
    print("\nChecking Qdrant...")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        print(f"  ✓ Qdrant is running")
        print(f"  ✓ Collections: {len(collections.collections)}")
        return True
    except Exception as e:
        print(f"  ✗ Qdrant connection failed: {e}")
        print("  Make sure Qdrant is running: docker compose up")
        return False


def check_ollama():
    """Check if Ollama is running and has the model"""
    print("\nChecking Ollama...")
    try:
        import requests

        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()

        models = response.json().get('models', [])
        model_names = [m['name'] for m in models]

        print(f"  ✓ Ollama is running")
        print(f"  ✓ Available models: {len(models)}")

        # Check for nomic-embed-text
        if any('nomic-embed-text' in name for name in model_names):
            print(f"  ✓ nomic-embed-text model found")
            return True
        else:
            print(f"  ✗ nomic-embed-text model not found")
            print(f"  Run: ollama pull nomic-embed-text")
            return False

    except requests.exceptions.RequestException as e:
        print(f"  ✗ Ollama connection failed: {e}")
        print("  Make sure Ollama is running")
        return False


def check_data_folder():
    """Check if data folder exists and has PDFs"""
    print("\nChecking data folder...")
    data_path = Path("data")

    if not data_path.exists():
        print(f"  ✗ data/ folder not found")
        print(f"  Create it with: mkdir data")
        return False

    pdf_files = list(data_path.glob("*.pdf"))

    print(f"  ✓ data/ folder exists")
    print(f"  ✓ Found {len(pdf_files)} PDF file(s)")

    if len(pdf_files) == 0:
        print(f"  ⚠ No PDF files found - add some PDFs to data/ folder")
        return False

    return True


def main():
    print("=" * 60)
    print("RAG Tutorial - Setup Verification")
    print("=" * 60)

    checks = [
        ("Python packages", check_requirements),
        ("Qdrant database", check_qdrant),
        ("Ollama embedding service", check_ollama),
        ("Data folder", check_data_folder),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"Error checking {name}: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = all(result for _, result in results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("=" * 60)

    if all_passed:
        print("\n✓ All checks passed! You're ready to run:")
        print("  python populate_qdrant.py")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()