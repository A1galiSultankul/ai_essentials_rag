import requests
from typing import List


class Embedder:
    """
    Embedder class to generate text embeddings using Ollama's nomic-embed-text model.
    """

    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        """
        Initialize the Embedder with the specified model.

        Args:
            model_name: Name of the embedding model (default: nomic-embed-text)
            base_url: Base URL for Ollama API (default: http://localhost:11434)
        """
        self.model_name = model_name
        self.base_url = base_url
        self.embed_url = f"{base_url}/api/embeddings"

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for the given text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": text
            }

            response = requests.post(self.embed_url, json=payload)
            response.raise_for_status()

            result = response.json()
            embedding = result.get("embedding", [])

            if not embedding:
                raise ValueError(f"No embedding returned for text: {text[:50]}...")

            return embedding

        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            raise
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings