from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Any
import uuid


class QdrantManager:
    """
    Manager class for interacting with Qdrant vector database.
    """

    def __init__(self, host: str = "localhost", port: int = 6333):
        """
        Initialize the Qdrant client.

        Args:
            host: Qdrant host address (default: localhost)
            port: Qdrant port (default: 6333)
        """
        self.client = QdrantClient(host=host, port=port)

    def create_collection(self, name: str, vector_size: int = 768):
        """
        Create a new collection in Qdrant.

        Args:
            name: Name of the collection
            vector_size: Dimension of the embedding vectors (default: 768 for nomic-embed-text)
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if name in collection_names:
                print(f"Collection '{name}' already exists.")
                return

            # Create new collection
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Collection '{name}' created successfully.")

        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def insert_point(self, embedding: List[float], collection_name: str, chunk_text: str,
                     metadata: Dict[str, Any] = None):
        """
        Insert a single point (embedding) into the collection.

        Args:
            embedding: Vector embedding
            collection_name: Name of the collection to insert into
            chunk_text: The original text chunk
            metadata: Additional metadata to store with the point
        """
        try:
            # Generate unique ID
            point_id = str(uuid.uuid4())

            # Prepare payload
            payload = {"text": chunk_text}
            if metadata:
                payload.update(metadata)

            # Create and insert point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )

            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )

        except Exception as e:
            print(f"Error inserting point: {e}")
            raise

    def insert_points_batch(self, embeddings: List[List[float]], collection_name: str,
                            chunk_texts: List[str], metadata_list: List[Dict[str, Any]] = None):
        """
        Insert multiple points in batch for better performance.

        Args:
            embeddings: List of vector embeddings
            collection_name: Name of the collection to insert into
            chunk_texts: List of original text chunks
            metadata_list: List of metadata dictionaries for each point
        """
        try:
            points = []
            for i, (embedding, text) in enumerate(zip(embeddings, chunk_texts)):
                point_id = str(uuid.uuid4())
                payload = {"text": text}

                if metadata_list and i < len(metadata_list):
                    payload.update(metadata_list[i])

                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)

            # Batch insert
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"Inserted {len(points)} points into collection '{collection_name}'")

        except Exception as e:
            print(f"Error inserting points batch: {e}")
            raise

    def search_points(self, embedding: List[float], collection_name: str,
                      top_k: int = 5, score_threshold: float = 0.75) -> List[str]:
        """
        Search for similar points in the collection.

        Args:
            embedding: Query embedding vector
            collection_name: Name of the collection to search in
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of text chunks that match the query
        """
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=embedding,
                limit=top_k,
                score_threshold=score_threshold
            )

            # Extract text chunks from results
            matched_texts = []
            for result in results:
                matched_texts.append(result.payload.get("text", ""))

            return matched_texts

        except Exception as e:
            print(f"Error searching points: {e}")
            raise

    def search_points_with_scores(self, embedding: List[float], collection_name: str,
                                  top_k: int = 5, score_threshold: float = 0.75) -> List[Dict[str, Any]]:
        """
        Search for similar points and return results with scores and metadata.

        Args:
            embedding: Query embedding vector
            collection_name: Name of the collection to search in
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of dictionaries containing text, score, and metadata
        """
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=embedding,
                limit=top_k,
                score_threshold=score_threshold
            )

            # Extract full results
            search_results = []
            for result in results:
                search_results.append({
                    "text": result.payload.get("text", ""),
                    "score": result.score,
                    "metadata": {k: v for k, v in result.payload.items() if k != "text"}
                })

            return search_results

        except Exception as e:
            print(f"Error searching points: {e}")
            raise

    def get_collection_info(self, collection_name: str):
        """
        Get information about a collection.

        Args:
            collection_name: Name of the collection
        """
        try:
            info = self.client.get_collection(collection_name=collection_name)
            print(f"Collection: {collection_name}")
            print(f"Points count: {info.points_count}")
            print(f"Vectors config: {info.config.params.vectors}")
            return info
        except Exception as e:
            print(f"Error getting collection info: {e}")
            raise

    def delete_collection(self, collection_name: str):
        """
        Delete a collection.

        Args:
            collection_name: Name of the collection to delete
        """
        try:
            self.client.delete_collection(collection_name=collection_name)
            print(f"Collection '{collection_name}' deleted successfully.")
        except Exception as e:
            print(f"Error deleting collection: {e}")
            raise