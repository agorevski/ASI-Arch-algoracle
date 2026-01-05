import logging
import os
import pickle
from threading import RLock
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np


class FAISSManager:
    """FAISS vector index manager for efficient vector similarity search."""

    def __init__(self, dimension: int = 4096, index_file: str = "faiss_index.bin", metadata_file: str = "faiss_metadata.pkl", index_type: str = "IP"):  # IP for Inner Product (cosine similarity)
        """Initialize FAISS manager.

        Creates a new FAISS index manager with the specified configuration.
        Attempts to load existing index from disk if available.

        Args:
            dimension: Vector dimension. Defaults to 4096.
            index_file: Path to the FAISS index file. Defaults to "faiss_index.bin".
            metadata_file: Path to the metadata pickle file. Defaults to "faiss_metadata.pkl".
            index_type: Type of index to create. 'IP' for Inner Product (cosine similarity),
                'L2' for Euclidean distance. Defaults to 'IP'.

        Raises:
            ValueError: If an unsupported index_type is provided.
        """
        self.dimension = dimension
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.index_type = index_type
        self.lock = RLock()

        # Initialize logging
        self.logger = logging.getLogger(__name__)

        # Create index
        if index_type == "IP":
            # Inner product index, suitable for cosine similarity search (requires normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "L2":
            # L2 distance index
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Store metadata (vector ID to actual data mapping)
        self.id_to_index_map = {}  # MongoDB ObjectId -> FAISS index position
        self.index_to_id_map = {}  # FAISS index position -> MongoDB ObjectId
        self.next_index = 0

        # Load existing index and metadata
        self._load_index()

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity computation.

        Applies L2 normalization to vectors when using inner product index type.
        This is required for cosine similarity search.

        Args:
            vectors: A 2D numpy array of shape (n, dimension) containing vectors to normalize.

        Returns:
            The normalized vectors if using IP index type, otherwise returns
            the original vectors unchanged.
        """
        if self.index_type != "IP":
            return vectors

        # L2 normalization
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def _load_index(self):
        """Load existing index and metadata from disk.

        Attempts to load a previously saved FAISS index and associated metadata
        from the configured file paths. If files don't exist or loading fails,
        initializes empty index and metadata structures.

        Returns:
            None
        """
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                # Load index
                self.index = faiss.read_index(self.index_file)

                # Load metadata
                with open(self.metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                    self.id_to_index_map = metadata.get('id_to_index_map', {})
                    self.index_to_id_map = metadata.get('index_to_id_map', {})
                    self.next_index = metadata.get('next_index', 0)

                self.logger.info(f"Successfully loaded FAISS index with {self.index.ntotal} vectors")
            else:
                self.logger.info("Existing index file not found, creating new index")
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            # Recreate index
            if self.index_type == "IP":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)

            self.id_to_index_map = {}
            self.index_to_id_map = {}
            self.next_index = 0

    def _save_index(self):
        """Save index and metadata to file.

        Persists the FAISS index and metadata mappings to disk. The index
        is saved using FAISS native format, and metadata is pickled.

        Returns:
            None
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)

            # Save metadata
            metadata = {
                'id_to_index_map': self.id_to_index_map,
                'index_to_id_map': self.index_to_id_map,
                'next_index': self.next_index
            }

            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)

            self.logger.info("Index and metadata saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")

    def add_vector(self, vector: List[float], data_id: str) -> bool:
        """Add vector to index.

        Adds a new vector to the FAISS index with the associated data ID.
        The vector is normalized if using inner product index type.

        Args:
            vector: Vector data as a list of floats. Must match the configured dimension.
            data_id: Data ID (typically a MongoDB ObjectId string) to associate with the vector.

        Returns:
            True if the vector was added successfully, False otherwise.
        """
        with self.lock:
            try:
                if len(vector) != self.dimension:
                    self.logger.error(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}")
                    return False

                # Convert to numpy array
                vector_array = np.array([vector], dtype=np.float32)

                # Normalize vector (if using inner product index)
                vector_array = self._normalize_vectors(vector_array)

                # Add to index
                self.index.add(vector_array)

                # Update metadata mapping
                current_index = self.next_index
                self.id_to_index_map[data_id] = current_index
                self.index_to_id_map[current_index] = data_id
                self.next_index += 1

                # Save index periodically
                if self.next_index % 100 == 0:
                    self._save_index()

                self.logger.debug(f"Successfully added vector, ID: {data_id}, index position: {current_index}")
                return True

            except Exception as e:
                self.logger.error(f"Failed to add vector: {e}")
                return False

    def search_similar(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for most similar vectors.

        Performs a similarity search in the FAISS index to find the k most
        similar vectors to the query vector. Filters out orphan vectors
        (vectors that have been logically deleted).

        Args:
            query_vector: Query vector as a list of floats. Must match the configured dimension.
            k: Number of results to return. Defaults to 5.

        Returns:
            A list of tuples containing (data_id, similarity_score) pairs,
            sorted by similarity score in descending order. Returns an empty
            list if the index is empty or an error occurs.
        """
        with self.lock:
            try:
                if len(query_vector) != self.dimension:
                    self.logger.error(f"Query vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}")
                    return []

                if self.index.ntotal == 0:
                    self.logger.warning("Index is empty, cannot perform search")
                    return []

                # Convert to numpy array
                query_array = np.array([query_vector], dtype=np.float32)

                # Normalize query vector
                query_array = self._normalize_vectors(query_array)

                # Since there might be orphan vectors, need to search more results to ensure k valid results
                search_k = min(k * 3, self.index.ntotal)  # Search 3x quantity to handle orphan vectors

                # Perform search
                if self.index_type == "IP":
                    # Inner product search, higher score is more similar
                    scores, indices = self.index.search(query_array, search_k)
                else:
                    # L2 distance search, smaller distance is more similar, need to convert to similarity
                    distances, indices = self.index.search(query_array, search_k)
                    # Convert distance to similarity (simple inverse relationship)
                    scores = 1.0 / (1.0 + distances)

                # Build results, filter out orphan vectors
                results = []
                for i in range(len(indices[0])):
                    faiss_index = indices[0][i]
                    score = float(scores[0][i])

                    # Get corresponding data ID, skip orphan vectors
                    data_id = self.index_to_id_map.get(faiss_index)
                    if data_id:
                        results.append((data_id, score))
                        # Stop when reaching required quantity
                        if len(results) >= k:
                            break

                self.logger.info(f"Search completed, found {len(results)} valid results among {self.index.ntotal} vectors")
                if len(results) < k and len(results) < len(indices[0]):
                    orphan_count = len([i for i in indices[0] if self.index_to_id_map.get(i) is None])
                    self.logger.warning(f"Found {orphan_count} orphan vectors (logically deleted but still in index)")

                return results

            except Exception as e:
                self.logger.error(f"Vector search failed: {e}")
                return []

    def remove_vector(self, data_id: str) -> bool:
        """Remove vector from index.

        Marks a vector for removal by deleting its ID mapping. Note that FAISS
        doesn't support direct vector deletion, so the vector remains in the
        index but becomes an orphan. Use clean_orphan_vectors() to physically
        remove orphaned vectors by rebuilding the index.

        Args:
            data_id: The data ID of the vector to remove.

        Returns:
            True if the vector was found and marked for removal, False otherwise.
        """
        with self.lock:
            try:
                if data_id in self.id_to_index_map:
                    faiss_index = self.id_to_index_map[data_id]

                    # Remove from mapping
                    del self.id_to_index_map[data_id]
                    del self.index_to_id_map[faiss_index]

                    self.logger.info(f"Marked vector for deletion: {data_id}")
                    return True
                else:
                    self.logger.warning(f"Vector to delete not found: {data_id}")
                    return False

            except Exception as e:
                self.logger.error(f"Failed to delete vector: {e}")
                return False

    def rebuild_index(self, vectors_data: List[Tuple[List[float], str]]):
        """Rebuild index from scratch.

        Clears all existing data and rebuilds the index with the provided vectors.
        This is useful for cleaning up orphan vectors or re-indexing data.

        Args:
            vectors_data: A list of tuples, where each tuple contains
                (vector, data_id). Vectors that don't match the configured
                dimension are skipped.

        Returns:
            None
        """
        with self.lock:
            try:
                # Recreate index
                if self.index_type == "IP":
                    self.index = faiss.IndexFlatIP(self.dimension)
                else:
                    self.index = faiss.IndexFlatL2(self.dimension)

                # Reset metadata
                self.id_to_index_map = {}
                self.index_to_id_map = {}
                self.next_index = 0

                # Batch add vectors
                if vectors_data:
                    vectors = []
                    data_ids = []

                    for vector, data_id in vectors_data:
                        if len(vector) == self.dimension:
                            vectors.append(vector)
                            data_ids.append(data_id)

                    if vectors:
                        # Convert to numpy array
                        vectors_array = np.array(vectors, dtype=np.float32)

                        # Normalize vectors
                        vectors_array = self._normalize_vectors(vectors_array)

                        # Batch add to index
                        self.index.add(vectors_array)

                        # Update metadata
                        for i, data_id in enumerate(data_ids):
                            self.id_to_index_map[data_id] = i
                            self.index_to_id_map[i] = data_id

                        self.next_index = len(data_ids)

                # Save rebuilt index
                self._save_index()

                self.logger.info(f"Index rebuild completed with {self.index.ntotal} vectors")

            except Exception as e:
                self.logger.error(f"Failed to rebuild index: {e}")

    def clean_orphan_vectors(self) -> Dict[str, int]:
        """Clean orphan vectors from the index.

        Orphan vectors are vectors that have been logically deleted (removed from
        ID mappings) but still exist in the FAISS index. This method detects such
        vectors and rebuilds the index to physically remove them.

        Returns:
            A dictionary containing cleanup statistics with keys:
                - total_vectors: Number of vectors after cleanup.
                - orphan_vectors: Number of orphan vectors found.
                - cleaned: Number of orphan vectors cleaned.
                - error: Error message if cleanup failed (only present on error).
        """
        with self.lock:
            try:
                if self.index.ntotal == 0:
                    return {"total_vectors": 0, "orphan_vectors": 0, "cleaned": 0}

                # Detect orphan vectors
                orphan_indices = []
                active_indices = set(self.index_to_id_map.keys())

                for i in range(self.next_index):
                    if i not in active_indices:
                        orphan_indices.append(i)

                orphan_count = len(orphan_indices)

                if orphan_count == 0:
                    self.logger.info("No orphan vectors found")
                    return {
                        "total_vectors": self.index.ntotal,
                        "orphan_vectors": 0,
                        "cleaned": 0
                    }

                self.logger.warning(f"Found {orphan_count} orphan vectors, will rebuild index to clean")

                # Rebuild index to clean orphan vectors
                vectors_data = []

                # Extract all valid vector data
                for faiss_index, data_id in self.index_to_id_map.items():
                    # Get vector from FAISS index
                    vector = self.index.reconstruct(faiss_index)
                    vectors_data.append((vector.tolist(), data_id))

                # Rebuild index
                self.rebuild_index(vectors_data)

                self.logger.info(f"Successfully cleaned {orphan_count} orphan vectors")

                return {
                    "total_vectors": self.index.ntotal,
                    "orphan_vectors": orphan_count,
                    "cleaned": orphan_count
                }

            except Exception as e:
                self.logger.error(f"Failed to clean orphan vectors: {e}")
                return {"error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        Calculates and returns various statistics about the FAISS index,
        including total vectors, dimension, index type, and health status.

        Returns:
            A dictionary containing index statistics with keys:
                - total_vectors: Total number of vectors in the FAISS index.
                - dimension: Vector dimension.
                - index_type: Type of index ('IP' or 'L2').
                - active_mappings: Number of active ID mappings.
                - orphan_vectors: Number of orphaned vectors.
                - next_index: Next available index position.
                - health_status: Health status string.
        """
        # Calculate orphan vector count
        active_indices = set(self.index_to_id_map.keys())
        orphan_count = 0
        for i in range(self.next_index):
            if i not in active_indices:
                orphan_count += 1

        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "active_mappings": len(self.id_to_index_map),
            "orphan_vectors": orphan_count,
            "next_index": self.next_index,
            "health_status": "healthy" if orphan_count == 0 else f"warning: {orphan_count} orphan vectors"
        }

    def save(self):
        """Manually save index.

        Triggers an immediate save of the index and metadata to disk.
        Use this to ensure data persistence before shutdown.

        Returns:
            None
        """
        self._save_index()


# Global FAISS manager instance
_faiss_manager = None


def get_faiss_manager() -> FAISSManager:
    """Get global FAISS manager instance.

    Returns a singleton instance of the FAISSManager. Creates a new
    instance with default parameters if one doesn't exist.

    Returns:
        The global FAISSManager instance.
    """
    global _faiss_manager
    if _faiss_manager is None:
        _faiss_manager = FAISSManager()
    return _faiss_manager
