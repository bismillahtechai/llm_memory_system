"""
Vector database integration for the RAG component.

This module provides adapters for various vector databases (FAISS, Weaviate)
to store and retrieve embeddings for the RAG component.
"""

import os
import json
import numpy as np
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseVectorDBAdapter:
    """Base class for vector database adapters."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vector database adapter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def initialize(self):
        """Initialize the vector database."""
        raise NotImplementedError("Subclasses must implement initialize method")
    
    def add_embedding(self, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """
        Add an embedding to the vector database.
        
        Args:
            embedding: Embedding vector
            metadata: Metadata for the embedding
            
        Returns:
            Identifier for the stored embedding
        """
        raise NotImplementedError("Subclasses must implement add_embedding method")
    
    def add_embeddings_batch(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> List[str]:
        """
        Add a batch of embeddings to the vector database.
        
        Args:
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            
        Returns:
            List of identifiers for the stored embeddings
        """
        raise NotImplementedError("Subclasses must implement add_embeddings_batch method")
    
    def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            
        Returns:
            List of results with similarity scores and metadata
        """
        raise NotImplementedError("Subclasses must implement search_similar method")
    
    def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete an embedding from the vector database.
        
        Args:
            embedding_id: Identifier for the embedding
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement delete_embedding method")
    
    def close(self):
        """Close the vector database connection."""
        pass


class FAISSVectorDBAdapter(BaseVectorDBAdapter):
    """Vector database adapter using FAISS."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FAISS vector database adapter.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Import FAISS
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            logger.error("FAISS not installed. Please install it with 'pip install faiss-cpu' or 'pip install faiss-gpu'")
            raise
        
        self.index_path = config.get('index_path', 'faiss_index.bin')
        self.metadata_path = config.get('metadata_path', 'faiss_metadata.pkl')
        self.dimension = config.get('dimension', 384)  # Default for all-MiniLM-L6-v2
        self.index = None
        self.metadata = {}
        self.id_to_index = {}
    
    def initialize(self):
        """Initialize the FAISS index."""
        # Check if index exists
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            # Load existing index
            try:
                self.index = self.faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data['metadata']
                    self.id_to_index = data['id_to_index']
                
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self._create_new_index()
        else:
            # Create new index
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.index_path)), exist_ok=True)
        
        # Create a new index
        self.index = self.faiss.IndexFlatL2(self.dimension)
        self.metadata = {}
        self.id_to_index = {}
        
        logger.info(f"Created new FAISS index with dimension {self.dimension}")
    
    def _save_index(self):
        """Save the FAISS index and metadata."""
        try:
            # Save index
            self.faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'id_to_index': self.id_to_index
                }, f)
            
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def add_embedding(self, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """
        Add an embedding to the FAISS index.
        
        Args:
            embedding: Embedding vector
            metadata: Metadata for the embedding
            
        Returns:
            Identifier for the stored embedding
        """
        # Ensure index is initialized
        if self.index is None:
            self.initialize()
        
        # Generate ID
        embedding_id = str(uuid.uuid4())
        
        # Convert embedding to numpy array
        embedding_np = np.array([embedding], dtype=np.float32)
        
        # Add to index
        index = self.index.ntotal
        self.index.add(embedding_np)
        
        # Store metadata
        self.metadata[embedding_id] = metadata
        self.id_to_index[embedding_id] = index
        
        # Save index
        self._save_index()
        
        return embedding_id
    
    def add_embeddings_batch(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> List[str]:
        """
        Add a batch of embeddings to the FAISS index.
        
        Args:
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            
        Returns:
            List of identifiers for the stored embeddings
        """
        # Ensure index is initialized
        if self.index is None:
            self.initialize()
        
        # Generate IDs
        embedding_ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
        
        # Convert embeddings to numpy array
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Add to index
        start_index = self.index.ntotal
        self.index.add(embeddings_np)
        
        # Store metadata
        for i, (embedding_id, metadata) in enumerate(zip(embedding_ids, metadatas)):
            self.metadata[embedding_id] = metadata
            self.id_to_index[embedding_id] = start_index + i
        
        # Save index
        self._save_index()
        
        return embedding_ids
    
    def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings in the FAISS index.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            
        Returns:
            List of results with similarity scores and metadata
        """
        # Ensure index is initialized
        if self.index is None:
            self.initialize()
        
        # If index is empty, return empty results
        if self.index.ntotal == 0:
            return []
        
        # Convert query embedding to numpy array
        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        
        # Search index
        distances, indices = self.index.search(query_embedding_np, min(limit, self.index.ntotal))
        
        # Convert distances to similarities (FAISS uses L2 distance)
        # For L2 distance, smaller is better, so we use a simple conversion
        max_distance = np.max(distances[0]) if distances[0].size > 0 else 1.0
        similarities = 1.0 - distances[0] / max_distance
        
        # Get results
        results = []
        
        for i, (index, similarity) in enumerate(zip(indices[0], similarities)):
            # Find embedding ID for this index
            embedding_id = None
            for eid, idx in self.id_to_index.items():
                if idx == index:
                    embedding_id = eid
                    break
            
            if embedding_id is None or embedding_id not in self.metadata:
                continue
            
            # Get metadata
            metadata = self.metadata[embedding_id]
            
            results.append({
                'id': embedding_id,
                'similarity': float(similarity),
                'metadata': metadata
            })
        
        return results
    
    def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete an embedding from the FAISS index.
        
        Note: FAISS doesn't support direct deletion, so we rebuild the index without the deleted embedding.
        
        Args:
            embedding_id: Identifier for the embedding
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure index is initialized
        if self.index is None:
            self.initialize()
        
        # Check if embedding exists
        if embedding_id not in self.metadata:
            return False
        
        # Get all embeddings except the one to delete
        embeddings = []
        metadatas = []
        embedding_ids = []
        
        for eid, idx in self.id_to_index.items():
            if eid != embedding_id:
                # Get embedding vector
                embedding_np = np.array([self.index.reconstruct(idx)], dtype=np.float32)
                embeddings.append(embedding_np[0].tolist())
                metadatas.append(self.metadata[eid])
                embedding_ids.append(eid)
        
        # Create new index
        self.index = self.faiss.IndexFlatL2(self.dimension)
        self.metadata = {}
        self.id_to_index = {}
        
        # Add embeddings back
        if embeddings:
            embeddings_np = np.array(embeddings, dtype=np.float32)
            self.index.add(embeddings_np)
            
            for i, (eid, metadata) in enumerate(zip(embedding_ids, metadatas)):
                self.metadata[eid] = metadata
                self.id_to_index[eid] = i
        
        # Save index
        self._save_index()
        
        return True
    
    def close(self):
        """Save the index before closing."""
        if self.index is not None:
            self._save_index()


class WeaviateVectorDBAdapter(BaseVectorDBAdapter):
    """Vector database adapter using Weaviate."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Weaviate vector database adapter.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Import Weaviate
        try:
            import weaviate
            self.weaviate = weaviate
        except ImportError:
            logger.error("Weaviate not installed. Please install it with 'pip install weaviate-client'")
            raise
        
        self.url = config.get('url', 'http://localhost:8080')
        self.api_key = config.get('api_key', None)
        self.class_name = config.get('class_name', 'MemoryChunk')
        self.client = None
    
    def initialize(self):
        """Initialize the Weaviate client and schema."""
        # Create client
        auth_config = self.weaviate.auth.AuthApiKey(api_key=self.api_key) if self.api_key else None
        
        try:
            self.client = self.weaviate.Client(
                url=self.url,
                auth_client_secret=auth_config
            )
            
            logger.info(f"Connected to Weaviate at {self.url}")
            
            # Check if class exists
            if not self.client.schema.exists(self.class_name):
                # Create class
                class_obj = {
                    "class": self.class_name,
                    "description": "Memory chunks for LLM memory system",
                    "vectorizer": "none",  # We'll provide our own vectors
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "Content of the memory chunk"
                        },
                        {
                            "name": "conversationId",
                            "dataType": ["string"],
                            "description": "ID of the conversation"
                        },
                        {
                            "name": "conversationTitle",
                            "dataType": ["string"],
                            "description": "Title of the conversation"
                        },
                        {
                            "name": "chunkType",
                            "dataType": ["string"],
                            "description": "Type of chunk"
                        },
                        {
                            "name": "timestamp",
                            "dataType": ["date"],
                            "description": "Timestamp of the chunk"
                        },
                        {
                            "name": "metadata",
                            "dataType": ["text"],
                            "description": "JSON metadata for the chunk"
                        }
                    ]
                }
                
                self.client.schema.create_class(class_obj)
                logger.info(f"Created Weaviate class: {self.class_name}")
        except Exception as e:
            logger.error(f"Error initializing Weaviate: {e}")
            raise
    
    def add_embedding(self, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """
        Add an embedding to Weaviate.
        
        Args:
            embedding: Embedding vector
            metadata: Metadata for the embedding
            
        Returns:
            Identifier for the stored embedding
        """
        # Ensure client is initialized
        if self.client is None:
            self.initialize()
        
        # Extract data from metadata
        content = metadata.get('content', '')
        conversation_id = metadata.get('conversation_id', '')
        conversation_title = metadata.get('conversation_title', '')
        chunk_type = metadata.get('chunk_type', 'unknown')
        timestamp_str = metadata.get('timestamp', datetime.now().isoformat())
        
        # Convert timestamp to datetime if it's a string
        if isinstance(timestamp_str, str):
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except ValueError:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()
        
        # Serialize metadata to JSON
        metadata_json = json.dumps(metadata)
        
        # Generate UUID
        embedding_id = str(uuid.uuid4())
        
        try:
            # Add object to Weaviate
            self.client.data_object.create(
                class_name=self.class_name,
                data_object={
                    "content": content,
                    "conversationId": conversation_id,
                    "conversationTitle": conversation_title,
                    "chunkType": chunk_type,
                    "timestamp": timestamp.isoformat(),
                    "metadata": metadata_json
                },
                uuid=embedding_id,
                vector=embedding
            )
            
            return embedding_id
        except Exception as e:
            logger.error(f"Error adding embedding to Weaviate: {e}")
            raise
    
    def add_embeddings_batch(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> List[str]:
        """
        Add a batch of embeddings to Weaviate.
        
        Args:
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            
        Returns:
            List of identifiers for the stored embeddings
        """
        # Ensure client is initialized
        if self.client is None:
            self.initialize()
        
        embedding_ids = []
        
        # Add each embedding individually
        # Note: Weaviate has batch import, but it's more complex to set up
        for embedding, metadata in zip(embeddings, metadatas):
            try:
                embedding_id = self.add_embedding(embedding, metadata)
                embedding_ids.append(embedding_id)
            except Exception as e:
                logger.error(f"Error adding embedding to batch: {e}")
        
        return embedding_ids
    
    def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings in Weaviate.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            
        Returns:
            List of results with similarity scores and metadata
        """
        # Ensure client is initialized
        if self.client is None:
            self.initialize()
        
        try:
            # Search Weaviate
            result = (
                self.client.query
                .get(self.class_name, ["content", "conversationId", "conversationTitle", "chunkType", "timestamp", "metadata", "_additional {id certainty}"])
                .with_near_vector({"vector": query_embedding})
                .with_limit(limit)
                .do()
            )
            
            # Extract results
            results = []
            
            if "data" in result and "Get" in result["data"] and self.class_name in result["data"]["Get"]:
                for item in result["data"]["Get"][self.class_name]:
                    # Extract data
                    embedding_id = item["_additional"]["id"]
                    similarity = item["_additional"]["certainty"]
                    
                    # Parse metadata
                    try:
                        metadata_dict = json.loads(item["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        metadata_dict = {}
                    
                    # Add other fields to metadata
                    metadata_dict.update({
                        "content": item["content"],
                        "conversation_id": item["conversationId"],
                        "conversation_title": item["conversationTitle"],
                        "chunk_type": item["chunkType"],
                        "timestamp": item["timestamp"]
                    })
                    
                    results.append({
                        "id": embedding_id,
                        "similarity": similarity,
                        "metadata": metadata_dict
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error searching Weaviate: {e}")
            return []
    
    def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete an embedding from Weaviate.
        
        Args:
            embedding_id: Identifier for the embedding
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure client is initialized
        if self.client is None:
            self.initialize()
        
        try:
            # Delete object from Weaviate
            self.client.data_object.delete(
                class_name=self.class_name,
                uuid=embedding_id
            )
            
            return True
        except Exception as e:
            logger.error(f"Error deleting embedding from Weaviate: {e}")
            return False
    
    def close(self):
        """Close the Weaviate client."""
        self.client = None


class VectorDBManager:
    """Manager for vector database adapters."""
    
    def __init__(self, db_type: str = 'faiss', config: Dict[str, Any] = None):
        """
        Initialize the vector database manager.
        
        Args:
            db_type: Type of vector database ('faiss' or 'weaviate')
            config: Configuration dictionary
        """
        self.db_type = db_type
        self.config = config or {}
        
        # Initialize adapter
        if db_type == 'faiss':
            self.adapter = FAISSVectorDBAdapter(self.config)
        elif db_type == 'weaviate':
            self.adapter = WeaviateVectorDBAdapter(self.config)
        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")
    
    def initialize(self):
        """Initialize the vector database."""
        self.adapter.initialize()
    
    def add_embedding(self, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """
        Add an embedding to the vector database.
        
        Args:
            embedding: Embedding vector
            metadata: Metadata for the embedding
            
        Returns:
            Identifier for the stored embedding
        """
        return self.adapter.add_embedding(embedding, metadata)
    
    def add_embeddings_batch(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> List[str]:
        """
        Add a batch of embeddings to the vector database.
        
        Args:
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            
        Returns:
            List of identifiers for the stored embeddings
        """
        return self.adapter.add_embeddings_batch(embeddings, metadatas)
    
    def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            
        Returns:
            List of results with similarity scores and metadata
        """
        return self.adapter.search_similar(query_embedding, limit)
    
    def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete an embedding from the vector database.
        
        Args:
            embedding_id: Identifier for the embedding
            
        Returns:
            True if successful, False otherwise
        """
        return self.adapter.delete_embedding(embedding_id)
    
    def close(self):
        """Close the vector database connection."""
        self.adapter.close()


# Example usage
if __name__ == "__main__":
    import os
    import numpy as np
    
    # Example configuration
    faiss_config = {
        'index_path': '/home/ubuntu/llm_memory_system/rag_component/faiss_index.bin',
        'metadata_path': '/home/ubuntu/llm_memory_system/rag_component/faiss_metadata.pkl',
        'dimension': 384  # Default for all-MiniLM-L6-v2
    }
    
    weaviate_config = {
        'url': 'http://localhost:8080',
        'class_name': 'MemoryChunk'
    }
    
    # Create vector database manager
    vector_db = VectorDBManager(db_type='faiss', config=faiss_config)
    
    # Initialize
    vector_db.initialize()
    
    # Example embeddings
    embeddings = [
        np.random.rand(384).tolist(),
        np.random.rand(384).tolist(),
        np.random.rand(384).tolist()
    ]
    
    # Example metadata
    metadatas = [
        {
            'content': 'This is a test chunk 1',
            'conversation_id': 'conv1',
            'conversation_title': 'Test Conversation 1',
            'chunk_type': 'message',
            'timestamp': datetime.now().isoformat()
        },
        {
            'content': 'This is a test chunk 2',
            'conversation_id': 'conv1',
            'conversation_title': 'Test Conversation 1',
            'chunk_type': 'message',
            'timestamp': datetime.now().isoformat()
        },
        {
            'content': 'This is a test chunk 3',
            'conversation_id': 'conv2',
            'conversation_title': 'Test Conversation 2',
            'chunk_type': 'exchange',
            'timestamp': datetime.now().isoformat()
        }
    ]
    
    # Add embeddings
    embedding_ids = vector_db.add_embeddings_batch(embeddings, metadatas)
    print(f"Added {len(embedding_ids)} embeddings")
    
    # Search similar
    query_embedding = np.random.rand(384).tolist()
    results = vector_db.search_similar(query_embedding, limit=2)
    
    print(f"Found {len(results)} similar embeddings")
    for result in results:
        print(f"ID: {result['id']}")
        print(f"Similarity: {result['similarity']}")
        print(f"Content: {result['metadata']['content']}")
        print()
    
    # Close
    vector_db.close()
