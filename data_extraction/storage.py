"""
Storage adapters for conversation data and embeddings.

This module provides storage adapters for saving and retrieving conversation data,
chunks, and embeddings using various storage backends.
"""

import os
import json
import shutil
import sqlite3
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Iterator
import logging
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseStorageAdapter:
    """Base class for storage adapters."""
    
    def __init__(self, storage_path: str):
        """
        Initialize the storage adapter.
        
        Args:
            storage_path: Path to the storage location
        """
        self.storage_path = storage_path
    
    def initialize(self):
        """Initialize the storage backend."""
        raise NotImplementedError("Subclasses must implement initialize method")
    
    def store_conversation(self, conversation: Dict[str, Any]) -> str:
        """
        Store a conversation.
        
        Args:
            conversation: Conversation data
            
        Returns:
            Identifier for the stored conversation
        """
        raise NotImplementedError("Subclasses must implement store_conversation method")
    
    def store_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Store a conversation chunk.
        
        Args:
            chunk: Chunk data
            
        Returns:
            Identifier for the stored chunk
        """
        raise NotImplementedError("Subclasses must implement store_chunk method")
    
    def store_chunks_batch(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Store a batch of conversation chunks.
        
        Args:
            chunks: List of chunk data
            
        Returns:
            List of identifiers for the stored chunks
        """
        raise NotImplementedError("Subclasses must implement store_chunks_batch method")
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a conversation by ID.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Conversation data or None if not found
        """
        raise NotImplementedError("Subclasses must implement get_conversation method")
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk data or None if not found
        """
        raise NotImplementedError("Subclasses must implement get_chunk method")
    
    def get_chunks_by_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a conversation.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            List of chunk data
        """
        raise NotImplementedError("Subclasses must implement get_chunks_by_conversation method")
    
    def search_similar_chunks(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks with similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            
        Returns:
            List of chunk data with similarity scores
        """
        raise NotImplementedError("Subclasses must implement search_similar_chunks method")
    
    def close(self):
        """Close the storage backend."""
        pass


class FileSystemStorageAdapter(BaseStorageAdapter):
    """Storage adapter using the file system."""
    
    def __init__(self, storage_path: str):
        """
        Initialize the file system storage adapter.
        
        Args:
            storage_path: Path to the storage directory
        """
        super().__init__(storage_path)
        self.conversations_dir = os.path.join(storage_path, 'conversations')
        self.chunks_dir = os.path.join(storage_path, 'chunks')
        self.index_file = os.path.join(storage_path, 'index.json')
        self.index = {
            'conversations': {},
            'chunks': {},
            'conversation_chunks': {}
        }
    
    def initialize(self):
        """Initialize the storage directories and index."""
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(self.conversations_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)
        
        # Load index if it exists
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
            except Exception as e:
                logger.error(f"Error loading index file: {e}")
                # Create a new index
                self.index = {
                    'conversations': {},
                    'chunks': {},
                    'conversation_chunks': {}
                }
    
    def _save_index(self):
        """Save the index to file."""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving index file: {e}")
    
    def store_conversation(self, conversation: Dict[str, Any]) -> str:
        """
        Store a conversation.
        
        Args:
            conversation: Conversation data
            
        Returns:
            Identifier for the stored conversation
        """
        # Ensure storage is initialized
        self.initialize()
        
        # Get or generate conversation ID
        conversation_id = conversation.get('metadata', {}).get('conversation_id', '')
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            if 'metadata' not in conversation:
                conversation['metadata'] = {}
            conversation['metadata']['conversation_id'] = conversation_id
        
        # Generate filename
        filename = f"{conversation_id}.json"
        file_path = os.path.join(self.conversations_dir, filename)
        
        # Save conversation to file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, indent=2, ensure_ascii=False)
            
            # Update index
            self.index['conversations'][conversation_id] = {
                'file_path': file_path,
                'title': conversation.get('metadata', {}).get('title', ''),
                'source': conversation.get('metadata', {}).get('source', ''),
                'create_time': conversation.get('metadata', {}).get('create_time', ''),
                'stored_time': datetime.now().isoformat()
            }
            
            # Initialize conversation chunks entry if not exists
            if conversation_id not in self.index['conversation_chunks']:
                self.index['conversation_chunks'][conversation_id] = []
            
            # Save index
            self._save_index()
            
            return conversation_id
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
            raise
    
    def store_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Store a conversation chunk.
        
        Args:
            chunk: Chunk data
            
        Returns:
            Identifier for the stored chunk
        """
        # Ensure storage is initialized
        self.initialize()
        
        # Get or generate chunk ID
        chunk_id = chunk.get('id', '')
        if not chunk_id:
            chunk_id = str(uuid.uuid4())
            chunk['id'] = chunk_id
        
        # Get conversation ID
        conversation_id = chunk.get('conversation_id', '')
        
        # Generate filename
        chunk_type = chunk.get('chunk_type', 'unknown')
        filename = f"{conversation_id}_{chunk_type}_{chunk_id}.json"
        file_path = os.path.join(self.chunks_dir, filename)
        
        # Save chunk to file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, indent=2, ensure_ascii=False)
            
            # Update index
            self.index['chunks'][chunk_id] = {
                'file_path': file_path,
                'conversation_id': conversation_id,
                'chunk_type': chunk_type,
                'stored_time': datetime.now().isoformat()
            }
            
            # Update conversation chunks index
            if conversation_id:
                if conversation_id not in self.index['conversation_chunks']:
                    self.index['conversation_chunks'][conversation_id] = []
                
                if chunk_id not in self.index['conversation_chunks'][conversation_id]:
                    self.index['conversation_chunks'][conversation_id].append(chunk_id)
            
            # Save index
            self._save_index()
            
            return chunk_id
        except Exception as e:
            logger.error(f"Error storing chunk: {e}")
            raise
    
    def store_chunks_batch(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Store a batch of conversation chunks.
        
        Args:
            chunks: List of chunk data
            
        Returns:
            List of identifiers for the stored chunks
        """
        chunk_ids = []
        
        for chunk in chunks:
            try:
                chunk_id = self.store_chunk(chunk)
                chunk_ids.append(chunk_id)
            except Exception as e:
                logger.error(f"Error storing chunk in batch: {e}")
        
        return chunk_ids
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a conversation by ID.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Conversation data or None if not found
        """
        # Ensure storage is initialized
        self.initialize()
        
        # Check if conversation exists in index
        if conversation_id not in self.index['conversations']:
            return None
        
        # Get file path from index
        file_path = self.index['conversations'][conversation_id]['file_path']
        
        # Load conversation from file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                conversation = json.load(f)
            
            return conversation
        except Exception as e:
            logger.error(f"Error retrieving conversation {conversation_id}: {e}")
            return None
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk data or None if not found
        """
        # Ensure storage is initialized
        self.initialize()
        
        # Check if chunk exists in index
        if chunk_id not in self.index['chunks']:
            return None
        
        # Get file path from index
        file_path = self.index['chunks'][chunk_id]['file_path']
        
        # Load chunk from file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunk = json.load(f)
            
            return chunk
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    def get_chunks_by_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a conversation.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            List of chunk data
        """
        # Ensure storage is initialized
        self.initialize()
        
        # Check if conversation exists in index
        if conversation_id not in self.index['conversation_chunks']:
            return []
        
        # Get chunk IDs from index
        chunk_ids = self.index['conversation_chunks'][conversation_id]
        
        # Load chunks
        chunks = []
        for chunk_id in chunk_ids:
            chunk = self.get_chunk(chunk_id)
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def search_similar_chunks(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks with similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            
        Returns:
            List of chunk data with similarity scores
        """
        # Ensure storage is initialized
        self.initialize()
        
        # Convert query embedding to numpy array
        query_embedding_np = np.array(query_embedding)
        
        # Calculate similarity for all chunks
        results = []
        
        for chunk_id, chunk_info in self.index['chunks'].items():
            chunk = self.get_chunk(chunk_id)
            
            if not chunk or 'embedding' not in chunk:
                continue
            
            # Calculate cosine similarity
            chunk_embedding = np.array(chunk['embedding'])
            similarity = np.dot(query_embedding_np, chunk_embedding) / (
                np.linalg.norm(query_embedding_np) * np.linalg.norm(chunk_embedding))
            
            results.append({
                'chunk': chunk,
                'similarity': float(similarity)
            })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top results
        return results[:limit]
    
    def close(self):
        """Save the index before closing."""
        self._save_index()


class SQLiteStorageAdapter(BaseStorageAdapter):
    """Storage adapter using SQLite database."""
    
    def __init__(self, storage_path: str):
        """
        Initialize the SQLite storage adapter.
        
        Args:
            storage_path: Path to the SQLite database file
        """
        super().__init__(storage_path)
        self.conn = None
    
    def initialize(self):
        """Initialize the SQLite database."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Connect to database
            self.conn = sqlite3.connect(self.storage_path)
            
            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")
            
            # Create tables if they don't exist
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    source TEXT,
                    create_time TEXT,
                    stored_time TEXT,
                    data TEXT
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    chunk_type TEXT,
                    content TEXT,
                    stored_time TEXT,
                    data TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    embedding BLOB,
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
                )
            """)
            
            # Create indices
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_conversation_id
                ON chunks (conversation_id)
            """)
            
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type
                ON chunks (chunk_type)
            """)
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")
            raise
    
    def store_conversation(self, conversation: Dict[str, Any]) -> str:
        """
        Store a conversation.
        
        Args:
            conversation: Conversation data
            
        Returns:
            Identifier for the stored conversation
        """
        # Ensure database is initialized
        if self.conn is None:
            self.initialize()
        
        # Get or generate conversation ID
        metadata = conversation.get('metadata', {})
        conversation_id = metadata.get('conversation_id', '')
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            if 'metadata' not in conversation:
                conversation['metadata'] = {}
            conversation['metadata']['conversation_id'] = conversation_id
        
        # Extract metadata
        title = metadata.get('title', '')
        source = metadata.get('source', '')
        create_time = metadata.get('create_time', '')
        stored_time = datetime.now().isoformat()
        
        # Serialize conversation data
        data_json = json.dumps(conversation, ensure_ascii=False)
        
        try:
            # Insert or replace conversation
            self.conn.execute("""
                INSERT OR REPLACE INTO conversations
                (id, title, source, create_time, stored_time, data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (conversation_id, title, source, create_time, stored_time, data_json))
            
            self.conn.commit()
            
            return conversation_id
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
            self.conn.rollback()
            raise
    
    def store_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Store a conversation chunk.
        
        Args:
            chunk: Chunk data
            
        Returns:
            Identifier for the stored chunk
        """
        # Ensure database is initialized
        if self.conn is None:
            self.initialize()
        
        # Get or generate chunk ID
        chunk_id = chunk.get('id', '')
        if not chunk_id:
            chunk_id = str(uuid.uuid4())
            chunk['id'] = chunk_id
        
        # Extract data
        conversation_id = chunk.get('conversation_id', '')
        chunk_type = chunk.get('chunk_type', 'unknown')
        content = chunk.get('content', '')
        stored_time = datetime.now().isoformat()
        
        # Check if embedding exists
        has_embedding = 'embedding' in chunk
        
        # Make a copy of chunk without embedding for storage
        chunk_without_embedding = {k: v for k, v in chunk.items() if k != 'embedding'}
        
        # Serialize chunk data
        data_json = json.dumps(chunk_without_embedding, ensure_ascii=False)
        
        try:
            # Insert or replace chunk
            self.conn.execute("""
                INSERT OR REPLACE INTO chunks
                (id, conversation_id, chunk_type, content, stored_time, data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chunk_id, conversation_id, chunk_type, content, stored_time, data_json))
            
            # Store embedding if exists
            if has_embedding:
                embedding = pickle.dumps(np.array(chunk['embedding']))
                
                self.conn.execute("""
                    INSERT OR REPLACE INTO embeddings
                    (chunk_id, embedding)
                    VALUES (?, ?)
                """, (chunk_id, embedding))
            
            self.conn.commit()
            
            return chunk_id
        except Exception as e:
            logger.error(f"Error storing chunk: {e}")
            self.conn.rollback()
            raise
    
    def store_chunks_batch(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Store a batch of conversation chunks.
        
        Args:
            chunks: List of chunk data
            
        Returns:
            List of identifiers for the stored chunks
        """
        # Ensure database is initialized
        if self.conn is None:
            self.initialize()
        
        chunk_ids = []
        
        try:
            for chunk in chunks:
                # Get or generate chunk ID
                chunk_id = chunk.get('id', '')
                if not chunk_id:
                    chunk_id = str(uuid.uuid4())
                    chunk['id'] = chunk_id
                
                # Extract data
                conversation_id = chunk.get('conversation_id', '')
                chunk_type = chunk.get('chunk_type', 'unknown')
                content = chunk.get('content', '')
                stored_time = datetime.now().isoformat()
                
                # Check if embedding exists
                has_embedding = 'embedding' in chunk
                
                # Make a copy of chunk without embedding for storage
                chunk_without_embedding = {k: v for k, v in chunk.items() if k != 'embedding'}
                
                # Serialize chunk data
                data_json = json.dumps(chunk_without_embedding, ensure_ascii=False)
                
                # Insert or replace chunk
                self.conn.execute("""
                    INSERT OR REPLACE INTO chunks
                    (id, conversation_id, chunk_type, content, stored_time, data)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (chunk_id, conversation_id, chunk_type, content, stored_time, data_json))
                
                # Store embedding if exists
                if has_embedding:
                    embedding = pickle.dumps(np.array(chunk['embedding']))
                    
                    self.conn.execute("""
                        INSERT OR REPLACE INTO embeddings
                        (chunk_id, embedding)
                        VALUES (?, ?)
                    """, (chunk_id, embedding))
                
                chunk_ids.append(chunk_id)
            
            self.conn.commit()
            
            return chunk_ids
        except Exception as e:
            logger.error(f"Error storing chunks batch: {e}")
            self.conn.rollback()
            raise
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a conversation by ID.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Conversation data or None if not found
        """
        # Ensure database is initialized
        if self.conn is None:
            self.initialize()
        
        try:
            # Query conversation
            cursor = self.conn.execute("""
                SELECT data FROM conversations
                WHERE id = ?
            """, (conversation_id,))
            
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            # Deserialize conversation data
            conversation = json.loads(row[0])
            
            return conversation
        except Exception as e:
            logger.error(f"Error retrieving conversation {conversation_id}: {e}")
            return None
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk data or None if not found
        """
        # Ensure database is initialized
        if self.conn is None:
            self.initialize()
        
        try:
            # Query chunk
            cursor = self.conn.execute("""
                SELECT c.data, e.embedding
                FROM chunks c
                LEFT JOIN embeddings e ON c.id = e.chunk_id
                WHERE c.id = ?
            """, (chunk_id,))
            
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            # Deserialize chunk data
            chunk = json.loads(row[0])
            
            # Add embedding if exists
            if row[1] is not None:
                embedding = pickle.loads(row[1])
                chunk['embedding'] = embedding.tolist()
            
            return chunk
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    def get_chunks_by_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a conversation.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            List of chunk data
        """
        # Ensure database is initialized
        if self.conn is None:
            self.initialize()
        
        try:
            # Query chunks
            cursor = self.conn.execute("""
                SELECT c.id
                FROM chunks c
                WHERE c.conversation_id = ?
            """, (conversation_id,))
            
            rows = cursor.fetchall()
            
            # Load chunks
            chunks = []
            for row in rows:
                chunk_id = row[0]
                chunk = self.get_chunk(chunk_id)
                if chunk:
                    chunks.append(chunk)
            
            return chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks for conversation {conversation_id}: {e}")
            return []
    
    def search_similar_chunks(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks with similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            
        Returns:
            List of chunk data with similarity scores
        """
        # Ensure database is initialized
        if self.conn is None:
            self.initialize()
        
        # Convert query embedding to numpy array
        query_embedding_np = np.array(query_embedding)
        
        try:
            # Query all embeddings
            cursor = self.conn.execute("""
                SELECT c.id, e.embedding
                FROM chunks c
                JOIN embeddings e ON c.id = e.chunk_id
            """)
            
            rows = cursor.fetchall()
            
            # Calculate similarity for all chunks
            results = []
            
            for row in rows:
                chunk_id = row[0]
                embedding = pickle.loads(row[1])
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding_np, embedding) / (
                    np.linalg.norm(query_embedding_np) * np.linalg.norm(embedding))
                
                # Get chunk data
                chunk = self.get_chunk(chunk_id)
                
                results.append({
                    'chunk': chunk,
                    'similarity': float(similarity)
                })
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Return top results
            return results[:limit]
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    def close(self):
        """Close the database connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None


class StorageManager:
    """Manager for storage adapters."""
    
    def __init__(self, storage_type: str = 'filesystem', storage_path: str = None):
        """
        Initialize the storage manager.
        
        Args:
            storage_type: Type of storage adapter ('filesystem' or 'sqlite')
            storage_path: Path to the storage location
        """
        self.storage_type = storage_type
        
        # Set default storage path if not provided
        if storage_path is None:
            if storage_type == 'filesystem':
                storage_path = os.path.join(os.getcwd(), 'data', 'storage')
            else:
                storage_path = os.path.join(os.getcwd(), 'data', 'storage.db')
        
        self.storage_path = storage_path
        
        # Initialize storage adapter
        if storage_type == 'filesystem':
            self.adapter = FileSystemStorageAdapter(storage_path)
        elif storage_type == 'sqlite':
            self.adapter = SQLiteStorageAdapter(storage_path)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
    
    def initialize(self):
        """Initialize the storage adapter."""
        self.adapter.initialize()
    
    def store_conversation(self, conversation: Dict[str, Any]) -> str:
        """
        Store a conversation.
        
        Args:
            conversation: Conversation data
            
        Returns:
            Identifier for the stored conversation
        """
        return self.adapter.store_conversation(conversation)
    
    def store_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Store a conversation chunk.
        
        Args:
            chunk: Chunk data
            
        Returns:
            Identifier for the stored chunk
        """
        return self.adapter.store_chunk(chunk)
    
    def store_chunks_batch(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Store a batch of conversation chunks.
        
        Args:
            chunks: List of chunk data
            
        Returns:
            List of identifiers for the stored chunks
        """
        return self.adapter.store_chunks_batch(chunks)
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a conversation by ID.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Conversation data or None if not found
        """
        return self.adapter.get_conversation(conversation_id)
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk data or None if not found
        """
        return self.adapter.get_chunk(chunk_id)
    
    def get_chunks_by_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a conversation.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            List of chunk data
        """
        return self.adapter.get_chunks_by_conversation(conversation_id)
    
    def search_similar_chunks(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks with similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            
        Returns:
            List of chunk data with similarity scores
        """
        return self.adapter.search_similar_chunks(query_embedding, limit)
    
    def close(self):
        """Close the storage adapter."""
        self.adapter.close()


# Example usage
if __name__ == "__main__":
    import os
    from chunker import ChunkingPipeline
    from parsers import ConversationProcessor
    from embeddings import EmbeddingGenerator
    
    # Example paths
    sample_dir = "/home/ubuntu/llm_memory_system/data_extraction/sample_data"
    storage_dir = "/home/ubuntu/llm_memory_system/data_extraction/storage"
    
    # Create storage manager
    storage_manager = StorageManager(
        storage_type='sqlite',
        storage_path=os.path.join(storage_dir, 'memory.db')
    )
    
    # Initialize storage
    storage_manager.initialize()
    
    # Process a sample file
    processor = ConversationProcessor()
    chunking_pipeline = ChunkingPipeline(
        strategies=['message', 'exchange'],
        chunk_size=1000,
        chunk_overlap=200
    )
    embedding_generator = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
    
    # Process the first file in the sample directory
    for filename in os.listdir(sample_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(sample_dir, filename)
            
            # Process the file
            conversation = processor.process_file(file_path)
            
            # Store conversation
            conversation_id = storage_manager.store_conversation(conversation)
            print(f"Stored conversation: {conversation_id}")
            
            # Apply chunking strategies
            chunks_by_strategy = chunking_pipeline.process_conversation(conversation)
            
            for strategy, chunks in chunks_by_strategy.items():
                # Generate embeddings
                embedded_chunks = embedding_generator.process_chunks_batch(chunks)
                
                # Store chunks
                chunk_ids = storage_manager.store_chunks_batch(embedded_chunks)
                print(f"Stored {len(chunk_ids)} chunks using {strategy} strategy")
            
            # Only process one file for this example
            break
    
    # Close storage
    storage_manager.close()
