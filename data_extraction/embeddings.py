"""
Embedding generator for conversation chunks.

This module provides functionality to generate embeddings for conversation chunks
using various embedding models.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from sentence_transformers import SentenceTransformer
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for conversation chunks."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
    
    def load_model(self):
        """Load the embedding model."""
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        if self.model is None:
            self.load_model()
        
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if self.model is None:
            self.load_model()
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def process_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a chunk by adding embedding.
        
        Args:
            chunk: Conversation chunk
            
        Returns:
            Chunk with embedding added
        """
        # Extract content for embedding
        content = chunk.get('content', '')
        
        if not content:
            logger.warning(f"Empty content in chunk {chunk.get('id', 'unknown')}")
            # Create a placeholder embedding of zeros
            if self.model is None:
                self.load_model()
            embedding = np.zeros(self.embedding_dim)
        else:
            # Generate embedding
            embedding = self.generate_embedding(content)
        
        # Add embedding to chunk
        chunk['embedding'] = embedding.tolist()
        
        return chunk
    
    def process_chunks_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of chunks by adding embeddings.
        
        Args:
            chunks: List of conversation chunks
            
        Returns:
            List of chunks with embeddings added
        """
        # Extract contents for embedding
        contents = []
        for chunk in chunks:
            content = chunk.get('content', '')
            if not content:
                logger.warning(f"Empty content in chunk {chunk.get('id', 'unknown')}")
                content = " "  # Use a space as placeholder
            contents.append(content)
        
        # Generate embeddings in batch
        embeddings = self.generate_embeddings_batch(contents)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
        
        return chunks
    
    def save_embedded_chunk(self, chunk: Dict[str, Any], output_dir: str) -> str:
        """
        Save an embedded chunk to a file.
        
        Args:
            chunk: Chunk with embedding
            output_dir: Directory to save the file
            
        Returns:
            Path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        chunk_id = chunk.get('id', str(uuid.uuid4()))
        conv_id = chunk.get('conversation_id', 'unknown')
        chunk_type = chunk.get('chunk_type', 'unknown')
        
        filename = f"{conv_id}_{chunk_type}_{chunk_id}.json"
        file_path = os.path.join(output_dir, filename)
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)
        
        return file_path


class EmbeddingPipeline:
    """Pipeline for generating embeddings for conversation chunks."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32):
        """
        Initialize the embedding pipeline.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            batch_size: Batch size for embedding generation
        """
        self.embedding_generator = EmbeddingGenerator(model_name)
        self.batch_size = batch_size
    
    def process_chunks_directory(self, chunks_dir: str, output_dir: str) -> Dict[str, List[str]]:
        """
        Process all chunks in a directory.
        
        Args:
            chunks_dir: Directory containing chunk files
            output_dir: Directory to save embedded chunks
            
        Returns:
            Dictionary mapping chunk types to lists of file paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # Process each subdirectory (chunk type)
        for chunk_type in os.listdir(chunks_dir):
            chunk_type_dir = os.path.join(chunks_dir, chunk_type)
            
            if not os.path.isdir(chunk_type_dir):
                continue
            
            # Create output subdirectory for this chunk type
            chunk_type_output_dir = os.path.join(output_dir, chunk_type)
            os.makedirs(chunk_type_output_dir, exist_ok=True)
            
            # Get all chunk files
            chunk_files = [f for f in os.listdir(chunk_type_dir) if f.endswith('.json')]
            
            if not chunk_files:
                logger.warning(f"No chunk files found in {chunk_type_dir}")
                continue
            
            # Process chunks in batches
            file_paths = []
            
            for i in range(0, len(chunk_files), self.batch_size):
                batch_files = chunk_files[i:i+self.batch_size]
                
                # Load chunks
                chunks = []
                for filename in batch_files:
                    file_path = os.path.join(chunk_type_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            chunk = json.load(f)
                        chunks.append(chunk)
                    except Exception as e:
                        logger.error(f"Error loading chunk file {file_path}: {e}")
                
                # Generate embeddings
                embedded_chunks = self.embedding_generator.process_chunks_batch(chunks)
                
                # Save embedded chunks
                for chunk in embedded_chunks:
                    try:
                        file_path = self.embedding_generator.save_embedded_chunk(
                            chunk, chunk_type_output_dir)
                        file_paths.append(file_path)
                    except Exception as e:
                        logger.error(f"Error saving embedded chunk: {e}")
            
            results[chunk_type] = file_paths
            logger.info(f"Processed {len(file_paths)} chunks of type {chunk_type}")
        
        return results
    
    def process_single_chunk_file(self, file_path: str, output_dir: str) -> str:
        """
        Process a single chunk file.
        
        Args:
            file_path: Path to the chunk file
            output_dir: Directory to save the embedded chunk
            
        Returns:
            Path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load chunk
            with open(file_path, 'r', encoding='utf-8') as f:
                chunk = json.load(f)
            
            # Generate embedding
            embedded_chunk = self.embedding_generator.process_chunk(chunk)
            
            # Save embedded chunk
            output_path = self.embedding_generator.save_embedded_chunk(embedded_chunk, output_dir)
            
            return output_path
        except Exception as e:
            logger.error(f"Error processing chunk file {file_path}: {e}")
            raise


# Example usage
if __name__ == "__main__":
    import os
    from chunker import ChunkingPipeline
    from parsers import ConversationProcessor
    
    # Example paths
    sample_dir = "/home/ubuntu/llm_memory_system/data_extraction/sample_data"
    processed_dir = "/home/ubuntu/llm_memory_system/data_extraction/processed_data"
    chunks_dir = "/home/ubuntu/llm_memory_system/data_extraction/chunks"
    embedded_dir = "/home/ubuntu/llm_memory_system/data_extraction/embedded_chunks"
    
    # Create directories
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(embedded_dir, exist_ok=True)
    
    # Process a sample file
    processor = ConversationProcessor()
    chunking_pipeline = ChunkingPipeline(
        strategies=['message', 'exchange'],
        chunk_size=1000,
        chunk_overlap=200
    )
    embedding_pipeline = EmbeddingPipeline(model_name='all-MiniLM-L6-v2')
    
    # Process the first file in the sample directory
    for filename in os.listdir(sample_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(sample_dir, filename)
            
            # Process the file
            conversation = processor.process_file(file_path)
            
            # Save processed conversation
            processed_path = processor.save_processed_conversation(conversation, processed_dir)
            print(f"Processed and saved: {processed_path}")
            
            # Apply chunking strategies
            chunks = chunking_pipeline.process_conversation(conversation)
            
            # Save chunks
            chunk_paths = chunking_pipeline.save_chunks(chunks, chunks_dir)
            
            for strategy, paths in chunk_paths.items():
                print(f"Created {len(paths)} chunks using {strategy} strategy")
            
            # Generate embeddings
            embedded_paths = embedding_pipeline.process_chunks_directory(chunks_dir, embedded_dir)
            
            for strategy, paths in embedded_paths.items():
                print(f"Generated embeddings for {len(paths)} chunks of type {strategy}")
            
            # Only process one file for this example
            break
