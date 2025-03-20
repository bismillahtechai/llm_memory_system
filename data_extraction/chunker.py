"""
Chunking strategies for conversation data.

This module provides various chunking strategies to break conversations into
meaningful units for embedding and retrieval.
"""

import re
import uuid
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseChunker:
    """Base class for conversation chunkers."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_conversation(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a conversation into smaller units.
        
        Args:
            conversation: Processed conversation data
            
        Returns:
            List of chunk dictionaries
        """
        raise NotImplementedError("Subclasses must implement chunk_conversation method")
    
    def _create_chunk(self, 
                     content: str, 
                     metadata: Dict[str, Any], 
                     chunk_type: str,
                     source_message_ids: List[str] = None) -> Dict[str, Any]:
        """
        Create a standardized chunk dictionary.
        
        Args:
            content: Chunk content
            metadata: Conversation metadata
            chunk_type: Type of chunk (e.g., 'message', 'exchange', 'topic')
            source_message_ids: List of message IDs that make up this chunk
            
        Returns:
            Chunk dictionary
        """
        # Generate a unique chunk ID
        chunk_id = str(uuid.uuid4())
        
        # Create timestamp
        timestamp = datetime.now().isoformat()
        
        return {
            'id': chunk_id,
            'content': content,
            'chunk_type': chunk_type,
            'source_message_ids': source_message_ids or [],
            'conversation_id': metadata.get('conversation_id', ''),
            'conversation_title': metadata.get('title', ''),
            'source': metadata.get('source', ''),
            'timestamp': timestamp,
            'metadata': {
                'conversation_metadata': metadata,
                'chunk_size': len(content),
                'creation_time': timestamp
            }
        }


class MessageChunker(BaseChunker):
    """Chunker that treats each message as a separate chunk."""
    
    def chunk_conversation(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a conversation by treating each message as a separate chunk.
        
        Args:
            conversation: Processed conversation data
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        metadata = conversation['metadata']
        messages = conversation['messages']
        
        for message in messages:
            # Skip empty messages
            if not message.get('content', '').strip():
                continue
            
            # Create a chunk for the message
            chunk = self._create_chunk(
                content=message['content'],
                metadata=metadata,
                chunk_type='message',
                source_message_ids=[message.get('id', '')]
            )
            
            # Add message-specific metadata
            chunk['metadata']['role'] = message.get('role', '')
            chunk['metadata']['message_timestamp'] = message.get('timestamp', '')
            
            chunks.append(chunk)
        
        return chunks


class ExchangeChunker(BaseChunker):
    """Chunker that groups user-assistant exchanges as chunks."""
    
    def chunk_conversation(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a conversation by grouping user-assistant exchanges.
        
        Args:
            conversation: Processed conversation data
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        metadata = conversation['metadata']
        messages = conversation['messages']
        
        # Group messages into exchanges (user + assistant)
        exchanges = []
        current_exchange = []
        current_exchange_ids = []
        
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '').strip()
            message_id = message.get('id', '')
            
            # Skip empty messages
            if not content:
                continue
            
            # If we have a user message and the current exchange already has content,
            # start a new exchange
            if role == 'user' and current_exchange:
                exchanges.append((current_exchange, current_exchange_ids))
                current_exchange = []
                current_exchange_ids = []
            
            # Add message to current exchange
            current_exchange.append(f"{role.upper()}: {content}")
            current_exchange_ids.append(message_id)
        
        # Add the last exchange if it has content
        if current_exchange:
            exchanges.append((current_exchange, current_exchange_ids))
        
        # Create chunks from exchanges
        for exchange_content, exchange_ids in exchanges:
            content = "\n\n".join(exchange_content)
            
            # Create a chunk for the exchange
            chunk = self._create_chunk(
                content=content,
                metadata=metadata,
                chunk_type='exchange',
                source_message_ids=exchange_ids
            )
            
            chunks.append(chunk)
        
        return chunks


class SlidingWindowChunker(BaseChunker):
    """Chunker that creates overlapping chunks of specified size."""
    
    def chunk_conversation(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a conversation using a sliding window approach.
        
        Args:
            conversation: Processed conversation data
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        metadata = conversation['metadata']
        messages = conversation['messages']
        
        # Concatenate all messages into a single text
        full_text = ""
        message_boundaries = []  # Track where each message starts
        message_ids = []
        
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '').strip()
            message_id = message.get('id', '')
            
            # Skip empty messages
            if not content:
                continue
            
            # Add message to full text
            start_pos = len(full_text)
            message_text = f"{role.upper()}: {content}\n\n"
            full_text += message_text
            
            # Record message boundary and ID
            message_boundaries.append((start_pos, len(full_text), message_id))
            message_ids.append(message_id)
        
        # Create sliding window chunks
        start = 0
        while start < len(full_text):
            end = start + self.chunk_size
            
            # If this is not the first chunk, include overlap
            if start > 0:
                start = max(0, start - self.chunk_overlap)
            
            # Get chunk text
            chunk_text = full_text[start:end]
            
            # Find which message IDs are included in this chunk
            chunk_message_ids = []
            for msg_start, msg_end, msg_id in message_boundaries:
                # If any part of the message is in the chunk, include its ID
                if (msg_start < end and msg_end > start):
                    chunk_message_ids.append(msg_id)
            
            # Create a chunk
            chunk = self._create_chunk(
                content=chunk_text,
                metadata=metadata,
                chunk_type='sliding_window',
                source_message_ids=chunk_message_ids
            )
            
            chunks.append(chunk)
            
            # Move to next chunk
            start = end
        
        return chunks


class SemanticChunker(BaseChunker):
    """Chunker that attempts to create semantically coherent chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 sentence_splitter: Optional[callable] = None):
        """
        Initialize the semantic chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            sentence_splitter: Optional function to split text into sentences
        """
        super().__init__(chunk_size, chunk_overlap)
        self.sentence_splitter = sentence_splitter or self._default_sentence_splitter
    
    def _default_sentence_splitter(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple regex-based sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_conversation(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a conversation into semantically coherent units.
        
        Args:
            conversation: Processed conversation data
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        metadata = conversation['metadata']
        messages = conversation['messages']
        
        # Process each message
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '').strip()
            message_id = message.get('id', '')
            
            # Skip empty messages
            if not content:
                continue
            
            # Split message into sentences
            sentences = self._default_sentence_splitter(content)
            
            # Group sentences into chunks
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # If adding this sentence would exceed chunk size and we already have content,
                # create a chunk and start a new one
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    chunk_text = f"{role.upper()}: " + " ".join(current_chunk)
                    
                    chunk = self._create_chunk(
                        content=chunk_text,
                        metadata=metadata,
                        chunk_type='semantic',
                        source_message_ids=[message_id]
                    )
                    
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    overlap_sentences = []
                    overlap_length = 0
                    
                    # Include some sentences from the previous chunk for context
                    for prev_sentence in reversed(current_chunk):
                        if overlap_length + len(prev_sentence) > self.chunk_overlap:
                            break
                        overlap_sentences.insert(0, prev_sentence)
                        overlap_length += len(prev_sentence) + 1  # +1 for space
                    
                    current_chunk = overlap_sentences
                    current_length = overlap_length
                
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space
            
            # Create a chunk from any remaining content
            if current_chunk:
                chunk_text = f"{role.upper()}: " + " ".join(current_chunk)
                
                chunk = self._create_chunk(
                    content=chunk_text,
                    metadata=metadata,
                    chunk_type='semantic',
                    source_message_ids=[message_id]
                )
                
                chunks.append(chunk)
        
        return chunks


class TopicChunker(BaseChunker):
    """
    Chunker that attempts to identify topic changes in conversations.
    
    Note: This is a simplified implementation. For production use,
    consider using more sophisticated topic detection algorithms.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 time_threshold: int = 3600):
        """
        Initialize the topic chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            time_threshold: Time threshold in seconds to consider a topic change
        """
        super().__init__(chunk_size, chunk_overlap)
        self.time_threshold = time_threshold
    
    def chunk_conversation(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a conversation based on potential topic changes.
        
        Args:
            conversation: Processed conversation data
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        metadata = conversation['metadata']
        messages = conversation['messages']
        
        # Group messages into potential topics
        topics = []
        current_topic = []
        current_topic_ids = []
        last_timestamp = None
        
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '').strip()
            message_id = message.get('id', '')
            timestamp_str = message.get('timestamp', '')
            
            # Skip empty messages
            if not content:
                continue
            
            # Parse timestamp
            current_timestamp = None
            if timestamp_str:
                try:
                    current_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    pass
            
            # Check for potential topic change based on time gap
            if (last_timestamp and current_timestamp and 
                (current_timestamp - last_timestamp).total_seconds() > self.time_threshold):
                # Time gap detected, start a new topic
                if current_topic:
                    topics.append((current_topic, current_topic_ids))
                    current_topic = []
                    current_topic_ids = []
            
            # Add message to current topic
            current_topic.append(f"{role.upper()}: {content}")
            current_topic_ids.append(message_id)
            
            # Update last timestamp
            if current_timestamp:
                last_timestamp = current_timestamp
        
        # Add the last topic if it has content
        if current_topic:
            topics.append((current_topic, current_topic_ids))
        
        # Create chunks from topics
        for topic_content, topic_ids in topics:
            content = "\n\n".join(topic_content)
            
            # If topic is too large, use sliding window chunker
            if len(content) > self.chunk_size:
                # Create a mini-conversation for the sliding window chunker
                mini_conv = {
                    'metadata': metadata,
                    'messages': [
                        {
                            'role': 'system',
                            'content': content,
                            'id': 'topic_' + str(uuid.uuid4())
                        }
                    ]
                }
                
                # Use sliding window chunker
                sliding_chunker = SlidingWindowChunker(self.chunk_size, self.chunk_overlap)
                topic_chunks = sliding_chunker.chunk_conversation(mini_conv)
                
                # Update chunk type and source message IDs
                for chunk in topic_chunks:
                    chunk['chunk_type'] = 'topic'
                    chunk['source_message_ids'] = topic_ids
                
                chunks.extend(topic_chunks)
            else:
                # Create a single chunk for the topic
                chunk = self._create_chunk(
                    content=content,
                    metadata=metadata,
                    chunk_type='topic',
                    source_message_ids=topic_ids
                )
                
                chunks.append(chunk)
        
        return chunks


class ChunkingPipeline:
    """Pipeline for applying multiple chunking strategies."""
    
    def __init__(self, strategies: List[str] = None, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the chunking pipeline.
        
        Args:
            strategies: List of chunking strategies to apply
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default strategies if none provided
        self.strategies = strategies or ['message', 'exchange', 'sliding_window']
        
        # Initialize chunkers
        self.chunkers = {
            'message': MessageChunker(chunk_size, chunk_overlap),
            'exchange': ExchangeChunker(chunk_size, chunk_overlap),
            'sliding_window': SlidingWindowChunker(chunk_size, chunk_overlap),
            'semantic': SemanticChunker(chunk_size, chunk_overlap),
            'topic': TopicChunker(chunk_size, chunk_overlap)
        }
    
    def process_conversation(self, conversation: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a conversation using multiple chunking strategies.
        
        Args:
            conversation: Processed conversation data
            
        Returns:
            Dictionary mapping strategy names to lists of chunks
        """
        results = {}
        
        for strategy in self.strategies:
            if strategy in self.chunkers:
                chunker = self.chunkers[strategy]
                try:
                    chunks = chunker.chunk_conversation(conversation)
                    results[strategy] = chunks
                except Exception as e:
                    logger.error(f"Error applying {strategy} chunking strategy: {e}")
            else:
                logger.warning(f"Unknown chunking strategy: {strategy}")
        
        return results
    
    def save_chunks(self, chunks: Dict[str, List[Dict[str, Any]]], output_dir: str) -> Dict[str, List[str]]:
        """
        Save chunks to files.
        
        Args:
            chunks: Dictionary mapping strategy names to lists of chunks
            output_dir: Directory to save the files
            
        Returns:
            Dictionary mapping strategy names to lists of file paths
        """
        import os
        import json
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for strategy, chunk_list in chunks.items():
            strategy_dir = os.path.join(output_dir, strategy)
            os.makedirs(strategy_dir, exist_ok=True)
            
            file_paths = []
            
            for chunk in chunk_list:
                # Generate filename
                chunk_id = chunk['id']
                conv_id = chunk['conversation_id']
                
                if not conv_id:
                    conv_id = 'unknown'
                
                filename = f"{conv_id}_{chunk_id}.json"
                file_path = os.path.join(strategy_dir, filename)
                
                # Save to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(chunk, f, indent=2, ensure_ascii=False)
                
                file_paths.append(file_path)
            
            results[strategy] = file_paths
        
        return results


# Example usage
if __name__ == "__main__":
    import os
    import json
    from parsers import ConversationProcessor
    
    # Example paths
    sample_dir = "/home/ubuntu/llm_memory_system/data_extraction/sample_data"
    processed_dir = "/home/ubuntu/llm_memory_system/data_extraction/processed_data"
    chunks_dir = "/home/ubuntu/llm_memory_system/data_extraction/chunks"
    
    # Create directories
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Process a sample file
    processor = ConversationProcessor()
    chunking_pipeline = ChunkingPipeline(
        strategies=['message', 'exchange', 'sliding_window', 'semantic', 'topic'],
        chunk_size=1000,
        chunk_overlap=200
    )
    
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
            
            # Only process one file for this example
            break
