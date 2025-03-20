"""
Mem0 component for personalized memory in LLM memory system.

This module provides the Mem0 component that stores and retrieves personalized memory
such as user preferences, past insights, and key conversation takeaways.
"""

import os
import json
import uuid
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryEntry:
    """A single memory entry in the Mem0 component."""
    
    def __init__(
        self,
        content: str,
        memory_type: str,
        source_conversation_id: Optional[str] = None,
        source_message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        memory_id: Optional[str] = None,
        created_at: Optional[str] = None,
        last_accessed: Optional[str] = None,
        access_count: int = 0
    ):
        """
        Initialize a memory entry.
        
        Args:
            content: Content of the memory
            memory_type: Type of memory (preference, insight, fact, etc.)
            source_conversation_id: ID of the source conversation
            source_message_id: ID of the source message
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)
            memory_id: Unique identifier (generated if not provided)
            created_at: Creation timestamp (current time if not provided)
            last_accessed: Last access timestamp
            access_count: Number of times this memory has been accessed
        """
        self.content = content
        self.memory_type = memory_type
        self.source_conversation_id = source_conversation_id
        self.source_message_id = source_message_id
        self.metadata = metadata or {}
        self.importance = importance
        self.memory_id = memory_id or str(uuid.uuid4())
        self.created_at = created_at or datetime.now().isoformat()
        self.last_accessed = last_accessed
        self.access_count = access_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the memory entry to a dictionary."""
        return {
            'memory_id': self.memory_id,
            'content': self.content,
            'memory_type': self.memory_type,
            'source_conversation_id': self.source_conversation_id,
            'source_message_id': self.source_message_id,
            'metadata': self.metadata,
            'importance': self.importance,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create a memory entry from a dictionary."""
        return cls(
            content=data['content'],
            memory_type=data['memory_type'],
            source_conversation_id=data.get('source_conversation_id'),
            source_message_id=data.get('source_message_id'),
            metadata=data.get('metadata', {}),
            importance=data.get('importance', 0.5),
            memory_id=data.get('memory_id'),
            created_at=data.get('created_at'),
            last_accessed=data.get('last_accessed'),
            access_count=data.get('access_count', 0)
        )
    
    def access(self) -> None:
        """Update access statistics for this memory entry."""
        self.last_accessed = datetime.now().isoformat()
        self.access_count += 1


class Mem0Component:
    """Mem0 component for personalized memory."""
    
    def __init__(self, storage_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Mem0 component.
        
        Args:
            storage_path: Path to the storage directory or file
            config: Configuration dictionary
        """
        self.storage_path = storage_path
        self.config = config or {}
        
        # Default configuration values
        self.max_memories = self.config.get('max_memories', 1000)
        self.memory_types = self.config.get('memory_types', [
            'preference', 'insight', 'fact', 'personal_info', 'conversation_summary'
        ])
        
        # Initialize memory store
        self.memories: Dict[str, MemoryEntry] = {}
        self.memories_by_type: Dict[str, List[str]] = {
            memory_type: [] for memory_type in self.memory_types
        }
        
        # Load existing memories if available
        self._load_memories()
    
    def _load_memories(self) -> None:
        """Load memories from storage."""
        if os.path.isfile(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load memories
                for memory_data in data.get('memories', []):
                    memory = MemoryEntry.from_dict(memory_data)
                    self.memories[memory.memory_id] = memory
                
                # Rebuild memories_by_type index
                self.memories_by_type = {
                    memory_type: [] for memory_type in self.memory_types
                }
                
                for memory_id, memory in self.memories.items():
                    if memory.memory_type in self.memories_by_type:
                        self.memories_by_type[memory.memory_type].append(memory_id)
                    else:
                        # Add new memory type if encountered
                        self.memories_by_type[memory.memory_type] = [memory_id]
                        if memory.memory_type not in self.memory_types:
                            self.memory_types.append(memory.memory_type)
                
                logger.info(f"Loaded {len(self.memories)} memories from {self.storage_path}")
            except Exception as e:
                logger.error(f"Error loading memories: {e}")
                # Initialize empty memories
                self.memories = {}
                self.memories_by_type = {
                    memory_type: [] for memory_type in self.memory_types
                }
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
    
    def _save_memories(self) -> None:
        """Save memories to storage."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
            
            # Convert memories to list of dictionaries
            memories_data = [
                memory.to_dict() for memory in self.memories.values()
            ]
            
            # Save to file
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'memories': memories_data,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.memories)} memories to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving memories: {e}")
    
    def add_memory(self, memory: MemoryEntry) -> str:
        """
        Add a memory entry.
        
        Args:
            memory: Memory entry to add
            
        Returns:
            Memory ID
        """
        # Add to memories
        self.memories[memory.memory_id] = memory
        
        # Add to memories_by_type
        if memory.memory_type in self.memories_by_type:
            self.memories_by_type[memory.memory_type].append(memory.memory_id)
        else:
            # Add new memory type if encountered
            self.memories_by_type[memory.memory_type] = [memory.memory_id]
            if memory.memory_type not in self.memory_types:
                self.memory_types.append(memory.memory_type)
        
        # Check if we need to prune memories
        if len(self.memories) > self.max_memories:
            self._prune_memories()
        
        # Save memories
        self._save_memories()
        
        return memory.memory_id
    
    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        Get a memory entry by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory entry or None if not found
        """
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.access()
            return memory
        
        return None
    
    def get_memories_by_type(self, memory_type: str, limit: int = 10) -> List[MemoryEntry]:
        """
        Get memory entries by type.
        
        Args:
            memory_type: Memory type
            limit: Maximum number of memories to return
            
        Returns:
            List of memory entries
        """
        if memory_type not in self.memories_by_type:
            return []
        
        memory_ids = self.memories_by_type[memory_type]
        
        # Get memories and sort by importance (descending)
        memories = [self.memories[memory_id] for memory_id in memory_ids]
        memories.sort(key=lambda m: m.importance, reverse=True)
        
        # Update access statistics
        for memory in memories[:limit]:
            memory.access()
        
        return memories[:limit]
    
    def search_memories(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """
        Search for memories by content.
        
        Args:
            query: Search query
            limit: Maximum number of memories to return
            
        Returns:
            List of memory entries
        """
        # Simple keyword search for now
        query_lower = query.lower()
        
        # Score memories by relevance
        scored_memories = []
        
        for memory in self.memories.values():
            content_lower = memory.content.lower()
            
            # Calculate simple relevance score
            if query_lower in content_lower:
                # Higher score for exact matches
                score = 1.0
            else:
                # Check for partial matches
                words = query_lower.split()
                matches = sum(1 for word in words if word in content_lower)
                score = matches / len(words) if words else 0.0
            
            # Combine with importance
            combined_score = (score * 0.7) + (memory.importance * 0.3)
            
            if combined_score > 0:
                scored_memories.append((memory, combined_score))
        
        # Sort by score (descending)
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Get top memories
        top_memories = [memory for memory, _ in scored_memories[:limit]]
        
        # Update access statistics
        for memory in top_memories:
            memory.access()
        
        return top_memories
    
    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a memory entry.
        
        Args:
            memory_id: Memory ID
            updates: Dictionary of updates
            
        Returns:
            True if successful, False otherwise
        """
        if memory_id not in self.memories:
            return False
        
        memory = self.memories[memory_id]
        
        # Update fields
        if 'content' in updates:
            memory.content = updates['content']
        
        if 'memory_type' in updates:
            old_type = memory.memory_type
            new_type = updates['memory_type']
            
            # Update memory type
            memory.memory_type = new_type
            
            # Update memories_by_type
            if old_type in self.memories_by_type and memory_id in self.memories_by_type[old_type]:
                self.memories_by_type[old_type].remove(memory_id)
            
            if new_type in self.memories_by_type:
                self.memories_by_type[new_type].append(memory_id)
            else:
                self.memories_by_type[new_type] = [memory_id]
                if new_type not in self.memory_types:
                    self.memory_types.append(new_type)
        
        if 'importance' in updates:
            memory.importance = updates['importance']
        
        if 'metadata' in updates:
            memory.metadata.update(updates['metadata'])
        
        # Save memories
        self._save_memories()
        
        return True
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory entry.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            True if successful, False otherwise
        """
        if memory_id not in self.memories:
            return False
        
        memory = self.memories[memory_id]
        
        # Remove from memories_by_type
        if memory.memory_type in self.memories_by_type and memory_id in self.memories_by_type[memory.memory_type]:
            self.memories_by_type[memory.memory_type].remove(memory_id)
        
        # Remove from memories
        del self.memories[memory_id]
        
        # Save memories
        self._save_memories()
        
        return True
    
    def _prune_memories(self) -> None:
        """Prune memories to stay within max_memories limit."""
        if len(self.memories) <= self.max_memories:
            return
        
        # Calculate how many memories to remove
        num_to_remove = len(self.memories) - self.max_memories
        
        # Score memories by importance, recency, and access count
        scored_memories = []
        
        for memory_id, memory in self.memories.items():
            # Calculate recency score (higher for newer memories)
            if memory.created_at:
                try:
                    created_dt = datetime.fromisoformat(memory.created_at.replace('Z', '+00:00'))
                    now = datetime.now()
                    age_days = (now - created_dt).days
                    recency_score = 1.0 / (1.0 + age_days)
                except (ValueError, TypeError):
                    recency_score = 0.5
            else:
                recency_score = 0.5
            
            # Calculate access score (higher for frequently accessed memories)
            access_score = min(1.0, memory.access_count / 10.0)
            
            # Combined score (higher is better to keep)
            combined_score = (
                (memory.importance * 0.5) +
                (recency_score * 0.3) +
                (access_score * 0.2)
            )
            
            scored_memories.append((memory_id, combined_score))
        
        # Sort by score (ascending, so we remove lowest scores first)
        scored_memories.sort(key=lambda x: x[1])
        
        # Remove lowest-scored memories
        for memory_id, _ in scored_memories[:num_to_remove]:
            self.delete_memory(memory_id)
    
    def get_memory_summary(self, memory_types: Optional[List[str]] = None, limit_per_type: int = 5) -> Dict[str, List[MemoryEntry]]:
        """
        Get a summary of memories by type.
        
        Args:
            memory_types: List of memory types to include (all if None)
            limit_per_type: Maximum number of memories per type
            
        Returns:
            Dictionary mapping memory types to lists of memory entries
        """
        types_to_include = memory_types or self.memory_types
        
        summary = {}
        
        for memory_type in types_to_include:
            memories = self.get_memories_by_type(memory_type, limit=limit_per_type)
            if memories:
                summary[memory_type] = memories
        
        return summary
    
    def format_memories_for_prompt(self, memories: List[MemoryEntry]) -> str:
        """
        Format memories for inclusion in an LLM prompt.
        
        Args:
            memories: List of memory entries
            
        Returns:
            Formatted string for prompt
        """
        if not memories:
            return ""
        
        # Group memories by type
        memories_by_type = {}
        
        for memory in memories:
            if memory.memory_type not in memories_by_type:
                memories_by_type[memory.memory_type] = []
            
            memories_by_type[memory.memory_type].append(memory)
        
        # Format each type
        sections = []
        
        for memory_type, type_memories in memories_by_type.items():
            # Format type name
            type_name = memory_type.replace('_', ' ').title()
            
            # Format memories
            memory_items = [f"- {memory.content}" for memory in type_memories]
            
            # Create section
            section = f"{type_name}:\n" + "\n".join(memory_items)
            sections.append(section)
        
        return "\n\n".join(sections)
    
    def get_relevant_memories_for_prompt(self, query: str, memory_types: Optional[List[str]] = None, limit: int = 10) -> str:
        """
        Get relevant memories formatted for inclusion in an LLM prompt.
        
        Args:
            query: User query
            memory_types: List of memory types to include (all if None)
            limit: Maximum number of memories to return
            
        Returns:
            Formatted string for prompt
        """
        # Search for relevant memories
        relevant_memories = self.search_memories(query, limit=limit)
        
        # Filter by memory types if specified
        if memory_types:
            relevant_memories = [
                memory for memory in relevant_memories
                if memory.memory_type in memory_types
            ]
        
        # Format for prompt
        formatted_memories = self.format_memories_for_prompt(relevant_memories)
        
        if not formatted_memories:
            return ""
        
        return f"""
Personal Memory:

{formatted_memories}

Use this personal memory to inform your response if relevant to the user's query.
"""


class MemoryExtractor:
    """Extract memories from conversation messages."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the memory extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default configuration values
        self.extraction_patterns = self.config.get('extraction_patterns', {
            'preference': [
                r"(?:I|user) (?:like|prefer|enjoy|love|hate|dislike) (.+)",
                r"(?:I|user) (?:don't|do not) (?:like|prefer|enjoy|love) (.+)",
                r"(?:my|user's) favorite (.+) is (.+)"
            ],
            'personal_info': [
                r"(?:I|user) (?:am|is) (?:from|in) (.+)",
                r"(?:I|user) (?:work|works) (?:at|for|as) (.+)",
                r"(?:my|user's) (?:name|email|phone|address) is (.+)"
            ]
        })
        
        self.extraction_keywords = self.config.get('extraction_keywords', {
            'preference': ['like', 'prefer', 'enjoy', 'love', 'hate', 'dislike', 'favorite'],
            'personal_info': ['name', 'email', 'phone', 'address', 'from', 'work', 'job', 'profession']
        })
    
    def extract_memories_from_message(self, message: Dict[str, Any]) -> List[MemoryEntry]:
        """
        Extract memories from a message.
        
        Args:
            message: Message dictionary
            
        Returns:
            List of extracted memory entries
        """
        # Extract message content
        content = message.get('content', '')
        if not content:
            return []
        
        # Extract message metadata
        conversation_id = message.get('conversation_id', '')
        message_id = message.get('id', '')
        timestamp = message.get('timestamp', datetime.now().isoformat())
        
        # Extract memories using patterns
        memories = []
        
        # Simple keyword-based extraction for now
        for memory_type, keywords in self.extraction_keywords.items():
            content_lower = content.lower()
            
            for keyword in keywords:
                if keyword in content_lower:
                    # Find sentences containing the keyword
                    sentences = content.split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            # Create memory entry
                            memory = MemoryEntry(
                                content=sentence.strip(),
                                memory_type=memory_type,
                                source_conversation_id=conversation_id,
                                source_message_id=message_id,
                                metadata={
                                    'extracted_from': 'message',
                                    'extraction_method': 'keyword',
                                    'keyword': keyword
                                },
                                importance=0.7,
                                created_at=timestamp
                            )
                            
                            memories.append(memory)
        
        return memories
    
    def extract_memories_from_conversation(self, conversation: Dict[str, Any]) -> List[MemoryEntry]:
        """
        Extract memories from a conversation.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            List of extracted memory entries
        """
        # Extract conversation metadata
        conversation_id = conversation.get('id', '')
        title = conversation.get('title', '')
        
        # Extract messages
        messages = conversation.get('messages', [])
        
        # Extract memories from each message
        all_memories = []
        
        for message in messages:
            memories = self.extract_memories_from_message(message)
            all_memories.extend(memories)
        
        # Create conversation summary memory
        if messages:
            summary = f"Conversation about {title}" if title else "Conversation"
            
            summary_memory = MemoryEntry(
                content=summary,
                memory_type='conversation_summary',
                source_conversation_id=conversation_id,
                metadata={
                    'conversation_title': title,
                    'message_count': len(messages)
                },
                importance=0.6
            )
            
            all_memories.append(summary_memory)
        
        return all_memories


# Example usage
if __name__ == "__main__":
    import os
    
    # Example paths
    storage_path = "/home/ubuntu/llm_memory_system/mem0_component/memories.json"
    
    # Create Mem0 component
    mem0 = Mem0Component(storage_path)
    
    # Create memory extractor
    extractor = MemoryExtractor()
    
    # Example message
    message = {
        'content': "I really enjoy programming in Python. It's my favorite language.",
        'conversation_id': 'conv1',
        'id': 'msg1',
        'timestamp': datetime.now().isoformat()
    }
    
    # Extract memories
    memories = extractor.extract_memories_from_message(message)
    
    print(f"Extracted {len(memories)} memories from message")
    
    # Add memories to Mem0
    for memory in memories:
        mem0.add_memory(memory)
    
    # Add some example memories
    mem0.add_memory(MemoryEntry(
        content="User prefers dark mode in applications",
        memory_type="preference",
        importance=0.8
    ))
    
    mem0.add_memory(MemoryEntry(
        content="User is allergic to peanuts",
        memory_type="personal_info",
        importance=0.9
    ))
    
    mem0.add_memory(MemoryEntry(
        content="User's favorite color is blue",
        memory_type="preference",
        importance=0.7
    ))
    
    # Search memories
    query = "What does the user like?"
    relevant_memories = mem0.search_memories(query)
    
    print(f"\nRelevant memories for query '{query}':")
    for memory in relevant_memories:
        print(f"- {memory.content} (type: {memory.memory_type}, importance: {memory.importance})")
    
    # Get formatted memories for prompt
    formatted_memories = mem0.get_relevant_memories_for_prompt(query)
    
    print("\nFormatted memories for prompt:")
    print(formatted_memories)
