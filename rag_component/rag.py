"""
RAG (Retrieval-Augmented Generation) component for LLM memory system.

This module provides the RAG component that retrieves relevant context from
past conversations to augment LLM prompts.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGComponent:
    """RAG component for retrieving relevant context from past conversations."""
    
    def __init__(self, storage_manager, embedding_generator, config: Dict[str, Any]):
        """
        Initialize the RAG component.
        
        Args:
            storage_manager: Storage manager instance
            embedding_generator: Embedding generator instance
            config: Configuration dictionary
        """
        self.storage_manager = storage_manager
        self.embedding_generator = embedding_generator
        self.config = config
        
        # Default configuration values
        self.max_results = config.get('max_results', 5)
        self.similarity_threshold = config.get('similarity_threshold', 0.6)
        self.token_budget = config.get('token_budget', 1000)
        self.preferred_chunk_types = config.get('preferred_chunk_types', 
                                              ['exchange', 'message', 'sliding_window'])
    
    def retrieve(self, query: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            conversation_id: Optional current conversation ID
            
        Returns:
            Dictionary with retrieved context and metadata
        """
        logger.info(f"Retrieving context for query: {query[:50]}...")
        
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search for similar chunks
        results = self.storage_manager.search_similar_chunks(
            query_embedding.tolist(), limit=self.max_results * 2)
        
        # Filter results by similarity threshold
        filtered_results = [
            r for r in results 
            if r['similarity'] >= self.similarity_threshold
        ]
        
        # Organize results by chunk type
        results_by_type = {}
        for result in filtered_results:
            chunk = result['chunk']
            chunk_type = chunk.get('chunk_type', 'unknown')
            
            if chunk_type not in results_by_type:
                results_by_type[chunk_type] = []
            
            results_by_type[chunk_type].append({
                'content': chunk.get('content', ''),
                'similarity': result['similarity'],
                'conversation_id': chunk.get('conversation_id', ''),
                'conversation_title': chunk.get('conversation_title', ''),
                'chunk_id': chunk.get('id', ''),
                'metadata': chunk.get('metadata', {})
            })
        
        # Select results based on preferred chunk types
        selected_results = []
        for chunk_type in self.preferred_chunk_types:
            if chunk_type in results_by_type:
                # Sort by similarity (descending)
                type_results = sorted(
                    results_by_type[chunk_type], 
                    key=lambda x: x['similarity'], 
                    reverse=True
                )
                
                # Add top results
                selected_results.extend(type_results[:self.max_results])
        
        # Limit to max_results
        selected_results = selected_results[:self.max_results]
        
        # Format context
        context = self._format_context(selected_results)
        
        return {
            'context': context,
            'results': selected_results,
            'query': query,
            'timestamp': datetime.now().isoformat()
        }
    
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved results into context string.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        
        for i, result in enumerate(results):
            content = result['content']
            similarity = result['similarity']
            conversation_title = result['conversation_title']
            
            # Format the context entry
            context_part = f"[{i+1}] {content}"
            
            # Add source information if available
            if conversation_title:
                context_part += f"\n(From: {conversation_title}, Relevance: {similarity:.2f})"
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def retrieve_and_format_for_prompt(self, query: str, conversation_id: Optional[str] = None) -> str:
        """
        Retrieve context and format it for inclusion in an LLM prompt.
        
        Args:
            query: User query
            conversation_id: Optional current conversation ID
            
        Returns:
            Formatted context string for prompt
        """
        retrieval_result = self.retrieve(query, conversation_id)
        context = retrieval_result['context']
        
        if not context:
            return ""
        
        return f"""
Relevant information from previous conversations:

{context}

Use this information to inform your response if relevant to the user's query.
"""


class RAGPromptManager:
    """Manager for augmenting prompts with RAG context."""
    
    def __init__(self, rag_component):
        """
        Initialize the RAG prompt manager.
        
        Args:
            rag_component: RAG component instance
        """
        self.rag_component = rag_component
    
    def augment_messages(self, messages: List[Dict[str, Any]], conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Augment a list of messages with RAG context.
        
        Args:
            messages: List of message dictionaries (role, content)
            conversation_id: Optional conversation ID
            
        Returns:
            Augmented list of messages
        """
        # Extract the last user message as the query
        user_messages = [m for m in messages if m.get('role') == 'user']
        
        if not user_messages:
            # No user messages to use as query
            return messages
        
        last_user_message = user_messages[-1]
        query = last_user_message.get('content', '')
        
        if not query:
            # Empty query
            return messages
        
        # Retrieve context
        context = self.rag_component.retrieve_and_format_for_prompt(query, conversation_id)
        
        if not context:
            # No relevant context found
            return messages
        
        # Create a new list of messages
        augmented_messages = []
        
        # Find the system message if it exists
        system_message_index = None
        for i, message in enumerate(messages):
            if message.get('role') == 'system':
                system_message_index = i
                break
        
        if system_message_index is not None:
            # Augment the existing system message
            system_message = messages[system_message_index]
            augmented_system_content = f"{system_message.get('content', '')}\n\n{context}"
            
            augmented_system_message = {
                'role': 'system',
                'content': augmented_system_content
            }
            
            # Add all messages with the augmented system message
            for i, message in enumerate(messages):
                if i == system_message_index:
                    augmented_messages.append(augmented_system_message)
                else:
                    augmented_messages.append(message)
        else:
            # Add a new system message with the context
            augmented_messages.append({
                'role': 'system',
                'content': context
            })
            
            # Add all original messages
            augmented_messages.extend(messages)
        
        return augmented_messages
    
    def augment_prompt(self, prompt: str, conversation_id: Optional[str] = None) -> str:
        """
        Augment a text prompt with RAG context.
        
        Args:
            prompt: Text prompt
            conversation_id: Optional conversation ID
            
        Returns:
            Augmented prompt
        """
        # Use the prompt as the query
        context = self.rag_component.retrieve_and_format_for_prompt(prompt, conversation_id)
        
        if not context:
            # No relevant context found
            return prompt
        
        # Augment the prompt with context
        augmented_prompt = f"{context}\n\n{prompt}"
        
        return augmented_prompt


class RAGQueryAnalyzer:
    """Analyzer for determining if a query needs RAG context."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the RAG query analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Keywords that suggest the need for context
        self.context_keywords = self.config.get('context_keywords', [
            'remember', 'recall', 'previous', 'last time', 'you said',
            'we discussed', 'earlier', 'before', 'mentioned', 'told me'
        ])
        
        # Keywords that suggest a new topic
        self.new_topic_keywords = self.config.get('new_topic_keywords', [
            'new', 'different', 'another', 'change', 'topic', 'instead',
            'forget', 'ignore'
        ])
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query to determine if it needs RAG context.
        
        Args:
            query: User query
            
        Returns:
            Analysis results
        """
        query_lower = query.lower()
        
        # Check for context keywords
        context_matches = [kw for kw in self.context_keywords if kw in query_lower]
        
        # Check for new topic keywords
        new_topic_matches = [kw for kw in self.new_topic_keywords if kw in query_lower]
        
        # Determine if query needs context
        needs_context = len(context_matches) > 0 and len(new_topic_matches) == 0
        
        # Determine query type
        if needs_context:
            if any(kw in query_lower for kw in ['you said', 'told me']):
                query_type = 'recall_statement'
            elif any(kw in query_lower for kw in ['remember', 'recall']):
                query_type = 'recall_request'
            else:
                query_type = 'context_reference'
        else:
            query_type = 'new_topic' if new_topic_matches else 'general'
        
        return {
            'needs_context': needs_context,
            'query_type': query_type,
            'context_keywords': context_matches,
            'new_topic_keywords': new_topic_matches
        }


# Example usage
if __name__ == "__main__":
    import os
    from embeddings import EmbeddingGenerator
    from storage import StorageManager
    
    # Example paths
    storage_dir = "/home/ubuntu/llm_memory_system/rag_component/storage"
    
    # Create storage manager
    storage_manager = StorageManager(
        storage_type='sqlite',
        storage_path=os.path.join(storage_dir, 'memory.db')
    )
    
    # Initialize storage
    storage_manager.initialize()
    
    # Create embedding generator
    embedding_generator = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
    
    # Create RAG component
    rag_config = {
        'max_results': 5,
        'similarity_threshold': 0.6,
        'token_budget': 1000,
        'preferred_chunk_types': ['exchange', 'message', 'sliding_window']
    }
    
    rag_component = RAGComponent(
        storage_manager=storage_manager,
        embedding_generator=embedding_generator,
        config=rag_config
    )
    
    # Create RAG prompt manager
    prompt_manager = RAGPromptManager(rag_component)
    
    # Example query
    query = "What did we discuss about the project structure last time?"
    
    # Retrieve context
    retrieval_result = rag_component.retrieve(query)
    
    print(f"Query: {query}")
    print(f"Retrieved {len(retrieval_result['results'])} results")
    print("\nContext:")
    print(retrieval_result['context'])
    
    # Example messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        {"role": "user", "content": query}
    ]
    
    # Augment messages
    augmented_messages = prompt_manager.augment_messages(messages)
    
    print("\nAugmented System Message:")
    for message in augmented_messages:
        if message['role'] == 'system':
            print(message['content'])
            break
    
    # Close storage
    storage_manager.close()
