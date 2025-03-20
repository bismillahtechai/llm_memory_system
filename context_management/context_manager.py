"""
Dynamic context management system for LLM memory system.

This module provides the dynamic context management system that integrates
RAG and Mem0 components to optimize context for LLM prompts.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContextManager:
    """Dynamic context manager for integrating RAG and Mem0 components."""
    
    def __init__(self, rag_client, mem0_client, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the context manager.
        
        Args:
            rag_client: Client for the RAG component
            mem0_client: Client for the Mem0 component
            config: Configuration dictionary
        """
        self.rag_client = rag_client
        self.mem0_client = mem0_client
        self.config = config or {}
        
        # Default configuration values
        self.token_budget = self.config.get('token_budget', 1000)
        self.rag_weight = self.config.get('rag_weight', 0.7)
        self.mem0_weight = self.config.get('mem0_weight', 0.3)
        self.max_rag_results = self.config.get('max_rag_results', 5)
        self.max_mem0_results = self.config.get('max_mem0_results', 5)
        self.memory_types = self.config.get('memory_types', [
            'preference', 'insight', 'personal_info'
        ])
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query to determine context needs.
        
        Args:
            query: User query
            
        Returns:
            Analysis results
        """
        # Keywords that suggest the need for factual context (RAG)
        factual_keywords = [
            'what', 'how', 'when', 'where', 'why', 'who', 'which',
            'explain', 'describe', 'tell me about', 'information',
            'remember', 'recall', 'previous', 'last time', 'you said',
            'we discussed', 'earlier', 'before', 'mentioned', 'told me'
        ]
        
        # Keywords that suggest the need for personal context (Mem0)
        personal_keywords = [
            'i like', 'i prefer', 'i enjoy', 'i love', 'i hate', 'i dislike',
            'my favorite', 'my preference', 'i want', 'i need',
            'remember me', 'remember that i', 'about me', 'my', 'mine'
        ]
        
        # Check for keywords
        query_lower = query.lower()
        
        factual_matches = [kw for kw in factual_keywords if kw in query_lower]
        personal_matches = [kw for kw in personal_keywords if kw in query_lower]
        
        # Determine context needs
        needs_factual = len(factual_matches) > 0
        needs_personal = len(personal_matches) > 0
        
        # If no specific needs detected, default to both
        if not needs_factual and not needs_personal:
            needs_factual = True
            needs_personal = True
        
        return {
            'needs_factual': needs_factual,
            'needs_personal': needs_personal,
            'factual_keywords': factual_matches,
            'personal_keywords': personal_matches,
            'query': query
        }
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def get_context(self, query: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get context for a query.
        
        Args:
            query: User query
            conversation_id: Optional conversation ID
            
        Returns:
            Context information
        """
        # Analyze query
        analysis = self.analyze_query(query)
        
        # Initialize context
        context = {
            'rag_context': '',
            'mem0_context': '',
            'combined_context': '',
            'token_count': 0,
            'analysis': analysis
        }
        
        # Get RAG context if needed
        if analysis['needs_factual']:
            try:
                rag_response = self.rag_client.query(
                    query=query,
                    conversation_id=conversation_id,
                    max_results=self.max_rag_results
                )
                
                context['rag_context'] = rag_response.get('context', '')
                context['rag_results'] = rag_response.get('results', [])
            except Exception as e:
                logger.error(f"Error getting RAG context: {e}")
                context['rag_context'] = ''
                context['rag_results'] = []
        
        # Get Mem0 context if needed
        if analysis['needs_personal']:
            try:
                mem0_response = self.mem0_client.get_relevant_memories(
                    query=query,
                    memory_types=self.memory_types,
                    limit=self.max_mem0_results
                )
                
                context['mem0_context'] = mem0_response.get('formatted_memories', '')
                context['mem0_results'] = mem0_response.get('memories', [])
            except Exception as e:
                logger.error(f"Error getting Mem0 context: {e}")
                context['mem0_context'] = ''
                context['mem0_results'] = []
        
        # Combine contexts
        context['combined_context'] = self.combine_contexts(
            context['rag_context'],
            context['mem0_context']
        )
        
        # Estimate token count
        context['token_count'] = self.estimate_tokens(context['combined_context'])
        
        return context
    
    def combine_contexts(self, rag_context: str, mem0_context: str) -> str:
        """
        Combine RAG and Mem0 contexts.
        
        Args:
            rag_context: Context from RAG component
            mem0_context: Context from Mem0 component
            
        Returns:
            Combined context
        """
        if not rag_context and not mem0_context:
            return ""
        
        if not rag_context:
            return mem0_context
        
        if not mem0_context:
            return rag_context
        
        # Combine contexts
        combined_context = f"{rag_context}\n\n{mem0_context}"
        
        return combined_context
    
    def optimize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize context to fit within token budget.
        
        Args:
            context: Context information
            
        Returns:
            Optimized context
        """
        # Check if context is within budget
        if context['token_count'] <= self.token_budget:
            return context
        
        # Calculate how much to reduce
        excess_tokens = context['token_count'] - self.token_budget
        
        # Calculate reduction for each component based on weights
        rag_tokens = self.estimate_tokens(context['rag_context'])
        mem0_tokens = self.estimate_tokens(context['mem0_context'])
        
        total_tokens = rag_tokens + mem0_tokens
        
        if total_tokens == 0:
            return context
        
        rag_reduction = int(excess_tokens * (rag_tokens / total_tokens))
        mem0_reduction = excess_tokens - rag_reduction
        
        # Reduce RAG context
        if rag_reduction > 0 and rag_tokens > 0:
            # Simple approach: truncate
            target_rag_tokens = rag_tokens - rag_reduction
            if target_rag_tokens <= 0:
                context['rag_context'] = ""
            else:
                # Approximate character count
                target_chars = target_rag_tokens * 4
                context['rag_context'] = context['rag_context'][:target_chars]
        
        # Reduce Mem0 context
        if mem0_reduction > 0 and mem0_tokens > 0:
            # Simple approach: truncate
            target_mem0_tokens = mem0_tokens - mem0_reduction
            if target_mem0_tokens <= 0:
                context['mem0_context'] = ""
            else:
                # Approximate character count
                target_chars = target_mem0_tokens * 4
                context['mem0_context'] = context['mem0_context'][:target_chars]
        
        # Recombine contexts
        context['combined_context'] = self.combine_contexts(
            context['rag_context'],
            context['mem0_context']
        )
        
        # Update token count
        context['token_count'] = self.estimate_tokens(context['combined_context'])
        
        return context
    
    def augment_messages(self, messages: List[Dict[str, Any]], conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Augment a list of messages with context.
        
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
        
        # Get context
        context = self.get_context(query, conversation_id)
        
        # Optimize context
        optimized_context = self.optimize_context(context)
        
        combined_context = optimized_context['combined_context']
        
        if not combined_context:
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
            augmented_system_content = f"{system_message.get('content', '')}\n\n{combined_context}"
            
            # Add all messages with the augmented system message
            for i, message in enumerate(messages):
                if i == system_message_index:
                    augmented_messages.append({
                        'role': 'system',
                        'content': augmented_system_content
                    })
                else:
                    augmented_messages.append(message)
        else:
            # Add a new system message with the context
            augmented_messages.append({
                'role': 'system',
                'content': combined_context
            })
            
            # Add all original messages
            augmented_messages.extend(messages)
        
        return augmented_messages
    
    def augment_prompt(self, prompt: str, conversation_id: Optional[str] = None) -> str:
        """
        Augment a text prompt with context.
        
        Args:
            prompt: Text prompt
            conversation_id: Optional conversation ID
            
        Returns:
            Augmented prompt
        """
        # Use the prompt as the query
        context = self.get_context(prompt, conversation_id)
        
        # Optimize context
        optimized_context = self.optimize_context(context)
        
        combined_context = optimized_context['combined_context']
        
        if not combined_context:
            # No relevant context found
            return prompt
        
        # Augment the prompt with context
        augmented_prompt = f"{combined_context}\n\n{prompt}"
        
        return augmented_prompt


class RAGClient:
    """Client for the RAG component."""
    
    def __init__(self, api_url: str):
        """
        Initialize the RAG client.
        
        Args:
            api_url: URL of the RAG API
        """
        self.api_url = api_url
    
    def query(self, query: str, conversation_id: Optional[str] = None, max_results: int = 5) -> Dict[str, Any]:
        """
        Query the RAG component.
        
        Args:
            query: User query
            conversation_id: Optional conversation ID
            max_results: Maximum number of results
            
        Returns:
            Query response
        """
        import requests
        
        try:
            response = requests.post(
                f"{self.api_url}/query",
                json={
                    "query": query,
                    "conversation_id": conversation_id,
                    "max_results": max_results
                }
            )
            
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"Error querying RAG component: {e}")
            return {
                "context": "",
                "results": [],
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
    
    def augment_messages(self, messages: List[Dict[str, Any]], conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Augment a list of messages with RAG context.
        
        Args:
            messages: List of message dictionaries (role, content)
            conversation_id: Optional conversation ID
            
        Returns:
            Augmented list of messages
        """
        import requests
        
        try:
            response = requests.post(
                f"{self.api_url}/augment-messages",
                json={
                    "messages": messages,
                    "conversation_id": conversation_id
                }
            )
            
            response.raise_for_status()
            
            return response.json().get("messages", messages)
        except Exception as e:
            logger.error(f"Error augmenting messages with RAG context: {e}")
            return messages


class Mem0Client:
    """Client for the Mem0 component."""
    
    def __init__(self, api_url: str):
        """
        Initialize the Mem0 client.
        
        Args:
            api_url: URL of the Mem0 API
        """
        self.api_url = api_url
    
    def get_relevant_memories(self, query: str, memory_types: Optional[List[str]] = None, limit: int = 5) -> Dict[str, Any]:
        """
        Get relevant memories for a query.
        
        Args:
            query: User query
            memory_types: List of memory types to include
            limit: Maximum number of memories
            
        Returns:
            Relevant memories
        """
        import requests
        
        try:
            response = requests.post(
                f"{self.api_url}/relevant-memories",
                json={
                    "query": query,
                    "memory_types": memory_types,
                    "limit": limit
                }
            )
            
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"Error getting relevant memories: {e}")
            return {
                "formatted_memories": "",
                "memories": []
            }
    
    def extract_memories(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract memories from a message.
        
        Args:
            message: Message dictionary
            
        Returns:
            List of extracted memories
        """
        import requests
        
        try:
            response = requests.post(
                f"{self.api_url}/extract",
                json={
                    "message": message
                }
            )
            
            response.raise_for_status()
            
            return response.json().get("memories", [])
        except Exception as e:
            logger.error(f"Error extracting memories: {e}")
            return []
    
    def add_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a memory.
        
        Args:
            memory: Memory dictionary
            
        Returns:
            Response with memory ID
        """
        import requests
        
        try:
            response = requests.post(
                f"{self.api_url}/memory",
                json=memory
            )
            
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return {
                "success": False,
                "memory_id": ""
            }


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'token_budget': 1000,
        'rag_weight': 0.7,
        'mem0_weight': 0.3,
        'max_rag_results': 5,
        'max_mem0_results': 5,
        'memory_types': ['preference', 'insight', 'personal_info']
    }
    
    # Create clients
    rag_client = RAGClient(api_url="http://localhost:8000")
    mem0_client = Mem0Client(api_url="http://localhost:8001")
    
    # Create context manager
    context_manager = ContextManager(
        rag_client=rag_client,
        mem0_client=mem0_client,
        config=config
    )
    
    # Example query
    query = "What did we discuss about the project structure last time? And remember I prefer Python over JavaScript."
    
    # Get context
    context = context_manager.get_context(query)
    
    print(f"Query: {query}")
    print(f"Needs factual context: {context['analysis']['needs_factual']}")
    print(f"Needs personal context: {context['analysis']['needs_personal']}")
    print(f"Token count: {context['token_count']}")
    
    print("\nRAG Context:")
    print(context['rag_context'])
    
    print("\nMem0 Context:")
    print(context['mem0_context'])
    
    print("\nCombined Context:")
    print(context['combined_context'])
    
    # Example messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        {"role": "user", "content": query}
    ]
    
    # Augment messages
    augmented_messages = context_manager.augment_messages(messages)
    
    print("\nAugmented System Message:")
    for message in augmented_messages:
        if message['role'] == 'system':
            print(message['content'])
            break
