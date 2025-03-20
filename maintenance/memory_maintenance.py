"""
Automated memory updates and maintenance for LLM memory system.

This script implements automated memory updates and maintenance processes
for the LLM memory system, including memory consolidation, summarization,
and optimization.
"""

import os
import json
import logging
import time
import schedule
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryMaintenance:
    """Memory maintenance system for LLM memory system."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the memory maintenance system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.rag_api_url = config.get('rag_api_url', 'http://localhost:8000')
        self.mem0_api_url = config.get('mem0_api_url', 'http://localhost:8001')
        self.context_api_url = config.get('context_api_url', 'http://localhost:8002')
        
        # Maintenance settings
        self.maintenance_interval = config.get('maintenance_interval', 24)  # hours
        self.memory_age_threshold = config.get('memory_age_threshold', 30)  # days
        self.consolidation_threshold = config.get('consolidation_threshold', 10)  # similar memories
        self.importance_threshold = config.get('importance_threshold', 0.3)  # minimum importance to keep
    
    def consolidate_similar_memories(self):
        """Consolidate similar memories in Mem0."""
        logger.info("Consolidating similar memories...")
        
        try:
            # Get all memories
            response = requests.get(f"{self.mem0_api_url}/memories/all")
            
            if response.status_code != 200:
                logger.error(f"Failed to get memories: {response.status_code}")
                return
            
            memories = response.json().get('memories', [])
            
            if not memories:
                logger.info("No memories to consolidate")
                return
            
            # Group memories by type
            memories_by_type = {}
            for memory in memories:
                memory_type = memory.get('memory_type')
                if memory_type not in memories_by_type:
                    memories_by_type[memory_type] = []
                memories_by_type[memory_type].append(memory)
            
            # Process each type
            for memory_type, type_memories in memories_by_type.items():
                # Skip if not enough memories
                if len(type_memories) < self.consolidation_threshold:
                    continue
                
                # Find similar memories
                similar_groups = self._find_similar_memories(type_memories)
                
                # Consolidate each group
                for group in similar_groups:
                    if len(group) >= 2:
                        self._consolidate_memory_group(group)
            
            logger.info("Memory consolidation completed")
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
    
    def _find_similar_memories(self, memories: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Find similar memories.
        
        Args:
            memories: List of memory dictionaries
            
        Returns:
            List of groups of similar memories
        """
        # This is a simplified implementation
        # In a real system, we would use embeddings and clustering
        
        # Group by content similarity (simple substring matching)
        groups = []
        processed = set()
        
        for i, memory in enumerate(memories):
            if i in processed:
                continue
            
            content = memory.get('content', '').lower()
            group = [memory]
            processed.add(i)
            
            for j, other_memory in enumerate(memories):
                if j in processed or i == j:
                    continue
                
                other_content = other_memory.get('content', '').lower()
                
                # Simple similarity check
                if content in other_content or other_content in content:
                    group.append(other_memory)
                    processed.add(j)
            
            if len(group) >= 2:
                groups.append(group)
        
        return groups
    
    def _consolidate_memory_group(self, memories: List[Dict[str, Any]]):
        """
        Consolidate a group of similar memories.
        
        Args:
            memories: List of similar memory dictionaries
        """
        # Sort by importance (descending)
        memories.sort(key=lambda m: m.get('importance', 0), reverse=True)
        
        # Keep the most important memory
        primary_memory = memories[0]
        
        # Update its importance
        new_importance = min(1.0, primary_memory.get('importance', 0.5) + 0.1)
        
        # Update the primary memory
        response = requests.put(
            f"{self.mem0_api_url}/memory/{primary_memory.get('memory_id')}",
            json={
                'importance': new_importance,
                'metadata': {
                    **primary_memory.get('metadata', {}),
                    'consolidated_from': [m.get('memory_id') for m in memories[1:]]
                }
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to update primary memory: {response.status_code}")
            return
        
        # Delete the other memories
        for memory in memories[1:]:
            requests.delete(f"{self.mem0_api_url}/memory/{memory.get('memory_id')}")
    
    def prune_old_memories(self):
        """Prune old, low-importance memories."""
        logger.info("Pruning old memories...")
        
        try:
            # Get all memories
            response = requests.get(f"{self.mem0_api_url}/memories/all")
            
            if response.status_code != 200:
                logger.error(f"Failed to get memories: {response.status_code}")
                return
            
            memories = response.json().get('memories', [])
            
            if not memories:
                logger.info("No memories to prune")
                return
            
            # Calculate age threshold
            age_threshold = datetime.now() - timedelta(days=self.memory_age_threshold)
            age_threshold_str = age_threshold.isoformat()
            
            # Find old, low-importance memories
            for memory in memories:
                created_at = memory.get('created_at')
                importance = memory.get('importance', 0.5)
                
                if created_at and created_at < age_threshold_str and importance < self.importance_threshold:
                    # Delete the memory
                    requests.delete(f"{self.mem0_api_url}/memory/{memory.get('memory_id')}")
            
            logger.info("Memory pruning completed")
        except Exception as e:
            logger.error(f"Error pruning memories: {e}")
    
    def summarize_conversations(self):
        """Summarize conversations and store as memories."""
        logger.info("Summarizing conversations...")
        
        try:
            # Get recent conversations
            response = requests.get(f"{self.rag_api_url}/conversations/recent")
            
            if response.status_code != 200:
                logger.error(f"Failed to get conversations: {response.status_code}")
                return
            
            conversations = response.json().get('conversations', [])
            
            if not conversations:
                logger.info("No conversations to summarize")
                return
            
            # Process each conversation
            for conversation in conversations:
                # Check if already summarized
                if conversation.get('metadata', {}).get('summarized'):
                    continue
                
                # Generate summary
                summary = self._generate_conversation_summary(conversation)
                
                if not summary:
                    continue
                
                # Store summary as memory
                response = requests.post(
                    f"{self.mem0_api_url}/memory",
                    json={
                        'content': summary,
                        'memory_type': 'conversation_summary',
                        'source_conversation_id': conversation.get('id'),
                        'importance': 0.7,
                        'metadata': {
                            'conversation_title': conversation.get('title'),
                            'conversation_date': conversation.get('created_at')
                        }
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to store summary: {response.status_code}")
                    continue
                
                # Mark conversation as summarized
                requests.put(
                    f"{self.rag_api_url}/conversation/{conversation.get('id')}",
                    json={
                        'metadata': {
                            **conversation.get('metadata', {}),
                            'summarized': True
                        }
                    }
                )
            
            logger.info("Conversation summarization completed")
        except Exception as e:
            logger.error(f"Error summarizing conversations: {e}")
    
    def _generate_conversation_summary(self, conversation: Dict[str, Any]) -> Optional[str]:
        """
        Generate a summary for a conversation.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            Summary string or None if failed
        """
        # This is a simplified implementation
        # In a real system, we would use an LLM to generate the summary
        
        messages = conversation.get('messages', [])
        
        if not messages:
            return None
        
        # Extract user messages
        user_messages = [m.get('content', '') for m in messages if m.get('role') == 'user']
        
        if not user_messages:
            return None
        
        # Simple summary: first and last user messages
        if len(user_messages) == 1:
            return f"Conversation about: {user_messages[0][:100]}"
        else:
            return f"Conversation starting with '{user_messages[0][:50]}' and ending with '{user_messages[-1][:50]}'"
    
    def optimize_vector_indexes(self):
        """Optimize vector indexes in the RAG component."""
        logger.info("Optimizing vector indexes...")
        
        try:
            # Trigger vector index optimization
            response = requests.post(f"{self.rag_api_url}/optimize-indexes")
            
            if response.status_code != 200:
                logger.error(f"Failed to optimize indexes: {response.status_code}")
                return
            
            logger.info("Vector index optimization completed")
        except Exception as e:
            logger.error(f"Error optimizing vector indexes: {e}")
    
    def run_maintenance(self):
        """Run all maintenance tasks."""
        logger.info("Running maintenance tasks...")
        
        # Consolidate similar memories
        self.consolidate_similar_memories()
        
        # Prune old memories
        self.prune_old_memories()
        
        # Summarize conversations
        self.summarize_conversations()
        
        # Optimize vector indexes
        self.optimize_vector_indexes()
        
        logger.info("Maintenance tasks completed")
    
    def schedule_maintenance(self):
        """Schedule maintenance tasks."""
        logger.info(f"Scheduling maintenance every {self.maintenance_interval} hours")
        
        # Schedule maintenance
        schedule.every(self.maintenance_interval).hours.do(self.run_maintenance)
        
        # Run immediately
        self.run_maintenance()
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)


def main():
    """Main function."""
    # Configuration
    config = {
        'rag_api_url': os.environ.get('RAG_API_URL', 'http://localhost:8000'),
        'mem0_api_url': os.environ.get('MEM0_API_URL', 'http://localhost:8001'),
        'context_api_url': os.environ.get('CONTEXT_API_URL', 'http://localhost:8002'),
        'maintenance_interval': int(os.environ.get('MAINTENANCE_INTERVAL', '24')),
        'memory_age_threshold': int(os.environ.get('MEMORY_AGE_THRESHOLD', '30')),
        'consolidation_threshold': int(os.environ.get('CONSOLIDATION_THRESHOLD', '10')),
        'importance_threshold': float(os.environ.get('IMPORTANCE_THRESHOLD', '0.3'))
    }
    
    # Create maintenance system
    maintenance = MemoryMaintenance(config)
    
    # Schedule maintenance
    maintenance.schedule_maintenance()


if __name__ == "__main__":
    main()
