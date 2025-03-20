"""
Test script for the LLM memory system.

This script tests the complete LLM memory system, including data extraction,
RAG, Mem0, context management, and TypingMind integration.
"""

import os
import json
import logging
import time
import unittest
import requests
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemorySystemTest(unittest.TestCase):
    """Test case for the LLM memory system."""
    
    def setUp(self):
        """Set up the test case."""
        # Configuration
        self.config = {
            'data_extraction': {
                'chatgpt_export_path': '/home/ubuntu/llm_memory_system/data_extraction/sample_data/ChatGPT-Hello_Assistance.json',
                'typingmind_export_path': '/home/ubuntu/llm_memory_system/data_extraction/sample_data/20250319_164817_typingmind_export.json'
            },
            'rag': {
                'api_url': 'http://localhost:8000'
            },
            'mem0': {
                'api_url': 'http://localhost:8001'
            },
            'context_management': {
                'api_url': 'http://localhost:8002'
            }
        }
        
        # Test data
        self.test_query = "What did we discuss about machine learning last time? And remember I prefer Python over JavaScript."
        self.test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
            {"role": "user", "content": self.test_query}
        ]
    
    def test_data_extraction(self):
        """Test data extraction from ChatGPT and TypingMind exports."""
        logger.info("Testing data extraction...")
        
        # Check if sample data files exist
        self.assertTrue(os.path.exists(self.config['data_extraction']['chatgpt_export_path']),
                        "ChatGPT export file not found")
        self.assertTrue(os.path.exists(self.config['data_extraction']['typingmind_export_path']),
                        "TypingMind export file not found")
        
        # Load sample data
        with open(self.config['data_extraction']['chatgpt_export_path'], 'r', encoding='utf-8') as f:
            chatgpt_data = json.load(f)
        
        with open(self.config['data_extraction']['typingmind_export_path'], 'r', encoding='utf-8') as f:
            typingmind_data = json.load(f)
        
        # Verify data structure
        self.assertIsInstance(chatgpt_data, dict, "ChatGPT data should be a dictionary")
        self.assertIsInstance(typingmind_data, dict, "TypingMind data should be a dictionary")
        
        logger.info("Data extraction test passed")
    
    def test_rag_component(self):
        """Test the RAG component."""
        logger.info("Testing RAG component...")
        
        try:
            # Test RAG API health
            response = requests.get(f"{self.config['rag']['api_url']}/")
            self.assertEqual(response.status_code, 200, "RAG API health check failed")
            
            # Test RAG query
            response = requests.post(
                f"{self.config['rag']['api_url']}/query",
                json={
                    "query": self.test_query,
                    "max_results": 3
                }
            )
            
            self.assertEqual(response.status_code, 200, "RAG query failed")
            
            data = response.json()
            self.assertIn("context", data, "RAG response should contain context")
            self.assertIn("results", data, "RAG response should contain results")
            
            logger.info("RAG component test passed")
        except requests.exceptions.ConnectionError:
            logger.warning("RAG API not available, skipping test")
            self.skipTest("RAG API not available")
    
    def test_mem0_component(self):
        """Test the Mem0 component."""
        logger.info("Testing Mem0 component...")
        
        try:
            # Test Mem0 API health
            response = requests.get(f"{self.config['mem0']['api_url']}/")
            self.assertEqual(response.status_code, 200, "Mem0 API health check failed")
            
            # Test memory extraction
            response = requests.post(
                f"{self.config['mem0']['api_url']}/extract",
                json={
                    "message": {
                        "content": "I really like Python programming and dislike JavaScript.",
                        "conversation_id": "test_conversation"
                    }
                }
            )
            
            self.assertEqual(response.status_code, 200, "Mem0 memory extraction failed")
            
            data = response.json()
            self.assertIn("memories", data, "Mem0 response should contain memories")
            
            # Test memory retrieval
            response = requests.post(
                f"{self.config['mem0']['api_url']}/relevant-memories",
                json={
                    "query": "What programming languages do I like?",
                    "limit": 5
                }
            )
            
            self.assertEqual(response.status_code, 200, "Mem0 memory retrieval failed")
            
            data = response.json()
            self.assertIn("formatted_memories", data, "Mem0 response should contain formatted_memories")
            self.assertIn("memories", data, "Mem0 response should contain memories")
            
            logger.info("Mem0 component test passed")
        except requests.exceptions.ConnectionError:
            logger.warning("Mem0 API not available, skipping test")
            self.skipTest("Mem0 API not available")
    
    def test_context_management(self):
        """Test the context management system."""
        logger.info("Testing context management system...")
        
        try:
            # Test context management API health
            response = requests.get(f"{self.config['context_management']['api_url']}/")
            self.assertEqual(response.status_code, 200, "Context management API health check failed")
            
            # Test query analysis
            response = requests.post(
                f"{self.config['context_management']['api_url']}/analyze-query",
                json={
                    "query": self.test_query
                }
            )
            
            self.assertEqual(response.status_code, 200, "Context management query analysis failed")
            
            data = response.json()
            self.assertIn("needs_factual", data, "Query analysis should contain needs_factual")
            self.assertIn("needs_personal", data, "Query analysis should contain needs_personal")
            
            # Test context retrieval
            response = requests.post(
                f"{self.config['context_management']['api_url']}/get-context",
                json={
                    "query": self.test_query
                }
            )
            
            self.assertEqual(response.status_code, 200, "Context management context retrieval failed")
            
            data = response.json()
            self.assertIn("rag_context", data, "Context retrieval should contain rag_context")
            self.assertIn("mem0_context", data, "Context retrieval should contain mem0_context")
            self.assertIn("combined_context", data, "Context retrieval should contain combined_context")
            
            # Test message augmentation
            response = requests.post(
                f"{self.config['context_management']['api_url']}/augment-messages",
                json={
                    "messages": self.test_messages
                }
            )
            
            self.assertEqual(response.status_code, 200, "Context management message augmentation failed")
            
            data = response.json()
            self.assertIn("messages", data, "Message augmentation should contain messages")
            self.assertIn("context_added", data, "Message augmentation should contain context_added")
            
            logger.info("Context management system test passed")
        except requests.exceptions.ConnectionError:
            logger.warning("Context management API not available, skipping test")
            self.skipTest("Context management API not available")
    
    def test_end_to_end(self):
        """Test the complete system end-to-end."""
        logger.info("Testing complete system end-to-end...")
        
        try:
            # Test context management API health
            response = requests.get(f"{self.config['context_management']['api_url']}/")
            
            # Test RAG API health
            rag_response = requests.get(f"{self.config['rag']['api_url']}/")
            
            # Test Mem0 API health
            mem0_response = requests.get(f"{self.config['mem0']['api_url']}/")
            
            if response.status_code != 200 or rag_response.status_code != 200 or mem0_response.status_code != 200:
                logger.warning("One or more APIs not available, skipping end-to-end test")
                self.skipTest("One or more APIs not available")
            
            # Test end-to-end flow
            # 1. Extract memories from a message
            mem0_extract_response = requests.post(
                f"{self.config['mem0']['api_url']}/extract",
                json={
                    "message": {
                        "content": "I really like Python programming and dislike JavaScript.",
                        "conversation_id": "test_conversation"
                    }
                }
            )
            
            self.assertEqual(mem0_extract_response.status_code, 200, "Memory extraction failed")
            
            # 2. Add extracted memories to Mem0
            memories = mem0_extract_response.json().get("memories", [])
            
            for memory in memories:
                memory_response = requests.post(
                    f"{self.config['mem0']['api_url']}/memory",
                    json={
                        "content": memory["content"],
                        "memory_type": memory["memory_type"],
                        "source_conversation_id": "test_conversation",
                        "importance": memory.get("importance", 0.7)
                    }
                )
                
                self.assertEqual(memory_response.status_code, 200, "Memory addition failed")
            
            # 3. Augment messages with context
            augment_response = requests.post(
                f"{self.config['context_management']['api_url']}/augment-messages",
                json={
                    "messages": self.test_messages,
                    "conversation_id": "test_conversation"
                }
            )
            
            self.assertEqual(augment_response.status_code, 200, "Message augmentation failed")
            
            augmented_messages = augment_response.json().get("messages", [])
            
            # 4. Verify that context was added
            system_messages = [m for m in augmented_messages if m["role"] == "system"]
            
            self.assertTrue(len(system_messages) > 0, "No system message found in augmented messages")
            
            if len(system_messages) > 0:
                system_content = system_messages[0]["content"]
                self.assertTrue(len(system_content) > 0, "System message content is empty")
                
                # Check if the system message contains memory-related content
                memory_keywords = ["memory", "context", "remember", "preference", "Python", "JavaScript"]
                found_keywords = [kw for kw in memory_keywords if kw.lower() in system_content.lower()]
                
                self.assertTrue(len(found_keywords) > 0, 
                                f"System message does not contain memory-related content: {system_content}")
            
            logger.info("End-to-end test passed")
        except requests.exceptions.ConnectionError:
            logger.warning("One or more APIs not available, skipping end-to-end test")
            self.skipTest("One or more APIs not available")


class MemorySystemOptimization:
    """Optimization for the LLM memory system."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the optimization.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def optimize_token_budget(self):
        """Optimize the token budget for context management."""
        logger.info("Optimizing token budget...")
        
        try:
            # Test different token budgets
            token_budgets = [500, 1000, 1500, 2000]
            results = {}
            
            test_query = "What did we discuss about machine learning last time? And remember I prefer Python over JavaScript."
            
            for budget in token_budgets:
                # Set token budget
                response = requests.post(
                    f"{self.config['context_management']['api_url']}/get-context",
                    json={
                        "query": test_query,
                        "token_budget": budget
                    }
                )
                
                if response.status_code != 200:
                    logger.warning(f"Failed to get context with token budget {budget}")
                    continue
                
                data = response.json()
                
                # Calculate metrics
                context_length = len(data.get("combined_context", ""))
                token_count = data.get("token_count", 0)
                
                results[budget] = {
                    "context_length": context_length,
                    "token_count": token_count,
                    "efficiency": context_length / budget if budget > 0 else 0
                }
            
            # Find optimal token budget
            if results:
                optimal_budget = max(results.items(), key=lambda x: x[1]["efficiency"])[0]
                logger.info(f"Optimal token budget: {optimal_budget}")
                logger.info(f"Optimization results: {results}")
                
                return optimal_budget, results
            
            logger.warning("No results for token budget optimization")
            return None, {}
        except requests.exceptions.ConnectionError:
            logger.warning("Context management API not available, skipping optimization")
            return None, {}
    
    def optimize_rag_mem0_weights(self):
        """Optimize the weights for RAG and Mem0 components."""
        logger.info("Optimizing RAG and Mem0 weights...")
        
        try:
            # Test different weight combinations
            rag_weights = [0.3, 0.5, 0.7, 0.9]
            results = {}
            
            test_query = "What did we discuss about machine learning last time? And remember I prefer Python over JavaScript."
            
            for rag_weight in rag_weights:
                mem0_weight = 1.0 - rag_weight
                
                # Set weights
                response = requests.post(
                    f"{self.config['context_management']['api_url']}/get-context",
                    json={
                        "query": test_query,
                        "rag_weight": rag_weight,
                        "mem0_weight": mem0_weight
                    }
                )
                
                if response.status_code != 200:
                    logger.warning(f"Failed to get context with RAG weight {rag_weight} and Mem0 weight {mem0_weight}")
                    continue
                
                data = response.json()
                
                # Calculate metrics
                rag_context_length = len(data.get("rag_context", ""))
                mem0_context_length = len(data.get("mem0_context", ""))
                combined_context_length = len(data.get("combined_context", ""))
                
                results[(rag_weight, mem0_weight)] = {
                    "rag_context_length": rag_context_length,
                    "mem0_context_length": mem0_context_length,
                    "combined_context_length": combined_context_length,
                    "balance": min(rag_context_length, mem0_context_length) / max(rag_context_length, mem0_context_length) if max(rag_context_length, mem0_context_length) > 0 else 0
                }
            
            # Find optimal weights
            if results:
                optimal_weights = max(results.items(), key=lambda x: x[1]["balance"])[0]
                logger.info(f"Optimal RAG weight: {optimal_weights[0]}, Mem0 weight: {optimal_weights[1]}")
                logger.info(f"Optimization results: {results}")
                
                return optimal_weights, results
            
            logger.warning("No results for weight optimization")
            return None, {}
        except requests.exceptions.ConnectionError:
            logger.warning("Context management API not available, skipping optimization")
            return None, {}
    
    def optimize_chunking_strategy(self):
        """Optimize the chunking strategy for data extraction."""
        logger.info("Optimizing chunking strategy...")
        
        # This would require modifying the data extraction pipeline and reindexing data,
        # which is beyond the scope of this test script. In a real implementation, we would
        # test different chunking strategies and measure retrieval performance.
        
        logger.info("Chunking strategy optimization requires modifying the data extraction pipeline")
        logger.info("Recommended chunking strategies based on research:")
        logger.info("1. Message-level chunking for short conversations")
        logger.info("2. Exchange-level chunking (user-assistant pairs) for medium-length conversations")
        logger.info("3. Semantic chunking for long conversations")
        logger.info("4. Hybrid approach: semantic chunking with overlap for best results")
        
        return "hybrid_semantic_chunking", {}
    
    def run_all_optimizations(self):
        """Run all optimizations."""
        logger.info("Running all optimizations...")
        
        results = {}
        
        # Optimize token budget
        token_budget, token_budget_results = self.optimize_token_budget()
        results["token_budget"] = {
            "optimal_value": token_budget,
            "results": token_budget_results
        }
        
        # Optimize RAG and Mem0 weights
        weights, weight_results = self.optimize_rag_mem0_weights()
        results["weights"] = {
            "optimal_value": weights,
            "results": weight_results
        }
        
        # Optimize chunking strategy
        chunking_strategy, chunking_results = self.optimize_chunking_strategy()
        results["chunking_strategy"] = {
            "optimal_value": chunking_strategy,
            "results": chunking_results
        }
        
        logger.info("All optimizations completed")
        logger.info(f"Optimization results: {results}")
        
        return results


def run_tests():
    """Run the tests."""
    logger.info("Running memory system tests...")
    
    # Run unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Run optimization
    config = {
        'context_management': {
            'api_url': 'http://localhost:8002'
        }
    }
    
    optimizer = MemorySystemOptimization(config)
    optimization_results = optimizer.run_all_optimizations()
    
    logger.info("Tests and optimization completed")
    
    return optimization_results


if __name__ == "__main__":
    run_tests()
