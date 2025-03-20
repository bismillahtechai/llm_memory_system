"""
Main pipeline for data extraction and consolidation.

This module provides a complete pipeline for extracting, processing, chunking,
embedding, and storing conversation data from ChatGPT and TypingMind exports.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from parsers import ConversationProcessor
from chunker import ChunkingPipeline
from embeddings import EmbeddingPipeline
from storage import StorageManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataExtractionPipeline:
    """Complete pipeline for data extraction and consolidation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data extraction pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Create directories
        os.makedirs(config['processed_dir'], exist_ok=True)
        os.makedirs(config['chunks_dir'], exist_ok=True)
        os.makedirs(config['embedded_dir'], exist_ok=True)
        os.makedirs(config['storage_dir'], exist_ok=True)
        
        # Initialize components
        self.conversation_processor = ConversationProcessor()
        
        self.chunking_pipeline = ChunkingPipeline(
            strategies=config['chunking_strategies'],
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap']
        )
        
        self.embedding_pipeline = EmbeddingPipeline(
            model_name=config['embedding_model'],
            batch_size=config['batch_size']
        )
        
        self.storage_manager = StorageManager(
            storage_type=config['storage_type'],
            storage_path=os.path.join(config['storage_dir'], config['storage_name'])
        )
        
        # Initialize storage
        self.storage_manager.initialize()
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single conversation file.
        
        Args:
            file_path: Path to the conversation file
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing file: {file_path}")
        
        results = {
            'file_path': file_path,
            'conversation_id': None,
            'chunks_count': {},
            'success': False,
            'error': None
        }
        
        try:
            # Process conversation
            conversation = self.conversation_processor.process_file(file_path)
            
            # Save processed conversation
            processed_path = self.conversation_processor.save_processed_conversation(
                conversation, self.config['processed_dir'])
            
            logger.info(f"Saved processed conversation: {processed_path}")
            
            # Store conversation in storage backend
            conversation_id = self.storage_manager.store_conversation(conversation)
            results['conversation_id'] = conversation_id
            
            logger.info(f"Stored conversation in backend: {conversation_id}")
            
            # Apply chunking strategies
            chunks_by_strategy = self.chunking_pipeline.process_conversation(conversation)
            
            # Save chunks to files if configured
            if self.config.get('save_chunks_to_files', False):
                chunk_paths = self.chunking_pipeline.save_chunks(
                    chunks_by_strategy, self.config['chunks_dir'])
                
                for strategy, paths in chunk_paths.items():
                    logger.info(f"Saved {len(paths)} chunks using {strategy} strategy")
            
            # Process chunks for each strategy
            for strategy, chunks in chunks_by_strategy.items():
                # Generate embeddings
                embedded_chunks = self.embedding_pipeline.process_chunks_batch(chunks)
                
                # Save embedded chunks to files if configured
                if self.config.get('save_embeddings_to_files', False):
                    strategy_dir = os.path.join(self.config['embedded_dir'], strategy)
                    os.makedirs(strategy_dir, exist_ok=True)
                    
                    for chunk in embedded_chunks:
                        self.embedding_pipeline.embedding_generator.save_embedded_chunk(
                            chunk, strategy_dir)
                
                # Store chunks in storage backend
                chunk_ids = self.storage_manager.store_chunks_batch(embedded_chunks)
                
                logger.info(f"Stored {len(chunk_ids)} embedded chunks using {strategy} strategy")
                results['chunks_count'][strategy] = len(chunk_ids)
            
            results['success'] = True
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
            results['error'] = str(e)
        
        return results
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all conversation files in a directory.
        
        Args:
            directory_path: Path to directory containing conversation files
            
        Returns:
            List of processing results
        """
        logger.info(f"Processing directory: {directory_path}")
        
        results = []
        
        # Get all JSON files
        json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        
        logger.info(f"Found {len(json_files)} JSON files")
        
        # Process each file
        for filename in json_files:
            file_path = os.path.join(directory_path, filename)
            result = self.process_file(file_path)
            results.append(result)
        
        return results
    
    def close(self):
        """Close the pipeline components."""
        self.storage_manager.close()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Process conversation exports')
    parser.add_argument('--input', required=True, help='Input file or directory')
    parser.add_argument('--config', default='config.json', help='Configuration file')
    parser.add_argument('--output', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        # Use default configuration
        config = {
            'processed_dir': 'data/processed',
            'chunks_dir': 'data/chunks',
            'embedded_dir': 'data/embedded',
            'storage_dir': 'data/storage',
            'storage_name': 'memory.db',
            'storage_type': 'sqlite',
            'chunking_strategies': ['message', 'exchange', 'sliding_window'],
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'embedding_model': 'all-MiniLM-L6-v2',
            'batch_size': 32,
            'save_chunks_to_files': True,
            'save_embeddings_to_files': True
        }
    
    # Override output directory if specified
    if args.output:
        config['processed_dir'] = os.path.join(args.output, 'processed')
        config['chunks_dir'] = os.path.join(args.output, 'chunks')
        config['embedded_dir'] = os.path.join(args.output, 'embedded')
        config['storage_dir'] = os.path.join(args.output, 'storage')
    
    # Initialize pipeline
    pipeline = DataExtractionPipeline(config)
    
    # Process input
    if os.path.isdir(args.input):
        results = pipeline.process_directory(args.input)
    else:
        results = [pipeline.process_file(args.input)]
    
    # Save results
    results_file = os.path.join(config['storage_dir'], 'extraction_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_file}")
    
    # Close pipeline
    pipeline.close()
    
    # Print summary
    success_count = sum(1 for r in results if r['success'])
    logger.info(f"Processed {len(results)} files, {success_count} successful")


if __name__ == "__main__":
    main()
