# Data Extraction and Consolidation Pipeline Design

## Overview

This document outlines the design for the data extraction and consolidation pipeline, which is responsible for processing conversation exports from ChatGPT and TypingMind, transforming them into a standardized format, and preparing them for storage in our dual memory system.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Data Extraction and Consolidation Pipeline              │
│                                                                         │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐    │
│  │ Export Format │    │ Conversation  │    │ Text Chunking &       │    │
│  │ Detection     │───▶│ Normalization │───▶│ Processing            │    │
│  └───────────────┘    └───────────────┘    └───────────────────────┘    │
│                                                        │                 │
│                                                        ▼                 │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐    │
│  │ Memory Store  │◀───│ Embedding     │◀───│ Metadata              │    │
│  │ Integration   │    │ Generation    │    │ Enrichment            │    │
│  └───────────────┘    └───────────────┘    └───────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Export Format Detection

#### Functionality:
- Identifies the source format of conversation exports (ChatGPT or TypingMind)
- Supports multiple export formats (JSON, Markdown, HTML, etc.)
- Validates file integrity and structure

#### Implementation:
```python
def detect_export_format(file_path):
    """
    Detect the format of the exported conversation file.
    
    Args:
        file_path: Path to the exported conversation file
        
    Returns:
        dict: Format information including source (chatgpt/typingmind) and type (json/md/html)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read(4096)  # Read first 4KB to detect format
    
    # Check for ChatGPT JSON format
    if content.strip().startswith('{') and '"mapping":' in content:
        return {'source': 'chatgpt', 'format': 'json'}
    
    # Check for TypingMind JSON format
    if content.strip().startswith('[') and '"role":' in content:
        return {'source': 'typingmind', 'format': 'json'}
    
    # Check for Markdown format
    if '# ' in content and ('**User:**' in content or '**Assistant:**' in content):
        return {'source': 'unknown', 'format': 'markdown'}
    
    # Check for HTML format
    if '<html' in content.lower() and '<body' in content.lower():
        return {'source': 'unknown', 'format': 'html'}
    
    return {'source': 'unknown', 'format': 'unknown'}
```

### 2. Conversation Normalization

#### Functionality:
- Parses different export formats into a standardized conversation schema
- Extracts message content, roles, timestamps, and metadata
- Handles format-specific quirks and inconsistencies

#### Implementation:
```python
def normalize_conversation(file_path, format_info):
    """
    Parse and normalize conversation from various export formats.
    
    Args:
        file_path: Path to the exported conversation file
        format_info: Format information from detect_export_format
        
    Returns:
        dict: Normalized conversation in standard schema
    """
    if format_info['source'] == 'chatgpt' and format_info['format'] == 'json':
        return parse_chatgpt_json(file_path)
    elif format_info['source'] == 'typingmind' and format_info['format'] == 'json':
        return parse_typingmind_json(file_path)
    elif format_info['format'] == 'markdown':
        return parse_markdown(file_path)
    elif format_info['format'] == 'html':
        return parse_html(file_path)
    else:
        raise ValueError(f"Unsupported format: {format_info}")
```

#### ChatGPT JSON Parser:
```python
def parse_chatgpt_json(file_path):
    """Parse ChatGPT JSON export format"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract conversation metadata
    conversation = {
        'conversation_id': data.get('id', str(uuid.uuid4())),
        'source': 'chatgpt',
        'timestamp': data.get('create_time', datetime.now().isoformat()),
        'title': data.get('title', 'Untitled Conversation'),
        'messages': [],
        'metadata': {
            'tags': [],
            'topics': [],
            'custom_fields': {}
        }
    }
    
    # Extract messages from mapping
    mapping = data.get('mapping', {})
    message_order = []
    
    # First pass: collect message IDs in order
    current_node = data.get('current_node')
    while current_node:
        message_order.insert(0, current_node)
        current_message = mapping.get(current_node, {})
        current_node = current_message.get('parent')
    
    # Second pass: extract message content
    for msg_id in message_order:
        if msg_id in mapping:
            msg_data = mapping[msg_id]
            if 'message' in msg_data:
                message = {
                    'role': msg_data['message'].get('author', {}).get('role', 'unknown'),
                    'content': msg_data['message'].get('content', {}).get('parts', [''])[0],
                    'timestamp': msg_data['message'].get('create_time', '')
                }
                conversation['messages'].append(message)
    
    return conversation
```

#### TypingMind JSON Parser:
```python
def parse_typingmind_json(file_path):
    """Parse TypingMind JSON export format"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract conversation metadata
    conversation = {
        'conversation_id': str(uuid.uuid4()),  # Generate ID if not available
        'source': 'typingmind',
        'timestamp': datetime.now().isoformat(),
        'title': 'TypingMind Conversation',
        'messages': [],
        'metadata': {
            'tags': [],
            'topics': [],
            'custom_fields': {}
        }
    }
    
    # Extract messages
    for msg in data:
        message = {
            'role': msg.get('role', 'unknown'),
            'content': msg.get('content', ''),
            'timestamp': msg.get('timestamp', datetime.now().isoformat())
        }
        conversation['messages'].append(message)
    
    return conversation
```

### 3. Text Chunking & Processing

#### Functionality:
- Splits conversations into appropriate chunks for embedding
- Implements multiple chunking strategies (fixed-size, semantic, turn-based)
- Preserves conversation context within chunks

#### Implementation:
```python
def chunk_conversation(conversation, strategy='turn'):
    """
    Split conversation into chunks based on specified strategy.
    
    Args:
        conversation: Normalized conversation object
        strategy: Chunking strategy ('turn', 'fixed', 'semantic')
        
    Returns:
        list: Chunks with metadata
    """
    chunks = []
    
    if strategy == 'turn':
        # Turn-based chunking (each message is a chunk)
        for i, message in enumerate(conversation['messages']):
            chunk = {
                'chunk_id': f"{conversation['conversation_id']}_chunk_{i}",
                'conversation_id': conversation['conversation_id'],
                'content': message['content'],
                'metadata': {
                    'source': conversation['source'],
                    'timestamp': message.get('timestamp', conversation['timestamp']),
                    'speaker': message['role'],
                    'chunk_type': 'turn',
                    'importance_score': 1.0  # Default score
                }
            }
            chunks.append(chunk)
    
    elif strategy == 'fixed':
        # Fixed-size chunking with overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        for i, message in enumerate(conversation['messages']):
            text_chunks = text_splitter.split_text(message['content'])
            
            for j, text in enumerate(text_chunks):
                chunk = {
                    'chunk_id': f"{conversation['conversation_id']}_chunk_{i}_{j}",
                    'conversation_id': conversation['conversation_id'],
                    'content': text,
                    'metadata': {
                        'source': conversation['source'],
                        'timestamp': message.get('timestamp', conversation['timestamp']),
                        'speaker': message['role'],
                        'chunk_type': 'fixed',
                        'importance_score': 1.0
                    }
                }
                chunks.append(chunk)
    
    elif strategy == 'semantic':
        # Semantic chunking (requires additional NLP)
        # Implementation would use NLP to identify topic boundaries
        pass
    
    return chunks
```

### 4. Metadata Enrichment

#### Functionality:
- Adds contextual metadata to chunks
- Extracts topics, entities, and key information
- Calculates importance scores for prioritization
- Tags chunks with relevant categories

#### Implementation:
```python
def enrich_chunks(chunks, nlp_pipeline=None):
    """
    Enrich chunks with additional metadata using NLP techniques.
    
    Args:
        chunks: List of conversation chunks
        nlp_pipeline: Optional NLP pipeline for advanced processing
        
    Returns:
        list: Enriched chunks with additional metadata
    """
    enriched_chunks = []
    
    for chunk in chunks:
        # Basic enrichment (always performed)
        enriched = chunk.copy()
        
        # Extract word count and complexity metrics
        word_count = len(chunk['content'].split())
        enriched['metadata']['word_count'] = word_count
        
        # Calculate basic importance score based on length
        if word_count > 100:
            enriched['metadata']['importance_score'] = min(1.5, enriched['metadata']['importance_score'] * 1.2)
        
        # Advanced enrichment (if NLP pipeline is provided)
        if nlp_pipeline:
            # Extract topics
            topics = extract_topics(chunk['content'], nlp_pipeline)
            enriched['metadata']['topics'] = topics
            
            # Extract entities
            entities = extract_entities(chunk['content'], nlp_pipeline)
            enriched['metadata']['entities'] = entities
            
            # Calculate semantic importance
            semantic_score = calculate_semantic_importance(chunk['content'], nlp_pipeline)
            enriched['metadata']['importance_score'] = (
                enriched['metadata']['importance_score'] + semantic_score
            ) / 2
        
        enriched_chunks.append(enriched)
    
    return enriched_chunks
```

### 5. Embedding Generation

#### Functionality:
- Generates vector embeddings for each chunk
- Supports multiple embedding models
- Handles batching for efficiency
- Includes caching to avoid redundant computation

#### Implementation:
```python
def generate_embeddings(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generate embeddings for chunks using the specified model.
    
    Args:
        chunks: List of conversation chunks
        model_name: Name of the embedding model to use
        
    Returns:
        list: Chunks with embeddings added
    """
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    
    # Process chunks in batches
    batch_size = 32
    embedded_chunks = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [chunk['content'] for chunk in batch]
        
        # Generate embeddings
        embeddings = embedding_model.embed_documents(texts)
        
        # Add embeddings to chunks
        for j, embedding in enumerate(embeddings):
            chunk = batch[j].copy()
            chunk['embedding'] = embedding
            embedded_chunks.append(chunk)
    
    return embedded_chunks
```

### 6. Memory Store Integration

#### Functionality:
- Stores processed chunks in appropriate memory systems
- Routes factual content to RAG vector store
- Routes personal information to Mem0 structured storage
- Handles deduplication and conflict resolution

#### Implementation:
```python
def store_in_memory_systems(chunks, rag_store, mem0_store):
    """
    Store processed chunks in the appropriate memory systems.
    
    Args:
        chunks: List of chunks with embeddings
        rag_store: RAG vector store instance
        mem0_store: Mem0 structured storage instance
        
    Returns:
        dict: Storage statistics and results
    """
    stats = {
        'total_chunks': len(chunks),
        'rag_stored': 0,
        'mem0_stored': 0,
        'errors': 0
    }
    
    for chunk in chunks:
        try:
            # Determine if chunk contains personal information
            is_personal = contains_personal_info(chunk)
            
            # Store in RAG system (all chunks go to RAG)
            rag_id = rag_store.add_chunk(
                chunk['content'],
                chunk['embedding'],
                chunk['metadata']
            )
            stats['rag_stored'] += 1
            
            # Store in Mem0 system (only personal information)
            if is_personal:
                mem0_id = mem0_store.extract_and_store(
                    chunk['content'],
                    chunk['metadata']
                )
                stats['mem0_stored'] += 1
                
        except Exception as e:
            stats['errors'] += 1
            print(f"Error storing chunk {chunk['chunk_id']}: {str(e)}")
    
    return stats
```

## Export Format Specifications

### ChatGPT JSON Format

ChatGPT exports conversations in a JSON format with the following structure:

```json
{
  "id": "conversation_id",
  "title": "Conversation Title",
  "create_time": "timestamp",
  "update_time": "timestamp",
  "mapping": {
    "message_id_1": {
      "id": "message_id_1",
      "parent": "parent_message_id",
      "children": ["child_message_id"],
      "message": {
        "id": "message_id_1",
        "author": {
          "role": "user|assistant|system"
        },
        "content": {
          "content_type": "text",
          "parts": ["message content"]
        },
        "create_time": "timestamp"
      }
    },
    "message_id_2": {
      // Similar structure
    }
  },
  "current_node": "latest_message_id"
}
```

### TypingMind JSON Format

TypingMind exports conversations in a simpler JSON array format:

```json
[
  {
    "role": "system",
    "content": "System instructions"
  },
  {
    "role": "user",
    "content": "User message"
  },
  {
    "role": "assistant",
    "content": "Assistant response"
  }
]
```

## Chunking Strategies

### 1. Turn-Based Chunking

- Each message (turn) in the conversation becomes a separate chunk
- Preserves the natural dialogue structure
- Maintains speaker attribution
- Best for conversational context and attribution

### 2. Fixed-Size Chunking

- Splits text into chunks of approximately equal size
- Uses overlap to maintain context across chunk boundaries
- Size is optimized for embedding model context window
- Best for long messages and consistent processing

### 3. Semantic Chunking

- Uses NLP to identify topic boundaries and semantic units
- Keeps related content together in the same chunk
- Requires more sophisticated processing
- Best for information retrieval and topic-based organization

## Metadata Enrichment Techniques

### Basic Metadata

- **Source**: Origin of the conversation (ChatGPT/TypingMind)
- **Timestamp**: When the message was created
- **Speaker**: Who wrote the message (user/assistant/system)
- **Conversation ID**: Identifier for the parent conversation
- **Chunk Type**: How the chunk was created (turn/fixed/semantic)

### Advanced Metadata

- **Topics**: Main subjects discussed in the chunk
- **Entities**: Named entities mentioned (people, places, organizations)
- **Sentiment**: Emotional tone of the content
- **Importance Score**: Calculated relevance or significance
- **Question/Answer**: Whether the chunk contains a question or answer
- **Action Items**: Tasks or to-dos mentioned in the content

## Implementation Considerations

### Performance Optimization

- **Batch Processing**: Process chunks in batches to optimize throughput
- **Parallel Processing**: Use multiprocessing for CPU-intensive tasks
- **Caching**: Cache embeddings for identical or similar content
- **Incremental Processing**: Support for processing new conversations incrementally

### Error Handling

- **Malformed Exports**: Gracefully handle corrupted or non-standard exports
- **Embedding Failures**: Retry logic for embedding generation
- **Storage Errors**: Transaction-like behavior for consistent storage
- **Recovery Mechanisms**: Ability to resume processing after failures

### Scalability

- **Large Exports**: Handle exports with thousands of messages
- **Multiple Exports**: Process multiple conversation exports in a single batch
- **Memory Efficiency**: Stream processing to avoid loading entire conversations into memory
- **Distributed Processing**: Design for potential distributed processing in the future

## Next Steps

1. Implement export format detection and parsing
2. Develop chunking strategies and metadata enrichment
3. Integrate with embedding generation pipeline
4. Connect to RAG and Mem0 storage systems
5. Add error handling and performance optimizations
6. Test with various export formats and conversation types
