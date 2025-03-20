# Dual Memory Architecture Design

## Overview

This document outlines the architecture for our cost-efficient, persistent memory system for LLMs. The system combines two complementary memory approaches (RAG and Mem0) to enable seamless recall of past interactions while keeping prompt sizes minimal.

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                     LLM Memory System                        │
│                                                             │
│  ┌───────────────┐    ┌───────────────┐    ┌──────────────┐ │
│  │ Data          │    │ Dual Memory   │    │ TypingMind   │ │
│  │ Extraction &  │───▶│ Framework     │───▶│ Integration  │ │
│  │ Consolidation │    │ (RAG + Mem0)  │    │              │ │
│  └───────────────┘    └───────────────┘    └──────────────┘ │
│                              │                               │
│                              ▼                               │
│                     ┌───────────────────┐                    │
│                     │ Dynamic Context   │                    │
│                     │ Management        │                    │
│                     └───────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### Component Interactions

1. **Data Extraction & Consolidation** processes conversation exports from ChatGPT and TypingMind
2. **Dual Memory Framework** stores and retrieves information using both RAG and Mem0 approaches
3. **Dynamic Context Management** determines what to store and retrieve based on conversation context
4. **TypingMind Integration** connects the memory system to the TypingMind frontend

## Detailed Component Design

### 1. Data Extraction & Consolidation

```
┌─────────────────────────────────────────────────────────────┐
│                Data Extraction & Consolidation               │
│                                                             │
│  ┌───────────────┐    ┌───────────────┐    ┌──────────────┐ │
│  │ Export        │    │ Parsing &     │    │ Chunking &   │ │
│  │ Handlers      │───▶│ Normalization │───▶│ Processing   │ │
│  │               │    │               │    │              │ │
│  └───────────────┘    └───────────────┘    └──────────────┘ │
│                                                    │        │
│                                                    ▼        │
│                                           ┌──────────────┐  │
│                                           │ Metadata     │  │
│                                           │ Enrichment   │  │
│                                           └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### Key Components:

- **Export Handlers**: Specialized parsers for ChatGPT and TypingMind export formats
- **Parsing & Normalization**: Converts exports to a standardized conversation format
- **Chunking & Processing**: Splits conversations into appropriate units for embedding
- **Metadata Enrichment**: Adds timestamps, speakers, topics, and other contextual information

#### Data Flow:

1. User provides conversation exports from ChatGPT and TypingMind
2. Export handlers detect format and apply appropriate parsing
3. Conversations are normalized to a standard schema
4. Text is chunked using semantic boundaries (turns, paragraphs, or topics)
5. Metadata is added to each chunk for improved retrieval

### 2. Dual Memory Framework

```
┌─────────────────────────────────────────────────────────────┐
│                     Dual Memory Framework                    │
│                                                             │
│  ┌───────────────────────────┐  ┌───────────────────────────┐
│  │          RAG              │  │          Mem0             │
│  │                           │  │                           │
│  │  ┌─────────────────────┐  │  │  ┌─────────────────────┐  │
│  │  │ Embedding Pipeline  │  │  │  │ Memory Extraction   │  │
│  │  └─────────────────────┘  │  │  └─────────────────────┘  │
│  │           │               │  │           │               │
│  │           ▼               │  │           ▼               │
│  │  ┌─────────────────────┐  │  │  ┌─────────────────────┐  │
│  │  │ Vector Database     │  │  │  │ Structured Storage  │  │
│  │  └─────────────────────┘  │  │  └─────────────────────┘  │
│  │           │               │  │           │               │
│  │           ▼               │  │           ▼               │
│  │  ┌─────────────────────┐  │  │  ┌─────────────────────┐  │
│  │  │ Retrieval Engine    │  │  │  │ Retrieval Logic     │  │
│  │  └─────────────────────┘  │  │  └─────────────────────┘  │
│  └───────────────────────────┘  └───────────────────────────┘
│                │                               │             │
│                └───────────────┬───────────────┘             │
│                                │                             │
│                                ▼                             │
│                     ┌─────────────────────┐                  │
│                     │ Memory Orchestrator │                  │
│                     └─────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

#### RAG Component:

- **Embedding Pipeline**: Converts text chunks to vector embeddings
- **Vector Database**: Stores embeddings with metadata (FAISS, Pinecone, or Weaviate)
- **Retrieval Engine**: Performs similarity search and reranking

#### Mem0 Component:

- **Memory Extraction**: Identifies personal preferences and context from conversations
- **Structured Storage**: Organizes memories in a structured, queryable format
- **Retrieval Logic**: Selects relevant personal context based on conversation

#### Memory Orchestrator:

- Coordinates between RAG and Mem0 components
- Allocates token budget between different memory types
- Resolves conflicts when both systems return relevant information
- Formats retrieved memories for inclusion in prompts

### 3. Dynamic Context Management

```
┌─────────────────────────────────────────────────────────────┐
│                   Dynamic Context Management                 │
│                                                             │
│  ┌───────────────┐    ┌───────────────┐    ┌──────────────┐ │
│  │ Context       │    │ Memory        │    │ Token        │ │
│  │ Analyzer      │───▶│ Selector      │───▶│ Optimizer    │ │
│  │               │    │               │    │              │ │
│  └───────────────┘    └───────────────┘    └──────────────┘ │
│         ▲                                          │        │
│         │                                          ▼        │
│  ┌──────────────┐                         ┌──────────────┐  │
│  │ User Query   │◀────────────────────────│ Context      │  │
│  │ Processor    │                         │ Assembler    │  │
│  └──────────────┘                         └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### Key Components:

- **Context Analyzer**: Determines what type of memory is needed for the current query
- **Memory Selector**: Decides which memory system(s) to query and with what parameters
- **Token Optimizer**: Ensures retrieved context fits within token budget
- **Context Assembler**: Formats selected memories for inclusion in the prompt
- **User Query Processor**: Analyzes user queries to guide memory retrieval

#### Operation Flow:

1. User query is analyzed to determine memory needs
2. Appropriate memory systems are queried (RAG, Mem0, or both)
3. Retrieved memories are optimized to fit token budget
4. Context is assembled into a format suitable for the LLM
5. After LLM response, new information is extracted for memory storage

### 4. TypingMind Integration

```
┌─────────────────────────────────────────────────────────────┐
│                     TypingMind Integration                   │
│                                                             │
│  ┌───────────────┐    ┌───────────────┐    ┌──────────────┐ │
│  │ API           │    │ System        │    │ Memory       │ │
│  │ Interceptor   │───▶│ Instruction   │───▶│ UI          │ │
│  │               │    │ Modifier      │    │ Components   │ │
│  └───────────────┘    └───────────────┘    └──────────────┘ │
│         ▲                      │                  │         │
│         │                      │                  │         │
│         │                      ▼                  │         │
│  ┌──────────────┐     ┌──────────────┐           │         │
│  │ TypingMind   │     │ LLM API      │           │         │
│  │ Frontend     │◀────│ Connector    │◀──────────┘         │
│  └──────────────┘     └──────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

#### Key Components:

- **API Interceptor**: Captures messages before they reach the LLM API
- **System Instruction Modifier**: Augments instructions with memory context
- **Memory UI Components**: Provides user interface for memory visibility and control
- **LLM API Connector**: Handles communication with the LLM provider
- **TypingMind Frontend**: The existing TypingMind user interface

#### Integration Approaches:

1. **Browser Extension**:
   - Injects JavaScript to intercept API calls
   - Adds UI elements for memory management
   - Communicates with memory backend via API

2. **Server-Side Proxy**:
   - Sits between TypingMind and LLM APIs
   - Handles memory operations server-side
   - Requires configuration of TypingMind to use proxy

3. **TypingMind Plugin** (if supported):
   - Uses official extension mechanisms
   - Integrates natively with TypingMind UI
   - Leverages existing plugin infrastructure

## Data Models

### Conversation Schema

```json
{
  "conversation_id": "string",
  "source": "chatgpt|typingmind",
  "timestamp": "ISO-8601 datetime",
  "title": "string",
  "messages": [
    {
      "role": "user|assistant|system",
      "content": "string",
      "timestamp": "ISO-8601 datetime"
    }
  ],
  "metadata": {
    "tags": ["string"],
    "topics": ["string"],
    "custom_fields": {}
  }
}
```

### Memory Chunk Schema

```json
{
  "chunk_id": "string",
  "conversation_id": "string",
  "content": "string",
  "embedding": [float],
  "metadata": {
    "source": "chatgpt|typingmind",
    "timestamp": "ISO-8601 datetime",
    "speaker": "user|assistant",
    "topic": "string",
    "chunk_type": "turn|paragraph|semantic",
    "importance_score": float
  }
}
```

### Mem0 Schema

```json
{
  "user_id": "string",
  "preferences": {
    "communication_style": "string",
    "interests": ["string"],
    "dislikes": ["string"]
  },
  "projects": {
    "project_name": {
      "description": "string",
      "status": "string",
      "key_requirements": ["string"]
    }
  },
  "interaction_patterns": {
    "response_style": "string",
    "follow_up_behavior": "string"
  },
  "facts": [
    {
      "content": "string",
      "confidence": float,
      "source_conversations": ["string"],
      "last_updated": "ISO-8601 datetime"
    }
  ]
}
```

## System Workflow

### Initialization Flow

1. User provides conversation exports from ChatGPT and TypingMind
2. System processes exports and builds initial memory stores
3. User installs TypingMind extension or configures proxy
4. System connects to TypingMind and begins intercepting conversations

### Conversation Flow

1. User sends message in TypingMind
2. API Interceptor captures the message
3. Context Analyzer determines memory needs
4. Memory systems (RAG and/or Mem0) are queried
5. Retrieved context is optimized and formatted
6. Augmented prompt is sent to LLM API
7. Response is returned to user
8. Memory Extraction identifies new information to store
9. Memory stores are updated

### Memory Maintenance Flow

1. Periodic consolidation of similar memories
2. Summarization of conversation clusters
3. Pruning of less relevant or outdated memories
4. Optimization of vector indices
5. Verification of high-impact memories with user

## Implementation Considerations

### Technology Stack

- **Backend**: Python with FastAPI or Flask
- **Vector Database**: FAISS (local development), Pinecone or Weaviate (production)
- **Embedding Models**: Sentence-transformers or OpenAI embeddings
- **Frontend**: JavaScript for TypingMind integration
- **Storage**: SQLite (development), PostgreSQL (production)
- **Deployment**: Docker containers for easy setup

### Scalability Considerations

- **Vector Database Scaling**: Strategies for handling growing embedding collections
- **Memory Pruning**: Automated approaches to maintain manageable memory size
- **Batch Processing**: Efficient handling of large conversation exports
- **Caching**: Optimizing frequent memory retrievals

### Privacy and Security

- **Data Encryption**: Protecting sensitive conversation data
- **User Control**: Allowing users to manage what is remembered
- **Data Minimization**: Storing only what's necessary
- **Local-First**: Prioritizing local storage where possible

## Next Steps

1. Implement data extraction and consolidation pipeline
2. Develop RAG component with vector database integration
3. Implement Mem0 component for personalized memory
4. Create dynamic context management system
5. Integrate with TypingMind frontend
6. Test and optimize the complete system
