"""
Docker configuration for the RAG component with Weaviate.

This module provides Docker configuration files for containerizing the RAG component
with Weaviate as the vector database.
"""

import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DockerConfigurator:
    """Generate Docker configuration files for the RAG component."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Docker configurator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = config.get('output_dir', '/home/ubuntu/llm_memory_system/rag_component/docker')
    
    def generate_dockerfile(self) -> str:
        """
        Generate a Dockerfile for the RAG component.
        
        Returns:
            Path to the generated Dockerfile
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Dockerfile path
        dockerfile_path = os.path.join(self.output_dir, 'Dockerfile')
        
        # Dockerfile content
        dockerfile_content = """
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "api.py"]
"""
        
        # Write Dockerfile
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content.strip())
        
        logger.info(f"Generated Dockerfile at {dockerfile_path}")
        
        return dockerfile_path
    
    def generate_requirements(self) -> str:
        """
        Generate a requirements.txt file for the RAG component.
        
        Returns:
            Path to the generated requirements.txt file
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Requirements path
        requirements_path = os.path.join(self.output_dir, 'requirements.txt')
        
        # Requirements content
        requirements_content = """
fastapi==0.95.1
uvicorn==0.22.0
pydantic==1.10.7
sentence-transformers==2.2.2
weaviate-client==3.15.4
python-dotenv==1.0.0
numpy==1.24.3
"""
        
        # Write requirements.txt
        with open(requirements_path, 'w') as f:
            f.write(requirements_content.strip())
        
        logger.info(f"Generated requirements.txt at {requirements_path}")
        
        return requirements_path
    
    def generate_docker_compose(self) -> str:
        """
        Generate a docker-compose.yml file for the RAG component with Weaviate.
        
        Returns:
            Path to the generated docker-compose.yml file
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Docker Compose path
        docker_compose_path = os.path.join(self.output_dir, 'docker-compose.yml')
        
        # Docker Compose content
        docker_compose_content = """
version: '3.4'

services:
  weaviate:
    image: semitechnologies/weaviate:1.19.6
    ports:
      - "8080:8080"
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: ''
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate

  rag-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    depends_on:
      - weaviate
    environment:
      - WEAVIATE_URL=http://weaviate:8080
      - EMBEDDING_MODEL=all-MiniLM-L6-v2
    volumes:
      - ./data:/app/data

volumes:
  weaviate_data:
"""
        
        # Write docker-compose.yml
        with open(docker_compose_path, 'w') as f:
            f.write(docker_compose_content.strip())
        
        logger.info(f"Generated docker-compose.yml at {docker_compose_path}")
        
        return docker_compose_path
    
    def generate_api_file(self) -> str:
        """
        Generate an API file for the RAG component.
        
        Returns:
            Path to the generated API file
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # API file path
        api_path = os.path.join(self.output_dir, 'api.py')
        
        # API file content
        api_content = """
import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG API",
    description="API for Retrieval-Augmented Generation (RAG) component",
    version="1.0.0"
)

# Load configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))

# Initialize components
embedding_model = None
vector_db = None

# Pydantic models for API
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    max_results: Optional[int] = None

class AugmentMessagesRequest(BaseModel):
    messages: List[Message]
    conversation_id: Optional[str] = None
    max_results: Optional[int] = None

class ChunkMetadata(BaseModel):
    content: str
    conversation_id: str
    conversation_title: Optional[str] = None
    chunk_type: str
    timestamp: Optional[str] = None

class RetrievalResult(BaseModel):
    id: str
    similarity: float
    metadata: ChunkMetadata

class QueryResponse(BaseModel):
    context: str
    results: List[RetrievalResult]
    query: str
    timestamp: str

class AugmentMessagesResponse(BaseModel):
    messages: List[Message]
    context_added: bool

class AddEmbeddingRequest(BaseModel):
    content: str
    embedding: List[float]
    conversation_id: str
    conversation_title: Optional[str] = None
    chunk_type: str = "message"
    metadata: Optional[Dict[str, Any]] = None

class AddEmbeddingResponse(BaseModel):
    id: str
    success: bool

# Initialize components on startup
@app.on_event("startup")
async def startup_event():
    global embedding_model, vector_db
    
    try:
        # Import vector database adapter
        from vector_db import VectorDBManager
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize vector database
        logger.info(f"Connecting to Weaviate at {WEAVIATE_URL}")
        vector_db = VectorDBManager(
            db_type='weaviate',
            config={
                'url': WEAVIATE_URL,
                'class_name': 'MemoryChunk'
            }
        )
        vector_db.initialize()
        
        logger.info("RAG API initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing RAG API: {e}")
        raise

# API endpoints
@app.get("/")
async def root():
    return {"message": "RAG API is running"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG component for relevant context.
    """
    try:
        # Generate embedding for query
        query_embedding = embedding_model.encode(request.query)
        
        # Set max results
        max_results = request.max_results or MAX_RESULTS
        
        # Search for similar chunks
        results = vector_db.search_similar(
            query_embedding.tolist(),
            limit=max_results
        )
        
        # Filter results by similarity threshold
        filtered_results = [
            r for r in results 
            if r['similarity'] >= SIMILARITY_THRESHOLD
        ]
        
        # Format results
        formatted_results = []
        for result in filtered_results:
            metadata = result['metadata']
            formatted_results.append(RetrievalResult(
                id=result['id'],
                similarity=result['similarity'],
                metadata=ChunkMetadata(
                    content=metadata.get('content', ''),
                    conversation_id=metadata.get('conversation_id', ''),
                    conversation_title=metadata.get('conversation_title', ''),
                    chunk_type=metadata.get('chunk_type', 'unknown'),
                    timestamp=metadata.get('timestamp', '')
                )
            ))
        
        # Format context
        context = format_context(formatted_results)
        
        return QueryResponse(
            context=context,
            results=formatted_results,
            query=request.query,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/augment-messages", response_model=AugmentMessagesResponse)
async def augment_messages(request: AugmentMessagesRequest):
    """
    Augment a list of messages with RAG context.
    """
    try:
        # Extract the last user message as the query
        user_messages = [m for m in request.messages if m.role == 'user']
        
        if not user_messages:
            # No user messages to use as query
            return AugmentMessagesResponse(
                messages=request.messages,
                context_added=False
            )
        
        last_user_message = user_messages[-1]
        query = last_user_message.content
        
        if not query:
            # Empty query
            return AugmentMessagesResponse(
                messages=request.messages,
                context_added=False
            )
        
        # Query for context
        query_response = await query(QueryRequest(
            query=query,
            conversation_id=request.conversation_id,
            max_results=request.max_results
        ))
        
        context = query_response.context
        
        if not context:
            # No relevant context found
            return AugmentMessagesResponse(
                messages=request.messages,
                context_added=False
            )
        
        # Create a new list of messages
        augmented_messages = []
        
        # Find the system message if it exists
        system_message_index = None
        for i, message in enumerate(request.messages):
            if message.role == 'system':
                system_message_index = i
                break
        
        if system_message_index is not None:
            # Augment the existing system message
            system_message = request.messages[system_message_index]
            augmented_system_content = f"{system_message.content}\\n\\n{format_context_for_prompt(context)}"
            
            # Add all messages with the augmented system message
            for i, message in enumerate(request.messages):
                if i == system_message_index:
                    augmented_messages.append(Message(
                        role='system',
                        content=augmented_system_content
                    ))
                else:
                    augmented_messages.append(message)
        else:
            # Add a new system message with the context
            augmented_messages.append(Message(
                role='system',
                content=format_context_for_prompt(context)
            ))
            
            # Add all original messages
            augmented_messages.extend(request.messages)
        
        return AugmentMessagesResponse(
            messages=augmented_messages,
            context_added=True
        )
    except Exception as e:
        logger.error(f"Error augmenting messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-embedding", response_model=AddEmbeddingResponse)
async def add_embedding(request: AddEmbeddingRequest):
    """
    Add an embedding to the vector database.
    """
    try:
        # Prepare metadata
        metadata = {
            'content': request.content,
            'conversation_id': request.conversation_id,
            'conversation_title': request.conversation_title or '',
            'chunk_type': request.chunk_type,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add user metadata if provided
        if request.metadata:
            metadata.update(request.metadata)
        
        # Add embedding
        embedding_id = vector_db.add_embedding(
            embedding=request.embedding,
            metadata=metadata
        )
        
        return AddEmbeddingResponse(
            id=embedding_id,
            success=True
        )
    except Exception as e:
        logger.error(f"Error adding embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
def format_context(results: List[RetrievalResult]) -> str:
    """
    Format retrieved results into context string.
    """
    if not results:
        return ""
    
    context_parts = []
    
    for i, result in enumerate(results):
        content = result.metadata.content
        similarity = result.similarity
        conversation_title = result.metadata.conversation_title or ''
        
        # Format the context entry
        context_part = f"[{i+1}] {content}"
        
        # Add source information if available
        if conversation_title:
            context_part += f"\\n(From: {conversation_title}, Relevance: {similarity:.2f})"
        
        context_parts.append(context_part)
    
    return "\\n\\n".join(context_parts)

def format_context_for_prompt(context: str) -> str:
    """
    Format context for inclusion in an LLM prompt.
    """
    return f"""
Relevant information from previous conversations:

{context}

Use this information to inform your response if relevant to the user's query.
"""

# Run the API server
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
"""
        
        # Write API file
        with open(api_path, 'w') as f:
            f.write(api_content.strip())
        
        logger.info(f"Generated API file at {api_path}")
        
        return api_path
    
    def generate_env_file(self) -> str:
        """
        Generate a .env file for the RAG component.
        
        Returns:
            Path to the generated .env file
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # .env file path
        env_path = os.path.join(self.output_dir, '.env')
        
        # .env file content
        env_content = """
# Weaviate configuration
WEAVIATE_URL=http://localhost:8080

# Embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# RAG configuration
MAX_RESULTS=5
SIMILARITY_THRESHOLD=0.6
"""
        
        # Write .env file
        with open(env_path, 'w') as f:
            f.write(env_content.strip())
        
        logger.info(f"Generated .env file at {env_path}")
        
        return env_path
    
    def generate_readme(self) -> str:
        """
        Generate a README.md file for the RAG component.
        
        Returns:
            Path to the generated README.md file
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # README.md file path
        readme_path = os.path.join(self.output_dir, 'README.md')
        
        # README.md file content
        readme_content = """
# RAG Component for LLM Memory System

This directory contains the RAG (Retrieval-Augmented Generation) component for the LLM memory system. The RAG component retrieves relevant context from past conversations to augment LLM prompts.

## Components

- `api.py`: FastAPI server for the RAG component
- `vector_db.py`: Vector database adapters for FAISS and Weaviate
- `Dockerfile`: Docker configuration for the RAG component
- `docker-compose.yml`: Docker Compose configuration for the RAG component with Weaviate
- `requirements.txt`: Python dependencies for the RAG component
- `.env`: Environment variables for the RAG component

## Deployment

### Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start Weaviate:
   ```
   docker run -d -p 8080:8080 --name weaviate semitechnologies/weaviate:1.19.6
   ```

3. Start the API server:
   ```
   python api.py
   ```

### Docker Deployment

1. Build and start the containers:
   ```
   docker-compose up -d
   ```

2. The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /`: Check if the API is running
- `POST /query`: Query the RAG component for relevant context
- `POST /augment-messages`: Augment a list of messages with RAG context
- `POST /add-embedding`: Add an embedding to the vector database

## Environment Variables

- `WEAVIATE_URL`: URL of the Weaviate instance (default: `http://localhost:8080`)
- `EMBEDDING_MODEL`: Name of the sentence-transformers model to use (default: `all-MiniLM-L6-v2`)
- `MAX_RESULTS`: Maximum number of results to return (default: `5`)
- `SIMILARITY_THRESHOLD`: Minimum similarity threshold for results (default: `0.6`)

## Render Deployment

To deploy on Render:

1. Create a new Web Service
2. Connect your GitHub repository
3. Set the following:
   - Build Command: `docker-compose build`
   - Start Command: `docker-compose up`
   - Environment Variables: Set as needed

Alternatively, you can use Render's native Docker support:

1. Create a new Web Service
2. Connect your GitHub repository
3. Select "Docker" as the environment
4. Set environment variables as needed
"""
        
        # Write README.md file
        with open(readme_path, 'w') as f:
            f.write(readme_content.strip())
        
        logger.info(f"Generated README.md file at {readme_path}")
        
        return readme_path
    
    def generate_all(self) -> Dict[str, str]:
        """
        Generate all Docker configuration files.
        
        Returns:
            Dictionary mapping file types to file paths
        """
        return {
            'dockerfile': self.generate_dockerfile(),
            'requirements': self.generate_requirements(),
            'docker_compose': self.generate_docker_compose(),
            'api': self.generate_api_file(),
            'env': self.generate_env_file(),
            'readme': self.generate_readme()
        }


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'output_dir': '/home/ubuntu/llm_memory_system/rag_component/docker'
    }
    
    # Create Docker configurator
    docker_configurator = DockerConfigurator(config)
    
    # Generate all files
    files = docker_configurator.generate_all()
    
    print("Generated Docker configuration files:")
    for file_type, file_path in files.items():
        print(f"- {file_type}: {file_path}")
