"""
Docker configuration for the dynamic context management system.

This module provides Docker configuration files for containerizing the context management system.
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
    """Generate Docker configuration files for the context management system."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Docker configurator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = config.get('output_dir', '/home/ubuntu/llm_memory_system/context_management/docker')
    
    def generate_dockerfile(self) -> str:
        """
        Generate a Dockerfile for the context management system.
        
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
EXPOSE 8002

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV RAG_API_URL=http://rag-api:8000
ENV MEM0_API_URL=http://mem0-api:8001

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
        Generate a requirements.txt file for the context management system.
        
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
python-dotenv==1.0.0
requests==2.28.2
"""
        
        # Write requirements.txt
        with open(requirements_path, 'w') as f:
            f.write(requirements_content.strip())
        
        logger.info(f"Generated requirements.txt at {requirements_path}")
        
        return requirements_path
    
    def generate_docker_compose(self) -> str:
        """
        Generate a docker-compose.yml file for the complete memory system.
        
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
      context: ../../rag_component/docker
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    depends_on:
      - weaviate
    environment:
      - WEAVIATE_URL=http://weaviate:8080
      - EMBEDDING_MODEL=all-MiniLM-L6-v2
    volumes:
      - rag_data:/app/data

  mem0-api:
    build:
      context: ../../mem0_component/docker
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    environment:
      - STORAGE_PATH=/app/data/memories.json
      - MAX_MEMORIES=1000
    volumes:
      - mem0_data:/app/data

  context-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8002:8002"
    depends_on:
      - rag-api
      - mem0-api
    environment:
      - RAG_API_URL=http://rag-api:8000
      - MEM0_API_URL=http://mem0-api:8001
      - TOKEN_BUDGET=1000
      - RAG_WEIGHT=0.7
      - MEM0_WEIGHT=0.3
      - MAX_RAG_RESULTS=5
      - MAX_MEM0_RESULTS=5
      - MEMORY_TYPES=preference,insight,personal_info

volumes:
  weaviate_data:
  rag_data:
  mem0_data:
"""
        
        # Write docker-compose.yml
        with open(docker_compose_path, 'w') as f:
            f.write(docker_compose_content.strip())
        
        logger.info(f"Generated docker-compose.yml at {docker_compose_path}")
        
        return docker_compose_path
    
    def generate_env_file(self) -> str:
        """
        Generate a .env file for the context management system.
        
        Returns:
            Path to the generated .env file
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # .env file path
        env_path = os.path.join(self.output_dir, '.env')
        
        # .env file content
        env_content = """
# Context management configuration
RAG_API_URL=http://localhost:8000
MEM0_API_URL=http://localhost:8001
TOKEN_BUDGET=1000
RAG_WEIGHT=0.7
MEM0_WEIGHT=0.3
MAX_RAG_RESULTS=5
MAX_MEM0_RESULTS=5
MEMORY_TYPES=preference,insight,personal_info
"""
        
        # Write .env file
        with open(env_path, 'w') as f:
            f.write(env_content.strip())
        
        logger.info(f"Generated .env file at {env_path}")
        
        return env_path
    
    def generate_readme(self) -> str:
        """
        Generate a README.md file for the context management system.
        
        Returns:
            Path to the generated README.md file
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # README.md file path
        readme_path = os.path.join(self.output_dir, 'README.md')
        
        # README.md file content
        readme_content = """
# Context Management System for LLM Memory System

This directory contains the dynamic context management system for the LLM memory system. The context management system integrates the RAG and Mem0 components to provide optimized context for LLM prompts.

## Components

- `context_manager.py`: Core context management system
- `api.py`: FastAPI server for the context management system
- `Dockerfile`: Docker configuration for the context management system
- `docker-compose.yml`: Docker Compose configuration for the complete memory system
- `requirements.txt`: Python dependencies for the context management system
- `.env`: Environment variables for the context management system

## Deployment

### Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start the API server:
   ```
   python api.py
   ```

### Docker Deployment

1. Build and start the containers:
   ```
   docker-compose up -d
   ```

2. The API will be available at `http://localhost:8002`

## API Endpoints

- `GET /`: Check if the API is running
- `POST /analyze-query`: Analyze a query to determine context needs
- `POST /get-context`: Get context for a query
- `POST /augment-messages`: Augment a list of messages with context
- `POST /augment-prompt`: Augment a text prompt with context

## Environment Variables

- `RAG_API_URL`: URL of the RAG API (default: `http://localhost:8000`)
- `MEM0_API_URL`: URL of the Mem0 API (default: `http://localhost:8001`)
- `TOKEN_BUDGET`: Maximum number of tokens for context (default: `1000`)
- `RAG_WEIGHT`: Weight for RAG context (default: `0.7`)
- `MEM0_WEIGHT`: Weight for Mem0 context (default: `0.3`)
- `MAX_RAG_RESULTS`: Maximum number of RAG results (default: `5`)
- `MAX_MEM0_RESULTS`: Maximum number of Mem0 results (default: `5`)
- `MEMORY_TYPES`: Comma-separated list of memory types (default: `preference,insight,personal_info`)

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
            'env': self.generate_env_file(),
            'readme': self.generate_readme()
        }


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'output_dir': '/home/ubuntu/llm_memory_system/context_management/docker'
    }
    
    # Create Docker configurator
    docker_configurator = DockerConfigurator(config)
    
    # Generate all files
    files = docker_configurator.generate_all()
    
    print("Generated Docker configuration files:")
    for file_type, file_path in files.items():
        print(f"- {file_type}: {file_path}")
