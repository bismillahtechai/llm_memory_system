"""
Docker configuration for the Mem0 component.

This module provides Docker configuration files for containerizing the Mem0 component.
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
    """Generate Docker configuration files for the Mem0 component."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Docker configurator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = config.get('output_dir', '/home/ubuntu/llm_memory_system/mem0_component/docker')
    
    def generate_dockerfile(self) -> str:
        """
        Generate a Dockerfile for the Mem0 component.
        
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

# Create data directory
RUN mkdir -p /app/data

# Expose port
EXPOSE 8001

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STORAGE_PATH=/app/data/memories.json

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
        Generate a requirements.txt file for the Mem0 component.
        
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
"""
        
        # Write requirements.txt
        with open(requirements_path, 'w') as f:
            f.write(requirements_content.strip())
        
        logger.info(f"Generated requirements.txt at {requirements_path}")
        
        return requirements_path
    
    def generate_docker_compose(self) -> str:
        """
        Generate a docker-compose.yml file for the Mem0 component.
        
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
  mem0-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    environment:
      - STORAGE_PATH=/app/data/memories.json
      - MAX_MEMORIES=1000
    volumes:
      - mem0_data:/app/data

volumes:
  mem0_data:
"""
        
        # Write docker-compose.yml
        with open(docker_compose_path, 'w') as f:
            f.write(docker_compose_content.strip())
        
        logger.info(f"Generated docker-compose.yml at {docker_compose_path}")
        
        return docker_compose_path
    
    def generate_env_file(self) -> str:
        """
        Generate a .env file for the Mem0 component.
        
        Returns:
            Path to the generated .env file
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # .env file path
        env_path = os.path.join(self.output_dir, '.env')
        
        # .env file content
        env_content = """
# Mem0 configuration
STORAGE_PATH=/app/data/memories.json
MAX_MEMORIES=1000
"""
        
        # Write .env file
        with open(env_path, 'w') as f:
            f.write(env_content.strip())
        
        logger.info(f"Generated .env file at {env_path}")
        
        return env_path
    
    def generate_readme(self) -> str:
        """
        Generate a README.md file for the Mem0 component.
        
        Returns:
            Path to the generated README.md file
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # README.md file path
        readme_path = os.path.join(self.output_dir, 'README.md')
        
        # README.md file content
        readme_content = """
# Mem0 Component for LLM Memory System

This directory contains the Mem0 component for the LLM memory system. The Mem0 component stores and retrieves personalized memory such as user preferences, past insights, and key conversation takeaways.

## Components

- `mem0.py`: Core Mem0 component for personalized memory
- `api.py`: FastAPI server for the Mem0 component
- `Dockerfile`: Docker configuration for the Mem0 component
- `docker-compose.yml`: Docker Compose configuration for the Mem0 component
- `requirements.txt`: Python dependencies for the Mem0 component
- `.env`: Environment variables for the Mem0 component

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

1. Build and start the container:
   ```
   docker-compose up -d
   ```

2. The API will be available at `http://localhost:8001`

## API Endpoints

- `GET /`: Check if the API is running
- `POST /memory`: Add a memory entry
- `GET /memory/{memory_id}`: Get a memory entry by ID
- `PUT /memory/{memory_id}`: Update a memory entry
- `DELETE /memory/{memory_id}`: Delete a memory entry
- `POST /memories/search`: Search for memories by content
- `POST /memories/type`: Get memory entries by type
- `POST /extract`: Extract memories from a message
- `POST /relevant-memories`: Get relevant memories formatted for inclusion in an LLM prompt

## Environment Variables

- `STORAGE_PATH`: Path to the storage file (default: `/app/data/memories.json`)
- `MAX_MEMORIES`: Maximum number of memories to store (default: `1000`)

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
        'output_dir': '/home/ubuntu/llm_memory_system/mem0_component/docker'
    }
    
    # Create Docker configurator
    docker_configurator = DockerConfigurator(config)
    
    # Generate all files
    files = docker_configurator.generate_all()
    
    print("Generated Docker configuration files:")
    for file_type, file_path in files.items():
        print(f"- {file_type}: {file_path}")
