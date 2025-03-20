"""
API for the Mem0 component.

This module provides a FastAPI server for the Mem0 component, allowing it to be
accessed via HTTP requests.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

from mem0 import MemoryEntry, Mem0Component, MemoryExtractor

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
    title="Mem0 API",
    description="API for the Mem0 component of the LLM memory system",
    version="1.0.0"
)

# Load configuration
STORAGE_PATH = os.getenv("STORAGE_PATH", "/app/data/memories.json")
MAX_MEMORIES = int(os.getenv("MAX_MEMORIES", "1000"))

# Initialize components
mem0 = None
memory_extractor = None

# Pydantic models for API
class MemoryEntryModel(BaseModel):
    content: str
    memory_type: str
    source_conversation_id: Optional[str] = None
    source_message_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    importance: float = 0.5
    memory_id: Optional[str] = None
    created_at: Optional[str] = None
    last_accessed: Optional[str] = None
    access_count: int = 0

class MemoryEntryResponse(MemoryEntryModel):
    memory_id: str
    created_at: str

class AddMemoryRequest(BaseModel):
    content: str
    memory_type: str
    source_conversation_id: Optional[str] = None
    source_message_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    importance: float = 0.5

class AddMemoryResponse(BaseModel):
    memory_id: str
    success: bool

class UpdateMemoryRequest(BaseModel):
    content: Optional[str] = None
    memory_type: Optional[str] = None
    importance: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class UpdateMemoryResponse(BaseModel):
    success: bool

class SearchMemoriesRequest(BaseModel):
    query: str
    limit: int = 10

class SearchMemoriesResponse(BaseModel):
    memories: List[MemoryEntryResponse]

class GetMemoriesRequest(BaseModel):
    memory_type: str
    limit: int = 10

class GetMemoriesResponse(BaseModel):
    memories: List[MemoryEntryResponse]

class MessageModel(BaseModel):
    content: str
    conversation_id: Optional[str] = None
    id: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ExtractMemoriesRequest(BaseModel):
    message: MessageModel

class ExtractMemoriesResponse(BaseModel):
    memories: List[MemoryEntryResponse]

class GetRelevantMemoriesRequest(BaseModel):
    query: str
    memory_types: Optional[List[str]] = None
    limit: int = 10

class GetRelevantMemoriesResponse(BaseModel):
    formatted_memories: str
    memories: List[MemoryEntryResponse]

# Initialize components on startup
@app.on_event("startup")
async def startup_event():
    global mem0, memory_extractor
    
    try:
        # Initialize Mem0 component
        logger.info(f"Initializing Mem0 component with storage path: {STORAGE_PATH}")
        mem0 = Mem0Component(
            storage_path=STORAGE_PATH,
            config={
                'max_memories': MAX_MEMORIES
            }
        )
        
        # Initialize memory extractor
        logger.info("Initializing memory extractor")
        memory_extractor = MemoryExtractor()
        
        logger.info("Mem0 API initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Mem0 API: {e}")
        raise

# API endpoints
@app.get("/")
async def root():
    return {"message": "Mem0 API is running"}

@app.post("/memory", response_model=AddMemoryResponse)
async def add_memory(request: AddMemoryRequest):
    """
    Add a memory entry.
    """
    try:
        # Create memory entry
        memory = MemoryEntry(
            content=request.content,
            memory_type=request.memory_type,
            source_conversation_id=request.source_conversation_id,
            source_message_id=request.source_message_id,
            metadata=request.metadata,
            importance=request.importance
        )
        
        # Add to Mem0
        memory_id = mem0.add_memory(memory)
        
        return AddMemoryResponse(
            memory_id=memory_id,
            success=True
        )
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/{memory_id}", response_model=MemoryEntryResponse)
async def get_memory(memory_id: str):
    """
    Get a memory entry by ID.
    """
    try:
        memory = mem0.get_memory(memory_id)
        
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory with ID {memory_id} not found")
        
        return MemoryEntryResponse(**memory.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/memory/{memory_id}", response_model=UpdateMemoryResponse)
async def update_memory(memory_id: str, request: UpdateMemoryRequest):
    """
    Update a memory entry.
    """
    try:
        updates = {}
        
        if request.content is not None:
            updates['content'] = request.content
        
        if request.memory_type is not None:
            updates['memory_type'] = request.memory_type
        
        if request.importance is not None:
            updates['importance'] = request.importance
        
        if request.metadata is not None:
            updates['metadata'] = request.metadata
        
        success = mem0.update_memory(memory_id, updates)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Memory with ID {memory_id} not found")
        
        return UpdateMemoryResponse(success=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/{memory_id}", response_model=UpdateMemoryResponse)
async def delete_memory(memory_id: str):
    """
    Delete a memory entry.
    """
    try:
        success = mem0.delete_memory(memory_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Memory with ID {memory_id} not found")
        
        return UpdateMemoryResponse(success=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/search", response_model=SearchMemoriesResponse)
async def search_memories(request: SearchMemoriesRequest):
    """
    Search for memories by content.
    """
    try:
        memories = mem0.search_memories(request.query, limit=request.limit)
        
        return SearchMemoriesResponse(
            memories=[MemoryEntryResponse(**memory.to_dict()) for memory in memories]
        )
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/type", response_model=GetMemoriesResponse)
async def get_memories_by_type(request: GetMemoriesRequest):
    """
    Get memory entries by type.
    """
    try:
        memories = mem0.get_memories_by_type(request.memory_type, limit=request.limit)
        
        return GetMemoriesResponse(
            memories=[MemoryEntryResponse(**memory.to_dict()) for memory in memories]
        )
    except Exception as e:
        logger.error(f"Error getting memories by type: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract", response_model=ExtractMemoriesResponse)
async def extract_memories(request: ExtractMemoriesRequest):
    """
    Extract memories from a message.
    """
    try:
        message_dict = {
            'content': request.message.content,
            'conversation_id': request.message.conversation_id,
            'id': request.message.id,
            'timestamp': request.message.timestamp or datetime.now().isoformat(),
            'metadata': request.message.metadata or {}
        }
        
        memories = memory_extractor.extract_memories_from_message(message_dict)
        
        return ExtractMemoriesResponse(
            memories=[MemoryEntryResponse(**memory.to_dict()) for memory in memories]
        )
    except Exception as e:
        logger.error(f"Error extracting memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/relevant-memories", response_model=GetRelevantMemoriesResponse)
async def get_relevant_memories(request: GetRelevantMemoriesRequest):
    """
    Get relevant memories formatted for inclusion in an LLM prompt.
    """
    try:
        # Search for relevant memories
        memories = mem0.search_memories(request.query, limit=request.limit)
        
        # Filter by memory types if specified
        if request.memory_types:
            memories = [
                memory for memory in memories
                if memory.memory_type in request.memory_types
            ]
        
        # Format for prompt
        formatted_memories = mem0.format_memories_for_prompt(memories)
        
        if formatted_memories:
            formatted_memories = f"""
Personal Memory:

{formatted_memories}

Use this personal memory to inform your response if relevant to the user's query.
"""
        
        return GetRelevantMemoriesResponse(
            formatted_memories=formatted_memories,
            memories=[MemoryEntryResponse(**memory.to_dict()) for memory in memories]
        )
    except Exception as e:
        logger.error(f"Error getting relevant memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the API server
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=False)
