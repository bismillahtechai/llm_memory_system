"""
API for the dynamic context management system.

This module provides a FastAPI server for the dynamic context management system,
allowing it to be accessed via HTTP requests.
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

from context_manager import ContextManager, RAGClient, Mem0Client

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
    title="Context Management API",
    description="API for the dynamic context management system of the LLM memory system",
    version="1.0.0"
)

# Load configuration
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
MEM0_API_URL = os.getenv("MEM0_API_URL", "http://localhost:8001")
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "1000"))
RAG_WEIGHT = float(os.getenv("RAG_WEIGHT", "0.7"))
MEM0_WEIGHT = float(os.getenv("MEM0_WEIGHT", "0.3"))
MAX_RAG_RESULTS = int(os.getenv("MAX_RAG_RESULTS", "5"))
MAX_MEM0_RESULTS = int(os.getenv("MAX_MEM0_RESULTS", "5"))
MEMORY_TYPES = os.getenv("MEMORY_TYPES", "preference,insight,personal_info").split(",")

# Initialize components
context_manager = None

# Pydantic models for API
class Message(BaseModel):
    role: str
    content: str

class AnalyzeQueryRequest(BaseModel):
    query: str

class AnalyzeQueryResponse(BaseModel):
    needs_factual: bool
    needs_personal: bool
    factual_keywords: List[str]
    personal_keywords: List[str]
    query: str

class GetContextRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class GetContextResponse(BaseModel):
    rag_context: str
    mem0_context: str
    combined_context: str
    token_count: int
    analysis: Dict[str, Any]

class AugmentMessagesRequest(BaseModel):
    messages: List[Message]
    conversation_id: Optional[str] = None

class AugmentMessagesResponse(BaseModel):
    messages: List[Message]
    context_added: bool
    token_count: int

class AugmentPromptRequest(BaseModel):
    prompt: str
    conversation_id: Optional[str] = None

class AugmentPromptResponse(BaseModel):
    augmented_prompt: str
    context_added: bool
    token_count: int

# Initialize components on startup
@app.on_event("startup")
async def startup_event():
    global context_manager
    
    try:
        # Initialize clients
        logger.info(f"Initializing RAG client with URL: {RAG_API_URL}")
        rag_client = RAGClient(api_url=RAG_API_URL)
        
        logger.info(f"Initializing Mem0 client with URL: {MEM0_API_URL}")
        mem0_client = Mem0Client(api_url=MEM0_API_URL)
        
        # Initialize context manager
        logger.info("Initializing context manager")
        context_manager = ContextManager(
            rag_client=rag_client,
            mem0_client=mem0_client,
            config={
                'token_budget': TOKEN_BUDGET,
                'rag_weight': RAG_WEIGHT,
                'mem0_weight': MEM0_WEIGHT,
                'max_rag_results': MAX_RAG_RESULTS,
                'max_mem0_results': MAX_MEM0_RESULTS,
                'memory_types': MEMORY_TYPES
            }
        )
        
        logger.info("Context Management API initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Context Management API: {e}")
        raise

# API endpoints
@app.get("/")
async def root():
    return {"message": "Context Management API is running"}

@app.post("/analyze-query", response_model=AnalyzeQueryResponse)
async def analyze_query(request: AnalyzeQueryRequest):
    """
    Analyze a query to determine context needs.
    """
    try:
        analysis = context_manager.analyze_query(request.query)
        
        return AnalyzeQueryResponse(
            needs_factual=analysis['needs_factual'],
            needs_personal=analysis['needs_personal'],
            factual_keywords=analysis['factual_keywords'],
            personal_keywords=analysis['personal_keywords'],
            query=analysis['query']
        )
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-context", response_model=GetContextResponse)
async def get_context(request: GetContextRequest):
    """
    Get context for a query.
    """
    try:
        context = context_manager.get_context(
            query=request.query,
            conversation_id=request.conversation_id
        )
        
        # Optimize context
        optimized_context = context_manager.optimize_context(context)
        
        return GetContextResponse(
            rag_context=optimized_context['rag_context'],
            mem0_context=optimized_context['mem0_context'],
            combined_context=optimized_context['combined_context'],
            token_count=optimized_context['token_count'],
            analysis=optimized_context['analysis']
        )
    except Exception as e:
        logger.error(f"Error getting context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/augment-messages", response_model=AugmentMessagesResponse)
async def augment_messages(request: AugmentMessagesRequest):
    """
    Augment a list of messages with context.
    """
    try:
        # Convert Pydantic models to dictionaries
        messages = [message.dict() for message in request.messages]
        
        # Augment messages
        augmented_messages = context_manager.augment_messages(
            messages=messages,
            conversation_id=request.conversation_id
        )
        
        # Check if context was added
        context_added = False
        token_count = 0
        
        for i, (orig_msg, aug_msg) in enumerate(zip(messages, augmented_messages)):
            if orig_msg.get('role') == 'system' and aug_msg.get('role') == 'system':
                if orig_msg.get('content') != aug_msg.get('content'):
                    context_added = True
                    token_count = context_manager.estimate_tokens(
                        aug_msg.get('content', '') - orig_msg.get('content', '')
                    )
                    break
        
        # If a new system message was added
        if len(augmented_messages) > len(messages):
            context_added = True
            if augmented_messages[0].get('role') == 'system':
                token_count = context_manager.estimate_tokens(
                    augmented_messages[0].get('content', '')
                )
        
        # Convert dictionaries back to Pydantic models
        augmented_message_models = [
            Message(role=msg.get('role', ''), content=msg.get('content', ''))
            for msg in augmented_messages
        ]
        
        return AugmentMessagesResponse(
            messages=augmented_message_models,
            context_added=context_added,
            token_count=token_count
        )
    except Exception as e:
        logger.error(f"Error augmenting messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/augment-prompt", response_model=AugmentPromptResponse)
async def augment_prompt(request: AugmentPromptRequest):
    """
    Augment a text prompt with context.
    """
    try:
        # Augment prompt
        augmented_prompt = context_manager.augment_prompt(
            prompt=request.prompt,
            conversation_id=request.conversation_id
        )
        
        # Check if context was added
        context_added = augmented_prompt != request.prompt
        
        # Estimate token count
        if context_added:
            token_count = context_manager.estimate_tokens(
                augmented_prompt) - context_manager.estimate_tokens(request.prompt
            )
        else:
            token_count = 0
        
        return AugmentPromptResponse(
            augmented_prompt=augmented_prompt,
            context_added=context_added,
            token_count=token_count
        )
    except Exception as e:
        logger.error(f"Error augmenting prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the API server
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=False)
