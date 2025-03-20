# TypingMind Integration Design

## Overview

This document outlines the design for integrating our dual memory system with TypingMind, enabling seamless recall of past interactions while keeping prompt sizes minimal. The integration will intercept messages, augment prompts with relevant memory, and provide user controls for the memory system.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TypingMind Integration                            │
│                                                                         │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐    │
│  │ Browser       │    │ API           │    │ Memory                │    │
│  │ Extension     │───▶│ Interceptor   │───▶│ Interface            │    │
│  └───────────────┘    └───────────────┘    └───────────────────────┘    │
│                                │                       │                 │
│                                ▼                       ▼                 │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐    │
│  │ Memory        │◀───│ Context       │◀───│ UI                    │    │
│  │ System API    │    │ Processor     │    │ Components            │    │
│  └───────────────┘    └───────────────┘    └───────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Browser Extension

#### Functionality:
- Injects JavaScript into TypingMind web interface
- Intercepts API calls to LLM providers
- Adds UI elements for memory management
- Communicates with memory system backend

#### Implementation:

**manifest.json**:
```json
{
  "manifest_version": 3,
  "name": "TypingMind Memory Extension",
  "version": "1.0",
  "description": "Adds persistent memory capabilities to TypingMind",
  "permissions": ["storage", "webRequest", "scripting"],
  "host_permissions": ["https://*.typingmind.com/*"],
  "background": {
    "service_worker": "background.js"
  },
  "content_scripts": [
    {
      "matches": ["https://*.typingmind.com/*"],
      "js": ["content.js"],
      "css": ["styles.css"]
    }
  ],
  "web_accessible_resources": [
    {
      "resources": ["ui_components.js"],
      "matches": ["https://*.typingmind.com/*"]
    }
  ]
}
```

**background.js**:
```javascript
// Background script for handling API communication
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "memory_query") {
    // Query memory system API
    fetch(MEMORY_API_URL + "/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        query: request.query,
        user_id: request.user_id,
        conversation_id: request.conversation_id
      })
    })
    .then(response => response.json())
    .then(data => {
      sendResponse({ success: true, data: data });
    })
    .catch(error => {
      console.error("Memory API error:", error);
      sendResponse({ success: false, error: error.toString() });
    });
    
    return true; // Required for async response
  }
});
```

**content.js**:
```javascript
// Content script for TypingMind integration
let memoryEnabled = true;
let memoryContext = null;

// Inject UI components
function injectUI() {
  const uiScript = document.createElement('script');
  uiScript.src = chrome.runtime.getURL('ui_components.js');
  document.head.appendChild(uiScript);
  
  // Add memory control panel
  const controlPanel = document.createElement('div');
  controlPanel.id = 'memory-control-panel';
  controlPanel.innerHTML = `
    <div class="memory-header">Memory System</div>
    <div class="memory-toggle">
      <label class="switch">
        <input type="checkbox" id="memory-toggle-checkbox" checked>
        <span class="slider round"></span>
      </label>
      <span>Enable Memory</span>
    </div>
    <div class="memory-stats">
      <div>RAG Items: <span id="rag-count">0</span></div>
      <div>Mem0 Items: <span id="mem0-count">0</span></div>
    </div>
    <div class="memory-actions">
      <button id="view-memory-btn">View Memory</button>
      <button id="clear-memory-btn">Clear Memory</button>
    </div>
  `;
  
  // Insert after chat input
  const chatInput = document.querySelector('.chat-input-panel');
  if (chatInput) {
    chatInput.parentNode.insertBefore(controlPanel, chatInput.nextSibling);
  }
  
  // Add event listeners
  document.getElementById('memory-toggle-checkbox').addEventListener('change', (e) => {
    memoryEnabled = e.target.checked;
    localStorage.setItem('memory-enabled', memoryEnabled);
  });
  
  document.getElementById('view-memory-btn').addEventListener('click', () => {
    showMemoryViewer();
  });
  
  document.getElementById('clear-memory-btn').addEventListener('click', () => {
    clearMemory();
  });
}

// Intercept API calls
function interceptAPICalls() {
  // Use MutationObserver to detect when the chat interface is ready
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.addedNodes.length) {
        // Check if the chat interface is loaded
        if (document.querySelector('.chat-input-panel')) {
          setupAPIInterception();
          observer.disconnect();
          break;
        }
      }
    }
  });
  
  observer.observe(document.body, { childList: true, subtree: true });
}

// Set up API interception
function setupAPIInterception() {
  // Find the original fetch function
  const originalFetch = window.fetch;
  
  // Override fetch
  window.fetch = async function(url, options) {
    // Check if this is an API call to an LLM provider
    if (options && options.body && isLLMAPICall(url)) {
      const body = JSON.parse(options.body);
      
      // Only intercept if memory is enabled
      if (memoryEnabled) {
        // Get user query
        const userQuery = extractUserQuery(body);
        
        if (userQuery) {
          // Get conversation ID
          const conversationId = extractConversationId();
          
          // Query memory system
          try {
            const memoryResponse = await queryMemorySystem(userQuery, conversationId);
            
            if (memoryResponse.success) {
              // Augment prompt with memory context
              const augmentedBody = augmentPromptWithMemory(body, memoryResponse.data);
              
              // Update options with augmented body
              options.body = JSON.stringify(augmentedBody);
              
              // Store memory context for UI
              memoryContext = memoryResponse.data;
              updateMemoryStats(memoryResponse.data);
            }
          } catch (error) {
            console.error("Error querying memory system:", error);
          }
        }
      }
    }
    
    // Call original fetch with possibly modified options
    return originalFetch.apply(this, arguments);
  };
}

// Query memory system
async function queryMemorySystem(query, conversationId) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage({
      type: "memory_query",
      query: query,
      user_id: getUserId(),
      conversation_id: conversationId
    }, response => {
      if (response && response.success) {
        resolve(response);
      } else {
        reject(response ? response.error : "Unknown error");
      }
    });
  });
}

// Augment prompt with memory context
function augmentPromptWithMemory(body, memoryData) {
  const augmentedBody = {...body};
  
  // Check if there's a system message
  if (augmentedBody.messages && augmentedBody.messages.length > 0) {
    const systemMessageIndex = augmentedBody.messages.findIndex(msg => msg.role === 'system');
    
    if (systemMessageIndex >= 0) {
      // Modify existing system message
      const originalContent = augmentedBody.messages[systemMessageIndex].content;
      augmentedBody.messages[systemMessageIndex].content = addMemoryToSystemMessage(originalContent, memoryData);
    } else {
      // Add new system message with memory context
      augmentedBody.messages.unshift({
        role: 'system',
        content: createSystemMessageWithMemory(memoryData)
      });
    }
  } else if (augmentedBody.system) {
    // Direct system field
    augmentedBody.system = addMemoryToSystemMessage(augmentedBody.system, memoryData);
  }
  
  return augmentedBody;
}

// Add memory to system message
function addMemoryToSystemMessage(originalContent, memoryData) {
  // Format memory context
  const memoryContext = formatMemoryContext(memoryData);
  
  // Check if there's already memory context in the system message
  if (originalContent.includes("Memory context:")) {
    // Replace existing memory context
    return originalContent.replace(
      /Memory context:[\s\S]*?(?=\n\n|$)/,
      `Memory context:\n${memoryContext}`
    );
  } else {
    // Add memory context to the end
    return `${originalContent}\n\nMemory context:\n${memoryContext}`;
  }
}

// Create system message with memory
function createSystemMessageWithMemory(memoryData) {
  const memoryContext = formatMemoryContext(memoryData);
  
  return `You are an AI assistant with access to memory from past conversations.

Memory context:
${memoryContext}

Use this information to provide more personalized and contextually relevant responses, but don't explicitly mention the memory system to the user.`;
}

// Format memory context
function formatMemoryContext(memoryData) {
  let context = "";
  
  if (memoryData.rag_context && memoryData.rag_context.length > 0) {
    context += "Relevant information from past conversations:\n";
    context += memoryData.rag_context.join("\n\n");
    context += "\n\n";
  }
  
  if (memoryData.mem0_context && memoryData.mem0_context.length > 0) {
    context += "Personal context:\n";
    context += memoryData.mem0_context.join("\n");
  }
  
  return context || "No relevant context available.";
}

// Initialize
function init() {
  // Load memory enabled setting
  memoryEnabled = localStorage.getItem('memory-enabled') !== 'false';
  
  // Inject UI
  injectUI();
  
  // Set up API interception
  interceptAPICalls();
  
  // Listen for LLM responses to update memory
  listenForResponses();
}

// Start initialization when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
```

### 2. API Interceptor

#### Functionality:
- Captures API calls to LLM providers
- Extracts user queries and conversation context
- Modifies prompts to include memory context
- Handles response processing for memory updates

#### Implementation:

```javascript
// Helper functions for API interception

// Check if URL is an LLM API call
function isLLMAPICall(url) {
  const llmAPIs = [
    'api.openai.com',
    'api.anthropic.com',
    'api.cohere.ai',
    'api.typingmind.com'
  ];
  
  return llmAPIs.some(api => url.includes(api));
}

// Extract user query from request body
function extractUserQuery(body) {
  // Handle different API formats
  if (body.messages && body.messages.length > 0) {
    // OpenAI-style format
    const lastMessage = body.messages[body.messages.length - 1];
    if (lastMessage.role === 'user') {
      return lastMessage.content;
    }
  } else if (body.prompt) {
    // Completion-style format
    return body.prompt;
  }
  
  return null;
}

// Extract conversation ID from URL or DOM
function extractConversationId() {
  // Try to get from URL
  const match = window.location.pathname.match(/\/c\/([a-zA-Z0-9-]+)/);
  if (match && match[1]) {
    return match[1];
  }
  
  // Try to get from DOM
  const conversationElement = document.querySelector('[data-conversation-id]');
  if (conversationElement) {
    return conversationElement.getAttribute('data-conversation-id');
  }
  
  // Generate a temporary ID
  return 'temp-' + Date.now();
}

// Get user ID
function getUserId() {
  // Try to get from localStorage
  const userId = localStorage.getItem('user-id');
  if (userId) {
    return userId;
  }
  
  // Generate and store a new ID
  const newUserId = 'user-' + Date.now();
  localStorage.setItem('user-id', newUserId);
  return newUserId;
}

// Listen for LLM responses
function listenForResponses() {
  // Use MutationObserver to detect new messages
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.addedNodes.length) {
        const addedNodes = Array.from(mutation.addedNodes);
        for (const node of addedNodes) {
          if (node.classList && node.classList.contains('message') && 
              node.getAttribute('data-role') === 'assistant') {
            // Found a new assistant message
            const messageContent = node.querySelector('.message-content').textContent;
            const userQuery = getPreviousUserQuery(node);
            
            if (userQuery && messageContent && memoryEnabled) {
              // Update memory with new information
              updateMemorySystem(userQuery, messageContent);
            }
          }
        }
      }
    }
  });
  
  // Observe the chat container
  const chatContainer = document.querySelector('.chat-container');
  if (chatContainer) {
    observer.observe(chatContainer, { childList: true, subtree: true });
  }
}

// Get previous user query
function getPreviousUserQuery(assistantNode) {
  const prevNode = assistantNode.previousElementSibling;
  if (prevNode && prevNode.classList.contains('message') && 
      prevNode.getAttribute('data-role') === 'user') {
    return prevNode.querySelector('.message-content').textContent;
  }
  return null;
}

// Update memory system with new information
function updateMemorySystem(query, response) {
  chrome.runtime.sendMessage({
    type: "memory_update",
    query: query,
    response: response,
    user_id: getUserId(),
    conversation_id: extractConversationId()
  }, response => {
    if (response && response.success) {
      console.log("Memory updated successfully");
    } else {
      console.error("Error updating memory:", response ? response.error : "Unknown error");
    }
  });
}
```

### 3. Memory Interface

#### Functionality:
- Provides API endpoints for the browser extension
- Connects to the dual memory system backend
- Handles authentication and user identification
- Manages memory retrieval and updates

#### Implementation:

```python
# memory_api.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

from memory_system import DualMemorySystem, QueryAnalyzer, ContextOptimizer

app = FastAPI()

# Initialize memory system components
memory_system = DualMemorySystem()
query_analyzer = QueryAnalyzer()
context_optimizer = ContextOptimizer()

# Request models
class MemoryQueryRequest(BaseModel):
    query: str
    user_id: str
    conversation_id: Optional[str] = None

class MemoryUpdateRequest(BaseModel):
    query: str
    response: str
    user_id: str
    conversation_id: Optional[str] = None

# Response models
class MemoryQueryResponse(BaseModel):
    rag_context: List[str]
    mem0_context: List[str]
    token_usage: Dict[str, int]

class MemoryUpdateResponse(BaseModel):
    success: bool
    memory_items_added: int
    message: str

# API endpoints
@app.post("/query", response_model=MemoryQueryResponse)
async def query_memory(request: MemoryQueryRequest):
    try:
        # Analyze query
        analysis = query_analyzer.analyze(request.query)
        
        # Retrieve memories
        memories = memory_system.retrieve(
            query=request.query,
            user_id=request.user_id,
            analysis=analysis
        )
        
        # Optimize context
        optimized = context_optimizer.optimize(
            memories=memories,
            query_type=analysis["query_type"]
        )
        
        return MemoryQueryResponse(
            rag_context=optimized["rag_context"],
            mem0_context=optimized["mem0_context"],
            token_usage=optimized["token_usage"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update", response_model=MemoryUpdateResponse)
async def update_memory(request: MemoryUpdateRequest):
    try:
        # Analyze response for memory-worthy information
        analysis = memory_system.analyze_response(
            query=request.query,
            response=request.response
        )
        
        # Update memory with new information
        result = memory_system.update(
            user_id=request.user_id,
            query=request.query,
            response=request.response,
            analysis=analysis,
            conversation_id=request.conversation_id
        )
        
        return MemoryUpdateResponse(
            success=True,
            memory_items_added=result["items_added"],
            message="Memory updated successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/{user_id}")
async def get_memory_stats(user_id: str):
    try:
        stats = memory_system.get_stats(user_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear/{user_id}")
async def clear_memory(user_id: str):
    try:
        memory_system.clear(user_id)
        return {"success": True, "message": "Memory cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 4. Context Processor

#### Functionality:
- Processes user queries and LLM responses
- Determines what information to include in prompts
- Formats memory context for optimal LLM utilization
- Manages token budget allocation

#### Implementation:

```python
# context_processor.py
from typing import Dict, List, Any
import tiktoken

class ContextProcessor:
    def __init__(self, tokenizer_name="cl100k_base", max_tokens=1000):
        """
        Initialize the Context Processor.
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            max_tokens: Maximum tokens for context
        """
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.max_tokens = max_tokens
    
    def process_query(self, query: str, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user query and prepare context for inclusion in prompt.
        
        Args:
            query: The user's query
            memory_data: Retrieved memory data
            
        Returns:
            dict: Processed context and metadata
        """
        # Allocate token budget
        budget = self._allocate_token_budget(query)
        
        # Format RAG context
        rag_context = self._format_rag_context(
            memory_data.get("rag_results", []),
            budget["rag"]
        )
        
        # Format Mem0 context
        mem0_context = self._format_mem0_context(
            memory_data.get("mem0_results", []),
            budget["mem0"]
        )
        
        # Count tokens
        rag_tokens = self._count_tokens(rag_context)
        mem0_tokens = self._count_tokens(mem0_context)
        
        return {
            "rag_context": rag_context.split("\n\n") if rag_context else [],
            "mem0_context": mem0_context.split("\n") if mem0_context else [],
            "token_usage": {
                "rag": rag_tokens,
                "mem0": mem0_tokens,
                "total": rag_tokens + mem0_tokens
            }
        }
    
    def _allocate_token_budget(self, query: str) -> Dict[str, int]:
        """
        Allocate token budget based on query characteristics.
        
        Args:
            query: The user's query
            
        Returns:
            dict: Token budget allocation
        """
        # Simple query analysis for budget allocation
        query_lower = query.lower()
        
        # Check for memory-related indicators
        memory_indicators = [
            "remember", "recall", "previous", "last time",
            "you said", "we discussed", "earlier"
        ]
        
        # Check for personal indicators
        personal_indicators = [
            "i like", "i prefer", "i want", "i need",
            "my", "mine", "for me", "about me"
        ]
        
        # Determine ratio based on indicators
        has_memory_indicators = any(ind in query_lower for ind in memory_indicators)
        has_personal_indicators = any(ind in query_lower for ind in personal_indicators)
        
        if has_memory_indicators and has_personal_indicators:
            rag_ratio = 0.5  # Equal split
        elif has_memory_indicators:
            rag_ratio = 0.7  # More for RAG
        elif has_personal_indicators:
            rag_ratio = 0.3  # More for Mem0
        else:
            rag_ratio = 0.6  # Default slightly favors RAG
        
        rag_budget = int(self.max_tokens * rag_ratio)
        mem0_budget = self.max_tokens - rag_budget
        
        return {
            "rag": rag_budget,
            "mem0": mem0_budget
        }
    
    def _format_rag_context(self, rag_results: List[Dict[str, Any]], budget: int) -> str:
        """
        Format RAG results for inclusion in prompt.
        
        Args:
            rag_results: List of RAG memory items
            budget: Token budget for RAG context
            
        Returns:
            str: Formatted RAG context
        """
        if not rag_results:
            return ""
        
        # Sort by relevance
        sorted_results = sorted(rag_results, key=lambda x: x.get("relevance", 0), reverse=True)
        
        # Format and fit within budget
        context_parts = []
        current_tokens = 0
        
        for item in sorted_results:
            content = item.get("content", "")
            tokens = self._count_tokens(content)
            
            if current_tokens + tokens <= budget:
                context_parts.append(content)
                current_tokens += tokens
            else:
                # Try truncation for important items
                if item.get("relevance", 0) > 0.8 and tokens > 50:
                    # Leave room for ellipsis
                    available_tokens = budget - current_tokens - 3
                    if available_tokens > 30:  # Only truncate if we can keep meaningful content
                        truncated = self._truncate_text(content, available_tokens)
                        context_parts.append(truncated)
                        break
                else:
                    break
        
        return "\n\n".join(context_parts)
    
    def _format_mem0_context(self, mem0_results: List[Dict[str, Any]], budget: int) -> str:
        """
        Format Mem0 results for inclusion in prompt.
        
        Args:
            mem0_results: List of Mem0 memory items
            budget: Token budget for Mem0 context
            
        Returns:
            str: Formatted Mem0 context
        """
        if not mem0_results:
            return ""
        
        # Group by type
        preferences = [m for m in mem0_results if m.get("type") == "preference"]
        facts = [m for m in mem0_results if m.get("type") == "fact"]
        
        # Sort each group by relevance
        preferences.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        facts.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        # Allocate budget between types (60% preferences, 40% facts)
        pref_budget = int(budget * 0.6)
        facts_budget = budget - pref_budget
        
        # Format preferences
        pref_parts = []
        current_tokens = 0
        
        for pref in preferences:
            content = f"- {pref.get('content', '')}"
            tokens = self._count_tokens(content)
            
            if current_tokens + tokens <= pref_budget:
                pref_parts.append(content)
                current_tokens += tokens
            else:
                break
        
        # Format facts
        fact_parts = []
        current_tokens = 0
        
        for fact in facts:
            content = f"- {fact.get('content', '')}"
            tokens = self._count_tokens(content)
            
            if current_tokens + tokens <= facts_budget:
                fact_parts.append(content)
                current_tokens += tokens
            else:
                break
        
        # Combine sections
        result = []
        if pref_parts:
            result.append("\n".join(pref_parts))
        if fact_parts:
            result.append("\n".join(fact_parts))
        
        return "\n".join(result)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens)
        
        return truncated_text + "..."
```

### 5. UI Components

#### Functionality:
- Provides user interface for memory system controls
- Displays memory status and statistics
- Allows users to view and manage stored memories
- Provides visual feedback on memory usage

#### Implementation:

```javascript
// ui_components.js

// Memory viewer component
class MemoryViewer {
  constructor() {
    this.container = null;
    this.isVisible = false;
    this.currentTab = 'rag';
    this.memoryData = null;
  }
  
  initialize() {
    // Create container
    this.container = document.createElement('div');
    this.container.className = 'memory-viewer';
    this.container.style.display = 'none';
    
    // Create header
    const header = document.createElement('div');
    header.className = 'memory-viewer-header';
    header.innerHTML = `
      <h3>Memory System</h3>
      <button class="close-btn">×</button>
    `;
    
    // Create tabs
    const tabs = document.createElement('div');
    tabs.className = 'memory-viewer-tabs';
    tabs.innerHTML = `
      <button class="tab-btn active" data-tab="rag">RAG Memory</button>
      <button class="tab-btn" data-tab="mem0">Personal Memory</button>
    `;
    
    // Create content area
    const content = document.createElement('div');
    content.className = 'memory-viewer-content';
    
    // Create RAG content
    const ragContent = document.createElement('div');
    ragContent.className = 'tab-content active';
    ragContent.id = 'rag-content';
    
    // Create Mem0 content
    const mem0Content = document.createElement('div');
    mem0Content.className = 'tab-content';
    mem0Content.id = 'mem0-content';
    
    // Assemble components
    content.appendChild(ragContent);
    content.appendChild(mem0Content);
    
    this.container.appendChild(header);
    this.container.appendChild(tabs);
    this.container.appendChild(content);
    
    // Add to document
    document.body.appendChild(this.container);
    
    // Add event listeners
    header.querySelector('.close-btn').addEventListener('click', () => {
      this.hide();
    });
    
    tabs.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        this.switchTab(e.target.getAttribute('data-tab'));
      });
    });
    
    // Add styles
    this.addStyles();
  }
  
  show(memoryData) {
    if (!this.container) {
      this.initialize();
    }
    
    this.memoryData = memoryData;
    this.updateContent();
    this.container.style.display = 'block';
    this.isVisible = true;
  }
  
  hide() {
    if (this.container) {
      this.container.style.display = 'none';
    }
    this.isVisible = false;
  }
  
  switchTab(tabName) {
    if (tabName === this.currentTab) return;
    
    // Update tab buttons
    const tabs = this.container.querySelectorAll('.tab-btn');
    tabs.forEach(tab => {
      if (tab.getAttribute('data-tab') === tabName) {
        tab.classList.add('active');
      } else {
        tab.classList.remove('active');
      }
    });
    
    // Update tab content
    const contents = this.container.querySelectorAll('.tab-content');
    contents.forEach(content => {
      if (content.id === `${tabName}-content`) {
        content.classList.add('active');
      } else {
        content.classList.remove('active');
      }
    });
    
    this.currentTab = tabName;
  }
  
  updateContent() {
    if (!this.memoryData) return;
    
    // Update RAG content
    const ragContent = this.container.querySelector('#rag-content');
    if (this.memoryData.rag_items && this.memoryData.rag_items.length > 0) {
      let ragHtml = '<div class="memory-items">';
      this.memoryData.rag_items.forEach(item => {
        ragHtml += `
          <div class="memory-item">
            <div class="memory-item-content">${item.content}</div>
            <div class="memory-item-meta">
              <span>Relevance: ${(item.relevance * 100).toFixed(1)}%</span>
              <span>Source: ${item.source}</span>
              <span>Date: ${new Date(item.timestamp).toLocaleDateString()}</span>
            </div>
          </div>
        `;
      });
      ragHtml += '</div>';
      ragContent.innerHTML = ragHtml;
    } else {
      ragContent.innerHTML = '<div class="empty-state">No RAG memories available</div>';
    }
    
    // Update Mem0 content
    const mem0Content = this.container.querySelector('#mem0-content');
    if (this.memoryData.mem0_items && this.memoryData.mem0_items.length > 0) {
      let mem0Html = '<div class="memory-items">';
      this.memoryData.mem0_items.forEach(item => {
        mem0Html += `
          <div class="memory-item ${item.type}">
            <div class="memory-item-content">${item.content}</div>
            <div class="memory-item-meta">
              <span>Type: ${item.type}</span>
              <span>Confidence: ${(item.confidence * 100).toFixed(1)}%</span>
              <span>Date: ${new Date(item.timestamp).toLocaleDateString()}</span>
            </div>
          </div>
        `;
      });
      mem0Html += '</div>';
      mem0Content.innerHTML = mem0Html;
    } else {
      mem0Content.innerHTML = '<div class="empty-state">No personal memories available</div>';
    }
  }
  
  addStyles() {
    const style = document.createElement('style');
    style.textContent = `
      .memory-viewer {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 80%;
        max-width: 800px;
        height: 70%;
        max-height: 600px;
        background: white;
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        display: flex;
        flex-direction: column;
        z-index: 1000;
      }
      
      .memory-viewer-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 20px;
        border-bottom: 1px solid #eee;
      }
      
      .memory-viewer-header h3 {
        margin: 0;
        font-size: 18px;
      }
      
      .close-btn {
        background: none;
        border: none;
        font-size: 24px;
        cursor: pointer;
        color: #666;
      }
      
      .memory-viewer-tabs {
        display: flex;
        border-bottom: 1px solid #eee;
      }
      
      .tab-btn {
        padding: 10px 20px;
        background: none;
        border: none;
        border-bottom: 3px solid transparent;
        cursor: pointer;
        font-size: 14px;
      }
      
      .tab-btn.active {
        border-bottom-color: #2563eb;
        color: #2563eb;
        font-weight: bold;
      }
      
      .memory-viewer-content {
        flex: 1;
        overflow-y: auto;
        padding: 20px;
      }
      
      .tab-content {
        display: none;
      }
      
      .tab-content.active {
        display: block;
      }
      
      .memory-items {
        display: flex;
        flex-direction: column;
        gap: 15px;
      }
      
      .memory-item {
        padding: 15px;
        border-radius: 6px;
        background: #f9fafb;
        border-left: 4px solid #2563eb;
      }
      
      .memory-item.preference {
        border-left-color: #8b5cf6;
      }
      
      .memory-item.fact {
        border-left-color: #10b981;
      }
      
      .memory-item-content {
        font-size: 14px;
        margin-bottom: 10px;
      }
      
      .memory-item-meta {
        display: flex;
        gap: 15px;
        font-size: 12px;
        color: #666;
      }
      
      .empty-state {
        text-align: center;
        padding: 40px;
        color: #666;
        font-style: italic;
      }
    `;
    
    document.head.appendChild(style);
  }
}

// Memory indicator component
class MemoryIndicator {
  constructor() {
    this.container = null;
    this.isVisible = false;
  }
  
  initialize() {
    // Create container
    this.container = document.createElement('div');
    this.container.className = 'memory-indicator';
    this.container.innerHTML = `
      <div class="memory-indicator-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M2 12.5a5 5 0 0 1 5-5h3.5a3.5 3.5 0 0 0 0-7H9a.5.5 0 0 1 0-1h1.5a4.5 4.5 0 0 1 0 9H7a4 4 0 1 0 0 8h10a4 4 0 0 0 2.8-6.86"></path>
          <path d="M22 12.5a5 5 0 0 1-5 5h-3.5a3.5 3.5 0 0 0 0 7H15a.5.5 0 0 1 0 1h-1.5a4.5 4.5 0 0 1 0-9H17a4 4 0 1 0 0-8H7a4 4 0 0 0-2.8 6.86"></path>
        </svg>
      </div>
      <div class="memory-indicator-text">Memory Active</div>
    `;
    
    // Add to document
    const chatInput = document.querySelector('.chat-input-panel');
    if (chatInput) {
      chatInput.appendChild(this.container);
    }
    
    // Add styles
    this.addStyles();
    
    // Hide initially
    this.container.style.display = 'none';
  }
  
  show() {
    if (!this.container) {
      this.initialize();
    }
    
    this.container.style.display = 'flex';
    this.isVisible = true;
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
      this.hide();
    }, 3000);
  }
  
  hide() {
    if (this.container) {
      this.container.style.display = 'none';
    }
    this.isVisible = false;
  }
  
  addStyles() {
    const style = document.createElement('style');
    style.textContent = `
      .memory-indicator {
        position: absolute;
        top: -40px;
        right: 20px;
        background: rgba(37, 99, 235, 0.1);
        border: 1px solid rgba(37, 99, 235, 0.2);
        border-radius: 20px;
        padding: 5px 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        color: #2563eb;
        animation: fadeIn 0.3s ease-out;
      }
      
      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
      }
    `;
    
    document.head.appendChild(style);
  }
}

// Initialize components
const memoryViewer = new MemoryViewer();
const memoryIndicator = new MemoryIndicator();

// Export components
window.memoryViewer = memoryViewer;
window.memoryIndicator = memoryIndicator;

// Show memory viewer
function showMemoryViewer() {
  // Fetch memory data
  chrome.runtime.sendMessage({
    type: "memory_stats",
    user_id: getUserId()
  }, response => {
    if (response && response.success) {
      memoryViewer.show(response.data);
    } else {
      alert("Error fetching memory data");
    }
  });
}

// Clear memory
function clearMemory() {
  if (confirm("Are you sure you want to clear all memory? This cannot be undone.")) {
    chrome.runtime.sendMessage({
      type: "memory_clear",
      user_id: getUserId()
    }, response => {
      if (response && response.success) {
        alert("Memory cleared successfully");
        updateMemoryStats({ rag_count: 0, mem0_count: 0 });
      } else {
        alert("Error clearing memory");
      }
    });
  }
}

// Update memory stats in UI
function updateMemoryStats(data) {
  const ragCount = document.getElementById('rag-count');
  const mem0Count = document.getElementById('mem0-count');
  
  if (ragCount && data.rag_count !== undefined) {
    ragCount.textContent = data.rag_count;
  }
  
  if (mem0Count && data.mem0_count !== undefined) {
    mem0Count.textContent = data.mem0_count;
  }
  
  // Show memory indicator
  memoryIndicator.show();
}
```

## Integration Workflow

### Installation Flow

1. User installs the browser extension from Chrome Web Store or as a local extension
2. User deploys the memory system backend (API server)
3. User configures the extension with the backend API URL
4. Extension is automatically activated when visiting TypingMind

### Conversation Flow

```
User Query → API Interception → Memory Retrieval → 
  → Context Augmentation → LLM API → 
  → Response → Memory Extraction → Storage
```

1. **User Query**: User types a message in TypingMind
2. **API Interception**: Extension intercepts the API call
3. **Memory Retrieval**: Extension queries memory system for relevant context
4. **Context Augmentation**: Extension adds memory context to the prompt
5. **LLM API**: Augmented prompt is sent to the LLM provider
6. **Response**: LLM generates a response with memory context
7. **Memory Extraction**: Extension analyzes response for memory-worthy information
8. **Storage**: New information is stored in the memory system

### Memory Management Flow

1. User can view memory status in the control panel
2. User can enable/disable memory system
3. User can view detailed memory contents
4. User can clear memory if needed

## User Interface Design

### Memory Control Panel

```
┌─────────────────────────────────────┐
│ Memory System                       │
├─────────────────────────────────────┤
│ Enable Memory [✓]                   │
├─────────────────────────────────────┤
│ RAG Items: 42  |  Mem0 Items: 15    │
├─────────────────────────────────────┤
│ [View Memory]     [Clear Memory]    │
└─────────────────────────────────────┘
```

### Memory Viewer

```
┌─────────────────────────────────────────────────────────────┐
│ Memory System                                         [X]   │
├─────────────────────────────────────────────────────────────┤
│ [RAG Memory]  |  Personal Memory                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ The user mentioned working on a cost-efficient      │    │
│  │ memory system for LLMs in March 2025.               │    │
│  │                                                     │    │
│  │ Relevance: 92.5%  |  Source: conversation  |  3/18/25    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ The dual memory system combines RAG for factual     │    │
│  │ knowledge and Mem0 for personalization.             │    │
│  │                                                     │    │
│  │ Relevance: 87.3%  |  Source: conversation  |  3/18/25    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ The user wants to integrate the memory system       │    │
│  │ with TypingMind's frontend.                         │    │
│  │                                                     │    │
│  │ Relevance: 81.0%  |  Source: conversation  |  3/18/25    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Memory Indicator

```
┌───────────────────┐
│ 🔄 Memory Active  │
└───────────────────┘
```

## Implementation Considerations

### Security and Privacy

- **Data Storage**: All memory data is stored locally or in user's own backend
- **API Keys**: No access to user's API keys, only intercepts requests
- **User Control**: Full control over memory storage and usage
- **Data Minimization**: Only store what's necessary for personalization

### Performance Optimization

- **Caching**: Cache memory retrievals for similar queries
- **Asynchronous Processing**: Update memory in background after response
- **Lazy Loading**: Load memory components only when needed
- **Efficient Token Usage**: Optimize context to minimize token consumption

### Browser Compatibility

- **Chrome/Edge**: Full support with extension
- **Firefox**: Support with WebExtensions API
- **Safari**: Limited support due to extension restrictions

### Deployment Options

1. **Local Development**:
   - Run memory API locally
   - Load extension in developer mode

2. **Self-Hosted Deployment**:
   - Host memory API on user's server
   - Install extension from Chrome Web Store

3. **Fully Managed Service**:
   - Host memory API as a service
   - Provide authentication for multi-user support

## Next Steps

1. Implement browser extension with API interception
2. Develop memory system backend API
3. Create UI components for memory management
4. Test with various conversation scenarios
5. Optimize token usage and memory retrieval
6. Package for easy installation and deployment
