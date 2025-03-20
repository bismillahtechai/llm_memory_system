/**
 * TypingMind Memory System Extension
 * 
 * This extension integrates the LLM memory system with TypingMind,
 * providing persistent memory and context-aware responses.
 */

// Configuration
const CONFIG = {
  // API endpoints
  contextApiUrl: 'https://context-api.example.com',
  ragApiUrl: 'https://rag-api.example.com',
  mem0ApiUrl: 'https://mem0-api.example.com',
  
  // Memory system settings
  tokenBudget: 1000,
  enableRag: true,
  enableMem0: true,
  
  // Debug settings
  debug: false,
  logApiCalls: false
};

// Main extension class
class MemorySystemExtension {
  constructor() {
    this.initialized = false;
    this.conversationId = null;
    this.messageQueue = [];
    this.processingQueue = false;
    
    // Bind methods
    this.initialize = this.initialize.bind(this);
    this.interceptMessages = this.interceptMessages.bind(this);
    this.processMessage = this.processMessage.bind(this);
    this.augmentPrompt = this.augmentPrompt.bind(this);
    this.extractMemories = this.extractMemories.bind(this);
    this.processMessageQueue = this.processMessageQueue.bind(this);
    this.log = this.log.bind(this);
  }
  
  /**
   * Initialize the extension
   */
  async initialize() {
    if (this.initialized) return;
    
    this.log('Initializing TypingMind Memory System Extension');
    
    // Get conversation ID from URL
    const urlParams = new URLSearchParams(window.location.search);
    this.conversationId = urlParams.get('id') || null;
    
    // Set up message interception
    this.setupMessageInterception();
    
    // Add UI elements
    this.addUI();
    
    this.initialized = true;
    this.log('Extension initialized');
    
    // Process any queued messages
    this.processMessageQueue();
  }
  
  /**
   * Set up interception of messages
   */
  setupMessageInterception() {
    // Intercept fetch requests
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
      const [url, options] = args;
      
      // Check if this is a message to the LLM API
      if (url.includes('/api/chat') && options.method === 'POST') {
        try {
          const body = JSON.parse(options.body);
          
          // Intercept the message
          const augmentedBody = await this.interceptMessages(body);
          
          // Replace the body
          const newOptions = {
            ...options,
            body: JSON.stringify(augmentedBody)
          };
          
          return originalFetch(url, newOptions);
        } catch (error) {
          this.log('Error intercepting message:', error);
          return originalFetch(...args);
        }
      }
      
      return originalFetch(...args);
    };
    
    // Observe DOM for new messages
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        if (mutation.type === 'childList') {
          const userMessages = document.querySelectorAll('.user-message');
          for (const message of userMessages) {
            // Check if we've processed this message already
            if (!message.dataset.processed) {
              message.dataset.processed = 'true';
              const messageText = message.textContent;
              
              // Queue the message for processing
              this.messageQueue.push({
                text: messageText,
                element: message
              });
              
              // Process the queue
              if (!this.processingQueue) {
                this.processMessageQueue();
              }
            }
          }
        }
      }
    });
    
    // Start observing
    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  }
  
  /**
   * Process the message queue
   */
  async processMessageQueue() {
    if (this.processingQueue || this.messageQueue.length === 0) return;
    
    this.processingQueue = true;
    
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      await this.processMessage(message);
    }
    
    this.processingQueue = false;
  }
  
  /**
   * Process a message for memory extraction
   */
  async processMessage(message) {
    try {
      // Extract memories from the message
      await this.extractMemories(message.text);
    } catch (error) {
      this.log('Error processing message:', error);
    }
  }
  
  /**
   * Intercept messages to the LLM API
   */
  async interceptMessages(body) {
    this.log('Intercepting message:', body);
    
    // Check if we have messages
    if (!body.messages || body.messages.length === 0) {
      return body;
    }
    
    try {
      // Augment the messages with context
      const augmentedMessages = await this.augmentPrompt(body.messages);
      
      // Return the augmented body
      return {
        ...body,
        messages: augmentedMessages
      };
    } catch (error) {
      this.log('Error augmenting messages:', error);
      return body;
    }
  }
  
  /**
   * Augment a prompt with context
   */
  async augmentPrompt(messages) {
    try {
      // Call the context API to augment the messages
      const response = await fetch(`${CONFIG.contextApiUrl}/augment-messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          messages: messages,
          conversation_id: this.conversationId
        })
      });
      
      if (!response.ok) {
        throw new Error(`Context API returned ${response.status}`);
      }
      
      const data = await response.json();
      
      // Check if context was added
      if (data.context_added) {
        this.log('Context added to prompt, token count:', data.token_count);
        this.updateUI({
          contextAdded: true,
          tokenCount: data.token_count
        });
      } else {
        this.log('No context added to prompt');
        this.updateUI({
          contextAdded: false,
          tokenCount: 0
        });
      }
      
      return data.messages;
    } catch (error) {
      this.log('Error augmenting prompt:', error);
      // Return original messages if there's an error
      return messages;
    }
  }
  
  /**
   * Extract memories from a message
   */
  async extractMemories(messageText) {
    if (!CONFIG.enableMem0) return;
    
    try {
      // Call the Mem0 API to extract memories
      const response = await fetch(`${CONFIG.mem0ApiUrl}/extract`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: {
            content: messageText,
            conversation_id: this.conversationId
          }
        })
      });
      
      if (!response.ok) {
        throw new Error(`Mem0 API returned ${response.status}`);
      }
      
      const data = await response.json();
      
      // Check if memories were extracted
      if (data.memories && data.memories.length > 0) {
        this.log(`Extracted ${data.memories.length} memories`);
        this.updateUI({
          memoriesExtracted: data.memories.length
        });
        
        // Add memories to Mem0
        for (const memory of data.memories) {
          await fetch(`${CONFIG.mem0ApiUrl}/memory`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              content: memory.content,
              memory_type: memory.memory_type,
              source_conversation_id: this.conversationId,
              importance: memory.importance || 0.7
            })
          });
        }
      } else {
        this.log('No memories extracted');
      }
    } catch (error) {
      this.log('Error extracting memories:', error);
    }
  }
  
  /**
   * Add UI elements
   */
  addUI() {
    // Create UI container
    const container = document.createElement('div');
    container.id = 'memory-system-ui';
    container.style.position = 'fixed';
    container.style.bottom = '20px';
    container.style.right = '20px';
    container.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    container.style.color = 'white';
    container.style.padding = '10px';
    container.style.borderRadius = '5px';
    container.style.zIndex = '9999';
    container.style.fontSize = '12px';
    container.style.fontFamily = 'monospace';
    container.style.display = CONFIG.debug ? 'block' : 'none';
    
    // Create status text
    const statusText = document.createElement('div');
    statusText.id = 'memory-system-status';
    statusText.textContent = 'Memory System: Active';
    container.appendChild(statusText);
    
    // Create context info
    const contextInfo = document.createElement('div');
    contextInfo.id = 'memory-system-context-info';
    contextInfo.textContent = 'Context: None';
    container.appendChild(contextInfo);
    
    // Create memory info
    const memoryInfo = document.createElement('div');
    memoryInfo.id = 'memory-system-memory-info';
    memoryInfo.textContent = 'Memories: 0';
    container.appendChild(memoryInfo);
    
    // Create toggle button
    const toggleButton = document.createElement('button');
    toggleButton.textContent = 'Toggle Debug';
    toggleButton.style.marginTop = '10px';
    toggleButton.style.padding = '5px';
    toggleButton.style.backgroundColor = '#333';
    toggleButton.style.color = 'white';
    toggleButton.style.border = 'none';
    toggleButton.style.borderRadius = '3px';
    toggleButton.style.cursor = 'pointer';
    
    toggleButton.addEventListener('click', () => {
      CONFIG.debug = !CONFIG.debug;
      container.style.display = CONFIG.debug ? 'block' : 'none';
    });
    
    container.appendChild(toggleButton);
    
    // Add to body
    document.body.appendChild(container);
    
    // Add settings to TypingMind settings panel if it exists
    this.addSettingsToPanel();
  }
  
  /**
   * Add settings to TypingMind settings panel
   */
  addSettingsToPanel() {
    // Check if settings panel exists
    const settingsInterval = setInterval(() => {
      const settingsPanel = document.querySelector('.settings-panel');
      if (settingsPanel) {
        clearInterval(settingsInterval);
        
        // Create memory system settings section
        const section = document.createElement('div');
        section.className = 'settings-section';
        
        const title = document.createElement('h3');
        title.textContent = 'Memory System';
        section.appendChild(title);
        
        // Create RAG toggle
        const ragToggle = document.createElement('div');
        ragToggle.className = 'settings-item';
        
        const ragLabel = document.createElement('label');
        ragLabel.textContent = 'Enable RAG (Factual Memory)';
        
        const ragCheckbox = document.createElement('input');
        ragCheckbox.type = 'checkbox';
        ragCheckbox.checked = CONFIG.enableRag;
        ragCheckbox.addEventListener('change', (e) => {
          CONFIG.enableRag = e.target.checked;
        });
        
        ragToggle.appendChild(ragLabel);
        ragToggle.appendChild(ragCheckbox);
        section.appendChild(ragToggle);
        
        // Create Mem0 toggle
        const mem0Toggle = document.createElement('div');
        mem0Toggle.className = 'settings-item';
        
        const mem0Label = document.createElement('label');
        mem0Label.textContent = 'Enable Mem0 (Personal Memory)';
        
        const mem0Checkbox = document.createElement('input');
        mem0Checkbox.type = 'checkbox';
        mem0Checkbox.checked = CONFIG.enableMem0;
        mem0Checkbox.addEventListener('change', (e) => {
          CONFIG.enableMem0 = e.target.checked;
        });
        
        mem0Toggle.appendChild(mem0Label);
        mem0Toggle.appendChild(mem0Checkbox);
        section.appendChild(mem0Toggle);
        
        // Create token budget input
        const tokenBudget = document.createElement('div');
        tokenBudget.className = 'settings-item';
        
        const tokenLabel = document.createElement('label');
        tokenLabel.textContent = 'Token Budget';
        
        const tokenInput = document.createElement('input');
        tokenInput.type = 'number';
        tokenInput.min = '100';
        tokenInput.max = '4000';
        tokenInput.step = '100';
        tokenInput.value = CONFIG.tokenBudget;
        tokenInput.addEventListener('change', (e) => {
          CONFIG.tokenBudget = parseInt(e.target.value, 10);
        });
        
        tokenBudget.appendChild(tokenLabel);
        tokenBudget.appendChild(tokenInput);
        section.appendChild(tokenBudget);
        
        // Add section to settings panel
        settingsPanel.appendChild(section);
      }
    }, 1000);
  }
  
  /**
   * Update UI elements
   */
  updateUI(data) {
    if (!CONFIG.debug) return;
    
    const contextInfo = document.getElementById('memory-system-context-info');
    const memoryInfo = document.getElementById('memory-system-memory-info');
    
    if (data.contextAdded) {
      contextInfo.textContent = `Context: Added (${data.tokenCount} tokens)`;
    }
    
    if (data.memoriesExtracted) {
      memoryInfo.textContent = `Memories: ${data.memoriesExtracted} extracted`;
    }
  }
  
  /**
   * Log messages if debug is enabled
   */
  log(...args) {
    if (CONFIG.debug) {
      console.log('[Memory System]', ...args);
    }
  }
}

// Initialize the extension
const memorySystem = new MemorySystemExtension();

// Wait for page to load
window.addEventListener('load', () => {
  // Initialize after a short delay to ensure TypingMind is loaded
  setTimeout(() => {
    memorySystem.initialize();
  }, 1000);
});
