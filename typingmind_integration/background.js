/**
 * TypingMind Memory System Extension - Background Script
 * 
 * This background script handles the extension's background processes,
 * including API communication and configuration management.
 */

// Default configuration
const DEFAULT_CONFIG = {
  contextApiUrl: 'https://context-api.example.com',
  ragApiUrl: 'https://rag-api.example.com',
  mem0ApiUrl: 'https://mem0-api.example.com',
  tokenBudget: 1000,
  enableRag: true,
  enableMem0: true,
  debug: false,
  logApiCalls: false
};

// Initialize configuration
chrome.runtime.onInstalled.addListener(() => {
  // Set default configuration
  chrome.storage.sync.get('memorySystemConfig', (data) => {
    if (!data.memorySystemConfig) {
      chrome.storage.sync.set({ memorySystemConfig: DEFAULT_CONFIG });
      console.log('Memory System: Default configuration set');
    }
  });
});

// Listen for messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'getConfig') {
    // Return configuration
    chrome.storage.sync.get('memorySystemConfig', (data) => {
      sendResponse({ config: data.memorySystemConfig || DEFAULT_CONFIG });
    });
    return true; // Required for async response
  } else if (request.type === 'setConfig') {
    // Update configuration
    chrome.storage.sync.set({ memorySystemConfig: request.config });
    sendResponse({ success: true });
    return true;
  } else if (request.type === 'apiRequest') {
    // Handle API request
    handleApiRequest(request.url, request.method, request.body)
      .then(response => sendResponse({ success: true, data: response }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true;
  }
});

/**
 * Handle API request
 * 
 * @param {string} url - API URL
 * @param {string} method - HTTP method
 * @param {object} body - Request body
 * @returns {Promise<object>} - API response
 */
async function handleApiRequest(url, method, body) {
  try {
    // Get configuration
    const config = await new Promise(resolve => {
      chrome.storage.sync.get('memorySystemConfig', (data) => {
        resolve(data.memorySystemConfig || DEFAULT_CONFIG);
      });
    });
    
    // Log API call if enabled
    if (config.logApiCalls) {
      console.log(`Memory System API Call: ${method} ${url}`, body);
    }
    
    // Make API request
    const response = await fetch(url, {
      method: method,
      headers: {
        'Content-Type': 'application/json'
      },
      body: body ? JSON.stringify(body) : undefined
    });
    
    // Check response
    if (!response.ok) {
      throw new Error(`API returned ${response.status}`);
    }
    
    // Parse response
    const data = await response.json();
    
    // Log response if enabled
    if (config.logApiCalls) {
      console.log(`Memory System API Response:`, data);
    }
    
    return data;
  } catch (error) {
    console.error('Memory System API Error:', error);
    throw error;
  }
}
