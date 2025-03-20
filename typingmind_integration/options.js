/**
 * TypingMind Memory System Extension - Options Script
 * 
 * This script handles the options page functionality for the
 * TypingMind Memory System Extension.
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

// DOM elements
const contextApiUrlInput = document.getElementById('contextApiUrl');
const ragApiUrlInput = document.getElementById('ragApiUrl');
const mem0ApiUrlInput = document.getElementById('mem0ApiUrl');
const tokenBudgetInput = document.getElementById('tokenBudget');
const enableRagCheckbox = document.getElementById('enableRag');
const enableMem0Checkbox = document.getElementById('enableMem0');
const debugCheckbox = document.getElementById('debug');
const logApiCallsCheckbox = document.getElementById('logApiCalls');
const saveButton = document.getElementById('saveButton');
const resetButton = document.getElementById('resetButton');
const statusElement = document.getElementById('status');

// Load configuration
function loadConfig() {
  chrome.storage.sync.get('memorySystemConfig', (data) => {
    const config = data.memorySystemConfig || DEFAULT_CONFIG;
    
    // Set input values
    contextApiUrlInput.value = config.contextApiUrl || '';
    ragApiUrlInput.value = config.ragApiUrl || '';
    mem0ApiUrlInput.value = config.mem0ApiUrl || '';
    tokenBudgetInput.value = config.tokenBudget || 1000;
    enableRagCheckbox.checked = config.enableRag !== false;
    enableMem0Checkbox.checked = config.enableMem0 !== false;
    debugCheckbox.checked = config.debug === true;
    logApiCallsCheckbox.checked = config.logApiCalls === true;
  });
}

// Save configuration
function saveConfig() {
  const config = {
    contextApiUrl: contextApiUrlInput.value,
    ragApiUrl: ragApiUrlInput.value,
    mem0ApiUrl: mem0ApiUrlInput.value,
    tokenBudget: parseInt(tokenBudgetInput.value, 10) || 1000,
    enableRag: enableRagCheckbox.checked,
    enableMem0: enableMem0Checkbox.checked,
    debug: debugCheckbox.checked,
    logApiCalls: logApiCallsCheckbox.checked
  };
  
  chrome.storage.sync.set({ memorySystemConfig: config }, () => {
    // Show success message
    showStatus('Settings saved successfully!', 'success');
  });
}

// Reset configuration
function resetConfig() {
  chrome.storage.sync.set({ memorySystemConfig: DEFAULT_CONFIG }, () => {
    // Reload configuration
    loadConfig();
    
    // Show success message
    showStatus('Settings reset to defaults!', 'success');
  });
}

// Show status message
function showStatus(message, type) {
  statusElement.textContent = message;
  statusElement.className = `status ${type}`;
  statusElement.style.display = 'block';
  
  // Hide after 3 seconds
  setTimeout(() => {
    statusElement.style.display = 'none';
  }, 3000);
}

// Test API connection
function testApiConnection() {
  const config = {
    contextApiUrl: contextApiUrlInput.value,
    ragApiUrl: ragApiUrlInput.value,
    mem0ApiUrl: mem0ApiUrlInput.value
  };
  
  // Test Context API
  fetch(`${config.contextApiUrl}/`)
    .then(response => {
      if (response.ok) {
        showStatus('Context API connection successful!', 'success');
      } else {
        showStatus('Context API connection failed!', 'error');
      }
    })
    .catch(() => {
      showStatus('Context API connection failed!', 'error');
    });
}

// Add event listeners
saveButton.addEventListener('click', saveConfig);
resetButton.addEventListener('click', resetConfig);

// Load configuration on page load
document.addEventListener('DOMContentLoaded', loadConfig);
