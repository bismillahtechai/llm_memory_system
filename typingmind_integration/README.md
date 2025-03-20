# TypingMind Memory System Extension

This browser extension integrates the LLM memory system with TypingMind, providing persistent memory and context-aware responses.

## Features

- **Message Interception**: Intercepts messages sent to the LLM API and augments them with relevant context
- **Context Augmentation**: Adds relevant context from both RAG (factual knowledge) and Mem0 (personal memory) components
- **Memory Extraction**: Automatically extracts memories from user messages
- **User Interface**: Provides UI elements for controlling the memory system
- **Configuration**: Allows users to configure the memory system through an options page

## Installation

### Chrome/Edge

1. Open Chrome/Edge and navigate to `chrome://extensions` or `edge://extensions`
2. Enable "Developer mode"
3. Click "Load unpacked" and select the `typingmind_integration` directory
4. The extension should now be installed and active

### Firefox

1. Open Firefox and navigate to `about:debugging#/runtime/this-firefox`
2. Click "Load Temporary Add-on..."
3. Select the `manifest.json` file in the `typingmind_integration` directory
4. The extension should now be installed and active

## Configuration

1. Click the extension icon in the browser toolbar
2. Click "Options" to open the configuration page
3. Set the API endpoints for the memory system components:
   - Context API URL: URL of the context management API
   - RAG API URL: URL of the RAG component API
   - Mem0 API URL: URL of the Mem0 component API
4. Configure memory system settings:
   - Token Budget: Maximum number of tokens to use for context
   - Enable RAG: Enable/disable the RAG component
   - Enable Mem0: Enable/disable the Mem0 component
5. Configure debug settings:
   - Enable Debug Mode: Show debug information in the TypingMind interface
   - Log API Calls: Log API calls to the console
6. Click "Save Settings" to apply the changes

## Usage

1. Navigate to TypingMind
2. The extension will automatically intercept messages and augment them with relevant context
3. To see debug information, enable debug mode in the options page
4. The debug overlay will show information about context augmentation and memory extraction

## Development

### Project Structure

- `memory_system_extension.js`: Main extension script
- `background.js`: Background script for API communication
- `options.html`: Options page HTML
- `options.js`: Options page JavaScript
- `manifest.json`: Extension manifest
- `icons/`: Extension icons

### Building

No build step is required for development. For production, you may want to minify the JavaScript files.

### Testing

1. Load the extension in your browser
2. Navigate to TypingMind
3. Open the browser console to see debug information
4. Test sending messages and verify that context is being added

## Troubleshooting

- If the extension is not working, check the browser console for errors
- Verify that the API endpoints are correct and accessible
- Make sure the memory system components are running and accessible
