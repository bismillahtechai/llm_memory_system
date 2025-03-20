// Create a directory for icons
mkdir -p /home/ubuntu/llm_memory_system/typingmind_integration/icons

// Create a README file for the TypingMind integration
cat > /home/ubuntu/llm_memory_system/typingmind_integration/README.md << 'EOF'
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
EOF

// Create a package.json file for the TypingMind integration
cat > /home/ubuntu/llm_memory_system/typingmind_integration/package.json << 'EOF'
{
  "name": "typingmind-memory-system",
  "version": "1.0.0",
  "description": "Browser extension that integrates the LLM memory system with TypingMind",
  "main": "memory_system_extension.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1",
    "build": "echo \"No build step required\""
  },
  "keywords": [
    "typingmind",
    "memory",
    "llm",
    "extension"
  ],
  "author": "Manus",
  "license": "MIT"
}
EOF

// Create a simple icon for the extension
cat > /home/ubuntu/llm_memory_system/typingmind_integration/create_icons.js << 'EOF'
const fs = require('fs');
const path = require('path');

// Create icons directory if it doesn't exist
const iconsDir = path.join(__dirname, 'icons');
if (!fs.existsSync(iconsDir)) {
  fs.mkdirSync(iconsDir, { recursive: true });
}

// Simple SVG icon template
const createIconSVG = (size) => `
<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" xmlns="http://www.w3.org/2000/svg">
  <rect width="${size}" height="${size}" rx="${size/8}" fill="#3498db" />
  <circle cx="${size/2}" cy="${size/3}" r="${size/6}" fill="#ffffff" />
  <rect x="${size/4}" y="${size/2}" width="${size/2}" height="${size/3}" rx="${size/16}" fill="#ffffff" />
</svg>
`;

// Create icons of different sizes
const sizes = [16, 48, 128];
sizes.forEach(size => {
  const iconPath = path.join(iconsDir, `icon${size}.svg`);
  fs.writeFileSync(iconPath, createIconSVG(size));
  console.log(`Created icon: ${iconPath}`);
});

console.log('Icons created successfully!');
EOF

// Create a zip script for packaging the extension
cat > /home/ubuntu/llm_memory_system/typingmind_integration/package_extension.js << 'EOF'
const fs = require('fs');
const path = require('path');
const archiver = require('archiver');

// Create a file to stream archive data to
const output = fs.createWriteStream(path.join(__dirname, 'typingmind_memory_system.zip'));
const archive = archiver('zip', {
  zlib: { level: 9 } // Sets the compression level
});

// Listen for all archive data to be written
output.on('close', function() {
  console.log(`Extension packaged successfully: ${archive.pointer()} total bytes`);
});

// Handle warnings and errors
archive.on('warning', function(err) {
  if (err.code === 'ENOENT') {
    console.warn(err);
  } else {
    throw err;
  }
});

archive.on('error', function(err) {
  throw err;
});

// Pipe archive data to the file
archive.pipe(output);

// Add files to the archive
const filesToInclude = [
  'memory_system_extension.js',
  'background.js',
  'options.html',
  'options.js',
  'manifest.json',
  'README.md'
];

filesToInclude.forEach(file => {
  archive.file(path.join(__dirname, file), { name: file });
});

// Add the icons directory
archive.directory(path.join(__dirname, 'icons'), 'icons');

// Finalize the archive
archive.finalize();
EOF

echo "TypingMind integration files created successfully!"
