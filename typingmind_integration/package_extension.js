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
