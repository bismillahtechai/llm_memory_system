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
