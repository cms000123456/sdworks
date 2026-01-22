#!/bin/bash
set -e

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: ./release.sh <version>"
    echo "Example: ./release.sh 3.1"
    exit 1
fi

echo "Creating release for version $VERSION..."

# Update version in index.html using Python for robust multi-line regex
python3 -c "
import re
import sys

version = '$VERSION'
try:
    with open('index.html', 'r+') as f:
        content = f.read()
        # Regex to find the span with id='appVersion' and replace the version number inside
        # Handles multi-line attributes in the opening tag
        pattern = r'(<span\s+id=\"appVersion\"[^>]*>)\s*v[0-9.]+\s*(</span>)'
        new_content = re.sub(pattern, r'\1v' + version + r'\2', content, flags=re.IGNORECASE | re.DOTALL)
        
        if new_content == content:
            print('WARNING: Version pattern not found or unchanged!')
        else:
            f.seek(0)
            f.write(new_content)
            f.truncate()
            print(f'Successfully updated index.html to v{version}')
except Exception as e:
    print(f'Error updating index.html: {e}')
    sys.exit(1)
"

# Create text release
TXT_FILE="release_v${VERSION}.txt"
echo "--- index.html ---" > "$TXT_FILE"
cat index.html >> "$TXT_FILE"
echo -e "\n--- style.css ---" >> "$TXT_FILE"
cat style.css >> "$TXT_FILE"
echo -e "\n--- app.js ---" >> "$TXT_FILE"
cat app.js >> "$TXT_FILE"

# Create zip release
ZIP_FILE="SDWorks_v${VERSION}.zip"
rm -f "$ZIP_FILE"
echo "Bundling files into $ZIP_FILE..."
# Include frontend, release scripts, and backend (excluding bulky models/cache)
zip -r "$ZIP_FILE" index.html style.css app.js "$TXT_FILE" release.sh backend \
    -x "backend/models/*" \
    -x "backend/huggingface-cache/*" \
    -x "*/__pycache__/*" \
    -x "*.pyc"

echo "✅ Full stack release successfully created!"
echo "  - Bundle: $ZIP_FILE (Includes Backend & Docker)"
echo "  - Source Text: $TXT_FILE (Frontend only)"
