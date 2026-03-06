#!/bin/bash
# Prepare files for Colab upload

echo "==================================="
echo "Cosmic Engine - Colab Upload Prep"
echo "==================================="
echo ""

# Create upload directory
UPLOAD_DIR="colab_upload"
mkdir -p "$UPLOAD_DIR"

echo "Copying files to $UPLOAD_DIR/..."

# Copy essential files
cp cosmic_engine_colab.ipynb "$UPLOAD_DIR/"
cp alien.obj "$UPLOAD_DIR/"
cp prehistoric.obj "$UPLOAD_DIR/"

# Copy Python source files (needed for factory.py)
for py_file in *.py; do
    if [ -f "$py_file" ]; then
        cp "$py_file" "$UPLOAD_DIR/"
    fi
done

# Copy documentation
cp COLAB_QUICK_START.md "$UPLOAD_DIR/" 2>/dev/null || true
cp UPLOAD_CHECKLIST.md "$UPLOAD_DIR/" 2>/dev/null || true

# Copy visualization images for reference
cp alien_structure_vis.png "$UPLOAD_DIR/" 2>/dev/null || true
cp prehistoric_vis.png "$UPLOAD_DIR/" 2>/dev/null || true

echo ""
echo "✓ Files copied to $UPLOAD_DIR/"
echo ""
ls -lh "$UPLOAD_DIR/"
echo ""

# Create zip for easy upload
ZIP_FILE="cosmic_engine_colab_package.zip"
zip -r "$ZIP_FILE" "$UPLOAD_DIR/" >/dev/null 2>&1

if [ -f "$ZIP_FILE" ]; then
    echo "✓ Created zip package: $ZIP_FILE"
    ls -lh "$ZIP_FILE"
    echo ""
fi

echo "==================================="
echo "Ready for Colab Upload!"
echo "==================================="
echo ""
echo "Option 1: Upload individual files from $UPLOAD_DIR/"
echo "Option 2: Upload $ZIP_FILE and extract in Colab"
echo ""
echo "Next steps:"
echo "  1. Go to https://colab.research.google.com"
echo "  2. Upload cosmic_engine_colab.ipynb"
echo "  3. Enable GPU runtime"
echo "  4. Follow UPLOAD_CHECKLIST.md"
echo ""
