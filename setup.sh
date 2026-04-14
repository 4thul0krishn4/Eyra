#!/bin/bash
set -e

echo "👁️ Setting up Eyra..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install it with: brew install python"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  Python $PYTHON_VERSION ✅"

# Create virtual environment
echo "  Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install dependencies
echo "  Installing dependencies (this may take a few minutes)..."
pip install -e . --quiet

echo ""
echo "✅ Eyra is ready!"
echo ""
echo "Next steps:"
echo "  1. Activate the environment:  source venv/bin/activate"
echo "  2. Index your images:         eyra index ~/Pictures"
echo "  3. Search:                    eyra search 'describe what you want'"
echo "  4. Start web UI:              eyra serve"
echo ""
