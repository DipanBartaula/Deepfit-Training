#!/usr/bin/env bash
# setup_env.sh
# Usage: ./setup_env.sh
# Creates a Python venv in ./venv and installs dependencies.

set -e

PYTHON=${PYTHON:-python3}

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Please ensure it exists in current directory."
    exit 1
fi

echo "Environment setup complete. To activate, run: source venv/bin/activate"
