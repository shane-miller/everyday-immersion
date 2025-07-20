#!/bin/bash

# Everyday Immersion - Conda Environment Setup Script
# 
# This script automates the creation and configuration of a conda environment
# for the language learning application. It handles dependency installation
# and provides clear feedback throughout the setup process.
#
# Usage: ./setup_env.sh

set -e  # Exit immediately if any command fails

echo "Setting up conda environment for everyday-immersion..."

# Verify conda installation
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed on your system."
    echo ""
    echo "Please install conda before running this script:"
    echo "  - Visit: https://docs.conda.io/en/latest/miniconda.html"
    echo "  - Download and install Miniconda for your operating system"
    echo "  - After installation, restart your terminal and run this script again"
    echo ""
    echo "Alternatively, you can install conda using your system's package manager:"
    echo "  - macOS (with Homebrew): brew install --cask miniconda"
    echo "  - Linux: Follow the instructions at https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if the conda environment already exists
if conda env list | grep -q "everyday-immersion-env"; then
    echo "Conda environment 'everyday-immersion-env' already exists."
else
    echo "Creating conda environment 'everyday-immersion-env'..."
    conda create -n everyday-immersion-env python=3.11 -y
    conda init zsh
    echo "Initializing conda in current shell..."
    eval "$(conda shell.zsh hook)"
    echo "Conda environment created successfully."
fi

# Activate the conda environment
echo "Activating conda environment..."
conda activate everyday-immersion-env

# Update pip to the latest version for optimal package installation
echo "Upgrading pip..."
pip install --upgrade pip

# Install all required dependencies from requirements.txt
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete! Conda environment is ready."
echo ""
echo "To activate the conda environment in the future, run:"
echo "conda activate everyday-immersion-env"
echo ""
echo "To deactivate the conda environment, run:"
echo "conda deactivate" 