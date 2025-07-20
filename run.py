#!/usr/bin/env python3
"""
Everyday Immersion Language Learning App - Startup Script

This script provides a user-friendly startup process for the language learning
application, including dependency checks, system requirements validation,
and graceful error handling.
"""

import sys
import os

def check_python_version():
    """
    Verify that the Python version meets minimum requirements.
    
    The application requires Python 3.8 or higher for compatibility
    with the required machine learning libraries.
    """
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def check_dependencies():
    """
    Verify that all required dependencies are installed.
    
    Returns:
        bool: True if all dependencies are available, False otherwise
    """
    try:
        import torch
        import transformers
        import flask
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_model_cache():
    """
    Check if the language model is already downloaded in the local cache.
    
    This helps users understand if they need to download the model
    on first run, which can take several minutes.
    
    Returns:
        bool: True if model exists in cache, False otherwise
    """
    from transformers import AutoTokenizer
    model_name = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
    
    try:
        # Attempt to load tokenizer to verify model availability
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Model is already downloaded")
        return True
    except Exception:
        print("⚠️  Model not found in cache")
        print("   The model will be downloaded on first run (~8GB)")
        return False

def main():
    """
    Main startup function that orchestrates the application launch process.
    
    This function performs all necessary checks and provides user feedback
    before starting the Flask web server.
    """
    print("🚀 Starting Everyday Immersion Language Learning App")
    print("=" * 50)
    
    # Perform system compatibility checks
    check_python_version()
    
    # Verify required dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model availability
    check_model_cache()
    
    # Display system requirements information
    print("\n📋 System Requirements:")
    print("   - At least 8GB RAM (16GB recommended)")
    print("   - GPU with CUDA support (optional but recommended)")
    print("   - Internet connection for model download")
    
    print("\n🎯 Starting the application...")
    print("   The model will be loaded on first run")
    print("   This may take several minutes")
    print("   You can monitor progress in the web interface")
    
    # Launch the Flask application
    try:
        from app import app
        print("\n🌐 Web interface will be available at: http://localhost:8080")
        print("   Press Ctrl+C to stop the server")
        print("-" * 50)
        
        app.run(debug=True, host="127.0.0.1", port=8080)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 