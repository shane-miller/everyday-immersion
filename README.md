# Everyday Immersion

Inductive-style language learning to learn like you're there.

## Overview

Everyday Immersion is an AI-powered language learning application that uses inductive learning methods to help users learn languages naturally through conversation. The app leverages the Llama-3.1-Swallow-8B-Instruct model to provide personalized language learning experiences.

## Features

- **Inductive Learning**: Learn languages through pattern recognition and discovery rather than rote memorization
- **Multi-language Support**: Currently supports English speakers learning Japanese (additional language pairs planned for future releases)
- **Adaptive Difficulty**: The AI adjusts complexity based on your proficiency level
- **Natural Conversation**: Practice real-world language use through interactive chat
- **Contextual Learning**: Learn grammar and vocabulary in meaningful contexts
- **Web-based Interface**: Clean, modern UI accessible from any browser

## Supported Languages

### Current Language Pair
- **English → Japanese**: Complete support with hiragana, katakana, kanji, and JLPT-level guidance

### Future Language Support
- Additional language pairs will be added in future releases
- Planned support for other native languages and target languages

## System Requirements

- **Python**: 3.8 or higher (3.11 recommended)
- **RAM**: At least 8GB (16GB recommended)
- **Storage**: ~8GB for model download
- **GPU**: CUDA-compatible GPU (optional but recommended for faster performance)
- **Internet**: Required for initial model download

## Installation

### Option 1: Using Conda (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd everyday-immersion
   ```

2. **Run the setup script**:
   ```bash
   chmod +x setup_env.sh
   ./setup_env.sh
   ```

3. **Activate the conda environment**:
   ```bash
   conda activate everyday-immersion-env
   ```

### Option 2: Manual Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Application

1. **Activate the environment** (if using conda):
   ```bash
   conda activate everyday-immersion-env
   ```

2. **Run the application**:
   ```bash
   python run.py
   ```
   
   Or alternatively:
   ```bash
   python app.py
   ```

3. **Access the web interface**:
   Open your browser and navigate to `http://localhost:8080`

### Using the Application

1. **Wait for model loading**: The AI model will load automatically on first run (this may take several minutes)

2. **Select your languages**:
   - Currently supports English speakers learning Japanese
   - Language selection is automatically configured for English → Japanese

3. **Start learning**:
   - Click "Begin Learning Session" to start
   - Type messages in your preferred language
   - The AI will respond with a mix of the target language and explanations
   - Practice conversation and ask questions naturally

4. **Learning methodology**:
   - The AI uses inductive learning techniques
   - It presents examples in context rather than isolated sentences
   - You'll discover patterns through guessing and exploration
   - Explanations are provided only after you attempt or request help
