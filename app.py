from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading

app = Flask(__name__)
CORS(app)

# Global model state variables
model = None
tokenizer = None
model_loaded = False

def load_model():
    """
    Load the Llama-3.1-Swallow-8B-Instruct model and tokenizer.
    
    This function initializes the language model for the chat interface.
    It automatically detects available hardware (GPU/CPU) and loads the model
    with appropriate optimizations for the detected environment.
    """
    global model, tokenizer, model_loaded
    
    if model_loaded:
        return
    
    try:
        print("\nLoading Llama-3.1-Swallow-8B-Instruct-v0.5 model...")
        
        # Detect and log available hardware
        if torch.cuda.is_available():
            print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("No CUDA GPU detected - using CPU")
        
        # Model configuration
        model_name = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
        
        # Initialize tokenizer with remote code trust
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Handle pad token configuration for proper tokenization
        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            # Use unknown token ID as pad token ID if available
            if hasattr(tokenizer, "unk_token_id") and tokenizer.unk_token_id is not None:
                tokenizer.pad_token_id = tokenizer.unk_token_id
            else:
                # Fallback to end-of-sequence token ID
                tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model with hardware-optimized configuration
        if torch.cuda.is_available():
            print("Loading model with GPU acceleration...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",  # Automatic GPU memory management
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            print("Loading model with CPU (this may take a while)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cpu",  # CPU-only mode
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        model_loaded = True
        print("Model loaded successfully!")
        print("Web interface is ready at: http://localhost:8080")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def create_language_context(preferred_language, learning_language):
    """
    Create a context prompt for inductive language learning.
    
    Args:
        preferred_language (str): The user's native/preferred language
        learning_language (str): The target language to learn
    
    Returns:
        str: Formatted context string for the language model
    """
    
    # Language-specific learning configurations
    language_configs = {
        "japanese": {
            "writing_systems": "hiragana, katakana, kanji",
            "proficiency_scale": "JLPT levels (N5-N1)",
            "special_features": "furigana readings, politeness levels, particles",
            "example_format": "漢字(かんじ) - include furigana for kanji"
        },
        "korean": {
            "writing_systems": "hangul",
            "proficiency_scale": "TOPIK levels",
            "special_features": "honorifics, particles, verb conjugations",
            "example_format": "한글 - standard hangul format"
        }
    }
    
    # Get configuration for the target language, or use defaults
    config = language_configs.get(learning_language, {
        "writing_systems": "standard script",
        "proficiency_scale": "beginner/intermediate/advanced",
        "special_features": "grammar patterns, cultural context",
        "example_format": "standard format"
    })
    
    preferred_lang = preferred_language.title()
    learning_lang = learning_language.title()
    
    return f"""You are an inductive language learning assistant helping {preferred_lang} speakers learn {learning_lang}.

CORE METHODOLOGY:
- Start by assessing user's proficiency level
- Present examples in context, not isolated sentences
- Let users discover patterns through guessing
- Provide hints and additional examples if needed
- Only explain after user attempts or requests help
- Encourage sentence creation once patterns are understood

LANGUAGE-SPECIFIC GUIDELINES:
- Writing systems: {config['writing_systems']}
- Proficiency scale: {config['proficiency_scale']}
- Special features: {config['special_features']}
- Example format: {config['example_format']}

RESPONSE STYLE:
- Concise and encouraging
- Mix {learning_lang} and {preferred_lang} appropriately
- Adapt complexity to user level
- Be patient and supportive
- Focus on practical, everyday language

User message: """

def generate_response(prompt, preferred_language="english", learning_language="japanese", max_length=512, temperature=0.7):
    """
    Generate a language learning response using the loaded model.
    
    Args:
        prompt (str): User's input message
        preferred_language (str): User's native language
        learning_language (str): Target language to learn
        max_length (int): Maximum tokens for response generation
        temperature (float): Sampling temperature for response creativity
    
    Returns:
        str: Generated response for language learning
    """
    if not model_loaded:
        return "Model not loaded yet. Please wait..."
    
    try:
        # Create language-specific learning context
        language_context = create_language_context(preferred_language, learning_language)
        
        # Format prompt for instruction-following model
        formatted_prompt = f"<|im_start|>user\n{language_context}{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize input with proper attention masking and truncation
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        )
        
        # Calculate available tokens for generation (respecting model limits)
        input_length = inputs.input_ids.shape[1]
        max_new_tokens = min(max_length, 2048 - input_length)
        
        # Generate response with optimized parameters
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0
            )
        
        # Decode the generated response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response portion
        response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        
        # Handle empty responses
        if not response:
            return "I'm sorry, I couldn't generate a response. Please try again."
        
        # Clean up any remaining special tokens
        response = response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
        
        return response
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}"

@app.route("/")
def index():
    """Serve the main chat interface HTML page."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """
    Handle chat requests from the frontend.
    
    Expected JSON payload:
    {
        "message": "user input text",
        "preferredLanguage": "english",
        "learningLanguage": "japanese"
    }
    
    Returns:
        JSON response with generated text or error message
    """
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        preferred_language = data.get("preferredLanguage", "english")
        learning_language = data.get("learningLanguage", "japanese")
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Generate language learning response
        response = generate_response(user_message, preferred_language, learning_language)
        
        return jsonify({
            "response": response,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/status")
def status():
    """
    Check the current status of the language model.
    
    Returns:
        JSON response indicating model loading status
    """
    if model_loaded:
        return jsonify({
            "model_loaded": True,
            "status": "ready",
            "message": "Model is ready"
        })
    else:
        # Check if the model loading thread is still active
        if model_thread.is_alive():
            return jsonify({
                "model_loaded": False,
                "status": "loading",
                "message": "Model is loading..."
            })
        else:
            return jsonify({
                "model_loaded": False,
                "status": "error",
                "message": "Model loading failed"
            })

def load_model_async():
    """Load the model in a background thread to avoid blocking the web interface."""
    load_model()

# Initialize model loading in background thread
model_thread = threading.Thread(target=load_model_async, daemon=True)
model_thread.start()

if __name__ == "__main__":
    # Start Flask development server
    app.run(debug=True, host="127.0.0.1", port=8080) 