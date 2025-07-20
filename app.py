from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

app = Flask(__name__)
CORS(app)

# ────────────────────────────────────────────────────────────
# Global model state
# ────────────────────────────────────────────────────────────
model = None
tokenizer = None
model_loaded = False
current_model_name = None

# ────────────────────────────────────────────────────────────
# Prompt constants
# ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a patient Japanese tutor for native English speakers.\n"
    "Your teaching style is inductive: learners must guess the meaning before you reveal it.\n"
    "After the first question to get the learner's level, you should immediately follow the following teaching protocol.\n\n"
    "Teaching protocol\n"
    "1. Never explain or translate before the learner takes a guess.\n"
    "2. Focus on one grammar point at a time.\n"
    "3. If the learner's guess is correct:\n"
    "   • Acknowledge success.\n"
    "   • Briefly explain ONLY the target grammar point that the sentence was designed to illustrate (do not break down every word).\n"
    "   • Offer a chance to create the learner's own sentence, or move on.\n"
    "   • Reveal the correct translation ONLY after the explanation, or if the learner explicitly asks for it.\n"
    "4. If the guess is wrong:\n"
    "   • Say it is not quite correct (do NOT reveal the correct answer or translation).\n"
    "   • Give a fresh example that highlights the SAME grammar point.\n"
    "   • Continue until the learner is correct or explicitly asks to see the answer.\n"
    "5. The learner can ask for definitions, clarifications, or the full translation at any time; only then may you reveal it.\n"
    "6. All sentences must show Kanji on the first line and full Hiragana on the second line.\n"
    "7. When an exercise is completed (after you have accepted a correct answer and provided your explanation), automatically create the next exercise sentence (Sentence + Reading) appropriate for the learner's proficiency level.\n\n"
    "Format for every exercise\n"
    "Sentence: <Japanese sentence>\n"
    "Reading:  <Hiragana reading>\n"
    "Please try translating this sentence.\n\n"
    "When a learner provides their level (beginner, intermediate, advanced, or JLPT level), immediately provide the first exercise following the teaching protocol above."
)

PROFILE_REQUEST = (
    "Before we begin, I'd like to tailor the lesson to your level. Roughly what is your JLPT level? If you are not sure, you can provide 'beginner' or 'intermediate' or 'advanced' instead."
)

# ────────────────────────────────────────────────────────────
# Available models
# ────────────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    "llama-3.1-swallow-8b": {
        "name": "Llama 3.1 Swallow 8B",
        "model_id": "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5",
        "description": "Fast and efficient 8B parameter model"
    },
    "llama-3.3-swallow-70b": {
        "name": "Llama 3.3 Swallow 70B",
        "model_id": "tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4",
        "description": "High-quality 70B parameter model"
    },
    "llama-3-youko-8b": {
        "name": "Llama 3 Youko 8B",
        "model_id": "rinna/llama-3-youko-8b-instruct",
        "description": "Japanese-optimized 8B parameter model by rinna"
    },
    "llm-jp-3.1-13b": {
        "name": "LLM-JP 3.1 13B",
        "model_id": "llm-jp/llm-jp-3.1-13b-instruct4",
        "description": "High-quality 13B parameter Japanese language model"
    },
    "mistral-small-24b": {
        "name": "Mistral Small 24B",
        "model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "description": "Powerful 24B parameter model with excellent multilingual capabilities"
    },
    "shisa-v2-12b": {
        "name": "Shisa V2 12B",
        "model_id": "shisa-ai/shisa-v2-mistral-nemo-12b",
        "description": "Japanese-optimized 12B parameter model based on Mistral"
    }
}

# ────────────────────────────────────────────────────────────
# Conversation state tracking
# ────────────────────────────────────────────────────────────
conversation_states = {}


def get_conversation_state(session_id):
    """Fetch (or create) per-session state."""
    if session_id not in conversation_states:
        conversation_states[session_id] = {
            "conversation_history": [
                {"role": "system", "content": SYSTEM_PROMPT}
            ],
            "profile_complete": False,
            "asked_profile": False,
            "learner_profile": {}
        }
    return conversation_states[session_id]



# ────────────────────────────────────────────────────────────
# Model loading
# ────────────────────────────────────────────────────────────
def load_model(model_key: str):
    global model, tokenizer, model_loaded, current_model_name

    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    if model_loaded and current_model_name == model_key:
        return

    cfg = AVAILABLE_MODELS[model_key]
    model_id = cfg["model_id"]

    print(f"\nLoading {cfg['name']} …")

    use_gpu = torch.cuda.is_available()
    dtype = torch.float16 if use_gpu else torch.float32
    device_map = "auto" if use_gpu else {"": "cpu"}

    if use_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"CUDA GPU detected: {gpu_name} — {gpu_mem:.1f} GB")
    else:
        print("No CUDA GPU detected — running on CPU")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model_loaded = True
    current_model_name = model_key
    print(f"Model {cfg['name']} loaded successfully!\nWeb UI → http://localhost:8080")

# ────────────────────────────────────────────────────────────
# Prompt helpers
# ────────────────────────────────────────────────────────────
def build_prompt(msgs):
    """
    Use the tokenizer's chat template to build a prompt string.
    msgs → list[dict] with “role” {user|assistant|system} and “content”
    """
    return tokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=False
    )



# ────────────────────────────────────────────────────────────
# Generation
# ────────────────────────────────────────────────────────────
def generate_response(user_text: str,
                      session_id: str = "default",
                      max_length: int = 512,
                      temperature: float = 0.7) -> str:
    """
    Produce a response using the active model *or* the fixed profile prompt.
    """
    if not model_loaded:
        return "Model not loaded yet. Please wait…"

    state = get_conversation_state(session_id)
    history = state["conversation_history"]

    # Record user turn
    history.append({"role": "user", "content": user_text})

    # ───────── 1) profile flow  ─────────
    if not state["profile_complete"]:
        if not state["asked_profile"]:
            # First assistant turn → ask the profile questions
            history.append({"role": "assistant", "content": PROFILE_REQUEST})
            state["asked_profile"] = True
            return PROFILE_REQUEST

        # We just got the learner's answers; store and mark complete
        state["learner_profile"] = {"raw_response": user_text}
        state["profile_complete"] = True

        # Add the learner's response to the conversation
        history.append({"role": "assistant", "content": PROFILE_REQUEST})
        
        # Add a user message to trigger the first exercise
        history.append({
            "role": "user", 
            "content": f"Based on my level ({user_text}), please provide the first exercise following the teaching protocol."
        })

    # ───────── 2) normal lesson turn  ─────────
    # Build chat prompt for the model
    prompt_text = build_prompt(history)

    inputs = tokenizer(prompt_text,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=4096).to(next(model.parameters()).device)

    input_len = inputs["input_ids"].shape[1]
    max_new = min(max_length, 4096 - input_len)

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=1.0,
        )[0]

    generated_ids = output_ids[input_len:]
    assistant_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    history.append({"role": "assistant", "content": assistant_text})

    return assistant_text

# ────────────────────────────────────────────────────────────
# Flask routes
# ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/models")
def get_models():
    return jsonify({"models": AVAILABLE_MODELS, "current_model": current_model_name})

@app.route("/load-model", methods=["POST"])
def load_selected_model():
    data = request.get_json()
    model_key = data.get("model_key")

    if not model_key:
        return jsonify({"error": "No model specified"}), 400
    if model_key not in AVAILABLE_MODELS:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400

    try:
        load_model(model_key)
        return jsonify({
            "status": "success",
            "message": f"Model {AVAILABLE_MODELS[model_key]['name']} loaded successfully",
            "model_key": model_key
        })
    except Exception as exc:
        print(f"Error loading model: {exc}")
        return jsonify({"error": str(exc)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message   = data.get("message", "")
    session_id     = data.get("sessionId", "default")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    if not model_loaded:
        return jsonify({"error": "No model loaded. Please load a model first."}), 400

    reply = generate_response(user_message, session_id)

    return jsonify({"response": reply, "status": "success"})

@app.route("/status")
def status():
    if model_loaded:
        info = AVAILABLE_MODELS.get(current_model_name, {})
        return jsonify({
            "model_loaded": True,
            "status": "ready",
            "message": f"Model {info.get('name', 'Unknown')} ready",
            "current_model": current_model_name,
            "model_info": info
        })
    return jsonify({
        "model_loaded": False,
        "status": "not_loaded",
        "message": "No model loaded. Please select and load a model.",
        "current_model": None
    })

@app.route("/reset", methods=["POST"])
def reset_session():
    """
    Reset the current session and return to initial state.
    This clears conversation history, resets profile completion,
    unloads models, and clears CUDA cache.
    """
    global model, tokenizer, model_loaded, current_model_name
    
    data = request.get_json()
    session_id = data.get("sessionId", "default")
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared")
    
    # Unload model and tokenizer
    if model_loaded:
        del model
        del tokenizer
        model = None
        tokenizer = None
        model_loaded = False
        current_model_name = None
        print("Model unloaded")
    
    # Reset conversation states
    if session_id in conversation_states:
        # Reset the conversation state to initial values
        conversation_states[session_id] = {
            "conversation_history": [
                {"role": "system", "content": SYSTEM_PROMPT}
            ],
            "profile_complete": False,
            "asked_profile": False,
            "learner_profile": {}
        }
    
    # Force garbage collection to free memory
    gc.collect()
    
    return jsonify({
        "status": "success",
        "message": "Session reset successfully - model unloaded and memory cleared"
    })



# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8080)
