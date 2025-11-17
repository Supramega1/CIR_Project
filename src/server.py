"""
Flask server:
1. Receive a .webm audio file
2. Transcribe it with Whisper (small.en)
3. Feed the text to Ollama (llama3.2:3b) with a chef system prompt
4. Keep a per-conversation message history
5. Return transcription + LLM recipe in JSON
"""

import whisper
from flask import Flask, request, jsonify
import os
from datetime import datetime
import ollama
import threading

# ------------------- CONFIG -------------------
# Whisper model (English-only, fast on CPU/GPU)
whisper_model = whisper.load_model("small.en")

# Ollama model to use for recipe generation
LLM_MODEL = "llama3.2"          # Might change to "llama3.2:1b" or "phi3" later

# Folder where uploaded audio files are temporarily stored
UPLOAD_FOLDER = "recordings"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory store for conversation histories
#   key = conversation_id
#   value = {"messages": [...], "last_used": datetime}
CONVERSATIONS = {}

# System prompt – defines the LLM persona
SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are YUMEYE, a creative English-speaking chef.
Generate ONLY one detailed recipe in English witht the ingredients provided by the user.
The recipe must include:
- Title
- Ingredients with US measurements (1 ½ cups, 2 tsp, 6 oz…)
- Numbered steps
- Prep + cook time
- 1 fun tip
Use ALL ingredients mentioned. Be precise and professional."""
}

app = Flask(__name__)

# ------------------- CLEANUP (We might need to change the behavior of the cleanup because we have a small model) -------------------
def cleanup_old_conversations():
    """Remove conversations inactive for more than 1 hour."""
    now = datetime.now()
    stale = [
        cid for cid, data in CONVERSATIONS.items()
        if (now - data["last_used"]).total_seconds() > 3600
    ]
    for cid in stale:
        CONVERSATIONS.pop(cid, None)

# Run cleanup every hour in the background
threading.Thread(
    target=lambda: [
        cleanup_old_conversations(),
        threading.Timer(3600, cleanup_old_conversations).start()
    ],
    daemon=True
).start()

# ------------------- MAIN ENDPOINT -------------------
@app.route("/upload", methods=["POST"])
def upload_audio():
    """
    Expected multipart/form-data:
        - audio: .webm file
        - conversation_id (optional): string to keep history
    Returns JSON with transcription + LLM recipe.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_file = request.files["audio"]
    # Use provided ID or create a new one based on timestamp
    conversation_id = request.form.get(
        "conversation_id",
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    # ---- 1. Save the audio temporarily ----
    filename = f"{conversation_id}.webm"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(filepath)

    try:
        # ---- 2. Transcribe with Whisper ----
        result = whisper_model.transcribe(filepath)
        user_text = result["text"].strip()
        print(f"[User] {user_text}")

        # ---- 3. Initialise / retrieve conversation ----
        if conversation_id not in CONVERSATIONS:
            CONVERSATIONS[conversation_id] = {
                "messages": [SYSTEM_PROMPT],
                "last_used": datetime.now()
            }

        conv = CONVERSATIONS[conversation_id]
        conv["last_used"] = datetime.now()

        # Append user message
        conv["messages"].append({"role": "user", "content": user_text})

        # ---- 4. Call Ollama ----
        response = ollama.chat(
            model=LLM_MODEL,
            messages=conv["messages"]
        )
        llm_reply = response["message"]["content"]

        # Append assistant reply to history
        conv["messages"].append({"role": "assistant", "content": llm_reply})
        print(f"[Chef] {llm_reply}...")

        # ---- 5. Clean up temporary file ----
        os.remove(filepath)

        # ---- 6. Return JSON ----
        return jsonify({
            "transcription": user_text,
            "llm_response": llm_reply,
            "conversation_id": conversation_id
        })

    except Exception as e:
        # Ensure file is removed even on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500
    
@app.route("/texte", methods=["POST"])
def texte_input():
    """
    Expected JSON:
        - text (string): user-written ingredients
        - conversation_id (optional): to keep history

    Returns JSON with LLM-generated recipe.
    """
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    user_text = data["text"].strip()
    if not user_text:
        return jsonify({"error": "Empty text"}), 400

    # Use provided ID or generate a new one
    conversation_id = data.get(
        "conversation_id",
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    print(f"[User TEXT] {user_text}")

    try:
        # ---- 1. Initialise / retrieve conversation ----
        if conversation_id not in CONVERSATIONS:
            CONVERSATIONS[conversation_id] = {
                "messages": [SYSTEM_PROMPT],
                "last_used": datetime.now()
            }

        conv = CONVERSATIONS[conversation_id]
        conv["last_used"] = datetime.now()

        # Append user message
        conv["messages"].append({"role": "user", "content": user_text})

        # ---- 2. Call Ollama ----
        response = ollama.chat(
            model=LLM_MODEL,
            messages=conv["messages"]
        )
        llm_reply = response["message"]["content"]

        # Append assistant reply to history
        conv["messages"].append({"role": "assistant", "content": llm_reply})
        print(f"[Chef] {llm_reply}")

        # ---- 3. Return JSON ----
        return jsonify({
            "transcription": user_text,
            "llm_response": llm_reply,
            "conversation_id": conversation_id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- HEALTH CHECK -------------------
@app.route("/health")
def health():
    """Simple health endpoint for monitoring."""
    return jsonify({
        "status": "ok",
        "active_conversations": len(CONVERSATIONS)
    })

# ------------------- RUN SERVER -------------------
if __name__ == "__main__":
    print(f"Server starting | Whisper: small.en | LLM: {LLM_MODEL}")
    app.run(host="0.0.0.0", port=5000, debug=False)