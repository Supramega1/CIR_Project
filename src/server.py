import whisper
from flask import Flask, request, jsonify, render_template, url_for, redirect
import os
from datetime import datetime
import ollama
import threading
import sounddevice as sd
import numpy as np
from PIL import Image
import cv2
from piper import PiperVoice
from Code_YOLO.food_detection import load_model, predict_on_frame
import re
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt

# ------------------- Database -------------------

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=1, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    
    repeat_password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Repeat Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')
        
    def validate_repeat_password(self, repeat_password):
        if repeat_password.data != self.password.data:
            raise ValidationError(
                'Passwords do not match. Please try again.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=1, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=1, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


@app.route('/')
def home_login():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if request.method == 'POST':
        if form.validate_on_submit():
            user = User.query.filter_by(username=form.username.data).first()
            if user and bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('index'))
            else:
                # General error for invalid credentials
                form.password.errors.append("Invalid username or password.")
        # If validation fails, errors will be shown automatically
    return render_template('login.html', form=form)


@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('index'))

    return render_template('register.html', form=form)



# ------------------- CONFIG -------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Whisper model (English-only, fast on CPU/GPU)
whisper_model = whisper.load_model("small.en")

#Yolo Model for ingredient detection 
MODEL_PATH_YOLO = os.path.join(SCRIPT_DIR, "Code_YOLO" , "models" , "yolo16c.pt")
yolo_model = load_model(model_path=MODEL_PATH_YOLO)

# Ollama model to use for recipe generation
LLM_MODEL = "llama3.2"          # Might change to "llama3.2:1b" or "phi3" later
MAX_HISTORY_MESSAGES = 6  # user + bot messages to keep in context

#Piper Model
MODEL_PATH = os.path.join(SCRIPT_DIR, "en_US-lessac-high.onnx")
voice = PiperVoice.load(MODEL_PATH, use_cuda=True)
current_thread = None
current_stop_event = None

# Folder where uploaded audio files are temporarily stored
UPLOAD_FOLDER = "recordings"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory store for conversation histories
#   key = conversation_id
#   value = {"messages": [...], "last_used": datetime}
CONVERSATIONS = {}

# Track active operations per user
USER_OPERATIONS = {}  # {user_id: {"tts_thread": thread, "tts_stop_event": event, "conversation_id": conv_id}}

# Track active users (users currently logged in)
ACTIVE_USERS = set()  # {user_id, user_id, ...}

# System prompt — defines the LLM persona
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



def sanitize_text_add_dots(text):
    """
    Keep basic punctuation, normalize spaces, preserve line breaks,
    and ensure each line ends with a dot if it doesn't already end with .!? 
    """
    # Split text into lines
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        # Remove unwanted characters but keep basic punctuation
        line = re.sub(r'[^A-Za-z0-9\s.,!?-]', '', line)
        # Normalize spaces
        line = re.sub(r'\s+', ' ', line).strip()
        if not line:
            cleaned_lines.append('')  # keep blank lines
            continue
        # Add dot if line doesn't end with punctuation
        if not line.endswith(('.', '!', '?')):
            line += '.'
        cleaned_lines.append(line)

    # Rejoin lines with line breaks
    print('\n'.join(cleaned_lines))
    return '\n'.join(cleaned_lines)


def tts_stream(texte: str, user_id: int = None):
    """
    Stream TTS audio for a given text.
    If user_id is provided, track the operation for cleanup on logout.
    IMPORTANT: Check if user is still active before starting TTS.
    """
    global current_thread, current_stop_event

    # Check if user is still logged in
    if user_id and user_id not in ACTIVE_USERS:
        print(f"User {user_id} is no longer active. Skipping TTS.")
        return None

    texte = sanitize_text_add_dots(texte)

    # If a TTS is already running, stop it. It should never happen with the current design, but just in case.
    if current_thread is not None and current_thread.is_alive():
        print("Stopping previous TTS stream")
        current_stop_event.set()
        current_thread.join()

    stop_event = threading.Event()
    current_stop_event = stop_event

    def worker():
        stream = None

        for chunk in voice.synthesize(texte):

            if stop_event.is_set():
                print("Stopping TTS stream on request")
                break

            audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)

            if stream is None:
                stream = sd.OutputStream(
                    samplerate=chunk.sample_rate,
                    channels=chunk.sample_channels,
                    dtype='int16'
                )
                stream.start()

            stream.write(audio)

        if stream is not None:
            stream.stop()
            stream.close()
            print("TTS stream finished")

    # Start TTS in a separate thread
    thread = threading.Thread(target=worker)
    current_thread = thread
    thread.start()
    
    # Track per-user if user_id provided
    if user_id:
        if user_id not in USER_OPERATIONS:
            USER_OPERATIONS[user_id] = {}
        USER_OPERATIONS[user_id]["tts_thread"] = thread
        USER_OPERATIONS[user_id]["tts_stop_event"] = stop_event
    
    return None


def cleanup_user_operations(user_id: int):
    """
    Stop all active operations for a user (TTS, recordings, etc.)
    """
    # Remove from active users
    ACTIVE_USERS.discard(user_id)
    
    if user_id not in USER_OPERATIONS:
        return
    
    user_ops = USER_OPERATIONS[user_id]
    
    # Stop TTS if running
    if "tts_stop_event" in user_ops and "tts_thread" in user_ops:
        thread = user_ops["tts_thread"]
        stop_event = user_ops["tts_stop_event"]
        
        if thread and thread.is_alive():
            print(f"Stopping TTS for user {user_id}")
            stop_event.set()
            thread.join(timeout=2)  # Wait max 2 seconds
    
    # Clear conversation history
    if "conversation_id" in user_ops:
        conv_id = user_ops["conversation_id"]
        if conv_id in CONVERSATIONS:
            print(f"Clearing conversation {conv_id} for user {user_id}")
            del CONVERSATIONS[conv_id]
    
    # Remove user from tracking
    del USER_OPERATIONS[user_id]
    print(f"Cleaned up operations for user {user_id}")


@app.route("/stop_tts", methods=["GET"])
def stop_tts():
    global current_thread, current_stop_event

    if current_thread and current_thread.is_alive():
        print("Stopping TTS stream")
        current_stop_event.set()
        current_thread.join()
        return jsonify({
            "status": "success",
            "message": "TTS stream stopped"
        }), 200

    return jsonify({
        "status": "idle",
        "message": "No active TTS stream to stop"
    }), 200

# ------------------- CLEANUP (We might need to change the behavior of the cleanup because we have a small model) -------------------
def trim_conversation(conv):
    """
    Keeps:
    - system prompt
    - last MAX_HISTORY_MESSAGES messages (user/assistant)
    """
    messages = conv["messages"]

    # Si on n'a que le system prompt → rien à faire
    if len(messages) <= MAX_HISTORY_MESSAGES + 1:
        return

    system_msg = messages[0]
    recent_msgs = messages[-MAX_HISTORY_MESSAGES:]

    conv["messages"] = [system_msg] + recent_msgs
    return

# ------------------- MAIN ENDPOINT -------------------
@app.route("/upload", methods=["POST"])
@login_required
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

    # Get user_id before processing
    user_id = current_user.id
    # Mark user as active
    ACTIVE_USERS.add(user_id)

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

        # Trim conversation history if too long
        trim_conversation(conv)

        # ---- 4. Call Ollama ----
        response = ollama.chat(
            model=LLM_MODEL,
            messages=conv["messages"]
        )
        llm_reply = response["message"]["content"]

        # Check if user is still active before starting TTS
        if user_id in ACTIVE_USERS:
            # Start TTS stream for LLM reply with user tracking
            tts_stream(llm_reply, user_id=user_id)
            
            # Track conversation for this user
            if user_id not in USER_OPERATIONS:
                USER_OPERATIONS[user_id] = {}
            USER_OPERATIONS[user_id]["conversation_id"] = conversation_id
        else:
            print(f"User {user_id} logged out during processing. Skipping TTS.")

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
    
@app.route("/upload_image", methods=["POST"])
@login_required
def upload_image():
    # We get the image but do nothing with it for now
    image = request.files.get("image")

    # STARTING TO BUILD

    temp_path = os.path.join(UPLOAD_FOLDER, "temp_image.jpg")
    image.save(temp_path)

    img = cv2.imread(temp_path)

    if img is None:
        # Fallback with PIL
        try:
            pil_img = Image.open(temp_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Failed to load image: {temp_path} ({e})")
            return jsonify({"Failed to load image": str(e)}), 500
        

    result = predict_on_frame(model=yolo_model, frame=img)

    os.remove(temp_path)

    # Get the list of class IDs predicted
    class_ids = result.boxes.cls.tolist()  # Convert tensor → Python list

    # Convert class IDs to class names
    detected_names = [yolo_model.names[int(c)] for c in class_ids]

    unique_ingredients = list(set(detected_names))

    if unique_ingredients:
        ingredients_str = ", ".join(unique_ingredients)
        user_text = f"Make me a recipe using: {ingredients_str}, please!"
    else:
        user_text = "No ingredients detected in the image. Notify the user ( me )."

    # END OF BUILDING

    # Use provided ID or generate a new one
    conversation_id = request.form.get(
        "conversation_id",
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    print(f"[User TEXT] {user_text}")

    # Get user_id before processing
    user_id = current_user.id
    # Mark user as active
    ACTIVE_USERS.add(user_id)

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

        # Trim conversation history if too long
        trim_conversation(conv)

        # ---- 2. Call Ollama ----
        response = ollama.chat(
            model=LLM_MODEL,
            messages=conv["messages"]
        )
        llm_reply = response["message"]["content"]

        # Check if user is still active before starting TTS
        if user_id in ACTIVE_USERS:
            # Start TTS stream for LLM reply with user tracking
            tts_stream(llm_reply, user_id=user_id)
            
            # Track conversation for this user
            if user_id not in USER_OPERATIONS:
                USER_OPERATIONS[user_id] = {}
            USER_OPERATIONS[user_id]["conversation_id"] = conversation_id
        else:
            print(f"User {user_id} logged out during processing. Skipping TTS.")

        # Append assistant reply to history
        conv["messages"].append({"role": "assistant", "content": llm_reply})
        print(f"[Chef] {llm_reply}")

        # ---- 3. Return JSON ----
        return jsonify({
            "transcription": user_text,
            "ingredients": unique_ingredients, 
            "llm_response": llm_reply,
            "conversation_id": conversation_id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


        
@app.route("/texte", methods=["POST"])
@login_required
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

    # Get user_id before processing
    user_id = current_user.id
    # Mark user as active
    ACTIVE_USERS.add(user_id)

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

        # Trim conversation history if too long
        trim_conversation(conv)

        # ---- 2. Call Ollama ----
        response = ollama.chat(
            model=LLM_MODEL,
            messages=conv["messages"]
        )
        llm_reply = response["message"]["content"]

        # Check if user is still active before starting TTS
        if user_id in ACTIVE_USERS:
            # Start TTS stream for LLM reply with user tracking
            tts_stream(llm_reply, user_id=user_id)
            
            # Track conversation for this user
            if user_id not in USER_OPERATIONS:
                USER_OPERATIONS[user_id] = {}
            USER_OPERATIONS[user_id]["conversation_id"] = conversation_id
        else:
            print(f"User {user_id} logged out during processing. Skipping TTS.")

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


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    user_id = current_user.id
    
    # Clean up all active operations for this user
    cleanup_user_operations(user_id)
    
    # Log out the user
    logout_user()
    
    return redirect(url_for('login'))


# ------------------- HEALTH CHECK -------------------
@app.route("/health")
def health():
    """Simple health endpoint for monitoring."""
    return jsonify({
        "status": "ok",
        "active_conversations": len(CONVERSATIONS),
        "active_users": len(ACTIVE_USERS)
    })

# ------------------- RUN SERVER -------------------
if __name__ == "__main__":
    print("Checking database...")
    with app.app_context():
        db.create_all()   # creates tables only if they don't exist
        print("Database ready.")
    print(f"Server starting | Whisper: small.en | LLM: {LLM_MODEL} | Piper TTS en_US-lessac-high")
    app.run(host="0.0.0.0", port=5000, debug=True)