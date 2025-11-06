import whisper
from flask import Flask, request, jsonify
import os
from datetime import datetime

model=whisper.load_model("small.en") # We load the small english model for whisper

app = Flask(__name__) #We create the flask app

UPLOAD_FOLDER = "recordings" #We check if the folder recordings exists, if not we create it
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"]) 
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400
    
    audio_file = request.files["audio"]
    
    # We create the audio file with a unique name based on the current timestamp
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".webm"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(filepath)

    result=model.transcribe(filepath) # We transcribe the audio file using whisper

    os. remove(filepath) # We delete the audio file after transcription

    print(result["text"]) # We print the transcribed text in the console (This line is optional and will be removed later)

    return jsonify({"message": result["text"], "filename": filename})

if __name__ == "__main__": #We run the app
    app.run(host="0.0.0.0", port=5000, debug=True)

