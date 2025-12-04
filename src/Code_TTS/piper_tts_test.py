from piper import *
import wave
import sounddevice as sd
import numpy as np
import tempfile
import os

# === Configuration ===
PHRASE = "Hello, this is a test of Piper TTS in Python."
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "en_US-lessac-high.onnx")  # Adapt to your model

# === VERIFY FILES EXIST ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model missing: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH + ".json"):
    raise FileNotFoundError(f"Config missing: {MODEL_PATH}.json")

# Load model once
print("Loading Piper model...")
VOICE = PiperVoice.load(MODEL_PATH)

def speak(text: str):
    # Create temp wav file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    print("Synthesizing speech...")
    with wave.open(wav_path, "wb") as wav_file:
        VOICE.synthesize_wav(text, wav_file)

    print("Playing audio...")
    with wave.open(wav_path, "rb") as wav_file:
        data = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(data, dtype=np.int16)
        samplerate = wav_file.getframerate()

        sd.play(audio, samplerate=samplerate)
        sd.wait()

    os.unlink(wav_path)
    print("Done speaking!")

# === RUN ===
if __name__ == "__main__":
    speak(PHRASE)
