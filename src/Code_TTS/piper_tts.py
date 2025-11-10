from piper import PiperVoice
import wave
import sounddevice as sd
import numpy as np
import tempfile
import os

# === Configuration ===
PHRASE = "Hello, this is a test of Piper TTS in Python."
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "en_US-lessac-high.onnx") # Change to your local model path

# === VERIFY FILES EXIST ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model missing: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH + ".json"):
    raise FileNotFoundError(f"Config missing: {MODEL_PATH}.json")

def speak(text):
    print("Loading Piper model...")
    voice = PiperVoice.load(MODEL_PATH)

    # Create temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    # This WILL work now (you have 1.3.0)
    print("Synthesizing speech...")
    audio = voice.synthesize(text, wav_path)

    # Play
    print("Playing audio...")
    with wave.open(wav_path, "rb") as wav_file:
        data = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(data, dtype=np.int16)
        samplerate = wav_file.getframerate()
        sd.play(audio, samplerate=samplerate)
        sd.wait()

    # Cleanup
    os.unlink(wav_path)
    print("Done speaking!")

# === RUN ===
if __name__ == "__main__":
    speak(PHRASE)