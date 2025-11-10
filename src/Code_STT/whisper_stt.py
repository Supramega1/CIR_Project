import whisper
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

audio_path = os.path.join(BASE_DIR, "audio", "audio.m4a")

model = whisper.load_model("small.en")
result = model.transcribe(audio_path)
print(result["text"])