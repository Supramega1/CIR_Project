import sounddevice as sd
import numpy as np
from piper import PiperVoice
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "en_US-lessac-low.onnx")  # Adapt to your model

voice = PiperVoice.load(MODEL_PATH)

texte = "Hello, this is a test of Piper TTS in Python."

stream = None

for chunk in voice.synthesize(texte):
    audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)

    if stream is None:
        stream = sd.OutputStream(
            samplerate=chunk.sample_rate,
            channels=chunk.sample_channels,
            dtype='int16'
        )
        stream.start()

    stream.write(audio)

stream.stop()
stream.close()
