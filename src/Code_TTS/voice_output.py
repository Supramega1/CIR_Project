import pyttsx3

try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 175) # speed of speech
    voices = tts_engine.getProperty('voices')
    tts_engine.setProperty('voice', voices[1].id) # only voice 1 works for me, idk why
except Exception as e:
    print(f"Could not initialize TTS engine: {e}")
    print("TTS functionality will be disabled.")
    tts_engine = None

def speak_text(text: str):
    """
    Uses the initialized TTS engine to speak the provided text aloud.
    """
    if tts_engine:
        try:
            print(f"YumEye says: {text}")
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"Error while speaking: {e}")
    else:
        print(f"TTS disabled. Would have said: {text}")

# usage
if __name__ == "__main__":
    speak_text("Hello! I am YumEye. I'm ready to help you cook.")
    speak_text("Please show me your ingredients.")