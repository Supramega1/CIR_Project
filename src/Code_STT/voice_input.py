import speech_recognition as sr

recognizer = sr.Recognizer()

def listen_for_command(prompt: str = "Listening for a command...") -> str | None:
    """
    Listens for a single command from the user's microphone.
    Returns:
        The transcribed text in lowercase, or None if an error occurred.
    """
    with sr.Microphone() as source:
        print(prompt)
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            print("Listening timed out. Please try again.")
            return None
        except Exception as e:
            print(f"Error capturing audio: {e}")
            return None


    # Recognize speech using Google Web Speech API
    try:
        print("Recognizing...")
        command = recognizer.recognize_google(audio)
        print(f"User said: {command}")
        return command.lower()
    except sr.UnknownValueError:
        print("Sorry, I did not understand what you said.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google; {e}")
        return None
    except Exception as e:
        print(f"An unknown error occurred during recognition: {e}")
        return None

# usage
if __name__ == "__main__":
    command = listen_for_command("Test: Say something...")
    if command:
        print(f"Command recognized: {command}")