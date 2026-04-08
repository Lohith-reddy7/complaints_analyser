import speech_recognition as sr

def transcribe_audio(audio_file_object):
    """
    Takes an audio file object (from st.audio_input) and uses
    SpeechRecognition to transcribe the Kannada audio to text.
    """
    recognizer = sr.Recognizer()
    
    try:
        # Load the audio file into the recognizer
        with sr.AudioFile(audio_file_object) as source:
            # You can also use recognizer.adjust_for_ambient_noise(source) if needed
            audio_data = recognizer.record(source)
            
        # Call Google's free API with Kannada localization
        text = recognizer.recognize_google(audio_data, language="kn-IN")
        return {"success": True, "text": text}
        
    except sr.UnknownValueError:
        return {"success": False, "error": "Could not understand or identify Kannada audio."}
    except sr.RequestError as e:
        return {"success": False, "error": f"API Request failed; {e}"}
    except Exception as e:
        return {"success": False, "error": f"An error occurred reading the audio: {e}"}
