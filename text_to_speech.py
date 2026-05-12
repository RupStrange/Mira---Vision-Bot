import os
import pyttsx3
from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def text_to_speech_with_elevenlabs(input_text, output_filepath):
    """Converts text to speech using ElevenLabs and saves to file.
    Falls back to pyttsx3 if ElevenLabs quota is exceeded."""
    try:
        voice_id = "cgSgspJ2msm6clMCkdW9"
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=input_text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        with open(output_filepath, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        return output_filepath

    except ApiError as e:
        print(f"⚠️ ElevenLabs quota exceeded — falling back to pyttsx3. ({e})")
        return _pyttsx3_fallback(input_text, output_filepath)

    except Exception as e:
        print(f"⚠️ ElevenLabs error — falling back to pyttsx3. ({e})")
        return _pyttsx3_fallback(input_text, output_filepath)


def _pyttsx3_fallback(text: str, output_path: str) -> str:
    """
    Offline TTS using pyttsx3. Saves as .wav (pyttsx3 doesn't support mp3).
    Returns the actual saved path.
    """
    # pyttsx3 only saves .wav reliably
    wav_path = output_path.replace(".mp3", ".wav")

    engine = pyttsx3.init()
    engine.setProperty("rate", 165)    # speed (words per minute)
    engine.setProperty("volume", 1.0)  # 0.0 – 1.0

    # Optional: pick a female voice if available
    voices = engine.getProperty("voices")
    for voice in voices:
        if "female" in voice.name.lower() or "zira" in voice.name.lower():
            engine.setProperty("voice", voice.id)
            break

    engine.save_to_file(text, wav_path)
    engine.runAndWait()

    print(f"✅ pyttsx3 TTS saved to {wav_path}")
    return wav_path