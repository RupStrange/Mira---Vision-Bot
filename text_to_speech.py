# import os
# from elevenlabs.client import ElevenLabs
# from elevenlabs.core.api_error import ApiError
# from dotenv import load_dotenv

# load_dotenv()

# client = ElevenLabs(
#     api_key=os.getenv("ELEVENLABS_API_KEY")
# )

# def text_to_speech_with_elevenlabs(input_text, output_filepath):
#     """
#     Converts text to speech using ElevenLabs and saves to file.
#     """

#     try:
#         voice_id = "cgSgspJ2msm6clMCkdW9"

#         audio = client.text_to_speech.convert(
#             voice_id=voice_id,
#             text=input_text,
#             model_id="eleven_multilingual_v2",
#             output_format="mp3_44100_128"
#         )

#         with open(output_filepath, "wb") as f:
#             for chunk in audio:
#                 f.write(chunk)

#         return output_filepath

#     except ApiError as e:
#         print(f"ElevenLabs API Error: {e}")
#         return None

#     except Exception as e:
#         print(f"TTS Error: {e}")
#         return None



import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError
from gtts import gTTS

load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

def _gtts_fallback(text, output_path):
    """
    Free fallback TTS using Google TTS
    """
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(output_path)

        print("✅ gTTS fallback successful")
        return output_path

    except Exception as e:
        print(f"❌ gTTS fallback failed: {e}")
        return None


def text_to_speech_with_elevenlabs(input_text, output_filepath):
    """
    Converts text to speech using ElevenLabs.
    Falls back to gTTS if ElevenLabs fails.
    """

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

        print("✅ ElevenLabs TTS successful")
        return output_filepath

    except ApiError as e:
        print(f"⚠️ ElevenLabs API Error: {e}")
        print("🔄 Switching to gTTS fallback...")

        return _gtts_fallback(input_text, output_filepath)

    except Exception as e:
        print(f"⚠️ TTS Error: {e}")
        print("🔄 Switching to gTTS fallback...")

        return _gtts_fallback(input_text, output_filepath)
