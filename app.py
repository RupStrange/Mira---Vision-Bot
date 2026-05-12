import os
import streamlit as st
import tempfile
import base64
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import time

from speech_to_text import transcribe_with_groq
from ai_agent import ask_agent
from text_to_speech import text_to_speech_with_elevenlabs


# =========================================================
# CONFIG
# =========================================================
CAMERA_FRAME_PATH = "cache/current_frame.jpg"

os.makedirs("cache", exist_ok=True)

st.set_page_config(
    page_title="✨ Mira AI Assistant",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.block-container{
    padding-top: 2rem;
}

[data-testid="stSidebar"]{
    background: linear-gradient(
        180deg,
        #0f172a 0%,
        #111827 100%
    );
}

[data-testid="stSidebar"] *{
    color: white;
}

.stChatMessage{
    border-radius: 18px;
    padding: 10px;
}

.stMetric{
    background-color: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

if "latest_audio" not in st.session_state:
    st.session_state.latest_audio = None

if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0


# =========================================================
# LIVE CAMERA STREAM
# =========================================================
class VideoCapture(VideoTransformerBase):

    last_save_time = 0

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        current_time = time.time()

        # Save frame every 2 seconds
        if current_time - self.last_save_time >= 2:

            cv2.imwrite(
                CAMERA_FRAME_PATH,
                img
            )

            self.last_save_time = current_time

        return img


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.title("👁️✨ Mira")

    st.caption(
        "🎙️ Real-Time Multimodal Vision Assistant"
    )

    st.divider()

    st.subheader("📸 Live Camera")

    webrtc_streamer(
        key="mira-camera",
        video_transformer_factory=VideoCapture,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        desired_playing_state=True,
        async_processing=True
    )

    if os.path.exists(CAMERA_FRAME_PATH):

        st.success("🟢 Camera Connected")

    else:

        st.warning("🟡 Waiting for Camera Access")

    st.divider()

    st.subheader("🛠️ How to Use Mira")
    st.subheader("🛠️ How to Use")

    st.markdown("""
    1️⃣ Enable Camera Access  
    2️⃣ Tap the 🎙️ microphone  
    3️⃣ Ask Mira anything  
    4️⃣ 👀 Mira sees + listens  
    5️⃣ 🔊 Get AI voice responses  
    """)
    st.divider()

    st.subheader("💡 Try Asking")

    example_prompts = [
        "👕 What am I wearing?",
        "🏠 Describe my surroundings",
        "📖 Read the text in front of me",
        "📦 What objects can you see?",
        "😊 What expression is on my face?",
        "💻 What's on my screen?"
    ]

    for prompt in example_prompts:
        st.caption(f"• {prompt}")

    st.divider()

    st.subheader("📊 Assistant Status")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "💬 Messages",
            len(st.session_state.messages)
        )

    with col2:
        st.metric(
            "🎤 Voice",
            "ACTIVE"
        )

    st.divider()

    st.caption(
        "⚡ Powered by Groq • ElevenLabs • Llama"
    )


# =========================================================
# MAIN HEADER
# =========================================================
st.title("👁️✨ Mira Vision Assistant")

st.caption(
    "Talk • Listen • See • Understand"
)

st.markdown("""
Your real-time AI assistant capable of:
- 🎙️ Listening to your voice
- 👀 Understanding your environment
- 🧠 Responding intelligently
- 🔊 Speaking back naturally
""")

st.divider()


# =========================================================
# AUTOPLAY AUDIO
# =========================================================
if st.session_state.latest_audio:

    with open(st.session_state.latest_audio, "rb") as f:
        audio_bytes = f.read()

    b64 = base64.b64encode(audio_bytes).decode()

    mime = (
        "audio/wav"
        if st.session_state.latest_audio.endswith(".wav")
        else "audio/mp3"
    )

    st.html(
        f"""
        <audio id="mira-audio" autoplay>
            <source src="data:{mime};base64,{b64}" type="{mime}">
        </audio>

        <script>
            var audio = document.getElementById("mira-audio");
            audio.load();

            audio.play().catch(function(e) {{
                console.log("Autoplay blocked:", e);
            }});
        </script>
        """
    )


# =========================================================
# CHAT SECTION
# =========================================================
st.subheader("💬 Conversation")

if not st.session_state.messages:

    st.info(
        "🌟 Start speaking below to begin interacting with Mira."
    )

else:

    for msg in st.session_state.messages:

        if msg["role"] == "user":

            with st.chat_message("user"):
                st.markdown(f"🧑 **You:** {msg['content']}")

        else:

            with st.chat_message("assistant"):
                st.markdown(f"🤖 **Mira:** {msg['content']}")


# =========================================================
# VOICE INPUT
# =========================================================
st.divider()

st.subheader("🎙️ Speak with Mira")

audio = st.audio_input(
    "Tap the microphone and start speaking...",
    key=f"audio_{st.session_state.audio_key}"
)


# =========================================================
# AUDIO PROCESSING
# =========================================================
if audio is not None:

    current_audio_id = audio.file_id

    if current_audio_id != st.session_state.last_audio_id:

        st.session_state.last_audio_id = current_audio_id

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as f:

            f.write(audio.read())
            audio_path = f.name

        with st.spinner(
            "🧠 Mira is analyzing your voice and surroundings..."
        ):

            user_text = transcribe_with_groq(audio_path)

            response = ask_agent(user_text)

            ai_audio_path = text_to_speech_with_elevenlabs(
                response,
                "cache/response.mp3"
            )

        st.session_state.latest_audio = ai_audio_path

        st.session_state.messages.append({
            "role": "user",
            "content": user_text
        })

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        st.session_state.audio_key += 1

        st.rerun()