import base64
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Canonical path where app.py writes the latest camera frame
CAMERA_FRAME_PATH = "cache/current_frame.jpg"


def analyze_image_with_query(query: str) -> str:
    """
    Use this tool when the user asks something that requires seeing through
    the webcam — e.g. 'what am I wearing?', 'what do you see?', 'how many
    people are here?'. Reads the latest frame saved to disk by Streamlit.
    """
    if not query:
        return "Error: 'query' field is required"

    # Read fresh from disk every time — avoids stale globals across Streamlit reruns
    if not os.path.exists(CAMERA_FRAME_PATH):
        return (
            "I can't see anything right now — the camera doesn't have a frame yet. "
            "Try pointing your camera and asking again."
        )

    with open(CAMERA_FRAME_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are Mira, a witty, clever, and helpful multimodal voice assistant.

Rules:
- Answer naturally, concisely, and in a voice-friendly way.
- If the user uses first-person words like "I", "me", "my", or "mine", preserve that perspective.
- Never rewrite user perspective in tool or vision queries.
- Analyze the provided image carefully and answer based on visible evidence.
- If something is unclear in the image, say so briefly instead of guessing.
"""
        ),
        (
            "human",
            [
                {"type": "text", "text": "{query}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,{img_b64}"
                    }
                }
            ]
        )
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query, "img_b64": img_b64})