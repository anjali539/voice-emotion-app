"""
import streamlit as st
import pandas as pd
from datetime import datetime
from transformers import pipeline

st.set_page_config(page_title="Emotion Detection App")

st.title("🧠 Emotion Detection App")
st.write("Enter text and see detected emotions")

text = st.text_area("Enter text")

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=True
    )

if "history" not in st.session_state:
    st.session_state.history = []

if text:
    emotion_model = load_model()   # 🔴 model loads ONLY after text
    results = emotion_model(text)[0]

    st.subheader("Detected Emotions")
    #for r in results:
        #st.write(f"{r['label']} : {round(r['score']*100, 2)}%")
    st.subheader("Detected Emotions")

if isinstance(results, list) and isinstance(results[0], dict):
    for r in results:
        st.write(f"{r['label']} : {round(r['score']*100, 2)}%")
else:
    st.write("Model output:", results)

    #top = max(results, key=lambda x: x["score"])
    if isinstance(results, list) and isinstance(results[0], dict):
        top = max(results, key=lambda x: x["score"])
    else:
        top = {"label": str(results), "score": 1.0}

    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "emotion": top["label"],
        "score": round(top["score"] * 100, 2)
    })

if st.session_state.history:
    st.subheader("Emotion Timeline")
    df = pd.DataFrame(st.session_state.history)
    st.line_chart(df.set_index("time")["score"])"""

import streamlit as st
import pandas as pd
from datetime import datetime
from transformers import pipeline
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Emotion Detection App")

st.title("🧠 Emotion Detection App")
st.write("Detect emotions from **text or voice**")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=True
    )

emotion_model = load_model()

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- TEXT INPUT ----------------
st.subheader("✍️ Text Input")
text_input = st.text_area("Enter text")

# ---------------- VOICE INPUT ----------------
st.subheader("🎤 Voice Input")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_buffer = []

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray()
        self.audio_buffer.append(audio)
        return frame

ctx = webrtc_streamer(
    key="voice",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

voice_text = None

if ctx.audio_processor and len(ctx.audio_processor.audio_buffer) > 20:
    audio_data = np.concatenate(ctx.audio_processor.audio_buffer).tobytes()

    audio = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)
    try:
        voice_text = ctx.audio_processor.recognizer.recognize_google(audio)
        st.success(f"Recognized Speech: {voice_text}")
        ctx.audio_processor.audio_buffer.clear()
    except:
        st.warning("Could not recognize voice")

# ---------------- FINAL INPUT ----------------
final_text = text_input or voice_text

# ---------------- EMOTION DETECTION ----------------
if final_text:
    results = emotion_model(final_text)[0]

    st.subheader("🎯 Detected Emotions")
    for r in results:
        st.write(f"{r['label']} : {round(r['score'] * 100, 2)}%")

    top = max(results, key=lambda x: x["score"])

    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "emotion": top["label"],
        "score": round(top["score"] * 100, 2)
    })

# ---------------- TIMELINE ----------------
if st.session_state.history:
    st.subheader("📈 Emotion Timeline")
    df = pd.DataFrame(st.session_state.history)
    st.line_chart(df.set_index("time")["score"])

