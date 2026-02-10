
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
    model = load_model()
    raw_output = model(text)

    # ✅ Normalize output
    if isinstance(raw_output, list) and isinstance(raw_output[0], list):
        results = raw_output[0]
    else:
        results = []

    st.subheader("Detected Emotions")

    if results:
        for r in results:
            label = r.get("label", "unknown")
            score = r.get("score", 0)
            st.write(f"{label} : {round(score * 100, 2)}%")

        top = max(results, key=lambda x: x["score"])
        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "emotion": top["label"],
            "score": round(top["score"] * 100, 2)
        })
    else:
        st.warning("No emotion detected")

if st.session_state.history:
    st.subheader("Emotion Timeline")
    df = pd.DataFrame(st.session_state.history)
    st.line_chart(df.set_index("time")["score"])
