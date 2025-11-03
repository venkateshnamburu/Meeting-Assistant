# meeting_assistant.py
# ---------------------------------------------
# 🎧 Smart Meeting Assistant
# Whisper + Gemini 1.5 Flash + Streamlit
# ---------------------------------------------

import streamlit as st
import whisper
import tempfile
import os
import subprocess
from google import genai

# -----------------------
# App Config
# -----------------------
st.set_page_config(page_title="Smart Meeting Assistant", page_icon="🎧", layout="wide")
st.title("🎧 Smart Meeting Assistant")
st.caption("Upload your meeting recording → Auto-transcribe → Summarize → Ask questions")

# -----------------------
# Function: Check ffmpeg
# -----------------------
def check_ffmpeg():
    """Check if ffmpeg is installed on the system."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

if not check_ffmpeg():
    st.error("⚠️ FFmpeg not found! Please install it first.\n\n"
             "➡️ Windows: Download from https://ffmpeg.org/download.html and add it to PATH.")
    st.stop()

# -----------------------
# Gemini Setup (Secure)
# -----------------------
API_KEY = "AIzaSyBrufvfsj2aMXo_qd6WCDRafwZxiQfBh4E"

if not API_KEY:
    st.error("❌ Gemini API key not found! Please set it in your environment variable as 'GEMINI_API_KEY'.")
    st.info("💡 Example (Windows): `setx GEMINI_API_KEY \"your_api_key_here\"` and restart your terminal.")
    st.stop()
else:
    client = genai.Client(api_key=API_KEY)

# -----------------------
# Audio Upload
# -----------------------
audio_file = st.file_uploader("🎤 Upload your meeting recording", type=["mp3", "wav", "m4a"])

# -----------------------
# Transcription with Whisper
# -----------------------
transcript_text = ""
if audio_file is not None:
    st.info("🔄 Transcribing audio... please wait (this may take a few minutes for long files)")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    model = whisper.load_model("tiny")
    result = model.transcribe(tmp_path)
    transcript_text = result["text"]

    st.success("✅ Transcription complete!")
    st.subheader("🗒️ Transcript")
    st.text_area("Full Transcript", transcript_text, height=250)

# -----------------------
# Gemini Summary
# -----------------------
if transcript_text:
    if st.button("📝 Generate Meeting Summary"):
        with st.spinner("Summarizing with Gemini Flash..."):
            prompt = (
                "You are a helpful meeting assistant. Summarize the following meeting transcript. "
                "Include key discussion points, decisions, and action items clearly.\n\n"
                f"Transcript:\n{transcript_text}"
            )
            summary = client.models.generate_content(
                model="gemini-1.5-flash", contents=prompt
            )
        st.subheader("📋 Meeting Summary / MOM")
        st.write(summary.text)

# -----------------------
# Q&A Section
# -----------------------
if transcript_text:
    st.divider()
    st.subheader("💬 Ask Questions About the Meeting")
    question = st.text_input("Type your question here:")
    if question:
        with st.spinner("Thinking..."):
            qa_prompt = (
                f"Based on the following meeting transcript, answer the question clearly and concisely.\n\n"
                f"Transcript:\n{transcript_text}\n\n"
                f"Question: {question}"
            )
            answer = client.models.generate_content(
                model="gemini-1.5-flash", contents=qa_prompt
            )
        st.write("**Answer:** ", answer.text)

# -----------------------
# Footer
# -----------------------
st.divider()
st.caption("Built with 💡 Whisper + Gemini Flash + Streamlit | Runs locally")
