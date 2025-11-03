# meeting_assistant_fast.py
# ---------------------------------------------
# 🎧 Smart Meeting Transcriber (Fast Version)
# Whisper + Streamlit
# ---------------------------------------------

import streamlit as st
import whisper
import tempfile
import os
import subprocess

# -----------------------
# App Config
# -----------------------
st.set_page_config(page_title="Smart Meeting Transcriber", page_icon="🎧", layout="wide")
st.title("🎧 Smart Meeting Transcriber")
st.caption("Upload your meeting recording → Get instant transcript (no summary, no Q&A)")

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
# Audio Upload
# -----------------------
audio_file = st.file_uploader("🎤 Upload your meeting recording", type=["mp3", "wav", "m4a"])

# -----------------------
# Transcription with Whisper
# -----------------------
if audio_file is not None:
    st.info("🔄 Transcribing audio... please wait (duration-based timing, usually 1x–2x real-time)")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    # Use the smallest model for speed
    model = whisper.load_model("tiny")  # Options: tiny, base, small, medium, large
    result = model.transcribe(tmp_path)
    transcript_text = result["text"]

    st.success("✅ Transcription complete!")
    st.subheader("🗒️ Transcript")
    st.text_area("Full Transcript", transcript_text, height=350)

    # Optionally allow download
    st.download_button(
        label="⬇️ Download Transcript as TXT",
        data=transcript_text,
        file_name="meeting_transcript.txt",
        mime="text/plain"
    )

# -----------------------
# Footer
# -----------------------
st.divider()
st.caption("Built with ⚡ Whisper + Streamlit | Optimized for Speed")
