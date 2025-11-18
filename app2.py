# fast_transcriber_with_progress.py
# ---------------------------------------------------
# ⚡ Fast Meeting Transcriber (Groq + Whisper + Chunked + Progress Bar)
# ---------------------------------------------------

import streamlit as st
from groq import Groq
import sys

# --- Compatibility Patch for Python 3.13 (pyaudioop missing) ---
if sys.version_info >= (3, 13):
    import audioop
    sys.modules['pyaudioop'] = audioop

from pydub import AudioSegment
import tempfile
import os
import math
import time

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="Fast Meeting Transcriber", page_icon="⚡", layout="wide")
st.title("⚡ Fast Meeting Transcriber")
st.caption("Upload long audio → Auto-split → Transcribe instantly with Groq Whisper")

# ---------------------------
# Groq API Setup
# ---------------------------
API_KEY = "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"

if not API_KEY:
    st.error("❌ Groq API key not found! Please set it as an environment variable or in Streamlit secrets.")
    st.info("Get your free API key from https://console.groq.com/keys")
    st.stop()

client = Groq(api_key=API_KEY)

# ---------------------------
# Helper Function: Split Audio
# ---------------------------
def split_audio(file_path, max_size_mb=23):
    """Split large audio into smaller chunks under given MB size."""
    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    num_parts = math.ceil(file_size_mb / max_size_mb)
    chunk_length = duration_ms / num_parts

    chunks = []
    for i in range(num_parts):
        start = i * chunk_length
        end = (i + 1) * chunk_length
        chunk = audio[start:end]
        tmp_chunk = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        chunk.export(tmp_chunk.name, format="mp3")
        chunks.append(tmp_chunk.name)
    return chunks

# ---------------------------
# Transcription Logic
# ---------------------------
audio_file = st.file_uploader("🎤 Upload meeting audio file", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    st.info("🔄 Processing... splitting large file if needed")
    chunks = split_audio(tmp_path)

    total_chunks = len(chunks)
    st.write(f"🔹 Audio split into {total_chunks} part(s) for processing")

    progress_bar = st.progress(0)
    status_text = st.empty()
    full_transcript = ""

    for idx, chunk_path in enumerate(chunks, start=1):
        status_text.text(f"⏳ Transcribing chunk {idx}/{total_chunks}...")
        with open(chunk_path, "rb") as f:
            try:
                result = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=f
                )
                full_transcript += result.text + "\n"
            except Exception as e:
                st.error(f"❌ Error on chunk {idx}: {e}")

        # Update progress bar
        progress = int((idx / total_chunks) * 100)
        progress_bar.progress(progress)
        time.sleep(0.3)

    status_text.text("✅ Transcription complete!")
    progress_bar.progress(100)

    # ---------------------------
    # Display Results
    # ---------------------------
    st.subheader("🗒️ Full Transcript")
    st.text_area("Transcribed Text", full_transcript, height=300)

    st.download_button(
        label="📥 Download Transcript as TXT",
        data=full_transcript,
        file_name="meeting_transcript.txt",
        mime="text/plain"
    )

# ---------------------------
# Footer
# ---------------------------
st.divider()
st.caption("Built with ⚡ Groq Whisper + Streamlit | Auto-chunked for long audio | Progress bar enabled")


