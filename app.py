import os
import streamlit as st
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import numpy as np
import torchaudio

# Install ffmpeg during deployment
os.system("apt-get update && apt-get install -y ffmpeg")

# Function to load the Whisper model
@st.cache_resource
def load_model():
    model_id = "openai/whisper-large-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=0 if device == "cuda" else -1,
    )

# Preprocess audio to ensure it's in the correct format
def preprocess_audio(audio_bytes):
    waveform, sample_rate = torchaudio.load(audio_bytes)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    return waveform

# Main Streamlit App
st.title("Whisper Audio Transcription App")

st.write("Record your audio and get a transcription using OpenAI's Whisper model.")

# Audio Recorder
audio_data = st.audio("Record your audio here:", type="wav")
if audio_data is not None:
    st.write("Recording received. Processing...")
    try:
        # Preprocess and transcribe
        processed_audio = preprocess_audio(audio_data)
        pipe = load_model()
        result = pipe(processed_audio)
        st.write("Transcription:")
        st.success(result["text"])
    except Exception as e:
        st.error(f"An error occurred while processing the audio: {e}")
