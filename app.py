import os
import streamlit as st
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torchaudio
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import numpy as np
import av

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
def preprocess_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    return waveform

# WebRTC audio processor
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio(self, frame: av.AudioFrame):
        self.frames.append(frame)
        return frame

    def save_audio(self, file_path: str):
        audio_data = b"".join([frame.to_ndarray().tobytes() for frame in self.frames])
        with open(file_path, "wb") as f:
            f.write(audio_data)

# Main Streamlit App
st.title("Whisper Audio Transcription App")

st.write("Record your audio and get a transcription using OpenAI's Whisper model.")

# Audio Recorder using WebRTC
ctx = webrtc_streamer(key="audio", mode="sendonly", audio_processor_factory=AudioProcessor)

if st.button("Stop Recording and Transcribe"):
    if ctx.audio_processor:
        file_path = "recorded_audio.wav"
        ctx.audio_processor.save_audio(file_path)

        st.write("Recording saved. Processing...")
        try:
            # Preprocess and transcribe
            processed_audio = preprocess_audio(file_path)
            pipe = load_model()
            result = pipe(processed_audio)
            st.write("Transcription:")
            st.success(result["text"])
        except Exception as e:
            st.error(f"An error occurred while processing the audio: {e}")
    else:
        st.error("No audio processor found. Please record your audio first.")
