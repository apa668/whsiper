import streamlit as st
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from io import BytesIO
import wave
import numpy as np

# Set up device and model
st.title("Audio Transcription App")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

@st.cache_resource
def load_model():
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

pipe = load_model()

# Function to convert audio bytes to numpy array
def read_audio(file):
    with wave.open(file, "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        return audio, wav_file.getframerate()

# Option to record audio (using streamlit-audiorecorder-like functionality)
if "audio_data" not in st.session_state:
    st.session_state["audio_data"] = None

def record_audio():
    st.session_state["audio_data"] = st.audio("audio/record")

# Audio recording or uploading
st.header("Record or Upload Audio")
option = st.radio("Choose an option:", ("Record", "Upload"))

if option == "Record":
    if st.button("Start Recording"):
        record_audio()
    audio_file = st.session_state["audio_data"]
elif option == "Upload":
    audio_file = st.file_uploader("Upload an audio file (MP3/WAV):", type=["wav", "mp3"])

# Transcription
if audio_file:
    st.audio(audio_file, format="audio/wav")
    st.write("Processing audio...")

    if isinstance(audio_file, BytesIO):
        audio_bytes = BytesIO(audio_file.read())
    else:
        audio_bytes = audio_file

    audio_data, sample_rate = read_audio(audio_bytes)

    result = pipe({"array": audio_data, "sampling_rate": sample_rate})
    st.write("**Transcription:**")
    st.text(result["text"])
