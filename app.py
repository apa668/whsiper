import streamlit as st
from streamlit_mic_recorder import speech_to_text
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    model_id = "openai/whisper-large-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
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

# Record and process audio
def record_voice(language="en"):
    state = st.session_state

    if "text_received" not in state:
        state.text_received = []

    # Record and transcribe using streamlit-mic-recorder
    text = speech_to_text(
        start_prompt="🎤 Click and speak to record audio",
        stop_prompt="⚠️ Stop recording 🚨",
        language=language,
        use_container_width=True,
        just_once=True,
    )

    if text:
        state.text_received.append(text)

    result = "".join(state.text_received)
    state.text_received = []  # Clear the state after use
    return result if result else None


# Main App
st.title("Streamlit Voice Recorder and Transcription")

st.write("Record your voice and get a transcription using Whisper or streamlit-mic-recorder.")

# Record voice
transcribed_text = record_voice(language="en")

if transcribed_text:
    st.write("Transcribed Text (from streamlit-mic-recorder):")
    st.success(transcribed_text)

    # Optional: Process text using Whisper
    use_whisper = st.checkbox("Use Whisper for transcription (optional)")
    if use_whisper:
        st.write("Processing with Whisper...")
        whisper_pipeline = load_whisper_model()
        result = whisper_pipeline(transcribed_text)
        st.write("Transcription (Whisper):")
        st.success(result["text"])
