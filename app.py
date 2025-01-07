import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import tempfile
import os

def initialize_whisper_pipeline():
    """Initialize the Whisper model pipeline"""
    model_id = "openai/whisper-large-v3"
    
    # Set device and dtype
    device = "cpu"  # Streamlit Cloud uses CPU
    torch_dtype = torch.float32  # Use float32 for CPU
    
    @st.cache_resource  # Cache the model to avoid reloading
    def load_model():
        # Load model with optimized settings
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        processor = AutoProcessor.from_pretrained(model_id)
        
        return model, processor
    
    model, processor = load_model()
    
    # Create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    return pipe

def main():
    st.title("🎙️ Audio Transcription App")
    st.write("Upload an audio file (MP3, WAV, M4A, or OGG) and get its transcription!")
    
    # File upload
    audio_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'm4a', 'ogg'])
    
    if audio_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            with st.spinner("Loading model and processing audio... This might take a few minutes."):
                # Initialize pipeline
                pipe = initialize_whisper_pipeline()
                
                # Perform transcription
                result = pipe(tmp_file_path)
                
                # Display results
                st.success("Transcription completed!")
                st.write("### Transcription:")
                st.write(result["text"])
                
                # Add download button for transcription
                st.download_button(
                    label="Download Transcription",
                    data=result["text"],
                    file_name="transcription.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    # Add usage instructions
    with st.expander("ℹ️ Usage Instructions"):
        st.write("""
        1. Click the 'Browse files' button to upload your audio file
        2. Wait for the model to process your audio (this may take a few minutes)
        3. View the transcription results
        4. Download the transcription as a text file if needed
        
        Note: This app works best with clear audio recordings. Background noise may affect accuracy.
        """)
    
    # Add footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>Made with ❤️ using Streamlit and Whisper</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
