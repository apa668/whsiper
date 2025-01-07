import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import tempfile
import os
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_whisper_pipeline():
    """Initialize the Whisper model pipeline"""
    try:
        # Using smaller model better suited for CPU
        model_id = "openai/whisper-medium"
        
        # Set device and dtype
        device = "cpu"
        torch_dtype = torch.float32
        
        st.info("Starting model initialization...")
        
        # Load model with memory optimizations
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map=None
        )
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        st.info("Model loaded successfully...")
        
        processor = AutoProcessor.from_pretrained(model_id)
        st.info("Processor loaded successfully...")
        
        # Create pipeline with memory efficient settings
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=30,  # Process audio in smaller chunks
            batch_size=1
        )
        st.info("Pipeline created successfully...")
        
        return pipe
    
    except Exception as e:
        logger.error(f"Error in initialize_whisper_pipeline: {str(e)}")
        st.error(f"Error initializing model: {str(e)}")
        raise

def transcribe_audio(pipe, audio_path):
    """Transcribe audio file with progress tracking"""
    try:
        return pipe(
            audio_path,
            return_timestamps=False,
            generate_kwargs={"max_new_tokens": 256}
        )
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise

def main():
    st.title("🎙️ Audio Transcription App")
    st.write("Upload an audio file (MP3, WAV, M4A, or OGG) and get its transcription!")
    
    # Memory usage warning
    st.warning("""Note: This app uses the 'whisper-small' model for better compatibility with Streamlit Cloud. 
                While transcription quality might be slightly lower than the large model, it should work more reliably.""")
    
    # File upload with size limit
    st.info("Please keep audio files under 4 minutes for best results")
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
                
                st.info("Starting transcription...")
                # Perform transcription
                result = transcribe_audio(pipe, tmp_file_path)
                
                # Clear memory
                del pipe
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
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
            logger.error(f"Error in main: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")
    
    # Add usage instructions
    with st.expander("ℹ️ Usage Instructions"):
        st.write("""
        1. Click the 'Browse files' button to upload your audio file
        2. Wait for the model to process your audio (this may take a few minutes)
        3. View the transcription results
        4. Download the transcription as a text file if needed
        
        Note: 
        - Keep audio files under 4 minutes for best results
        - Clear audio recordings work best
        - Background noise may affect accuracy
        """)
    
    # Add footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>Made with ❤️ using Streamlit and Whisper</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"Application error: {str(e)}")
