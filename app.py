
import streamlit as st
import requests
import os
import io
import base64
import numpy as np
import soundfile as sf

from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")

# NVIDIA API endpoints
ASR_URL = (
    "https://1598d209-5e27-4d3c-8079-4751568b1081"
    ".invocation.api.nvcf.nvidia.com/v1/audio/transcriptions"
)

IMAGE_URL = (
    "https://ai.api.nvidia.com/v1/genai/"
    "black-forest-labs/flux.1-schnell"
)

st.set_page_config(
    page_title="AI Speech to Image",
    page_icon="🎨",
    layout="centered"
)

st.title("🎙️ AI Speech to Image Generator")
st.write("Speak your imagination and turn it into an image!")
st.caption("Powered by NVIDIA AI")

if not API_KEY:
    st.error("Please set NVIDIA_API_KEY in your .env file.")
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}


# Convert microphone audio to mono WAV
def prepare_audio(audio_file):
    audio_bytes = audio_file.getvalue()

    samples, sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32"
    )

    # Convert stereo to mono
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    # Encode as 16-bit PCM WAV
    wav_buffer = io.BytesIO()
    sf.write(
        wav_buffer,
        samples,
        sample_rate,
        format="WAV",
        subtype="PCM_16"
    )

    return wav_buffer.getvalue()


# Speech to text using NVIDIA Parakeet
def speech_to_text(audio_bytes):
    response = requests.post(
        ASR_URL,
        headers=HEADERS,
        files={
            "file": ("input.wav", audio_bytes, "audio/wav")
        },
        data={
            "language": "en-US",
            "word_time_offsets": "False"
        },
        timeout=120
    )

    response.raise_for_status()
    result = response.json()

    return result.get("text", "").strip()


# Generate image using NVIDIA FLUX
def generate_image(prompt):
    payload = {
        "prompt": prompt,
        "height": 1024,
        "width": 1024,
        "cfg_scale": 0,
        "mode": "base",
        "samples": 1,
        "seed": 0,
        "steps": 4
    }

    headers = {
        **HEADERS,
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    response = requests.post(
        IMAGE_URL,
        headers=headers,
        json=payload,
        timeout=180
    )

    response.raise_for_status()
    result = response.json()

    artifacts = result.get("artifacts", [])

    if not artifacts:
        raise ValueError(
            f"No image returned by NVIDIA: {result}"
        )

    image_base64 = artifacts[0].get("base64")

    if not image_base64:
        raise ValueError("Missing base64 image data.")

    return base64.b64decode(image_base64)


# Audio input
st.subheader("Step 1: Record your voice")

audio = st.audio_input("Click to record your description")

if audio is not None:
    st.audio(audio)

    if st.button(
        "✨ Generate Image",
        type="primary",
        use_container_width=True
    ):
        try:
            with st.spinner("Converting speech to text..."):
                audio_bytes = prepare_audio(audio)
                transcript = speech_to_text(audio_bytes)

            if not transcript:
                st.warning("No speech detected. Please try again.")
                st.stop()

            st.success("Speech converted successfully!")

            st.subheader("Step 2: Recognized Text")
            st.write(transcript)

            with st.spinner("Generating your AI image..."):
                image_bytes = generate_image(transcript)

            st.success("Image generated successfully!")

            st.subheader("Step 3: Generated Image")
            st.image(
                image_bytes,
                caption=transcript,
                use_container_width=True
            )

            st.download_button(
                label="⬇️ Download Image",
                data=image_bytes,
                file_name="generated_image.png",
                mime="image/png",
                use_container_width=True
            )

        except requests.HTTPError as e:
            details = (
                e.response.text[:1000]
                if e.response is not None
                else str(e)
            )
            st.error(f"NVIDIA API error: {details}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
