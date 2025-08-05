import streamlit as st
import google.generativeai as genai
import os

st.set_page_config(page_title="Gemini Image Generator", page_icon=":art:")
st.title("🖼️  2.0 Preview Image Generator")

# API key setup
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except (KeyError, FileNotFoundError):
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        st.error("API key not found. Please set it in Streamlit secrets or as an environment variable.")
        st.stop()

MODEL_NAME = "gemini-2.0-preview"
model = genai.GenerativeModel(MODEL_NAME)

prompt = st.text_area("Enter a prompt to generate an image:")
if st.button("Generate Image") and prompt:
    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "image/png"}
        )
        # Check if image data exists
        parts = response.candidates[0].content.parts
        img_bytes = None
        for part in parts:
            if hasattr(part, "data") and part.data:
                img_bytes = part.data
                break
        if img_bytes:
            st.image(img_bytes, caption="Generated Image", use_container_width=True)
        else:
            st.warning("No image was generated. Try a different prompt or check model/image support.")
    except Exception as e:
        st.error(f"Image generation failed: {e}")