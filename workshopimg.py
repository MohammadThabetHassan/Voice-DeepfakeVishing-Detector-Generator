import streamlit as st
import google.generativeai as genai
import os
st.set_page_config(page_title="Gemini Chatbot", page_icon=":robot_face:")
st.title("🤖 Mohammad Chatbot")
st.caption("A friendly AI assistant ")
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
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

for message in st.session_state.chat_session.history:
    with st.chat_message(name=message.role):
        st.markdown(message.parts[0].text)

user_input = st.chat_input("Ask me anything...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    try:
        gemini_response = st.session_state.chat_session.send_message(user_input)
        with st.chat_message("model"):
            st.markdown(gemini_response.text)

    except Exception as e:
        st.error(f"An error occurred: {e}")