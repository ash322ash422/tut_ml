# app.py
# To run:  python -m  streamlit run  app.py
# Open URL: http://localhost:8501

from openai import OpenAI
import streamlit as st

st.title("ChatGPT-like Clone")

# Sidebar API Key
api_key = st.sidebar.text_input(
    "Enter OpenAI API Key",
    type="password"
)

if not api_key:
    st.info("Please enter your OpenAI API Key in the sidebar.")
    st.stop()

client = OpenAI(api_key=api_key)

# Model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What is up?"):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=st.session_state.messages,
            stream=True,
        )

        response = st.write_stream(stream)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )