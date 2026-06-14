"""
ATLAS AI — Streamlit frontend that talks to the FastAPI backend.

Run locally:
    streamlit run streamlit_app.py

Env vars (.env):
    API_BASE_URL=http://localhost:8000
"""

import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="ATLAS AI", page_icon="🤖")
st.title("🤖 ATLAS AI")
st.caption(f"Backend: {API_BASE_URL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask ATLAS AI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/chat",
                    json={"message": prompt},
                    timeout=60,
                )
                resp.raise_for_status()
                text = resp.json()["response"]
            except requests.RequestException as e:
                text = f"Error contacting backend: {e}"
            st.markdown(text)

    st.session_state.messages.append({"role": "assistant", "content": text})