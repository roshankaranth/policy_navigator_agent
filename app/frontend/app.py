import streamlit as st

st.set_page_config( page_title="Policy Navigator – General Chat", page_icon="📜")

import time
import requests

def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.title("📜 Policy Navigator – General Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a policy or legal question"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = requests.get("http://127.0.0.1:8000/chat", params={"query": prompt}, verify=False)
    response_text = response.json()[0]

    with st.chat_message("assistant"):
        streamed_response = st.write_stream(response_generator(response_text))

    st.session_state.messages.append({"role": "assistant", "content": streamed_response})
