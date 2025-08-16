import streamlit as st

st.set_page_config(page_title="Document Chat", page_icon="📄")

import requests
import time

def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.title("📄 Upload Document and Chat")

if "doc_messages" not in st.session_state:
    st.session_state.doc_messages = []
if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = False

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Submit to Assistant"):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        with st.spinner("Processing document..."):
            response = requests.post("http://127.0.0.1:8000/upload-doc", files=files)

        if response.status_code == 200:
            response_text = response.json()[0]
            with st.chat_message("assistant"):
                streamed_response = st.write_stream(response_generator(response_text))
            st.session_state.doc_messages.append({"role": "assistant", "content": streamed_response})
            st.session_state.doc_uploaded = True
        else:
            st.error("Something went wrong while processing the file.")

if st.session_state.doc_uploaded:
    st.markdown("---")
    st.subheader("Chat about your uploaded document")

    for message in st.session_state.doc_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something about the uploaded document"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.doc_messages.append({"role": "user", "content": prompt})

        response = requests.get("http://127.0.0.1:8000/chat-upload", params={"query": prompt}, verify=False)
        response_text = response.json()["response"]

        with st.chat_message("assistant"):
            streamed_response = st.write_stream(response_generator(response_text))

        st.session_state.doc_messages.append({"role": "assistant", "content": streamed_response})
