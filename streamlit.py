import streamlit as st
import requests
import os


st.title("AI-Task")


option = st.selectbox("Choose an action:", ["Upload file", "Ask Question"])

if option == "Upload file":
    st.header("Upload file")


    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        pdf_path = os.path.join("e:\\Coding\\Projects\\PythonProjects\\AI-Task\\uploaded_pdfs\\", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        response = requests.post("http://localhost:5000/create_embeddings", json={'fileName': uploaded_file.name})
        if response.status_code == 200:
            st.success(response.json()['message'])
        else:
            st.error("Error creating embeddings.")

elif option == "Ask Question":
    st.header("Ask Question")

    question = st.text_input("Enter your question:")
    if st.button("Ask"):

        response = requests.post("http://localhost:5000/ask_question", json={'question': question})
        if response.status_code == 200:
            st.success(response.json()['answer'])
        else:
            st.error("Error asking the question.")

