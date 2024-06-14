import streamlit as st
import os
import tempfile
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import time

from dotenv import load_dotenv
load_dotenv()

# Load the NVIDIA API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            return tmpfile.name
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

def fetch_vectorstore(documents, embeddings):
    if not documents:
        st.error("No documents were loaded from the file.")
        return None
    if not all(doc.page_content.strip() for doc in documents):  # Ensure all documents have content
        st.error("One or more documents are empty after splitting.")
        return None
    return FAISS.from_documents(documents, embeddings)

def vector_embedding(file_path):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        loader = PyPDFLoader(file_path)
        documents = loader.load_and_split()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents)
        st.session_state.vectors = fetch_vectorstore(st.session_state.final_documents, st.session_state.embeddings)
        if st.session_state.vectors is None:
            return  # Exit if vector store creation failed

st.title("NVIDIA NIM Demo")
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Always give the answer in detail and perfect explanation and keep the answer concise.\
<context>
{context}
<context>
Questions:{input}"""
qa_prompt = ChatPromptTemplate.from_template(qa_system_prompt)

# Initialize chat history if it doesn't exist in the session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Sidebar for file imports and processing
with st.sidebar:
    uploaded_file = st.file_uploader("Upload PDF file", type='pdf')
    if st.button("Process Uploaded PDF"):
        if uploaded_file is not None:
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                vector_embedding(file_path)
                st.success("Vector Store DB is ready.")
                # Reset chat history after successfully processing a new file
                st.session_state.chat_history = []
                os.unlink(file_path)  # Optional: remove the file after processing
        else:
            st.error("Please upload a PDF file to proceed.")

# Main interface for question input and display
prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Ask") and prompt1:
    if 'vectors' in st.session_state and st.session_state.vectors is not None:
        retriever = st.session_state.vectors.as_retriever()
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": prompt1, "chat_history": st.session_state.chat_history})
        # Assuming response['answer'] gives a properly formatted message
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt1),
            HumanMessage(content=response['answer'].content if hasattr(response['answer'], 'content') else response['answer'])
        ])
        st.write(response['answer'].content if hasattr(response['answer'], 'content') else response['answer'])
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.error("Please upload and process a PDF first or ensure vectors are properly loaded.")

