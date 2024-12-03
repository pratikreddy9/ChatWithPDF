Share


You said:
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from io import BytesIO

# Configure Streamlit
st.set_page_config(page_title="Chat With PDFs")

# Your Google API Key
GENAI_API_KEY = "AIzaSyAWeNKsOj_pSoqvbsMz1tkYkGEhsJLzgR8"
genai.configure(api_key=GENAI_API_KEY)

# Function to extract text from PDFs
def getPdfText(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(BytesIO(pdf.read()))
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

# Split text into chunks
def getTextChunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Create FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Create a conversational chain
def get_conversational_chain():
    Prompt_Template = """ 
Answer the question as detailed as possible based on the provided context. 
If the context does not contain the information, respond with "Answer not available in the context." 
Do not fabricate an answer.\n\n
Context:\n {context}\n
Question: \n{question}\n
Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=Prompt_Template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Handle user input and respond
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {
                "input_documents": docs, "question": user_question
            }, return_only_outputs=True)
        st.write("Reply: ", response.get("output_text", "No output text found"))
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# Main function
def main():
    st.title("Chat with PDF")
    user_question = st.text_input("Ask a question based on the uploaded PDF(s)")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    try:
                        raw_text = getPdfText(pdf_docs)
                        text_chunks = getTextChunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Processing error: {e}")
                else:
                    st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()


i have this code thats deployed on streamlit cloud via github i cant keep api key in script i have the .env file 

Google_API_KEY=AIzaSyAWeNKsOj_pSoqvbsMz1tkYkGEhsJLzgR8

with this 

can we use that inthe script
ChatGPT said:
ChatGPT
Yes, you can securely use the API key from the .env file in your script by leveraging the python-dotenv library to load the environment variables. Here’s how you can update your script:

Steps:
Install the python-dotenv library if not already installed:

bash
Copy code
pip install python-dotenv
Update your .env file to include the key:

plaintext
Copy code
Google_API_KEY=AIzaSyAWeNKsOj_pSoqvbsMz1tkYkGEhsJLzgR8
Modify your script to load the API key securely using dotenv:

python
Copy code
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from io import BytesIO
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get the API key from .env
GENAI_API_KEY = os.getenv("Google_API_KEY")

# Ensure the API key is set
if not GENAI_API_KEY:
    st.error("Google API key not found. Please check your .env file.")
    st.stop()

# Configure the API
genai.configure(api_key=GENAI_API_KEY)

# Configure Streamlit
st.set_page_config(page_title="Chat With PDFs")


# Function to extract text from PDFs
def getPdfText(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(BytesIO(pdf.read()))
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

# Split text into chunks
def getTextChunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Create FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Create a conversational chain
def get_conversational_chain():
    Prompt_Template = """ 
Answer the question as detailed as possible based on the provided context. 
If the context does not contain the information, respond with "Answer not available in the context." 
Do not fabricate an answer.\n\n
Context:\n {context}\n
Question: \n{question}\n
Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=Prompt_Template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Handle user input and respond
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {
                "input_documents": docs, "question": user_question
            }, return_only_outputs=True)
        st.write("Reply: ", response.get("output_text", "No output text found"))
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# Main function
def main():
    st.title("Chat with PDF")
    user_question = st.text_input("Ask a question based on the uploaded PDF(s)")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    try:
                        raw_text = getPdfText(pdf_docs)
                        text_chunks = getTextChunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Processing error: {e}")
                else:
                    st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
