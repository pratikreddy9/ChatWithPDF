import streamlit as st
import os
from PyPDF2 import PdfReader
from google.generativeai import configure, generate_embeddings, chat
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
from io import BytesIO

# Configure Google Generative AI
GOOGLE_API_KEY = "your_google_api_key_here"  # Replace with your actual API key
configure(api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="Chat With PDFs")

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
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

# Function to split text into manageable chunks
def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# Generate embeddings for text chunks
def generate_text_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        try:
            embedding = generate_embeddings(chunk)["embeddings"]
            embeddings.append(embedding)
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
    return embeddings

# Save embeddings and chunks for later use
def save_embeddings(chunks, embeddings):
    np.save("text_chunks.npy", chunks)
    np.save("text_embeddings.npy", embeddings)

# Load embeddings and chunks
def load_embeddings():
    try:
        chunks = np.load("text_chunks.npy", allow_pickle=True)
        embeddings = np.load("text_embeddings.npy", allow_pickle=True)
        return chunks, embeddings
    except FileNotFoundError:
        st.error("No embeddings found. Please process PDFs first.")
        return None, None

# Find the most similar chunk to the query
def find_similar_chunks(query_embedding, embeddings, top_k=3):
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return top_indices, similarities[top_indices]

# Generate a response using Google Generative AI
def generate_response(context, question):
    prompt = f"""
    Answer the question as detailed as possible based on the provided context. 
    If the context does not contain the information, respond with "Answer not available in the context."
    
    Context: {context}
    Question: {question}
    Answer:
    """
    try:
        response = chat(prompt, temperature=0.3)
        return response["text"]
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Error generating response."

# Main Streamlit app logic
def main():
    st.title("Chat with PDFs")
    st.header("Chat with your uploaded PDF files")
    
    user_question = st.text_input("Ask a question about the PDFs:")
    if user_question:
        chunks, embeddings = load_embeddings()
        if chunks is not None and embeddings is not None:
            try:
                query_embedding = generate_embeddings(user_question)["embeddings"]
                top_indices, _ = find_similar_chunks(query_embedding, embeddings)
                context = " ".join([chunks[i] for i in top_indices])
                response = generate_response(context, user_question)
                st.write("Response:", response)
            except Exception as e:
                st.error(f"Error processing question: {e}")
    
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        chunks = split_text_into_chunks(raw_text)
                        embeddings = generate_text_embeddings(chunks)
                        save_embeddings(chunks, embeddings)
                        st.success("PDFs processed and embeddings saved!")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {e}")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
