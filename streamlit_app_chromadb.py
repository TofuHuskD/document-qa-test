
import streamlit as st
import sqlite3
import os
import re
import PyPDF2
from chromadb import Client
from chromadb.config import Settings

# ChromaDB configuration
CHROMADB_PATH = '/path/to/your/vector.db'  # Specify your vector database path
client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMADB_PATH))

# Create or get your collection
collection = client.get_collection("your_collection_name")  # Replace with your collection name

# Initialize session state
if 'documents_uploaded' not in st.session_state:
    st.session_state['documents_uploaded'] = False

# Document Processing Function
def process_documents(uploaded_files):
    """Processes uploaded documents: extracts text and inserts into ChromaDB."""
    total_files = len(uploaded_files)
    progress_bar = st.progress(0)

    for idx, uploaded_file in enumerate(uploaded_files, 1):
        try:
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
                section = "Full Document"  # Modify as needed
            else:
                text = uploaded_file.getvalue().decode("utf-8")
                section = "Full Document"

            if text:
                # Preprocess the extracted text
                text = preprocess_text(text)

                # Add document to ChromaDB
                collection.add(documents=[text], metadatas=[{"source": uploaded_file.name, "section": section}])

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")

        # Update progress
        progress = idx / total_files
        progress_bar.progress(progress)

    st.session_state['documents_uploaded'] = True
    st.success("🎉 New documents processed and added to the knowledge base successfully!")

# Query Handling Function
def query_knowledge(query_text, num_results=5):
    """Function to query knowledge from the ChromaDB."""
    results = collection.query(query_texts=[query_text], n_results=num_results)
    documents = results['documents']
    return documents

# Streamlit User Interface for Document Upload
with st.sidebar.expander("📂 Upload Documents"):
    uploaded_files = st.file_uploader("Upload documents (TXT, PDF):", type=["txt", "pdf"], accept_multiple_files=True)
    if uploaded_files:
        if st.button("Start Upload and Processing"):
            process_documents(uploaded_files)

# Streamlit User Interface for Query
if st.session_state['documents_uploaded']:
    query = st.text_input("Enter your query:")
    if st.button("Search"):
        if query.strip() == "":
            st.error("❌ Query cannot be empty.")
        else:
            with st.spinner("🔍 Generating answer..."):
                results = query_knowledge(query)
                st.subheader("💡 Answers:")
                for doc in results:
                    st.write(doc)
