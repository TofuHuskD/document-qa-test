import os
import threading
from dotenv import load_dotenv
import streamlit as st
import PyPDF2
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# Load environment variables from the .env file
load_dotenv()

# Get the OpenAI API key from the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set it in the .env file.")

# Predefined directory for PDF ingestion
PREDEFINED_DIRECTORY = "/Users/aidenzf/Documents/GitHub/document-qa-test/combined"  # Replace with your actual directory

# Global variables to hold the vector store and sections
vector_store = None
all_sections = None
building_in_progress = True


def extract_text_and_sections_from_directory(directory_path):
    """Extract text and sections from all PDFs in a directory."""
    all_texts = []
    all_sections = []
    
    pdf_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.lower().endswith(".pdf")]
    
    if not pdf_files:
        st.error(f"No PDF files found in directory: {directory_path}")
        return all_texts, all_sections

    for pdf_file in pdf_files:
        with open(pdf_file, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            extracted_text = ""
            sections = []
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text() or ""  # Handle potential None
                
                section_headers = [line for line in page_text.split('\n') if "section" in line.lower() or "step" in line.lower()]
                
                for header in section_headers:
                    sections.append((header, page_num))  # Store section title and page number
                extracted_text += page_text
            
            all_texts.append(extracted_text)
            all_sections.append(sections)
    
    return all_texts, all_sections


def build_vector_store(texts):
    """Build a vector store from texts."""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    doc_store = FAISS.from_texts(texts, embeddings)
    return doc_store


def background_database_builder():
    """Background thread function to build the database."""
    global vector_store, all_sections, building_in_progress

    try:
        texts, sections = extract_text_and_sections_from_directory(PREDEFINED_DIRECTORY)
        vector_store = build_vector_store(texts)
        all_sections = sections
    except Exception as e:
        st.error(f"Error building knowledge base: {e}")
    finally:
        building_in_progress = False


def create_formal_rag_chain():
    """Create a formal RAG chain for querying."""
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, max_tokens=1000)

    prompt_template = PromptTemplate(
        template=(
            "You are an expert assistant that provides highly structured and detailed step-by-step instructions "
            "based only on the provided documents (SOPs). Each step you provide must be clearly derived from the content "
            "and must cite the section from where it was taken. If the query cannot be answered from the documents, reply with: "
            "'I do not know based on the provided SOPs.' and ask for clarification.\n\n"
            "Documents: {context}\n\n"
            "Query: {query}\n\n"
            "Provide your answer as a step-by-step guide, ensuring that every step cites a specific section of the SOP."
        ),
        input_variables=["query", "context"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template, "document_variable_name": "context"}
    )


# Streamlit Web App
st.title("SOP handling intelligent agent")

if OPENAI_API_KEY:
    # Automatically start building the database in the background
    if building_in_progress:
        #st.info("Building the knowledge base in the background...")
        threading.Thread(target=background_database_builder).start()

    if building_in_progress:
        st.info("Knowledge base is being built. Please wait...")
    elif vector_store is None or all_sections is None:
        st.error("The knowledge base could not be built. Please check the logs.")
    else:
        # Query Interface
        query = st.text_input("What would you like to know?", placeholder="e.g., What are the steps for section 1.2?")
        if query:
            qa_chain = create_formal_rag_chain()

            with st.spinner("Generating formal step-by-step plan..."):
                context = "\n\n".join(
                    [f"{section[0]} (Page {section[1] + 1})" for section in all_sections[0]]
                )  # Use the first document's sections

                response = qa_chain.invoke({"query": query, "context": context})
                st.write(f"**Response:**\n\n{response['result']}")
else:
    st.warning("Please set your OpenAI API key in the .env file to proceed.")