import os
import re
import streamlit as st
import openai
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Retrieve the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="SOPhia", page_icon="🤵‍♀️")
st.title("📄 SOP handling intelligent agent 🤵‍♀️")

# Predefined directory to load PDF files
PREDEFINED_DIRECTORY =os.path.join(os.getcwd(), "combined")# Replace with the actual path
# Disclaimer 
with st.expander("Disclaimer", expanded=True):
    st.write("""

IMPORTANT NOTICE: This web application is developed as a proof-of-concept prototype. The information provided here is NOT intended for actual usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.

Always consult with qualified professionals for accurate and personalized advice.

""")

about_tab, methodology_tab = st.tabs(["About Us", "Methodology"])

with about_tab:
    st.header("About Us")
    st.markdown("""
   
### Project Scope
	
The scope of the project is to create a knowledge management Retrieval-Augmented Generation (RAG) chatbot that ingest various Standard Operating Procedure (SOP) documents from Operations Command Centre (OCC) and to churn out quick and accurate responses to any user queries related to these SOPs. 

As part of the project scope, the SOPhia team had spoken to the users to gather their requirements, shared the prototype for a simple tryout and received valuable feedbacks, and incorporated most of the features that are deemed useful for operations. This was done to ensure that the project 	is driven by user requirements, addresses the problem statement, and curated to fit the needs of the users.

### Objectives

This project serves to address two primary objectives :

1. Provide OCC users with a knowledge retention tool that is capable of fetching data from a ever-growing pool of SOP documents and providing near real-time responses to users on queries related to their SOPs. As traffic situation on roads are dynamic and fast-moving, there are certain types 	of incident that are not often faced by OCC operators, and SOPs for such cases might not be at the fingertips of even the senior operators. The 	chatbot could potentially plug these gaps with readily available and accurate information on how to respond quickly to such cases.

2. To aid OCC in transiting new OCC staff into full time roles in the quickest way possible. The chatbot could be deployed instantly for the new staff to use for real-time operations, used as a tool to supplement the training of new staffs, and for familiarisation of SOP. 

### Data Sources

Sources are provided by OCC with a data sensitivity of "Official (Closed)".

[1]   Abnormal Ops & Special Requirements
- Handling of Ad-hoc Urgent & Emergency roadworks
- Handling & Managing of Major incidents
- Handling of systems failure
- Handling Unplanned Expressway,Tunnel & Road closure
- Handling VIP Movement
- Handling of Special Scheduled Events and Planned  Road works involving total closure
- Sentosa Gateway Operations
                
[2]   Emergency Crisis Ops
- Handling a Tunnel Fire
- Facility Building Fire
- Handling critical Infrastructure or Road damage or Collapse
- Handling Terrorist Related Incident
- Handling Hazmat Incident
- Short Tunnels
                
[3]   General Information
- Div Quality policy Objective
- Singapore Road Network
- Overview of various equipment & system in OCC
- Function of OCC & Zonal concept
- Staff Roles, Responsibilities & Guidelines
- Communications Protocol
                
[4]   Normal Daily Operations
- Shift Handing Over & Taking Over
- Peak Hour Traffic Routine
- Off Peak Hour Routine
- Monitoring & Reporting of equip & sys fault
- Handling & Managing of Road incidents
- Checking of Incident,Accident,Patrol Report
- Reporting Of Damaged Road Furniture
- Coordinating with TP, TW & other external agencies
- Handling customer complaints & Feedback
- Handling of Abandon Items & Vehicle
- Coordination with other OCC
- Control of access to FBs
- Work Permit
- Sentosa Gateway
- Arterial Road Operations
    
### Features
	
The project was designed to include the following features:

1. A side bar with "Upload Document" section that allows upload of PDF and TXT files.
2. A .env file must be loaded in for the chatbot to work. The API key is not coded within.
3. A query tab for users to input their prompts
4. A history tab for users to refer back to the their last prompts and responses from the chatbot.

Some additional user friendly features are:

1. After the document is uploaded, the system stores these documents into a vector database for future retrieval, without the need for the chatbot 	to read the database again when another query is entered. 
2. Every action taken by users in the application is met with a response from the chatbot to inform users that their actions were captured and 	processed.
3. Every response from the chatbot cites the document retrieved and the section it has retrieved from so that users can refer back to the SOP 	document if needed.
4. Query tab is cleared after every query is entered and a corresponding response was given.

""")


with methodology_tab:
    st.header("Method Writeup")
    st.markdown("""

    ### System Architecture

    1. User Interface (UI): Streamlit is utilized to create an interactive web interface, allowing users to upload documents, query SOPs, and view query history.
    2. Data Storage: An SQLite database serves as the primary datastore for storing document content and metadata, ensuring efficient retrieval and management of documents.
    3. Vector Store: FAISS (Facebook AI Similarity Search) is employed for high-performance similarity search and document retrieval, enabling quick responses to user queries.
    4. Natural Language Processing: The system integrates OpenAI's language models via Langchain, providing the capability to understand and generate human-like responses based on the uploaded SOPs.

    ### Workflow Overview
    
    1. Environment Setup:

        - Environment variables are loaded to retrieve the OpenAI API key securely.
        - Initial session states are established to manage user interactions and data flow.
                
    2. Document Upload and Processing:

        - Users can upload documents in TXT and PDF formats through a file uploader component.
        - Uploaded documents are validated for their type before processing.
        - Text extraction occurs from the documents, particularly from PDF files using PyPDF2. The text is then preprocessed to remove excessive whitespace and ensure a clean format.
        
    3. Database Interaction:

        - A singleton SQLite connection is established to handle document storage.
        - Each document's content is added to the database if it does not already exist, ensuring unique entries.
        - Document metadata, including source and section information, is stored alongside the content.
        
    4. Vector Store Initialization:

        - A FAISS vector store is initialized either by loading an existing index or creating a new one if no prior index is found.
        - OpenAI embeddings are used to convert document texts into vector representations for efficient similarity searching.
        
    5. Document Processing:

        - Upon successful upload, the system processes the documents by splitting the content into manageable chunks using RecursiveCharacterTextSplitter.
        - The processed documents are added to the FAISS vector store, enabling fast retrieval based on user queries.
        
    6. Query Handling:

        - Once documents are processed, users can submit queries through the UI.
        - A custom prompt template guides the language model to generate structured, context-aware responses.
        - The system employs a RetrievalQA chain that combines the language model with the vector store, ensuring answers are derived from the content of the SOPs.
        
    7. Response Generation:

        - User queries are processed by the QA chain, which retrieves relevant documents and generates answers while adhering to specified guidelines for citation and response structure.
        
    8. Query History and Session Management:

        - The application maintains a history of queries and responses for user reference, allowing users to review previous interactions.
        - Users can reset the knowledge base and query history via the UI.

    _____________________________________________________________________

    """)

st.markdown("""
### 🌟 **Welcome to SOPhia (SOP handling intelligent agent)🤵‍♀️!**

This application leverages **LangChain** and **OpenAI's** powerful language models to provide an interactive question-and-answer interface based on your uploaded documents.

#### **Key Features:**
- **📁 Upload Multiple Documents**: Support for PDF and TXT files.
- **🔍 Intelligent Search**: Quickly find relevant information within your documents.
- **📑 Detailed Sources**: Answers come with references to the specific document sections.
- **⚡ Fast and Efficient**: Optimized for quick processing and responses.

#### **Getting Started:**
1. **Upload Documents**:
   - Click on the sidebar's "Upload Documents" section.
   - Select and upload your PDF or TXT files.
2. **Process Documents**:
   - After uploading, click on "Start Upload and Processing".
   - Wait for the progress bar to complete the processing.
3. **Ask Questions**:
   - Once processing is done, the question input box will appear after app is rerun.
   - Enter your query and receive detailed answers.
4. **Review History**:
   - Access your query history in the "Query History" tab to revisit past interactions.

#### **Why Use This App?**
- **Efficiency**: Save time by extracting information without manually searching through documents.
- **Accuracy**: Get precise answers backed by the content of your documents.
- **Transparency**: Always know the source of the information provided.

#### **Note:**
- Ensure your `.env` file is properly configured with your OpenAI API key.
- Uploaded documents are securely stored and processed locally.

🔒 **Your data privacy is our priority!**
""")

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None

if 'documents_uploaded' not in st.session_state:
    st.session_state['documents_uploaded'] = False

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Define the path for the FAISS index
FAISS_INDEX_PATH = 'vector_store.faiss'

def preprocess_text(text):
    """
    Preprocesses the text by removing excessive whitespace and newlines.
    """
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with a space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file with source metadata.
    """
    try:
        text = ""
        pdf_reader = PyPDF2.PdfReader(file_path)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                st.warning(f"⚠️ No text found on page {page_num + 1} of {os.path.basename(file_path)}.")
        return preprocess_text(text)
    except Exception as e:
        st.error(f"❌ Error extracting text from {os.path.basename(file_path)}: {e}")
        return ""

def process_directory(directory):
    """
    Processes all PDF files in the directory: extracts text and updates the FAISS vector store.
    """
    documents_to_add = []
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.pdf')]

    if not pdf_files:
        st.error("❌ No PDF files found in the specified directory.")
        return

    st.info(f"📂 Found {len(pdf_files)} PDF files. Processing...")
    for file_path in pdf_files:
        text = extract_text_from_pdf(file_path)
        if text:
            # Add extracted text to documents for vector store processing
            documents_to_add.append({"text": text, "source": os.path.basename(file_path)})

    # Split documents into chunks for vectorization
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = [
        {"text": chunk, "source": doc["source"]}
        for doc in documents_to_add
        for chunk in text_splitter.split_text(doc["text"])
    ]

    # Convert chunks to LangChain Document objects
    docs = [Document(page_content=chunk["text"], metadata={"source": chunk["source"]}) for chunk in chunks]

    # Build or update FAISS vector store
    if st.session_state['vectorstore'] is None:
        st.session_state['vectorstore'] = FAISS.from_documents(docs, embeddings)
    else:
        st.session_state['vectorstore'].add_documents(docs)

    # Save the updated vector store
    st.session_state['vectorstore'].save_local(FAISS_INDEX_PATH)
    st.success("🎉 Knowledge base created and saved successfully!")
    st.session_state['documents_uploaded'] = True

# Automatically build the vector store if not already built
if st.session_state['vectorstore'] is None:
    process_directory(PREDEFINED_DIRECTORY)

# Query Tab
if st.session_state['documents_uploaded'] and st.session_state['vectorstore']:
    st.subheader("🗨️ Ask a Question:")
    query = st.text_input("Enter your query:", placeholder="What information are you looking for?")
    search_button = st.button("Search")

    if search_button and query.strip():
        # Create QA chain
        custom_prompt = """
        You are an expert assistant that provides detailed step-by-step instructions derived only from the provided documents (SOPs).
        Each response must include citations of the source document and section.
        You must directly perform all instructions with reference to the appropriate sections of the knowledge base
        You must only refer to sections of the knowledge base which is relevant to your task.
        You must always review your output to determine if the facts are consistent with the knowledge base
        Do not do math calculations and just cite the data as it is.
        Cite text in verbatim as far as possible
        In your output, retain the keywords and tone from the documents.
        If the output to the instructions cannot be derived from the knowledge base, strictly only reply “There is no relevant information, please only query about SOP related information”.
        Documents: {context}
        
        Question: {question}
        
        Provide your answer as a detailed guide with citations.
        """
        prompt_template = PromptTemplate(input_variables=["context", "question"], template=custom_prompt)
        llm = OpenAI(temperature=0, openai_api_key=api_key, max_tokens=1000)
        qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt_template)

        retriever = st.session_state['vectorstore'].as_retriever(search_type="similarity", search_kwargs={"k": 5})
        qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever)

        with st.spinner("🔍 Searching the knowledge base..."):
            try:
                answer = qa.run(query)
                st.subheader("💡 Answer:")
                st.write(answer)

                # Display relevant documents
                relevant_docs = retriever.get_relevant_documents(query)
                if relevant_docs:
                    st.subheader("📄 Source Documents:")
                    for doc in relevant_docs:
                        st.write(f"**Source:** {doc.metadata['source']}")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
    elif search_button:
        st.error("❌ Query cannot be empty.")
else:
    st.info("📝 Loading knowledge base. Please wait or ensure the directory contains valid PDF files.")
