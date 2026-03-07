import streamlit as st
st.set_page_config(
    page_title="Enterprise Hybrid AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
from app.agent import build_agent # Importing your actual backend brain
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.tools import vector_store # We import your live database

# 1. Page Configuration


# 2. Booting the Engine (Cached so it only runs ONCE)
#@st.cache_resource(show_spinner="Booting up the Enterprise Data Layer...")
def get_agent(model_choice):
    return build_agent(model_name=model_choice)



# 3. The Sidebar (Data Ingestion Zone)
with st.sidebar:
    st.header("⚙️ Local AI Engine")
    
    model_options = {
        "Llama 3.2 1B (Ultra Light - 2GB RAM)": "llama3.2:1b",
        "Llama 3.2 3B (Newest Small - 8GB RAM)": "llama3.2",
        "Llama 3.1 8B (Standard - 16GB RAM)": "llama3.1:8b",
        "DeepSeek-R1 32B (Distilled Reasoning - 32GB RAM)": "deepseek-r1:32b",
        "Llama 3.3 70B (Newest Expert - 64GB RAM+)": "llama3.3:70b"
    }
    
    selected_friendly_name = st.selectbox(
        "Select your Model:",
        options=list(model_options.keys()),
        help="Choose a model that fits your system's VRAM/RAM."
    )
    
    target_model = model_options[selected_friendly_name]
    st.caption("Ensure this model is downloaded via Ollama before chatting:")
    st.code(f"ollama pull {target_model}", language="bash")
    st.divider()
    st.header("🗄️ Database Injection")
    st.markdown("Upload your unstructured and structured data here.")
    
    # Create a memory bank for processed files
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    
    uploaded_pdf = st.file_uploader("Upload Engineering PDF", type=["pdf"])
    
    if uploaded_pdf:
        # Check if we ALREADY processed this exact file
        if uploaded_pdf.name not in st.session_state.processed_files:
            with st.spinner("Injecting PDF into AI Brain..."):
                # 1. Save the uploaded file temporarily
                import tempfile
                from langchain_community.document_loaders import PyPDFLoader
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                from app.tools import vector_store
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_pdf.getvalue())
                    temp_path = temp_file.name
                
                # 2. Parse and split the PDF into readable chunks
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = splitter.split_documents(docs)
                
                # 3. THE KILL SWITCH: Generate deterministic IDs to force Upserts
                chunk_ids = [f"{uploaded_pdf.name}_chunk_{i}" for i in range(len(split_docs))]
                
                # Inject into the live ChromaDB using the custom IDs
                vector_store.add_documents(split_docs, ids=chunk_ids)
                
                # 4. MEMORY LOCK: Add this file to our "already processed" list
                st.session_state.processed_files.add(uploaded_pdf.name)
                
                st.success(f"Successfully Indexed: {uploaded_pdf.name}")
                os.remove(temp_path) # Clean up the temp file
        else:
            # If it's already in memory, just show a green badge
            st.success(f"Loaded and active: {uploaded_pdf.name}")
        
    st.divider()
    
    
    uploaded_csv = st.file_uploader("Upload CSV Database", type=["csv"])
    if uploaded_csv:
        if uploaded_csv.name not in st.session_state.processed_files:
            with st.spinner("Injecting CSV into SQL Engine..."):
                import pandas as pd
                import sqlite3
                
                # 1. Read the uploaded CSV into a Pandas DataFrame
                df = pd.read_csv(uploaded_csv)
                
                # 2. Connect to the local SQLite database
                BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
                
                # --- THE BULLETPROOF FIX ---
                db_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "database"))
                os.makedirs(db_dir, exist_ok=True) # Creates the folder if it's missing!
                db_path = os.path.join(db_dir, "structured_data.db")
                # ---------------------------
                
                conn = sqlite3.connect(db_path)
                
                # 3. Overwrite the database with a new table called 'uploaded_data'
                df.to_sql('uploaded_data', conn, if_exists='replace', index=False)
                conn.close()
                
                # 4. MEMORY LOCK
                st.session_state.processed_files.add(uploaded_csv.name)
                st.success(f"SQL Database Overwritten with: {uploaded_csv.name}")
        else:
            st.success(f"Loaded and active: {uploaded_csv.name}")
agent_executor = get_agent(target_model)            
# 4. The Main Chat Interface
st.title("🧠 Enterprise ReAct Agent")
st.markdown("Ask questions across your PDFs and SQL databases simultaneously.")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. The Execution Loop
if prompt := st.chat_input("Ask the Database or the Documents..."):
    
    # Render user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Trigger the AI Response
    with st.chat_message("assistant"):
        with st.spinner("Agent is analyzing data..."):
            try:
                # We feed the prompt directly into your Groq-powered Agent!
                raw_response = agent_executor.invoke({"input": prompt})
                final_answer = raw_response["output"]
                
                # Render the final answer
                st.markdown(final_answer)
                
            except Exception as e:
                final_answer = f"⚠️ System Error: {e}"
                st.error(final_answer)
                
    # Save AI response to history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
