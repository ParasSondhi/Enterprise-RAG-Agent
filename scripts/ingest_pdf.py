import os
import shutil  # <-- 1. Add this built-in library
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Define Production Paths
PDF_PATH = os.path.join("data", "sample.pdf")
DB_DIR = os.path.join("database", "chroma_db")

def build_vector_store():
    # 2. THE KILL SWITCH: Wipe old database if it exists
    if os.path.exists(DB_DIR):
        print(f"⚠️ Old database detected at {DB_DIR}. Deleting to prevent duplicates...")
        shutil.rmtree(DB_DIR)
    
    print(f"\nLoading document from: {PDF_PATH}")
    # 3. Extract Data
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # 4. Chunk the Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Document split into {len(chunks)} computational chunks.")

    # 5. Initialize Embeddings
    print("Spinning up embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 6. Build and Persist the NEW Chroma Database
    print("Writing fresh data to local ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print("✅ Vector Database successfully rebuilt!")

if __name__ == "__main__":
    build_vector_store()