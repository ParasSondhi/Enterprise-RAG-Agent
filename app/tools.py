import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import create_retriever_tool
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool

# 1. Path Definitions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "database", "chroma_db")
SQL_URI = f"sqlite:///{os.path.join(BASE_DIR, 'database', 'structured_data.db')}"

print("Booting up the Data Layer Interfaces...")

# ---------------------------------------------------------
# TOOL 1: THE UNSTRUCTURED RAG ENGINE (ChromaDB)
# ---------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vector_store.as_retriever(
    search_type="mmr", 
    search_kwargs={
        "k": 3,          # <--- DROPPED FROM 5 to 3. Saves 40% of your tokens instantly.
        "fetch_k": 20,   
        "lambda_mult": 0.5 
    }
)

# Wrap the retriever into a tool the Agent can use
pdf_search_tool = create_retriever_tool(
    retriever,
    name="pdf_search",
    description="Use this tool to search the uploaded PDFs and documents for text-based information, concepts, and general knowledge. Do NOT use this tool for math, SQL databases, or calculating revenue."
)

# ---------------------------------------------------------
# TOOL 2: THE STRUCTURED SQL ENGINE (SQLite)
# ---------------------------------------------------------
# Connect LangChain to the SQLite file
# ---------------------------------------------------------
# TOOL 2: THE STRUCTURED SQL ENGINE (SQLite)
# ---------------------------------------------------------
# Connect LangChain to the SQLite file
sql_db = SQLDatabase.from_uri(SQL_URI)

# We dynamically pull the exact CREATE TABLE schema so the AI knows the exact column names
schema_info = sql_db.get_table_info()

sql_description = (
    "Execute SQLite queries. YOU MUST ONLY INPUT VALID SQL CODE. NEVER INPUT ENGLISH. "
    "CRITICAL RULES: "
    "1. The ONLY table available is named 'uploaded_data'. Do not guess table names. "
    "2. Example Action Input: SELECT AVG(speed), AVG(height) FROM uploaded_data; "
    "3. Use LIKE '%keyword%' for text searches. "
    f"Here is the database schema you MUST use: {schema_info}"
)

sql_tool = QuerySQLDatabaseTool(db=sql_db, verbose=True) 
sql_tool.name = "sql_db_query" 
sql_tool.description = sql_description

agent_tools = [pdf_search_tool, sql_tool] # Replace 'pdf_search' with your actual PDF tool variable name if it's different

print("SUCCESS: Tools initialized and ready for Agent attachment.")