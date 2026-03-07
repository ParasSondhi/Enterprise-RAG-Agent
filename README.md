# 🧠 Enterprise RAG & SQL Agent (Zero-Leakage Architecture)

### Live Demonstration 
**[🎥 Click here to watch the full Agent in action](https://drive.google.com/file/d/1Ib78egf57JLCYflITnxK3eL06--PIfZ4/view?usp=drive_link)**

A privacy-first, fully local AI agent that combines **Retrieval-Augmented Generation (RAG)** with deterministic SQL querying. Built using a ReAct architecture, this agent dynamically routes user queries across unstructured PDFs and structured databases without exposing sensitive data to external cloud APIs.

### ⚙️ Core Engineering & Data Pipeline
* **Hybrid RAG Routing:** Seamlessly switches between semantic search over document vectors (ChromaDB) and executing strict SQL queries (SQLite).
* **Automated Data Cleaning:** Raw CSV files are dynamically pre-processed using **Pandas** (handling missing values, datatype casting, and formatting) before being loaded into the SQL engine.
* **Smart Document Ingestion:** Automated chunking and ID-tagging for PDFs via LangChain to prevent duplicate vectorization across sessions.
* **Defensive Prompt Engineering:** Custom ReAct prompts with aggressive format-error handling and infinite-loop kill switches to prevent agent crashes during complex reasoning.

### 🛠️ Skills & Tech Stack
* **AI/ML Concepts:** RAG, ReAct Agent Architecture, Prompt Engineering, Semantic Search, Local LLM Quantization.
* **Frameworks & Libraries:** LangChain, Pandas, Streamlit, PyPDF, HuggingFace Embeddings.
* **Databases:** ChromaDB (Vector), SQLite (Relational).
* **LLM Engine:** Ollama (running local GGUF quantized models).

### 🖥️ Hardware & Quantization Matrix
ReAct agents require strict adherence to formatting and advanced multi-step reasoning. Smaller models will frequently struggle to output the correct tool-routing syntax. To run these models on consumer hardware, this project utilizes **Q4_K_M 4-bit quantization** via Ollama. 

| Base Model | Minimum VRAM | Quantization | Performance Profile |
| :--- | :--- | :--- | :--- |
| **Llama 3.2 1B** | 2GB | Default 4-bit (GGUF) | Basic pipeline testing only. Expect frequent formatting failures. |
| **Llama 3.2 3B** | 8GB | Default 4-bit (GGUF) | Light queries and entry-level reasoning. |
| **Llama 3.1 8B** | 16GB | Default 4-bit (GGUF) | Standard performance. Recommended baseline for ReAct logic. |
| **DeepSeek-R1 32B** | 32GB | Default 4-bit (GGUF) | Advanced reasoning and high data-extraction accuracy. |
| **Llama 3.3 70B** | 64GB+ | Default 4-bit (GGUF) | Enterprise-grade accuracy. Flawless tool selection and formatting. |

### 🚀 Local Deployment

**1. Install Dependencies:**
```Bash
pip install -r requirements.txt
```
**2. Pull the Quantized Engine (e.g., Llama 3.3 70B):**

```Bash
ollama pull llama3.3:70b
```
 **Troubleshooting for Windows Users:** If you get an error saying 'ollama is not recognized as an internal or external command', ensure you have opened a fresh terminal after installing Ollama, or provide the direct path to your `ollama.exe` file.
 
***3. Initialize the Agent Interface:***

```Bash
streamlit run app/ui.py
