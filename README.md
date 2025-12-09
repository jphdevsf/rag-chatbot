# RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot built with Python, using Ollama for local LLM inference, ChromaDB for vector storage, and Gradio for the web interface. It processes documents (e.g., PDFs) to answer questions based on their content.

## Features

- **Local RAG Pipeline**: Runs entirely offline using Ollama for LLM inference and embeddings, ensuring privacy and no API costs.
- **Multi-Format Document Support**: Handles PDFs (via PyMuPDF), DOCX, Markdown, JSON (structured entities/relations), and plain text files.
- **Efficient Processing**: Uses recursive text splitting with configurable chunk sizes, batch embedding generation with progress tracking, and persistent ChromaDB storage to avoid reprocessing.
- **Strict Context-Based Answers**: LLM responses are constrained to document content only, preventing hallucinations or external knowledge leakage.
- **User-Friendly Interface**: Web-based Gradio UI for easy interaction, with real-time question-answering.
- **Customizable**: Easy to swap models (e.g., Qwen2.5-Coder for LLM, All-MiniLM for embeddings) and adjust parameters like chunk size or batch processing.


## Prerequisites

Before setting up the project, ensure you have the following:

1. **Python 3.8 or higher**: Download from [python.org](https://www.python.org/downloads/). Verify with `python --version`.

2. **Ollama**: A tool for running large language models locally.
   - Download and install from [ollama.com](https://ollama.com/download).
   - Start Ollama: Run `ollama serve` in your terminal (or it may start automatically).
   - Validate Ollama is running: Open a new terminal and run `curl http://localhost:11434`. You should see a response like `Ollama is running`.
   - Pull required models:
     - For the main LLM: `ollama pull qwen2.5-coder:1.5b`
     - For embeddings: `ollama pull all-minilm` (Note: Ensure Ollama supports this embedding model; if not available, you may need to adjust the code to use a compatible embedding model like `nomic-embed-text`).

3. **Documents**: Place your input file (e.g., `Accessibility_Web_WCAG2_2_easy_EN_2025_revilla_carreras.pdf`) in the `resources/` folder. This folder is gitignored, so add your own files.

## Installation

1. **Clone the Repository**:
   ```
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
   ```

2. **Create a Virtual Environment** (Recommended to isolate dependencies):
   - On macOS/Linux:
     ```
     python -m venv venv
     source venv/bin/activate
     ```
   - On Windows (if using Git Bash or similar):
     ```
     python -m venv venv
     source venv/Scripts/activate
     ```
   - Verify activation: Your terminal prompt should show `(venv)`.

3. **Install Python Dependencies**:
   Run the following command to install required packages:
   ```
   pip install ollama gradio numpy langchain langchain-community langchain-chroma chromadb pymupdf python-docx unstructured markdown
   ```
   - This covers: Ollama client, Gradio UI, NumPy, LangChain (text splitters, loaders for PDF/DOCX/MD/JSON/TXT), ChromaDB, and PyMuPDF for PDF handling.
   - If you encounter issues with `unstructured`, it may require additional system dependencies (e.g., `libmagic` on macOS via `brew install libmagic`).

4. **Optional: Create a requirements.txt**:
   For reproducibility, generate one after installation:
   ```
   pip freeze > requirements.txt
   ```
   Then install via `pip install -r requirements.txt` in future setups.

## Running the Application

1. Ensure Ollama is running (see Prerequisites).

2. Run the chatbot script:
   ```
   python chatbot.py
   ```

3. The Gradio interface will launch in your browser (usually at `http://127.0.0.1:7860`). Ask questions about the document in the textbox.

## Usage

- The script processes the PDF in `resources/` on first run, creating a ChromaDB vector store in `./chroma_db_*` (gitignored).
- Subsequent runs reuse the database unless deleted.
- Questions are answered strictly based on the document context using the RAG pipeline.
- To process a different file, update `FILE_PATH` in `chatbot.py` and rerun.

## Troubleshooting

- **Ollama not responding**: Ensure `ollama serve` is running and models are pulled.
- **Embedding errors**: Verify the embedding model is compatible with Ollama; fallback to `nomic-embed-text` if needed.
- **PDF loading issues**: Install `pymupdf` via `pip install pymupdf`.
- **Port conflicts**: Gradio may use a different port; check the terminal output.

## Notes

- The vector database is stored locally and persists across runs.
- For production, consider Dockerizing or adding error handling.
- This setup runs entirely locally for privacy.

If you encounter issues, check the console output or provide more details.