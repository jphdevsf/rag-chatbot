import ollama
import re
import gradio as gr
import time
import numpy as np
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    Docx2txtLoader, 
    UnstructuredMarkdownLoader, 
    JSONLoader,
    TextLoader
)
from chromadb import PersistentClient
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

FILE_PATH = "resources/Accessibility_Web_WCAG2_2_easy_EN_2025_revilla_carreras.pdf"

# Extract filename from PDF_PATH and create database name
filename = os.path.basename(FILE_PATH)
name_without_ext = os.path.splitext(filename)[0]
CHROMA_DB_PATH = f"./chroma_db_{name_without_ext}"

COLLECTION_NAME = "my_collection"
MODEL_NAME = "qwen2.5-coder:1.5b"
EMBED_MODEL_NAME = "all-minilm"  # Much faster embedding model
BATCH_SIZE = 50  # Process 50 chunks at once
CHUNK_SIZE = 500  # Larger chunks = fewer total chunks
CHUNK_OVERLAP = 100

# Check if collection already exists to avoid reprocessing
def check_if_collection_exists(client, collection_name):
    try:
        collection = client.get_collection(name=collection_name)
        return collection.count() > 0
    except:
        return False

# 2. EMBEDDING FUNCTION SETUP (Universal: "Semantic Representation")
class FixedOllamaEmbeddingFunction:
    def __init__(self, model_name, url):
        self.embedding_function = OllamaEmbeddingFunction(
            model_name=model_name,
            url=url
        )
    
    def embed_query(self, text):
        """Embed a query with proper list conversion"""
        embedding = self.embedding_function.embed_query(text)
        
        # Convert numpy array to regular Python list
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        elif isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        elif isinstance(embedding, list) and len(embedding) > 0 and hasattr(embedding[0], 'tolist'):
            # Handle case where we have a list containing a numpy array
            embedding = embedding[0].tolist()
        
        return embedding
    
    def embed_documents(self, texts):
        """Embed multiple documents with proper list conversion"""
        embeddings = self.embedding_function.embed_documents(texts)
        
        # Convert each embedding to list
        converted_embeddings = []
        for embedding in embeddings:
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            elif isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            elif isinstance(embedding, list) and len(embedding) > 0 and hasattr(embedding[0], 'tolist'):
                # Handle case where we have a list containing a numpy array
                embedding = embedding[0].tolist()
            converted_embeddings.append(embedding)
        
        return converted_embeddings
    
    def __call__(self, input):
        """Make this callable like the original"""
        if isinstance(input, list):
            return self.embed_documents(input)
        else:
            return self.embed_query(input)

embedding_function = FixedOllamaEmbeddingFunction(
    model_name=EMBED_MODEL_NAME,
    url="http://localhost:11434"
)

def generate_embeddings_efficient(chunks):
    """Generate embeddings efficiently using sequential processing (Ollama is single-threaded)"""
    embeddings = []
    print(f"Generating embeddings for {len(chunks)} chunks using all-minilm model...")
    
    start_time = time.time()
    
    for i, chunk in enumerate(chunks):
        try:
            embedding = embedding_function.embed_query(chunk.page_content)
            
            # Convert numpy array to regular Python list
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            elif isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            elif isinstance(embedding, list) and len(embedding) > 0 and hasattr(embedding[0], 'tolist'):
                # Handle case where we have a list containing a numpy array
                embedding = embedding[0].tolist()
            
            embeddings.append(embedding)
            
            # Progress reporting every 50 chunks
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(chunks) - i - 1) / rate if rate > 0 else 0
                print(f"Processed {i+1}/{len(chunks)} chunks ({rate:.1f} chunks/sec, ETA: {eta:.1f}s)")
                
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            # Use a zero vector as fallback for failed embeddings
            embeddings.append([0.0] * 768)  # Standard embedding dimension
    
    total_time = time.time() - start_time
    print(f"Generated {len(embeddings)} embeddings in {total_time:.1f}s ({len(chunks)/total_time:.1f} chunks/sec)")
    
    return embeddings

# 3. VECTOR DATABASE CONFIGURATION (Universal: "Vector Storage")
# Use PersistentClient to store data on disk with file specific database name
client = PersistentClient(path=CHROMA_DB_PATH)

collection_exists = check_if_collection_exists(client, COLLECTION_NAME)

if not collection_exists:
    print("No existing collection found. Processing documents and creating embeddings...")
    
    # 1. DYNAMIC DOCUMENT LOADING based on file extension
    file_ext = os.path.splitext(FILE_PATH)[1].lower()
    
    if file_ext == '.pdf':
        loader = PyMuPDFLoader(FILE_PATH)
    elif file_ext == '.docx':
        loader = Docx2txtLoader(FILE_PATH)
    elif file_ext == '.md':
        loader = UnstructuredMarkdownLoader(FILE_PATH)
    # elif file_ext == '.json':
    #     try:
    #         # Tailored for your structure: array of entities (with observations) and relations
    #         loader = JSONLoader(
    #             file_path=FILE_PATH,
    #             jq_schema='''
    #                 .[] | 
    #                 if .type == "entity" then 
    #                     (.name + ": " + (.observations | join(" | "))) 
    #                 else 
    #                     (.from + " -> " + .to + " (" + .relationType + ")")
    #                 end
    #             ''',
    #             content_key='content',  # Arbitrary key for the extracted/formatted string
    #             metadata_func=lambda x: {
    #                 'source': FILE_PATH,
    #                 **{k: v for k, v in x.items() if k in ['name', 'type', 'entityType', 'from', 'to', 'relationType']}  # Preserve key fields
    #             }
    #         )
    #         # Quick validation
    #         test_docs = loader.load()
    #         if not test_docs:
    #             raise ValueError("No documents extracted")
    #         print(f"JSON loaded: {len(test_docs)} documents (entities/relations formatted).")
    #     except (ValueError, Exception) as e:
    #         print(f"JSONLoader failed ({e}). Falling back to raw text.")
    #         loader = TextLoader(FILE_PATH, encoding='utf-8')
    else:
        # Fallback for other text-based files
        loader = TextLoader(FILE_PATH, encoding='utf-8')
    
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    
    # Generate embeddings
    print("Starting embedding generation...")
    embeddings = generate_embeddings_efficient(chunks)
    print(f"Generated {len(embeddings)} embeddings successfully!")
    
    # Create new collection
    collection = client.create_collection(name=COLLECTION_NAME)
    for idx, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk.page_content], 
            metadatas=[{'id': idx}], 
            embeddings=[embeddings[idx]], 
            ids=[str(idx)]
        )
    print(f"Collection '{COLLECTION_NAME}' created with {len(chunks)} documents!")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists with data. Skipping document processing.")
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Using existing collection with {collection.count()} documents.")


# 4. RETRIEVAL PIPELINE (Universal: "Context Retrieval")
retriever = Chroma(collection_name=COLLECTION_NAME, client=client, embedding_function=embedding_function).as_retriever()

def retrieve_context(question):
    results = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in results])
    return context

# 5. RAG WORKFLOW INTEGRATION (Universal: "Response Synthesis")
def query_llm(question, context):
    formatted_prompt = f"""You are a strict question-answering assistant. You must ONLY use the information provided in the context below to answer the question. Do NOT use any external knowledge, general knowledge, or information not explicitly stated in the context.

IMPORTANT RULES:
1. Only answer based on the information in the provided context
2. If the context doesn't contain the answer, clearly state that the information is not available in the document
3. Do not add explanations, examples, or details that are not in the context
4. Do not use your general knowledge to supplement the answer

Question: {question}

Context: {context}

Instructions: Answer the question using ONLY the information from the context above. If the answer cannot be found in the context, state that the information is not available in the document."""
    
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': formatted_prompt}]
    )
    response_content = response['message']['content']
    final_answer = re.sub(r'Settings.*?RECHECK', '', response_content, flags=re.DOTALL).strip()
    return final_answer

# RAG Pipeline: Complete flow from user question to final answer
def ask_question(question):
    context = retrieve_context(question)
    answer = query_llm(question, context)
    return answer

# User Interface: Web interface for users to interact with the RAG system
interface = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(lines=10, label="Question"),
    outputs=gr.Textbox(lines=10, label="Response"),
    title="RAG Chatbot",
    description=f"Ask me a question about: {filename}"
)
interface.launch()

