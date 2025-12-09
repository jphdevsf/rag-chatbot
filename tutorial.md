# DeepSeek R1 RAG Chatbot With Chroma, Ollama, and Gradio
Tutorial scraped from [https://www.datacamp.com/tutorial/deepseek-r1-rag](https://www.datacamp.com/tutorial/deepseek-r1-rag).

Learn how to build a local RAG chatbot using DeepSeek-R1 with Ollama, LangChain, and Chroma.

## Why Use DeepSeek-R1 With RAG?

- High-performance retrieval: DeepSeek-R1 handles large document collections with low latency.
- Fine-grained relevance ranking: It ensures accurate retrieval of passages by computing semantic similarity.
- Cost and privacy benefits: You can run DeepSeek-R1 locally to avoid API fees and keep sensitive data secure.
- Easy integration: It easily integrates with vector databases like Chroma.
- Offline capabilities: With DeepSeek-R1 you can build retrieval systems that work even without internet access once the model is downloaded.

## Overview: Building a RAG Chatbot With DeepSeek-R1

Our demo project focuses on building a RAG chatbot using DeepSeek-R1 and Gradio.

The process begins with loading and splitting a PDF into text chunks, followed by generating embeddings for those chunks. These embeddings are stored in a Chroma database for efficient retrieval. When a user submits a query, the system retrieves the most relevant text chunks and uses DeepSeek-R1 to generate an answer based on the retrieved context.

## Step 1: Prerequisites

Before we start, let’s ensure that we have the following tools and libraries installed:

- Python 3.8+
- Langchain
- Chromadb
- Gradio

Run the following commands to install the necessary dependencies:

```bash
!pip install langchain chromadb gradio ollama pymupdf
!pip install -U langchain-community
```

Then import the necessary libraries:

```python
import ollama
import re
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from chromadb.config import Settings
from chromadb import Client
from langchain.vectorstores import Chroma
```

## Step 2: Load the PDF Using PyMuPDFLoader

We will use LangChain’s PyMuPDFLoader to extract the text from the PDF version of the book Foundations of LLMs by Tong Xiao and Jingbo Zhu.

```python
# Load the document using PyMuPDFLoader
loader = PyMuPDFLoader("/path/to/Foundations_of_llms.pdf")
documents = loader.load()
```

## Step 3: Split the Document Into Smaller Chunks

Split the extracted text into smaller, overlapping chunks for better context retrieval.

```python
# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
```

## Step 4: Generate Embeddings Using DeepSeek-R1

Use Ollama Embeddings based on DeepSeek-R1 to generate document embeddings.

```python
# Initialize Ollama embeddings using DeepSeek-R1
embedding_function = OllamaEmbeddings(model="deepseek-r1")
# Parallelize embedding generation
def generate_embedding(chunk):
    return embedding_function.embed_query(chunk.page_content)
with ThreadPoolExecutor() as executor:
    embeddings = list(executor.map(generate_embedding, chunks))
```

## Step 5: Store Embeddings in Chroma Vector Store

Store the embeddings and corresponding text chunks in Chroma.

```python
# Initialize Chroma client and create/reset the collection
client = Client(Settings())
client.delete_collection(name="foundations_of_llms")
collection = client.create_collection(name="foundations_of_llms")
# Add documents and embeddings to Chroma
for idx, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk.page_content], 
        metadatas=[{'id': idx}], 
        embeddings=[embeddings[idx]], 
        ids=[str(idx)]
    )
```

## Step 6: Initialize the Retriever

Initialize the Chroma retriever.

```python
# Initialize retriever using Ollama embeddings for queries
retriever = Chroma(collection_name="foundations_of_llms", client=client, embedding_function=embedding_function).as_retriever()
```

## Step 7: Define the RAG pipeline

Retrieve relevant chunks and format them for DeepSeek-R1.

```python
def retrieve_context(question):
    results = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in results])
    return context
```

## Step 8: Query DeepSeek-R1 for contextual answers

Send the question and context to DeepSeek-R1.

```python
def query_deepseek(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = embedding_function.chat(
        model="deepseek-r1",
        messages=[{'role': 'user', 'content': formatted_prompt}]
    )
    response_content = response['message']['content']
    final_answer = re.sub(r'Settings.*?RECHECK', '', response_content, flags=re.DOTALL).strip()
    return final_answer
```

## Step 9: Build the Gradio Interface

Create an interactive interface for users.

```python
def ask_question(question):
    context = retrieve_context(question)
    answer = query_deepseek(question, context)
    return answer

interface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs="text",
    title="RAG Chatbot: Foundations of LLMs",
    description="Ask any question about the Foundations of LLMs book. Powered by DeepSeek-R1."
)
interface.launch()
```

## Optimizations

- Adjust `chunk_size` and `chunk_overlap` for better performance.
- Use smaller model versions (e.g., `deepseek-r1:7b`) if needed.
- Scale using Faiss for larger documents.
- Batch processing for faster embedding generation.

## Conclusion

In this tutorial, we built a RAG-based local chatbot using DeepSeek-R1 and Chroma for retrieval, ensuring accurate, contextually rich answers to questions based on a large knowledge base.