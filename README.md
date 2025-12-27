# Mini RAG Chatbot – Assignment Submission

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that answers user queries strictly based on uploaded internal documents. The main objective of the system is to avoid hallucinations by grounding every response in retrieved document content and explicitly handling cases where the required information is not available.

The application is built using **Streamlit** and provides a simple interface where users can upload documents and interact with the chatbot.

---

## Features

- Upload and process **PDF** and **Markdown (.md)** documents  
- Semantic retrieval using **vector embeddings and FAISS**  
- Answers generated **only from retrieved document context**  
- Clear fallback response when information is missing  
- Transparent display of:
  - Retrieved document chunks  
  - Source file names and page numbers  
- Clean and readable UI focused on clarity and usability  

---
---

## Local LLM Testing (Ollama)

During development, the chatbot was also tested using a local open-source language model( phi3-mini) running via **Ollama**. This was done to compare response quality, grounding to retrieved context, and latency against a cloud-based LLM.

Local inference was used only for development and evaluation, while the deployed application uses a cloud LLM since local models are not accessible in hosted environments.

Local LLMs are better for privacy, cost, latency, and strict grounding, while cloud-based LLMs are better for answer quality, reasoning capability, and scalable deployment.


## Project Structure
````
mini-rag/
│
├── app.py # Streamlit UI
├── rag_pipeline.py # Document processing and RAG logic
│-data|--doc1.md
      |--doc2.md
      |--doc3.md
├── requirements.txt
└── README.md
|__ .gitignore
````


The project is structured to keep UI logic separate from document processing and retrieval logic, making the code easier to understand and maintain.

---

## How the System Works

### 1. Document Upload
Users upload PDF or Markdown files through the Streamlit interface.

---

### 2. Indexing (Explicit Step)

- Uploaded documents are temporarily stored during processing  
- Text is split into chunks with overlap to preserve context  
- Each chunk is converted into embeddings using a sentence-transformer model 
-I used API-based embeddings to avoid PyTorch dependency issues on hosted environments. Sentence-transformer embeddings were used locally during development.” 
- Embeddings are stored in a **local FAISS vector index**  

Indexing is triggered explicitly to avoid repeated processing during normal interactions.

---

### 3. Query Processing

- User queries are embedded using the same embedding model  
- Top-K relevant chunks are retrieved using semantic similarity  
- Retrieved chunks are displayed to the user for transparency  

---

### 4. Answer Generation

- A **strict system prompt** enforces grounding rules  
- Answers are generated **only from retrieved document content**  
- If the answer cannot be found in the documents, the system returns a clear fallback message  

---

## Grounding and Hallucination Control

The system prompt enforces the following rules:

- Use only the provided document context  
- Do not use external knowledge or assumptions  
- Do not invent or exaggerate details  
- Respond with  
  **"I don't have enough information in the provided documents."**  
  when the answer cannot be derived from the retrieved content  

This ensures the chatbot behaves as a reliable internal knowledge assistant rather than a general conversational model.

---

## Evaluation Methodology

The system was evaluated using a curated set of test questions derived directly from the provided documents. The evaluation focused on:

- Relevance of retrieved chunks  
- Grounding of generated answers  
- Presence of hallucinations or unsupported claims    
Manual evaluation showed consistent grounding behavior, with no hallucinated or unsupported responses observed.

---

## How to Run the Project Locally

### 1. Create a virtual environment
```bash
uv venv .venv
```
### 2. Activate the environment
```
.venv\Scripts\activate
```
### 3. Install dependencies
```
uv pip install -r requirements.txt
```
### 4. Set environment variables
```
OPENAI_API_KEY=your_api_ke
```
### 5. Run the application
```
streamlit run app.py
```








