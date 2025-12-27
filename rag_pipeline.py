import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
# from langchain_ollama import Chatollama

load_dotenv()
VECTORSTORE_PATH = "vectorstore/faiss_index"

# LOAD DOCUMENTS FROM STREAMLIT UPLOADS
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_documents(uploaded_files):
    documents = []

    for file in uploaded_files:
        original_name = file.name
        suffix = os.path.splitext(original_name)[1]

        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name

        # Choose loader
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".md":
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            continue

        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = original_name
            if "page" not in doc.metadata:
                doc.metadata["page"] = "N/A"
        documents.extend(docs)

        os.remove(tmp_path)
    return documents

# BUILD VECTOR STORE FROM DOCUMENTS
def build_vectorstore(uploaded_files):
    documents = load_documents(uploaded_files)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=80
    )
    chunks = splitter.split_documents(documents)

    # embeddings = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2"
    # )
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

# LOAD EXISTING VECTOR STORE

def load_vectorstore():
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2"
    # )
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
# def get_llm(llm_type="openrouter"): #tested with ollama and openai
#     if llm_type == "ollama":
#         return ChatOllama(
#             model="phi3:mini",
#             temperature=0
#         )
#     else:
#         return ChatOpenAI(
#             model="gpt-3.5-turbo",
#             temperature=0,
#             openai_api_key=os.getenv("OPENAI_API_KEY"),
#         )

def retrieve_context(query, vectorstore, k=3):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
        )
    return retriever.get_relevant_documents(query)

def generate_answer(query, retrieved_docs):
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    system_prompt = """
You are an internal AI assistant for Indecimal.
Your role is to answer user questions strictly using the provided document context.
The documents contain internal reference information such as:
- Company overview and customer journey
- Package pricing and specifications
- Quality assurance systems and policies
- Payment safety, timelines, and maintenance programs
RULES (MANDATORY):
1. Use ONLY the information present in the provided context.
2. Do NOT use external knowledge, assumptions, or general construction industry knowledge.
3. If the answer is not explicitly mentioned or cannot be inferred directly from the context, respond with:
   "I don't have enough information in the provided documents."
4. Do NOT exaggerate, guarantee, or invent details.
5. When multiple packages or options exist, clearly differentiate them based on the context.
6. Maintain a professional, factual, and neutral tone suitable for an internal knowledge assistant.
The final answer must be concise, accurate, and grounded in the retrieved content.
"""
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,  
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{query}")
    ]
    return llm(messages).content




