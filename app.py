import os
import re
import time
from typing import List, Dict, Any, Optional
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "google/flan-t5-base"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 200
MAX_FILE_SIZE_MB = 10
PERSIST_DIR = "chroma_db"  # For persistent storage

# Initialize models
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource(show_spinner="Loading language model...")
def load_llm():
    try:
        pipe = pipeline(
            "text2text-generation",
            model=LLM_NAME,
            max_length=800,
            temperature=0.3,
            device_map="auto"
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        st.stop()

# Document processing
def extract_text_from_file(uploaded_file) -> Optional[str]:
    try:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.warning(f"File {uploaded_file.name} exceeds {MAX_FILE_SIZE_MB}MB limit")
            return None

        text = ""
        if uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif uploaded_file.type == "text/plain":
            text = str(uploaded_file.read(), "utf-8", errors="replace")
        elif uploaded_file.type.endswith("wordprocessingml.document"):
            doc = Document(uploaded_file)
            text = "\n".join(para.text for para in doc.paragraphs)
        return text if text.strip() else None
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return None

def split_documents(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "• ", "  "],
        length_function=len
    )
    return text_splitter.split_text(text)

# Vector store with persistence
def create_vector_store(_embedding_model, chunks: List[str]):
    with st.spinner("Creating search index..."):
        # Clear previous persist directory if exists
        if os.path.exists(PERSIST_DIR):
            import shutil
            shutil.rmtree(PERSIST_DIR)
            
        return Chroma.from_texts(
            texts=chunks,
            embedding=_embedding_model,
            persist_directory=PERSIST_DIR,
            metadatas=[{"source": f"chunk-{i}"} for i in range(len(chunks))]
        )

# Tools
def calculate(expression: str) -> str:
    try:
        expression = re.sub(r'[^0-9+\-*/.() ]', '', expression)
        if len(expression) > 50:
            return "Expression too complex"
        result = eval(expression, {'__builtins__': None}, {})
        return f"Result: {expression} = {result}"
    except:
        return "Could not compute the expression"

def define_word(word: str) -> str:
    definitions = {
        "algorithm": "A set of rules to solve problems",
        "neural network": "Computing system inspired by biological neurons",
        "api": "Application Programming Interface",
        "rmse": "Root Mean Square Error - prediction accuracy measure",
        "eda": "Exploratory Data Analysis"
    }
    return definitions.get(word.lower(), "Not in dictionary. Ask about documents.")

# Query processing
def route_query(query: str, vector_store, llm) -> Dict[str, Any]:
    start_time = time.time()
    log = {"query": query, "steps": [], "time_elapsed": 0}
    
    # Calculator
    if re.search(r'calculate|compute|\d\s*[+\-*/]\s*\d', query, re.I):
        match = re.search(r'([-+]?\d*\.?\d+\s*[-+*/]\s*[-+]?\d*\.?\d+)', query)
        if match:
            return {
                "result": calculate(match.group(1)),
                "tool": "calculator",
                "time_elapsed": round(time.time() - start_time, 2)
            }
    
    # Dictionary
    if re.search(r'define|what is|meaning of', query, re.I):
        if word := re.findall(r'(?:define|what is|meaning of)\s+([^\s?.]+)', query, re.I):
            return {
                "result": define_word(word[0]),
                "tool": "dictionary",
                "time_elapsed": round(time.time() - start_time, 2)
            }
    
    # RAG Pipeline
    try:
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10}
        )
        
        prompt = PromptTemplate(
            template="""Answer based ONLY on:
            Context: {context}
            Question: {question}
            If unsure, say "Not mentioned in documents".""",
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        
        result = chain({"query": query})
        return {
            "result": result["result"],
            "tool": "RAG",
            "time_elapsed": round(time.time() - start_time, 2),
            "retrieved_chunks": [c.page_content for c in retriever.get_relevant_documents(query)]
        }
    except Exception as e:
        return {
            "result": f"Error: {str(e)}",
            "tool": "error",
            "time_elapsed": round(time.time() - start_time, 2)
        }

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Document Q&A",
        page_icon="📄",
        layout="centered"
    )
    
    st.title("📄 Document Q&A Assistant")
    st.caption("Upload documents and get AI-powered answers")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )
    
    # Initialize
    if "vector_store" not in st.session_state:
        st.session_state.vs = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    
    # Process documents
    if uploaded_files and not st.session_state.processed:
        with st.status("Processing...", expanded=True) as status:
            texts = [extract_text_from_file(f) for f in uploaded_files]
            if any(texts):
                chunks = split_documents("\n\n".join(t for t in texts if t))
                st.session_state.vs = create_vector_store(load_embedding_model(), chunks)
                st.session_state.processed = True
                status.update(label=f"Processed {len(chunks)} chunks", state="complete")
            else:
                st.error("No valid text extracted")
    
    # Query interface
    if st.session_state.processed:
        if query := st.chat_input("Ask about your documents"):
            with st.spinner("Thinking..."):
                result = route_query(query, st.session_state.vs, load_llm())
                
                with st.chat_message("assistant"):
                    st.write(result["result"])
                    if st.toggle("Show details"):
                        st.caption(f"Method: {result['tool']} | Time: {result['time_elapsed']}s")
                        if "retrieved_chunks" in result:
                            st.subheader("Relevant passages")
                            for i, c in enumerate(result["retrieved_chunks"], 1):
                                st.text_area(f"Passage {i}", c, height=100)
    else:
        st.info("Upload documents to begin")

if __name__ == "__main__":
    main()