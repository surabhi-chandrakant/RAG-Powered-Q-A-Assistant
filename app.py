import os
import re
import time
from typing import List, Dict, Any, Optional
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"  # Upgraded to base model for better quality
CHUNK_SIZE = 800  # Reduced for better performance
CHUNK_OVERLAP = 150
MAX_FILE_SIZE_MB = 10  # Limit file size for cloud deployment

# Initialize models with better caching
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
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            LLM_MODEL,
            low_cpu_mem_usage=True,
            torch_dtype="auto"
        )
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=800,  # Reduced for cloud deployment
            temperature=0.3,
            repetition_penalty=1.2
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        st.stop()

# Enhanced document processing with error handling
def extract_text_from_file(uploaded_file) -> Optional[str]:
    try:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.warning(f"File {uploaded_file.name} exceeds {MAX_FILE_SIZE_MB}MB limit")
            return None

        text = ""
        if uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() or ""  # Handle None returns
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
        length_function=len,
        add_start_index=True
    )
    return text_splitter.split_text(text)

# Vector store with progress indication
def create_vector_store(_embedding_model, chunks: List[str]):
    with st.spinner("Creating search index..."):
        return FAISS.from_texts(chunks, _embedding_model)

# Enhanced tools
def calculate(expression: str) -> str:
    try:
        # Safer evaluation with only basic math operations
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Invalid characters in expression"
        
        # Limit complexity
        if len(expression) > 50:
            return "Expression too complex"
            
        result = eval(expression, {'__builtins__': None}, {})
        return f"Result: {expression} = {result}"
    except Exception:
        return "Could not compute the expression"

def define_word(word: str) -> str:
    definitions = {
        "algorithm": "A set of rules or steps to solve a problem",
        "neural network": "A computing system inspired by biological neural networks",
        "API": "Application Programming Interface - a way for programs to communicate",
    }
    return definitions.get(word.lower(), 
                         f"No definition found for '{word}'. Try asking about the document content.")

# Improved agent logic
def route_query(query: str, vector_store, llm) -> Dict[str, Any]:
    start_time = time.time()
    log = {
        "query": query, 
        "steps": [],
        "time_elapsed": 0
    }
    
    # Normalize query for matching
    clean_query = query.lower().strip()
    
    # Calculator routing
    if any(keyword in clean_query for keyword in ["calculate", "compute", "+", "-", "*", "/"]):
        log["steps"].append("Detected calculation request")
        match = re.search(r'([-+]?\d*\.?\d+\s*[-+*/]\s*[-+]?\d*\.?\d+)', query)
        if match:
            expression = match.group(1)
            result = calculate(expression)
            log.update({
                "result": result,
                "tool": "calculator",
                "time_elapsed": round(time.time() - start_time, 2)
            })
            return log
    
    # Dictionary routing
    if any(keyword in clean_query for keyword in ["define", "what is", "meaning of"]):
        log["steps"].append("Detected definition request")
        words = re.findall(r'(?:define|what is|meaning of)\s+([^\s?.]+)', query, re.I)
        if words:
            word = words[0].lower()
            result = define_word(word)
            log.update({
                "result": result,
                "tool": "dictionary",
                "time_elapsed": round(time.time() - start_time, 2)
            })
            return log
    
    # Default to RAG
    log["steps"].append("Using RAG pipeline")
    
    try:
        retriever = vector_store.as_retriever(
            search_type="mmr",  # Max marginal relevance for better diversity
            search_kwargs={"k": 3}
        )
        
        relevant_chunks = retriever.get_relevant_documents(query)
        log["retrieved_chunks"] = [chunk.page_content for chunk in relevant_chunks]
        
        prompt_template = """Answer based only on this context:
        {context}
        
        Question: {question}
        
        If the answer isn't in the context, say "I don't know"."""
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
            }
        )
        
        result = qa_chain({"query": query})
        log.update({
            "result": result["result"],
            "tool": "RAG",
            "time_elapsed": round(time.time() - start_time, 2)
        })
        
    except Exception as e:
        log.update({
            "result": f"Error processing your query: {str(e)}",
            "tool": "error",
            "time_elapsed": round(time.time() - start_time, 2)
        })
    
    return log

# Streamlit UI with better layout
def main():
    st.set_page_config(
        page_title="Document Q&A Assistant",
        page_icon="📄",
        layout="centered"
    )
    
    st.title("📄 Document Q&A Assistant")
    st.caption("Upload documents and get answers powered by AI")
    
    with st.sidebar:
        st.header("Settings")
        max_files = st.slider("Max files to process", 1, 10, 3)
        show_details = st.checkbox("Show processing details", True)
    
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Maximum 10MB per file"
    )
    
    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    
    # Process documents
    if uploaded_files and not st.session_state.processed:
        files_to_process = uploaded_files[:max_files]
        with st.status("Processing documents...", expanded=True) as status:
            all_text = []
            for uploaded_file in files_to_process:
                st.write(f"Reading {uploaded_file.name}...")
                text = extract_text_from_file(uploaded_file)
                if text:
                    all_text.append(text)
            
            if all_text:
                st.write("Splitting documents...")
                chunks = split_documents("\n\n".join(all_text))
                
                st.write("Creating search index...")
                embedding_model = load_embedding_model()
                st.session_state.vector_store = create_vector_store(embedding_model, chunks)
                st.session_state.processed = True
                status.update(label="Processing complete!", state="complete", expanded=False)
                st.success(f"Processed {len(files_to_process)} files with {len(chunks)} chunks")
            else:
                st.error("No valid text extracted from documents")
                st.session_state.processed = False
    
    # Query interface
    if st.session_state.processed:
        query = st.chat_input("Ask a question about the documents")
        
        if query:
            with st.spinner("Thinking..."):
                llm = load_llm()
                result = route_query(query, st.session_state.vector_store, llm)
                
                # Display results
                with st.chat_message("assistant"):
                    st.markdown(f"**Answer:** {result['result']}")
                    
                    if show_details:
                        with st.expander("Details"):
                            st.write(f"**Tool used:** {result['tool']}")
                            st.write(f"**Processing time:** {result['time_elapsed']}s")
                            
                            if "retrieved_chunks" in result:
                                st.subheader("Relevant passages")
                                for i, chunk in enumerate(result["retrieved_chunks"], 1):
                                    st.text_area(f"Passage {i}", chunk, height=150)
                            
                            st.subheader("Processing steps")
                            for step in result["steps"]:
                                st.write(f"- {step}")
    else:
        st.info("Please upload documents to begin")

if __name__ == "__main__":
    main()