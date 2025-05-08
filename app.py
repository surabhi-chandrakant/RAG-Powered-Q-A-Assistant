import os
import re
import time
from typing import List, Dict, Any, Optional
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS  # Changed from Chroma to FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
# Uncomment if needed for fallback loading
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Configuration - Using smaller models better suited for Streamlit Cloud
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Small embedding model
LLM_NAME = "google/flan-t5-small"  # Small T5 model that should work with current dependencies
CHUNK_SIZE = 600
CHUNK_OVERLAP = 200
MAX_FILE_SIZE_MB = 5  # Reduced max file size

# Initialize models with better caching and error handling
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        st.error(f"Failed to load embedding model: {str(e)}")
        st.stop()

@st.cache_resource(show_spinner="Loading language model...")
def load_llm():
    try:
        # First try loading with pipeline
        pipe = pipeline(
            "text2text-generation",
            model=LLM_NAME,
            max_length=512,  # Reduced max length
            temperature=0.3,
            device_map="auto"
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        st.info("Try using a smaller model like 'google/flan-t5-small' instead")
        st.stop()
        
        # Commented out fallback loading to simplify
        # st.warning(f"Pipeline loading failed, trying alternative method: {str(e)}")
        # try:
        #     # Try explicit loading as fallback
        #     tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
        #     model = AutoModelForSeq2SeqLM.from_pretrained(LLM_NAME)
        #     pipe = pipeline(
        #         "text2text-generation",
        #         model=model,
        #         tokenizer=tokenizer,
        #         max_length=512,
        #         temperature=0.3
        #     )
        #     return HuggingFacePipeline(pipeline=pipe)
        # except Exception as e2:
        #     st.error(f"Failed to load LLM with both methods: {str(e2)}")
        #     st.stop()

# Document processing
def extract_text_from_file(uploaded_file) -> Optional[str]:
    try:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.warning(f"File {uploaded_file.name} exceeds {MAX_FILE_SIZE_MB}MB limit ({file_size_mb:.2f}MB)")
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
        else:
            st.warning(f"Unsupported file type: {uploaded_file.type}")
            return None
            
        if not text.strip():
            st.warning(f"No text extracted from {uploaded_file.name}")
            return None
            
        return text
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

# Vector store (changed to FAISS which doesn't require persistence)
def create_vector_store(_embedding_model, chunks: List[str]):
    with st.spinner("Creating search index..."):
        try:
            return FAISS.from_texts(
                texts=chunks,
                embedding=_embedding_model,
                metadatas=[{"source": f"chunk-{i}"} for i in range(len(chunks))]
            )
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            st.stop()

# Tools
def calculate(expression: str) -> str:
    try:
        # Sanitize input more thoroughly
        expression = re.sub(r'[^0-9+\-*/.() ]', '', expression)
        if len(expression) > 50:
            return "Expression too complex"
        result = eval(expression, {'__builtins__': None}, {})
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Could not compute the expression: {str(e)}"

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
            search_kwargs={"k": 3}  # Simplified retriever config
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
        st.error(f"RAG error: {str(e)}")
        return {
            "result": "Sorry, I encountered an error processing your query. Please try again with a different question.",
            "tool": "error",
            "time_elapsed": round(time.time() - start_time, 2)
        }

# Streamlit UI with improved error handling
def main():
    st.set_page_config(
        page_title="Document Q&A",
        page_icon="📄",
        layout="centered"
    )
    
    st.title("📄 Document Q&A Assistant")
    st.caption("Upload documents and get AI-powered answers")
    
    # Show system info
    with st.expander("System Info"):
        st.write(f"Using embedding model: {EMBEDDING_MODEL}")
        st.write(f"Using language model: {LLM_NAME}")
        st.write(f"Max file size: {MAX_FILE_SIZE_MB}MB")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )
    
    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "processing_error" not in st.session_state:
        st.session_state.processing_error = False
    
    # Process documents button
    col1, col2 = st.columns([3, 1])
    process_button = col2.button("Process Documents", disabled=not uploaded_files)
    
    if process_button and not st.session_state.processed:
        st.session_state.processing_error = False
        with st.status("Processing...", expanded=True) as status:
            try:
                # Load models first to catch any model loading errors early
                embedding_model = load_embedding_model()
                llm = load_llm()
                
                # Process files
                texts = []
                for file in uploaded_files:
                    with st.spinner(f"Processing {file.name}..."):
                        text = extract_text_from_file(file)
                        if text:
                            texts.append(text)
                
                if not texts:
                    st.error("No valid text extracted from any of the uploaded files")
                    st.session_state.processing_error = True
                else:
                    all_text = "\n\n".join(texts)
                    chunks = split_documents(all_text)
                    st.info(f"Extracted {len(chunks)} text chunks")
                    
                    if len(chunks) > 0:
                        st.session_state.vector_store = create_vector_store(embedding_model, chunks)
                        st.session_state.processed = True
                        status.update(label=f"Processed {len(chunks)} chunks", state="complete")
                    else:
                        st.error("No text chunks created")
                        st.session_state.processing_error = True
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
                st.session_state.processing_error = True
    
    # Reset button
    if st.session_state.processed or st.session_state.processing_error:
        if st.button("Reset"):
            st.session_state.vector_store = None
            st.session_state.processed = False
            st.session_state.processing_error = False
            st.experimental_rerun()
    
    # Query interface
    if st.session_state.processed:
        st.success("Documents processed successfully! You can now ask questions.")
        if query := st.chat_input("Ask about your documents"):
            st.chat_message("user").write(query)
            with st.spinner("Thinking..."):
                llm = load_llm()  # Ensure model is loaded
                result = route_query(query, st.session_state.vector_store, llm)
                
                with st.chat_message("assistant"):
                    st.write(result["result"])
                    
                    with st.expander("Show details"):
                        st.caption(f"Method: {result['tool']} | Time: {result['time_elapsed']}s")
                        if "retrieved_chunks" in result:
                            st.subheader("Relevant passages")
                            for i, c in enumerate(result["retrieved_chunks"], 1):
                                st.text_area(f"Passage {i}", c, height=100)
    elif not st.session_state.processing_error:
        st.info("Upload documents and click 'Process Documents' to begin")

if __name__ == "__main__":
    main()