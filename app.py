import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 200
MAX_FILE_SIZE_MB = 5

# Initialize embedding model
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

# Document processing
def extract_text_from_file(uploaded_file):
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

def split_documents(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "• ", "  "],
        length_function=len
    )
    return text_splitter.split_text(text)

# Vector store
def create_vector_store(embedding_model, chunks):
    with st.spinner("Creating search index..."):
        try:
            return FAISS.from_texts(
                texts=chunks,
                embedding=embedding_model,
                metadatas=[{"source": f"chunk-{i}"} for i in range(len(chunks))]
            )
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            st.stop()

# Streamlit UI (simplified)
def main():
    st.set_page_config(page_title="Document Processor", page_icon="📄")
    
    st.title("📄 Document Processor")
    st.caption("This app demonstrates document processing and embedding")
    
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
    
    # Process documents button
    if st.button("Process Documents", disabled=not uploaded_files) and not st.session_state.processed:
        with st.status("Processing...", expanded=True) as status:
            try:
                # Load embedding model
                embedding_model = load_embedding_model()
                
                # Process files
                texts = []
                for file in uploaded_files:
                    with st.spinner(f"Processing {file.name}..."):
                        text = extract_text_from_file(file)
                        if text:
                            texts.append(text)
                
                if not texts:
                    st.error("No valid text extracted from any of the uploaded files")
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
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
    
    # Display results
    if st.session_state.processed:
        st.success("Documents processed successfully!")
        
        # Show sample chunks
        vs = st.session_state.vector_store
        if vs:
            st.subheader("Sample Document Chunks")
            # Get sample chunks
            sample_docs = vs.similarity_search("", k=3)
            for i, doc in enumerate(sample_docs):
                with st.expander(f"Chunk {i+1}"):
                    st.write(doc.page_content)

    # Reset button
    if st.session_state.processed:
        if st.button("Reset"):
            st.session_state.vector_store = None
            st.session_state.processed = False
            st.experimental_rerun()

if __name__ == "__main__":
    main()