# RAG-Powered-Q-A-Assistant



📄 Document Q&A Assistant
An intelligent AI-powered assistant that allows users to upload PDF, DOCX, or TXT files and ask questions about their contents. This assistant uses a Retrieval-Augmented Generation (RAG) approach combining Hugging Face models, FAISS vector search, and Streamlit UI to deliver fast and context-aware responses.

🚀 Features
✅ Supports PDF, TXT, and Word (DOCX) file formats

✅ Extracts and splits documents into intelligent chunks for semantic search

✅ Uses FAISS for fast vector-based document retrieval

✅ Integrates HuggingFace models for embeddings and text generation

✅ Includes a built-in calculator and dictionary tool

✅ Designed with a responsive and intuitive Streamlit UI

✅ Efficient error handling and detailed response logs

🧠 Tech Stack
Component	Description
Streamlit	Frontend framework for interactive UI
FAISS	Fast vector similarity search for chunk retrieval
sentence-transformers	For generating embeddings from text chunks
google/flan-t5-base	LLM used for generating answers from context
LangChain	Orchestration of retrieval and QA chain
transformers	Hugging Face pipelines for text generation
PyPDF2, docx	For extracting text from PDF and DOCX files
