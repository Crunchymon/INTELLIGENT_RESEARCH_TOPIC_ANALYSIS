import os
import streamlit as st
from pypdf import PdfReader

def load_sample_corpus(sample_dir_name):
    """Load text and pdf files from a specific sample corpus directory."""
    raw_docs = []
    filenames = []
    
    base_dir = "sample_corpora"
    full_path = os.path.join(base_dir, sample_dir_name)
    
    if os.path.exists(full_path):
        for filename in sorted(os.listdir(full_path)):
            if filename.endswith(".txt") or filename.endswith(".pdf"):
                file_path = os.path.join(full_path, filename)
                text = read_file_content(file_path, filename)
                if text:
                    raw_docs.append(text)
                    filenames.append(filename)
    else:
        st.error(f"Sample corpus directory '{full_path}' not found.")
        
    return raw_docs, filenames

def process_uploaded_files(uploaded_files):
    """Process uploaded Streamlit file objects."""
    raw_docs = []
    filenames = []
    
    for file in uploaded_files:
        if file.name.lower().endswith((".txt", ".pdf")):
            text = read_uploaded_file(file)
            if text:
                raw_docs.append(text)
                filenames.append(file.name)
        else:
            st.warning(f"Unsupported file format: {file.name}. Only .txt and .pdf are supported.")
            
    return raw_docs, filenames

def read_file_content(path, filename):
    """Read content from a local file path."""
    text = ""
    try:
        if filename.lower().endswith(".pdf"):
            pdf = PdfReader(path)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        else:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
    except Exception as e:
        st.error(f"Error reading file {filename}: {str(e)}")
        return None
    return text

def read_uploaded_file(file):
    """Read content from a Streamlit UploadedFile object."""
    text = ""
    try:
        if file.name.lower().endswith(".pdf"):
            pdf = PdfReader(file)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        else:
            text = file.getvalue().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Error extracting text from {file.name}: {str(e)}")
        return None
    return text
