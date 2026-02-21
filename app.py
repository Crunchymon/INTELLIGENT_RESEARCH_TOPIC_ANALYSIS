import streamlit as st
import pandas as pd
from preprocessing import preprocess
from modeling import *
from utils import simple_summary
from pypdf import PdfReader

st.title("📄 Document Intelligence Tool")

uploaded_files = st.file_uploader(
    "Upload text files",
    accept_multiple_files=True
)

if uploaded_files:

    raw_docs = []
    filenames = []

    for file in uploaded_files:
        if file.name.lower().endswith(".pdf"):
            pdf = PdfReader(file)
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        else:
            text = file.getvalue().decode("utf-8", errors="ignore")
            
        raw_docs.append(text)
        filenames.append(file.name)

    processed_docs = [preprocess(doc) for doc in raw_docs]

    X, vectorizer = build_tfidf(processed_docs)

    if len(raw_docs) >= 2:
        st.subheader("🔍 Cosine Similarity Matrix")
        similarity = compute_similarity(X)
        st.write(pd.DataFrame(similarity, index=filenames, columns=filenames))

        st.subheader("📊 Clustering")
        max_k = min(len(raw_docs), 6)
        k = st.slider("Number of clusters", min_value=1, max_value=max_k, value=min(3, max_k))
        
        if k > 1:
            labels = cluster_docs(X, k)
            cluster_df = pd.DataFrame({"Document": filenames, "Cluster": labels})
            st.write(cluster_df)
        else:
            st.info("Need at least 2 clusters to perform clustering.")
    else:
        st.warning("Please upload at least 2 documents to view similarity and clustering.")

    st.subheader("🏷 Keywords")
    keywords = extract_keywords(vectorizer, X)
    for name, words in zip(filenames, keywords):
        st.write(f"**{name}**:", ", ".join(words))

    st.subheader("🧠 Extractive Summary")
    for name, doc in zip(filenames, raw_docs):
        summary = simple_summary(doc, vectorizer)
        st.write(f"**{name} Summary:**")
        st.write(summary)