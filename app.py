import streamlit as st
import pandas as pd
from preprocessing import preprocess
from modeling import *
from utils import simple_summary
from pypdf import PdfReader
import plotly.express as px
import os


st.title("📄 Document Intelligence Tool")

use_sample = st.checkbox("Use Sample Documents")

uploaded_files = st.file_uploader(
    "Upload text files",
    accept_multiple_files=True,
    disabled=use_sample
)

raw_docs = []
filenames = []

if use_sample:
    sample_dir = "sample_corpus"
    if os.path.exists(sample_dir):
        for filename in sorted(os.listdir(sample_dir)):
            if filename.endswith(".txt") or filename.endswith(".pdf"):
                path = os.path.join(sample_dir, filename)
                if filename.lower().endswith(".pdf"):
                    pdf = PdfReader(path)
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    raw_docs.append(text)
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        raw_docs.append(f.read())
                filenames.append(filename)
    else:
        st.error(f"Sample corpus directory '{sample_dir}' not found.")
elif uploaded_files:
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

if raw_docs:
    processed_docs = [preprocess(doc) for doc in raw_docs]

    X, vectorizer = build_tfidf(processed_docs)

    st.subheader("🔍 Cosine Similarity Matrix")
    if len(raw_docs) >= 2:
        similarity = compute_similarity(X)
        sim_df = pd.DataFrame(similarity, index=filenames, columns=filenames)
        
        fig_sim = px.imshow(
            sim_df, 
            text_auto=".2f", 
            aspect="auto", 
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig_sim, use_container_width=True)
    else:
        st.warning("Please upload at least 2 documents to view similarity.")

    st.subheader("📊 Clustering")
    if len(raw_docs) >= 2:
        max_k = min(len(raw_docs), 6)
        k = st.slider("Number of clusters", min_value=1, max_value=max_k, value=min(3, max_k))
        
        if k > 1:
            labels = cluster_docs(X, k)
            cluster_df = pd.DataFrame({"Document": filenames, "Cluster": labels})
            st.write(cluster_df)
            
            coords = reduce_dimensions(X, n_components=2)
            cluster_df['PCA1'] = coords[:, 0]
            cluster_df['PCA2'] = coords[:, 1]
            cluster_df['Cluster'] = cluster_df['Cluster'].astype(str)
            
            fig = px.scatter(
                cluster_df, x='PCA1', y='PCA2', color='Cluster', 
                hover_name='Document', title="Cluster Visualization (2D PCA)"
            )
            fig.update_traces(marker=dict(size=20, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig)
            
            st.subheader("🏷 Keywords by Cluster")
            cluster_texts = {i: "" for i in range(k)}
            for label, text in zip(labels, raw_docs):
                cluster_texts[label] += text + " "
                
            cluster_list = [cluster_texts[i] for i in range(k)]
            processed_clusters = [preprocess(c) for c in cluster_list]
            cluster_X, cluster_vectorizer = build_tfidf(processed_clusters)
            keywords = extract_keywords(cluster_vectorizer, cluster_X)
            
            for i, words in enumerate(keywords):
                st.write(f"**Cluster {i}**:", ", ".join(words))

            st.subheader("🧠 Extractive Summary by Cluster")
            for i, text in enumerate(cluster_list):
                summary = simple_summary(text, cluster_vectorizer, 2)
                st.write(f"**Cluster {i} Summary:**")
                st.write(summary)
        else:
            st.info("Need at least 2 clusters to perform clustering.")
    else:
        st.warning("Please upload at least 2 documents to view clustering.")

    if len(raw_docs) < 2 or (len(raw_docs) >= 2 and k == 1):
        st.subheader("🏷 Keywords")
        keywords = extract_keywords(vectorizer, X)
        for name, words in zip(filenames, keywords):
            st.write(f"**{name}**:", ", ".join(words))

        st.subheader("🧠 Extractive Summary")
        for name, doc in zip(filenames, raw_docs):
            summary = simple_summary(doc, vectorizer, 2)
            st.write(f"**{name} Summary:**")
            st.write(summary)