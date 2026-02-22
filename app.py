import streamlit as st
import pandas as pd
from preprocessing import preprocess
from modeling import *
from utils import simple_summary
from pypdf import PdfReader
import plotly.express as px
import os
import re

def highlight_text(text, keywords, summary_sentences):
    # Highlight summary sentences first (blue)
    for sent in summary_sentences:
        pattern = re.escape(sent.strip())
        text = re.sub(
            pattern,
            f"<mark style='background-color:#90caf9; color:black'>{sent}</mark>",
            text,
            flags=re.IGNORECASE
        )

    # Highlight keywords (yellow)
    for word in keywords:
        pattern = r"\b" + re.escape(word) + r"\b"
        text = re.sub(
            pattern,
            f"<span style='background-color:#ffd54f; color:black'>{word}</span>",
            text,
            flags=re.IGNORECASE
        )

    return text


@st.dialog("Document View", width="large")
def show_document_modal(doc_name, doc_text, keywords, summary_sentences):
    st.markdown(
        f"### {doc_name}\n\n"
        "**Legend:** \n"
        "<span style='background-color:#90caf9; color:black; padding: 2px 6px; border-radius: 4px; font-weight: bold;'>Summary Sentence</span> \n"
        "&nbsp;&nbsp;\n"
        "<span style='background-color:#ffd54f; color:black; padding: 2px 6px; border-radius: 4px; font-weight: bold;'>Keyword</span>\n"
        "<hr style='margin-top: 10px; margin-bottom: 10px;'>", 
        unsafe_allow_html=True
    )
    highlighted = highlight_text(doc_text, keywords, summary_sentences)
    st.markdown(
        f"""
        <div style="
            max-height: 500px;
            overflow-y: auto;
            padding: 1rem;
            border-radius: 8px;
            background-color: #111;
            line-height: 1.6;
            font-size: 16px;
        ">
        {highlighted}
        </div>
        """,
        unsafe_allow_html=True
    )

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
            
            # Compute cluster-level features to show under cluster heading
            cluster_texts = {i: "" for i in range(k)}
            for label, text in zip(labels, raw_docs):
                cluster_texts[label] += text + " "
                
            cluster_list = [cluster_texts[i] for i in range(k)]
            processed_clusters = [preprocess(c) for c in cluster_list]
            cluster_X, cluster_vectorizer = build_tfidf(processed_clusters)
            cluster_keywords = extract_keywords(cluster_vectorizer, cluster_X, top_n=8)
            cluster_summaries = [simple_summary(c, cluster_vectorizer, top_n=3) for c in cluster_list]

            st.subheader("📂 Document Clusters")
            for cluster_id in range(k):
                st.write(f"### Cluster {cluster_id}")
                
                # Show cluster-level insights
                st.markdown(f"**Cluster Keywords:** {', '.join(cluster_keywords[cluster_id])}")
                st.markdown(f"**Cluster Summary:** {' '.join(cluster_summaries[cluster_id])}")
                st.markdown("---")
                
                cluster_docs_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
                for idx in cluster_docs_indices:
                    doc_name = filenames[idx]
                    if st.button(doc_name, key=f"btn_cluster_{cluster_id}_{idx}"):
                        show_document_modal(
                            doc_name, 
                            raw_docs[idx], 
                            cluster_keywords[cluster_id], 
                            cluster_summaries[cluster_id]
                        )
        else:
            st.info("Need at least 2 clusters to perform clustering.")
            
            st.subheader("📂 All Documents")
            # Fallback to computing global insights when k=1 or not clustered
            global_keywords = extract_keywords(vectorizer, X, top_n=8)
            global_summaries = [simple_summary(doc, vectorizer, top_n=4) for doc in raw_docs]
            for idx, doc_name in enumerate(filenames):
                if st.button(doc_name, key=f"btn_all_{idx}"):
                    show_document_modal(doc_name, raw_docs[idx], global_keywords[idx], global_summaries[idx])
    else:
        st.warning("Please upload at least 2 documents to view clustering.")
        
        st.subheader("📂 All Documents")
        # Fallback to computing global insights when fewer than 2 docs entirely
        global_keywords = extract_keywords(vectorizer, X, top_n=8)
        global_summaries = [simple_summary(doc, vectorizer, top_n=4) for doc in raw_docs]
        for idx, doc_name in enumerate(filenames):
            if st.button(doc_name, key=f"btn_single_{idx}"):
                show_document_modal(doc_name, raw_docs[idx], global_keywords[idx], global_summaries[idx])