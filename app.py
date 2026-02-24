import streamlit as st
import pandas as pd
import os

from preprocessing.text_cleaning import preprocess, clean_text_for_summary
from utils.file_handling import load_sample_corpus, process_uploaded_files
from utils.highlight import show_document_modal
from modeling import build_tfidf, cluster_docs, suggest_optimal_k, reduce_dimensions, extract_keywords, compute_similarity
from summarization import simple_summary
from visualization import plot_similarity_heatmap, plot_cluster_scatter, plot_silhouette_scores

st.set_page_config(page_title="Document Intelligence Tool", layout="wide")
st.title("📄 Document Intelligence Tool")

with st.sidebar:
    preserve_numbers = st.toggle("Preserve Numeric Content (e.g., 2023, 5.2%)", value=False)
    
    st.header("Data Source")
    use_sample = st.checkbox("Use Sample Documents")
    
    sample_corpus_name = None
    if use_sample:
        sample_corpora_options = {
            "Select a Corpus...": None,
            "Demo": "optimal_demo",
            "Limitation Demo": "semantic_limitation"
        }
        selected_corpus = st.selectbox("Select Sample Corpus", options=list(sample_corpora_options.keys()))
        sample_corpus_name = sample_corpora_options[selected_corpus]
        
        if sample_corpus_name == "optimal_demo":
            st.info(
                "**Demo Corpus**\n\n"
                "This corpus contains several machine learning and systems research papers "
                "(like Attention Is All You Need, BERT, ResNet, MapReduce, GFS) along with one unrelated document "
                "(a Cricket Rule Book).\n\n"
                "This perfectly demonstrates how TF-IDF and K-Means can successfully cluster distinctly themed documents "
                "into their appropriate technical domains based on terminology, isolating unrelated topics."
            )
        elif sample_corpus_name == "semantic_limitation":
            st.warning(
                "**Limitation Demo**\n\n"
                "This corpus contains three very short documents describing the exact same event "
                "(a customer purchasing a product online).\n\n"
                "However, each document uses completely different vocabulary (e.g., 'customer purchased' vs. 'buyer bought' vs. 'individual acquired').\n\n"
                "**Where our model fails:** Because TF-IDF relies entirely on exact word matches (lexical similarity) "
                "rather than the underlying meaning of words (semantic similarity), it will fail to recognize that these three documents mean the exact same thing. "
                "This will likely result in a low similarity score and poor clustering, demonstrating the limitations of classical TF-IDF approaches compared to modern semantic embeddings."
            )

    uploaded_files = st.file_uploader(
        "Upload text or PDF files (Minimum 2 required)",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        disabled=use_sample
    )

raw_docs = []
filenames = []

if use_sample and sample_corpus_name:
    raw_docs, filenames = load_sample_corpus(sample_corpus_name)
elif uploaded_files:
    if len(uploaded_files) < 2:
        st.warning("Please upload at least 2 documents to proceed with analysis.")
    else:
        raw_docs, filenames = process_uploaded_files(uploaded_files)

@st.cache_data
def run_pipeline(docs, preserve):
    processed = [preprocess(doc, preserve_numeric=preserve) for doc in docs]
    X, vectorizer = build_tfidf(processed)
    return processed, X, vectorizer

if raw_docs:
    st.success(f"Loaded {len(raw_docs)} documents.")
    
    with st.spinner("Vectorizing documents..."):
        processed_docs, X, vectorizer = run_pipeline(raw_docs, preserve_numbers)

    # ---------------------------------------------------------
    # 1. Similarity Matrix
    # ---------------------------------------------------------
    st.subheader("🔍 Cosine Similarity Matrix")
    with st.expander("What does this mean?"):
        st.write("Cosine similarity measures how similar two documents are based on the words they contain, ignoring document length. A score of 1 means they are exactly the same conceptually, while 0 means they have no words in common.")

    if len(raw_docs) >= 2:
        with st.spinner("Computing similarities..."):
            similarity = compute_similarity(X)
            fig_sim = plot_similarity_heatmap(similarity, filenames)
            st.plotly_chart(fig_sim, use_container_width=True)
    else:
        st.info("Similarity requires at least 2 documents. Upload more documents to compare them.")

    # ---------------------------------------------------------
    # 2. Clustering & Silhouette Score
    # ---------------------------------------------------------
    st.subheader("📊 Clustering")
    with st.expander("What does this mean?"):
        st.write("Clustering groups similar documents together automatically based on their text content. It helps discover underlying themes across a large collection of files without human labeling.")

    if len(raw_docs) >= 2:
        with st.spinner("Evaluating optimal clusters..."):
            suggested_k, silhouette_scores_dict = suggest_optimal_k(X)
        
        st.write("### Silhouette Score Evaluation")
        with st.expander("What does Silhouette Score mean?"):
             st.write("Silhouette score measures how well-separated the clusters are (range: -1 to 1). Higher is better. It tells us the optimal number of groups for the documents.")
             
        if len(raw_docs) >= 3 and suggested_k > 0:
            fig_silhouette = plot_silhouette_scores(silhouette_scores_dict)
            if fig_silhouette:
                st.plotly_chart(fig_silhouette, use_container_width=True)
                st.caption("Silhouette score measures how well-separated the clusters are (range: -1 to 1). Higher is better.")
        else:
            st.info("Cannot compute meaningful Silhouette scores for less than 3 documents.")

        k = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=max(2, min(6, len(raw_docs))),
            value=max(2, suggested_k) if suggested_k > 0 else 2
        )
        
        if k > 1 and len(raw_docs) >= k:
            with st.spinner("Clustering documents..."):
                labels = cluster_docs(X, k)
                coords = reduce_dimensions(X, n_components=2)
                
            fig_scatter = plot_cluster_scatter(coords, labels, filenames)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Compute cluster-level features
            cluster_texts = {i: "" for i in range(k)}
            for label, text in zip(labels, raw_docs):
                cluster_texts[label] += text + " "
                
            cluster_list = [cluster_texts[i] for i in range(k)]
            processed_clusters = [preprocess(c, preserve_numeric=preserve_numbers) for c in cluster_list]
            cluster_X, cluster_vectorizer = build_tfidf(processed_clusters)
            
            cluster_vocab_size = len(cluster_vectorizer.get_feature_names_out())
            dynamic_top_n = max(3, min(10, int(0.1 * cluster_vocab_size)))

            cluster_keywords = extract_keywords(cluster_vectorizer, cluster_X, top_n=dynamic_top_n)
            
            # Use custom cleaner for summarization
            cleaned_clusters_for_summary = [clean_text_for_summary(c, preserve_numeric=preserve_numbers) for c in cluster_list]
            cluster_summaries = [simple_summary(c, cluster_vectorizer, top_n=3) for c in cleaned_clusters_for_summary]

            st.subheader("📂 Document Clusters")
            for cluster_id in range(k):
                st.write(f"### Cluster {cluster_id}")
                st.markdown(f"**Cluster Keywords:** {', '.join(cluster_keywords[cluster_id])}")
                st.markdown(f"**Cluster Summary:** {' '.join(cluster_summaries[cluster_id])}")
                st.markdown("---")
                
                cluster_docs_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
                for idx in cluster_docs_indices:
                    doc_name = filenames[idx]
                    if st.button(doc_name, key=f"btn_cluster_{cluster_id}_{idx}"):
                        cleaned_doc = clean_text_for_summary(raw_docs[idx], preserve_numeric=preserve_numbers)
                        show_document_modal(
                            doc_name, 
                            raw_docs[idx],
                            cleaned_doc,
                            cluster_keywords[cluster_id], 
                            cluster_summaries[cluster_id]
                        )
        else:
            st.info(f"Need at least {k} documents to form {k} clusters.")
    else:
        st.info("Clustering requires at least 2 documents.")
        