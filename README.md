# Document Intelligence Tool

A Streamlit-based Natural Language Processing application that analyzes, compares, and clusters multiple documents (text or PDF) simultaneously. The tool allows users to uncover underlying themes across file collections using classical NLP techniques, offering deep explainability and visual introspection.

## Problem Statement

Given a collection of heterogeneous documents, the goal is to:
1. Quantify pairwise similarity
2. Automatically group documents into coherent clusters
3. Extract representative keywords and summaries
4. Provide interpretable visualizations

The challenge lies in performing this analysis using classical NLP methods without semantic embeddings.

## Features

- **Document Ingestion:** Upload multiple PDFs or text files to analyze them simultaneously.
- **Preprocessing Pipeline:** Cleans text, removes stopwords, and lemmatizes words. Provides the option to preserve numerical values, decimals, and percentage symbols.
- **TF-IDF Vectorization:** Converts documents into numerical vectors based on Term Frequency-Inverse Document Frequency, supporting both unigrams and bigrams.
- **Cosine Similarity Matrix:** Visualizes the lexical similarity between documents using an interactive heatmap.
- **K-Means Clustering & Silhouette Analysis:** Groups documents automatically based on content. Recommends the optimal number of clusters using Silhouette scores, but allows for manual adjustment via a slider.
- **Cluster Introspection:** For each cluster, the app automatically extracts characteristic **Keywords** and generates an **Extractive Summary**.
- **Document Modal Viewer:** Click on individual documents within a cluster to view their original and cleaned contents, with keywords and summary sentences interactively highlighted.

## Methodology

1. Lexical Preprocessing
   - Tokenization
   - POS-aware Lemmatization
   - Stopword Removal
   - Optional Numeric Preservation

2. Vector Representation
   - TF-IDF (Unigrams + Bigrams)
   - Dynamic Feature Scaling

3. Similarity Computation
   - Cosine Similarity (Normalized Dot Product)

4. Clustering
   - K-Means
   - Silhouette Score for Auto-K Selection

5. Cluster-Level Analysis
   - Keyword Extraction
   - Sentence-Level Extractive Summarization

6. Visualization
   - Similarity Heatmap
   - PCA Projection
   - Interactive Document Modal

## Evaluation

Since this is an unsupervised system, traditional accuracy metrics do not apply.

We evaluate performance using:
- Silhouette Score (cluster separation quality)
- Intra-domain vs Inter-domain similarity margins
- Qualitative keyword interpretability

## Optimization

- Implemented dynamic TF-IDF feature scaling to reduce sparsity.
- Compared static vs dynamic vocabulary sizes.
- Used silhouette-based automatic cluster selection.
- Evaluated cosine vs Euclidean distance behavior in sparse space.

## Assumptions & Reasoning

*   **Lexical Importance (TF-IDF):** This tool operates under the assumption that the frequency of specific terms (and continuous multi-word structures) relative to the broader corpus is a reliable proxy for a document's central themes.
*   **Dimensionality and Distance (Cosine Similarity):** Cosine distance is used over Euclidean distance to normalize against arbitrary document lengths, analyzing only the frequency vectors of the content.
*   **Cluster Structure (K-Means):** It is assumed that similar documents will group together into relatively uniform, spherical clusters in the high-dimensional TF-IDF space.
*   **Transparency (Extractive Summarization over Abstractive):** Sentences are extracted directly from the text based on cumulative TF-IDF scores rather than abstractive generative (LLM) summaries. This eliminates "hallucinations" and ensures provenance and transparency for analysis.

## Limitations

*   **Lack of Semantic Understanding:** TF-IDF relies almost entirely on exact string matching (lexical similarity). It does not understand synonyms, context, or language semantics. For example, "automobile" and "car" are treated as completely different dimensions in the vector space.
*   **Order Agnostic:** Underneath, TF-IDF is largely a "Bag of Words" approach. While bigrams (two-word sequences) capture some local context, the overall paragraph structure and grammar rules are ignored.
*   **Dimensionality / Sparsity:** Scaling to thousands of documents with massive unique vocabularies creates highly sparse matrices that might require heavy dimensionality reduction before clustering.

## Built-in Demos

The application ships with sample corpora to demonstrate its capabilities and constraints.

### 1. Optimal Demo (`optimal_demo`)
*   **Contents:** Contains several landmark Research Papers across computer science fields (e.g., *Attention Is All You Need*, *BERT*, *ResNet*, *MapReduce*, *GFS*) and one completely unrelated outlier (a *Cricket Rule Book*).
*   **Purpose:** Perfectly demonstrates when this classical framework shines. The vocabulary domains are distinct and highly specific (e.g., neural, blocks, arrays, innings, umpire). K-Means easily separates the documents into proper groupings based entirely on domain terminology, isolating unrelated topics.

### 2. Limitation Demo (`semantic_limitation`)
*   **Contents:** Contains three extremely short texts describing the exact same event (a customer buying a product online). However, each document uses completely different vocabulary (e.g., "customer purchased" vs. "buyer bought" vs. "individual acquired").
*   **Purpose:** Shows the critical failure point of TF-IDF. Because there is little to no overlap in the actual characters of the words, the Cosine Similarity will report near-zero similarity, proving that classical approaches cannot inherently link synonymous terms without relying on modern Semantic Embeddings.

## Installation

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```
