# 📄 Document Intelligence Tool

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)
![NLP](https://img.shields.io/badge/NLP-Classical-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An interactive **Streamlit-based NLP system** that analyzes, compares, and clusters multiple documents (PDF or TXT) using classical Natural Language Processing techniques.

This tool uncovers thematic structure, similarity patterns, and representative summaries across document collections — while maintaining full explainability and algorithmic transparency.

---

## 🚀 Overview

When working with multiple research papers, reports, or articles, it becomes difficult to:

- Identify which documents are similar  
- Discover emerging themes  
- Group related documents automatically  
- Extract meaningful keywords and summaries  

This application solves that problem using **classical NLP (no embeddings, no LLMs)** — ensuring interpretability and transparency.

---

## 🎯 Core Objectives

Given a collection of heterogeneous documents:

1. Quantify pairwise document similarity  
2. Automatically cluster documents into coherent groups  
3. Extract representative keywords per cluster  
4. Generate extractive summaries  
5. Provide intuitive visualizations  

---

## 🧠 Key Features

### 📂 Document Ingestion
- Upload multiple PDF or TXT files simultaneously
- Built-in demo datasets is also included

### 🧹 Intelligent Preprocessing Pipeline
- Tokenization  
- POS-aware lemmatization  
- Stopword removal  
- Optional numeric preservation (%, decimals, figures)

### 📊 TF-IDF Vectorization
- Unigrams + Bigrams  
- Dynamic feature scaling  
- Sparse matrix optimization  

### 🔎 Similarity Analysis
- Cosine similarity matrix  
- Interactive heatmap visualization  

### 📌 Automatic Clustering
- K-Means clustering  
- Automatic K selection using Silhouette Score  
- Manual override via UI slider  

### 🔍 Cluster Introspection
For each cluster:
- Characteristic keyword extraction  
- Sentence-level extractive summarization  
- Interactive document viewer with highlighted insights  

### 📈 Visual Analytics
- Cosine Similarity Heatmap  
- PCA-based 2D Projection  
- Interactive document modal  

---

## ⚙️ Methodology

### 1️⃣ Lexical Processing
- Tokenization  
- POS-aware Lemmatization  
- Stopword Removal  
- Optional Numeric Preservation  

### 2️⃣ Vector Representation
- TF-IDF (Unigrams + Bigrams)  
- Dynamic Vocabulary Scaling  

### 3️⃣ Similarity Computation
- Cosine Similarity (length-normalized dot product)

### 4️⃣ Clustering
- K-Means in high-dimensional sparse space  
- Silhouette-based automatic K selection  

### 5️⃣ Cluster-Level Analysis
- Keyword scoring via TF-IDF weights  
- Extractive summarization via sentence scoring  

### 6️⃣ Visualization
- Heatmaps  
- PCA projections  
- Interactive document modal  

---

## 📊 Evaluation Strategy

Since this is an **unsupervised system**, traditional accuracy metrics do not apply.

Instead, evaluation is performed using:

- **Silhouette Score** — cluster separation quality  
- **Intra vs Inter-cluster similarity margins**  
- **Qualitative keyword interpretability**

---

## 🛠 Optimization Decisions

- Dynamic TF-IDF feature scaling to reduce sparsity  
- Compared static vs adaptive vocabulary sizes  
- Evaluated cosine vs Euclidean distance in sparse space  
- Implemented silhouette-based automatic K selection  

---

## 🧩 Assumptions & Design Philosophy

### 📌 Lexical Importance (TF-IDF)
Term frequency relative to corpus frequency approximates thematic importance.

### 📌 Distance Metric Choice
Cosine similarity normalizes document length and focuses on directional similarity.

### 📌 Cluster Geometry
K-Means assumes relatively spherical clusters in high-dimensional TF-IDF space.

### 📌 Transparency First
Extractive summarization is used instead of abstractive generation to:
- Avoid hallucinations  
- Maintain provenance  
- Ensure explainability  

---

## ⚠️ Limitations

### ❌ No Semantic Understanding
TF-IDF relies strictly on lexical overlap.  
Example: *"car"* and *"automobile"* are treated as different features.

### ❌ Order-Agnostic Representation
Bag-of-Words assumption ignores deeper syntactic and contextual structure.

### ❌ High-Dimensional Sparsity
Scaling to very large corpora may require dimensionality reduction.

---

## 🧪 Built-in Demonstrations

### 1️⃣ Optimal Demo (`optimal_demo`)

**Contents:**
- Attention Is All You Need  
- BERT  
- ResNet  
- MapReduce  
- GFS  
- Cricket Rule Book (outlier)

**Purpose:**  
Demonstrates strong clustering when vocabulary domains are clearly distinct.

---

### 2️⃣ Semantic Limitation Demo (`semantic_limitation`)

**Contents:**  
Three short texts describing the same event using completely different vocabulary.

Example:
- "Customer purchased"  
- "Buyer bought"  
- "Individual acquired"

**Purpose:**  
Shows TF-IDF’s inability to recognize semantic similarity without embeddings.

---

## 🏗 Tech Stack

- Python  
- Streamlit  
- Scikit-learn  
- NLTK / SpaCy  
- NumPy  
- Pandas  
- Matplotlib / Seaborn  

---

## 💻 Installation

```bash
pip install -r requirements.txt
