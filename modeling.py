from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def build_tfidf(docs):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=3000
    )
    X = vectorizer.fit_transform(docs)
    return X, vectorizer

def compute_similarity(X):
    return cosine_similarity(X)

def cluster_docs(X, k=3):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    return labels

def extract_keywords(vectorizer, X, top_n=10):
    feature_names = np.array(vectorizer.get_feature_names_out())
    keywords = []

    for doc_vector in X:
        sorted_indices = doc_vector.toarray().flatten().argsort()[-top_n:]
        keywords.append(feature_names[sorted_indices])

    return keywords

def topic_modeling(X, n_topics=3):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    return lda

def reduce_dimensions(X, n_components=2):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(X)