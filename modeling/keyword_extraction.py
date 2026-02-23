import numpy as np

def extract_keywords(vectorizer, X, top_n=10):
    feature_names = np.array(vectorizer.get_feature_names_out())
    keywords = []

    for doc_vector in X:
        sorted_indices = doc_vector.toarray().flatten().argsort()[-top_n:]
        keywords.append(feature_names[sorted_indices].tolist())

    return keywords
