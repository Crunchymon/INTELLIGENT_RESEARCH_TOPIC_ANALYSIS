from sklearn.feature_extraction.text import TfidfVectorizer

def dynamic_max_features(docs):
    all_tokens = " ".join(docs).split()
    unique_vocab = len(set(all_tokens))
    
    # Keep 15% of unique vocabulary
    max_features = int(0.15 * unique_vocab)
    
    max_features = max(20, max_features)   # lower bound
    max_features = min(150, max_features)  # upper bound

    return max_features

def build_tfidf(docs):
    n_docs = len(docs)

    min_df = max(2, int(0.1 * n_docs))
    max_df = 0.85  # remove overly common words
    max_features = dynamic_max_features(docs)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df= min_df,
        max_df = max_df,
        max_features= max_features
    )

    X = vectorizer.fit_transform(docs)

    return X, vectorizer
