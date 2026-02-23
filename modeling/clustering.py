from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def cluster_docs(X, k=3):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    return labels

def suggest_optimal_k(X, max_k=6):
    n_docs = X.shape[0]
    
    # We need at least 2 clusters and at most n_docs - 1
    upper_bound = min(max_k, n_docs - 1)

    best_k = 2
    best_score = -1
    scores_dict = {}

    if n_docs < 3:
        # Cannot compute silhouette for range of k properly if extremely few docs
        return best_k, scores_dict

    for k in range(2, upper_bound + 1):
        try:
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X)
            
            # Need at least 2 unique labels to compute silhouette
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
            else:
                score = -1
                
            scores_dict[k] = float(score)

            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            scores_dict[k] = -1.0

    return best_k, scores_dict
