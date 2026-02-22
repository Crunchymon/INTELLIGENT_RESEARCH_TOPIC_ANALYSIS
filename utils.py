import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def simple_summary(text, vectorizer, top_n=3):
    # Filter empty sentences
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 5]
    
    if not sentences:
        return ""
        
    sentence_vectors = vectorizer.transform(sentences)
    sim_matrix = cosine_similarity(sentence_vectors)
    scores = sim_matrix.sum(axis=1)

    # Make sure we don't try to extract more sentences than exist
    n_sentences = min(top_n, len(sentences))
    
    # Get indices of top ranked sentences
    ranked_indices = np.argsort(scores)[-n_sentences:]
    
    # Sort indices so the summary reads chronologically
    chronological_indices = sorted(ranked_indices)
    
    summary = [sentences[i].strip() + "." for i in chronological_indices]

    return summary