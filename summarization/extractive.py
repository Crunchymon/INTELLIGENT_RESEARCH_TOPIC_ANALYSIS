import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

def simple_summary(text, vectorizer, top_n=3):
    # Use NLTK for robust sentence splitting
    raw_sentences = sent_tokenize(text)
    
    # Filter extremely short sentences
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]
    
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
    
    # Build final summary sentences, ensuring punctuation is reasonable
    summary = []
    for i in chronological_indices:
        sent = sentences[i].strip()
        if not sent.endswith(('.', '!', '?', '"', "'")):
            sent += "."
        summary.append(sent)

    return summary
