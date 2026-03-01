import nltk
import streamlit as st
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True) 
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Ensure downloads happen on import
download_nltk_data()

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """Map POS tag to first character used by WordNetLemmatizer"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # Default

def clean_text(text, preserve_numeric=True):
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    
    if preserve_numeric:
        # Step 1: Keep letters, digits, whitespace, dot and %
        text = re.sub(r'[^a-z0-9\s.%]', ' ', text)

        # Step 2: Remove dots not between digits (avoid stray periods)
        text = re.sub(r'(?<!\d)\.(?!\d)', ' ', text)

        # Step 3: Remove % not directly attached to a digit
        text = re.sub(r'(?<!\d)%', ' ', text)
    else:
        # Remove all numbers and symbols
        text = re.sub(r'[^a-z\s]', ' ', text)
        
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def lemmatize_tokens(tokens):
    pos_tags = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]

def preprocess(text, preserve_numeric=True):
    cleaned = clean_text(text, preserve_numeric)
    tokens = tokenize_text(cleaned)
    tokens = lemmatize_tokens(tokens)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)

def clean_text_for_summary(text, preserve_numeric=True):
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Rejoin words split by a hyphen at a line break
    text = re.sub(r'-\s*\n\s*', '', text)
    
    if preserve_numeric:
        # Keep letters, digits, whitespace, %, sentence boundary punctuation, and hyphens
        text = re.sub(r'[^a-zA-Z0-9\s.%!?,\"\'-]', ' ', text)

        # Remove dots that are just standalone (avoid stray periods)
        text = re.sub(r'(?<!\d)\.(?!\d)(?!\s)', ' ', text)

        # Remove % not directly attached to a digit
        text = re.sub(r'(?<!\d)%', ' ', text)
    else:
        # Remove all numbers, but keep essential punctuation and hyphens
        text = re.sub(r'[^a-zA-Z\s.!?,\"\'-]', ' ', text)
        
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
