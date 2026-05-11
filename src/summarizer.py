import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.gemini_service import get_gemini_key_points

# Download necessary NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def extract_key_sentences(text, num_sentences=10):
    """
    Extract key sentences from the text using TF-IDF.
    
    Args:
        text (str): Text to summarize
        num_sentences (int): Number of sentences to extract
        
    Returns:
        list: List of key sentences
    """
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # If we have fewer sentences than requested, return all sentences
    if len(sentences) <= num_sentences:
        return sentences
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    # Fit and transform the sentences
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calculate sentence scores based on TF-IDF values
    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    
    # Get indices of top sentences
    top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
    
    # Sort indices to maintain original order
    top_indices = sorted(top_indices)
    
    # Extract top sentences
    key_sentences = [sentences[i] for i in top_indices]
    
    return key_sentences

def extract_key_points(text):
    """
    Extract key points from the text using a hybrid approach.
    First tries using Gemini AI, then falls back to extractive summarization.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        list: List of key points
    """
    try:
        # Try using Gemini AI first
        ai_points = get_gemini_key_points(text)
        
        # If we get meaningful results, return them
        if ai_points and len(ai_points) >= 5:
            return ai_points
    except Exception as e:
        print(f"Error using Gemini for key points: {str(e)}")
    
    # Fall back to extractive summarization if Gemini fails
    key_sentences = extract_key_sentences(text, num_sentences=10)
    
    # Clean up and format the sentences
    key_points = [f"{sent.strip()}" for sent in key_sentences]
    
    return key_points

def generate_extractive_summary(text, ratio=0.3):
    """
    Generate an extractive summary of the text.
    
    Args:
        text (str): Text to summarize
        ratio (float): Proportion of original text to keep
        
    Returns:
        str: Extractive summary
    """
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Calculate the number of sentences to keep
    num_sentences = max(3, int(len(sentences) * ratio))
    
    # Extract key sentences
    key_sentences = extract_key_sentences(text, num_sentences)
    
    # Join the sentences to form the summary
    summary = " ".join(key_sentences)
    
    return summary

def identify_legal_entities(text):
    """
    Identify important legal entities in the text.
    
    Args:
        text (str): Legal text
        
    Returns:
        dict: Dictionary of entity types and their instances
    """
    # This is a simple implementation - a more sophisticated approach would use NER
    entities = {
        "cases": [],
        "statutes": [],
        "courts": [],
        "parties": []
    }
    
    # Simple pattern matching for common formats
    # Case citations (very simplified)
    case_patterns = [
        r'[A-Z][a-z]+ v[s]?\.? [A-Z][a-z]+',  # Simple case name pattern
        r'\(\d{4}\)\s+\d+\s+[A-Z]+\s+\d+'     # Year and reporter pattern
    ]
    
    # Statute patterns
    statute_patterns = [
        r'section \d+',
        r'act(?: of)? \d{4}',
        r'article \d+'
    ]
    
    # Court patterns
    court_patterns = [
        r'Supreme Court',
        r'High Court',
        r'District Court',
        r'Court of Appeal',
        r'Tribunal'
    ]
    
    # Simple pattern matching for each entity type
    for pattern in case_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["cases"].extend(matches)
    
    for pattern in statute_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["statutes"].extend(matches)
    
    for pattern in court_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["courts"].extend(matches)
    
    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities