import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from textstat import flesch_kincaid_grade

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# List of common legal terms for analysis
LEGAL_TERMS = [
    'plaintiff', 'defendant', 'court', 'appeal', 'judgment', 'jurisdiction',
    'statute', 'appellant', 'respondent', 'petition', 'decree', 'conviction',
    'prosecution', 'section', 'act', 'charge', 'evidence', 'testimony',
    'witness', 'submission', 'verdict', 'acquittal', 'suit', 'claim',
    'damages', 'liability', 'compensation', 'injunction', 'provision', 'clause',
    'regulation', 'obligation', 'contract', 'agreement', 'amendment', 'ordinance',
    'rule', 'law', 'legal', 'judicial', 'affidavit', 'pleading', 'deposition',
    'summons', 'writ', 'notice', 'hearing', 'trial', 'proceedings', 'tribunal',
    'order', 'motion', 'assessment', 'tax', 'revenue'
]

def preprocess_text(text):
    """
    Preprocess the legal text for analysis.
    
    Args:
        text (str): Raw legal text
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits (but keep periods for sentence detection)
    text = re.sub(r'[^\w\s.]', ' ', text)
    
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Fix common legal abbreviations before sentence tokenization
    text = re.sub(r'(?<=\w)\.(?=\w)', '. ', text)
    text = re.sub(r'vs\.', 'vs', text)
    text = re.sub(r'etc\.', 'etc', text)
    
    return text

def tokenize_text(text):
    """
    Tokenize text into words and sentences.
    
    Args:
        text (str): Preprocessed text
        
    Returns:
        tuple: (words, sentences)
    """
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize into words
    words = word_tokenize(text)
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words 
             and word not in string.punctuation]
    
    return words, sentences

def extract_legal_terms(words):
    """
    Extract and count legal terms from the text.
    
    Args:
        words (list): List of tokenized words
        
    Returns:
        dict: Frequency of legal terms
    """
    legal_term_freq = {}
    
    for term in LEGAL_TERMS:
        count = words.count(term)
        if count > 0:
            legal_term_freq[term] = count
    
    # Sort by frequency (descending)
    legal_term_freq = dict(sorted(legal_term_freq.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True))
    
    return legal_term_freq

def calculate_readability(text):
    """
    Calculate readability score using Flesch-Kincaid Grade Level.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        float: Readability score
    """
    return flesch_kincaid_grade(text)

def extract_text_features(text):
    """
    Extract various features from the text for analysis.
    
    Args:
        text (str): Preprocessed text
        
    Returns:
        dict: Dictionary of text features
    """
    words, sentences = tokenize_text(text)
    
    # Calculate basic statistics
    word_count = len(words)
    sentence_count = len(sentences)
    char_count = len(text)
    
    # Calculate averages
    avg_word_length = sum(len(word) for word in words) / max(1, word_count)
    avg_sentence_length = word_count / max(1, sentence_count)
    
    # Get most common words
    freq_dist = FreqDist(words)
    most_common_words = dict(freq_dist.most_common(20))
    
    # Get legal terms frequency
    legal_terms_freq = extract_legal_terms(words)
    
    # Calculate readability
    readability_score = calculate_readability(text)
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "char_count": char_count,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,
        "most_common_words": most_common_words,
        "legal_terms_freq": legal_terms_freq,
        "readability_score": readability_score
    }