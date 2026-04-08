import os
import re

# We will try to import indic_nlp, but provide a fallback if it fails or requires extra setup
try:
    from indicnlp.tokenize import indic_tokenize
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    INDIC_NLP_AVAILABLE = True
except ImportError:
    INDIC_NLP_AVAILABLE = False


def load_stopwords(filepath="stopwords_kn.txt"):
    """Loads stopwords from a text file into a set."""
    stopwords = set()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
    return stopwords

def normalize_text(text):
    """Normalize Kannada text to remove zero-width joiners and fix typical input issues."""
    if INDIC_NLP_AVAILABLE:
        factory = IndicNormalizerFactory()
        normalizer = factory.get_normalizer("kn")
        return normalizer.normalize(text)
    return text

def tokenize(text):
    """Tokenize the text into words."""
    if INDIC_NLP_AVAILABLE:
        return indic_tokenize.trivial_tokenize(text)
    # Basic unicode word boundary tokenization fallback
    return re.findall(r"[\w']+", text)

def remove_stopwords(tokens, stop_words_set):
    """Filter out stopwords from a list of tokens."""
    return [t for t in tokens if t not in stop_words_set]

def rule_based_stemmer(tokens):
    """
    A basic rule-based suffix-stripping approach for Kannada.
    This simulates lemmatization for highly agglutinative languages.
    """
    stemmed_tokens = []
    # Common suffixes in Kannada (not exhaustive, for demonstration purposes)
    suffixes = [
        'ನ್ನು', 'ಇಂದ', 'ಗೆ', 'ಅಲ್ಲಿ', 'ಗಳ', 'ಗಳು', 'ಅವರ', 
        'ಆಗಿ', 'ದಲ್ಲಿ', 'ಗಾಗಿ', 'ಕೋಸ್ಕರ', 'ಇಗೆ', 'ಗೆ'
    ]
    
    for token in tokens:
        original = token
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                token = token[:-len(suffix)]
                break  # Only strip the longest matching end once for simplicity
        stemmed_tokens.append(token)
        
    return stemmed_tokens

def process_complaint(text, stopwords_filepath="stopwords_kn.txt"):
    """Main pipeline for preprocessing a Kannada complaint."""
    stopwords = load_stopwords(stopwords_filepath)
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    
    # Optional: basic cleaning (removing punctuation tokens)
    clean_tokens = [t for t in tokens if not re.match(r'^[^\w\s]+$', t)]
    
    without_stops = remove_stopwords(clean_tokens, stopwords)
    lemmatized = rule_based_stemmer(without_stops)
    
    return {
        "original": text,
        "normalized": normalized,
        "tokens": clean_tokens,
        "without_stopwords": without_stops,
        "lemmatized": lemmatized
    }

if __name__ == "__main__":
    # Test
    sample = "ನಮ್ಮ ಬೀದಿಯಲ್ಲಿ ಕಸದ ತೊಟ್ಟಿ ತುಂಬಿ ತುಳುಕುತ್ತಿದೆ, ದಯವಿಟ್ಟು ಬೇಗ ಸ್ವಚ್ಛಗೊಳಿಸಿ."
    res = process_complaint(sample)
    print("Tokens:", res['tokens'])
    print("No Stops:", res['without_stopwords'])
    print("Lemmatized:", res['lemmatized'])
