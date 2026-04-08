from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define reference texts for specific categories based on vocabulary
CATEGORY_CORPUS = {
    "Drainage (ಚರಂಡಿ)": "ಚರಂಡಿ ಒಳಚರಂಡಿ ಬ್ಲಾಕ್ ನೀರು ನಿಂತಿದೆ ಡ್ರೈನೇಜ್ ತಡೆಗಟ್ಟಿ ಸ್ವಚ್ಛ ಕೊಳಚೆ",
    "Garbage (ಕಸ)": "ಕಸ ತ್ಯಾಜ್ಯ ತೊಟ್ಟಿ ದುರ್ವಾಸನೆ ಗಬ್ಬು ಸ್ವಚ್ಛಗೊಳಿಸಿ ಪ್ಲಾಸ್ಟಿಕ್ ವಿಲೇವಾರಿ",
    "Roads (ರಸ್ತೆ)": "ರಸ್ತೆ ಗುಂಡಿ ಡಾಂಬರು ರಿಪೇರಿ ಹಳ್ಳ ಕಾಂಕ್ರೀಟ್ ಮಣ್ಣು ಕಾಮಗಾರಿ",
    "Electricity (ವಿದ್ಯುತ್)": "ವಿದ್ಯುತ್ ಕರೆಂಟ್ ಕಂಬ ದೀಪ ಲೈನ್ಮ್ಯಾನ್ ಪವರ್ ತಂತಿ ಕಡಿತ",
    "Water (ನೀರು)": "ನೀರು ಸರಬರಾಜು ಕೊಳವೆ ಬತ್ತಿ ಪೈಪ್ ಕುಡಿಯುವ ನಳ ಬರುವ"
}

# Define reference texts for priority
PRIORITY_CORPUS = {
    "High": "ತುರ್ತು ಕೂಡಲೇ ಅಪಾಯಕಾರಿ ತಕ್ಷಣ ಬೇಗ ಪ್ರಾಣಾಪಾಯ ಜರೂರು ಆಪತ್ತು ಗಂಭೀರ",
    "Medium": "ತೊಂದರೆ ಸಮಸ್ಯೆ ದಿನಗಟ್ಟಲೆ ಅನಾನುಕೂಲ ಕಷ್ಟ ತಡವಾಗಿ ತೊಂದರೆಯಾಗುತ್ತಿದೆ",
    "Low": "ಮಾಹಿತಿ ಮನವಿ ಅವಕಾಶ ಕೇಳಿಕೆ ಭವಿಷ್ಯದಲ್ಲಿ ವಿಚಾರಣೆ ದಯವಿಟ್ಟು"
}

def analyze_complaint(original_text, preprocessed_tokens):
    """
    Uses TF-IDF + Cosine Similarity to assign Category AND Priority.
    """
    text = " ".join(preprocessed_tokens)
    
    def get_best_match(corpus, input_text, default_key):
        keys = list(corpus.keys())
        docs = list(corpus.values())
        docs.append(input_text)
        
        # Using char_wb analyzer to handle Kannada agglutination without needing perfect lemmatizers
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6))
        try:
            tfidf_matrix = vectorizer.fit_transform(docs)
            input_vector = tfidf_matrix[-1]
            corpus_vectors = tfidf_matrix[:-1]
            
            similarities = cosine_similarity(input_vector, corpus_vectors)[0]
            best_idx = similarities.argmax()
            
            # If there's strictly NO overlap with any vector, it stays 0.0
            if similarities[best_idx] == 0.0:
                return default_key, 0.0
                
            return keys[best_idx], similarities[best_idx]
        except Exception:
            return default_key, 0.0

    best_category, cat_score = get_best_match(CATEGORY_CORPUS, text, "Other")
    
    # Analyze Priority using TF-IDF and Cosine Similarity, defaulting to Medium if 0.0 match
    best_priority, pri_score = get_best_match(PRIORITY_CORPUS, text, "Medium")
    
    if pri_score == 0.0:
        pri_reason = "No priority keyword overlap found. Defaulting to Medium."
    else:
        pri_reason = f"Priority matched with TF-IDF similarity score: {pri_score:.2f}."
        
    reason = f"Category match score: {cat_score:.2f}. {pri_reason}"
    
    return {
        "category": best_category,
        "priority": best_priority,
        "reason": reason
    }
