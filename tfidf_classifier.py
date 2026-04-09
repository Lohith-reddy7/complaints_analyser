import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Massive Expansion for Categories
CATEGORY_CORPUS = {
    "Drainage (ಚರಂಡಿ)": "ಚರಂಡಿ ಡ್ರೈನೇಜ್ ಯುಜಿಡಿ ಮ್ಯಾನ್‌ಹೋಲ್ ಬ್ಲಾಕ್ ಕೊಳಚೆ ನಿಂತಿದೆ ನೀರು ಲೀಕ್ ujd drainage manhole",
    "Garbage (ಕಸ)": "ಕಸ ತ್ಯಾಜ್ಯ ತೊಟ್ಟಿ ಗಬ್ಬು ದುರ್ವಾಸನೆ ವಿಲೇವಾರಿ ಕಸದ ಲಾರಿ ಬಿನ್ ಪ್ಲಾಸ್ಟಿಕ್ ಡಸ್ಟ್ ಬಿನ್ garbage waste bin",
    "Roads (ರಸ್ತೆ)": "ರಸ್ತೆ ಗುಂಡಿ ಡಾಂಬರು ಹಳ್ಳ ಮಣ್ಣು ಕಾಂಕ್ರೀಟ್ ಕಾಮಗಾರಿ ಫುಟ್‌ಪಾತ್ ಟ್ರಾಫಿಕ್ ಅಪಘಾತ ರಸ್ತೆಗಳು road potholes",
    "Electricity (ವಿದ್ಯುತ್)": "ವಿದ್ಯುತ್ ಕರೆಂಟ್ ಲೈನ್ಮ್ಯಾನ್ ಕಂಬ ಕೆಬಿ ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್ ವೈರ್ ಶಾರ್ಟ್ ಸರ್ಕ್ಯೂಟ್ ಬೆಂಕಿ ಸಿಡಿದು ಪವರ್ electricity power",
    "Water (ನೀರು)": "ನೀರು ವಾಟರ್ ನಲ್ಲಿ ಪೈಪ್ ಕುಡಿಯುವ ಬತ್ತಿ ಸರಬರಾಜು ಬೋರ್ ವೆಲ್ ಕಾವೇರಿ ಟ್ಯಾಂಕರ್ water drinking"
}

# Extensive Priority Rules
HIGH_WORDS = ["ತುರ್ತು", "ಆಪತ್ತು", "ಅಪಾಯ", "ಅಪಾಯಕಾರಿ", "ಬೆಂಕಿ", "ಸಿಡಿದು", "ಪ್ರಾಣಾಪಾಯ", "ತಕ್ಷಣ", "ಕೂಡಲೇ", "ಜರೂರು", "ಎಮರ್ಜೆನ್ಸಿ", "ಸಾವು", "ಹಾನಿ"]
LOW_WORDS = ["ಮಾಹಿತಿ", "ದಯವಿಟ್ಟು", "ಮನವಿ", "ಕೇಳಿಕೆ", "ಭವಿಷ್ಯದಲ್ಲಿ", "ವಿಚಾರಣೆ"]

def analyze_complaint(original_text, preprocessed_tokens):
    text = " ".join(preprocessed_tokens)
    
    # 1. Advanced Category Classification Using Core TF-IDF
    def get_best_category(corpus, input_text):
        keys = list(corpus.keys())
        docs = list(corpus.values())
        docs.append(input_text)
        
        # char_wb helps with Kannada prefixes/suffixes immensely. 
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
        try:
            tfidf_matrix = vectorizer.fit_transform(docs)
            similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
            best_idx = similarities.argmax()
            score = similarities[best_idx]
            
            if score < 0.05:
                return "Other (ಇತರೆ)", score
            return keys[best_idx], score
        except:
            return "Other (ಇತರೆ)", 0.0

    best_category, cat_score = get_best_category(CATEGORY_CORPUS, text)
    
    # Check for irrelevant requests
    if cat_score < 0.05:
        return {
            "category": "Other (ಇತರೆ)",
            "priority": "Low",
            "reason": "Very low confidence score. This may not be a municipal complaint."
        }

    # 2. Rule-Based + TF-IDF Hybrid Prioritization
    # Rule overriding for extreme accuracy
    joined_text = " ".join(preprocessed_tokens) + " " + original_text
    
    best_priority = "Medium"
    reason_pri = "Default assigned based on regular municipal request logic."
    
    # Explicit urgent checks
    if any(urgent_word in joined_text for urgent_word in HIGH_WORDS):
        best_priority = "High"
        reason_pri = "Critical danger or extreme urgency keywords detected in the complaint!"
    elif any(low_word in joined_text for low_word in LOW_WORDS):
        best_priority = "Low"
        reason_pri = "General inquiry or information request detected."
    else:
        # Fallback to TF-IDF logic for priority assessing regular distress levels
        PRIORITY_CORPUS = {
            "High": "ತುರ್ತು ಕೂಡಲೇ ಅಪಾಯಕಾರಿ ತಕ್ಷಣ ಬೇಗ ಪ್ರಾಣಾಪಾಯ ಜರೂರು ಆಪತ್ತು ಗಂಭೀರ ಬೆಂಕಿ ಸಿಡಿದು",
            "Medium": "ತೊಂದರೆ ಸಮಸ್ಯೆ ದಿನಗಟ್ಟಲೆ ಅನಾನುಕೂಲ ಕಷ್ಟ ತಡವಾಗಿ ತೊಂದರೆಯಾಗುತ್ತಿದೆ ದುರಸ್ತಿ",
            "Low": "ಮಾಹಿತಿ ಮನವಿ ಅವಕಾಶ ಕೇಳಿಕೆ ಭವಿಷ್ಯದಲ್ಲಿ ವಿಚಾರಣೆ ದಯವಿಟ್ಟು"
        }
        test_docs = list(PRIORITY_CORPUS.values()) + [text]
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
        try:
            matrix = vec.fit_transform(test_docs)
            sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
            if max(sims) > 0.05:
                idx = sims.argmax()
                best_priority = list(PRIORITY_CORPUS.keys())[idx]
                reason_pri = f"Mathematically aligned with '{best_priority}' urgency level based on text semantics."
        except:
            pass
            
    return {
        "category": best_category,
        "priority": best_priority,
        "reason": f"Category match confidence: {cat_score*100:.1f}%. {reason_pri}"
    }
