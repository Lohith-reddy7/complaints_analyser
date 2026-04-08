# 📝 Kannada Language Complaint Analyzer

An intelligent, accessible, and automated Natural Language Processing (NLP) system designed to process, categorize, and prioritize public grievances submitted in the Kannada language. 

## ✨ Features

- **🗣️ Voice & Text Input:** Users can either type their complaints in Kannada or use the integrated microphone to simply speak their issues (Speech-to-Text).
- **🌍 Real-Time Translation:** Automatically translates incoming Kannada complaints into English using `deep-translator` to bridge the language gap for backend operators.
- **🧠 Native Kannada Processing:** Uses custom text preprocessing (including custom stopword removal and rule-based lemmatization) via the `indic-nlp-library` to intelligently handle Kannada's complex structure.
- **📊 Smart Categorization & Triage:** Employs an advanced Scikit-Learn TF-IDF engine (utilizing Character N-Grams to handle agglutination) to automatically assign the complaint to the correct operational department (e.g., Water, Roads, Garbage) and tag it with an Urgency Priority (High, Medium, Low).
- **🌐 Interactive Dashboard:** Built entirely on **Streamlit** to provide a clean, modern, and responsive user interface.

## 🛠️ Tech Stack & Requirements

- **Frontend:** Streamlit 
- **Backend Core:** Python 3.9+
- **Machine Learning / NLP:** Scikit-Learn, indic-nlp-library
- **Audio Processing:** SpeechRecognition
- **Translation:** deep-translator

## 🚀 Installation & Setup

1. **Clone or Download the Repository**
2. **Navigate to the Project Directory**
   ```bash
   cd kannada_complaints
   ```
3. **Install the Required Dependencies**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: You may need a working microphone for the audio recording feature.)*

4. **Launch the Application**
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
- `app.py`: The main Streamlit dashboard application file.
- `preprocess.py`: Contains the native Kannada NLP pipelines (tokenization, stopword removal).
- `tfidf_classifier.py`: The smart, Character N-Gram level Machine Learning categorizer.
- `audio_handler.py`: Handles the Speech-to-Text recording from the web UI.
- `stopwords_kn.txt`: Custom dictionary for removing Kannada filler words.

## 💡 How it Works
1. A user speaks or types a complaint: *"ನಮ್ಮ ಬೀದಿಯಲ್ಲಿ ಕಸದ ತೊಟ್ಟಿ ತುಂಬಿ ತುಳುಕುತ್ತಿದೆ"*
2. The system instantly translates it into English to inform the operator.
3. The original Kannada text is processed (noise filtered out).
4. The system mathematically matches the complaint's meaning, detects that it is about **Garbage**, assesses its frustration level, and labels it as a **High Priority**.
