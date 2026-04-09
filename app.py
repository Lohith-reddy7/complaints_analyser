import streamlit as st
from preprocess import process_complaint
from tfidf_classifier import analyze_complaint
from audio_handler import transcribe_audio

st.set_page_config(page_title="Kannada Complaint Analyzer", page_icon="📝", layout="wide")

st.title("📝 Kannada Language Complaint Analyzer")
st.markdown("Enter a municipal complaint in Kannada (via text or voice). The system will preprocess it (Tokenization, Stopword Removal, Lemmatization) and use **TF-IDF + Cosine Similarity** to categorize and prioritize it.")

st.divider()

with st.container(border=True):
    st.markdown("### 🗣️ File Your Complaint")
    st.markdown("You can **type** your complaint in the box below, or click the **Microphone Icon** (🎤) to speak it directly.")
    
    complaint_text_input = st.text_area("Type here:", 
                                   placeholder="ಉದಾಹರಣೆ: ನಮ್ಮ ಬೀದಿಯಲ್ಲಿ ಕಸದ ತೊಟ್ಟಿ ತುಂಬಿ ತುಳುಕುತ್ತಿದೆ...", 
                                   height=100,
                                   label_visibility="collapsed")
                                   
    audio_file = st.audio_input("🎤 Tap to record your voice:")
st.divider()

if st.button("🔍 Analyze Complaint", type="primary", use_container_width=True):
    final_text_to_analyze = None
    
    # Priority: If audio matches, transcribe first
    if audio_file is not None:
        with st.spinner("🎙️ Transcribing Kannada Audio to Text..."):
            transcription = transcribe_audio(audio_file)
            if transcription["success"]:
                final_text_to_analyze = transcription["text"]
                st.success(f"**Transcribed Text:** {final_text_to_analyze}")
            else:
                st.error(f"Failed to transcribe audio: {transcription['error']}")
                st.info("Try speaking clearly or typing your complaint instead.")
                
    elif complaint_text_input.strip() != "":
        final_text_to_analyze = complaint_text_input.strip()
    else:
        st.warning("⚠️ Please either record a voice message or type a complaint!")

    # If we have valid text (either from text box or successfully transcribed audio)
    if final_text_to_analyze:
        import re
        import requests
        import urllib.parse
        
        # If the input contains English letters (Kanglish transliteration or direct English)
        if re.search(r'[a-zA-Z]', final_text_to_analyze):
            with st.spinner("🔄 Transliterating Kanglish to accurate Kannada script..."):
                try:
                    # Use Google Input Tools API for flawless Kanglish transliteration
                    encoded_text = urllib.parse.quote(final_text_to_analyze)
                    url = f"https://inputtools.google.com/request?text={encoded_text}&itc=kn-t-i0-und&num=1"
                    resp = requests.get(url)
                    resp_json = resp.json()
                    
                    if resp_json[0] == 'SUCCESS':
                        converted_kannada = resp_json[1][0][1][0]
                        st.success(f"**Auto-Corrected to Kannada Script:** {converted_kannada}")
                        final_text_to_analyze = converted_kannada
                    else:
                        raise Exception("API failure")
                        
                except Exception as e:
                    try:
                        # Fallback to translation
                        from deep_translator import GoogleTranslator
                        converted_kannada = GoogleTranslator(source='auto', target='kn').translate(final_text_to_analyze)
                        st.success(f"**Translated to Kannada:** {converted_kannada}")
                        final_text_to_analyze = converted_kannada
                    except:
                        st.warning("⚠️ Failed to auto-convert English text to Kannada.")
        with st.spinner("🌍 Translating to English..."):
            try:
                from deep_translator import GoogleTranslator
                english_translation = GoogleTranslator(source='kn', target='en').translate(final_text_to_analyze)
                st.info(f"**Translated (English):** {english_translation}")
            except Exception as e:
                st.error("⚠️ Failed to translate the text.")
                
        with st.spinner("🛠️ Processing NLP steps..."):
            # Step 1: Preprocessing
            nlp_results = process_complaint(final_text_to_analyze)
            
        with st.expander("🧐 View NLP Preprocessing Breakdown", expanded=True):
            scol1, scol2, scol3 = st.columns(3)
            with scol1:
                st.markdown("**1. Tokens (ಟೋಕನ್‌ಗಳು)**")
                st.info(", ".join(nlp_results['tokens']) if nlp_results['tokens'] else "None")
            with scol2:
                st.markdown("**2. Without Stopwords (ಸ್ಟಾಪ್ ವರ್ಡ್ಸ್ ಇಲ್ಲದೆ)**")
                st.success(", ".join(nlp_results['without_stopwords']) if nlp_results['without_stopwords'] else "None")
            with scol3:
                st.markdown("**3. Lemmatized (ಮೂಲ ಪದಗಳು)**")
                st.warning(", ".join(nlp_results['lemmatized']) if nlp_results['lemmatized'] else "None")
                
        with st.spinner("📈 Calculating TF-IDF similarities..."):
            # Step 2: TF-IDF Classification
            report = analyze_complaint(nlp_results['original'], nlp_results['lemmatized'])
            
        st.subheader("📊 Final Analysis Report")
        
        category = report.get("category", "Unknown")
        priority = report.get("priority", "Unknown")
        reason = report.get("reason", "No reason provided.")
        
        final_c1, final_c2 = st.columns(2)
        
        with final_c1:
            st.metric(label="📌 Category", value=category)
        
        with final_c2:
            # Color code priority
            color = "green"
            if priority.lower() == "high":
                color = "red"
            elif priority.lower() == "medium":
                color = "orange"
                
            st.markdown(f"### 🚨 Priority: <span style='color:{color}'>{priority}</span>", unsafe_allow_html=True)
            
        st.markdown(f"**📝 Similarity Info:** {reason}")
