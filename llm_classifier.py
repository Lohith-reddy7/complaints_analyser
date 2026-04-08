import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def analyze_complaint(original_text, lemmatized_tokens):
    """
    Calls the OpenAI API to classify the complaint and assign a priority.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # We join lemmatized tokens to show the LLM the preprocessed form, 
    # but we also provide original text to maintain full semantic context.
    preprocessed_text = " ".join(lemmatized_tokens)
    
    system_prompt = """
You are a Kannada Language Complaint Analyzer for municipal and public services.
Your job is to read a complaint written in Kannada (and its preprocessed lemmatized tokens) and output a JSON response containing:
1. "category": The department/category the complaint belongs to. Choose one from: [Drainage, Garbage, Roads, Electricity, Water, Streetlights, Public Health, Other].
2. "priority": The urgency of the complaint. Choose one from: [High, Medium, Low].
3. "reason": A short reason (1-2 sentences) in English explaining why this priority and category were assigned.

Only return valid JSON without markdown wrapping.
    """
    
    user_prompt = f"""
Original Complaint (Kannada): {original_text}
Preprocessed Tokens: {preprocessed_text}

Please analyze and return the JSON.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # or gpt-3.5-turbo if 4o-mini is unavailable, but gpt-4o-mini is optimal
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ],
            response_format={ "type": "json_object" },
            temperature=0.2
        )
        
        result_content = response.choices[0].message.content
        return json.loads(result_content)
        
    except Exception as e:
        return {
            "error": str(e),
            "category": "Unknown",
            "priority": "Unknown",
            "reason": f"An error occurred: {str(e)}"
        }
