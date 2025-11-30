import os
import json
import logging
from typing import List, Dict
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not found in environment variables. Alert extraction will be disabled.")

def extract_alerts_from_text(text: str) -> List[Dict]:
    """
    Extracts alerts, reminders, and notifications from the given text using Gemini.
    
    Parameters:
        text (str): The conversation text to analyze.
        
    Returns:
        List[Dict]: A list of extracted alerts. Each alert is a dictionary.
    """
    if not GEMINI_API_KEY:
        logger.warning("Gemini API key missing. Skipping alert extraction.")
        return []

    if not text or not text.strip():
        return []

    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        
        # Get current time context
        now = datetime.now()
        current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        current_day = now.strftime("%A")
        
        prompt = f"""
        Current Date and Time: {current_time_str} ({current_day})
        
        Analyze the following conversation text and extract any alerts, reminders, or notifications that the user might want to set.
        Ignore general conversation. Focus on specific requests to remember something, remind someone, or set an alert.
        
        Conversation Text:
        {text}
        
        Return the result as a JSON list of objects. Each object should have the following fields:
        - "alert_text": The content of the alert/reminder.
        - "date": The specific date for the alert in "YYYY-MM-DD" format. Calculate based on "tomorrow", "next tuesday", etc. relative to current date. If no date is mentioned, use null.
        - "time": The specific time for the alert in "HH:MM" format (24-hour). If no time is mentioned, use null.
        - "type": "reminder", "alert", or "notification".
        - "confidence": A score from 0.0 to 1.0 indicating confidence that this is a genuine user request.
        
        If no alerts are found, return an empty list [].
        Output ONLY the JSON.
        """

        # print(prompt)
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        alerts = json.loads(response_text)
        
        # Add timestamp and validate
        valid_alerts = []
        for alert in alerts:
            if alert.get("confidence", 0) > 0.6: # Filter low confidence
                alert["created_at"] = datetime.now().isoformat()
                valid_alerts.append(alert)
                
        if valid_alerts:
            logger.info(f"🔔 Extracted {len(valid_alerts)} alerts from text.")
            
        return valid_alerts

    except Exception as e:
        logger.error(f"❌ Error extracting alerts with Gemini: {e}")
        return []
