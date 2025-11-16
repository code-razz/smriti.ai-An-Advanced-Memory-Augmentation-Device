# response.py

import os
import io
import json
import pygame
from datetime import datetime
import google.generativeai as genai
from gtts import gTTS
from prompt import RESPONDER_PROMPT
from search import get_query_and_memories

# ✅ Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ Initialize Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")

# ✅ Fetch query + detailed memories
question, memories_list = get_query_and_memories()
memories_json = json.dumps(memories_list, indent=2)  # Pretty JSON for clarity

# ✅ Build final prompt
today_date = datetime.now().strftime("%Y-%m-%d")
prompt_text = RESPONDER_PROMPT.format(
    today_date=today_date,
    memories=memories_json,
    question=question
)

# ✅ Generate response using Gemini
try:
    response = model.generate_content(prompt_text)
    answer = response.text
    print("\n--- AISmriti Answer ---")
    print(answer)
except Exception as e:
    print("❌ Error during Gemini execution:", e)
    answer = "Sorry, I couldn’t generate a response."

# ✅ Convert answer to speech and play it
tts = gTTS(text=answer, lang="en")
pygame.mixer.init()
fp = io.BytesIO()
tts.write_to_fp(fp)
fp.seek(0)
pygame.mixer.music.load(fp, "mp3")
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    continue
