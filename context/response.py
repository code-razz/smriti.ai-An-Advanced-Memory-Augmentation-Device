# response.py

import os
import json
from datetime import datetime
import google.generativeai as genai
from prompt import RESPONDER_PROMPT
from search import get_query_and_memories
import time

# ✅ Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ Initialize Gemini model
# model = genai.GenerativeModel("gemini-2.5-flash-lite")
model = genai.GenerativeModel("gemini-2.5-flash")
# model = genai.GenerativeModel("gemini-2.0-flash")

def generate_answer(query_text=None):
    """
    Generates an answer for the given query text using Gemini and retrieved memories.
    If query_text is None, it prompts the user for input (via search.get_query_and_memories).
    
    Returns:
        str: The generated answer text.
    """
    # ✅ Fetch query + detailed memories
    question, memories_list = get_query_and_memories(query_text)
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
        return answer
    except Exception as e:
        print("❌ Error during Gemini execution:", e)
        return "Sorry, I couldn’t generate a response."

def generate_answer_stream(query_text=None):
    """
    Generates an answer for the given query text using Gemini and retrieved memories,
    yielding chunks of text as they are generated.
    
    Yields:
        str: Chunks of the generated answer text.
    """
    # ✅ Fetch query + detailed memories
    question, memories_list = get_query_and_memories(query_text)
    memories_json = json.dumps(memories_list, indent=2)

    # ✅ Build final prompt
    today_date = datetime.now().strftime("%Y-%m-%d")
    prompt_text = RESPONDER_PROMPT.format(
        today_date=today_date,
        memories=memories_json,
        question=question
    )

    # ✅ Generate response using Gemini (Streaming)
    try:
        print(f"🕒[PERF] Sent to LLM for answer at {time.strftime('%H:%M:%S')}")
        llm_start = time.time()
        response = model.generate_content(prompt_text, stream=True)
        print("\n--- AISmriti Answer (Streaming) ---")
        for chunk in response:
            if llm_start:
                 print(f"🕒[PERF] Received First LLM Token in {time.time() - llm_start:.4f}s")
                 llm_start = None # Only log once
            text_chunk = chunk.text
            print(text_chunk, end="", flush=True)
            yield text_chunk
    except Exception as e:
        print("❌ Error during Gemini execution:", e)
        yield "Sorry, I couldn’t generate a response."


if __name__ == "__main__":
    # Original behavior for testing
    import io
    import pygame
    from gtts import gTTS

    answer = generate_answer()

    # ✅ Convert answer to speech and play it (Local testing only)
    tts = gTTS(text=answer, lang="en")
    pygame.mixer.init()
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    pygame.mixer.music.load(fp, "mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue
