import asyncio
import edge_tts
import io

# Voice options: en-US-AriaNeural, en-US-GuyNeural, en-US-JennyNeural, etc.
VOICE = "en-US-AriaNeural"

async def _get_tts_audio_async(text, voice=VOICE):
    """
    Generates TTS audio using Edge-TTS asynchronously.
    Returns MP3 bytes.
    """
    communicate = edge_tts.Communicate(text, voice)
    mp3_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_data += chunk["data"]
    return mp3_data

def generate_tts(text, voice=VOICE):
    """
    Synchronous wrapper for generating TTS audio.
    Returns MP3 bytes.
    """
    try:
        return asyncio.run(_get_tts_audio_async(text, voice))
    except Exception as e:
        print(f"❌ Error in Edge-TTS generation: {e}")
        return None
