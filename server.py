# server.py
import warnings
# Suppress torchaudio deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*")
warnings.filterwarnings("ignore", message=".*TorchAudio.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*")
warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom.*")
warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")
# Suppress SpeechBrain deprecation warnings
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*deprecated.*")
# Suppress pyannote warnings about symlinks on Windows
warnings.filterwarnings("ignore", message=".*Pretrainer collection using symlinks on Windows.*")

# from flask import Flask, request
from flask import Flask, request, jsonify, current_app
from flask_socketio import SocketIO, emit
import wave
import threading
import time
import sys
import os
from pathlib import Path

import logging
from typing import Optional, Tuple

from PIL import Image

# Add paths for importing processing modules
project_root = Path(__file__).parent
stt_diarization_path = project_root / "stt_diarization"
context_path = project_root / "context"

sys.path.insert(0, str(stt_diarization_path))
sys.path.insert(0, str(context_path))

import numpy as np
import io
# from gtts import gTTS  # Replaced by edge-tts
from tts_engine import generate_tts
from pydub import AudioSegment

# Import processing functions
try:
    from process_audio import process_audio_file
    from process_audio_streaming import process_audio_segment
    from process_chunks import process_and_store_conversation
    from response import generate_answer, generate_answer_stream
    from transcriber import transcribe_audio
    from core_processing import get_models
    PROCESSING_AVAILABLE = True
    print("✅ Processing modules loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import processing modules: {e}")
    print("⚠️ Context audio will be saved but not processed automatically.")
    PROCESSING_AVAILABLE = False

# Streaming processing configuration
SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # 16-bit = 2 bytes per sample, mono
SEGMENT_DURATION_SECONDS = 50.0  # Process every 50 seconds of audio
SEGMENT_DURATION_BYTES = int(SEGMENT_DURATION_SECONDS * BYTES_PER_SECOND)
OVERLAP_SECONDS = 2.0  # 2 second overlap between segments for continuity
OVERLAP_BYTES = int(OVERLAP_SECONDS * BYTES_PER_SECOND)

# -----------------------------
# Load reply WAV (16kHz, mono, 16-bit)
# -----------------------------
with open("reply_16k.wav", "rb") as f:
    SAMPLE_REPLY = f.read()

# -----------------------------
# Flask + SocketIO setup
# -----------------------------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Per-client buffers/state
partial_audio = {}               # sid -> bytearray for query recording
partial_context_audio = {}       # sid -> bytearray for context recording
is_recording = {}                # sid -> bool: query currently active
is_context_recording = {}        # sid -> bool: context currently active
reply_stream_stop_flags = {}     # sid -> bool flag to tell background streamer to stop
reply_streaming_task = {}        # sid -> task handle (if needed)

# Streaming processing state
context_processed_bytes = {}     # sid -> int: bytes already processed
context_processing_lock = {}     # sid -> threading.Lock: prevent concurrent processing
context_accumulated_transcript = {}  # sid -> str: accumulated transcript from all segments

# Chunk counters for clean logging
query_chunk_count = {}           # sid -> int: number of query chunks received
# Chunk counters for clean logging
query_chunk_count = {}           # sid -> int: number of query chunks received
context_chunk_count = {}         # sid -> int: number of context chunks received
finalizing_processing = {}       # sid -> bool: flag to indicate completion handler is running

CHUNK_SIZE = 4096
STREAM_DELAY = 0.04  # seconds between sending chunks (tune for latency vs jitter)


@socketio.on("connect")
def on_connect():
    sid = request.sid
    print(f"✅ Client connected: {sid}")
    partial_audio[sid] = bytearray()
    partial_context_audio[sid] = bytearray()
    is_recording[sid] = False
    is_context_recording[sid] = False
    reply_stream_stop_flags[sid] = False
    context_processed_bytes[sid] = 0
    context_processing_lock[sid] = threading.Lock()
    context_accumulated_transcript[sid] = ""
    query_chunk_count[sid] = 0
    context_chunk_count[sid] = 0


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    # Clean up client state
    if finalizing_processing.get(sid):
        print(f"⏳ Deferring cleanup for {sid} to handle_context_audio_complete")
        return

    # If processing is active, we try to acquire lock to ensure we don't delete state 
    # while handle_context_audio_complete is using it.
    lock = context_processing_lock.get(sid)
    if lock:
        # Try to acquire lock (blocking) to wait for any active processing to finish
        with lock:
            _perform_cleanup(sid)
    else:
        _perform_cleanup(sid)


def _perform_cleanup(sid):
    """Helper to clean up client state."""
    print(f"🧹 Cleaning up state for {sid}")
    partial_audio.pop(sid, None)
    partial_context_audio.pop(sid, None)
    is_recording.pop(sid, None)
    is_context_recording.pop(sid, None)
    reply_stream_stop_flags.pop(sid, None)
    reply_streaming_task.pop(sid, None)
    context_processed_bytes.pop(sid, None)
    context_processing_lock.pop(sid, None)
    context_accumulated_transcript.pop(sid, None)
    query_chunk_count.pop(sid, None)
    context_chunk_count.pop(sid, None)
    finalizing_processing.pop(sid, None)


# -----------------------------
# Query (existing) handlers
# -----------------------------
@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    """
    Receive small PCM chunks from client while query-recording (press-hold).
    Behavior: if we receive 'audio_chunk' and is_recording[sid] is False,
    treat this as the *start* of a new query recording: clear the previous buffer and set is_recording True.
    """
    sid = request.sid
    if sid not in partial_audio:
        partial_audio[sid] = bytearray()
    if sid not in is_recording:
        is_recording[sid] = False

    if not is_recording[sid]:
        partial_audio[sid] = bytearray()   # clear previous query recording
        is_recording[sid] = True
        query_chunk_count[sid] = 0
        print(f"\n🆕 [QueryStream] Starting new recording for {sid}")

    partial_audio[sid].extend(data)
    query_chunk_count[sid] = query_chunk_count.get(sid, 0) + 1
    total_bytes = len(partial_audio[sid])
    total_mb = total_bytes / (1024 * 1024)
    print(f"\r[QueryStream] {total_mb:.2f} MB received | {query_chunk_count[sid]} chunks processed", end="", flush=True)


@socketio.on("audio_complete")
def handle_audio_complete():
    """
    Client indicated query recording finished. Save WAV and start streaming reply back in a background task.
    """
    sid = request.sid
    data = partial_audio.get(sid, None)
    if not data:
        print(f"⚠️ audio_complete received but no query data for {sid}")
    else:
        filename = f"received_{sid}.wav"
        try:
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16
                wf.setframerate(16000)
                wf.writeframes(bytes(data))
            total_mb = len(data) / (1024 * 1024)
            print(f"\n💾 [QueryStream] Saved '{filename}' | {total_mb:.2f} MB | {query_chunk_count.get(sid, 0)} chunks")
        except Exception as e:
            print(f"❌ Failed to write QUERY WAV for {sid}: {e}")

    # Mark recording finished so next audio_chunk will clear buffer
    is_recording[sid] = False

    # Reset stop flag (ensure new stream allowed)
    reply_stream_stop_flags[sid] = False

    # Start streaming reply back (background task)
    print(f"🔁 Starting reply stream to {sid} (background task)")
    if data:
        reply_streaming_task[sid] = socketio.start_background_task(process_and_stream_reply, sid, data)
    else:
        print(f"⚠️ No data for {sid}, cannot process query.")


def process_and_stream_reply(sid, audio_data_bytes):
    """
    Process the query audio: STT (In-Memory) -> LLM (Stream) -> TTS (Stream) -> Stream.
    """
    print(f"🧠 Processing query for {sid} (In-Memory)")
    try:
        if PROCESSING_AVAILABLE:
            # 1. STT (In-Memory)
            _, _, whisper_model, _ = get_models()
            
            print("📝 Transcribing query (In-Memory)...")
            # Convert PCM bytes to float32 numpy array for Whisper
            # Assumes 16-bit PCM, 16kHz
            audio_np = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            result = transcribe_audio(whisper_model, audio_np)
            query_text = result["text"].strip()
            print(f"🗣️ Query: {query_text}")

            if not query_text:
                print("⚠️ Empty transcription, sending default reply.")
                stream_reply_to_client(sid)
                return

            # 2. Generate Answer & Stream TTS
            print("🤔 Generating answer (Streaming)...")
            
            text_buffer = ""
            # Added comma and other punctuation for faster chunking
            sentence_endings = ['.', '?', '!', '.\n', '?\n', '!\n', ',', ':', ';']
            
            start_time = time.time()
            first_token_time = None
            is_first_chunk = True
            
            for chunk_text in generate_answer_stream(query_text):
                if first_token_time is None:
                    first_token_time = time.time()
                    print(f"⏱️ Time to first LLM token: {first_token_time - start_time:.2f}s")
                
                text_buffer += chunk_text
                
                # Aggressive split for the very first chunk to reduce latency
                if is_first_chunk:
                    words = text_buffer.split()
                    if len(words) >= 5:  # Wait for at least 5 words
                        # Find the position of the 5th space to cut strictly there
                        # This ensures we don't process a huge chunk if the LLM yields a lot at once
                        split_index = -1
                        space_count = 0
                        for i, char in enumerate(text_buffer):
                            if char == ' ':
                                space_count += 1
                                if space_count == 5:
                                    split_index = i
                                    break
                        
                        if split_index != -1:
                            sentence_to_speak = text_buffer[:split_index].strip()
                            remaining_text = text_buffer[split_index:]
                            
                            print(f"\n🗣️ TTS First Phrase (Strict 5 words): {sentence_to_speak}")
                            tts_start = time.time()
                            _generate_and_stream_tts(sid, sentence_to_speak)
                            print(f"⏱️ TTS processing time (First): {time.time() - tts_start:.2f}s")
                            
                            text_buffer = remaining_text
                            is_first_chunk = False
                    continue

                # Normal splitting for subsequent chunks
                last_punct_idx = -1
                for punct in sentence_endings:
                    idx = text_buffer.rfind(punct)
                    if idx != -1 and idx > last_punct_idx:
                        last_punct_idx = idx + len(punct)
                
                if last_punct_idx != -1:
                    sentence_to_speak = text_buffer[:last_punct_idx].strip()
                    remaining_text = text_buffer[last_punct_idx:]
                    
                    if sentence_to_speak:
                        # Don't split if it's too short unless it's a real sentence end
                        if len(sentence_to_speak) < 5 and sentence_to_speak[-1] not in ['.', '?', '!']:
                            continue
                            
                        print(f"\n🗣️ TTS Phrase: {sentence_to_speak}")
                        tts_start = time.time()
                        _generate_and_stream_tts(sid, sentence_to_speak)
                        print(f"⏱️ TTS processing time: {time.time() - tts_start:.2f}s")
                    
                    text_buffer = remaining_text
            
            # Process any remaining text
            if text_buffer.strip():
                print(f"\n🗣️ TTS Final Chunk: {text_buffer}")
                _generate_and_stream_tts(sid, text_buffer)
                
            print(f"✅ Finished streaming reply to {sid}. Total time: {time.time() - start_time:.2f}s")

        else:
            print("⚠️ Processing modules not available, sending default reply.")
            stream_reply_to_client(sid)

    except Exception as e:
        print(f"❌ Error in process_and_stream_reply: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to default reply on error
        stream_reply_to_client(sid)


def _generate_and_stream_tts(sid, text):
    """Helper to generate TTS for a text chunk and stream it."""
    try:
        if not text.strip():
            return
            
        # TTS (Edge-TTS)
        mp3_bytes = generate_tts(text)
        if not mp3_bytes:
            print("⚠️ TTS generation failed.")
            return
            
        mp3_fp = io.BytesIO(mp3_bytes)
        mp3_fp.seek(0)

        # Convert to WAV 16kHz Mono
        audio = AudioSegment.from_file(mp3_fp, format="mp3")
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        wav_fp = io.BytesIO()
        audio.export(wav_fp, format="wav")
        wav_bytes = wav_fp.getvalue()

        # Stream chunks
        total = len(wav_bytes)
        sent = 0
        while sent < total:
            if reply_stream_stop_flags.get(sid, False):
                print(f"⏹️ Streaming stopped by client request for {sid}")
                break
            end = min(sent + CHUNK_SIZE, total)
            chunk = wav_bytes[sent:end]
            try:
                socketio.emit("server_audio_chunk", chunk, to=sid)
            except Exception as e:
                print(f"❌ Emit error: {e}")
                break
            sent = end
            time.sleep(STREAM_DELAY)
            
    except Exception as e:
        print(f"❌ Error in TTS generation: {e}")



def stream_reply_to_client(sid, audio_data=None):
    """
    Stream audio_data bytes in CHUNK_SIZE pieces to client.
    If audio_data is None, uses default SAMPLE_REPLY.
    Honors reply_stream_stop_flags[sid] to stop early when client requests.
    Emits 'server_audio_chunk' events and a final 'server_audio_complete'.
    """
    print(f"\n▶️ stream_reply_to_client started for {sid}")
    
    data_to_send = audio_data if audio_data else SAMPLE_REPLY
    total = len(data_to_send)
    sent = 0
    while sent < total:
        # stop check
        if reply_stream_stop_flags.get(sid, False):
            print(f"⏹️ Server-side streaming stopped by client request for {sid}")
            break
        end = min(sent + CHUNK_SIZE, total)
        chunk = data_to_send[sent:end]
        try:
            socketio.emit("server_audio_chunk", chunk, to=sid)
        except Exception as e:
            print(f"❌ Emit error while streaming to {sid}: {e}")
            break
        sent = end
        time.sleep(STREAM_DELAY)
    # signal end (unless we were stopped intentionally - still useful)
    try:
        socketio.emit("server_audio_complete", to=sid)
    except Exception:
        pass
    print(f"\n✅ Finished (or stopped) streaming reply to {sid} (sent={sent}/{total})")


@socketio.on("stop_server_stream")
def on_stop_server_stream():
    """
    Client asked server to stop streaming (client pressed 'a' to start new query recording).
    """
    sid = request.sid
    reply_stream_stop_flags[sid] = True
    print(f"\n📴 Received stop_server_stream from {sid}")


# -----------------------------
# Context handlers (new)
# -----------------------------
@socketio.on("context_start")
def on_context_start():
    """
    Client indicates the user started a NEW context-recording session (press push-button2 to start).
    Clear any previous context buffer and mark as recording (server will accept context_audio_chunk).
    """
    sid = request.sid
    print(f"\n🟣 [AudioStream] Starting NEW context session for {sid} (clearing previous buffer)")
    partial_context_audio[sid] = bytearray()
    is_context_recording[sid] = True
    context_processed_bytes[sid] = 0
    context_accumulated_transcript[sid] = ""
    context_chunk_count[sid] = 0


@socketio.on("context_resume")
def on_context_resume():
    """
    Client indicates context recording is resuming (after being paused by a query or previously paused).
    Do NOT clear the buffer — continue appending.
    """
    sid = request.sid
    is_context_recording[sid] = True
    print(f"\n🟣 [AudioStream] Resuming context recording for {sid} (no buffer clear)")


@socketio.on("context_pause")
def on_context_pause():
    """
    Client indicates context recording is paused (e.g., because user started a query-recording).
    """
    sid = request.sid
    is_context_recording[sid] = False
    print(f"\n🟣 [AudioStream] Pausing context recording for {sid}")


@socketio.on("context_audio_chunk")
def handle_context_audio_chunk(data):
    """
    Receive PCM chunks from client while context-recording is active.
    Process segments in real-time as audio accumulates.
    """
    sid = request.sid
    if sid not in partial_context_audio:
        partial_context_audio[sid] = bytearray()
    if not is_context_recording.get(sid, False):
        print(f"\n⚠️ [AudioStream] Received chunk for {sid} but context is paused; ignoring.")
        return

    partial_context_audio[sid].extend(data)
    total_bytes = len(partial_context_audio[sid])
    processed_bytes = context_processed_bytes.get(sid, 0)
    unprocessed_bytes = total_bytes - processed_bytes
    
    # Check if we have enough unprocessed audio to process a segment
    if unprocessed_bytes >= SEGMENT_DURATION_BYTES and PROCESSING_AVAILABLE:
        # Process the segment in background (non-blocking)
        socketio.start_background_task(process_context_segment_streaming, sid)
    
    context_chunk_count[sid] = context_chunk_count.get(sid, 0) + 1
    total_mb = total_bytes / (1024 * 1024)
    processed_mb = processed_bytes / (1024 * 1024)
    print(f"\r[AudioStream] {total_mb:.2f} MB received | {context_chunk_count[sid]} chunks | {processed_mb:.2f} MB processed", end="", flush=True)


def process_context_segment_streaming(sid):
    """
    Process a segment of context audio in real-time (streaming mode).
    Processes accumulated audio segments as they reach the threshold duration.
    """
    if sid not in context_processing_lock:
        return
    
    # Use lock to prevent concurrent processing of same client
    lock = context_processing_lock[sid]
    if not lock.acquire(blocking=False):
        # Another processing task is already running for this client
        return
    
    try:
        audio_buffer = partial_context_audio.get(sid)
        if not audio_buffer:
            return
        
        processed_bytes = context_processed_bytes.get(sid, 0)
        total_bytes = len(audio_buffer)
        unprocessed_bytes = total_bytes - processed_bytes
        
        if unprocessed_bytes < SEGMENT_DURATION_BYTES:
            return
        
        # Extract segment to process (with overlap for continuity)
        segment_start = max(0, processed_bytes - OVERLAP_BYTES)
        segment_end = processed_bytes + SEGMENT_DURATION_BYTES
        segment_bytes = bytes(audio_buffer[segment_start:segment_end])
        
        if len(segment_bytes) < 1600:  # Less than 0.05 seconds
            return
        
        # Calculate time offset for this segment
        segment_offset = segment_start / BYTES_PER_SECOND
        
        segment_mb = len(segment_bytes) / (1024 * 1024)
        print(f"\n🔄 [AudioStream] Processing segment: {segment_mb:.2f} MB (offset: {segment_offset:.2f}s)")
        
        # Process the segment
        segment_transcript = process_audio_segment(segment_bytes, segment_offset)
        
        if segment_transcript and segment_transcript.strip():
            # Accumulate transcript
            if context_accumulated_transcript.get(sid):
                context_accumulated_transcript[sid] += "\n" + segment_transcript
            else:
                context_accumulated_transcript[sid] = segment_transcript
            
            # Update processed bytes (account for overlap)
            new_processed_bytes = segment_end - OVERLAP_BYTES
            context_processed_bytes[sid] = new_processed_bytes
            
            processed_mb = new_processed_bytes / (1024 * 1024)
            print(f"✅ [AudioStream] Processed segment: {len(segment_transcript.splitlines())} lines | {processed_mb:.2f} MB processed")
            
            # Store chunks incrementally (when enough text accumulated)
            # Note: With 50-second segments, we'll store after each segment is processed
            accumulated_text = context_accumulated_transcript.get(sid, "")
            if len(accumulated_text) > 100:  # Store when we have text (lower threshold since segments are longer)
                print(f"📦 Storing accumulated transcript chunks for {sid}...")
                success = process_and_store_conversation(accumulated_text)
                if success:
                    # Clear accumulated transcript after successful storage
                    context_accumulated_transcript[sid] = ""
                    print(f"✅ Stored chunks for {sid}, cleared accumulated transcript")
        else:
            # Even if no transcript, advance processed bytes to avoid getting stuck
            context_processed_bytes[sid] = segment_end - OVERLAP_BYTES
            
    except Exception as e:
        print(f"❌ Error processing streaming segment for {sid}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        lock.release()


def process_context_audio_background(audio_filename):
    """
    Background task to process remaining context audio after recording completes.
    Processes any remaining unprocessed audio and finalizes storage.
    """
    try:
        print(f"🔄 Starting final processing for {audio_filename}...")
        
        # Step 1: Process audio through transcription and diarization
        print(f"📝 Step 1/3: Transcribing and diarizing audio...")
        transcript_text = process_audio_file(audio_filename)
        
        if not transcript_text or not transcript_text.strip():
            print(f"⚠️ No transcript generated from {audio_filename}")
            return False
        
        print(f"✅ Transcription and diarization complete. Generated transcript with {len(transcript_text.splitlines())} lines")
        
        # Step 2: Chunk conversation and store in vector database
        print(f"📦 Step 2/3: Chunking conversation and storing in vector database...")
        success = process_and_store_conversation(transcript_text)
        
        if success:
            print(f"✅ Step 3/3: Successfully processed and stored context audio from {audio_filename}")
        else:
            print(f"❌ Failed to store chunks in vector database for {audio_filename}")
        
        return success
        
    except Exception as e:
        print(f"❌ Error processing context audio {audio_filename}: {e}")
        import traceback
        traceback.print_exc()
        return False


@socketio.on("context_audio_complete")
def handle_context_audio_complete():
    """
    Client indicated context-recording finished. 
    Process any remaining unprocessed audio and finalize storage.
    """
    sid = request.sid
    # Mark context recording finished so next context_start will clear buffer
    is_context_recording[sid] = False
    finalizing_processing[sid] = True
    
    print(f"🏁 handle_context_audio_complete called for {sid}")
    
    try:
        # Acquire lock EARLY to prevent on_disconnect from cleaning up state while we work
        lock = context_processing_lock.get(sid)
        if not lock:
            print(f"⚠️ Lock not found for {sid} in handle_context_audio_complete (already disconnected?)")
            return

        print(f"🔒 Attempting to acquire lock for {sid}...")
        with lock:
            print(f"🔒 Lock acquired for {sid}")
            
            data = partial_context_audio.get(sid, None)
            if not data:
                print(f"⚠️ context_audio_complete received but no context data for {sid}")
                return

            filename = f"received_{sid}_context.wav"
            try:
                # Save the complete audio file
                with wave.open(filename, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # int16
                    wf.setframerate(16000)
                    wf.writeframes(bytes(data))
                total_mb = len(data) / (1024 * 1024)
                print(f"\n💾 [AudioStream] Saved '{filename}' | {total_mb:.2f} MB | {context_chunk_count.get(sid, 0)} chunks")
                
            except Exception as e:
                print(f"❌ Failed to write CONTEXT WAV for {sid}: {e}")
                # Continue processing even if save fails? Yes.

            if PROCESSING_AVAILABLE:
                # Process any remaining unprocessed audio
                processed_bytes = context_processed_bytes.get(sid, 0)
                total_bytes = len(data)
                remaining_bytes = total_bytes - processed_bytes
                print(f"📊 Stats: Total={total_bytes}, Processed={processed_bytes}, Remaining={remaining_bytes}")
                
                if remaining_bytes > 1600:  # More than 0.05 seconds remaining
                    print(f"🔄 Processing remaining {remaining_bytes} bytes for {sid}...")
                    # Extract and process remaining segment
                    remaining_segment = bytes(data[processed_bytes:])
                    segment_offset = processed_bytes / BYTES_PER_SECOND
                    remaining_transcript = process_audio_segment(remaining_segment, segment_offset)
                    
                    print(f"📄 Remaining transcript length: {len(remaining_transcript) if remaining_transcript else 0}")
                    if remaining_transcript:
                        print(f"📄 Transcript start: {remaining_transcript[:50]}...")

                    if remaining_transcript and remaining_transcript.strip():
                        # Add to accumulated transcript
                        if context_accumulated_transcript.get(sid):
                            context_accumulated_transcript[sid] += "\n" + remaining_transcript
                        else:
                            context_accumulated_transcript[sid] = remaining_transcript
                else:
                    print(f"⚠️ Remaining bytes {remaining_bytes} too small to process")
                
                # Store any remaining accumulated transcript
                accumulated_transcript = context_accumulated_transcript.get(sid)
                if accumulated_transcript:
                    print(f"📦 Storing final accumulated transcript for {sid}...")
                    process_and_store_conversation(accumulated_transcript)
                    context_accumulated_transcript[sid] = ""
                
                print(f"✅ [AudioStream] Completed streaming processing for {sid}")
            else:
                print(f"⚠️ Processing modules not available. Audio saved but not processed.")
    
    finally:
        # Ensure cleanup happens if we took ownership
        _perform_cleanup(sid)


# -----------------------------
# Face Recognition Endpoint
# -----------------------------
# Add camera_system to path
sys.path.append(str(project_root / "camera_system"))

# Example config defaults you can override in app.config
app.config.setdefault("MAX_CONTENT_LENGTH", 5 * 1024 * 1024)  # 5 MB max upload
app.config.setdefault("ALLOWED_MIMETYPES", {"image/jpeg", "image/png"})
app.config.setdefault("FACE_CROP_MARGIN", 0.12)  # fraction to expand crop (12%)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Try to import face recognition related modules; mark availability
try:
    import face_recognition  # face_recognition package
    from cloudinary_utils import upload_face_image
    from face_server_utils import search_face, enroll_face
    FACE_RECOGNITION_AVAILABLE = True
except Exception as e:
    logger.warning("Face recognition modules not available: %s", e)
    FACE_RECOGNITION_AVAILABLE = False


def allowed_file_mimetype(mimetype: Optional[str]) -> bool:
    if not mimetype:
        return False
    return mimetype.lower() in current_app.config["ALLOWED_MIMETYPES"]


def expand_box(
    top: int, right: int, bottom: int, left: int, img_h: int, img_w: int, margin_frac: float
) -> Tuple[int, int, int, int]:
    """Expand box by fraction of width/height while staying inside image bounds."""
    h = bottom - top
    w = right - left
    top_new = max(0, int(top - margin_frac * h))
    bottom_new = min(img_h, int(bottom + margin_frac * h))
    left_new = max(0, int(left - margin_frac * w))
    right_new = min(img_w, int(right + margin_frac * w))
    return top_new, right_new, bottom_new, left_new


@app.route("/process_face", methods=["POST"])
def process_face():
    """
    POST multipart/form-data:
      - image: file
      - name: optional -> if present, we ENROLL, else we RECOGNIZE

    JSON response:
      - status: "recognized" | "unknown" | "enrolled" | "error"
      - name: if recognized/enrolled
      - image_url: if enrolled or matched
      - message: debug/info
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return jsonify({"status": "error", "message": "Face recognition backend not loaded."}), 500

    # file presence
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image file provided."}), 400

    file = request.files["image"]
    name = request.form.get("name")  # optional: triggers enrollment

    # Basic file validation
    if not allowed_file_mimetype(file.mimetype):
        return jsonify({"status": "error", "message": f"Unsupported image type: {file.mimetype}"}), 415

    try:
        # Ensure the file stream is at start (face_recognition.load_image_file reads file-like)
        file.stream.seek(0)

        # Load image into numpy array (face_recognition accepts file-like)
        pil = Image.open(file.stream).convert("RGB")
        image_np = np.array(pil)  # shape (H, W, 3)
        img_h, img_w = image_np.shape[:2]

        # Detect faces (HOG faster; use model="cnn" if you have GPU and need accuracy)
        face_locations = face_recognition.face_locations(image_np, model="hog")

        if not face_locations:
            return jsonify({"status": "error", "message": "No face detected in image."}), 400

        # If multiple faces, choose the largest face (most likely intended subject)
        def area(loc):
            t, r, b, l = loc
            return (b - t) * (r - l)

        chosen_index = max(range(len(face_locations)), key=lambda i: area(face_locations[i]))
        chosen_location = face_locations[chosen_index]

        # Generate embedding for the chosen face only
        face_encodings = face_recognition.face_encodings(image_np, [chosen_location], num_jitters=10)
        if not face_encodings:
            return jsonify({"status": "error", "message": "Could not generate embedding."}), 400

        embedding = face_encodings[0].tolist()  # convert to Python list for storage

        # Enrollment flow
        if name:
            logger.info("Enrolling new face: %s", name)

            # Crop and expand a bit
            top, right, bottom, left = chosen_location
            top, right, bottom, left = expand_box(
                top, right, bottom, left, img_h, img_w, current_app.config["FACE_CROP_MARGIN"]
            )
            face_np = image_np[top:bottom, left:right]

            # Convert to JPEG bytes
            pil_face = Image.fromarray(face_np)
            img_io = io.BytesIO()
            pil_face.save(img_io, format="JPEG", quality=95)
            img_io.seek(0)
            image_bytes = img_io.getvalue()

            # Upload to Cloudinary (your helper should return a dict with secure_url)
            upload_result = upload_face_image(image_bytes, name)
            image_url = upload_result.get("secure_url") or upload_result.get("url")
            if not image_url:
                logger.exception("Cloudinary upload failed: %s", upload_result)
                return jsonify({"status": "error", "message": "Failed to upload image to storage."}), 500

            # Enroll in vector DB (Pinecone / whatever) via helper
            success = enroll_face(name, embedding, image_url)
            if success:
                return jsonify({"status": "enrolled", "name": name, "image_url": image_url}), 201
            else:
                logger.exception("Failed to save enrollment to vector DB for %s", name)
                return jsonify({"status": "error", "message": "Failed to save to vector DB."}), 500

        # Recognition flow
        else:
            logger.info("Recognizing face...")
            match_metadata = search_face(embedding)
            if match_metadata:
                return jsonify(
                    {
                        "status": "recognized",
                        "name": match_metadata.get("name"),
                        "image_url": match_metadata.get("image_url"),
                    }
                )
            else:
                return jsonify({"status": "unknown", "message": "No match found."}), 200

    except Exception as exc:
        logger.exception("Error in /process_face: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500

if __name__ == "__main__":
    print("🚀 Starting server on 0.0.0.0:5000")
    
    # Warmup models
    if PROCESSING_AVAILABLE:
        print("🔥 Warming up models...")
        try:
            get_models()
            print("✅ Models warmed up")
        except Exception as e:
            print(f"⚠️ Model warmup failed: {e}")

    socketio.run(app, host="0.0.0.0", port=5000)
