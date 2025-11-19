# server.py
from flask import Flask, request
from flask_socketio import SocketIO, emit
import wave
import threading
import time
import sys
import os
from pathlib import Path

# Add paths for importing processing modules
project_root = Path(__file__).parent
stt_diarization_path = project_root / "stt_diarization"
context_path = project_root / "context"

sys.path.insert(0, str(stt_diarization_path))
sys.path.insert(0, str(context_path))

# Import processing functions
try:
    from process_audio import process_audio_file
    from process_audio_streaming import process_audio_segment
    from process_chunks import process_and_store_conversation
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


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    print(f"❌ Client disconnected: {sid}")
    partial_audio.pop(sid, None)
    partial_context_audio.pop(sid, None)
    is_recording.pop(sid, None)
    is_context_recording.pop(sid, None)
    reply_stream_stop_flags.pop(sid, None)
    reply_streaming_task.pop(sid, None)
    context_processed_bytes.pop(sid, None)
    context_processing_lock.pop(sid, None)
    context_accumulated_transcript.pop(sid, None)


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
        print(f"🆕 (query) Starting new recording for {sid} — cleared previous query buffer")

    partial_audio[sid].extend(data)
    print(f"🎤 (query) Received chunk from {sid}: {len(data)} bytes (total_query={len(partial_audio[sid])})")


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
            print(f"💾 Saved recorded QUERY audio as '{filename}', {len(data)} bytes")
        except Exception as e:
            print(f"❌ Failed to write QUERY WAV for {sid}: {e}")

    # Mark recording finished so next audio_chunk will clear buffer
    is_recording[sid] = False

    # Reset stop flag (ensure new stream allowed)
    reply_stream_stop_flags[sid] = False

    # Start streaming reply back (background task)
    print(f"🔁 Starting reply stream to {sid} (background task)")
    reply_streaming_task[sid] = socketio.start_background_task(stream_reply_to_client, sid)


def stream_reply_to_client(sid):
    """
    Stream SAMPLE_REPLY bytes in CHUNK_SIZE pieces to client.
    Honors reply_stream_stop_flags[sid] to stop early when client requests.
    Emits 'server_audio_chunk' events and a final 'server_audio_complete'.
    """
    print(f"▶️ stream_reply_to_client started for {sid}")
    total = len(SAMPLE_REPLY)
    sent = 0
    while sent < total:
        # stop check
        if reply_stream_stop_flags.get(sid, False):
            print(f"⏹️ Server-side streaming stopped by client request for {sid}")
            break
        end = min(sent + CHUNK_SIZE, total)
        chunk = SAMPLE_REPLY[sent:end]
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
    print(f"✅ Finished (or stopped) streaming reply to {sid} (sent={sent}/{total})")


@socketio.on("stop_server_stream")
def on_stop_server_stream():
    """
    Client asked server to stop streaming (client pressed 'a' to start new query recording).
    """
    sid = request.sid
    reply_stream_stop_flags[sid] = True
    print(f"📴 Received stop_server_stream from {sid}")


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
    print(f"🟣 Received context_start from {sid} -> starting NEW context session (clearing previous buffer)")
    partial_context_audio[sid] = bytearray()
    is_context_recording[sid] = True
    context_processed_bytes[sid] = 0
    context_accumulated_transcript[sid] = ""


@socketio.on("context_resume")
def on_context_resume():
    """
    Client indicates context recording is resuming (after being paused by a query or previously paused).
    Do NOT clear the buffer — continue appending.
    """
    sid = request.sid
    is_context_recording[sid] = True
    print(f"🟣 Received context_resume from {sid} -> resuming context recording (no buffer clear)")


@socketio.on("context_pause")
def on_context_pause():
    """
    Client indicates context recording is paused (e.g., because user started a query-recording).
    """
    sid = request.sid
    is_context_recording[sid] = False
    print(f"🟣 Received context_pause from {sid} -> pausing context recording")


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
        print(f"⚠️ (context) Received chunk for {sid} but server thinks context is paused; ignoring.")
        return

    partial_context_audio[sid].extend(data)
    total_bytes = len(partial_context_audio[sid])
    processed_bytes = context_processed_bytes.get(sid, 0)
    unprocessed_bytes = total_bytes - processed_bytes
    
    # Check if we have enough unprocessed audio to process a segment
    if unprocessed_bytes >= SEGMENT_DURATION_BYTES and PROCESSING_AVAILABLE:
        # Process the segment in background (non-blocking)
        socketio.start_background_task(process_context_segment_streaming, sid)
    
    print(f"🎤 (context) Received chunk from {sid}: {len(data)} bytes (total={total_bytes}, processed={processed_bytes})")


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
        
        print(f"🔄 Processing streaming segment for {sid}: {len(segment_bytes)} bytes (offset: {segment_offset:.2f}s)")
        
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
            
            print(f"✅ Processed segment for {sid}: {len(segment_transcript.splitlines())} lines (total processed: {new_processed_bytes} bytes)")
            
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
    data = partial_context_audio.get(sid, None)
    if not data:
        print(f"⚠️ context_audio_complete received but no context data for {sid}")
    else:
        filename = f"received_{sid}_context.wav"
        try:
            # Save the complete audio file
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16
                wf.setframerate(16000)
                wf.writeframes(bytes(data))
            print(f"💾 Saved recorded CONTEXT audio as '{filename}', {len(data)} bytes")
            
            if PROCESSING_AVAILABLE:
                # Process any remaining unprocessed audio
                processed_bytes = context_processed_bytes.get(sid, 0)
                total_bytes = len(data)
                remaining_bytes = total_bytes - processed_bytes
                
                if remaining_bytes > 1600:  # More than 0.05 seconds remaining
                    print(f"🔄 Processing remaining {remaining_bytes} bytes for {sid}...")
                    # Extract and process remaining segment
                    remaining_segment = bytes(data[processed_bytes:])
                    segment_offset = processed_bytes / BYTES_PER_SECOND
                    remaining_transcript = process_audio_segment(remaining_segment, segment_offset)
                    
                    if remaining_transcript and remaining_transcript.strip():
                        # Add to accumulated transcript
                        if context_accumulated_transcript.get(sid):
                            context_accumulated_transcript[sid] += "\n" + remaining_transcript
                        else:
                            context_accumulated_transcript[sid] = remaining_transcript
                
                # Store any remaining accumulated transcript
                accumulated_transcript = context_accumulated_transcript.get(sid)
                if accumulated_transcript:
                    print(f"📦 Storing final accumulated transcript for {sid}...")
                    process_and_store_conversation(accumulated_transcript)
                    context_accumulated_transcript[sid] = ""
                
                print(f"✅ Completed streaming processing for {sid}")
            else:
                print(f"⚠️ Processing modules not available. Audio saved but not processed.")
                
        except Exception as e:
            print(f"❌ Failed to write CONTEXT WAV for {sid}: {e}")

    # Mark context recording finished so next context_start will clear buffer
    is_context_recording[sid] = False


if __name__ == "__main__":
    print("🚀 Starting server on 0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000)
