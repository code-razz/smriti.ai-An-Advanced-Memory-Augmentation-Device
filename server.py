# server.py
from flask import Flask, request
from flask_socketio import SocketIO, emit
import wave
import threading
import time

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
    Only append if the server-side is_context_recording[sid] is True (safety).
    """
    sid = request.sid
    if sid not in partial_context_audio:
        partial_context_audio[sid] = bytearray()
    if not is_context_recording.get(sid, False):
        # Defensive: ignore context chunks if not marked as recording on server.
        # In normal flow client will only send chunks while it has set recording True,
        # but network races can happen so guard here.
        print(f"⚠️ (context) Received chunk for {sid} but server thinks context is paused; ignoring.")
        return

    partial_context_audio[sid].extend(data)
    print(f"🎤 (context) Received chunk from {sid}: {len(data)} bytes (total_context={len(partial_context_audio[sid])})")


@socketio.on("context_audio_complete")
def handle_context_audio_complete():
    """
    Client indicated context-recording finished. Save WAV for later server-side processing.
    """
    sid = request.sid
    data = partial_context_audio.get(sid, None)
    if not data:
        print(f"⚠️ context_audio_complete received but no context data for {sid}")
    else:
        filename = f"received_{sid}_context.wav"
        try:
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16
                wf.setframerate(16000)
                wf.writeframes(bytes(data))
            print(f"💾 Saved recorded CONTEXT audio as '{filename}', {len(data)} bytes")
        except Exception as e:
            print(f"❌ Failed to write CONTEXT WAV for {sid}: {e}")

    # Mark context recording finished so next context_start will clear buffer
    is_context_recording[sid] = False


if __name__ == "__main__":
    print("🚀 Starting server on 0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000)
