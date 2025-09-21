"""
Chunk conversation into objects matching your sample.
Writes output to 'conversation_chunks.txt' and prints chunks.

Behavior:
- Chunks contain only metadata keys: conversation_id, timestamp, location, participants, tags.
- Chunking respects sentence boundaries: if a speaker's text would exceed the chunk size,
  finish the current sentence, flush the chunk, then continue that speaker in the next chunk.
"""

from datetime import datetime
from zoneinfo import ZoneInfo
import json
import re
import uuid
from typing import Callable, List, Dict, Tuple


import os
import sys

def run_diarization_and_get_output():
    """
    Run the STT diarization process and return the output content.
    """
    try:
        # Add the parent directory to sys.path to import from stt_diarization
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        stt_diarization_path = os.path.join(parent_dir, 'stt_diarization')
        sys.path.insert(0, stt_diarization_path)
        
        # Import and run the main function
        from main import main
        main()
        
        # Read the output file
        output_file = os.path.join(stt_diarization_path, 'output', 'named_diarized_output.txt')
        with open(output_file, 'r', encoding='utf-8') as file:
            output_content = file.read()
        return output_content

    except Exception as e:
        print(f"Error running diarization: {e}")
        return None
    finally:
        # Clean up sys.path
        if stt_diarization_path in sys.path:
            sys.path.remove(stt_diarization_path)


def split_into_utterances(conversation: str) -> List[Tuple[str, str]]:
    """
    Parse conversation into a list of (speaker, text) tuples.
    Expects lines with 'Speaker: text'. If no speaker label is found, speaker='Unknown'.
    """
    utterances = []
    # Normalize newlines and split lines
    lines = [ln.strip() for ln in conversation.splitlines() if ln.strip()]
    if not lines:
        return []
    speaker_re = re.compile(r'^\s*([^:]{1,50}):\s*(.*)$')
    for ln in lines:
        m = speaker_re.match(ln)
        if m:
            speaker = m.group(1).strip()
            text = m.group(2).strip()
        else:
            speaker = "Unknown"
            text = ln
        utterances.append((speaker, text))
    return utterances

def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences, keeping punctuation.
    Uses a conservative regex: split on (?<=[.!?])\s+.
    If no sentence boundaries found, returns the whole text as a single sentence.
    """
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return []
    return parts

def chunk_conversation_from_func(
    conversation_getter: Callable[[], str],
    max_chars: int = 900,
    tz_name: str = "Asia/Kolkata",
    out_filename: str = "created_conversation_chunks.txt",
    write_to_file: bool = True
) -> List[Dict]:
    """
    Produce conversation_chunks list of dicts like the sample, print them,
    and store JSON to out_filename.
    """
    full_text = conversation_getter().strip()
    if not full_text:
        return []

    utterances = split_into_utterances(full_text)

    # base conversation id same for all chunks (like sample)
    base_conv_id = f"conv_{datetime.now(ZoneInfo(tz_name)).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    conversation_chunks = []
    buffer_lines: List[str] = []
    buffer_len = 0  # approximate char count including newline per line
    chunk_counter = 1

    def flush_buffer():
        nonlocal buffer_lines, buffer_len, chunk_counter
        if not buffer_lines:
            return
        text = "\n".join(buffer_lines)
        # participants: extract speakers from each buffer line (before ':')
        participants = []
        for bl in buffer_lines:
            sp = bl.split(":", 1)[0].strip()
            if sp not in participants:
                participants.append(sp)
        ts = datetime.now(ZoneInfo(tz_name)).isoformat()
        conversation_chunks.append({
            "text": text,
            "metadata": {
                "conversation_id": base_conv_id,   # same for all chunks
                "timestamp": ts,
                "location": "not_provided",
                "participants": participants,
                "tags": "not_provided"
            }
        })
        chunk_counter += 1
        buffer_lines = []
        buffer_len = 0

    # Process each utterance and sentences
    for speaker, text in utterances:
        sentences = split_sentences(text)
        if not sentences:
            continue
        # We will append sentences one by one, possibly joining them into the speaker's line.
        for i, sent in enumerate(sentences):
            # Determine how much length will be added if we append this sentence.
            if buffer_lines and buffer_lines[-1].startswith(f"{speaker}:"):
                # we would append to existing last line for same speaker in buffer
                length_increase = 1 + len(sent)  # space + sentence
            else:
                # new line: "Speaker: sentence"
                length_increase = len(speaker) + 2 + len(sent)  # speaker + ": " + sentence

            # If adding would exceed max and buffer is not empty, flush buffer first
            if buffer_lines and (buffer_len + length_increase > max_chars):
                # finish current chunk at end of last full sentence (we're already at sentence boundary)
                flush_buffer()

            # After possible flush, add the sentence to buffer
            if buffer_lines and buffer_lines[-1].startswith(f"{speaker}:"):
                # append to last line
                buffer_lines[-1] = buffer_lines[-1] + " " + sent
                buffer_len += 1 + len(sent)
            else:
                # new line for this speaker
                new_line = f"{speaker}: {sent}"
                buffer_lines.append(new_line)
                buffer_len += len(new_line) + 1  # +1 for newline when joined

            # Edge case: if buffer was empty and the single sentence itself is larger than max_chars,
            # we still keep it (as one chunk) and will flush on next addition
            if buffer_len > max_chars and len(buffer_lines) == 1:
                # allow an oversized single-sentence chunk; we'll flush on next addition or end
                pass

    # flush any remaining lines
    flush_buffer()

    # Write to file as JSON in the same directory as this script (if requested)
    if write_to_file:
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(script_dir, out_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(conversation_chunks, f, ensure_ascii=False, indent=4)
            print(f"Wrote {len(conversation_chunks)} chunks to '{out_filename}'")
        except Exception as e:
            print("Warning: failed to write file:", e)

    # Print chunks summary (matching sample style)
    for idx, ch in enumerate(conversation_chunks, start=1):
        print(f"=== Conversation {idx} ===")
        print(ch["text"])
        print("metadata:", ch["metadata"])
        print()

    return conversation_chunks

def default_get_conversation() -> str:
    """
    Get conversation from STT diarization output or return sample data.
    """
    # Try to get conversation from diarization output
    conversation = run_diarization_and_get_output()
    if conversation:
        return conversation
    
    # Fallback to sample conversation if diarization fails
    return """User: Did you bring the medicine?
Alex: Yes, I picked it up on the way.
User: Great, I need to take it after dinner.
Alex: Remember, the doctor said to avoid coffee.
User: Got it. I’ll skip the coffee tonight.
Alex: That’s good. Better safe than sorry."""
# Generate conversation chunks and make them available for import
conversation_chunks = chunk_conversation_from_func(default_get_conversation, max_chars=900, write_to_file=True)

# Example run if module executed
if __name__ == "__main__":
    # When run directly, also write to file
    chunks_with_file = chunk_conversation_from_func(default_get_conversation, max_chars=900, write_to_file=True)
    print(f"Generated {len(chunks_with_file)} conversation chunks")
    print("Chunks are available as 'conversation_chunks' variable for import")
    print(f"Chunks saved to 'created_conversation_chunks.txt' in the context folder")
