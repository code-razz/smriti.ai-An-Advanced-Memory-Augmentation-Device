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

def default_get_conversation() -> str:
    # Your sample conversation (replace this function with your real source)
    return """Unknown_SPEAKER_00: everyone. topic for group discussion is artificial intelligence. Artificial intelligence is the ability of a computer program to learn and think. Anything can be considered as artificial intelligence if it involves a program doing something that we would normally rely on our intelligence.
Stwmonica: And here to talk about the advantages of artificial intelligence, the one of the biggest advantages is that it reduces human error. Human error is one of humans make mistakes, mistakes like usually and it happens all the time. So computers however do not make all these mistakes. With artificial intelligence the decisions are taken but they previously gathered information and it uses a set of algorithms. So errors are reduced and chances of reaching accuracy is high.
Sthross: Yes, I am going to agree with you but isn't this a making the human so lazy with its applications? Like most of its applicants are automated. The majority of its work is being automated. So like if humans get addicted to this type of inventions, this might cause a lot of problem to the future generations. Like they might become so lazy.
Sforachel: need not making people lazy. Instead people will be still working on their creativity part. for the decisions where they have to think and make something, people will be working on that. Whereas for the jobs which are repetitive like sending mail and making documents. And in the documents where the errors should be reduced. For those particular things the robots or the AI machines will be working. This will reduce people from working on their boring parts and instead use their mind and energy on the creative parts.
Sfichandler: Yeah, I see I said but I think for the repetitive jobs and the boring jobs you are talking about. I don't think there is any reason to have robots that are mature made of higher costs and also it requires maintenance of these robots. So for these repetitive jobs, I don't think you need to put this much price and you need to get a robot to do these silly jobs.
Stwmonica: Yes, I agree but it's a one -time investment. So this is considered because artificial intelligence is available for humans, customers or any one. Like for more than 24 hours like 24
Sthross: only it is available 24 by seven but isn't this availability making a cause of unemployment? Like if this AI is replacing the majority of this repetitive tasks. So this might cause a lot of unemployment. Like every organization is looking to replace the minimum qualified individuals with the error of what it means. This will cause a lot of unemployment in the future.
Sforachel: This is not going to cause unemployment because the first point is the people are only ones who are making the robots. So if the unemployment will be caused in this sector, it will be created in some other sectors. The employment will be created in some other sector. And along with that, for the jobs with the risk involved, where the human life is at risk. We can make machines do those tasks. Like in fire extinguishing work or in any natural disaster or human disaster. In those cases instead if a human will do the work, there is a risk of their lives. Where the machine works, there are more accuracy, more efficiency and the risk is less.
Sfichandler: As I see, but what I think is these robots are pre -programmed. So given this natural disaster, that you are talking about. If the robots doesn't really know the solution to, like if they cannot really act on spot, like the new problem arises during this natural disaster. Then I don't think there is any robot that can solve that can think on its own and solve a problem accordingly.
Stwmonica: it also helps us with digital assistance, education institutions, helpline centers, hospitals, etc. uses digital assistance. So organized organizations have a customer support team. So they work all the time and at any time humans take the help and it's useful.
Sthross: only it is available in digital assistance. But this unit has one thing is that machines can't alter their responses to changing environments. So like that is the basic premise on which these AI are built. This repetitive nature of work where the input doesn't change. machines can't get what is right, what is wrong or they are not capable of what is legal or illegal, that kind of things.
Sforachel: the new task we still have humans to work for those kind of things. But for the task which are predefined, like we can tell the machine that this is right and this is wrong. For those particular tasks, the decision making by the robots will be faster and will be more efficient. Like the error will be less in those decisions and there will be faster. So this will reduce human efforts along with that it will increase their efficiency of that particular task.
Sfichandler: As a sheep, but I think even in that task, like the old task, that's which you are talking about there. There needs to be improvements, right? Like the world is changing and it has to be updating. They like every day. So I don't think the robots will be able to update themselves and again the human involvement has to be there to update the robots. So it's I think she's time taking and also it's very much of cost effective.
Unknown_SPEAKER_00: guys. Those were some nice points as we have seen these have these are some advantages and disadvantages of artificial intelligence. As we know every new invention or breakthrough will have both advantages and disadvantages. It is upon us how we use it. So when we use the artificial intelligence to its limit, we all could really enjoy the outcomes of it. you.
"""

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
    out_filename: str = "conversation_chunks.txt"
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

    # Write to file as JSON
    try:
        with open(out_filename, "w", encoding="utf-8") as f:
            json.dump(conversation_chunks, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print("Warning: failed to write file:", e)

    # Print chunks summary (matching sample style)
    for idx, ch in enumerate(conversation_chunks, start=1):
        print(f"=== Conversation {idx} ===")
        print(ch["text"])
        print("metadata:", ch["metadata"])
        print()

    print(f"Wrote {len(conversation_chunks)} chunks to '{out_filename}'")
    return conversation_chunks

# Example run if module executed
if __name__ == "__main__":
    chunks = chunk_conversation_from_func(default_get_conversation, max_chars=900)
