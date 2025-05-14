import sounddevice as sd
import numpy as np
import queue
import threading
import time
from resemblyzer import VoiceEncoder, preprocess_wav
from faster_whisper import WhisperModel
from scipy.io.wavfile import write
import os

# Initialize components
q = queue.Queue()
encoder = VoiceEncoder()
model = WhisperModel("base.en", device="cpu")  # You can use 'medium.en' or 'large-v2' if GPU is available

# Audio settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # seconds
NUM_SAMPLES = SAMPLE_RATE * CHUNK_DURATION

# For speaker tracking
speaker_embeddings = []
speaker_labels = []
speaker_counter = 1
SIMILARITY_THRESHOLD = 0.75


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    q.put(indata.copy())


def speaker_id(new_embed):
    global speaker_embeddings, speaker_labels, speaker_counter

    if not speaker_embeddings:
        speaker_embeddings.append(new_embed)
        speaker_labels.append("Speaker 1")
        return "Speaker 1"

    similarities = [np.dot(embed, new_embed) for embed in speaker_embeddings]
    max_sim = max(similarities)
    if max_sim > SIMILARITY_THRESHOLD:
        idx = similarities.index(max_sim)
        return speaker_labels[idx]
    else:
        speaker_counter += 1
        label = f"Speaker {speaker_counter}"
        speaker_embeddings.append(new_embed)
        speaker_labels.append(label)
        return label


def process_audio():
    print("🎙️ Starting real-time transcription + speaker labeling...\n")

    buffer = np.zeros((0, 1), dtype=np.float32)

    while True:
        chunk = q.get()
        buffer = np.concatenate((buffer, chunk), axis=0)

        if len(buffer) >= NUM_SAMPLES:
            audio_chunk = buffer[:NUM_SAMPLES]
            buffer = buffer[NUM_SAMPLES:]

            filename = "chunk.wav"
            write(filename, SAMPLE_RATE, (audio_chunk * 32767).astype(np.int16))

            # Voice embedding
            wav = preprocess_wav(filename)
            embed = encoder.embed_utterance(wav)
            speaker = speaker_id(embed)

            # Transcribe
            segments, _ = model.transcribe(filename)
            for segment in segments:
                print(f"[{speaker}] {segment.text}")

            os.remove(filename)


# Start audio stream
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)
with stream:
    t = threading.Thread(target=process_audio)
    t.start()
    while True:
        time.sleep(0.1)
