import gradio as gr
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from resemblyzer import preprocess_wav, VoiceEncoder
from faster_whisper import WhisperModel
import os
import tempfile
from datetime import datetime
import requests
import json

# Constants
SAMPLE_RATE = 16000
SIMILARITY_THRESHOLD = 0.8  # Threshold for speaker differentiation
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Default Ollama API endpoint

# Load models
encoder = VoiceEncoder()
model = WhisperModel("base.en", device="cpu", compute_type="float32")

# Global speaker state
speaker_embeddings = []
speaker_labels = []
speaker_counter = 1

def get_summary(text: str) -> str:
    """Get summary of the transcript using Ollama."""
    try:
        prompt = f"{text}"  # Only the transcript

        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": "meeting-summarizer",
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error getting summary: {response.status_code}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

def speaker_id(new_embed):
    """Assign speaker label based on embedding similarity."""
    global speaker_embeddings, speaker_labels, speaker_counter

    # Compare the embedding of the new utterance with existing ones
    if not speaker_embeddings:
        speaker_embeddings.append(new_embed)
        speaker_labels.append(f"Speaker {speaker_counter}")
        return f"Speaker {speaker_counter}"

    similarities = [np.dot(embed, new_embed) for embed in speaker_embeddings]
    max_sim = max(similarities)

    if max_sim > SIMILARITY_THRESHOLD:
        idx = similarities.index(max_sim)
        return speaker_labels[idx]
    else:
        # If no match is found, it means this is a new speaker
        speaker_counter += 1
        new_label = f"Speaker {speaker_counter}"
        speaker_embeddings.append(new_embed)
        speaker_labels.append(new_label)
        return new_label

def record_and_transcribe(minutes: float, title: str = ""):
    """Record audio, transcribe with Whisper, identify speaker, return transcript + file."""
    duration = int(minutes * 60)  # convert minutes to seconds
    print(f"Recording audio for {minutes:.2f} minutes...")

    # Record audio
    audio = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()

    # Save audio to temporary file
    wav_path = "temp_chunk.wav"
    write(wav_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    if np.any(audio):
        try:
            wav = preprocess_wav(wav_path)
            embed = encoder.embed_utterance(wav)
            speaker = speaker_id(embed)  # Identify the speaker

            # Transcribe using Whisper
            segments, _ = model.transcribe(wav_path)
            os.remove(wav_path)

            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_title = title.replace(" ", "_") if title else "Untitled"
            filename = f"{timestamp}_{filename_title}"

            full_text = ""
            for seg in segments:
                start = str(datetime.utcfromtimestamp(seg.start).strftime("%H:%M:%S"))
                end = str(datetime.utcfromtimestamp(seg.end).strftime("%H:%M:%S"))
                full_text += f"[{start} - {end}] {speaker}: {seg.text.strip()}\n"

            # Get summary using Ollama
            summary = get_summary(full_text)

            # Save transcript and summary to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tf:
                tf.write(f"=== {filename} ===\n\n")
                tf.write("=== TRANSCRIPT ===\n\n")
                tf.write(full_text)
                tf.write("\n\n=== SUMMARY ===\n\n")
                tf.write(summary)
                transcript_path = tf.name

            return full_text, summary, transcript_path
        except Exception as e:
            return f"Error during transcription: {e}", "Error getting summary", None
    else:
        return "Silent audio detected.", "No summary available for silent audio.", None

# Gradio UI
with gr.Blocks(title="🎙️ Real-time Transcription + Speaker Diarization") as demo:
    gr.Markdown("## 🎙️ Real-time Transcription + Speaker Diarization")
    gr.Markdown("Enter title and duration, then click Record:")

    # UI Elements
    title_input = gr.Textbox(label="Conversation Title", placeholder="Enter a title for this conversation")
    duration_input = gr.Number(label="Duration (minutes)", value=0.1, precision=2, minimum=0.1)
    record_btn = gr.Button("🔴 Start Recording")
    output_text = gr.Textbox(label="Transcription", lines=10)
    summary_text = gr.Textbox(label="Summary", lines=5)
    download_btn = gr.File(label="Download Transcript & Summary (.txt)", interactive=True)

    # Button Actions
    record_btn.click(
        fn=record_and_transcribe, 
        inputs=[duration_input, title_input], 
        outputs=[output_text, summary_text, download_btn]
    )

# Launch the Gradio interface
demo.launch(share=True)
