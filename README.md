# 🎙️ Real-time Speech Transcription and Speaker Diarization

This project provides a real-time transcription system with speaker diarization using the `faster-whisper` model for transcription and `resemblyzer` for speaker identification. It listens through your microphone, segments audio every few seconds, and prints out the transcription along with identified speaker tags.

---

## 🧠 Features

- 🔊 Live audio recording using your microphone
- 🗣️ Speaker recognition and labeling (Speaker 1, Speaker 2, ...)
- 📄 Real-time transcription using Whisper (base.en by default)
- 🧠 Embedding comparison via cosine similarity for speaker tracking

---

## 📦 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
