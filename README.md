# 🎥 AI Video Voice Translation

An AI-powered pipeline that translates the spoken audio in a video into another language and **re-syncs the speaker's lips** to the new voice — combining speech recognition, text-to-speech, and Wav2Lip-based lip synchronization.

> ⚠️ **Disclaimer:** Built for educational, research, and portfolio purposes. Generated translations and synthesized voices may contain errors and should be reviewed before professional or public use. Only process content you have permission to use.

**[▶ Demo Video](https://drive.google.com/file/d/1MqyN-5GsmSXOKeB2Al7iacZp5E8GV-Xf/view?usp=drive_link)**

---

## 🚀 Overview

This project demonstrates a full multimodal AI pipeline — audio extraction, speech-to-text, machine translation, TTS voice generation, and AI-driven video re-synchronization — wrapped in a lightweight Python web interface. It's an end-to-end example of chaining several ML models into a single production-style workflow.

## 🔄 Pipeline

```
Video Upload → Audio Extraction → Speech Recognition → Text Translation
            → Target-Language TTS → Wav2Lip Sync → Translated Video Output
```

## ✨ Key Features

- 🎙️ Speech recognition and audio extraction from video
- 🌍 Text translation into a target language
- 🗣️ Text-to-speech generation of translated audio
- 👄 AI-based lip synchronization using **Wav2Lip**
- 🌐 Simple web interface for upload → processing → download
- 🧪 Installation/environment self-test script

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Core** | Python |
| **AI / ML** | Wav2Lip (deep-learning lip sync), Speech-to-Text, TTS |
| **Media** | FFmpeg-based audio/video processing (MP4, MP3, WAV) |
| **Web** | Python web app + HTML templates |

## 📂 Project Structure

```
project/
├── app.py                 # Web interface entry point
├── main.py                # Core processing workflow
├── setup_models.py        # AI model setup/download
├── test_installation.py   # Environment/dependency check
├── wav2lip/                # Lip-sync components
├── templates/              # Web UI templates
├── uploads/ | outputs/ | work/   # Runtime I/O (gitignored)
└── requirements.txt
```

> Large pretrained model weights and generated media are excluded from version control — use Git LFS or external storage if needed.

## 💻 Quick Start

```bash
# Clone & enter the repo
git clone https://github.com/ritin3098-bit/AI-Video-Voice-Translation.git
cd AI-Video-Voice-Translation

# Set up environment
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt

# Download/configure AI models
python setup_models.py

# Verify environment
python test_installation.py

# Run
python app.py
```

Visit **http://127.0.0.1:5000/**

## 🎯 Core Components

| File / Folder | Purpose |
|---|---|
| `app.py` | Web interface entry point |
| `main.py` | End-to-end processing workflow |
| `setup_models.py` | Downloads/configures required AI models |
| `test_installation.py` | Validates dependencies and environment |
| `wav2lip/` | Lip-synchronization components |
| `templates/` | Frontend HTML templates |

## ⚙️ Infrastructure Notes

Lip-sync and video processing are compute-intensive — expect meaningful CPU/RAM/disk use and significant speedups from GPU inference. Basic free-tier hosting is generally insufficient for the full pipeline; size infrastructure to the model and video-processing load.

## 🔐 Security & Privacy

- Never commit API keys, secrets, or `.env` files
- Validate uploaded file types/sizes; restrict access to private media
- Store secrets via environment variables; clean up temp files regularly
- Recommended `.gitignore`: `.env`, `venv/`, `__pycache__/`, `uploads/*`, `outputs/*`, `work/*`, `*.mp3`, `*.wav`, `*.mp4`, `*.pth`, `*.pt`, `*.ckpt`

## 🔮 Roadmap

- Multi-language support + improved voice cloning
- Faster GPU inference and progress indicators
- Cloud storage integration + user authentication
- Subtitle generation and voice-quality enhancement
- Production-ready deployment architecture

## 👨‍💻 Author

**Ritin Setia** — [GitHub](https://github.com/ritin3098-bit)

## ⚠️ Responsible Use

AI-generated voice and lip-sync technology can be misused for impersonation or misleading content. Only process media you have rights to use, and clearly disclose AI-generated or translated content where appropriate.

---
*Educational/portfolio project.*
