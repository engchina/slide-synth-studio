# Slide Synth Studio 🎥🗣️

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![GPLv3 License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://opensource.org/licenses/GPL-3.0)
[![GitHub stars](https://img.shields.io/github/stars/yourname/SlideSynth-Studio?style=social)](https://github.com/yourname/SlideSynth-Studio)

**Intelligent Presentation Automation Suite** - Transform PPT slides into narrated videos with AI-powered voiceovers

![Demo Preview](https://via.placeholder.com/800x400.png?text=SlideSynth+Studio+Demo)

## 🌟 Key Features

### Core Capabilities
- **Smart Slide Processing**
  - Multi-threaded PPT/PPTX to image conversion (PNG/JPEG)
  - Resolution control up to 600dpi
  - Batch processing for multiple files

- **AI Content Understanding**
  - Automatic script generation from slide content
  - Key point extraction and structure analysis
  - Multi-language support (EN/CN/JP)

- **Interactive Workflow**
  - Web-based UI with Gradio
  - Visual preview system with thumbnail gallery
  - Real-time processing status monitoring

### Upcoming Features (Q4 2024)
- **Voice Synthesis Engine**
  - Text-to-speech conversion with emotion control
  - Multi-voice character support
  - Timing synchronization with slides

- **Video Production**
  - Automated video rendering with transitions
  - Customizable templates for different styles
  - Direct export to YouTube/MP4 format

## 🚀 Quick Start

### Prerequisites
- Ubuntu 24.04+ (Recommended)
- Python 3.1+
- LibreOffice 7.0+
- FFmpeg (For upcoming video features)

### Installation
```bash
# Clone repository
git clone https://github.com/engchina/slide-synth-studio.git
cd slide-synth-studio

# Install system dependencies
sudo apt install libreoffice libjpeg-dev zlib1g-dev
pip install pytesseract pdf2image

# Setup Python environment
pip install -r requirements.txt
# pip list --format=freeze > requirements.txt