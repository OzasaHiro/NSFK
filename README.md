# NSFK? (Not Safe For Kids?)
**AI-Powered YouTube Video Safety Analyzer for Parents**

*Weekend Agent Hack - Mountain View, June 14th, 2025*

## 🎯 Project Overview

NSFK is an AI-powered tool that analyzes YouTube videos to help busy parents determine if content is appropriate for their children. The system provides quick safety assessments with detailed category-based scoring and summaries.

## ✨ Key Features

- **🔍 Comprehensive Video Analysis** - Downloads and analyzes video content, audio, and visual elements
- **🎯 Safety Scoring System** - 0-100 point scale with category breakdowns:
  - Violence (20 pts max)
  - Language (15 pts max) 
  - Scary Content (20 pts max)
  - Sexual Content (15 pts max)
  - Substance Use (10 pts max)
  - Dangerous Behavior (10 pts max)
  - Educational Value (10 pts max)
- **🤖 AI-Powered Analysis** - Uses Google Gemini for intelligent content evaluation
- **📊 Category-Based Scoring** - Detailed breakdown by content type
- **💡 Smart Recommendations** - Safe/Review Required/Not Recommended guidance
- **🔗 REST API** - FastAPI-based web service for easy integration
- **🌐 Web Interface** - Simple test page for immediate use

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/OzasaHiro/NSFK.git
cd NSFK

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up Gemini API key
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 2. Start API Server
```bash
python3 start_api.py
```

### 3. Test with Web Interface
```bash
open web_test.html
```

## 🛠️ API Usage

### Basic Analysis
```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtube.com/watch?v=VIDEO_ID"}'
```

### Example Response
```json
{
  "safety_score": 79,
  "recommendation": "Review Required",
  "category_scores": {
    "Violence": 18,
    "Language": 15,
    "Scary Content": 11,
    "Sexual Content": 15,
    "Substance Use": 10,
    "Dangerous Behavior": 10,
    "Educational Value": 0
  },
  "summary": "Animated music video with vampire-themed content...",
  "keywords": ["Monsters", "Vampires", "Dark visuals", "Spooky", "Animated"]
}
```

## 📁 Project Structure

```
NSFK/
├── nsfk_analyzer.py        # Main analysis engine
├── youtube_downloader.py   # YouTube video download functionality
├── api.py                 # FastAPI web service
├── start_api.py           # API server startup script
├── web_test.html          # Web testing interface
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
├── reports/               # Analysis report outputs
└── docs/                  # API documentation
```

## 🔧 Tech Stack

- **AI/ML**: Google Gemini API for content analysis
- **Video Processing**: yt-dlp, OpenCV, PySceneDetect
- **Audio Analysis**: Whisper for transcription
- **API Framework**: FastAPI with async support
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Deployment**: Docker support included

## 📖 Documentation

- **API Documentation**: http://127.0.0.1:8000/docs (when server is running)
- **Integration Guide**: See `API_INTEGRATION_GUIDE.md`
- **Detailed API Reference**: See `API_README.md`

## 🎯 Scoring System

| Score Range | Recommendation | Description |
|-------------|----------------|-------------|
| 80-100 | ✅ Safe | Suitable for children |
| 60-79 | ⚠️ Review Required | Parent judgment needed |
| 0-59 | ❌ Not Recommended | Inappropriate for children |

## 🔮 Future Development

- **Chrome Extension** - Browser integration for YouTube pages
- **Comments Analysis** - Evaluate YouTube comments for safety
- **Reddit/Wiki Integration** - Additional context from community discussions
- **Batch Processing** - Analyze multiple videos simultaneously
- **Custom Filtering** - User-defined safety criteria

## 🤝 Contributing

This project was developed during the Weekend Agent Hack. Contributions and improvements are welcome!

## 📄 License

MIT License - see LICENSE file for details.

---

*Built with ❤️ using Claude Code at Weekend Agent Hack - Mountain View, June 14th, 2025*
