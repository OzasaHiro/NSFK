# NSFK? (Not Safe For Kids?)
**AI-Powered YouTube Video Safety Analyzer for 10-Year-Olds**

*Weekend Agent Hack - Mountain View, June 14th, 2025*

## 🎯 Project Overview

NSFK is an AI-powered tool that analyzes YouTube videos to help busy parents determine if content is appropriate for 10-year-old children. The system provides quick safety assessments with detailed category-based scoring, comment analysis, and channel reputation evaluation.

## ✨ Key Features

- **🔍 Comprehensive Video Analysis** - Downloads and analyzes video content and visual elements
- **👦 10-Year-Old Focused** - All analysis specifically tailored for 10-year-old appropriateness
- **🎯 Safety Scoring System** - 0-100 point scale with positive category breakdowns:
  - Non-Violence (20 pts max) - Absence of violence, weapons, fighting
  - Appropriate Language (15 pts max) - Clean, family-friendly language
  - Non-Scary Content (20 pts max) - No horror, jump scares, frightening imagery
  - Family-Friendly Content (15 pts max) - No inappropriate themes
  - Substance-Free (10 pts max) - No drugs, alcohol, smoking
  - Safe Behavior (10 pts max) - No risky activities kids might imitate
  - Educational Value (10 pts max) - Positive learning content for 10-year-olds
- **💬 Comment Analysis** - Analyzes YouTube comments for safety concerns and age-appropriateness
- **🌐 Channel Reputation** - Evaluates channel/creator reputation for child safety
- **🤖 Advanced AI Architecture** - Primary: Meta Llama-4-Maverick-17B-128E-Instruct-FP8 via GMI API + Gemini/OpenAI fallbacks
- **📊 Category-Based Scoring** - Detailed breakdown with intuitive positive scoring
- **💡 Smart Recommendations** - Safe/Review Required/Not Recommended guidance
- **🔗 REST API** - FastAPI-based web service for easy integration
- **🌐 Web Interface** - Enhanced test page with additional analysis sections
- **🎨 Chrome Extension** - Browser extension for seamless YouTube integration

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

# Set up API keys
echo "GMI_API_KEY=your_gmi_api_key_here" > .env
echo "GEMINI_API_KEY=your_gemini_api_key_here" >> .env
echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
echo "YOUTUBE_API_KEY=your_youtube_api_key_here" >> .env
```

### 2. Start API Server
```bash
python3 start_api.py
```

### 3. Test with Web Interface
```bash
open web_test.html
```

### 4. Chrome Extension (Optional)
1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked" and select the `chrome_extension` folder
4. Navigate to any YouTube video page
5. Click the nsfK? extension icon to analyze the video

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
  "safety_score": 85,
  "recommendation": "Safe",
  "category_scores": {
    "Non-Violence": 18,
    "Appropriate Language": 15,
    "Non-Scary Content": 16,
    "Family-Friendly Content": 15,
    "Substance-Free": 10,
    "Safe Behavior": 10,
    "Educational Value": 1
  },
  "summary": "Educational TED Talk appropriate for 10-year-olds with positive messaging about perseverance...",
  "keywords": ["Education", "Motivation", "Learning", "Perseverance", "TED"],
  "comment_analysis": "Mixed sentiment with positive educational reactions. No safety concerns detected for 10-year-olds.",
  "channel_name": "TED",
  "web_reputation": "TED content is generally family-friendly, focusing on educational talks. Overall rating: Safe."
}
```

## 📁 Project Structure

```
NSFK/
├── nsfk_analyzer_quality_optimized.py  # Main optimized analysis engine
├── nsfk_analyzer.py                    # Original analysis engine
├── youtube_downloader.py               # YouTube video download functionality
├── api.py                              # FastAPI web service
├── start_api.py                        # API server startup script
├── web_test.html                       # Enhanced web testing interface
├── requirements.txt                    # Python dependencies
├── .env                                # Environment variables (API keys)
├── chrome_extension/                   # Chrome browser extension
│   ├── manifest.json                   # Extension configuration
│   ├── popup.html                      # Extension popup interface
│   ├── popup.js                        # Popup functionality
│   ├── popup.css                       # Popup styling
│   ├── background.js                   # Background service worker
│   ├── content.js                      # YouTube page integration
│   └── icons/                          # Extension icons
├── reports/                            # Analysis report outputs
├── speed_optimization_analysis.md      # Performance optimization documentation
└── docs/                               # API documentation
```

## 🎯 Minimal Setup Requirements

For basic functionality, you only need these essential files:

### Core Files (Required)
```
NSFK/
├── nsfk_analyzer_quality_optimized.py  # ⭐ Main analysis engine
├── youtube_downloader.py               # ⭐ YouTube downloader
├── api.py                              # ⭐ REST API server
├── start_api.py                        # ⭐ Server launcher
├── requirements.txt                    # ⭐ Dependencies
└── .env                                # ⭐ API keys (create this)
```

### Chrome Extension (Optional)
```
chrome_extension/
├── manifest.json                       # Extension config
├── popup.html                          # UI interface
├── popup.js                            # Frontend logic
├── popup.css                           # Styling
├── background.js                       # Background worker
├── content.js                          # YouTube integration
└── icons/                              # Icons (16px, 32px, 128px)
```

### Additional Files (Enhanced Experience)
```
├── web_test.html                       # Web testing interface
├── nsfk_analyzer.py                    # Original analyzer (backup)
└── docker-compose.yml                  # Docker deployment
```

## 🔧 Tech Stack

- **AI/ML**: 
  - **Primary**: Meta Llama-4-Maverick-17B-128E-Instruct-FP8 via GMI API (1M token context, FP8 optimization)
  - **Fallback**: Google Gemini 2.0 Flash for video frame analysis
  - **Secondary**: OpenAI GPT-4o-mini for specialized analysis
- **Video Processing**: yt-dlp, OpenCV, PySceneDetect
- **Audio Analysis**: Whisper for transcription (currently disabled for speed)
- **YouTube Integration**: YouTube Data API v3 for comments and channel data
- **API Framework**: FastAPI with async support
- **Frontend**: Enhanced HTML/CSS/JavaScript with additional analysis sections
- **Deployment**: Docker support included

## 🧠 AI Model Details

### Primary Model: Llama-4-Maverick-17B-128E-Instruct-FP8
- **Provider**: GMI API (https://api.gmi-serving.com)
- **Context Window**: 1,048,576 tokens (1M tokens)
- **Optimization**: FP8 quantization for faster inference
- **Use Cases**: Comprehensive safety report generation, comment analysis, channel reputation assessment
- **Advantages**: Large context window, cost-effective, specialized for instruction-following

### Fallback Models
- **Gemini 2.0 Flash**: Image analysis (when multimodal GMI models timeout)
- **OpenAI GPT-4o-mini**: Secondary option for comment/reputation analysis

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

## 🎨 Chrome Extension Features

The included Chrome extension provides seamless YouTube integration:

- **🚀 One-Click Analysis** - Analyze videos directly from YouTube pages
- **🎯 Real-time Detection** - Automatically detects when you visit YouTube videos
- **📊 In-browser Results** - View safety scores and recommendations in a popup
- **💾 Smart Caching** - Remembers analysis results to avoid duplicate processing
- **⚡ Fast Integration** - Works with your local API server via ngrok tunnel
- **🛡️ Privacy-Focused** - All analysis happens on your local machine

### Extension Installation
1. Ensure your API server is running (`python3 start_api.py`)
2. Load the extension in Chrome Developer Mode
3. Visit any YouTube video page
4. Click the nsfK? extension icon to get instant safety analysis

## 🔮 Future Development
- **Audio Transcription Re-enabling** - Restore Whisper-based audio analysis
- **Enhanced GMI Integration** - Leverage Llama-4-Scout multimodal capabilities for image analysis
- **Reddit/Wiki Integration** - Additional context from community discussions
- **Batch Processing** - Analyze multiple videos simultaneously
- **Custom Age Targeting** - Support for different age groups beyond 10-year-olds
- **Advanced Comment Filtering** - More sophisticated comment safety analysis
- **Real-time Analysis** - Live analysis during video playback
- **Chrome Extension Distribution** - Publish to Chrome Web Store
- **Mobile App** - Native mobile application for on-the-go analysis

## 🤝 Contributing

This project was developed during the Weekend Agent Hack. Contributions and improvements are welcome!

## 📄 License

MIT License - see LICENSE file for details.

---

*Built with ❤️ using Claude Code at Weekend Agent Hack - Mountain View, June 14th, 2025*
