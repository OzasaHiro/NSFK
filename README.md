# NSFK? (Not Safe For Kids?)
**AI-Powered YouTube Video Safety Analyzer for Children 10 and Below**

*Weekend Agent Hack - Mountain View, June 14th, 2025*

## 🎯 Project Overview

NSFK is an AI-powered tool that analyzes YouTube videos to help busy parents determine if content is appropriate for children 10 and below. The system provides comprehensive safety assessments with detailed category-based scoring, comment analysis, channel reputation evaluation, and audio transcript analysis. Features a Chrome extension with intelligent caching and a local REST API for seamless integration.

## ✨ Key Features

### 🔍 Core Analysis Capabilities
- **Comprehensive Video Analysis** - Downloads and analyzes video frames with scene change detection
- **Audio Transcript Analysis** - Whisper-based audio transcription and safety assessment
- **Comment Analysis** - YouTube comments evaluation for safety concerns
- **Channel Reputation** - Web-based channel/creator reputation assessment
- **Dynamic Scoring System** - Weighted component scoring with real-time adjustments

### 🎯 Safety Scoring System (Out of 10 Scale)
- **Violence** (10 pts max) - Absence of violence, weapons, fighting
- **Language** (10 pts max) - Clean, family-friendly language
- **Scary Content** (10 pts max) - No horror, jump scares, frightening imagery
- **Sexual Content** (10 pts max) - No inappropriate themes
- **Substance Use** (10 pts max) - No drugs, alcohol, smoking
- **Dangerous Behavior** (10 pts max) - No risky activities kids might imitate
- **Educational Value** (10 pts max) - Positive learning content

### 🤖 Multi-Model AI Architecture
- **Primary**: Meta Llama-4-Maverick-17B-128E-Instruct-FP8 via GMI API
- **Fallback**: Gemini 1.5 Flash for rate limit scenarios
- **Audio**: OpenAI Whisper for transcription
- **Comments/Web**: OpenAI GPT models for text analysis

### 🎨 Chrome Extension Features
- **Smart Caching** - 1-hour cache with debugging capabilities
- **Visual Indicators** - Red highlighting for category scores below 5
- **Progress Tracking** - Real-time analysis progress with stage updates
- **Cache Status** - Shows "from cache" vs fresh analysis
- **Compact Layout** - 2-3 categories per row for space efficiency

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
  "recommendation": "Safe for children 10 and below",
  "title": "Educational TED Talk - How to Build Confidence",
  "channel_name": "TED",
  "duration": 720,
  "category_scores": {
    "Violence": 9,
    "Language": 10,
    "Scary Content": 9,
    "Sexual Content": 10,
    "Substance Use": 10,
    "Dangerous Behavior": 9,
    "Educational Value": 8
  },
  "dynamic_scoring": {
    "final_score": 85,
    "component_scores": {
      "video_content": 88,
      "audio_transcript": 90,
      "category_analysis": 82,
      "comments": 75,
      "web_reputation": 95
    },
    "weights_used": {
      "video_content": 0.3,
      "audio_transcript": 0.2,
      "category_analysis": 0.25,
      "comments": 0.15,
      "web_reputation": 0.1
    }
  },
  "summary": "Educational TED Talk appropriate for children with positive messaging about confidence building...",
  "keywords": ["Education", "Confidence", "Learning", "TED"],
  "comment_analysis": "Positive educational reactions. No safety concerns detected.",
  "web_reputation": "TED content is family-friendly, focusing on educational talks.",
  "audio_transcript": "Today I want to talk about building confidence in young people...",
  "analysis_timestamp": "2025-06-15T10:30:45Z",
  "processing_time": 45.2
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

- **AI/ML Multi-Model Architecture**:
  - **Video Analysis**: Gemini 2.0 Flash Experimental (vision processing, frame analysis)
  - **Text Analysis**: Gemini 1.5 Flash (comments & reputation - separate rate limits)
  - **Primary Fallback**: Meta Llama-4-Maverick-17B-128E-Instruct-FP8 via GMI API (1M token context)
  - **Rate Limit Optimization**: Dual-model system prevents API conflicts
- **Video Processing**: yt-dlp, OpenCV, PySceneDetect
- **Audio Analysis**: Whisper for transcription (currently disabled for speed)
- **YouTube Integration**: YouTube Data API v3 for comments and channel data
- **API Framework**: FastAPI with async support and comprehensive error handling
- **Frontend**: Enhanced HTML/CSS/JavaScript with bullet point formatting and timing display
- **Chrome Extension**: Modern popup with progress bars, compact category layout, and real-time feedback
- **Deployment**: Docker support included

## 🧠 AI Model Details

### Multi-Model Architecture (Optimized for Quality & Speed)

#### Primary Video Analysis
- **Model**: Meta Llama-4-Maverick-17B-128E-Instruct-FP8
- **Provider**: GMI API (https://api.gmi-serving.com)
- **Context Window**: 1,048,576 tokens (1M tokens)
- **Use Cases**: Primary video frame analysis, comprehensive safety assessment
- **Advantages**: Large context window, multimodal capabilities, cost-effective
- **Rate Limits**: High throughput with intelligent batching

#### Fallback System
- **Model**: Gemini 1.5 Flash
- **Provider**: Google AI (generativelanguage.googleapis.com)
- **Use Cases**: Fallback when GMI API hits rate limits
- **Advantages**: Reliable fallback, good vision capabilities
- **Rate Limits**: Separate pool from primary analysis

#### Audio Transcription
- **Model**: OpenAI Whisper (base model)
- **Provider**: Local processing
- **Use Cases**: Audio content transcription and analysis
- **Advantages**: High accuracy, local processing, no API costs

#### Text Analysis (Comments & Web Reputation)
- **Model**: OpenAI GPT models
- **Provider**: OpenAI API
- **Use Cases**: Comment analysis, channel reputation assessment
- **Advantages**: Excellent text understanding, specialized for safety analysis

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

The Chrome extension provides seamless YouTube integration with advanced caching and visual feedback:

### 🚀 Core Features
- **One-Click Analysis** - Analyze videos directly from YouTube pages
- **Smart Caching System** - 1-hour cache with size limits (100 entries max)
- **Cache Debugging** - Built-in cache status checking and logging
- **Real-time Detection** - Automatically detects YouTube video navigation
- **Local API Integration** - Works with local API server (no external dependencies)

### 🎯 Visual Enhancements
- **Progress Bar Interface** - Real-time progress with stage-specific feedback
- **Category Score Highlighting** - Scores below 5 highlighted in red for safety concerns
- **Cache Indicators** - Shows "from cache" vs fresh analysis timing
- **Compact Layout** - 2-3 categories per row for optimal space usage
- **Formatted Text Display** - Proper bullet points for analysis sections

### 💾 Caching System
- **Duration**: 1 hour per video analysis
- **Size Limit**: 100 cached analyses maximum
- **Auto-Cleanup**: Periodic cleanup of expired entries
- **Debug Functions**: `nsfkDebug.checkCacheStatus()` for troubleshooting
- **Cache Indicators**: Visual feedback when results are loaded from cache

### Extension Installation
1. Ensure your API server is running (`python3 start_api.py`)
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode" (toggle in top right)
4. Click "Load unpacked" and select the `chrome_extension` folder
5. Visit any YouTube video page
6. Click the nsfK? extension icon to analyze

### Debug Features
- **Cache Status**: Check cache contents with browser console
- **Detailed Logging**: Background script logs cache hits/misses
- **Performance Timing**: Shows analysis vs cache retrieval times
- **Error Handling**: Comprehensive error reporting and retry logic

## 🔮 Future Development
- **Audio Transcription Re-enabling** - Restore Whisper-based audio analysis
- **Enhanced GMI Integration** - Leverage Llama-4-Scout multimodal capabilities for image analysis
- **Reddit/Wiki Integration** - Additional context from community discussions
- **Batch Processing** - Analyze multiple videos simultaneously
- **Custom Age Targeting** - Support for different age groups beyond children 10 and below
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
