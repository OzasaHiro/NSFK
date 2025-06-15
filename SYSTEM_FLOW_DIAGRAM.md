# NSFK System Architecture Flow Diagram

## 🏗️ System Overview

```mermaid
graph TB
    subgraph "User Interface Layer"
        CE[Chrome Extension]
        WI[Web Interface]
        API_DOCS[API Documentation]
    end
    
    subgraph "API Layer"
        API[FastAPI Server<br/>Port 8000]
        BG[Background Worker]
        CACHE[Analysis Cache<br/>1 hour TTL]
    end
    
    subgraph "Core Analysis Engine"
        ANALYZER[NSFK Quality Analyzer]
        DOWNLOADER[YouTube Downloader]
        FRAME_EXT[Frame Extractor]
        AUDIO_EXT[Audio Extractor]
    end
    
    subgraph "AI Models"
        LLAMA[Llama-4-Maverick<br/>GMI API<br/>Primary]
        GEMINI[Gemini 1.5 Flash<br/>Fallback]
        WHISPER[Whisper<br/>Audio Transcription]
        OPENAI[OpenAI GPT<br/>Comments/Web]
    end
    
    subgraph "External APIs"
        YT_API[YouTube Data API<br/>Comments]
        YT_DL[YouTube Video<br/>Download]
        WEB[Web Search<br/>Channel Reputation]
    end
    
    subgraph "Output"
        REPORTS[JSON/TXT Reports]
        RESPONSE[API Response]
        UI_DISPLAY[UI Display]
    end
    
    %% User interactions
    CE --> API
    WI --> API
    
    %% API processing
    API --> CACHE
    CACHE -.->|Cache Hit| RESPONSE
    API --> ANALYZER
    
    %% Analysis flow
    ANALYZER --> DOWNLOADER
    DOWNLOADER --> YT_DL
    ANALYZER --> FRAME_EXT
    ANALYZER --> AUDIO_EXT
    
    %% AI processing
    FRAME_EXT --> LLAMA
    LLAMA -.->|Rate Limit| GEMINI
    AUDIO_EXT --> WHISPER
    ANALYZER --> OPENAI
    
    %% External data
    ANALYZER --> YT_API
    ANALYZER --> WEB
    
    %% Output generation
    ANALYZER --> REPORTS
    ANALYZER --> RESPONSE
    RESPONSE --> UI_DISPLAY
    
    %% Cache storage
    ANALYZER --> CACHE
    
    %% Styling
    classDef userLayer fill:#e1f5fe
    classDef apiLayer fill:#f3e5f5
    classDef coreLayer fill:#e8f5e8
    classDef aiLayer fill:#fff3e0
    classDef externalLayer fill:#fce4ec
    classDef outputLayer fill:#f1f8e9
    
    class CE,WI,API_DOCS userLayer
    class API,BG,CACHE apiLayer
    class ANALYZER,DOWNLOADER,FRAME_EXT,AUDIO_EXT coreLayer
    class LLAMA,GEMINI,WHISPER,OPENAI aiLayer
    class YT_API,YT_DL,WEB externalLayer
    class REPORTS,RESPONSE,UI_DISPLAY outputLayer
```

## 🔄 Detailed Analysis Flow

```mermaid
sequenceDiagram
    participant User
    participant Extension as Chrome Extension
    participant API as FastAPI Server
    participant Cache as Analysis Cache
    participant Analyzer as NSFK Analyzer
    participant YouTube as YouTube APIs
    participant AI as AI Models
    
    User->>Extension: Click Analyze Button
    Extension->>API: POST /analyze {url}
    
    API->>Cache: Check cache for video ID
    alt Cache Hit
        Cache-->>API: Return cached analysis
        API-->>Extension: Cached results (source: cache)
        Extension-->>User: Display results (from cache)
    else Cache Miss
        API->>Analyzer: Start analysis
        
        par Video Download
            Analyzer->>YouTube: Download video
            YouTube-->>Analyzer: Video file
        and Metadata Fetch
            Analyzer->>YouTube: Get video metadata
            YouTube-->>Analyzer: Title, channel, duration
        end
        
        par Frame Analysis
            Analyzer->>AI: Analyze video frames
            AI-->>Analyzer: Safety scores per frame
        and Audio Analysis
            Analyzer->>AI: Transcribe & analyze audio
            AI-->>Analyzer: Audio safety score
        and Comment Analysis
            Analyzer->>YouTube: Fetch comments
            YouTube-->>Analyzer: Comment data
            Analyzer->>AI: Analyze comments
            AI-->>Analyzer: Comment safety score
        and Web Reputation
            Analyzer->>AI: Analyze channel reputation
            AI-->>Analyzer: Channel safety score
        end
        
        Analyzer->>Analyzer: Calculate dynamic score
        Analyzer->>Cache: Store analysis result
        Analyzer-->>API: Analysis complete
        API-->>Extension: Fresh results (source: analysis)
        Extension-->>User: Display results (with timing)
    end
```

## 🎯 Chrome Extension Detailed Flow

```mermaid
graph TD
    subgraph "Content Script (content.js)"
        DETECT[Video Detection]
        META[Metadata Extraction]
        NOTIFY[Notify Background]
    end
    
    subgraph "Background Script (background.js)"
        CACHE_CHECK[Cache Check]
        API_CALL[API Request]
        CACHE_STORE[Cache Storage]
        BADGE[Badge Update]
    end
    
    subgraph "Popup (popup.js)"
        UI[User Interface]
        PROGRESS[Progress Bar]
        RESULTS[Results Display]
        DEBUG[Debug Functions]
    end
    
    subgraph "Cache System"
        MEMORY[In-Memory Map]
        TTL[1 Hour TTL]
        CLEANUP[Periodic Cleanup]
        SIZE_LIMIT[100 Entry Limit]
    end
    
    %% Flow connections
    DETECT --> META
    META --> NOTIFY
    NOTIFY --> CACHE_CHECK
    
    CACHE_CHECK -.->|Hit| RESULTS
    CACHE_CHECK -->|Miss| API_CALL
    API_CALL --> CACHE_STORE
    API_CALL --> RESULTS
    
    UI --> PROGRESS
    PROGRESS --> RESULTS
    
    CACHE_STORE --> MEMORY
    MEMORY --> TTL
    TTL --> CLEANUP
    MEMORY --> SIZE_LIMIT
    
    API_CALL --> BADGE
    CACHE_CHECK --> BADGE
    
    %% Debug connections
    DEBUG -.-> MEMORY
    DEBUG -.-> TTL
    
    %% Styling
    classDef contentScript fill:#e3f2fd
    classDef background fill:#f3e5f5
    classDef popup fill:#e8f5e8
    classDef cache fill:#fff3e0
    
    class DETECT,META,NOTIFY contentScript
    class CACHE_CHECK,API_CALL,CACHE_STORE,BADGE background
    class UI,PROGRESS,RESULTS,DEBUG popup
    class MEMORY,TTL,CLEANUP,SIZE_LIMIT cache
```

## 🤖 AI Model Processing Flow

```mermaid
graph LR
    subgraph "Video Analysis Pipeline"
        FRAMES[Video Frames<br/>261 frames extracted]
        BATCH[Batch Processing<br/>20 frames per batch]
        LLAMA_V[Llama-4-Maverick<br/>Primary Vision Model]
        GEMINI_V[Gemini 1.5 Flash<br/>Fallback Model]
        FRAME_SCORES[Frame Safety Scores]
    end

    subgraph "Audio Analysis Pipeline"
        AUDIO[Audio Track]
        WHISPER_T[Whisper Transcription]
        TRANSCRIPT[Text Transcript]
        LLAMA_T[Llama-4 Text Analysis]
        AUDIO_SCORE[Audio Safety Score]
    end

    subgraph "Text Analysis Pipeline"
        COMMENTS[YouTube Comments]
        CHANNEL[Channel Info]
        OPENAI_C[OpenAI Comment Analysis]
        OPENAI_W[OpenAI Web Reputation]
        TEXT_SCORES[Text Safety Scores]
    end

    subgraph "Dynamic Scoring Engine"
        WEIGHTS[Component Weights<br/>Video: 30%<br/>Audio: 20%<br/>Categories: 25%<br/>Comments: 15%<br/>Web: 10%]
        COMBINE[Score Combination]
        ADJUSTMENTS[Safety Adjustments]
        FINAL[Final Safety Score]
    end

    %% Video flow
    FRAMES --> BATCH
    BATCH --> LLAMA_V
    LLAMA_V -.->|Rate Limit| GEMINI_V
    LLAMA_V --> FRAME_SCORES
    GEMINI_V --> FRAME_SCORES

    %% Audio flow
    AUDIO --> WHISPER_T
    WHISPER_T --> TRANSCRIPT
    TRANSCRIPT --> LLAMA_T
    LLAMA_T --> AUDIO_SCORE

    %% Text flow
    COMMENTS --> OPENAI_C
    CHANNEL --> OPENAI_W
    OPENAI_C --> TEXT_SCORES
    OPENAI_W --> TEXT_SCORES

    %% Scoring flow
    FRAME_SCORES --> WEIGHTS
    AUDIO_SCORE --> WEIGHTS
    TEXT_SCORES --> WEIGHTS
    WEIGHTS --> COMBINE
    COMBINE --> ADJUSTMENTS
    ADJUSTMENTS --> FINAL

    %% Styling
    classDef video fill:#e3f2fd
    classDef audio fill:#f3e5f5
    classDef text fill:#e8f5e8
    classDef scoring fill:#fff3e0

    class FRAMES,BATCH,LLAMA_V,GEMINI_V,FRAME_SCORES video
    class AUDIO,WHISPER_T,TRANSCRIPT,LLAMA_T,AUDIO_SCORE audio
    class COMMENTS,CHANNEL,OPENAI_C,OPENAI_W,TEXT_SCORES text
    class WEIGHTS,COMBINE,ADJUSTMENTS,FINAL scoring
```

## 📊 Category Scoring System

```mermaid
graph TD
    subgraph "Category Analysis (Out of 10)"
        VIOLENCE[Violence<br/>Weapons, Fighting]
        LANGUAGE[Language<br/>Profanity, Inappropriate]
        SCARY[Scary Content<br/>Horror, Jump Scares]
        SEXUAL[Sexual Content<br/>Inappropriate Themes]
        SUBSTANCE[Substance Use<br/>Drugs, Alcohol]
        DANGEROUS[Dangerous Behavior<br/>Risky Activities]
        EDUCATIONAL[Educational Value<br/>Learning Content]
    end

    subgraph "Visual Indicators"
        NORMAL[Score ≥ 5<br/>Blue Color]
        WARNING[Score < 5<br/>Red Highlight]
    end

    subgraph "Final Recommendation"
        SAFE[80-100: Safe<br/>✅ Green]
        REVIEW[60-79: Review Required<br/>⚠️ Yellow]
        UNSAFE[0-59: Not Recommended<br/>❌ Red]
    end

    %% Category connections
    VIOLENCE --> NORMAL
    VIOLENCE --> WARNING
    LANGUAGE --> NORMAL
    LANGUAGE --> WARNING
    SCARY --> NORMAL
    SCARY --> WARNING
    SEXUAL --> NORMAL
    SEXUAL --> WARNING
    SUBSTANCE --> NORMAL
    SUBSTANCE --> WARNING
    DANGEROUS --> NORMAL
    DANGEROUS --> WARNING
    EDUCATIONAL --> NORMAL
    EDUCATIONAL --> WARNING

    %% Recommendation flow
    NORMAL --> SAFE
    NORMAL --> REVIEW
    WARNING --> REVIEW
    WARNING --> UNSAFE

    %% Styling
    classDef category fill:#e8f5e8
    classDef indicator fill:#fff3e0
    classDef recommendation fill:#f3e5f5

    class VIOLENCE,LANGUAGE,SCARY,SEXUAL,SUBSTANCE,DANGEROUS,EDUCATIONAL category
    class NORMAL,WARNING indicator
    class SAFE,REVIEW,UNSAFE recommendation
```

## 🔧 Technical Implementation Details

### API Endpoints
- **POST /analyze** - Main analysis endpoint
- **GET /health** - Health check and status
- **GET /docs** - Interactive API documentation
- **GET /** - API information and endpoints

### Cache Implementation
- **Storage**: In-memory JavaScript Map
- **TTL**: 1 hour (3,600,000 ms)
- **Size Limit**: 100 entries (LRU eviction)
- **Cleanup**: Every 10 minutes
- **Debug**: `nsfkDebug.checkCacheStatus()`

### Error Handling
- **Retry Logic**: 2 attempts with 5-second delays
- **Fallback Models**: Gemini when Llama rate-limited
- **Timeout Handling**: 4-minute API timeout
- **Network Errors**: Graceful degradation

### Performance Optimizations
- **Concurrent Processing**: Parallel frame analysis
- **Batch Processing**: 20 frames per batch
- **Scene Detection**: Smart frame selection
- **Rate Limit Management**: Model switching
- **Caching**: Aggressive result caching

## 🚀 Deployment Architecture

```mermaid
graph TB
    subgraph "Local Development"
        DEV_API[Local API Server<br/>127.0.0.1:8000]
        DEV_EXT[Chrome Extension<br/>Developer Mode]
        DEV_FILES[Local Files<br/>Videos, Reports]
    end

    subgraph "External Services"
        GMI[GMI API<br/>Llama-4-Maverick]
        GOOGLE[Google AI<br/>Gemini Models]
        OPENAI_API[OpenAI API<br/>GPT Models]
        YOUTUBE_API[YouTube Data API<br/>Comments & Metadata]
    end

    subgraph "Optional Cloud Deployment"
        DOCKER[Docker Container]
        CLOUD[Cloud Platform<br/>Railway/Heroku/GCP]
        CDN[Content Delivery<br/>Static Assets]
    end

    %% Local connections
    DEV_EXT --> DEV_API
    DEV_API --> DEV_FILES

    %% External API connections
    DEV_API --> GMI
    DEV_API --> GOOGLE
    DEV_API --> OPENAI_API
    DEV_API --> YOUTUBE_API

    %% Cloud deployment
    DEV_API -.->|Optional| DOCKER
    DOCKER -.-> CLOUD
    CLOUD -.-> CDN

    %% Styling
    classDef local fill:#e8f5e8
    classDef external fill:#fff3e0
    classDef cloud fill:#f3e5f5

    class DEV_API,DEV_EXT,DEV_FILES local
    class GMI,GOOGLE,OPENAI_API,YOUTUBE_API external
    class DOCKER,CLOUD,CDN cloud
```

## 📈 System Metrics & Monitoring

### Performance Metrics
- **Analysis Time**: 30-120 seconds per video
- **Cache Hit Rate**: ~80% for repeated videos
- **API Success Rate**: >95% with fallback models
- **Frame Processing**: 261 frames per video average
- **Batch Efficiency**: 20 frames per batch optimal

### Quality Metrics
- **Model Accuracy**: Llama-4-Maverick primary (99.6% usage)
- **Fallback Usage**: Gemini <1% (rate limit scenarios)
- **Audio Transcription**: Whisper base model
- **Comment Coverage**: YouTube Data API v3

### Resource Usage
- **Memory**: ~2GB peak during analysis
- **Storage**: Temporary video files (auto-cleanup)
- **Network**: 8MB average per video download
- **CPU**: Intensive during frame extraction

---

## 🎯 Key System Benefits

1. **Intelligent Caching** - Reduces analysis time from minutes to seconds
2. **Multi-Model Fallback** - Ensures high availability despite rate limits
3. **Real-time Feedback** - Progress tracking and visual indicators
4. **Local Processing** - Privacy-focused with local API deployment
5. **Comprehensive Analysis** - Video, audio, comments, and reputation
6. **Parent-Friendly** - Simple scoring system with clear recommendations
