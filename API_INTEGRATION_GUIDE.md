# NSFK API Integration Guide

This guide explains how to integrate the NSFK Video Analyzer API into web applications, Chrome extensions, or other services.

## Quick Start

1. **Start the API server:**
```bash
python3 start_api.py
```

2. **Test the API:**
```bash
python3 test_api.py
```

3. **Access the documentation:**
- Open http://localhost:8000/docs in your browser

## Integration Examples

### Chrome Extension Integration

For the NSFK Chrome extension, the API provides a clean interface:

```javascript
// In your Chrome extension content script
async function analyzeCurrentVideo() {
    const videoUrl = window.location.href;
    
    if (!videoUrl.includes('youtube.com/watch')) {
        return;
    }
    
    try {
        const response = await fetch('http://localhost:8000/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: videoUrl })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const analysis = await response.json();
        
        // Display results in the extension UI
        displaySafetyResults(analysis);
        
    } catch (error) {
        console.error('Analysis failed:', error);
        displayError('Unable to analyze this video');
    }
}

function displaySafetyResults(analysis) {
    const safetyScore = analysis.safety_score;
    const recommendation = analysis.recommendation;
    const summary = analysis.summary;
    
    // Create or update the safety indicator
    const indicator = document.getElementById('nsfk-indicator') || 
                     document.createElement('div');
    
    indicator.id = 'nsfk-indicator';
    indicator.innerHTML = `
        <div class="nsfk-score ${getScoreClass(safetyScore)}">
            Safety Score: ${safetyScore}/100
        </div>
        <div class="nsfk-recommendation">
            ${recommendation}
        </div>
        <div class="nsfk-summary">
            ${summary}
        </div>
    `;
    
    // Add to YouTube page
    const target = document.querySelector('#primary-inner') || 
                  document.querySelector('#player');
    if (target && !document.getElementById('nsfk-indicator')) {
        target.insertBefore(indicator, target.firstChild);
    }
}

function getScoreClass(score) {
    if (score >= 80) return 'safe';
    if (score >= 60) return 'warning';
    return 'danger';
}
```

### Web Application Integration

For web applications, you can use the API with modern frameworks:

#### React Example
```jsx
import React, { useState } from 'react';

function VideoAnalyzer() {
    const [url, setUrl] = useState('');
    const [analysis, setAnalysis] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    
    const analyzeVideo = async () => {
        if (!url) return;
        
        setLoading(true);
        setError(null);
        
        try {
            const response = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Analysis failed');
            }
            
            const result = await response.json();
            setAnalysis(result);
            
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <div className="video-analyzer">
            <div className="input-section">
                <input
                    type="url"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="Enter YouTube URL..."
                    disabled={loading}
                />
                <button 
                    onClick={analyzeVideo}
                    disabled={loading || !url}
                >
                    {loading ? 'Analyzing...' : 'Analyze Video'}
                </button>
            </div>
            
            {error && (
                <div className="error">
                    Error: {error}
                </div>
            )}
            
            {analysis && (
                <div className="results">
                    <h3>{analysis.title}</h3>
                    <div className={`safety-score score-${getSafetyLevel(analysis.safety_score)}`}>
                        Safety Score: {analysis.safety_score}/100
                    </div>
                    <div className="recommendation">
                        Recommendation: {analysis.recommendation}
                    </div>
                    <div className="summary">
                        {analysis.summary}
                    </div>
                    <div className="keywords">
                        Keywords: {analysis.keywords.join(', ')}
                    </div>
                </div>
            )}
        </div>
    );
}

function getSafetyLevel(score) {
    if (score >= 80) return 'safe';
    if (score >= 60) return 'warning';
    return 'danger';
}

export default VideoAnalyzer;
```

### Backend Service Integration

For server-to-server integration:

#### Python Example
```python
import asyncio
import aiohttp
from typing import Dict, Any

class NSFKClient:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
    
    async def analyze_video(self, youtube_url: str) -> Dict[str, Any]:
        """Analyze a YouTube video for safety"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base_url}/analyze",
                json={"url": youtube_url}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.json()
                    raise Exception(f"Analysis failed: {error_data.get('error', 'Unknown error')}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_base_url}/health") as response:
                return await response.json()

# Usage
async def main():
    client = NSFKClient()
    
    # Check if API is healthy
    health = await client.health_check()
    print(f"API Status: {health['status']}")
    
    # Analyze a video
    result = await client.analyze_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print(f"Safety Score: {result['safety_score']}/100")
    print(f"Recommendation: {result['recommendation']}")

if __name__ == "__main__":
    asyncio.run(main())
```

#### Node.js Example
```javascript
const axios = require('axios');

class NSFKClient {
    constructor(apiBaseUrl = 'http://localhost:8000') {
        this.apiBaseUrl = apiBaseUrl;
    }
    
    async analyzeVideo(youtubeUrl) {
        try {
            const response = await axios.post(`${this.apiBaseUrl}/analyze`, {
                url: youtubeUrl
            });
            return response.data;
        } catch (error) {
            if (error.response) {
                throw new Error(`Analysis failed: ${error.response.data.error}`);
            }
            throw error;
        }
    }
    
    async healthCheck() {
        const response = await axios.get(`${this.apiBaseUrl}/health`);
        return response.data;
    }
}

// Usage
async function main() {
    const client = new NSFKClient();
    
    try {
        // Check API health
        const health = await client.healthCheck();
        console.log(`API Status: ${health.status}`);
        
        // Analyze a video
        const result = await client.analyzeVideo('https://www.youtube.com/watch?v=dQw4w9WgXcQ');
        console.log(`Safety Score: ${result.safety_score}/100`);
        console.log(`Recommendation: ${result.recommendation}`);
        
    } catch (error) {
        console.error('Error:', error.message);
    }
}

main();
```

## Deployment Options

### Development
```bash
python3 start_api.py
```

### Production with Uvicorn
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or with Docker directly
docker build -t nsfk-api .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key_here nsfk-api
```

### Cloud Deployment

The API is designed to work with cloud platforms:

- **Railway/Heroku**: Use the included `Dockerfile`
- **AWS Lambda**: Can be adapted with Mangum
- **Google Cloud Run**: Direct Docker deployment
- **Azure Container Instances**: Direct Docker deployment

## Error Handling Best Practices

Always handle these common scenarios:

1. **API unavailable** (network errors)
2. **Invalid URLs** (422 validation errors)
3. **Analysis failures** (500 server errors)
4. **Rate limiting** (429 errors)
5. **Long processing times** (timeouts)

```javascript
async function robustAnalyze(url, maxRetries = 3) {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            const response = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url }),
                timeout: 300000 // 5 minute timeout
            });
            
            if (response.ok) {
                return await response.json();
            }
            
            if (response.status === 422) {
                // Validation error - don't retry
                const error = await response.json();
                throw new Error(`Invalid URL: ${error.error}`);
            }
            
            if (attempt === maxRetries) {
                throw new Error(`Analysis failed after ${maxRetries} attempts`);
            }
            
            // Wait before retry
            await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
            
        } catch (error) {
            if (attempt === maxRetries) {
                throw error;
            }
        }
    }
}
```

## Performance Considerations

- **Video analysis takes 1-5 minutes** - implement proper loading states
- **Use caching** - cache results for the same video
- **Implement timeouts** - don't let requests hang indefinitely
- **Queue long tasks** - for high-traffic applications, consider job queues
- **Monitor API health** - regularly check the `/health` endpoint

## Security Notes

- The API includes CORS support for web clients
- In production, configure CORS origins appropriately
- Consider rate limiting for public deployments
- Secure your Gemini API key (never expose in client-side code)
- Use HTTPS in production environments