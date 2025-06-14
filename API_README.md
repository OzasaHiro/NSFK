# NSFK Video Analyzer API

A FastAPI application that provides REST endpoints for YouTube video safety analysis.

## Features

- **POST /analyze**: Analyze YouTube videos for child safety
- **GET /health**: Health check endpoint
- **GET /**: API information and available endpoints
- **Async support**: Built with FastAPI and asyncio for efficient processing
- **CORS enabled**: Ready for web client integration
- **Error handling**: Comprehensive error responses with appropriate HTTP status codes
- **Input validation**: YouTube URL validation with helpful error messages

## Installation

1. Make sure you have the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Create .env file with your Gemini API key
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

## Running the API

### Development Server
```bash
python api.py
```

### Production Server
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- **Local**: http://127.0.0.1:8000
- **Documentation**: http://127.0.0.1:8000/docs (Swagger UI)
- **Alternative docs**: http://127.0.0.1:8000/redoc

## API Endpoints

### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "analyzer": "ready",
  "environment": "configured",
  "version": "1.0.0"
}
```

### Analyze Video
```http
POST /analyze
Content-Type: application/json

{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

Response:
```json
{
  "video_url": "https://youtube.com/watch?v=VIDEO_ID",
  "title": "Video Title",
  "duration": 180,
  "safety_score": 85,
  "category_scores": {
    "Violence": 18,
    "Language": 15,
    "Scary Content": 20,
    "Sexual Content": 15,
    "Substance Use": 10,
    "Dangerous Behavior": 7,
    "Educational Value": 0
  },
  "summary": "Educational content about science concepts, suitable for children.",
  "risk_factors": [],
  "keywords": ["education", "science", "learning", "kids", "safe"],
  "recommendation": "Safe",
  "audio_transcript": "Hello and welcome to...",
  "analysis_timestamp": "2024-01-01T12:00:00",
  "report_path": "reports/nsfk_report_20240101_120000.json"
}
```

## Testing

Run the test suite:
```bash
python test_api.py
```

This will test:
- Health check endpoint
- Root endpoint
- Invalid URL handling
- Video analysis functionality

## Error Handling

The API returns appropriate HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid URL, validation errors)
- **422**: Unprocessable Entity (Pydantic validation errors)
- **500**: Internal Server Error
- **503**: Service Unavailable (analyzer not ready, API quota exceeded)

Error response format:
```json
{
  "error": "Error description",
  "detail": "Additional details",
  "timestamp": "2024-01-01T12:00:00"
}
```

## Usage Examples

### cURL
```bash
# Health check
curl http://localhost:8000/health

# Analyze video
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

### Python
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Analyze video
response = requests.post(
    "http://localhost:8000/analyze",
    json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
)
result = response.json()
print(f"Safety Score: {result['safety_score']}/100")
```

### JavaScript/Fetch
```javascript
// Health check
fetch('http://localhost:8000/health')
  .then(response => response.json())
  .then(data => console.log(data));

// Analyze video
fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
  })
})
  .then(response => response.json())
  .then(data => console.log(`Safety Score: ${data.safety_score}/100`));
```

## Performance Notes

- Video analysis can take 1-5 minutes depending on video length
- The API includes timeouts and rate limiting for Gemini API calls
- Large videos (>10 minutes) are sampled for efficiency
- Audio transcription uses Whisper which requires some processing time

## Development

The API is built with:
- **FastAPI**: Modern Python web framework
- **Pydantic**: Data validation and serialization
- **CORS middleware**: Cross-origin request support
- **Uvicorn**: ASGI server for production deployment

For Chrome extension integration, the CORS settings allow all origins (configure appropriately for production).