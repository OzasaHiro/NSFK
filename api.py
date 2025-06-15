#!/usr/bin/env python3
"""
NSFK Video Analyzer FastAPI Application
Provides REST API endpoints for YouTube video safety analysis
"""

import os
import logging
from datetime import datetime
from typing import Dict, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import uvicorn

# Import the NSFK analyzer (quality optimized version)
from nsfk_analyzer_quality_optimized import QualityPreservingNSFKAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global analyzer instance
analyzer = None

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan manager"""
    global analyzer
    
    # Startup
    logger.info("Starting NSFK API server...")
    analyzer = QualityPreservingNSFKAnalyzer()
    
    yield
    
    # Shutdown
    logger.info("Shutting down NSFK API server...")


# Create FastAPI app
app = FastAPI(
    title="NSFK Video Analyzer API",
    description="AI-powered YouTube video safety analysis for parents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AnalyzeRequest(BaseModel):
    """Request model for video analysis"""
    url: str
    
    @validator('url')
    def validate_youtube_url(cls, v):
        """Validate that the URL is a YouTube URL"""
        if not v:
            raise ValueError("URL cannot be empty")
        
        # Basic YouTube URL validation
        youtube_domains = [
            'youtube.com', 'www.youtube.com', 'm.youtube.com',
            'youtu.be', 'www.youtu.be'
        ]
        
        if not any(domain in v.lower() for domain in youtube_domains):
            raise ValueError("URL must be a valid YouTube URL")
            
        return v

class CategoryScores(BaseModel):
    """Category scores model"""
    Violence: Optional[int] = 0
    Language: Optional[int] = 0
    ScaryContent: Optional[int] = 0
    SexualContent: Optional[int] = 0
    SubstanceUse: Optional[int] = 0
    DangerousBehavior: Optional[int] = 0
    EducationalValue: Optional[int] = 0

class DynamicScoring(BaseModel):
    """Dynamic scoring breakdown model"""
    final_score: int
    component_scores: Dict[str, int]
    weights_used: Dict[str, float]
    explanation: str
    adjustments_applied: List[str]

class AnalyzeResponse(BaseModel):
    """Response model for video analysis"""
    video_url: str
    title: str
    duration: int
    safety_score: int
    category_scores: Dict[str, int]
    summary: str
    risk_factors: list
    keywords: list
    recommendation: str
    audio_transcript: Optional[str] = None
    comment_analysis: Optional[str] = None
    channel_name: Optional[str] = None
    web_reputation: Optional[str] = None
    dynamic_scoring: Optional[DynamicScoring] = None
    analysis_timestamp: str
    report_path: Optional[str] = None

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: str


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with basic API information"""
    return {
        "name": "NSFK Video Analyzer API",
        "version": "1.0.0",
        "description": "AI-powered YouTube video safety analysis for parents",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global analyzer
    
    # Check if analyzer is initialized
    analyzer_status = "ready" if analyzer is not None else "not_initialized"
    
    # Check environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    env_status = "configured" if gemini_api_key else "missing_api_key"
    
    health_status = {
        "status": "healthy" if analyzer_status == "ready" and env_status == "configured" else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "analyzer": analyzer_status,
        "environment": env_status,
        "version": "1.0.0"
    }
    
    status_code = status.HTTP_200_OK if health_status["status"] == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(content=health_status, status_code=status_code)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_video(request: AnalyzeRequest):
    """
    Analyze a YouTube video for safety content
    
    Args:
        request: AnalyzeRequest containing YouTube URL
        
    Returns:
        AnalyzeResponse with safety analysis results
        
    Raises:
        HTTPException: For various error conditions
    """
    global analyzer
    
    if analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analyzer not initialized"
        )
    
    logger.info(f"Analyzing video: {request.url}")
    
    try:
        # Perform analysis
        result = await analyzer.analyze_youtube_video_quality_optimized(request.url)
        
        # Check if analysis failed
        if 'error' in result:
            error_msg = result['error']
            logger.error(f"Analysis failed: {error_msg}")
            
            # Determine appropriate HTTP status code based on error type
            if "download" in error_msg.lower():
                status_code = status.HTTP_400_BAD_REQUEST
            elif "api" in error_msg.lower() or "quota" in error_msg.lower():
                status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            else:
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                
            raise HTTPException(
                status_code=status_code,
                detail=error_msg
            )
        
        # Return successful analysis
        logger.info(f"Analysis completed successfully. Safety score: {result.get('safety_score', 'N/A')}")
        return AnalyzeResponse(**result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during analysis"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=f"HTTP {exc.status_code}",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(_request, exc):
    """General exception handler for unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


# Development server configuration
def start_server():
    """Start the development server"""
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()