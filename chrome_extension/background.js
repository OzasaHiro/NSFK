// ===========================================
// background.js - Clean Version with Direct API Mapping
// ===========================================

console.log('nsfK? Background script loaded');

// Extension state
let currentVideoData = null;
let analysisCache = new Map();
let isAnalyzing = false;

// Configuration
const CONFIG = {
    CACHE_DURATION: 60 * 60 * 1000, // 1 hour
    MAX_CACHE_SIZE: 100,
    API_ENDPOINT: 'https://5e45-107-194-242-26.ngrok-free.app/analyze',
    API_TIMEOUT: 240000, // 4 minutes
    RETRY_ATTEMPTS: 2,
    RETRY_DELAY: 5000 // 5 seconds between retries
};

// Listen for messages from content script and popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('nsfK? Background received message:', message.type);
    
    switch (message.type) {
        case 'YOUTUBE_VIDEO_DETECTED':
            handleVideoDetected(message, sender);
            sendResponse({ success: true });
            break;
            
        case 'YOUTUBE_VIDEO_LEFT':
            handleVideoLeft();
            sendResponse({ success: true });
            break;
            
        case 'REQUEST_ANALYSIS':
            handleAnalysisRequest(message, sendResponse);
            return true; // Keep message channel open for async response
            
        case 'GET_CURRENT_VIDEO_DATA':
            sendResponse({
                success: true,
                data: currentVideoData,
                isAnalyzing: isAnalyzing
            });
            break;
            
        case 'CLEAR_CACHE':
            analysisCache.clear();
            sendResponse({ success: true, message: 'Cache cleared' });
            break;
            
        case 'TEST_API':
            testAPIConnection(sendResponse);
            return true;
            
        default:
            console.warn('nsfK? Unknown message type:', message.type);
            sendResponse({ success: false, error: 'Unknown message type' });
    }
});

// Handle video detection from content script
function handleVideoDetected(message, sender) {
    currentVideoData = {
        videoId: message.videoId,
        metadata: message.metadata,
        tabId: sender.tab.id,
        timestamp: message.timestamp,
        url: sender.tab.url
    };
    
    console.log('nsfK? Video detected:', message.videoId, message.metadata.title);
    updateBadge('📺', '#4F46E5');
    
    // Check cache
    const cached = getCachedAnalysis(message.videoId);
    if (cached) {
        console.log('nsfK? Using cached analysis for:', message.videoId);
        currentVideoData.analysis = cached;
        updateBadge('✅', '#10B981');
    }
}

// Handle leaving video page
function handleVideoLeft() {
    currentVideoData = null;
    updateBadge('', '#666666');
    console.log('nsfK? Left video page');
}

// Handle analysis request from popup
async function handleAnalysisRequest(message, sendResponse) {
    if (isAnalyzing) {
        sendResponse({
            success: false,
            error: 'Analysis already in progress'
        });
        return;
    }
    
    const videoUrl = message.videoUrl || currentVideoData?.url;
    let videoId = message.videoId || currentVideoData?.videoId;
    
    if (!videoUrl) {
        sendResponse({
            success: false,
            error: 'No video URL available'
        });
        return;
    }
    
    // Extract video ID from URL if not provided
    if (!videoId && videoUrl) {
        videoId = extractVideoIdFromUrl(videoUrl);
    }
    
    // Check cache first
    if (videoId) {
        const cached = getCachedAnalysis(videoId);
        if (cached) {
            console.log('nsfK? Returning cached analysis for:', videoId);
            sendResponse({
                success: true,
                data: cached,
                source: 'cache'
            });
            return;
        }
    }
    
    // Perform new analysis
    try {
        isAnalyzing = true;
        updateBadge('⏳', '#F59E0B');
        
        const analysisResult = await performAnalysisWithRetry(videoUrl);
        
        // Cache the result
        if (videoId) {
            setCachedAnalysis(videoId, analysisResult);
        }
        
        // Update current video data
        if (currentVideoData && currentVideoData.videoId === videoId) {
            currentVideoData.analysis = analysisResult;
        }
        
        updateBadge('✅', '#10B981');
        
        sendResponse({
            success: true,
            data: analysisResult,
            source: 'analysis'
        });
        
    } catch (error) {
        console.error('nsfK? Analysis failed:', error);
        updateBadge('❌', '#EF4444');
        
        sendResponse({
            success: false,
            error: error.message || 'Analysis failed'
        });
    } finally {
        isAnalyzing = false;
    }
}

// Analysis with retry logic
async function performAnalysisWithRetry(videoUrl) {
    let lastError;
    
    for (let attempt = 1; attempt <= CONFIG.RETRY_ATTEMPTS; attempt++) {
        try {
            console.log(`nsfK? Analysis attempt ${attempt}/${CONFIG.RETRY_ATTEMPTS}`);
            
            const result = await performAnalysis(videoUrl);
            console.log(`nsfK? Analysis succeeded on attempt ${attempt}`);
            return result;
            
        } catch (error) {
            console.error(`nsfK? Attempt ${attempt} failed:`, error);
            lastError = error;
            
            // Don't retry on certain errors
            if (error.message.includes('404') || error.message.includes('not found')) {
                throw error;
            }
            
            // Wait before retry (except on last attempt)
            if (attempt < CONFIG.RETRY_ATTEMPTS) {
                console.log(`nsfK? Waiting ${CONFIG.RETRY_DELAY}ms before retry...`);
                await new Promise(resolve => setTimeout(resolve, CONFIG.RETRY_DELAY));
            }
        }
    }
    
    throw new Error(`Analysis failed after ${CONFIG.RETRY_ATTEMPTS} attempts. Last error: ${lastError.message}`);
}

// Main analysis function
async function performAnalysis(videoUrl) {
    console.log('nsfK? Starting analysis for:', videoUrl);
    console.log('nsfK? Using API endpoint:', CONFIG.API_ENDPOINT);
    
    const startTime = Date.now();
    
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            console.log('nsfK? Request timed out after', CONFIG.API_TIMEOUT, 'ms');
            controller.abort();
        }, CONFIG.API_TIMEOUT);
        
        const response = await fetch(CONFIG.API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'ngrok-skip-browser-warning': 'true'
            },
            body: JSON.stringify({ url: videoUrl }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        const responseTime = Date.now() - startTime;
        console.log('nsfK? API response received in', responseTime, 'ms');
        console.log('nsfK? Response status:', response.status);
        
        if (!response.ok) {
            let errorText;
            try {
                const errorData = await response.json();
                errorText = errorData.message || errorData.error || errorData.detail || response.statusText;
            } catch (e) {
                errorText = await response.text() || response.statusText;
            }
            throw new Error(`API request failed: ${response.status} - ${errorText}`);
        }
        
        const result = await response.json();
        console.log('nsfK? Analysis completed successfully');
        console.log('nsfK? Raw API response:', result);
        
        // Return the API response with minimal transformation
        return cleanApiResponse(result, responseTime);
        
    } catch (error) {
        const errorTime = Date.now() - startTime;
        console.error('nsfK? API error after', errorTime, 'ms:', error);
        
        if (error.name === 'AbortError') {
            throw new Error(`Request timed out after ${CONFIG.API_TIMEOUT / 1000} seconds.`);
        } else if (error.message.includes('fetch')) {
            throw new Error('Network error: Unable to reach API endpoint.');
        }
        
        throw error;
    }
}

// Clean and standardize API response
function cleanApiResponse(apiResult, processingTime) {
    console.log('📊 Cleaning API response');
    
    // Direct mapping of API fields to our standard format
    const cleanedResponse = {
        // Core fields from API
        title: apiResult.title || 'Unknown Title',
        channel: apiResult.channel_name || 'Unknown Channel',
        duration: apiResult.duration || 0,
        
        // Safety information
        safetyScore: apiResult.safety_score || 0,
        recommendation: apiResult.recommendation || 'No recommendation available',
        
        // Content analysis
        summary: apiResult.summary || '',
        keywords: apiResult.keywords || [],
        
        // Category scores - pass through as-is
        categories: apiResult.category_scores || {},
        
        // Additional analysis fields
        commentAnalysis: apiResult.comment_analysis || '',
        webReputation: apiResult.web_reputation || '',
        
        // Risk factors if they exist (but don't create if not present)
        riskFactors: apiResult.risk_factors || [],
        
        // Metadata
        videoUrl: apiResult.video_url || '',
        analysisTimestamp: apiResult.analysis_timestamp || new Date().toISOString(),
        reportPath: apiResult.report_path || '',
        
        // Audio transcript if available
        audioTranscript: apiResult.audio_transcript || '',
        
        // Processing metadata
        processingTime: processingTime,
        apiEndpoint: CONFIG.API_ENDPOINT,
        
        // Keep original response for debugging
        _originalResponse: apiResult
    };
    
    console.log('✅ Cleaned response:', cleanedResponse);
    return cleanedResponse;
}

// Test API connection
async function testAPIConnection(sendResponse) {
    try {
        console.log('nsfK? Testing API connection...');
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000);
        
        const response = await fetch(CONFIG.API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true'
            },
            body: JSON.stringify({ url: 'test' }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        sendResponse({
            success: response.status < 500,
            message: response.status < 500 ? 'API endpoint is responding' : 'API endpoint error',
            endpoint: CONFIG.API_ENDPOINT,
            status: response.status
        });
        
    } catch (error) {
        sendResponse({
            success: false,
            error: error.message,
            endpoint: CONFIG.API_ENDPOINT
        });
    }
}

// Cache management
function getCachedAnalysis(videoId) {
    const cached = analysisCache.get(videoId);
    if (!cached) return null;
    
    if (Date.now() - cached.timestamp > CONFIG.CACHE_DURATION) {
        analysisCache.delete(videoId);
        return null;
    }
    
    return cached.data;
}

function setCachedAnalysis(videoId, data) {
    if (analysisCache.size >= CONFIG.MAX_CACHE_SIZE) {
        const oldestKey = analysisCache.keys().next().value;
        analysisCache.delete(oldestKey);
    }
    
    analysisCache.set(videoId, {
        data: data,
        timestamp: Date.now()
    });
    
    console.log('nsfK? Cached analysis for:', videoId);
}

// Update extension badge
function updateBadge(text, color) {
    try {
        chrome.action.setBadgeText({ text: text });
        chrome.action.setBadgeBackgroundColor({ color: color });
    } catch (error) {
        console.error('nsfK? Badge update failed:', error);
    }
}

// Extension lifecycle events
chrome.runtime.onInstalled.addListener((details) => {
    console.log('nsfK? Extension installed:', details.reason);
    if (details.reason === 'install') {
        chrome.tabs.create({
            url: 'https://www.youtube.com'
        });
    }
    updateBadge('', '#666666');
});

chrome.runtime.onStartup.addListener(() => {
    console.log('nsfK? Extension started');
    analysisCache.clear();
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url && tab.url.includes('youtube.com/watch')) {
        console.log('nsfK? YouTube tab updated:', tab.url);
    }
});

// Helper function to extract video ID from YouTube URL
function extractVideoIdFromUrl(url) {
    if (!url) return null;
    const match = url.match(/[?&]v=([^&]+)/);
    return match ? match[1] : null;
}

// Periodic cache cleanup
setInterval(() => {
    let cleanedCount = 0;
    const now = Date.now();
    
    for (const [videoId, cached] of analysisCache.entries()) {
        if (now - cached.timestamp > CONFIG.CACHE_DURATION) {
            analysisCache.delete(videoId);
            cleanedCount++;
        }
    }
    
    if (cleanedCount > 0) {
        console.log(`nsfK? Cleaned ${cleanedCount} expired cache entries`);
    }
}, 10 * 60 * 1000); // Clean every 10 minutes

console.log('nsfK? Background script initialized');
console.log('nsfK? API endpoint:', CONFIG.API_ENDPOINT);
console.log('nsfK? Timeout:', CONFIG.API_TIMEOUT / 1000, 'seconds');