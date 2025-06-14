// ===========================================
// background.js - Final Complete Version
// ===========================================

console.log('nsfK? Background script loaded');

// Extension state
let currentVideoData = null;
let analysisCache = new Map();
let isAnalyzing = false;

// Configuration - UPDATE YOUR API ENDPOINT HERE
const CONFIG = {
    CACHE_DURATION: 60 * 60 * 1000, // 1 hour
    MAX_CACHE_SIZE: 100,
    API_ENDPOINT: 'https://2bcb-107-194-242-26.ngrok-free.app/analyze', // Update this!
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
    const videoId = message.videoId || currentVideoData?.videoId;
    
    if (!videoUrl) {
        sendResponse({
            success: false,
            error: 'No video URL available'
        });
        return;
    }
    
    // Check cache first
    if (videoId) {
        const cached = getCachedAnalysis(videoId);
        if (cached) {
            console.log('nsfK? Returning cached analysis');
            sendResponse({
                success: true,
                data: cached,
                source: 'cache'
            });
            return;
        }
    }
    
    // Perform new analysis with retries
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
            
            // Test API connection first on first attempt
            if (attempt === 1) {
                const isAPIAlive = await quickAPITest();
                if (!isAPIAlive) {
                    throw new Error('API endpoint not responding to basic requests');
                }
            }
            
            const result = await performAnalysis(videoUrl);
            console.log(`nsfK? Analysis succeeded on attempt ${attempt}`);
            return result;
            
        } catch (error) {
            console.error(`nsfK? Attempt ${attempt} failed:`, error);
            lastError = error;
            
            // Don't retry on certain errors
            if (error.message.includes('404') || error.message.includes('not found')) {
                throw error; // Don't retry 404s
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

// Quick API test to check if endpoint is alive
async function quickAPITest() {
    try {
        console.log('nsfK? Testing API connection...');
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout for test
        
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
        
        console.log('nsfK? API test response:', response.status);
        return response.status < 500; // Accept any response except server errors
        
    } catch (error) {
        console.error('nsfK? API test failed:', error);
        return false;
    }
}

// Main analysis function
async function performAnalysis(videoUrl) {
    console.log('nsfK? Starting analysis for:', videoUrl);
    console.log('nsfK? Using API endpoint:', CONFIG.API_ENDPOINT);
    
    const startTime = Date.now();
    
    const requestData = {
        url: videoUrl
    };
    
    console.log('nsfK? Request payload:', requestData);
    
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            console.log('nsfK? Request timed out after', CONFIG.API_TIMEOUT, 'ms');
            controller.abort();
        }, CONFIG.API_TIMEOUT);
        
        console.log(`nsfK? Making request with ${CONFIG.API_TIMEOUT/1000}s timeout...`);
        
        const response = await fetch(CONFIG.API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'ngrok-skip-browser-warning': 'true'
            },
            body: JSON.stringify(requestData),
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
                errorText = errorData.message || errorData.error || response.statusText;
            } catch (e) {
                errorText = await response.text() || response.statusText;
            }
            throw new Error(`API request failed: ${response.status} - ${errorText}`);
        }
        
        const result = await response.json();
        console.log('nsfK? Analysis completed successfully');
        
        if (!result || typeof result !== 'object') {
            throw new Error('Invalid API response format');
        }
        
        const processingTime = Date.now() - startTime;
        return formatApiResponse(result, processingTime);
        
    } catch (error) {
        const errorTime = Date.now() - startTime;
        console.error('nsfK? API error after', errorTime, 'ms:', error);
        
        if (error.name === 'AbortError') {
            throw new Error(`Request timed out after ${CONFIG.API_TIMEOUT / 1000} seconds. Your API might be processing slowly or not responding.`);
        } else if (error.message.includes('fetch')) {
            throw new Error('Network error: Unable to reach API endpoint. Check if your ngrok tunnel is running.');
        }
        
        throw error;
    }
}

// Test API connection function
async function testAPIConnection(sendResponse) {
    try {
        console.log('nsfK? Testing API connection...');
        
        const isAlive = await quickAPITest();
        
        if (isAlive) {
            sendResponse({
                success: true,
                message: 'API endpoint is responding',
                endpoint: CONFIG.API_ENDPOINT
            });
        } else {
            sendResponse({
                success: false,
                error: 'API endpoint not responding',
                endpoint: CONFIG.API_ENDPOINT
            });
        }
    } catch (error) {
        sendResponse({
            success: false,
            error: error.message,
            endpoint: CONFIG.API_ENDPOINT
        });
    }
}

// ===========================================
// DYNAMIC API RESPONSE FORMATTER
// ===========================================

// Dynamic formatApiResponse function that adapts to any API response format
function formatApiResponse(apiResult, processingTime) {
    console.log('🔍 Raw API result:', apiResult);
    
    // Extract the actual data from various possible wrapper formats
    let rawData = apiResult;
    if (apiResult.data) rawData = apiResult.data;
    else if (apiResult.analysis) rawData = apiResult.analysis;
    else if (apiResult.result) rawData = apiResult.result;
    else if (apiResult.response) rawData = apiResult.response;
    
    console.log('📊 Extracted raw data:', rawData);
    
    // Dynamic field mapping - check for various possible field names
    const dynamicAnalysis = {
        // Video Information
        title: findField(rawData, [
            'title', 'video_title', 'name', 'video_name', 'videoTitle'
        ]) || 'Unknown Title',
        
        channel: findField(rawData, [
            'channel', 'channel_name', 'channelName', 'author', 'uploader', 'creator'
        ]) || 'Unknown Channel',
        
        duration: formatDuration(findField(rawData, [
            'duration', 'length', 'video_duration', 'videoDuration', 'time'
        ])),
        
        thumbnail: findField(rawData, [
            'thumbnail', 'thumbnail_url', 'thumbnailUrl', 'image', 'preview'
        ]) || '',
        
        // Safety Scores - handle different scale formats
        safetyRating: calculateSafetyRating(rawData),
        safetyColor: '', // Will be calculated later
        
        // Age and Recommendations
        ageRecommendation: findField(rawData, [
            'recommendation', 'age_recommendation', 'ageRecommendation', 
            'suggested_age', 'age_rating', 'rating'
        ]) || 'Parental guidance recommended',
        
        confidence: findField(rawData, [
            'confidence', 'confidence_score', 'confidenceScore', 'certainty'
        ]) || calculateConfidence(rawData),
        
        // Content Analysis
        reasoning: findField(rawData, [
            'summary', 'reasoning', 'explanation', 'analysis', 'description',
            'report', 'conclusion', 'assessment'
        ]) || 'Analysis completed',
        
        // Warnings and Risk Factors
        contentWarnings: extractWarnings(rawData),
        positiveAspects: extractPositives(rawData),
        
        // Categories - handle various category formats
        categories: extractCategories(rawData),
        
        // Additional Dynamic Fields
        keywords: findField(rawData, [
            'keywords', 'tags', 'labels', 'topics'
        ]) || [],
        
        // Audio/Text Content
        audioTranscript: findField(rawData, [
            'audio_transcript', 'transcript', 'audioTranscript', 'speech_text', 'captions'
        ]),
        
        // Metadata
        analysisTime: new Date().toISOString(),
        processingTime: processingTime,
        apiEndpoint: CONFIG.API_ENDPOINT,
        originalApiResponse: rawData, // Keep full original response
        
        // Dynamic extra fields - capture anything we might have missed
        extraFields: extractExtraFields(rawData)
    };
    
    // Calculate safety color based on rating
    dynamicAnalysis.safetyColor = calculateSafetyColor(dynamicAnalysis.safetyRating);
    
    console.log('✅ Dynamic analysis result:', dynamicAnalysis);
    return dynamicAnalysis;
}

// Helper function to find a field by checking multiple possible names
function findField(obj, fieldNames) {
    if (!obj || typeof obj !== 'object') return null;
    
    for (const name of fieldNames) {
        // Check exact match
        if (obj.hasOwnProperty(name) && obj[name] !== null && obj[name] !== undefined) {
            return obj[name];
        }
        
        // Check case-insensitive match
        const keys = Object.keys(obj);
        const foundKey = keys.find(key => key.toLowerCase() === name.toLowerCase());
        if (foundKey && obj[foundKey] !== null && obj[foundKey] !== undefined) {
            return obj[foundKey];
        }
    }
    
    return null;
}

// Dynamic safety rating calculation
function calculateSafetyRating(data) {
    // Try direct safety fields first
    const directSafety = findField(data, [
        'safety_rating', 'safetyRating', 'safety_score', 'safetyScore',
        'overall_rating', 'overallRating', 'rating', 'score'
    ]);
    
    if (directSafety !== null) {
        // Convert different scales to 0-10
        if (directSafety <= 1) return Math.round(directSafety * 10); // 0-1 scale
        if (directSafety <= 5) return Math.round(directSafety * 2); // 0-5 scale
        if (directSafety <= 10) return Math.round(directSafety); // 0-10 scale
        if (directSafety <= 100) return Math.round(directSafety / 10); // 0-100 scale
    }
    
    // Try to calculate from category scores
    const categories = extractCategories(data);
    if (Object.keys(categories).length > 0) {
        const avgScore = Object.values(categories).reduce((a, b) => a + b, 0) / Object.values(categories).length;
        return Math.round(avgScore);
    }
    
    // Default fallback
    return 5;
}

// Extract warnings from various possible formats
function extractWarnings(data) {
    const warnings = findField(data, [
        'warnings', 'risk_factors', 'riskFactors', 'concerns', 'issues',
        'content_warnings', 'contentWarnings', 'alerts', 'flags'
    ]);
    
    if (Array.isArray(warnings)) return warnings;
    if (typeof warnings === 'string') return [warnings];
    
    // Look for individual warning fields
    const warningTypes = [
        'violence', 'language', 'sexual_content', 'substance_use', 
        'scary_content', 'inappropriate_content'
    ];
    
    const foundWarnings = [];
    warningTypes.forEach(type => {
        const value = findField(data, [type, type.replace('_', ''), type + '_warning']);
        if (value && (value === true || (typeof value === 'number' && value > 50))) {
            foundWarnings.push(`Contains ${type.replace('_', ' ')}`);
        }
    });
    
    return foundWarnings;
}

// Extract positive aspects
function extractPositives(data) {
    const positives = findField(data, [
        'positives', 'positive_aspects', 'positiveAspects', 'benefits',
        'educational', 'educational_value', 'good_aspects'
    ]);
    
    if (Array.isArray(positives)) return positives;
    if (typeof positives === 'string') return [positives];
    
    // Check for educational indicators
    const foundPositives = [];
    const educational = findField(data, ['educational', 'educational_value', 'learning']);
    if (educational && (educational === true || (typeof educational === 'number' && educational > 50))) {
        foundPositives.push('Educational content');
    }
    
    return foundPositives;
}

// Extract and normalize categories
function extractCategories(data) {
    const categories = {};
    
    // Try to find category scores object
    const categoryScores = findField(data, [
        'categories', 'category_scores', 'categoryScores', 'scores', 'ratings'
    ]);
    
    if (categoryScores && typeof categoryScores === 'object') {
        Object.entries(categoryScores).forEach(([key, value]) => {
            if (typeof value === 'number') {
                const normalizedKey = key.toLowerCase().replace(/[^a-z]/g, '');
                // Normalize to 0-10 scale and invert if it seems like a risk score
                let normalizedValue = value;
                if (value <= 1) normalizedValue = value * 10;
                else if (value <= 5) normalizedValue = value * 2;
                else if (value > 10) normalizedValue = value / 10;
                
                // If this seems like a risk score (higher = worse), invert it
                if (key.toLowerCase().includes('risk') || key.toLowerCase().includes('danger')) {
                    normalizedValue = 10 - normalizedValue;
                }
                
                categories[normalizedKey] = Math.round(Math.max(0, Math.min(10, normalizedValue)));
            }
        });
    }
    
    return categories;
}

// Calculate confidence if not provided
function calculateConfidence(data) {
    // Look for confidence indicators
    const hasTranscript = findField(data, ['transcript', 'audio_transcript']);
    const hasCategories = Object.keys(extractCategories(data)).length > 0;
    const hasWarnings = extractWarnings(data).length > 0;
    
    let confidence = 5; // Base confidence
    if (hasTranscript) confidence += 2;
    if (hasCategories) confidence += 2;
    if (hasWarnings) confidence += 1;
    
    return Math.min(10, confidence);
}

// Calculate safety color
function calculateSafetyColor(rating) {
    if (rating >= 8) return '#10B981'; // Green
    if (rating >= 6) return '#F59E0B'; // Yellow
    if (rating >= 4) return '#F97316'; // Orange
    return '#EF4444'; // Red
}

// Format duration from various formats
function formatDuration(duration) {
    if (!duration) return 'Unknown';
    
    if (typeof duration === 'number') {
        // Assume seconds
        const minutes = Math.floor(duration / 60);
        const seconds = duration % 60;
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }
    
    if (typeof duration === 'string') {
        // Return as-is if already formatted
        return duration;
    }
    
    return 'Unknown';
}

// Extract any extra fields we might want to display
function extractExtraFields(data) {
    const extraFields = {};
    const commonFields = new Set([
        'title', 'channel', 'duration', 'thumbnail', 'safety_rating', 'safetyRating',
        'recommendation', 'summary', 'warnings', 'categories', 'keywords', 'transcript'
    ]);
    
    Object.entries(data).forEach(([key, value]) => {
        const normalizedKey = key.toLowerCase();
        if (!commonFields.has(normalizedKey) && !commonFields.has(key)) {
            // Only include interesting extra fields
            if (typeof value === 'string' || typeof value === 'number' || Array.isArray(value)) {
                extraFields[key] = value;
            }
        }
    });
    
    return extraFields;
}

// ===========================================
// CACHE AND UTILITY FUNCTIONS
// ===========================================

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
            url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
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
}, 10 * 60 * 1000);

console.log('nsfK? Background script initialized with dynamic response handling');
console.log('nsfK? API endpoint:', CONFIG.API_ENDPOINT);
console.log('nsfK? Timeout:', CONFIG.API_TIMEOUT / 1000, 'seconds');