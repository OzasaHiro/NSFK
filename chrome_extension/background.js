// ===========================================
// background.js - Fixed version
// ===========================================

console.log('nsfK? Background script loaded');

// Extension state
let currentVideoData = null;
let analysisCache = new Map();
let isAnalyzing = false;

// Configuration
const CONFIG = {
    CACHE_DURATION: 60 * 60 * 1000,
    MAX_CACHE_SIZE: 100,
    N8N_WEBHOOK_URL: 'https://ankitaj224.app.n8n.cloud/webhook-test/video-analysis' // Replace this!
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
            
        case 'GET_CACHE_STATS':
            sendResponse({
                success: true,
                cacheSize: analysisCache.size,
                maxSize: CONFIG.MAX_CACHE_SIZE
            });
            break;
            
        case 'OPEN_POPUP':
            // This functionality is not available in service workers
            // The popup will open when user clicks the extension icon
            console.log('nsfK? Popup open requested');
            sendResponse({ success: true, message: 'Click extension icon to open popup' });
            break;
            
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
    
    // Update badge to show video is detected
    updateBadge('📺', '#4F46E5');
    
    // Preload analysis if we have cached data
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
    
    const videoId = message.videoId || currentVideoData?.videoId;
    if (!videoId) {
        sendResponse({
            success: false,
            error: 'No video ID available'
        });
        return;
    }
    
    // Check cache first
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
    
    // Perform new analysis
    try {
        isAnalyzing = true;
        updateBadge('⏳', '#F59E0B');
        
        const analysisResult = await performAnalysis(videoId, message.videoUrl || currentVideoData?.url);
        
        // Cache the result
        setCachedAnalysis(videoId, analysisResult);
        
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

// Perform analysis via n8n webhook
async function performAnalysis(videoId, videoUrl) {
    console.log('nsfK? Starting analysis for:', videoId);
    
    const startTime = Date.now();
    
    const requestData = {
        videoUrl: videoUrl,
        videoId: videoId,
        userAgent: 'Chrome Extension nsfK',
        timestamp: new Date().toISOString(),
        source: 'chrome_extension'
    };
    
    try {
        console.log('nsfK? Making fetch request to n8n...');
        
        const response = await fetch(CONFIG.N8N_WEBHOOK_URL, {
            method: 'POST',
            mode: 'cors', // Explicitly set CORS mode
            credentials: 'omit', // Don't send credentials
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        console.log('nsfK? Response status:', response.status);
        
        if (!response.ok) {
            // Try to get error details
            let errorText;
            try {
                errorText = await response.text();
            } catch (e) {
                errorText = 'Unable to read error response';
            }
            throw new Error(`Analysis request failed: ${response.status} ${response.statusText} - ${errorText}`);
        }
        
        const result = await response.json();
        console.log('nsfK? Analysis response received');
        
        if (!result.success) {
            throw new Error(result.error || 'Analysis failed on server');
        }
        
        const processingTime = Date.now() - startTime;
        console.log(`nsfK? Analysis completed in ${processingTime}ms`);
        
        return {
            ...result.analysis,
            processingTime: processingTime,
            analysisDate: new Date().toISOString()
        };
        
    } catch (error) {
        console.error('nsfK? Analysis error:', error);
        
        // If CORS error, provide helpful message
        if (error.message.includes('CORS') || error.message.includes('fetch')) {
            throw new Error('CORS error: Please configure your n8n webhook to allow cross-origin requests');
        }
        
        throw error;
    }
}

// Cache management
function getCachedAnalysis(videoId) {
    const cached = analysisCache.get(videoId);
    if (!cached) return null;
    
    // Check if cache is expired
    if (Date.now() - cached.timestamp > CONFIG.CACHE_DURATION) {
        analysisCache.delete(videoId);
        return null;
    }
    
    return cached.data;
}

function setCachedAnalysis(videoId, data) {
    // Clean old cache if too large
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

// Handle extension installation
chrome.runtime.onInstalled.addListener((details) => {
    console.log('nsfK? Extension installed:', details.reason);
    
    if (details.reason === 'install') {
        // First time installation - open YouTube
        chrome.tabs.create({
            url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
        });
    }
    
    // Set initial badge
    updateBadge('', '#666666');
});

// Handle extension startup
chrome.runtime.onStartup.addListener(() => {
    console.log('nsfK? Extension started');
    analysisCache.clear(); // Clear cache on startup
});

// Cleanup when extension is suspended
chrome.runtime.onSuspend.addListener(() => {
    console.log('nsfK? Extension suspending');
});

// Handle tab updates to detect YouTube navigation
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url && tab.url.includes('youtube.com/watch')) {
        console.log('nsfK? YouTube tab updated:', tab.url);
    }
});

// Periodic cache cleanup (every 10 minutes)
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

console.log('nsfK? Background script initialized');