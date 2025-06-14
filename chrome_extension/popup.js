// popup.js - Updated for dynamic response handling

console.log('nsfK? Popup script loaded');

document.addEventListener('DOMContentLoaded', function() {
    console.log('nsfK? DOM loaded, initializing popup');
    checkYouTubePage();
    
    // Add event listeners
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeCurrentVideo);
    }
});

async function checkYouTubePage() {
    try {
        console.log('nsfK? Checking current page...');
        
        const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
        console.log('nsfK? Current tab URL:', tab.url);
        
        if (!tab.url || !tab.url.includes('youtube.com/watch')) {
            console.log('nsfK? Not on YouTube video page');
            showElement('not-youtube');
            hideElement('analyze-section');
            updateVideoInfo('Not on YouTube video page');
            return;
        }
        
        // Extract video ID from URL
        const videoId = extractVideoId(tab.url);
        console.log('nsfK? Video ID:', videoId);
        
        if (videoId) {
            updateVideoInfo(`Video detected: ${videoId}`);
            showElement('analyze-section');
            hideElement('not-youtube');
            
            // Check if we have cached results
            const cached = await getCachedResult(videoId);
            if (cached) {
                console.log('nsfK? Found cached results');
                displayResults(cached);
                return;
            }
        } else {
            updateVideoInfo('Could not detect video ID');
        }
        
    } catch (error) {
        console.error('nsfK? Error checking page:', error);
        showError('Error checking current page: ' + error.message);
    }
}

async function analyzeCurrentVideo() {
    try {
        console.log('nsfK? Starting video analysis...');
        showLoading();
        
        const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
        const videoUrl = tab.url;
        
        if (!videoUrl || !videoUrl.includes('youtube.com/watch')) {
            throw new Error('Please navigate to a YouTube video page');
        }
        
        console.log('nsfK? Analyzing URL:', videoUrl);
        
        // Send analysis request to background script
        const response = await chrome.runtime.sendMessage({
            type: 'REQUEST_ANALYSIS',
            videoUrl: videoUrl,
            videoId: extractVideoId(videoUrl)
        });
        
        console.log('nsfK? Analysis response:', response);
        
        if (!response.success) {
            throw new Error(response.error || 'Analysis failed');
        }
        
        // Cache the result
        const videoId = extractVideoId(videoUrl);
        if (videoId) {
            await cacheResult(videoId, response);
        }
        
        // Display results
        displayResults(response);
        
    } catch (error) {
        console.error('nsfK? Analysis failed:', error);
        showError('Analysis failed: ' + error.message);
    } finally {
        hideLoading();
    }
}

function displayResults(result) {
    console.log('nsfK? Displaying results:', result);
    
    // Handle nested response structure - get the actual analysis data
    const analysis = result.data || result.analysis || result;
    
    if (!analysis) {
        showError('No analysis data received');
        return;
    }
    
    // Use the dynamic fields from background.js formatApiResponse
    const safetyRating = analysis.safetyRating || 5;
    const safetyColor = analysis.safetyColor || getSafetyColor(safetyRating);
    const safetyLevel = getSafetyLevel(safetyRating);
    
    const resultsHTML = `
        <div class="video-info">
            <div class="video-details">
                <h3 class="video-title">${escapeHtml(analysis.title) || 'Unknown Title'}</h3>
                <p class="video-meta">
                    <span class="channel">Channel: ${escapeHtml(analysis.channel) || 'Unknown'}</span><br>
                    <span class="duration">Duration: ${analysis.duration || 'Unknown'}</span>
                    ${analysis.analysisTime ? `<br><span class="timestamp">Analyzed: ${formatTimestamp(analysis.analysisTime)}</span>` : ''}
                </p>
                ${analysis.keywords && analysis.keywords.length > 0 ? `
                    <div class="keywords-section">
                        <strong>Keywords:</strong>
                        <div class="keywords-tags">
                            ${analysis.keywords.map(keyword => `<span class="keyword-tag">${escapeHtml(keyword)}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>
        </div>
        
        <div class="safety-rating" style="border-left: 4px solid ${safetyColor}">
            <div class="rating-header">
                <span class="rating-score" style="color: ${safetyColor}">
                    ${safetyRating}/10
                </span>
                <span class="age-recommendation" style="color: ${safetyColor}">
                    ${safetyLevel}
                </span>
            </div>
            ${analysis.ageRecommendation ? `
                <p class="age-details">${escapeHtml(analysis.ageRecommendation)}</p>
            ` : ''}
            ${analysis.confidence ? `
                <small class="confidence">Confidence: ${analysis.confidence}/10</small>
            ` : ''}
        </div>
        
        ${analysis.reasoning ? `
        <div class="summary-section">
            <h4>📋 Analysis Summary</h4>
            <p class="summary-text">${escapeHtml(analysis.reasoning)}</p>
        </div>
        ` : ''}
        
        ${analysis.contentWarnings && analysis.contentWarnings.length > 0 ? `
        <div class="warnings-section">
            <h4>⚠️ Content Warnings</h4>
            <ul class="warnings-list">
                ${analysis.contentWarnings.map(warning => `<li>${escapeHtml(warning)}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
        
        ${analysis.positiveAspects && analysis.positiveAspects.length > 0 ? `
        <div class="positives-section">
            <h4>✅ Positive Aspects</h4>
            <ul class="positives-list">
                ${analysis.positiveAspects.map(positive => `<li>${escapeHtml(positive)}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
        
        ${analysis.categories && Object.keys(analysis.categories).length > 0 ? `
        <div class="categories-section">
            <h4>📊 Content Categories</h4>
            <div class="category-bars">
                ${Object.entries(analysis.categories).map(([category, score]) => {
                    const categoryColor = getCategoryColor(score);
                    const percentage = Math.min(score * 10, 100);
                    const formattedCategory = category.replace(/([A-Z])/g, ' $1').trim();
                    
                    return `
                        <div class="category-bar">
                            <span class="category-label">${escapeHtml(formattedCategory)}</span>
                            <div class="bar-container">
                                <div class="bar-fill" style="width: ${percentage}%; background-color: ${categoryColor}"></div>
                                <span class="bar-score">${score}/10</span>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        </div>
        ` : ''}
        
        ${analysis.audioTranscript ? `
        <div class="transcript-section">
            <h4>🎤 Audio Transcript</h4>
            <div class="transcript-content">
                <p class="transcript-text">${escapeHtml(analysis.audioTranscript)}</p>
            </div>
        </div>
        ` : ''}
        
        ${analysis.extraFields && Object.keys(analysis.extraFields).length > 0 ? `
        <div class="extra-fields-section">
            <h4>📌 Additional Information</h4>
            <div class="extra-fields">
                ${Object.entries(analysis.extraFields).map(([key, value]) => {
                    const formattedKey = key.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ').trim();
                    let formattedValue = value;
                    
                    if (Array.isArray(value)) {
                        formattedValue = value.join(', ');
                    } else if (typeof value === 'object') {
                        formattedValue = JSON.stringify(value, null, 2);
                    }
                    
                    return `
                        <div class="extra-field">
                            <strong>${escapeHtml(formattedKey)}:</strong> 
                            <span>${escapeHtml(String(formattedValue))}</span>
                        </div>
                    `;
                }).join('')}
            </div>
        </div>
        ` : ''}
        
        <div class="analysis-meta">
            <small>
                ${analysis.processingTime ? `Analysis took ${(analysis.processingTime / 1000).toFixed(1)}s • ` : ''}
                ${analysis.source ? `Source: ${analysis.source} • ` : ''}
                Powered by ${analysis.apiEndpoint ? analysis.apiEndpoint.split('/').slice(2, 3).join('') : 'nsfK'}
            </small>
        </div>
        
        <div class="debug-section" style="display: none;">
            <h4>🐛 Debug Information</h4>
            <pre>${JSON.stringify(analysis.originalApiResponse || analysis, null, 2)}</pre>
        </div>
    `;
    
    document.getElementById('results').innerHTML = resultsHTML;
    showElement('results');
    hideElement('analyze-section');
    
    // Add debug toggle functionality
    addDebugToggle();
}

// Add debug toggle functionality
function addDebugToggle() {
    const metaSection = document.querySelector('.analysis-meta');
    if (metaSection) {
        const debugToggle = document.createElement('a');
        debugToggle.href = '#';
        debugToggle.textContent = ' • Show Debug';
        debugToggle.onclick = (e) => {
            e.preventDefault();
            const debugSection = document.querySelector('.debug-section');
            if (debugSection) {
                if (debugSection.style.display === 'none') {
                    debugSection.style.display = 'block';
                    debugToggle.textContent = ' • Hide Debug';
                } else {
                    debugSection.style.display = 'none';
                    debugToggle.textContent = ' • Show Debug';
                }
            }
        };
        metaSection.appendChild(debugToggle);
    }
}

// ===========================================
// UTILITY FUNCTIONS FOR DYNAMIC DISPLAY
// ===========================================

// Get safety color based on score (0-10 scale)
function getSafetyColor(score) {
    if (!score && score !== 0) return '#666';
    if (score >= 8) return '#22c55e'; // Green - Safe
    if (score >= 6) return '#eab308'; // Yellow - Caution
    if (score >= 4) return '#f97316'; // Orange - Warning
    return '#ef4444'; // Red - Unsafe
}

// Get safety level text
function getSafetyLevel(score) {
    if (!score && score !== 0) return 'Not analyzed';
    if (score >= 8) return 'Safe for Kids';
    if (score >= 6) return 'Mostly Safe';
    if (score >= 4) return 'Caution Advised';
    return 'Not Safe for Kids';
}

// Get category color based on score (0-10 scale, where 10 is safest)
function getCategoryColor(score) {
    if (!score && score !== 0) return '#666';
    if (score >= 8) return '#22c55e'; // Green - Safe
    if (score >= 5) return '#eab308'; // Yellow - Medium
    if (score >= 3) return '#f97316'; // Orange - Concerning
    return '#ef4444'; // Red - High concern
}

// Format timestamp to readable format
function formatTimestamp(timestamp) {
    if (!timestamp) return 'Unknown';
    
    try {
        const date = new Date(timestamp);
        return date.toLocaleString();
    } catch (error) {
        console.error('Error formatting timestamp:', error);
        return timestamp;
    }
}

// Escape HTML to prevent XSS attacks
function escapeHtml(text) {
    if (!text && text !== 0) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

// ===========================================
// EXISTING UTILITY FUNCTIONS
// ===========================================

// Extract video ID from URL
function extractVideoId(url) {
    if (!url) return null;
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = url.match(regExp);
    return (match && match[2].length === 11) ? match[2] : null;
}

function updateVideoInfo(text) {
    const infoElement = document.getElementById('video-info');
    if (infoElement) {
        infoElement.textContent = text;
    }
}

async function getCachedResult(videoId) {
    return new Promise((resolve) => {
        chrome.storage.local.get([`analysis_${videoId}`], (result) => {
            const cached = result[`analysis_${videoId}`];
            // Extended cache time to 24 hours for better performance
            if (cached && Date.now() - cached.timestamp < 86400000) { // 24 hours cache
                resolve(cached.data);
            } else {
                // Clean up expired cache
                if (cached) {
                    chrome.storage.local.remove([`analysis_${videoId}`]);
                }
                resolve(null);
            }
        });
    });
}

async function cacheResult(videoId, data) {
    const cacheData = {
        data: data,
        timestamp: Date.now()
    };
    chrome.storage.local.set({[`analysis_${videoId}`]: cacheData});
    console.log('nsfK? Cached result for video:', videoId);
}

function showElement(id) {
    const element = document.getElementById(id);
    if (element) {
        element.classList.remove('hidden');
    }
}

function hideElement(id) {
    const element = document.getElementById(id);
    if (element) {
        element.classList.add('hidden');
    }
}

function showLoading() {
    showElement('loading');
    hideElement('results');
    hideElement('error');
    hideElement('analyze-section');
    hideElement('not-youtube');
    
    // Update loading message
    const loadingText = document.querySelector('#loading p');
    if (loadingText) {
        loadingText.textContent = 'Analyzing video content... This may take up to 4 minutes for longer videos.';
    }
    
    // Disable analyze button during loading
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
    }
}

function hideLoading() {
    hideElement('loading');
    
    // Re-enable analyze button
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze Video';
    }
}

function showError(message) {
    const errorElement = document.getElementById('error-message');
    if (errorElement) {
        errorElement.textContent = message;
    }
    showElement('error');
    hideElement('results');
    hideElement('loading');
    hideElement('analyze-section');
}

// ===========================================
// BUTTON HANDLERS
// ===========================================

function retry() {
    hideElement('error');
    checkYouTubePage();
}

function openYouTube() {
    chrome.tabs.create({
        url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    });
}

function clearCache() {
    chrome.storage.local.clear(() => {
        console.log('nsfK? Cache cleared');
        updateVideoInfo('Cache cleared - refresh page to reanalyze');
        // Hide results to show cache was cleared
        hideElement('results');
        showElement('analyze-section');
    });
}

function openSettings() {
    // Future: Open settings page
    console.log('nsfK? Settings clicked (not implemented yet)');
    // You could open an options page here:
    // chrome.runtime.openOptionsPage();
}

// Add CSS for new sections
const style = document.createElement('style');
style.textContent = `
    .positives-section {
        margin-top: 15px;
        padding: 10px;
        background: #f0fdf4;
        border-radius: 5px;
    }
    
    .positives-list {
        list-style: none;
        padding: 0;
        margin: 5px 0;
    }
    
    .positives-list li {
        padding: 5px 0;
        padding-left: 20px;
        position: relative;
    }
    
    .positives-list li:before {
        content: "✅";
        position: absolute;
        left: 0;
    }
    
    .extra-fields-section {
        margin-top: 15px;
        padding: 10px;
        background: #f9fafb;
        border-radius: 5px;
    }
    
    .extra-field {
        margin: 5px 0;
        font-size: 14px;
    }
    
    .confidence {
        display: block;
        margin-top: 5px;
        opacity: 0.7;
        font-size: 12px;
    }
    
    .age-details {
        margin-top: 8px;
        font-size: 14px;
        line-height: 1.4;
    }
    
    .debug-section {
        margin-top: 15px;
        padding: 10px;
        background: #f3f4f6;
        border-radius: 5px;
        font-size: 12px;
    }
    
    .debug-section pre {
        white-space: pre-wrap;
        word-wrap: break-word;
        max-height: 300px;
        overflow-y: auto;
    }
`;
document.head.appendChild(style);

console.log('nsfK? Popup script initialized with dynamic response handling');