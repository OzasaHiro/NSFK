// n8n webhook URL - Replace with your actual webhook
const N8N_WEBHOOK_URL = 'https://your-n8n-instance.com/webhook/video-analysis';

document.addEventListener('DOMContentLoaded', function() {
    checkYouTubePage();
    
    document.getElementById('analyze-btn').addEventListener('click', analyzeCurrentVideo);
});

async function checkYouTubePage() {
    try {
        const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
        
        if (!tab.url.includes('youtube.com/watch')) {
            showElement('not-youtube');
            hideElement('analyze-section');
            return;
        }
        
        // Check if we have cached results for this video
        const videoId = extractVideoId(tab.url);
        if (videoId) {
            const cached = await getCachedResult(videoId);
            if (cached) {
                displayResults(cached);
                return;
            }
        }
        
        showElement('analyze-section');
        
    } catch (error) {
        showError('Error checking current page: ' + error.message);
    }
}

async function analyzeCurrentVideo() {
    try {
        showLoading();
        
        const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
        const videoUrl = tab.url;
        
        if (!videoUrl.includes('youtube.com/watch')) {
            throw new Error('Please navigate to a YouTube video page');
        }
        
        // Send request to n8n webhook
        const response = await fetch(N8N_WEBHOOK_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                videoUrl: videoUrl,
                userAgent: navigator.userAgent,
                timestamp: new Date().toISOString()
            })
        });
        
        if (!response.ok) {
            throw new Error(`Analysis failed: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Analysis failed');
        }
        
        // Cache the result
        await cacheResult(extractVideoId(videoUrl), result);
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        showError('Analysis failed: ' + error.message);
    } finally {
        hideLoading();
    }
}

function displayResults(result) {
    const analysis = result.analysis;
    
    const resultsHTML = `
        <div class="video-info">
            <img src="${analysis.thumbnail}" alt="Video thumbnail" class="thumbnail">
            <div class="video-details">
                <h3 class="video-title">${analysis.title}</h3>
                <p class="channel-name">by ${analysis.channel}</p>
                <p class="video-meta">${analysis.duration} • ${formatNumber(analysis.viewCount)} views</p>
            </div>
        </div>
        
        <div class="safety-rating" style="border-left: 4px solid ${analysis.safetyColor}">
            <div class="rating-header">
                <span class="rating-score" style="color: ${analysis.safetyColor}">
                    ${analysis.safetyRating}/10
                </span>
                <span class="age-recommendation">${analysis.ageRecommendation}</span>
            </div>
            <div class="confidence">Confidence: ${analysis.confidence}/10</div>
        </div>
        
        ${analysis.contentWarnings.length > 0 ? `
        <div class="warnings-section">
            <h4>⚠️ Content Warnings</h4>
            <ul class="warnings-list">
                ${analysis.contentWarnings.map(warning => `<li>${warning}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
        
        ${analysis.positiveAspects.length > 0 ? `
        <div class="positive-section">
            <h4>✅ Positive Aspects</h4>
            <ul class="positive-list">
                ${analysis.positiveAspects.map(aspect => `<li>${aspect}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
        
        <div class="reasoning-section">
            <h4>🤔 Analysis Reasoning</h4>
            <p class="reasoning-text">${analysis.reasoning}</p>
        </div>
        
        ${analysis.timestamps.length > 0 ? `
        <div class="timestamps-section">
            <h4>⏰ Important Timestamps</h4>
            <ul class="timestamps-list">
                ${analysis.timestamps.map(ts => `<li><strong>${ts.time}</strong>: ${ts.note}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
        
        <div class="analysis-meta">
            <small>Analysis completed in ${analysis.processingTimeMs}ms</small>
        </div>
    `;
    
    document.getElementById('results').innerHTML = resultsHTML;
    showElement('results');
    hideElement('analyze-section');
}

// Utility functions
function extractVideoId(url) {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = url.match(regExp);
    return (match && match[2].length === 11) ? match[2] : null;
}

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
}

async function getCachedResult(videoId) {
    return new Promise((resolve) => {
        chrome.storage.local.get([`analysis_${videoId}`], (result) => {
            const cached = result[`analysis_${videoId}`];
            if (cached && Date.now() - cached.timestamp < 3600000) { // 1 hour cache
                resolve(cached.data);
            } else {
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
}

function showElement(id) {
    document.getElementById(id).classList.remove('hidden');
}

function hideElement(id) {
    document.getElementById(id).classList.add('hidden');
}

function showLoading() {
    showElement('loading');
    hideElement('results');
    hideElement('error');
    hideElement('analyze-section');
}

function hideLoading() {
    hideElement('loading');
}

function showError(message) {
    document.getElementById('error-message').textContent = message;
    showElement('error');
    hideElement('results');
    hideElement('loading');
}

function retry() {
    hideElement('error');
    checkYouTubePage();
}