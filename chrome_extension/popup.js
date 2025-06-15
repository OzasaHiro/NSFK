// popup.js - Clean minimal version

console.log('nsfK? Popup loaded');

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    checkYouTubePage();
    
    // Add click listener to analyze button
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeCurrentVideo);
    }
});

// Check if we're on a YouTube video page
async function checkYouTubePage() {
    try {
        const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
        
        if (!tab.url || !tab.url.includes('youtube.com/watch')) {
            showElement('not-youtube');
            hideElement('analyze-section');
            return;
        }
        
        showElement('analyze-section');
        hideElement('not-youtube');
        
    } catch (error) {
        console.error('Error checking page:', error);
        showError('Error: ' + error.message);
    }
}

// Analyze the current video
async function analyzeCurrentVideo() {
    try {
        showLoading();
        
        const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
        
        // Send analysis request to background script
        const response = await chrome.runtime.sendMessage({
            type: 'REQUEST_ANALYSIS',
            videoUrl: tab.url
        });
        
        if (!response.success) {
            throw new Error(response.error || 'Analysis failed');
        }
        
        displayResults(response);
        
    } catch (error) {
        console.error('Analysis failed:', error);
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// Display the analysis results
function displayResults(response) {
    const data = response.data || {};
    
    // Get values with defaults
    const safetyScore = data.safetyScore || 0;
    const recommendation = data.recommendation || 'No recommendation';
    const summary = data.summary || 'No summary available';
    const categories = data.categories || {};
    const keywords = data.keywords || [];
    
    // Build HTML
    let html = `
        <div class="score-section">
            <div class="total-score ${getScoreClass(safetyScore)}">${safetyScore}/100</div>
            <div class="recommendation ${getScoreClass(safetyScore)}">${recommendation}</div>
        </div>
        
        <div class="categories">`;
    
    // Add categories
    for (const [name, score] of Object.entries(categories)) {
        html += `
            <div class="category">
                <div class="category-name">${name}</div>
                <div class="category-score">${score}/${getCategoryMax(name)}</div>
            </div>`;
    }
    
    html += `</div>
        
        <div class="summary">
            <h3>Summary</h3>
            <p>${summary}</p>`;
    
    // Add keywords if any
    if (keywords.length > 0) {
        html += `<h3>Keywords</h3><div class="keywords">`;
        for (const keyword of keywords) {
            html += `<span class="keyword">${keyword}</span>`;
        }
        html += `</div>`;
    }
    
    html += `</div>`;
    
    // Add additional analysis if available
    if (data.channel || data.webReputation || data.commentAnalysis) {
        html += `<div class="additional-analysis"><h3>Additional Analysis</h3>`;
        
        if (data.channel) {
            html += `
                <div class="analysis-section">
                    <h4>📺 Channel Information</h4>
                    <p>${data.channel}</p>
                </div>`;
        }
        
        if (data.webReputation) {
            html += `
                <div class="analysis-section">
                    <h4>🌐 Web Reputation</h4>
                    <p>${data.webReputation}</p>
                </div>`;
        }
        
        if (data.commentAnalysis) {
            html += `
                <div class="analysis-section">
                    <h4>💬 Comment Analysis</h4>
                    <p>${data.commentAnalysis}</p>
                </div>`;
        }
        
        html += `</div>`;
    }
    
    // Set the HTML and show results
    document.getElementById('results').innerHTML = html;
    showElement('results');
    hideElement('analyze-section');
}

// Utility functions
function getScoreClass(score) {
    if (score >= 80) return 'safe';
    if (score >= 60) return 'review';
    return 'danger';
}

function getCategoryMax(name) {
    const maxScores = {
        'Non-Violence': 20,
        'Appropriate Language': 15,
        'Non-Scary Content': 20,
        'Family-Friendly Content': 15,
        'Substance-Free': 10,
        'Safe Behavior': 10,
        'Educational Value': 10
    };
    return maxScores[name] || 10;
}

function showElement(id) {
    const el = document.getElementById(id);
    if (el) el.classList.remove('hidden');
}

function hideElement(id) {
    const el = document.getElementById(id);
    if (el) el.classList.add('hidden');
}

function showLoading() {
    showElement('loading');
    hideElement('results');
    hideElement('error');
    
    const btn = document.getElementById('analyze-btn');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Analyzing...';
    }
}

function hideLoading() {
    hideElement('loading');
    
    const btn = document.getElementById('analyze-btn');
    if (btn) {
        btn.disabled = false;
        btn.textContent = 'Analyze Video';
    }
}

function showError(message) {
    const errorEl = document.getElementById('error-message');
    if (errorEl) errorEl.textContent = message;
    
    showElement('error');
    hideElement('results');
    hideElement('loading');
}

// Button handlers for popup.html
function retry() {
    hideElement('error');
    checkYouTubePage();
}

function openYouTube() {
    chrome.tabs.create({ url: 'https://www.youtube.com' });
}

// Add basic styles
const style = document.createElement('style');
style.textContent = `
    .score-section {
        text-align: center;
        padding: 20px;
        background: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    .total-score {
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .total-score.safe { color: #28a745; }
    .total-score.review { color: #ffc107; }
    .total-score.danger { color: #dc3545; }
    
    .recommendation {
        font-size: 1.2em;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
    }
    
    .recommendation.safe {
        background: #d4edda;
        color: #155724;
    }
    
    .recommendation.review {
        background: #fff3cd;
        color: #856404;
    }
    
    .recommendation.danger {
        background: #f8d7da;
        color: #721c24;
    }
    
    .categories {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 10px;
        margin: 20px 0;
    }
    
    .category {
        background: white;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .category-name {
        font-weight: bold;
        color: #333;
        font-size: 0.9em;
    }
    
    .category-score {
        font-size: 1.3em;
        font-weight: bold;
        color: #667eea;
    }
    
    .summary {
        background: white;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .summary h3 {
        margin-bottom: 10px;
        color: #333;
    }
    
    .keywords {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 10px;
    }
    
    .keyword {
        background: #667eea;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.9em;
    }
    
    .additional-analysis {
        background: white;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .analysis-section {
        margin-bottom: 15px;
        padding: 15px;
        border-left: 4px solid #17a2b8;
        background: #f8f9fa;
    }
    
    .analysis-section h4 {
        color: #17a2b8;
        margin-bottom: 8px;
    }
    
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .hidden {
        display: none !important;
    }
    
    #results {
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
`;
document.head.appendChild(style);

console.log('nsfK? Popup initialized');