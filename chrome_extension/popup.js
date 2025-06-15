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
        // Start timing the analysis
        analysisStartTime = Date.now();

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

        // End timing the analysis
        analysisEndTime = Date.now();
        const analysisTime = analysisEndTime - analysisStartTime;

        // Complete progress before showing results
        completeProgress();

        // Small delay to show completion, then display results
        setTimeout(() => {
            displayResults(response, analysisTime);
        }, 500);

    } catch (error) {
        console.error('Analysis failed:', error);

        // Reset timing on error
        analysisStartTime = null;
        analysisEndTime = null;

        showError(error.message);
        hideLoading();
    }
}

// Display the analysis results
function displayResults(response, analysisTime = null) {
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

    html += `</div>`;

    // Add component scoring breakdown if available
    if (data.dynamicScoring) {
        const ds = data.dynamicScoring;
        html += `
            <div class="component-breakdown">
                <h3>📊 Safety Analysis Breakdown</h3>
                <div class="component-scores">
                    <div class="component-item">
                        <div class="component-header">
                            <span class="component-icon">📹</span>
                            <span class="component-label">Video Content</span>
                            <span class="component-score">${ds.component_scores.video_content}/100</span>
                        </div>
                        <div class="component-weight">70% of final score</div>
                    </div>
                    <div class="component-item">
                        <div class="component-header">
                            <span class="component-icon">💬</span>
                            <span class="component-label">Comments</span>
                            <span class="component-score">${ds.component_scores.comments}/100</span>
                        </div>
                        <div class="component-weight">20% of final score</div>
                    </div>
                    <div class="component-item">
                        <div class="component-header">
                            <span class="component-icon">🌐</span>
                            <span class="component-label">Web Reputation</span>
                            <span class="component-score">${ds.component_scores.web_reputation}/100</span>
                        </div>
                        <div class="component-weight">10% of final score</div>
                    </div>
                </div>
            </div>`;
    }

    html += `
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

        if (data.channel && data.channel !== 'Unknown Channel') {
            html += `
                <div class="analysis-section">
                    <h4>📺 Channel Information</h4>
                    <p class="channel-info">${data.channel}</p>
                </div>`;
        }

        if (data.webReputation) {
            const formattedWebReputation = formatBulletPoints(data.webReputation);
            html += `
                <div class="analysis-section">
                    <h4>🌐 Web Reputation</h4>
                    ${formattedWebReputation}
                </div>`;
        }

        if (data.commentAnalysis) {
            const isError = data.commentAnalysis.includes('disabled') ||
                           data.commentAnalysis.includes('not configured') ||
                           data.commentAnalysis.includes('Error') ||
                           data.commentAnalysis.includes('quota exceeded');

            if (isError) {
                html += `
                    <div class="analysis-section">
                        <h4>💬 Comment Analysis</h4>
                        <p style="color: #856404; font-style: italic;">${data.commentAnalysis}</p>
                    </div>`;
            } else {
                const formattedCommentAnalysis = formatBulletPoints(data.commentAnalysis);
                html += `
                    <div class="analysis-section">
                        <h4>💬 Comment Analysis</h4>
                        ${formattedCommentAnalysis}
                    </div>`;
            }
        }

        html += `</div>`;
    }

    // Add analysis timing information
    if (analysisTime) {
        const formattedTime = formatAnalysisTime(analysisTime);

        html += `
            <div class="analysis-timing">
                <div class="timing-info">
                    <span class="timing-icon">⏱️</span>
                    <span class="timing-text">Analysis completed in ${formattedTime}</span>
                </div>
            </div>`;
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

function formatAnalysisTime(milliseconds) {
    const seconds = milliseconds / 1000;

    if (seconds < 60) {
        return `${seconds.toFixed(1)}s`;
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}m ${remainingSeconds}s`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    }
}

function formatBulletPoints(text) {
    if (!text) return '<p>No information available</p>';

    // Common bullet point patterns to detect
    const bulletPatterns = [
        /^[•·▪▫‣⁃]\s*/gm,  // Unicode bullet points
        /^[-*]\s*/gm,       // Dash or asterisk bullets
        /^\d+\.\s*/gm,      // Numbered lists
        /^[a-zA-Z]\.\s*/gm  // Letter bullets
    ];

    // Check if text contains bullet points
    const hasBullets = bulletPatterns.some(pattern => pattern.test(text));

    if (hasBullets) {
        // Split text into lines
        const lines = text.split('\n').map(line => line.trim()).filter(line => line.length > 0);

        let html = '';
        let currentList = [];
        let inList = false;

        for (const line of lines) {
            // Check if line is a bullet point
            const isBullet = bulletPatterns.some(pattern => pattern.test(line));

            if (isBullet) {
                // Remove bullet characters and add to list
                let cleanLine = line;
                bulletPatterns.forEach(pattern => {
                    cleanLine = cleanLine.replace(pattern, '');
                });
                currentList.push(cleanLine.trim());
                inList = true;
            } else {
                // If we were in a list, close it
                if (inList && currentList.length > 0) {
                    html += '<ul class="bullet-list">';
                    currentList.forEach(item => {
                        html += `<li>${item}</li>`;
                    });
                    html += '</ul>';
                    currentList = [];
                    inList = false;
                }

                // Add regular paragraph
                if (line.length > 0) {
                    html += `<p>${line}</p>`;
                }
            }
        }

        // Close any remaining list
        if (inList && currentList.length > 0) {
            html += '<ul class="bullet-list">';
            currentList.forEach(item => {
                html += `<li>${item}</li>`;
            });
            html += '</ul>';
        }

        return html || `<p>${text}</p>`;
    } else {
        // No bullet points detected, return as regular paragraph
        return `<p>${text}</p>`;
    }
}

function getCategoryMax(name) {
    const maxScores = {
        // API returns these category names - All out of 10
        'Violence': 10,
        'Language': 10,
        'ScaryContent': 10,
        'Scary Content': 10,
        'SexualContent': 10,
        'Sexual Content': 10,
        'SubstanceUse': 10,
        'Substance Use': 10,
        'DangerousBehavior': 10,
        'Dangerous Behavior': 10,
        'EducationalValue': 10,
        'Educational Value': 10,
        // Legacy names for backward compatibility - All out of 10
        'Non-Violence': 10,
        'Appropriate Language': 10,
        'Non-Scary Content': 10,
        'Family-Friendly Content': 10,
        'Substance-Free': 10,
        'Safe Behavior': 10
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

    // Start progress simulation
    startProgressSimulation();
}

function hideLoading() {
    hideElement('loading');
    resetProgress();

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
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 6px;
        margin: 12px 0;
    }

    .category {
        background: white;
        padding: 8px;
        border-radius: 6px;
        border-left: 3px solid #667eea;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .category-name {
        font-weight: bold;
        color: #333;
        font-size: 0.8em;
        line-height: 1.2;
        margin-bottom: 2px;
    }

    .category-score {
        font-size: 1.1em;
        font-weight: bold;
        color: #667eea;
    }

    .component-breakdown {
        background: white;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }

    .component-breakdown h3 {
        color: #667eea;
        margin-bottom: 15px;
        font-size: 1.1em;
    }

    .component-scores {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    .component-item {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 3px solid #667eea;
    }

    .component-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
    }

    .component-icon {
        font-size: 1.2em;
        margin-right: 8px;
    }

    .component-label {
        font-weight: 600;
        color: #495057;
        flex-grow: 1;
    }

    .component-score {
        font-weight: bold;
        color: #667eea;
        font-size: 1.1em;
    }

    .component-weight {
        font-size: 0.85em;
        color: #6c757d;
        font-style: italic;
        margin-left: 28px;
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

    .channel-info {
        font-weight: bold;
        color: #495057;
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

// Progress bar functionality
let progressInterval = null;
let currentProgress = 0;
let analysisStartTime = null;
let analysisEndTime = null;

function startProgressSimulation() {
    currentProgress = 0;
    updateProgress(0, 'Initializing analysis...');

    // Simulate realistic progress stages
    const progressStages = [
        { progress: 10, text: 'Downloading video metadata...', delay: 500 },
        { progress: 25, text: 'Extracting video frames...', delay: 1000 },
        { progress: 40, text: 'Processing audio content...', delay: 1500 },
        { progress: 55, text: 'Analyzing visual content...', delay: 2000 },
        { progress: 70, text: 'Running AI safety analysis...', delay: 2500 },
        { progress: 85, text: 'Generating safety report...', delay: 1000 },
        { progress: 95, text: 'Finalizing results...', delay: 500 }
    ];

    let stageIndex = 0;

    progressInterval = setInterval(() => {
        if (stageIndex < progressStages.length) {
            const stage = progressStages[stageIndex];
            updateProgress(stage.progress, stage.text);
            stageIndex++;
        } else {
            // Keep at 95% until actual completion
            updateProgress(95, 'Almost done...');
        }
    }, 800);
}

function updateProgress(percentage, text) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    if (progressFill) {
        progressFill.style.width = percentage + '%';
    }

    if (progressText) {
        progressText.textContent = text;
    }

    currentProgress = percentage;
}

function completeProgress() {
    updateProgress(100, 'Analysis complete!');

    // Clear the interval
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }

    // Brief delay to show completion
    setTimeout(() => {
        hideLoading();
    }, 500);
}

function resetProgress() {
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    currentProgress = 0;
    analysisStartTime = null;
    analysisEndTime = null;
    updateProgress(0, 'Ready to analyze...');
}

console.log('nsfK? Popup initialized');