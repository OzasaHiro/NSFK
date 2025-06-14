// ===========================================
// content.js
// ===========================================
// This script runs on YouTube pages and detects video changes

(function() {
    'use strict';
    
    let currentVideoId = null;
    let isYouTubePage = window.location.hostname === 'www.youtube.com' || window.location.hostname === 'youtube.com';
    
    console.log('nsfK? Content script loaded on:', window.location.href);
    
    // Check if we're on a YouTube video page
    function isVideoPage() {
        return window.location.pathname === '/watch' && window.location.search.includes('v=');
    }
    
    // Extract video ID from current URL
    function getCurrentVideoId() {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('v');
    }
    
    // Get video metadata from the page
    function getVideoMetadata() {
        try {
            const title = document.querySelector('h1.ytd-video-primary-info-renderer')?.textContent?.trim() ||
                         document.querySelector('h1.title')?.textContent?.trim() ||
                         document.querySelector('#title h1')?.textContent?.trim();
            
            const channel = document.querySelector('#channel-name a')?.textContent?.trim() ||
                           document.querySelector('.ytd-channel-name a')?.textContent?.trim();
            
            const views = document.querySelector('#count .view-count')?.textContent?.trim() ||
                         document.querySelector('.view-count')?.textContent?.trim();
            
            const duration = document.querySelector('.ytp-time-duration')?.textContent?.trim();
            
            const thumbnail = document.querySelector('video')?.poster ||
                             document.querySelector('.html5-video-container video')?.poster;
            
            return {
                title: title || 'Unknown Title',
                channel: channel || 'Unknown Channel',
                views: views || '0 views',
                duration: duration || '0:00',
                thumbnail: thumbnail || null,
                url: window.location.href,
                videoId: getCurrentVideoId()
            };
        } catch (error) {
            console.error('nsfK? Error getting video metadata:', error);
            return {
                title: 'Error loading title',
                channel: 'Error loading channel',
                views: '0 views',
                duration: '0:00',
                thumbnail: null,
                url: window.location.href,
                videoId: getCurrentVideoId()
            };
        }
    }
    
    // Send video change notification to extension
    function notifyVideoChange(videoId) {
        const metadata = getVideoMetadata();
        
        // Send message to background script
        chrome.runtime.sendMessage({
            type: 'YOUTUBE_VIDEO_DETECTED',
            videoId: videoId,
            metadata: metadata,
            timestamp: Date.now()
        }).catch(error => {
            console.error('nsfK? Error sending message to background:', error);
        });
        
        console.log('nsfK? Video detected:', videoId, metadata.title);
    }
    
    // Handle URL changes (YouTube is a SPA)
    function handleUrlChange() {
        if (!isYouTubePage) return;
        
        const newVideoId = getCurrentVideoId();
        
        if (isVideoPage() && newVideoId && newVideoId !== currentVideoId) {
            currentVideoId = newVideoId;
            
            // Wait a bit for page to load content
            setTimeout(() => {
                notifyVideoChange(newVideoId);
            }, 1000);
        } else if (!isVideoPage()) {
            currentVideoId = null;
            
            // Notify that we left a video page
            chrome.runtime.sendMessage({
                type: 'YOUTUBE_VIDEO_LEFT',
                timestamp: Date.now()
            }).catch(error => {
                console.error('nsfK? Error sending leave message:', error);
            });
        }
    }
    
    // Listen for URL changes (YouTube uses pushState/replaceState)
    let lastUrl = location.href;
    new MutationObserver(() => {
        const url = location.href;
        if (url !== lastUrl) {
            lastUrl = url;
            setTimeout(handleUrlChange, 500); // Give YouTube time to update
        }
    }).observe(document, { subtree: true, childList: true });
    
    // Initial check when script loads
    setTimeout(handleUrlChange, 1000);
    
    // Listen for messages from popup
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        if (message.type === 'GET_CURRENT_VIDEO') {
            const videoId = getCurrentVideoId();
            const metadata = isVideoPage() ? getVideoMetadata() : null;
            
            sendResponse({
                success: true,
                isVideoPage: isVideoPage(),
                videoId: videoId,
                metadata: metadata
            });
        }
        
        if (message.type === 'REFRESH_VIDEO_DATA') {
            if (isVideoPage()) {
                const videoId = getCurrentVideoId();
                if (videoId) {
                    setTimeout(() => {
                        notifyVideoChange(videoId);
                    }, 500);
                }
                sendResponse({ success: true });
            } else {
                sendResponse({ success: false, error: 'Not on video page' });
            }
        }
        
        return true; // Keep message channel open for async response
    });
    
    // Add visual indicator when nsfK is active (optional)
    function addVisualIndicator() {
        if (!isVideoPage() || document.getElementById('nsfk-indicator')) return;
        
        const indicator = document.createElement('div');
        indicator.id = 'nsfk-indicator';
        indicator.innerHTML = '🛡️ nsfK?';
        indicator.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            background: linear-gradient(135deg, #4F46E5, #7C3AED);
            color: white;
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            z-index: 9999;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
            opacity: 0.8;
            transition: opacity 0.3s ease;
            cursor: pointer;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        `;
        
        indicator.onmouseover = () => indicator.style.opacity = '1';
        indicator.onmouseout = () => indicator.style.opacity = '0.8';
        indicator.onclick = () => {
            chrome.runtime.sendMessage({ type: 'OPEN_POPUP' });
        };
        
        document.body.appendChild(indicator);
        
        // Remove indicator after 3 seconds
        setTimeout(() => {
            if (indicator.parentNode) {
                indicator.style.opacity = '0';
                setTimeout(() => indicator.remove(), 300);
            }
        }, 3000);
    }
    
    // Show indicator when video is detected
    if (isVideoPage()) {
        setTimeout(addVisualIndicator, 2000);
    }
    
    console.log('nsfK? Content script initialized');
})();