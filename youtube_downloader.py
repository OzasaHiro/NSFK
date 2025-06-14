"""
YouTube video download functionality
Improved for prototype based on youtube_dl.py from Reference.md
"""
import yt_dlp
import os
from typing import Optional

class YouTubeDownloader:
    def __init__(self, output_path: str = "./videos"):
        self.output_path = output_path
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        if "youtube.com/watch?v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        else:
            # If URL is already a video ID
            return url if len(url) == 11 else None
    
    def download_video(self, url_or_id: str) -> tuple[bool, str, dict]:
        """
        Download YouTube video
        
        Args:
            url_or_id: YouTube URL or video ID
            
        Returns:
            tuple: (success flag, file path, video info)
        """
        video_id = self.extract_video_id(url_or_id)
        if not video_id:
            return False, "", {"error": "Invalid YouTube URL or ID"}
        
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        try:
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'best[height<=720]/best',  # Best quality up to 720p
                'outtmpl': f'{self.output_path}/{video_id}.%(ext)s',
                'noplaylist': True,  # Ignore playlists
                'extract_flat': False,
            }

            # Execute download
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video information
                info = ydl.extract_info(video_url, download=False)
                video_info = {
                    "title": info.get("title", "Unknown"),
                    "duration": info.get("duration", 0),
                    "description": info.get("description", ""),
                    "uploader": info.get("uploader", "Unknown"),
                    "view_count": info.get("view_count", 0),
                    "upload_date": info.get("upload_date", ""),
                }
                
                # Execute download
                ydl.download([video_url])
                
                # Identify the path of the downloaded file
                for ext in ['mp4', 'webm', 'mkv']:
                    file_path = f'{self.output_path}/{video_id}.{ext}'
                    if os.path.exists(file_path):
                        print(f"Download completed: {file_path}")
                        return True, file_path, video_info
                
                return False, "", {"error": "Downloaded file not found"}
                
        except Exception as e:
            print(f"Download error: {e}")
            return False, "", {"error": str(e)}

if __name__ == "__main__":
    # For testing
    downloader = YouTubeDownloader()
    
    # Test video ID (short video)
    test_video_id = "dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up
    
    success, file_path, info = downloader.download_video(test_video_id)
    if success:
        print(f"✅ Download successful: {file_path}")
        print(f"📝 Title: {info['title']}")
        print(f"⏱️ Duration: {info['duration']}s")
    else:
        print(f"❌ Download failed: {info.get('error', 'Unknown error')}")