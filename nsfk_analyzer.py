#!/usr/bin/env python3
"""
NSFK Video Analyzer - Integration Script
Combines YouTube downloading and video safety analysis
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the downloader
from youtube_downloader import YouTubeDownloader

# Import analyzer components
import cv2
import tempfile
import base64
import numpy as np
from pydub import AudioSegment
import whisper
import aiohttp
import nest_asyncio

# Apply nest_asyncio for Jupyter/interactive environments
nest_asyncio.apply()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file")
    sys.exit(1)


class NSFKAnalyzer:
    """Main analyzer class for NSFK project"""
    
    def __init__(self):
        self.downloader = YouTubeDownloader()
        self.whisper_model = None
        
    def load_whisper_model(self):
        """Load Whisper model for audio transcription"""
        if not self.whisper_model:
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            print("Whisper model loaded.")
            
    def detect_scene_changes_opencv(self, video_filepath: str, threshold: int = 2000000) -> set:
        """Detect scene changes using OpenCV frame differencing"""
        print("Detecting scene changes...")
        scene_change_frames = set()
        cap = cv2.VideoCapture(video_filepath)
        
        if not cap.isOpened():
            print(f"Error: Could not open video for scene detection: {video_filepath}")
            return scene_change_frames
            
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return scene_change_frames
            
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_frame_number = 0
        
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break
                
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(current_gray, prev_gray)
            diff_sum = np.sum(frame_diff)
            
            if diff_sum > threshold:
                scene_change_frames.add(current_frame_number)
                if current_frame_number > 0:
                    scene_change_frames.add(current_frame_number - 1)
                    
            prev_gray = current_gray
            current_frame_number += 1
            
        cap.release()
        print(f"Detected {len(scene_change_frames)} scene changes.")
        return scene_change_frames
        
    def extract_frames_and_audio(self, video_filepath: str, temp_dir: str, 
                                frames_per_second: float = 0.2) -> tuple:
        """Extract frames and audio from video file"""
        if not os.path.exists(video_filepath):
            print(f"Error: Video file not found at '{video_filepath}'")
            return [], None
            
        cap = cv2.VideoCapture(video_filepath)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{video_filepath}'")
            return [], None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration_seconds = total_frames / fps
        
        print(f"Video Info: Total frames={total_frames}, FPS={fps:.2f}, Duration={video_duration_seconds:.2f}s")
        
        # Detect scene changes
        scene_change_frame_numbers = self.detect_scene_changes_opencv(video_filepath)
        
        # Re-open video for frame extraction
        cap = cv2.VideoCapture(video_filepath)
        if not cap.isOpened():
            print(f"Error: Could not re-open video file")
            return [], None
            
        sampled_frames_data = {}
        frame_interval = int(fps / frames_per_second) if frames_per_second > 0 else 1
        if frame_interval == 0:
            frame_interval = 1
            
        current_frame_number = 0
        success, image = cap.read()
        
        print(f"Extracting frames (every {frame_interval}th frame + {len(scene_change_frame_numbers)} scene changes)...")
        
        while success:
            if (current_frame_number % frame_interval == 0) or (current_frame_number in scene_change_frame_numbers):
                _, buffer = cv2.imencode('.jpg', image)
                sampled_frames_data[current_frame_number] = base64.b64encode(buffer).decode('utf-8')
                
            success, image = cap.read()
            current_frame_number += 1
            
        cap.release()
        
        # Sort frames chronologically
        sorted_frame_numbers = sorted(sampled_frames_data.keys())
        final_sampled_frames_base64 = [sampled_frames_data[fn] for fn in sorted_frame_numbers]
        
        print(f"Total unique frames extracted: {len(final_sampled_frames_base64)}")
        
        # Extract audio
        audio_output_filepath = os.path.join(temp_dir, "extracted_audio.wav")
        print(f"Extracting audio...")
        
        try:
            audio = AudioSegment.from_file(video_filepath)
            audio.export(audio_output_filepath, format="wav")
            print("Audio extraction complete.")
            return final_sampled_frames_base64, audio_output_filepath
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return final_sampled_frames_base64, None
            
    def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribe audio using Whisper"""
        if not audio_file_path or not os.path.exists(audio_file_path):
            print("No audio file for transcription.")
            return ""
            
        self.load_whisper_model()
        
        print("Transcribing audio...")
        try:
            result = self.whisper_model.transcribe(audio_file_path)
            transcription_text = result["text"]
            print("Transcription complete.")
            return transcription_text
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""
            
    async def analyze_frame_with_gemini(self, session: aiohttp.ClientSession, 
                                      base64_image: str, frame_number: int, 
                                      video_title: str) -> str:
        """Analyze a single frame with Gemini for safety"""
        headers = {'Content-Type': 'application/json'}
        
        prompt_text = (
            f"Analyze this image from the video titled '{video_title}'. "
            f"Specifically describe anything that could be potentially offensive, inappropriate, or unsafe "
            f"for young children (e.g., violence, weapons, nudity, explicit content, drug/alcohol use, "
            f"dangerous situations, scary imagery, jump scares, disturbing visual effects). "
            f"If the image appears completely safe and normal for kids, respond with 'SAFE_FOR_KIDS'."
        )
        
        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {"inlineData": {"mimeType": "image/jpeg", "data": base64_image}}
                ]
            }]
        }
        
        try:
            async with session.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", 
                                  headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                
                if (result.get('candidates') and 
                    result['candidates'][0].get('content') and 
                    result['candidates'][0]['content'].get('parts')):
                    
                    description = result['candidates'][0]['content']['parts'][0]['text'].strip()
                    if description == "SAFE_FOR_KIDS":
                        return None
                    else:
                        print(f"  Frame {frame_number}: Potential Issue: {description[:100]}...")
                        return f"Frame {frame_number}: {description}"
                else:
                    print(f"  Frame {frame_number}: No valid response from LLM.")
                    return f"Frame {frame_number}: No analysis available"
                    
        except Exception as e:
            print(f"  Frame {frame_number}: Error during analysis: {e}")
            return f"Frame {frame_number}: Analysis error"
            
    async def analyze_video_safety(self, video_filepath: str, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Main video safety analysis function"""
        video_title = video_info.get('title', 'Unknown Video')
        
        print(f"\n--- Starting Video Safety Analysis ---")
        print(f"Title: {video_title}")
        print(f"Duration: {video_info.get('duration', 0)}s")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract frames and audio
            frames_base64, audio_filepath = self.extract_frames_and_audio(
                video_filepath, temp_dir, frames_per_second=0.2
            )
            
            # Transcribe audio
            audio_transcript = self.transcribe_audio(audio_filepath)
            
            # Analyze frames with rate limiting
            print("\n--- Analyzing video frames for unsafe content ---")
            visual_safety_observations = []
            
            # Limit to analyzing max 30 frames to avoid rate limits
            frames_to_analyze = frames_base64[:30] if len(frames_base64) > 30 else frames_base64
            print(f"Analyzing {len(frames_to_analyze)} frames (sampled from {len(frames_base64)} total)")
            
            async with aiohttp.ClientSession() as session:
                # Process in batches of 5 to avoid rate limits
                batch_size = 5
                for i in range(0, len(frames_to_analyze), batch_size):
                    batch = frames_to_analyze[i:i+batch_size]
                    tasks = []
                    
                    for j, frame_data in enumerate(batch):
                        frame_number = i + j + 1
                        task = self.analyze_frame_with_gemini(
                            session, frame_data, frame_number, video_title
                        )
                        tasks.append(task)
                    
                    batch_results = await asyncio.gather(*tasks)
                    visual_safety_observations.extend([obs for obs in batch_results if obs is not None])
                    
                    # Add delay between batches to avoid rate limits
                    if i + batch_size < len(frames_to_analyze):
                        await asyncio.sleep(1)
                
            # Generate final safety assessment
            report_data = await self.generate_safety_report(
                video_title, audio_transcript, visual_safety_observations
            )
            
            total_score = report_data['total_score']
            
            return {
                "video_url": f"https://youtube.com/watch?v={video_info.get('video_id', '')}",
                "title": video_title,
                "duration": video_info.get('duration', 0),
                "safety_score": total_score,
                "category_scores": report_data['category_scores'],
                "summary": report_data['summary'],
                "risk_factors": visual_safety_observations[:5],  # Top 5 issues
                "keywords": report_data['keywords'],
                "recommendation": "Safe" if total_score >= 80 else "Review Required" if total_score >= 60 else "Not Recommended",
                "audio_transcript": audio_transcript[:500] + "..." if len(audio_transcript) > 500 else audio_transcript,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
    async def generate_safety_report(self, video_title: str, audio_transcript: str, 
                                   visual_observations: list) -> dict:
        """Generate final safety report using Gemini"""
        print("\n--- Generating final safety assessment ---")
        
        combined_analysis = f"Video Title: '{video_title}'\n\n"
        combined_analysis += "--- Audio Transcript ---\n"
        combined_analysis += audio_transcript[:1000] + "...\n\n" if len(audio_transcript) > 1000 else audio_transcript + "\n\n"
        
        combined_analysis += "--- Visual Safety Observations ---\n"
        if visual_observations:
            for obs in visual_observations[:10]:  # Limit to 10 observations
                combined_analysis += obs + "\n"
        else:
            combined_analysis += "No unsafe visual content detected.\n"
            
        prompt = (
            "Based on the following video analysis, provide a detailed safety assessment with category scores.\n"
            "Each category should be scored based on how SAFE the content is (higher score = safer):\n\n"
            "Categories (all out of 10 points):\n"
            "- Violence: 10 points max (physical violence, fighting, weapons)\n"
            "- Language: 10 points max (profanity, inappropriate language)\n"
            "- Scary Content: 10 points max (horror elements, jump scares, frightening imagery)\n"
            "- Sexual Content: 10 points max (nudity, suggestive themes)\n"
            "- Substance Use: 10 points max (drugs, alcohol, smoking)\n"
            "- Dangerous Behavior: 10 points max (risky activities kids might imitate)\n"
            "- Educational Value: 10 points max (positive learning content)\n\n"
            "Provide your response as JSON with:\n"
            "- category_scores: object with each category and its score\n"
            "- total_score: sum of all category scores (out of 100)\n"
            "- keywords: array of 5 key content keywords\n"
            "- summary: 1-2 line summary for parents\n\n"
            f"Analysis:\n{combined_analysis}"
        )
        
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", 
                                      headers=headers, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    if (result.get('candidates') and 
                        result['candidates'][0].get('content') and 
                        result['candidates'][0]['content'].get('parts')):
                        
                        response_text = result['candidates'][0]['content']['parts'][0]['text'].strip()
                        
                        # Try to parse JSON response
                        try:
                            import re
                            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                            if json_match:
                                parsed = json.loads(json_match.group())
                                return {
                                    'category_scores': parsed.get('category_scores', {}),
                                    'total_score': parsed.get('total_score', 50),
                                    'keywords': parsed.get('keywords', []),
                                    'summary': parsed.get('summary', 'Analysis complete.')
                                }
                        except:
                            pass
                            
                        # Fallback if JSON parsing fails
                        return {
                            'category_scores': {
                                'Violence': 10, 'Language': 10, 'Scary Content': 10,
                                'Sexual Content': 10, 'Substance Use': 5, 
                                'Dangerous Behavior': 5, 'Educational Value': 0
                            },
                            'total_score': 50,
                            'keywords': ["content", "video"],
                            'summary': "Analysis complete. Manual review recommended."
                        }
                        
        except Exception as e:
            print(f"Error generating final report: {e}")
            return {
                'category_scores': {},
                'total_score': 50,
                'keywords': ["error"],
                'summary': "Analysis encountered an error."
            }
            
    async def analyze_youtube_video(self, youtube_url: str) -> Dict[str, Any]:
        """Complete pipeline: download and analyze YouTube video"""
        print(f"\n{'='*50}")
        print(f"NSFK Video Analyzer")
        print(f"{'='*50}")
        
        # Download video
        print("\n[1/3] Downloading video...")
        success, file_path, video_info = self.downloader.download_video(youtube_url)
        
        if not success:
            return {
                "error": f"Failed to download video: {video_info.get('error', 'Unknown error')}"
            }
            
        video_info['video_id'] = self.downloader.extract_video_id(youtube_url)
        
        # Analyze video
        print("\n[2/3] Analyzing video content...")
        try:
            analysis_result = await self.analyze_video_safety(file_path, video_info)
            
            # Generate report
            print("\n[3/3] Generating report...")
            report_path = self.save_report(analysis_result)
            analysis_result['report_path'] = report_path
            
            return analysis_result
            
        finally:
            # Clean up downloaded video
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"\nCleaned up temporary video file.")
                
    def save_report(self, analysis_result: Dict[str, Any]) -> str:
        """Save analysis report to file"""
        reports_dir = "reports"
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{reports_dir}/nsfk_report_{timestamp}.json"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            
        print(f"Report saved to: {report_filename}")
        
        # Also create a human-readable text report
        text_filename = f"{reports_dir}/nsfk_report_{timestamp}.txt"
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write(f"NSFK Video Safety Analysis Report\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Video Title: {analysis_result.get('title', 'Unknown')}\n")
            f.write(f"Video URL: {analysis_result.get('video_url', 'Unknown')}\n")
            f.write(f"Analysis Date: {analysis_result.get('analysis_timestamp', 'Unknown')}\n")
            f.write(f"Duration: {analysis_result.get('duration', 0)} seconds\n\n")
            
            f.write(f"Total Safety Score: {analysis_result.get('safety_score', 'N/A')}/100\n")
            f.write(f"Recommendation: {analysis_result.get('recommendation', 'Unknown')}\n\n")
            
            # Add category scores
            f.write(f"Category Scores:\n")
            category_scores = analysis_result.get('category_scores', {})
            if category_scores:
                f.write(f"- Violence: {category_scores.get('Violence', 0)}/10\n")
                f.write(f"- Language: {category_scores.get('Language', 0)}/10\n")
                f.write(f"- Scary Content: {category_scores.get('Scary Content', 0)}/10\n")
                f.write(f"- Sexual Content: {category_scores.get('Sexual Content', 0)}/10\n")
                f.write(f"- Substance Use: {category_scores.get('Substance Use', 0)}/10\n")
                f.write(f"- Dangerous Behavior: {category_scores.get('Dangerous Behavior', 0)}/10\n")
                f.write(f"- Educational Value: {category_scores.get('Educational Value', 0)}/10\n\n")
            else:
                f.write("No category scores available\n\n")
            
            f.write(f"Summary: {analysis_result.get('summary', 'No summary available')}\n\n")
            
            f.write(f"Keywords: {', '.join(analysis_result.get('keywords', []))}\n\n")
            
            if analysis_result.get('risk_factors'):
                f.write(f"Risk Factors:\n")
                for factor in analysis_result.get('risk_factors', []):
                    f.write(f"- {factor}\n")
                    
        print(f"Text report saved to: {text_filename}")
        
        return report_filename


async def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        youtube_url = sys.argv[1]
    else:
        youtube_url = input("Enter YouTube URL or video ID: ")
        
    analyzer = NSFKAnalyzer()
    result = await analyzer.analyze_youtube_video(youtube_url)
    
    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
    else:
        print(f"\n{'='*50}")
        print(f"Analysis Complete!")
        print(f"{'='*50}")
        print(f"Total Safety Score: {result['safety_score']}/100")
        print(f"Recommendation: {result['recommendation']}")
        
        # Display category scores
        print(f"\nCategory Scores:")
        category_scores = result.get('category_scores', {})
        if category_scores:
            print(f"- Violence: {category_scores.get('Violence', 0)}/10")
            print(f"- Language: {category_scores.get('Language', 0)}/10")
            print(f"- Scary Content: {category_scores.get('Scary Content', 0)}/10")
            print(f"- Sexual Content: {category_scores.get('Sexual Content', 0)}/10")
            print(f"- Substance Use: {category_scores.get('Substance Use', 0)}/10")
            print(f"- Dangerous Behavior: {category_scores.get('Dangerous Behavior', 0)}/10")
            print(f"- Educational Value: {category_scores.get('Educational Value', 0)}/10")
        
        print(f"\nSummary: {result['summary']}")
        print(f"Keywords: {', '.join(result['keywords'])}")
        print(f"\nFull report saved to: {result['report_path']}")


if __name__ == "__main__":
    asyncio.run(main())