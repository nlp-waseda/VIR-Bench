import os
import time
from typing import Dict, Any
import google.generativeai as genai


class VideoAnalyzer:
    """Class to upload videos to Gemini and return file reference"""
    
    def __init__(self):
        # Configure genai
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
        else:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
    
    def upload_video(self, video_path: str) -> Dict[str, Any]:
        """Upload video to Gemini and return file reference

        Args:
            video_path: Path to video file
        """
        try:
            print(f"🎥 Uploading video: {video_path}")
            
            # Check if video file exists
            if not os.path.exists(video_path):
                return {
                    "success": False,
                    "error": f"Video file not found: {video_path}",
                    "video_file": None
                }
            
            print("📤 Uploading video to Gemini...")
            
            # Upload video file to Gemini
            video_file = genai.upload_file(video_path)
            print(f"✅ Upload completed: {video_file.uri}")
            
            # Wait for file to be processed
            while video_file.state.name == "PROCESSING":
                print("⏳ Processing video...")
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                return {
                    "success": False,
                    "error": "Video processing failed",
                    "video_file": None
                }
            
            print("✅ Video ready")
            
            return {
                "success": True,
                "video_file": video_file,
                "video_file_name": video_file.name,
                "video_file_uri": video_file.uri
            }
            
        except Exception as e:
            print(f"❌ Video upload error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "video_file": None
            }