import os
import cv2
import time
import json
import numpy as np
import requests
from PIL import Image
import io
from datetime import datetime
from pathlib import Path
import traceback

class MultimodalLLMAnalyzer:
    """
    Class for analyzing warehouse CCTV footage using multimodal LLMs via Hugging Face.
    """
    
    def __init__(self, output_dir="outputs"):
        """
        Initialize the LLM analyzer.
        
        Args:
            output_dir (str): Base directory for saving outputs
        """
        self.output_dir = output_dir
        
        # Create output directories for LLM analysis
        self.llm_analysis_dir = os.path.join(output_dir, "llm_analysis")
        self.frame_captures_dir = os.path.join(output_dir, "frame_captures")
        
        os.makedirs(self.llm_analysis_dir, exist_ok=True)
        os.makedirs(self.frame_captures_dir, exist_ok=True)
        
        # Store latest results
        self.latest_results = None
        
        # Hugging Face API settings
        self.api_token = os.environ.get("HF_API_TOKEN")
        if not self.api_token:
            print("Warning: HF_API_TOKEN environment variable not set. LLM analysis will not work.")
            print("Please create a .env file with your HF_API_TOKEN or set it in your environment.")
        
        # Default model or from environment
        self.model_id = os.environ.get("HF_MODEL_ID", "Salesforce/blip-itm-base-coco")
        print(f"Using Hugging Face model: {self.model_id}")
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        
        # Note: LLM analyzer will now use VideoManager from API for index management
    
    def query_huggingface(self, image_path, prompt):
        """
        Query the Hugging Face API with an image and text prompt.
        
        Args:
            image_path (str): Path to the image file
            prompt (str): Text prompt for the model
            
        Returns:
            dict: The JSON response from the API
        """
        if not self.api_token:
            return {"error": "API token not configured. Set the HF_API_TOKEN environment variable."}
        
        headers = {
            "Authorization": f"Bearer {self.api_token}"
        }
        
        try:
            with open(image_path, "rb") as img_file:
                data = {
                    "inputs": prompt,
                }
                files = {
                    "image": img_file.read(),
                }
                response = requests.post(self.api_url, headers=headers, data=data, files=files)
                
            if response.status_code != 200:
                return {"error": f"API request failed with status code: {response.status_code}", "details": response.text}
                
            return response.json()
        except Exception as e:
            print(f"Error in API request: {str(e)}")
            return {"error": str(e)}
    
    def extract_and_analyze_frames(self, video_path, frame_interval=30):
        """
        Extract key frames from video and analyze with multimodal LLM.
        
        Args:
            video_path (str): Path to the video file
            frame_interval (int): Interval between frames to analyze
            
        Returns:
            dict: Analysis results
        """
        try:
            print(f"Extracting frames from video: {video_path}")
            
            # Create a timestamp-based directory for frame captures
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            frames_dir = os.path.join(self.frame_captures_dir, f"{video_name}_{timestamp}")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize results
            analysis_results = {
                "video_path": video_path,
                "frame_analyses": [],
                "summary": None,
                "timestamp": datetime.now().isoformat()
            }
            
            # Extract and analyze frames
            frame_count = 0
            analyzed_frames = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame
                if frame_count % frame_interval == 0:
                    # Calculate timestamp
                    timestamp_sec = frame_count / fps
                    
                    # Save frame as image
                    frame_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    print(f"Analyzing frame {frame_count} at {timestamp_sec:.2f}s")
                    
                    # Create prompt for this frame
                    prompt = f"""
                    Analyze this frame from a warehouse surveillance video at timestamp {timestamp_sec:.2f} seconds.
                    
                    Describe:
                    1. How many employees are visible
                    2. What activities they appear to be performing
                    3. Any safety concerns or inefficiencies you notice
                    4. The general workflow patterns visible
                    
                    Be specific, clear and concise.
                    """
                    
                    # Query multimodal LLM
                    llm_response = self.query_huggingface(frame_path, prompt)
                    
                    # Store analysis
                    frame_analysis = {
                        "timestamp": timestamp_sec,
                        "frame_idx": frame_count,
                        "frame_path": frame_path,
                        "llm_response": llm_response
                    }
                    
                    analysis_results["frame_analyses"].append(frame_analysis)
                    analyzed_frames += 1
                
                frame_count += 1
                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
            
            cap.release()
            print(f"Completed frame extraction and analysis. Analyzed {analyzed_frames} frames.")
            
            # Generate overall summary if we have frame analyses
            if analysis_results["frame_analyses"]:
                summary_prompt = self.generate_summary_prompt(analysis_results["frame_analyses"])
                # For summary, we can use text-only LLM or just pass the last frame with the summary prompt
                if analysis_results["frame_analyses"]:
                    last_frame_path = analysis_results["frame_analyses"][-1]["frame_path"]
                    summary_response = self.query_huggingface(last_frame_path, summary_prompt)
                    analysis_results["summary"] = summary_response
            
            # Store results
            self.latest_results = analysis_results
            
            # Save results to file for future use
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filename = f"llm_analysis_{video_name}_{timestamp}.json"
            results_path = os.path.join(self.llm_analysis_dir, results_filename)
            
            with open(results_path, 'w') as f:
                json.dump(analysis_results, f, indent=4)
            
            print(f"LLM Analysis saved to: {results_path}")
            
            return analysis_results
            
        except Exception as e:
            print(f"Error in extract_and_analyze_frames: {str(e)}")
            print(traceback.format_exc())
            return {"error": str(e), "video_path": video_path}
    
    def generate_summary_prompt(self, frame_analyses):
        """Generate a prompt for summarizing frame analyses."""
        # Extract text from frame analyses
        analysis_texts = []
        for analysis in frame_analyses:
            if "llm_response" in analysis and isinstance(analysis["llm_response"], list):
                for item in analysis["llm_response"]:
                    if "generated_text" in item:
                        analysis_texts.append(f"Timestamp {analysis['timestamp']:.2f}s: {item['generated_text']}")
        
        # Combine analyses (limit to avoid token limits)
        combined_analyses = "\n\n".join(analysis_texts[:10])  # Limit to first 10 analyses
        
        # Create summary prompt
        prompt = f"""
        Based on these observations from a warehouse surveillance video, provide an overall analysis:
        
        {combined_analyses}
        
        Summarize:
        1. The general workforce activities observed
        2. Any patterns in employee behavior
        3. Safety concerns or inefficiencies
        4. Recommendations for improving workflow
        
        Be concise and focus on actionable insights.
        """
        
        return prompt
        
    def get_latest_results(self):
        """Get the latest LLM analysis results."""
        return self.latest_results 