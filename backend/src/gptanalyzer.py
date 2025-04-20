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
import base64

# Import the new detailed captions module
from .detailed_captions import generate_detailed_caption

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
        self.model_id = os.environ.get("HF_MODEL_ID", "Salesforce/blip-image-captioning-large")
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
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Exponential backoff parameters
        max_retries = 5
        base_delay = 1  # Start with 1 second delay
        
        # Prepare the request payload once (outside the retry loop)
        try:
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create payload with base64-encoded image
            payload = {
                "inputs": {
                    "image": image_base64,
                    "text": prompt
                }
            }
        except Exception as e:
            print(f"Error preparing image: {str(e)}")
            print(traceback.format_exc())
            return {"error": f"Failed to process image: {str(e)}"}
        
        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                # Calculate delay with exponential backoff
                delay = base_delay * (2 ** attempt)
                
                if attempt > 0:
                    print(f"Retry attempt {attempt}/{max_retries-1} after {delay}s delay...")
                    time.sleep(delay)
                
                print(f"Sending request to: {self.api_url}")
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    json=payload,
                    timeout=60  # 60 second timeout
                )
                
                # Success case
                if response.status_code == 200:
                    try:
                        result = response.json()
                        print(f"Received successful response")
                        return result
                    except json.JSONDecodeError:
                        return {"text": response.text}
                
                # Handle specific error codes
                if response.status_code == 503:
                    print(f"Server unavailable (503). Will retry after delay.")
                    # Continue to next iteration (which has delay)
                    continue
                
                # Other error codes - don't retry
                print(f"API request failed: {response.status_code} - {response.text}")
                return {"error": f"API request failed with status code: {response.status_code}", "details": response.text}
                
            except requests.exceptions.RequestException as e:
                print(f"Request exception: {str(e)}")
                # Network errors should be retried
                if attempt < max_retries - 1:
                    continue
                else:
                    return {"error": f"Network error after {max_retries} attempts: {str(e)}"}
        
        # If we've exhausted all retries
        return {"error": f"Failed after {max_retries} retry attempts. Last error was service unavailable (503)."}
    
    def generate_summary_prompt(self, frame_analyses):
        """Generate a prompt for summarizing frame analyses.
        
        Args:
            frame_analyses (list): List of frame analysis dictionaries
            
        Returns:
            str: A prompt for the summary model
        """
        # Extract text from frame analyses
        analysis_texts = []
        
        for analysis in frame_analyses:
            timestamp = analysis.get("timestamp", 0)
            
            # Process regular LLM responses
            if "llm_response" in analysis:
                response = analysis["llm_response"]
                
                # Skip if it's an error
                if isinstance(response, dict) and "error" in response:
                    continue
                    
                # Try to extract text based on different response formats
                regular_text = None
                
                # Case 1: Response is a list of items with generated_text
                if isinstance(response, list):
                    for item in response:
                        if isinstance(item, dict) and "generated_text" in item:
                            regular_text = item["generated_text"]
                            break
                
                # Case 2: Response is a dict with generated_text directly
                elif isinstance(response, dict) and "generated_text" in response:
                    regular_text = response["generated_text"]
                    
                # Case 3: Response is a dict with text or output fields
                elif isinstance(response, dict):
                    for field in ["text", "output", "content", "caption"]:
                        if field in response:
                            regular_text = response[field]
                            break
                
                if regular_text:
                    analysis_texts.append(f"Timestamp {timestamp:.2f}s (Standard): {regular_text}")
            
            # Process detailed LLM responses from Florence model
            if "llm_detailed_response" in analysis:
                detailed_response = analysis["llm_detailed_response"]
                
                # Skip if it's an error
                if isinstance(detailed_response, dict) and "error" in detailed_response:
                    continue
                    
                # Try to extract text based on different response formats
                detailed_text = None
                
                # Case 1: Response is a list of items with generated_text
                if isinstance(detailed_response, list):
                    for item in detailed_response:
                        if isinstance(item, dict) and "generated_text" in item:
                            detailed_text = item["generated_text"]
                            break
                
                # Case 2: Response is a dict with generated_text directly
                elif isinstance(detailed_response, dict) and "generated_text" in detailed_response:
                    detailed_text = detailed_response["generated_text"]
                    
                # Case 3: Response is a dict with text or output fields
                elif isinstance(detailed_response, dict):
                    for field in ["text", "output", "content", "caption"]:
                        if field in detailed_response:
                            detailed_text = detailed_response[field]
                            break
                
                if detailed_text:
                    analysis_texts.append(f"Timestamp {timestamp:.2f}s (Detailed): {detailed_text}")
        
        # If we couldn't extract any text, provide a fallback
        if not analysis_texts:
            return """
            Please provide a summary of the warehouse surveillance footage.
            Since no specific observations were available, focus on general 
            insights about warehouse operations, safety considerations, 
            and efficiency recommendations.
            """
        
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

    def generate_detailed_summary(self, frame_analyses):
        """Generate a comprehensive summary of all detailed frame analyses using a local LLM.
        
        Args:
            frame_analyses (list): List of frame analysis dictionaries
            custom_prompt (str, optional): User-provided summarization prompt
            
        Returns:
            dict: Summary result from LLM
        """
        # Extract all detailed responses
        detailed_texts = []
        
        for analysis in frame_analyses:
            timestamp = analysis.get("timestamp", 0)
            
            if "llm_detailed_response" in analysis:
                detailed_response = analysis["llm_detailed_response"]
                
                # Extract the text from the response
                detailed_text = None
                
                # Handle different response formats
                if "error" not in detailed_response:
                    detailed_text = detailed_response
                
                if detailed_text:
                    detailed_texts.append(f"Timestamp {timestamp:.2f}s: {detailed_text}")
        
        # If no detailed texts found, return early
        if not detailed_texts:
            return {"error": "No detailed analysis found to summarize"}
        
        # Combine all detailed texts
        all_detailed_text = "\n\n".join(detailed_texts)
        
        
        prompt = f"""
            Based on these frame based descriptions from a surveillance video:
            
            {all_detailed_text}
            
            Please provide:
            1. A comprehensive summary of employee activities
            2. Key observations about worker behavior and patterns
            3. Safety concerns identified
            4. Efficiency recommendations
            5. Any unusual or noteworthy events
            """
        
        # Use text-based LLM for summarization
        summary_result = self._call_summary_llm(prompt)

        print(f"Detailed Summary result: {summary_result}")
        
        return summary_result

    def _call_summary_llm(self, prompt):
        """
        Call a text-based LLM via Hugging Face API for summarization of detailed analyses.
        
        Args:
            prompt (str): Text prompt containing detailed analyses
            
        Returns:
            dict: Summary result from the LLM
        """
        if not self.api_token:
            return {
                "error": "API token not configured. Set the HF_API_TOKEN environment variable.",
                "summary": "No summary available - API token missing",
                "generated_at": datetime.now().isoformat()
            }
        
        max_retries = 5
        base_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                from huggingface_hub import InferenceClient
            
                # Configure the retry strategy explicitly
                # Default is already exponential backoff, but we can customize it

                # Create client with explicit retry config
                client = InferenceClient(
                    token=self.api_token
                    
                )
                print(f"prompt: {prompt}")
                # Actual API call with the retry mechanism active
                generated_text = client.text_generation(prompt=prompt, model="mistralai/Mistral-7B-Instruct-v0.3")

                return {
                    "summary": generated_text,
                    "generated_at": datetime.now().isoformat(),
                    "model_name": "mistralai/Mistral-7B-Instruct-v0.3"
                }
            except Exception as e:
                print(f"Error in _call_summary_llm: {str(e)}")
                print(traceback.format_exc())
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    attempt += 1
                    continue
                else:
                    return {
                    "error": f"Error in _call_summary_llm: {str(e)}",
                    "summary": "Summary generation failed",
                    "generated_at": datetime.now().isoformat(),
                    "model_name": "huggingface_hub"
                }
    
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
                    
                    # Verify image was saved correctly
                    if not os.path.exists(frame_path) or os.path.getsize(frame_path) == 0:
                        print(f"Warning: Failed to save frame at {frame_path} or file is empty")
                        continue
                    
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
                    
                    # Query primary LLM (original analysis)
                    llm_response = self.query_huggingface(frame_path, prompt)
                    
                    # Get detailed caption using Florence model
                    print(f"Getting detailed caption for frame {frame_count}...")
                    llm_detailed_response = generate_detailed_caption(
                        frame_path, 
                        instruction="<MIXED_CAPTION_PLUS>"
                    )
                    
                    # Store analysis with both responses
                    frame_analysis = {
                        "timestamp": timestamp_sec,
                        "frame_idx": frame_count,
                        "frame_path": frame_path,
                        "llm_response": llm_response,
                        "llm_detailed_response": llm_detailed_response
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
            
            # Generate detailed summary with custom prompt (can be configured externally)
            detailed_summary = self.generate_detailed_summary(analysis_results["frame_analyses"])
            analysis_results["detailed_summary"] = detailed_summary
            
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
