import os
import json
import shutil
from datetime import datetime
from pathlib import Path


class VideoManager:
    """
    Class for managing video files, their indexes, and associated results.
    Handles video listings, result storage, and retrieval.
    """
    
    def __init__(self, data_dir="data"):
        """Initialize the video manager.
        
        Args:
            data_dir (str): Base directory for storing video data
        """
        self.data_dir = data_dir
        
        # Create data directories if they don't exist
        self.results_dir = os.path.join(data_dir, "results")
        self.heatmaps_dir = os.path.join(data_dir, "heatmaps")
        self.llm_results_dir = os.path.join(data_dir, "llm_results")
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.heatmaps_dir, exist_ok=True)
        os.makedirs(self.llm_results_dir, exist_ok=True)
        
        # Initialize video index and LLM analysis index
        self.initialize_video_index()
        self.initialize_llm_analysis_index()
        
    def initialize_video_index(self):
        """Initialize or load the video index file.
        
        Returns:
            dict: The video index data
        """
        index_path = os.path.join(self.data_dir, "video_index.json")
        
        if not os.path.exists(index_path):
            # Create empty index
            with open(index_path, "w") as f:
                json.dump({"videos": {}, "latest_video": None}, f)
        
        # Load the index
        with open(index_path, "r") as f:
            return json.load(f)
    
    def update_video_index(self, video_path, results, heatmap_path):
        """Update the video index with new results.
        
        Args:
            video_path (str): Path to the video file
            results (dict): Analytics results
            heatmap_path (str): Path to the generated heatmap
        """
        # Get video name from path
        video_name = os.path.basename(video_path)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Create filenames
        results_file = f"results_{os.path.splitext(video_name)[0]}_{timestamp}.json"
        heatmap_file = f"heatmap_{os.path.splitext(video_name)[0]}_{timestamp}.png"
        
        # Save results to file
        with open(os.path.join(self.results_dir, results_file), "w") as f:
            json.dump(results, f)
        
        # Copy heatmap to storage location
        shutil.copy(heatmap_path, os.path.join(self.heatmaps_dir, heatmap_file))
        
        # Update index
        index = self.initialize_video_index()
        index["videos"][video_name] = {
            "last_analyzed": datetime.now().isoformat(),
            "results_file": results_file,
            "heatmap_file": heatmap_file
        }
        index["latest_video"] = video_name
        
        # Save updated index
        with open(os.path.join(self.data_dir, "video_index.json"), "w") as f:
            json.dump(index, f, indent=2)
    
    def get_video_results(self, video_name=None):
        """Get results for a specific video.
        
        Args:
            video_name (str, optional): Name of the video. If None, uses latest.
            
        Returns:
            dict: Analytics results for the video or None if not found
        """
        index = self.initialize_video_index()
        
        # If no video specified, use latest
        if video_name is None:
            video_name = index.get("latest_video")
            if not video_name:
                return None  # No videos processed yet
        
        # Check if video exists in index
        if video_name not in index["videos"]:
            return None
        
        # Get results path
        results_file = index["videos"][video_name]["results_file"]
        
        # Load and return results
        with open(os.path.join(self.results_dir, results_file), "r") as f:
            return json.load(f)
    
    def get_video_heatmap_path(self, video_name=None):
        """Get heatmap path for a specific video.
        
        Args:
            video_name (str, optional): Name of the video. If None, uses latest.
            
        Returns:
            str: Path to the heatmap or None if not found
        """
        index = self.initialize_video_index()
        
        # If no video specified, use latest
        if video_name is None:
            video_name = index.get("latest_video")
            if not video_name:
                return None  # No videos processed yet
        
        # Check if video exists in index
        if video_name not in index["videos"]:
            return None
        
        # Return heatmap path
        heatmap_file = index["videos"][video_name]["heatmap_file"]
        return os.path.join(self.heatmaps_dir, heatmap_file)
    
    def list_analyzed_videos(self):
        """List all available analyzed videos.
        
        Returns:
            list: List of video information dictionaries
        """
        index = self.initialize_video_index()
        videos = []
        
        for video_name, info in index["videos"].items():
            videos.append({
                "name": video_name,
                "last_analyzed": info["last_analyzed"]
            })
        
        return videos
    
    def list_available_videos(self):
        """List all video files available in the data directory.
        
        Returns:
            list: List of video filenames
        """
        # Supported video extensions
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        
        # Find all video files
        video_files = []
        
        if os.path.exists(self.data_dir) and os.path.isdir(self.data_dir):
            for file in os.listdir(self.data_dir):
                # Check if the file has a video extension
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(file)
        
        return video_files
    
    def initialize_llm_analysis_index(self):
        """Initialize or load the LLM analysis index file.
        
        Returns:
            dict: The LLM analysis index data
        """
        index_path = os.path.join(self.data_dir, "llm_analysis_index.json")
        
        if not os.path.exists(index_path):
            # Create empty index
            with open(index_path, "w") as f:
                json.dump({"videos": {}, "latest_video": None}, f)
        
        # Load the index
        with open(index_path, "r") as f:
            return json.load(f)
            
    def update_llm_analysis_index(self, video_path, llm_analysis_path):
        """Update the LLM analysis index with new results.
        
        Args:
            video_path (str): Path to the video file
            llm_analysis_path (str): Path to the generated LLM analysis JSON file
        """
        # Get video name from path
        video_name = os.path.basename(video_path)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Create filename for the copied file
        llm_results_file = f"llm_results_{os.path.splitext(video_name)[0]}_{timestamp}.json"
        
        # Copy analysis to storage location
        llm_results_dest = os.path.join(self.llm_results_dir, llm_results_file)
        shutil.copy(llm_analysis_path, llm_results_dest)
        
        # Update LLM analysis index
        index = self.initialize_llm_analysis_index()
        index["videos"][video_name] = {
            "last_analyzed": datetime.now().isoformat(),
            "results_file": llm_results_file
        }
        index["latest_video"] = video_name
        
        # Save updated index
        with open(os.path.join(self.data_dir, "llm_analysis_index.json"), "w") as f:
            json.dump(index, f, indent=2)
            
    def get_video_llm_analysis(self, video_name=None):
        """Get LLM analysis results path for a specific video.
        
        Args:
            video_name (str, optional): Name of the video. If None, uses latest.
            
        Returns:
            str: Path to the LLM analysis results file or None if not found
        """
        index = self.initialize_llm_analysis_index()
        
        # If no video specified, use latest
        if video_name is None:
            video_name = index.get("latest_video")
            if not video_name:
                return None  # No videos processed yet
        
        # If video name already has an extension, remove it
        video_name = os.path.splitext(os.path.basename(video_name))[0]
        
        # Find the video in the index
        for full_video_name in index["videos"].keys():
            base_name = os.path.splitext(full_video_name)[0]
            if base_name == video_name:
                # Get results path
                results_file = index["videos"][full_video_name]["results_file"]
                return os.path.join(self.llm_results_dir, results_file)
        
        return None 