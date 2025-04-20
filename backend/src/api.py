from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter, Query
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
import os
from pathlib import Path
import shutil
import time
from typing import Optional, List
import cv2

from .detection import EmployeeDetector
from .tracking import EmployeeTracker
from .analytics import WarehouseAnalytics
from .gptanalyzer import MultimodalLLMAnalyzer
from .videomanager import VideoManager

# Define API models
class VideoAnalysisRequest(BaseModel):
    video_path: str

class VideoAnalysisResponse(BaseModel):
    employee_count: int
    distance_traveled: dict
    heatmap_image_path: str
    
class VideoInfo(BaseModel):
    name: str
    last_analyzed: str

class LLMAnalysisRequest(BaseModel):
    video_path: str
    frame_interval: Optional[int] = 30  # Analyze every 30 frames by default
    
# Initialize global state
detector = None
tracker = None
analytics = None
llm_analyzer = None
video_manager = None

# Track processing status
analysis_status = {
    "is_processing": False,
    "llm_processing": False,
    "last_video_path": None,
    "start_time": None,
    "end_time": None
}

def initialize_modules():
    """Initialize all required modules."""
    global detector, tracker, analytics, llm_analyzer, video_manager
    
    if detector is None:
        detector = EmployeeDetector()
    
    if tracker is None:
        tracker = EmployeeTracker()
    
    if analytics is None:
        analytics = WarehouseAnalytics()
    
    if llm_analyzer is None:
        llm_analyzer = MultimodalLLMAnalyzer()
    
    if video_manager is None:
        video_manager = VideoManager()

def process_video_task(video_path: str):
    """Background task to process video.
    
    Args:
        video_path (str): Path to the video file to process
    """
    global analysis_status
    
    try:
        # Update status
        analysis_status["is_processing"] = True
        analysis_status["last_video_path"] = video_path
        analysis_status["start_time"] = time.time()
        analysis_status["end_time"] = None
        
        # Initialize modules if not already initialized
        initialize_modules()
        
        # Step 1: Detect employees in the video
        detections = detector.process_video(video_path)
        
        # Step 2: Track employees
        tracking_results = tracker.process_video(video_path, detections)
        
        # Step 3: Analyze tracking results
        results = analytics.analyze_tracking(tracking_results, video_path)
        
        # Step 4: Update the video index with the results
        heatmap_path = results.get("heatmap_image_path", "")
        video_manager.update_video_index(video_path, results, heatmap_path)
        
        # Update status
        analysis_status["is_processing"] = False
        analysis_status["end_time"] = time.time()
        
    except Exception as e:
        # Handle errors
        analysis_status["is_processing"] = False
        analysis_status["end_time"] = time.time()
        print(f"Error processing video: {str(e)}")

def process_video_with_llm_task(video_path: str, frame_interval: int = 30):
    """Background task to analyze video with LLM.
    
    Args:
        video_path (str): Path to the video file to process
        frame_interval (int): Interval between frames to analyze
    """
    global analysis_status
    
    try:
        # Update status
        analysis_status["llm_processing"] = True
        analysis_status["last_video_path"] = video_path
        analysis_status["start_time"] = time.time()
        analysis_status["end_time"] = None
        
        # Initialize modules if not already initialized
        initialize_modules()
        
        # Use LLM to analyze key frames
        llm_results = llm_analyzer.extract_and_analyze_frames(video_path, frame_interval)
        
        # LLM results are now stored in the analyzer's latest_results
        # We could add them to our video_manager if needed for persistence
        
        # Update status
        analysis_status["llm_processing"] = False
        analysis_status["end_time"] = time.time()
        
    except Exception as e:
        # Handle errors
        analysis_status["llm_processing"] = False
        analysis_status["end_time"] = time.time()
        print(f"Error in LLM video analysis: {str(e)}")

# Create API router
api_router = APIRouter()

@api_router.post("/analyze-video", response_model=VideoAnalysisResponse)
async def analyze_video(request: VideoAnalysisRequest, background_tasks: BackgroundTasks):
    # Validate video path
    video_path = request.video_path
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
    
    # Check if analysis is already running
    if analysis_status["is_processing"]:
        raise HTTPException(
            status_code=409, 
            detail="Another video is currently being processed"
        )
    
    # Initialize modules if not already initialized
    initialize_modules()
    
    # Start processing in background
    background_tasks.add_task(process_video_task, video_path)
    
    return JSONResponse(
        status_code=202,
        content={
            "message": f"Processing video: {video_path}",
            "status": "processing"
        }
    )

@api_router.post("/analyze-video-llm")
async def analyze_video_llm(request: LLMAnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze video using multimodal LLM."""
    # Validate video path
    video_path = request.video_path
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
    
    # Check if analysis is already running
    if analysis_status["llm_processing"]:
        raise HTTPException(
            status_code=409, 
            detail="Another video is currently being processed with LLM"
        )
    
    # Initialize modules if not already initialized
    initialize_modules()
    
    # Make sure the API token is set
    if not llm_analyzer.api_token:
        raise HTTPException(
            status_code=400,
            detail="Hugging Face API token not configured. Set the HF_API_TOKEN environment variable."
        )
    
    # Start processing in background
    background_tasks.add_task(
        process_video_with_llm_task, 
        video_path, 
        request.frame_interval
    )
    
    return JSONResponse(
        status_code=202,
        content={
            "message": f"Processing video with LLM: {video_path}",
            "status": "processing",
            "frame_interval": request.frame_interval
        }
    )

@api_router.get("/llm-analysis")
async def get_llm_analysis(video_name: Optional[str] = None):
    """Get LLM-based analysis results."""
    # Initialize modules if not already initialized
    initialize_modules()
    
    # For now, we only have the latest results since we're not storing
    # LLM results in the VideoManager yet
    results = llm_analyzer.get_latest_results()
    if results is None:
        raise HTTPException(
            status_code=404,
            detail="No LLM analysis available. Process a video first."
        )
    
    return results

@api_router.get("/llm-analyzed-videos")
async def list_llm_analyzed_videos():
    """Get a list of all videos analyzed using LLM."""
    # Initialize modules if not already initialized
    initialize_modules()
    
    # This is a placeholder until we implement storage of LLM results
    # in the VideoManager
    if llm_analyzer.latest_results:
        video_path = llm_analyzer.latest_results.get("video_path", "")
        if video_path:
            return [{
                "name": os.path.basename(video_path),
                "last_analyzed": llm_analyzer.latest_results.get("timestamp", "")
            }]
    return []

@api_router.get("/analytics", response_model=VideoAnalysisResponse)
async def get_analytics(video_name: Optional[str] = None):
    # Initialize modules if not already initialized
    initialize_modules()
    
    # Get results for the specified video or latest
    if video_name:
        results = video_manager.get_video_results(video_name)
        if results is None:
            raise HTTPException(
                status_code=404,
                detail=f"No analytics results available for video: {video_name}"
            )
    else:
        # Fall back to latest results for backward compatibility
        results = analytics.get_latest_results()
        if results is None:
            raise HTTPException(
                status_code=404,
                detail="No analytics results available. Process a video first."
            )
    
    return results

@api_router.get("/heatmap")
async def get_heatmap(video_name: Optional[str] = None):
    # Initialize modules if not already initialized
    initialize_modules()
    
    # Get heatmap for the specified video or latest
    if video_name:
        heatmap_path = video_manager.get_video_heatmap_path(video_name)
        if heatmap_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"No heatmap available for video: {video_name}"
            )
    else:
        # Fall back to latest results for backward compatibility
        results = analytics.get_latest_results()
        if results is None:
            raise HTTPException(
                status_code=404,
                detail="No heatmap available. Process a video first."
            )
        heatmap_path = results["heatmap_image_path"]
    
    if not os.path.exists(heatmap_path):
        raise HTTPException(
            status_code=404,
            detail=f"Heatmap file not found: {heatmap_path}"
        )
    
    return FileResponse(
        path=heatmap_path,
        media_type="image/png",
        filename=os.path.basename(heatmap_path)
    )

@api_router.get("/stream/{video_name}")
async def stream_video(video_name: str):
    """Stream a video file by name."""
    # Construct path to the video file
    video_path = os.path.join("data", video_name)
    
    # Check if the video file exists
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_name}")
    
    # Define a generator function to stream the file
    def iterfile():
        with open(video_path, "rb") as file:
            # Read and yield chunks of the file
            while chunk := file.read(1024 * 1024):  # 1MB chunks
                yield chunk
    
    # Determine the media type based on file extension
    media_type = None
    if video_name.endswith(".mp4"):
        media_type = "video/mp4"
    elif video_name.endswith(".avi"):
        media_type = "video/x-msvideo"
    elif video_name.endswith(".mov"):
        media_type = "video/quicktime"
    else:
        media_type = "application/octet-stream"  # Generic binary
    
    # Return a streaming response
    return StreamingResponse(
        iterfile(),
        media_type=media_type,
        headers={"Content-Disposition": f"inline; filename={video_name}"}
    )

@api_router.get("/available-videos", response_model=List[str])
async def list_available_videos():
    """List all video files available in the data directory."""
    # Initialize modules if not already initialized
    initialize_modules()
    
    # Get list of available videos from VideoManager
    return video_manager.list_available_videos()

@api_router.get("/videos", response_model=List[VideoInfo])
async def list_videos():
    """Get a list of all analyzed videos."""
    # Initialize modules if not already initialized
    initialize_modules()
    
    # Get list of analyzed videos from VideoManager
    return video_manager.list_analyzed_videos()

@api_router.get("/status")
async def get_status():
    return {
        "is_processing": analysis_status["is_processing"],
        "llm_processing": analysis_status.get("llm_processing", False),
        "last_video_path": analysis_status["last_video_path"],
        "start_time": analysis_status["start_time"],
        "end_time": analysis_status["end_time"],
        "elapsed_time": (
            time.time() - analysis_status["start_time"] 
            if (analysis_status["is_processing"] or analysis_status.get("llm_processing", False)) and analysis_status["start_time"] 
            else None
        )
    }

# Export the router
api = api_router 