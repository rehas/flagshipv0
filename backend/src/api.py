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
    
# Initialize global state
detector = None
tracker = None
analytics = None

# Track processing status
analysis_status = {
    "is_processing": False,
    "last_video_path": None,
    "start_time": None,
    "end_time": None
}

def initialize_modules():
    """Initialize all required modules."""
    global detector, tracker, analytics
    
    if detector is None:
        detector = EmployeeDetector()
    
    if tracker is None:
        tracker = EmployeeTracker()
    
    if analytics is None:
        analytics = WarehouseAnalytics()
        
    # Initialize the video index
    if analytics:
        analytics.initialize_video_index()

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
        analytics.update_video_index(video_path, results, heatmap_path)
        
        # Update status
        analysis_status["is_processing"] = False
        analysis_status["end_time"] = time.time()
        
    except Exception as e:
        # Handle errors
        analysis_status["is_processing"] = False
        analysis_status["end_time"] = time.time()
        print(f"Error processing video: {str(e)}")

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

@api_router.get("/analytics", response_model=VideoAnalysisResponse)
async def get_analytics(video_name: Optional[str] = None):
    # Initialize modules if not already initialized
    initialize_modules()
    
    # Get results for the specified video or latest
    if video_name:
        results = analytics.get_video_results(video_name)
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
        heatmap_path = analytics.get_video_heatmap_path(video_name)
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

@api_router.get("/videos", response_model=List[VideoInfo])
async def list_videos():
    """Get a list of all analyzed videos."""
    # Initialize modules if not already initialized
    initialize_modules()
    
    videos = analytics.list_available_videos()
    
    return videos

@api_router.get("/status")
async def get_status():
    return {
        "is_processing": analysis_status["is_processing"],
        "last_video_path": analysis_status["last_video_path"],
        "start_time": analysis_status["start_time"],
        "end_time": analysis_status["end_time"],
        "elapsed_time": (
            time.time() - analysis_status["start_time"] 
            if analysis_status["is_processing"] and analysis_status["start_time"] 
            else None
        )
    }

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
    # Path to the data directory
    data_dir = "data"
    
    # Supported video extensions
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    
    # Find all video files
    video_files = []
    
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        for file in os.listdir(data_dir):
            # Check if the file has a video extension
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(file)
    
    return video_files

# Export the router
api = api_router 