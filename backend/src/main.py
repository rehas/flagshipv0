import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv

from .api import api, initialize_modules

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="FlagShip AI",
    description="REST API for analyzing CCTV footage",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for the POC
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create outputs directory if it doesn't exist
os.makedirs("outputs/heatmaps", exist_ok=True)
os.makedirs("outputs/analytics", exist_ok=True)
os.makedirs("outputs/llm_analysis", exist_ok=True)
os.makedirs("outputs/frame_captures", exist_ok=True)

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Mount static files for serving heatmaps and other static content
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/data", StaticFiles(directory="data"), name="data")

# Include API router
app.include_router(api)

# Add startup event to initialize modules
@app.on_event("startup")
async def startup_event():
    print("Initializing modules and video index...")
    # Check for HF API token
    hf_token = os.environ.get("HF_API_TOKEN")
    if hf_token:
        print("Hugging Face API token found in environment variables.")
    else:
        print("WARNING: Hugging Face API token not found. LLM analysis will not work.")
        print("Create a .env file with your HF_API_TOKEN or set it in your environment.")
        
    initialize_modules()
    print("Initialization complete!")

# Root endpoint with basic info
@app.get("/")
async def root():
    return {
        "application": "FlagShip AI",
        "version": "0.1.0",
        "endpoints": [
            {"path": "/analyze-video", "method": "POST", "description": "Analyze a warehouse CCTV video (tracks employees, calculates distance and idle time)"},
            {"path": "/analytics", "method": "GET", "description": "Get latest analytics results including employee counts, distances, and idle times"},
            {"path": "/heatmap", "method": "GET", "description": "Get heatmap visualization of employee movement"},
            {"path": "/status", "method": "GET", "description": "Get current processing status"},
            {"path": "/videos", "method": "GET", "description": "Get list of analyzed videos"},
            {"path": "/analyze-video-llm", "method": "POST", "description": "Analyze video with multimodal LLM"},
            {"path": "/llm-analysis", "method": "GET", "description": "Get LLM analysis results"},
            {"path": "/llm-analyzed-videos", "method": "GET", "description": "Get list of videos analyzed with LLM"},
            {"path": "/available-videos", "method": "GET", "description": "Get list of all video files available"},
            {"path": "/stream/{video_name}", "method": "GET", "description": "Stream a video file"}
        ],
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True) 