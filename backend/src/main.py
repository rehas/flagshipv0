import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from .api import api, initialize_modules

app = FastAPI(
    title="Warehouse CCTV Analyzer",
    description="REST API for analyzing warehouse CCTV footage",
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

# Create data directories for the video index
os.makedirs("data/results", exist_ok=True)
os.makedirs("data/heatmaps", exist_ok=True)

# Mount static files for serving heatmaps and other static content
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/data", StaticFiles(directory="data"), name="data")

# Include API router
app.include_router(api)

# Add startup event to initialize modules
@app.on_event("startup")
async def startup_event():
    print("Initializing modules and video index...")
    initialize_modules()
    print("Initialization complete!")

# Root endpoint with basic info
@app.get("/")
async def root():
    return {
        "application": "Warehouse CCTV Analyzer",
        "version": "0.1.0",
        "endpoints": [
            {"path": "/analyze-video", "method": "POST", "description": "Analyze a warehouse CCTV video"},
            {"path": "/analytics", "method": "GET", "description": "Get latest analytics results"},
            {"path": "/heatmap", "method": "GET", "description": "Get heatmap visualization of employee movement"},
            {"path": "/status", "method": "GET", "description": "Get current processing status"},
            {"path": "/videos", "method": "GET", "description": "Get list of analyzed videos"}
        ],
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True) 