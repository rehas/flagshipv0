# CCTV Warehouse Analyzer (REST API)

This is a backend REST API for analyzing warehouse CCTV footage. It provides endpoints to detect employees, track their movements, and generate analytics such as employee count, distance traveled, and heatmaps.

## Features

- **Employee Detection**: Uses YOLOv8 for accurate employee detection in video footage
- **Employee Tracking**: Implements DeepSORT for persistent tracking of employees across frames
- **Analytics Generation**: Calculates metrics like total employee count and distance traveled
- **Heatmap Visualization**: Creates heatmaps of employee movement patterns
- **REST API**: Provides convenient endpoints for video analysis and results retrieval

## Project Structure

```
.
├── data
│   └── warehouse-footage.mp4
├── models
│   └── [auto-downloaded YOLO/DeepSORT models]
├── outputs
│   ├── heatmaps
│   └── analytics
├── src
│   ├── detection.py      # YOLO (object detection)
│   ├── tracking.py       # DeepSORT (object tracking)
│   ├── analytics.py      # Distance calculation, heatmaps
│   ├── api.py            # REST API endpoints
│   └── main.py           # FastAPI entry point
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository
2. Create a virtual environment (recommended)
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

## Usage

### Starting the Server

Run the following command to start the FastAPI server:

```
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### API Documentation

Once the server is running, you can access the automatically generated Swagger documentation at:

```
http://localhost:8000/docs
```

### API Endpoints

- **POST /analyze-video**
  - Triggers analysis on selected video
  - JSON input: `{ "video_path": "data/warehouse-footage.mp4" }`
  - Returns a 202 Accepted response with processing status

- **GET /analytics**
  - Returns stored analytics of the latest processed video
  - JSON response:
    ```json
    {
      "employee_count": 5,
      "distance_traveled": {
        "employee_1": 45.7,
        "employee_2": 23.5
      },
      "heatmap_image_path": "outputs/heatmaps/heatmap.png"
    }
    ```

- **GET /heatmap**
  - Returns the generated heatmap image (as downloadable image file)

- **GET /status**
  - Returns the current processing status

## Example Usage

### Python Client

```python
import requests
import json

# Start video analysis
response = requests.post(
    "http://localhost:8000/analyze-video",
    json={"video_path": "data/warehouse-footage.mp4"}
)
print(response.json())

# Check status
status = requests.get("http://localhost:8000/status").json()
print(f"Processing: {status['is_processing']}")

# Get analytics (after processing is complete)
analytics = requests.get("http://localhost:8000/analytics").json()
print(f"Employee count: {analytics['employee_count']}")
print(f"Distance traveled: {analytics['distance_traveled']}")

# Download heatmap
with open("heatmap.png", "wb") as f:
    f.write(requests.get("http://localhost:8000/heatmap").content)
```

### cURL

```bash
# Start video analysis
curl -X POST http://localhost:8000/analyze-video \
  -H "Content-Type: application/json" \
  -d '{"video_path": "data/warehouse-footage.mp4"}'

# Get analytics
curl -X GET http://localhost:8000/analytics

# Download heatmap
curl -X GET http://localhost:8000/heatmap --output heatmap.png
```

## Technical Details

- **Detection**: YOLOv8 model focused on person detection
- **Tracking**: DeepSORT with ReID model for persistent tracking
- **Analytics**: Custom distance calculation and heatmap generation
- **API**: FastAPI for high-performance REST endpoints

## License

This project is for demonstration purposes only. 