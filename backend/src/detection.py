import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
from pathlib import Path
import traceback  # Added for better error reporting

class EmployeeDetector:
    def __init__(self, model_path=None):
        """Initialize the YOLO-based employee detector.
        
        Args:
            model_path (str, optional): Path to custom YOLO model. 
                If None, will use pretrained YOLOv8n model.
        """
        # If no model provided, use YOLOv8n
        if model_path is None:
            self.model = YOLO("yolov8n.pt")
        else:
            self.model = YOLO(model_path)
            
        # We'll focus on detecting people (class 0 in COCO dataset)
        self.target_class = 0  # 'person' in COCO
    
    def detect_frame(self, frame):
        """Detect employees in a single frame.
        
        Args:
            frame (numpy.ndarray): Input frame (BGR format)
            
        Returns:
            detections (numpy.ndarray): Array of detection results with format
                [x1, y1, x2, y2, confidence, class_id]
        """
        try:
            results = self.model(frame, verbose=False)[0]
            
            # Filter detections to keep only people
            detections = []
            
            for result in results.boxes.data.cpu().numpy():
                x1, y1, x2, y2, confidence, class_id = result
                
                # Only keep person detections (class 0)
                if int(class_id) == self.target_class and confidence > 0.5:
                    detections.append([x1, y1, x2, y2, confidence, class_id])
            
            # Return empty array with correct shape if no detections
            if len(detections) == 0:
                return np.zeros((0, 6))
                    
            return np.array(detections)
        except Exception as e:
            print(f"Error in detect_frame: {str(e)}")
            print(traceback.format_exc())
            # Return empty array in case of error
            return np.zeros((0, 6))
    
    def process_video(self, video_path, output_dir=None, save_output=False):
        """Process a video file and detect employees in all frames.
        
        Args:
            video_path (str): Path to video file
            output_dir (str, optional): Directory to save output video
            save_output (bool): Whether to save the output video with detections
        
        Returns:
            all_detections (dict): Dictionary mapping frame indices to detections
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            # Check if video opened successfully
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
                
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Processing video: {video_path}")
            print(f"Dimensions: {frame_width}x{frame_height}, FPS: {fps}, Total frames: {total_frames}")
            
            # Set up output video writer if needed
            out = None
            if save_output and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir, 
                    f"detection_{Path(video_path).stem}.mp4"
                )
                out = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (frame_width, frame_height)
                )
            
            # Storage for all detections
            all_detections = {}
            
            # Process each frame
            frame_idx = 0
            while cap.isOpened():
                success, frame = cap.read()
                
                if not success:
                    break
                    
                # Run detection on frame
                detections = self.detect_frame(frame)
                
                # Ensure detections is a proper numpy array with the right shape
                if not isinstance(detections, np.ndarray) or detections.ndim != 2:
                    print(f"Warning: Invalid detections format at frame {frame_idx}. Converting to empty array.")
                    detections = np.zeros((0, 6))
                
                all_detections[frame_idx] = detections
                
                # Print progress every 100 frames
                if frame_idx % 100 == 0:
                    print(f"Processed {frame_idx}/{total_frames} frames")
                
                # If saving output, draw boxes and write frame
                if save_output and out:
                    # Create box annotator
                    box_annotator = sv.BoxAnnotator(
                        thickness=2,
                        text_thickness=2,
                        text_scale=1
                    )
                    
                    # Prepare detections for supervision format
                    if len(detections) > 0:
                        detections_sv = sv.Detections(
                            xyxy=detections[:, :4],
                            confidence=detections[:, 4],
                            class_id=detections[:, 5].astype(int),
                        )
                        
                        # Create labels
                        labels = [
                            f"Employee: {confidence:.2f}"
                            for confidence in detections_sv.confidence
                        ]
                        
                        # Draw bounding boxes
                        frame = box_annotator.annotate(
                            scene=frame, 
                            detections=detections_sv,
                            labels=labels
                        )
                    
                    out.write(frame)
                
                frame_idx += 1
            
            # Release resources
            cap.release()
            if out:
                out.release()
                
            print(f"Video processing complete. Processed {frame_idx} frames.")
            return all_detections
            
        except Exception as e:
            print(f"Error in process_video: {str(e)}")
            print(traceback.format_exc())
            # Return empty dict in case of error
            return {} 