import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
from pathlib import Path
import traceback  # Added for better error reporting

class EmployeeTracker:
    def __init__(self, max_age=30, n_init=3):
        """Initialize DeepSORT tracker for employee tracking.
        
        Args:
            max_age (int): Maximum number of frames to keep track of 
                           disappeared objects
            n_init (int): Number of frames needed to confirm a track
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            embedder="mobilenet",
            embedder_gpu=True,
            embedder_model_name="osnet_ain_x1_0"
        )
    
    def update(self, frame, detections):
        """Update tracker with new detections.
        
        Args:
            frame (numpy.ndarray): Current frame
            detections (numpy.ndarray): Array of detections with format
                [x1, y1, x2, y2, confidence, class_id]
                
        Returns:
            tracks (list): List of tracks with format
                [track_id, class_id, x1, y1, x2, y2]
        """
        try:
            # Check if detections is a proper numpy array with the right shape
            if not isinstance(detections, np.ndarray):
                print(f"Warning: Detections is not a numpy array. Type: {type(detections)}")
                return []
            
            # Check if detections is a scalar (float32)
            if np.isscalar(detections):
                print(f"Warning: Detections is a scalar value: {detections}")
                return []
            
            # Check if detections is empty (numpy array with shape (0, 6))
            if detections.shape[0] == 0:
                return []
            
            # Format detections according to DeepSORT's expected structure:
            # List[ Tuple[ List[float or int], float, str ] ]
            formatted_detections = []
            
            for detection in detections:
                x1, y1, x2, y2, confidence, class_id = detection
                
                # Convert to tlwh format
                tlwh = [
                    float(x1),           # Top-left x
                    float(y1),           # Top-left y
                    float(x2 - x1),      # Width
                    float(y2 - y1)       # Height
                ]
                
                # Create tuple in the format (bbox, confidence, class)
                # Class is hardcoded as 'person' since we're tracking employees
                det_tuple = (tlwh, float(confidence), 'person')
                formatted_detections.append(det_tuple)
            
            
            # Call update_tracks with the properly formatted detections
            tracks = self.tracker.update_tracks(formatted_detections, frame=frame)
            
            # Convert tracks to our format [track_id, class_id, x1, y1, x2, y2]
            results = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = ltrb
                
                # Assuming all detections are employees (class_id = 0)
                results.append([track_id, 0, x1, y1, x2, y2])
            
            return results
        
        except Exception as e:
            print(f"Error in update: {str(e)}")
            print(traceback.format_exc())
            return []
    
    def process_video(self, video_path, detections_dict, output_dir=None, save_output=False):
        """Process a video file with DeepSORT tracking.
        
        Args:
            video_path (str): Path to video file
            detections_dict (dict): Dictionary mapping frame indices to detections
            output_dir (str, optional): Directory to save output video
            save_output (bool): Whether to save the output video with tracks
            
        Returns:
            tracking_results (dict): Dictionary with tracking information
                - "tracks": {track_id: [positions]}
                - "per_frame": {frame_idx: [tracks]}
        """
        try:
            # Print detections_dict info for debugging
            num_frames = len(detections_dict)
            print(f"Number of frames with detections: {num_frames}")
            
            if num_frames == 0:
                print("Warning: No frames with detections found")
                return {"tracks": {}, "per_frame": {}}
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            # Check if video opened successfully
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
                
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Tracking video: {video_path}")
            print(f"Dimensions: {frame_width}x{frame_height}, FPS: {fps}, Total frames: {total_frames}")
            
            # Set up output video writer if needed
            out = None
            if save_output and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir, 
                    f"tracking_{Path(video_path).stem}.mp4"
                )
                out = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (frame_width, frame_height)
                )
                
            # Storage for tracking results
            tracking_results = {
                "tracks": {},  # track_id -> list of positions
                "per_frame": {}  # frame_idx -> list of tracks
            }
            
            # Process each frame
            frame_idx = 0
            while cap.isOpened():
                success, frame = cap.read()
                
                if not success:
                    break
                    
                # Get detections for this frame
                if frame_idx in detections_dict:
                    # Get detections and ensure it's properly formatted
                    detections = detections_dict[frame_idx]
                    
                    # Check for scalar values and convert to empty arrays
                    if np.isscalar(detections):
                        print(f"Warning: Scalar detection at frame {frame_idx}. Converting to empty array.")
                        detections = np.zeros((0, 6))
                    
                    # Ensure it's a numpy array
                    if not isinstance(detections, np.ndarray):
                        print(f"Warning: Non-array detection at frame {frame_idx}. Converting to empty array.")
                        detections = np.zeros((0, 6))
                    
                    # Ensure it has the right dimensionality
                    if detections.ndim != 2:
                        print(f"Warning: Wrong dimensions for detection at frame {frame_idx}. Converting to empty array.")
                        detections = np.zeros((0, 6))
                    
                    # Update tracker
                    tracks = self.update(frame, detections)
                    
                    # Store tracking results
                    tracking_results["per_frame"][frame_idx] = tracks
                    
                    # Update track positions
                    for track in tracks:
                        track_id, class_id, x1, y1, x2, y2 = track
                        
                        # Calculate center point of bounding box
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        
                        # Store position
                        if track_id not in tracking_results["tracks"]:
                            tracking_results["tracks"][track_id] = []
                            
                        tracking_results["tracks"][track_id].append(
                            {"frame": frame_idx, "x": cx, "y": cy, "bbox": [x1, y1, x2, y2]}
                        )
                    
                    # If saving output, draw tracks and write frame
                    if save_output and out:
                        # Draw tracks
                        for track in tracks:
                            track_id, class_id, x1, y1, x2, y2 = track
                            
                            # Generate a color based on track_id
                            color = (
                                hash(track_id) % 256,
                                hash(track_id * 2) % 256,
                                hash(track_id * 3) % 256
                            )
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            
                            # Draw ID
                            cv2.putText(
                                frame,
                                f"ID: {track_id}",
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2
                            )
                            
                        out.write(frame)
                
                # Print progress every 100 frames
                if frame_idx % 100 == 0:
                    print(f"Processed {frame_idx}/{total_frames} frames for tracking")
                
                frame_idx += 1
                
            # Release resources
            cap.release()
            if out:
                out.release()
                
            print(f"Tracking complete. Found {len(tracking_results['tracks'])} unique tracks.")
            return tracking_results
            
        except Exception as e:
            print(f"Error in process_video: {str(e)}")
            print(traceback.format_exc())
            return {"tracks": {}, "per_frame": {}} 