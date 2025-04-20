import numpy as np
# Set matplotlib backend to 'Agg' to avoid issues with GUI in background threads
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
import math
import json
from datetime import datetime
import traceback  # Added for better error reporting
import shutil

class WarehouseAnalytics:
    def __init__(self, output_dir="outputs"):
        """Initialize the analytics module.
        
        Args:
            output_dir (str): Base directory for saving outputs
        """
        self.output_dir = output_dir
        
        # Create output directories if they don't exist
        self.heatmap_dir = os.path.join(output_dir, "heatmaps")
        self.analytics_dir = os.path.join(output_dir, "analytics")
        
        os.makedirs(self.heatmap_dir, exist_ok=True)
        os.makedirs(self.analytics_dir, exist_ok=True)
        
        # Store the latest analytics results
        self.latest_results = None
        
    def calculate_distance(self, track_positions):
        """Calculate the total distance traveled by an employee.
        
        Args:
            track_positions (list): List of position dictionaries with x, y coordinates
            
        Returns:
            float: Total distance traveled in pixels
        """
        total_distance = 0.0
        
        # Need at least 2 points to calculate distance
        if len(track_positions) < 2:
            return total_distance
            
        # Sum distances between consecutive points
        for i in range(1, len(track_positions)):
            prev_pos = track_positions[i-1]
            curr_pos = track_positions[i]
            
            # Calculate Euclidean distance
            dx = curr_pos["x"] - prev_pos["x"]
            dy = curr_pos["y"] - prev_pos["y"]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Only add if distance is reasonable (to filter out tracking errors)
            if distance < 100:  # Threshold to ignore big jumps
                total_distance += distance
                
        return total_distance
    
    def generate_heatmap(self, tracking_results, video_path, resolution=(500, 500)):
        """Generate a heatmap of employee movements.
        
        Args:
            tracking_results (dict): Dictionary with tracking information
            video_path (str): Path to the original video
            resolution (tuple): Resolution of the heatmap
            
        Returns:
            str: Path to the generated heatmap image
        """
        try:
            # Check if tracking results contains any tracks
            if not tracking_results or 'tracks' not in tracking_results or not tracking_results['tracks']:
                print("Warning: No tracks found for heatmap generation. Creating an empty heatmap.")
                plt.figure(figsize=(10, 8))
                plt.text(0.5, 0.5, 'No employee tracks detected', 
                        horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                heatmap_filename = f"empty_heatmap_{Path(video_path).stem}_{timestamp}.png"
                heatmap_path = os.path.join(self.heatmap_dir, heatmap_filename)
                plt.savefig(heatmap_path)
                plt.close()
                
                return heatmap_path
                
            # Get video dimensions
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
                
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Create accumulation map
            heatmap = np.zeros(resolution, dtype=np.float32)
            
            # Scale factors to convert video coordinates to heatmap coordinates
            scale_x = resolution[0] / video_width
            scale_y = resolution[1] / video_height
            
            # Print debug info
            print(f"Generating heatmap for {len(tracking_results['tracks'])} tracks")
            print(f"Video dimensions: {video_width}x{video_height}, Heatmap resolution: {resolution}")
            
            # Add all track points to heatmap
            track_points_total = 0
            for track_id, positions in tracking_results['tracks'].items():
                track_points = len(positions)
                track_points_total += track_points
                print(f"Track {track_id}: {track_points} positions")
                
                for pos in positions:
                    x, y = float(pos.get("x", 0)), float(pos.get("y", 0))
                    
                    # Scale coordinates to heatmap resolution
                    heatmap_x = int(x * scale_x)
                    heatmap_y = int(y * scale_y)
                    
                    # Ensure within bounds
                    if 0 <= heatmap_x < resolution[0] and 0 <= heatmap_y < resolution[1]:
                        # Add Gaussian distribution around the point
                        sigma = 5  # Standard deviation of the Gaussian
                        for i in range(max(0, heatmap_x - 15), min(resolution[0], heatmap_x + 15)):
                            for j in range(max(0, heatmap_y - 15), min(resolution[1], heatmap_y + 15)):
                                dx = i - heatmap_x
                                dy = j - heatmap_y
                                distance = math.sqrt(dx*dx + dy*dy)
                                
                                # Gaussian distribution
                                intensity = math.exp(-(distance*distance) / (2*sigma*sigma))
                                heatmap[j, i] += intensity
            
            print(f"Added {track_points_total} points to heatmap")
            
            # Normalize heatmap
            heatmap_max = np.max(heatmap)
            print(f"Heatmap max value: {heatmap_max}")
            
            if heatmap_max > 0:
                heatmap = heatmap / heatmap_max
            
            # Create visualization
            plt.figure(figsize=(10, 8))
            plt.imshow(heatmap, cmap='jet')
            plt.colorbar(label='Density')
            plt.title('Employee Movement Heatmap')
            
            # Save heatmap
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            heatmap_filename = f"heatmap_{Path(video_path).stem}_{timestamp}.png"
            heatmap_path = os.path.join(self.heatmap_dir, heatmap_filename)
            plt.savefig(heatmap_path)
            plt.close()
            
            print(f"Heatmap saved to: {heatmap_path}")
            return heatmap_path
            
        except Exception as e:
            print(f"Error in generate_heatmap: {str(e)}")
            print(traceback.format_exc())
            
            # Create a dummy heatmap in case of error
            plt.figure(figsize=(10, 8))
            plt.text(0.5, 0.5, 'Error generating heatmap', 
                     horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            heatmap_filename = f"error_heatmap_{Path(video_path).stem}_{timestamp}.png"
            heatmap_path = os.path.join(self.heatmap_dir, heatmap_filename)
            plt.savefig(heatmap_path)
            plt.close()
            
            return heatmap_path
    
    def analyze_tracking(self, tracking_results, video_path):
        """Analyze tracking results to compute employee metrics.
        
        Args:
            tracking_results (dict): Dictionary with tracking information
            video_path (str): Path to the original video
            
        Returns:
            dict: Dictionary with analytics results
        """
        try:
            # Ensure tracking_results contains the expected structure
            if not tracking_results or 'tracks' not in tracking_results:
                print("Warning: Tracking results missing or invalid. Creating empty results.")
                tracking_results = {"tracks": {}}
                
            # Print incoming tracking data for debugging
            track_count = len(tracking_results.get('tracks', {}))
            print(f"Analyzing tracking results with {track_count} tracks")
            
            # Count unique employees (track IDs)
            employee_count = track_count
            
            # Calculate distance traveled per employee
            distance_traveled = {}
            for track_id, positions in tracking_results.get("tracks", {}).items():
                distance = self.calculate_distance(positions)
                distance_traveled[f"employee_{track_id}"] = round(distance, 2)
                print(f"Employee {track_id}: distance traveled = {distance:.2f} pixels")
            
            # Generate heatmap
            heatmap_path = self.generate_heatmap(tracking_results, video_path)
            
            # Prepare results
            results = {
                "video_path": video_path,
                "employee_count": employee_count,
                "distance_traveled": distance_traveled,
                "heatmap_image_path": heatmap_path,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save results to JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analytics_filename = f"analytics_{Path(video_path).stem}_{timestamp}.json"
            analytics_path = os.path.join(self.analytics_dir, analytics_filename)
            
            with open(analytics_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            # Store the latest results
            self.latest_results = results
            
            print(f"Analytics saved to: {analytics_path}")
            return results
            
        except Exception as e:
            print(f"Error in analyze_tracking: {str(e)}")
            print(traceback.format_exc())
            
            # Create a basic result in case of error
            heatmap_path = None
            try:
                # Try to generate an error heatmap
                plt.figure(figsize=(10, 8))
                plt.text(0.5, 0.5, 'Error analyzing tracking data', 
                         horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                heatmap_filename = f"error_heatmap_{Path(video_path).stem}_{timestamp}.png"
                heatmap_path = os.path.join(self.heatmap_dir, heatmap_filename)
                plt.savefig(heatmap_path)
                plt.close()
            except:
                pass
            
            results = {
                "video_path": video_path,
                "employee_count": 0,
                "distance_traveled": {},
                "heatmap_image_path": heatmap_path or "",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
            
            # Store the error results
            self.latest_results = results
            return results
    
    def get_latest_results(self):
        """Get the latest analytics results.
        
        Returns:
            dict: Latest analytics results or None if not available
        """
        return self.latest_results 