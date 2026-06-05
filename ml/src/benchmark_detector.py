import time
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.src.vehicle_detector import VehicleDetector

def benchmark():
    video_path = r"C:\Users\User\OneDrive\Desktop\traffic-django\media\videos\vecteezy_car-and-truck-traffic-on-the-highway-in-europe-poland_7957364.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    detector = VehicleDetector()
    print("Inference started...")
    
    start_time = time.time()
    count = 0
    
    # We use stream_inference to test the MJPEG path
    for frame_data in detector.stream_inference(video_path):
        count += 1
        if count >= 100:  # Benchmark 100 frames
            break
            
    end_time = time.time()
    total_time = end_time - start_time
    avg_fps = count / total_time
    
    print(f"Processed {count} frames in {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    benchmark()
