import cv2
import time
import os

video_path = r"C:\Users\User\OneDrive\Desktop\traffic-django\media\videos\vecteezy_car-and-truck-traffic-on-the-highway-in-europe-poland_7957364.mp4"

def test_cv2_speed():
    if not os.path.exists(video_path):
        print("Video not found")
        return

    cap = cv2.VideoCapture(video_path)
    start_time = time.time()
    count = 0
    while count < 100:
        ok, frame = cap.read()
        if not ok:
            break
        count += 1
    
    end_time = time.time()
    print(f"Read {count} frames in {end_time - start_time:.2f}s")
    print(f"FPS: {count / (end_time - start_time):.2f}")
    cap.release()

if __name__ == "__main__":
    test_cv2_speed()
