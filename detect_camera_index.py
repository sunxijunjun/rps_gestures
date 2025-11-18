
"""
detect camera index
"""

import cv2

def list_cameras(max_index=5):
    print("Scanning cameras...")
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera index {i} is AVAILABLE")
            cap.release()
        else:
            print(f"Camera index {i} not available")

if __name__ == "__main__":
    list_cameras(5)
