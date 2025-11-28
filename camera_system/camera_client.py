import requests
import cv2
import time
import os
from io import BytesIO
from PIL import Image

# Configuration
SERVER_URL = "http://127.0.0.1:5000"  # REPLACE WITH YOUR WINDOWS PC IP
PROCESS_FACE_ENDPOINT = f"{SERVER_URL}/process_face"

def get_face_image_bytes():
    """
    Captures an image from the camera.
    Uses OpenCV for compatibility (works on Pi with legacy camera enabled or USB cam).
    For Picamera2 specific code, refer to face_capture.py.
    """
    # Set Resolution to HD (1280x720)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Could not open camera.")
        return None

    print("📷 Camera opened (HD). Press 'c' to capture (Smart Burst), 'q' to quit.")
    
    captured_bytes = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break
            
        cv2.imshow("Face Client", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("\n📸 Starting Smart Burst Capture (5 frames)...")
            best_frame = None
            best_score = -1.0
            
            # Capture 5 frames in quick succession
            for i in range(5):
                ret, burst_frame = cap.read()
                if not ret:
                    continue
                
                # Calculate sharpness using Laplacian Variance
                gray = cv2.cvtColor(burst_frame, cv2.COLOR_BGR2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                print(f"   - Frame {i+1}: Sharpness Score = {score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_frame = burst_frame
                
                # Small delay to allow focus/exposure to adjust if needed, 
                # but fast enough to be a "burst"
                time.sleep(0.05) 
            
            if best_frame is not None:
                print(f"✨ Selected best frame (Score: {best_score:.2f})")
                # Convert to JPEG bytes
                is_success, buffer = cv2.imencode(".jpg", best_frame)
                if is_success:
                    captured_bytes = buffer.tobytes()
                    print("✅ Image prepared.")
                else:
                    print("❌ Failed to encode image.")
            else:
                print("❌ Failed to capture any valid frames.")
                
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return captured_bytes

def main():
    print("=== AI Smriti: Face Recognition Client ===")
    print(f"Server URL: {SERVER_URL}")
    
    while True:
        cmd = input("\nPress Enter to capture face, or 'q' to quit: ").strip().lower()
        if cmd == 'q':
            break
            
        image_bytes = get_face_image_bytes()
        
        if not image_bytes:
            continue
            
        print("🚀 Sending to server for recognition...")
        try:
            files = {'image': ('face.jpg', image_bytes, 'image/jpeg')}
            response = requests.post(PROCESS_FACE_ENDPOINT, files=files)
            
            if response.status_code != 200:
                print(f"❌ Server Error: {response.text}")
                continue
                
            result = response.json()
            status = result.get("status")
            
            if status == "recognized":
                name = result.get("name")
                print(f"\n✨ RECOGNIZED: {name} ✨")
                # Optional: Text-to-Speech here
                
            elif status == "unknown":
                print("\n❓ Face Unknown.")
                name = input("Enter name to enroll (or leave empty to skip): ").strip()
                
                if name:
                    print(f"📝 Enrolling {name}...")
                    # Send again with name
                    files = {'image': ('face.jpg', image_bytes, 'image/jpeg')}
                    data = {'name': name}
                    enroll_response = requests.post(PROCESS_FACE_ENDPOINT, files=files, data=data)
                    
                    if enroll_response.status_code in [200, 201]:
                        print(f"✅ Successfully enrolled {name}!")
                    else:
                        print(f"❌ Enrollment failed: {enroll_response.text}")
            
            elif status == "error":
                print(f"❌ Error: {result.get('message')}")
                
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            print("Make sure the server is running and IP is correct.")

if __name__ == "__main__":
    main()
