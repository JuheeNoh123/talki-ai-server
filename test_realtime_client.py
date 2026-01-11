import asyncio
import websockets
import cv2
import json
import time
import os
import mediapipe as mp

# Install: pip install websockets opencv-python mediapipe

SERVER_URL = "ws://127.0.0.1:8000/realtime"
VIDEO_PATH = "temp_video_posing.mp4" 

# 랜드마크 인덱스 정의 (서버와 동일)
LEFT_IRIS  = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263
UPPER_LID = [159, 386] 
LOWER_LID = [145, 374] 

REQUIRED_FACE_IDX = (
    LEFT_IRIS + RIGHT_IRIS + 
    [LEFT_EYE_OUTER, LEFT_EYE_INNER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER] +
    UPPER_LID + LOWER_LID
)

REQUIRED_POSE_IDX = [13, 14, 15, 16] # Elbows and Wrists

def extract_landmarks(face_result, pose_result):
    """MediaPipe 결과에서 필요한 랜드마크만 추출하여 딕셔너리로 변환"""
    data = {"face": {}, "pose": {}}
    
    # 1. Face
    if face_result.multi_face_landmarks:
        lms = face_result.multi_face_landmarks[0].landmark
        for idx in REQUIRED_FACE_IDX:
            lm = lms[idx]
            data["face"][str(idx)] = {"x": lm.x, "y": lm.y}
            
    # 2. Pose
    if pose_result.pose_landmarks:
        lms = pose_result.pose_landmarks.landmark
        for idx in REQUIRED_POSE_IDX:
            lm = lms[idx]
            data["pose"][str(idx)] = {"x": lm.x, "y": lm.y}
            
    return data

async def test_realtime():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file '{VIDEO_PATH}' not found.")
        return

    # MediaPipe 초기화
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True, 
        max_num_faces=1,
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    pose = mp_pose.Pose(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )


    print(f"Connecting to {SERVER_URL}...")
    try:
        async with websockets.connect(SERVER_URL) as websocket:
            print("✅ Connected to WebSocket Server (Client-Side Inference Mode)")
            first_msg = await websocket.recv()
            first_data = json.loads(first_msg)

            if first_data.get("type") == "session_start":
                presentation_id = first_data["presentationId"]
                print("🎤 Presentation ID:", presentation_id)
            
            cap = cv2.VideoCapture(VIDEO_PATH)
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("End of video.")
                    break
                
                # Resize specifically for model inference performance
                frame_resized = cv2.resize(frame, (640, 360))
                rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # --- Client-Side Inference ---
                face_res = face_mesh.process(rgb)
                pose_res = pose.process(rgb)
                
                # Extract Landmarks
                payload_data = extract_landmarks(face_res, pose_res)
                payload_data["timestamp"] = time.time() * 1000
                
                # Send JSON
                await websocket.send(json.dumps(payload_data))
                
                # Receive response
                response = await websocket.recv()
                data = json.loads(response)
                
                raw = data.get('raw_result', {})
                message = data.get("data", "")
                if not message: message = "..."

                gaze_info = raw.get('gaze')
                gaze_str = gaze_info.get('horiz', 'N/A') if gaze_info else 'N/A'
                dx_val = gaze_info.get('dx', 0.0) if gaze_info else 0.0
                
                print(f"[{frame_count}] Client-Inference -> Gaze: {gaze_str:<6} (dx:{dx_val:+.2f}) | Feedback: {message}")
                
                frame_count += 1
                
                # Simulate 10 Hz (Recommended)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Processing time should be subtracted from 0.1s ideally, but simple sleep 0.1s is close enough for test
                # Or better: await asyncio.sleep(0.1)
                await asyncio.sleep(0.1) 
                
            cap.release()
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(test_realtime())
    except KeyboardInterrupt:
        pass
