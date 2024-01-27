'''
Only storage
'''

import cv2
import insightface
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
import time
import os

# Define variables
name = 'Unknown'
# Define variables
video_url = "C:/Users/pg_ma/Videos/senaha.mp4"
#"rtsp://192.168.11.25:8554"
train_dir = 'C:/Users/pg_ma/personal_folders/senaha/Unity/script/faceRecognition/train_dir_demo/'
save_dir = os.path.join(train_dir, name)
#"C:/Users/pg_ma/Videos/test002.mp4"
#"C:/Users/pg_ma/Videos/test001.mp4"
#"C:/Users/pg_ma/Videos/conceptForEval3.mp4"
#"rtsp://192.168.11.8:8554"

# Initialize InsightFace detector
insightface_detector = insightface.app.FaceAnalysis()
insightface_detector.prepare(ctx_id=0)  # Use 

# Initialize the MTCNN module
mtcnn = MTCNN()

# Initialize a pre-trained InceptionResnetV1 model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load the trained SVM model
cap = cv2.VideoCapture(video_url)
data = []
face_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faceNum = 0
    faces = insightface_detector.get(frame)
    if faces:
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1] - 1, x2)
            y2 = min(frame.shape[0] - 1, y2)

            # Extract the face from the frame
            face_img = frame[y1:y2, x1:x2]
            face_filename = f'{save_dir}/face_{face_id}.jpg'
            success = cv2.imwrite(face_filename, face_img)
            if not success:
                print(f"Failed to save image: {face_filename}")
            face_id += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # 映像の解像度を下げる
    frame = cv2.resize(frame, (1920, 1080))     
    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()