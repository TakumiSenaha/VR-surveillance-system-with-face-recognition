'''
Update the video without changing the predefined number of rows in the database
For example, the database is always configured as follows
face_data = [
        1,AI01,0,0.0,0.0,0.0,vip member
        2,AI02,0,0.0,0.0,0.0,vip member
        3,AI03,0,0.0,0.0,0.0,vip member
        4,AI04,0,0.0,0.0,0.0,vip member
        5,AI05,0,0.0,0.0,0.0,vip member
        ...
    ]
'''

import cv2
import insightface
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
import csv
import sqlite3
import math
import time


# Define variables
FACE_THRESHOLD = 0.9
SKIP_FRAMES = 15  # Number of frames to skip
TIME_THRESHOLD = 5.0
DISTANCE_THRESHOLD = 0.001

csv_file = 'csv/face_data.csv'
db_path = '/~/Unity/FaceRecgnitionForVR/Assets/database/face_data.db'
svm_model_path = 'model/svm_model.pkl'
video_url = "rtsp://192.168.11.1:8554"
# or for test "video/360-degreees_video_for_test.mp4"
name_list = {}  # Dictionary to record the last seen time for each face


def get_3d_pos(equi_x, equi_y, width, height):
    # Normalize coordinates
    norm_x = equi_x / (width - 1)
    norm_y = equi_y / (height - 1)
    # Calculate longitude and latitude
    longitude = norm_x * 2 * math.pi
    latitude = (norm_y - 0.5) * math.pi
    x = math.cos(latitude) * math.sin(longitude)
    y = math.cos(latitude) * math.cos(longitude)
    z = math.sin(latitude)
    return x, y, z

def distance(coord1, coord2):
    x1, y1, z1 = coord1
    x2, y2, z2 = coord2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def init():
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # ヘッダ行をスキップ
        for row in reader:
            name = row[1]  # nameの列を取得
            ID = int(row[0])  # IDの列を取得
            name_list[name] = ID
    print(name_list)
    num_of_learned_faces_ = ID

    # データベースに接続
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # テーブル作成（既存のテーブルがある場合は削除して再作成）
    c.execute('''DROP TABLE IF EXISTS faces''')
    c.execute('''CREATE TABLE faces
                (ID INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, flag INTEGER, x REAL, y REAL, z REAL, profile TEXT)''')
    # データ挿入
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            name = row['name']
            flag = int(row['flag'])
            x = float(row['x'])
            y = float(row['y'])
            z = float(row['z'])
            profile = row['profile']
            c.execute('INSERT INTO faces (ID, name, flag, x, y, z, profile) VALUES (?, ?, ?, ?, ?, ?, ?)', (name_list[name], name, flag, x, y, z, profile))
    # コミットと接続のクローズ
    conn.commit()
    conn.close()

    return num_of_learned_faces_

def face_classification(unknown_face_id):
    frame_count = 0  # Frame counter
    last_seen = {}  # Dictionary to record the last seen time for each face
    last_coords = {}  # Dictionary to record the last coordinates for each face
    unknown_face_last_seen = 0  # 未知の顔が最後に検出された時刻

    # Initialize InsightFace detector
    insightface_detector = insightface.app.FaceAnalysis()
    insightface_detector.prepare(ctx_id=-1)  # CPU:-1, GPU:0
    mtcnn = MTCNN() # Initialize the MTCNN module
    resnet = InceptionResnetV1(pretrained='vggface2').eval() # Initialize a pre-trained InceptionResnetV1 model
    # Load the trained SVM model
    with open(svm_model_path, 'rb') as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(video_url)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    while True:
        frame_count += 1
        if(frame_count % SKIP_FRAMES != 0):
            continue
        frame_count = 0
        ret, frame = cap.read()
        
        if not ret:
            break
        # Get the current time
        now = time.time()
        # Check all IDs (both known and unknown faces)
        for ID in list(last_seen.keys()):  # last_seen には既知および未知の顔のIDが含まれる
            if now - last_seen[ID] > TIME_THRESHOLD:
                # If the face has not been seen for a certain time, reset the flag to 0
                c.execute("UPDATE faces SET flag = 0 WHERE ID = ?", (ID,))
                conn.commit()

                # print('\n'+'flag reset to 0 for ID:', ID)
        # Detect faces in the image
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
                if face_img.size > 0 and face_img.shape[0] > 20 and face_img.shape[1] > 20:  # Check if the face image is not empty
                    face_cropped = mtcnn(face_img)
                    if face_cropped is not None:
                        face_embedding = resnet(face_cropped.unsqueeze(0))
                        # Predict the label of the face
                        proba = model.predict_proba(face_embedding.detach().numpy())
                        max_proba = np.max(proba)

                        # Update the coordinates and flag in the database
                        sphere_x, sphere_y, sphere_z = get_3d_pos(x1, y1, frame.shape[1], frame.shape[0])
                        unity_x = -sphere_y
                        unity_y = sphere_z  # Swap Y and Z
                        unity_z = -sphere_x  # Flip Z

                        if max_proba > FACE_THRESHOLD:
                            label = model.predict(face_embedding.detach().numpy())
                        else:
                            label = ['Unknown']
                            if now - unknown_face_last_seen > TIME_THRESHOLD:
                                unknown_face_id += 1  # 新しいIDを割り当てる
                            ID = unknown_face_id
                            unknown_face_last_seen = now  # 最後に検出された時刻を更新
                            
                            if ID not in last_seen or now - last_seen[ID] > TIME_THRESHOLD:
                                # 未知の顔のデータベースへの追加処理
                                c.execute("INSERT INTO faces (ID, name, flag, x, y, z, profile) VALUES (?, ?, ?, ?, ?, ?, ?)", (ID, 'Unknown', 1, unity_x, unity_y, unity_z, 'May be an outsider!!'))
                                conn.commit()

                            last_seen[ID] = now  # 最後に検出された時間を記録
                            last_coords[ID] = (unity_x, unity_y, unity_z)  # 最後に検出された座標を記録
                        # Draw rectangle around the face and put label
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # cv2.putText(frame, label[0], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                        if label[0] != 'Unknown':
                            ID = name_list[label[0]]
                            now = time.time()
                            last_seen[ID] = now  # Update the last seen time
                            if ID not in last_coords or distance(last_coords[ID], (unity_x, unity_y, unity_z)) > DISTANCE_THRESHOLD:
                                # If the face has moved a certain distance, update the coordinates in the database
                                c.execute("UPDATE faces SET x = ?, y = ?, z = ?, flag = 1 WHERE ID = ?", (unity_x, unity_y, unity_z, ID))
                                conn.commit()
                                # print('/n'+'move ')
                            last_coords[ID] = (unity_x, unity_y, unity_z)  # Update the last coordinates
                            # print('/n'+'lastdoords')
                            # print(last_coords,last_seen)
    # Display the resulting frame
    # frame = cv2.resize(frame, (1920, 1080))
    # cv2.imshow('Video', frame)
                            
    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    try:
        num_of_learned_faces = init()
        face_classification(num_of_learned_faces)
    except Exception as e:
        if "init" in str(e):
            print(f"An error occurred: {e} in init()")
        else:
            print(f"An error occurred: {e} in face_classification()")

